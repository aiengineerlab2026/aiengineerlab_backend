"""Downloading remote images without opening the server to SSRF.

Used only by the admin ingestion script, and written as if the URL were hostile:

* https only, no credentials in the URL, no unusual ports
* the host must be on an allow-list (``COUNTRY_MATCHER_ALLOWED_IMAGE_HOSTS``, subdomains included)
* every address the name resolves to must be a public one (no loopback, private, link-local or reserved ranges),
  re-checked on every redirect hop; redirects are followed by hand, at most ``MAX_REDIRECTS`` of them
* a hard cap on the body size, a content-type check and timeouts
* polite to the source: limited concurrency, a gap between requests, retries with backoff on 429 / 5xx

Remaining gap: the name is resolved once by this check and again by the HTTP client (DNS rebinding). The allow-list
makes that exploitable only by whoever controls the DNS of an allowed host.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import time
from collections.abc import Sequence
from urllib.parse import urljoin, urlsplit

import httpx

from app.errors import ExternalServiceError, ValidationFailed

log = logging.getLogger("country_matcher.download")

MAX_REDIRECTS = 3
MAX_BYTES = 15 * 1024 * 1024
ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/webp")


class UnsafeUrl(ValidationFailed):
    code = "unsafe_url"


def host_allowed(host: str, allowed_hosts: Sequence[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in allowed_hosts)


def is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 is loopback in disguise
    return ip.is_global and not ip.is_multicast


async def check_url(url: str, allowed_hosts: Sequence[str]) -> str:
    """Raise ``UnsafeUrl`` unless ``url`` is safe to fetch. Returns the (lower-cased) host."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise UnsafeUrl("Only https URLs are allowed.")
    if parts.username or parts.password:
        raise UnsafeUrl("Credentials in a URL are not allowed.")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeUrl("The URL has no host.")
    try:
        port = parts.port or 443
    except ValueError as exc:
        raise UnsafeUrl("The URL has an invalid port.") from exc
    if port != 443:
        raise UnsafeUrl("Only the default https port is allowed.")
    if not host_allowed(host, allowed_hosts):
        raise UnsafeUrl(f"Host not allowed: {host}")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeUrl(f"Could not resolve {host}.") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or not all(is_public_ip(a) for a in addresses):
        raise UnsafeUrl(f"{host} does not resolve to a public address.")
    return host


class SafeImageDownloader:
    def __init__(
        self,
        allowed_hosts: Sequence[str],
        user_agent: str,
        *,
        max_bytes: int = MAX_BYTES,
        concurrency: int = 2,
        min_interval_seconds: float = 0.3,
        attempts: int = 4,
        client: httpx.AsyncClient | None = None,
    ):
        self.allowed_hosts = list(allowed_hosts)
        self.max_bytes = max_bytes
        self.attempts = attempts
        self.min_interval = min_interval_seconds
        self._sem = asyncio.Semaphore(concurrency)
        self._gap = asyncio.Lock()
        self._last = 0.0
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent}, follow_redirects=False, timeout=httpx.Timeout(30, connect=10)
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def download(self, url: str) -> tuple[bytes, str]:
        """Bytes and content-type of the image at ``url``. Retries transient failures; raises ``UnsafeUrl`` /
        ``ValidationFailed`` for URLs and bodies that must not be used, ``ExternalServiceError`` when the source keeps failing."""
        last: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                async with self._sem:
                    await self._wait_turn()
                    return await self._fetch(url)
            except _Transient as exc:
                last = exc
                delay = exc.retry_after or min(2 ** attempt, 30)
                log.warning("download failed, retrying", extra={"url": url, "attempt": attempt, "reason": str(exc), "wait_s": delay})
                await asyncio.sleep(delay)
        raise ExternalServiceError(f"Could not download {url}: {last}") from last

    async def _wait_turn(self) -> None:
        async with self._gap:  # one request starts at least ``min_interval`` after the previous one
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

    async def _fetch(self, url: str) -> tuple[bytes, str]:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            await check_url(current, self.allowed_hosts)
            try:
                async with self.client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise UnsafeUrl("Redirect without a location.")
                        current = urljoin(current, location)
                        continue
                    if response.status_code in (429, 500, 502, 503, 504):
                        retry_after = response.headers.get("retry-after", "")
                        raise _Transient(f"HTTP {response.status_code}", float(retry_after) if retry_after.isdigit() else None)
                    if response.status_code != 200:
                        raise ValidationFailed(f"The source answered HTTP {response.status_code}.")
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if content_type not in ALLOWED_CONTENT_TYPES:
                        raise ValidationFailed(f"Not an accepted image type: {content_type or 'unknown'}.")
                    declared = response.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > self.max_bytes:
                        raise ValidationFailed("The image is larger than the allowed size.")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > self.max_bytes:
                            raise ValidationFailed("The image is larger than the allowed size.")
                    return bytes(body), content_type
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise _Transient(type(exc).__name__) from exc
        raise UnsafeUrl("Too many redirects.")


class _Transient(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after
