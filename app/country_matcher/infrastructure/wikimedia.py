"""Wikimedia Commons as an image source.

Commons states a license for every file, and this module keeps only those that allow reuse with attribution: CC0, public
domain, CC BY and CC BY-SA (never the NonCommercial variants). The license, the author and the file's page are stored with
every image so the app can show proper credit. Only the public MediaWiki API is used, no HTML scraping.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from collections.abc import Sequence

import httpx

from app.country_matcher.domain.models import Candidate
from app.errors import ExternalServiceError

log = logging.getLogger("country_matcher.wikimedia")

API_URL = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 960  # one of Wikimedia's standard thumbnail steps (other widths are rate-limited)
ACCEPTED_MIME = {"image/jpeg", "image/png", "image/webp"}
MIN_WIDTH, MIN_HEIGHT = 600, 400
MAX_ASPECT = 3.0  # skips panoramas and banners
TITLE_BLOCKLIST = re.compile(r"\b(map|logo|flag|coat of arms|diagram|plan|poster|screenshot|sign|icon|postcard|postkarte)\b", re.I)
_LICENSE_OK = re.compile(r"^(cc[ -]by(-sa)?[ -]\d(\.\d)?(\s+\w+)?|cc0.*|cc[ -]zero.*|public domain.*|pd[ -].*|pd)$", re.I)
_TAGS = re.compile(r"<[^>]+>")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def license_allowed(short_name: str) -> bool:
    name = short_name.strip()
    if re.search(r"\b(nc|nd)\b", name, re.I):
        return False
    return bool(_LICENSE_OK.match(name))


def plain_text(markup: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAGS.sub(" ", markup or ""))).strip()


class WikimediaCommonsSource:
    def __init__(self, user_agent: str, *, client: httpx.AsyncClient | None = None, min_interval_seconds: float = 0.3):
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(headers={"User-Agent": user_agent}, timeout=httpx.Timeout(30, connect=10))
        self.min_interval = min_interval_seconds

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _query(self, params: dict) -> dict:
        full = {"action": "query", "format": "json", "formatversion": 2, **params}
        for attempt in range(1, 5):
            await asyncio.sleep(self.min_interval)
            try:
                response = await self.client.get(API_URL, params=full)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            else:
                if response.status_code == 200:
                    return response.json()
                last = f"HTTP {response.status_code}"
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
            log.warning("Commons API failed, retrying", extra={"attempt": attempt, "reason": last})
            await asyncio.sleep(min(2**attempt, 20))
        raise ExternalServiceError(f"Wikimedia Commons API failed: {last}")

    async def category_info(self, categories: Sequence[str]) -> dict[str, dict | None]:
        """``{category: {"files": n, "subcategories": n}}``, or ``None`` for a category that does not exist."""
        data = await self._query({"prop": "categoryinfo", "titles": "|".join(f"Category:{c}" for c in categories)})
        out: dict[str, dict | None] = {}
        for page in data["query"]["pages"]:
            name = page["title"].removeprefix("Category:")
            info = page.get("categoryinfo")
            out[name] = None if page.get("missing") or not info else {"files": info.get("files", 0), "subcategories": info.get("subcats", 0)}
        return out

    async def candidates(
        self, *, category: str, country_code: str, country_name: str, city: str | None, limit: int, since_year: int | None = None
    ) -> list[Candidate]:
        """Up to ``limit`` acceptable files of ``category`` (right type, size and license), in the order Commons lists them.

        With ``since_year`` only photos whose recorded capture date (EXIF) is that year or later are kept; undated ones are skipped."""
        found: list[Candidate] = []
        params = {
            "generator": "categorymembers", "gcmtitle": f"Category:{category}", "gcmtype": "file", "gcmlimit": 50,
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": THUMB_WIDTH,
            "iiextmetadatafilter": "LicenseShortName|Artist|Credit|Restrictions|DateTimeOriginal",
        }
        cont: dict = {}
        while len(found) < limit:
            data = await self._query({**params, **cont})
            for page in data.get("query", {}).get("pages", []):
                candidate = _candidate(page, category, country_code, country_name, city, since_year)
                if candidate:
                    found.append(candidate)
            cont = data.get("continue")
            if not cont:
                break
        return found[:limit]


def _candidate(
    page: dict, category: str, country_code: str, country_name: str, city: str | None, since_year: int | None = None
) -> Candidate | None:
    info = (page.get("imageinfo") or [None])[0]
    if not info or info.get("mime") not in ACCEPTED_MIME:
        return None
    title = page.get("title", "").removeprefix("File:")
    width, height = info.get("width") or 0, info.get("height") or 0
    if width < MIN_WIDTH or height < MIN_HEIGHT or max(width, height) / max(1, min(width, height)) > MAX_ASPECT:
        return None
    if TITLE_BLOCKLIST.search(title.replace("_", " ")):  # file names use underscores, which are word characters to a regex
        return None
    meta = info.get("extmetadata") or {}

    def value(key: str) -> str:
        return plain_text((meta.get(key) or {}).get("value", ""))

    if since_year is not None:
        year = _YEAR.search(value("DateTimeOriginal"))
        if not year or int(year.group()) < since_year:
            return None
    license_name = value("LicenseShortName")
    if not license_allowed(license_name):
        return None
    if "personality" in value("Restrictions").lower():  # a recognisable person's rights are involved
        return None
    author = value("Artist") or value("Credit") or "Unknown author"
    download_url = info.get("thumburl") or info.get("url")
    if not download_url or not info.get("url"):
        return None
    return Candidate(
        country_code=country_code, country_name=country_name, city=city, category=category, title=title,
        download_url=download_url, original_url=info["url"], page_url=info.get("descriptionurl"), license=license_name,
        attribution=f"{author[:200]} via Wikimedia Commons", author=author, width=width, height=height,
    )
