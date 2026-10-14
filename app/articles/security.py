"""Password hashing (scrypt) and signed, expiring login tokens (HMAC-SHA256). Standard library only."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=_N, r=_R, p=_P)
        return hmac.compare_digest(actual, bytes.fromhex(digest))
    except ValueError:
        return False


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _sign(secret: str, payload: str) -> str:
    return _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())


def password_fingerprint(stored_hash: str | None) -> str:
    """Identifies the current password, so changing it invalidates every token issued before."""
    return hashlib.sha256((stored_hash or "default").encode()).hexdigest()[:16]


def make_token(secret: str, username: str, fingerprint: str, ttl_seconds: int, now: float | None = None) -> str:
    payload = _b64(f"{username}|{int((now or time.time()) + ttl_seconds)}|{fingerprint}".encode())
    return f"{payload}.{_sign(secret, payload)}"


def read_token(secret: str, token: str, now: float | None = None) -> tuple[str, str] | None:
    """(username, fingerprint) of a genuine, unexpired token, otherwise None."""
    payload, _, signature = token.partition(".")
    if not payload or not hmac.compare_digest(signature, _sign(secret, payload)):
        return None
    try:
        username, expires, fingerprint = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode().split("|")
        if int(expires) < (now or time.time()):
            return None
    except ValueError:
        return None
    return username, fingerprint
