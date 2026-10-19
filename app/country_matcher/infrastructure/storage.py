"""Where the dataset images live: Supabase Storage (S3-compatible API), or a local folder for development and tests.

Credentials are read from the settings (environment) by the server and the admin scripts only. The browser is only ever
given image URLs: public ones, or presigned ones that expire (a presigned URL carries the access key *id* and a
signature, never the secret key).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import Settings
from app.errors import NotFoundError, ServiceNotConfigured

log = logging.getLogger("country_matcher.storage")

LOCAL_URL_PREFIX = "/api/v1/country-matcher/local-images/"


class S3ImageStorage:
    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        public_base_url: str = "",
        signed_url_ttl_seconds: int = 3600,
    ):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.public_base_url = public_base_url.rstrip("/")
        self.ttl = signed_url_ttl_seconds
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},  # Supabase's S3 endpoint is path-style
                retries={"max_attempts": 5, "mode": "standard"},  # backoff on throttling and 5xx
                connect_timeout=10,
                read_timeout=60,
            ),
        )

    async def put(self, path: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self._client.put_object, Bucket=self.bucket, Key=path, Body=data, ContentType=content_type,
            CacheControl="public, max-age=31536000, immutable",  # the path contains the content hash: it never changes
        )

    async def get(self, path: str) -> bytes:
        from botocore.exceptions import ClientError

        def read() -> bytes:
            try:
                return self._client.get_object(Bucket=self.bucket, Key=path)["Body"].read()
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                    raise NotFoundError(f"Stored image not found: {path}") from exc
                raise

        return await asyncio.to_thread(read)

    async def exists(self, path: str) -> bool:
        from botocore.exceptions import ClientError

        def head() -> bool:
            try:
                self._client.head_object(Bucket=self.bucket, Key=path)
                return True
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                    return False
                raise

        return await asyncio.to_thread(head)

    def url(self, path: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{path}"
        return self._client.generate_presigned_url(  # computed locally, no network call
            "get_object", Params={"Bucket": self.bucket, "Key": path}, ExpiresIn=self.ttl
        )


class LocalImageStorage:
    """Files in a directory. ``url`` points at the API's own ``/local-images`` route (registered only in this mode)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):  # no ../ escapes
            raise NotFoundError("Not found.")
        return target

    async def put(self, path: str, data: bytes, content_type: str) -> None:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def get(self, path: str) -> bytes:
        target = self.resolve(path)
        if not target.is_file():
            raise NotFoundError(f"Stored image not found: {path}")
        return await asyncio.to_thread(target.read_bytes)

    async def exists(self, path: str) -> bool:
        return self.resolve(path).is_file()

    def url(self, path: str) -> str:
        return f"{LOCAL_URL_PREFIX}{path}"


class UnconfiguredStorage:
    """Stands in when S3 credentials are missing so the API still starts; using it explains what to set."""

    async def put(self, path: str, data: bytes, content_type: str) -> None:
        self._fail()

    async def get(self, path: str) -> bytes:
        self._fail()

    async def exists(self, path: str) -> bool:
        self._fail()

    def url(self, path: str) -> str:
        self._fail()

    @staticmethod
    def _fail():
        raise ServiceNotConfigured(
            "Image storage is not configured. Set COUNTRY_MATCHER_S3_* in backend/.env (or COUNTRY_MATCHER_STORAGE=local)."
        )


def build_storage(settings: Settings):
    if settings.country_matcher_storage == "local":
        return LocalImageStorage(settings.country_matcher_local_dir)
    if not (
        settings.country_matcher_s3_endpoint
        and settings.country_matcher_s3_access_key_id
        and settings.country_matcher_s3_secret_access_key
    ):
        log.warning("Country Matcher storage is not configured (COUNTRY_MATCHER_S3_*)")
        return UnconfiguredStorage()
    return S3ImageStorage(
        endpoint=settings.country_matcher_s3_endpoint,
        region=settings.country_matcher_s3_region,
        bucket=settings.country_matcher_s3_bucket,
        access_key_id=settings.country_matcher_s3_access_key_id,
        secret_access_key=settings.country_matcher_s3_secret_access_key,
        public_base_url=settings.country_matcher_public_base_url,
        signed_url_ttl_seconds=settings.country_matcher_signed_url_ttl_seconds,
    )
