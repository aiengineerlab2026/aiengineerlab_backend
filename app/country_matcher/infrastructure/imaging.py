"""Decoding, validating and fingerprinting images (uploads and dataset downloads share this)."""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps, UnidentifiedImageError

from app.errors import ValidationFailed

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_PIXELS = 40_000_000  # refuse decompression bombs: a small file can decode to a huge bitmap
MIN_SIDE = 64
STORED_MAX_SIDE = 1024  # stored copies are downscaled to this (enough for retrieval and display, small to host)
STORED_QUALITY = 88


def decode_image(data: bytes, *, min_side: int = MIN_SIDE) -> Image.Image:
    """Bytes -> RGB image, or ``ValidationFailed`` with a message safe to show the user."""
    if not data:
        raise ValidationFailed("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, size = probe.format, probe.size
            if fmt not in ALLOWED_FORMATS:
                raise ValidationFailed("Unsupported image format. Please upload a JPEG, PNG or WebP image.")
            if size[0] * size[1] > MAX_PIXELS:
                raise ValidationFailed("The image is too large to process. Please upload a smaller one.")
            probe.verify()  # structural check; the image must be reopened to be read afterwards
        image = Image.open(io.BytesIO(data))
        image.load()
    except ValidationFailed:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ValidationFailed("The file is not a readable image.") from exc
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
    else:
        image = image.convert("RGB")
    if min(image.size) < min_side:
        raise ValidationFailed(f"The image is too small: both sides must be at least {min_side} pixels.")
    return image


def normalise_for_storage(image: Image.Image) -> tuple[bytes, int, int]:
    """Downscale (never upscale) and encode as JPEG. Returns the bytes and the stored width and height."""
    img = image.copy()
    img.thumbnail((STORED_MAX_SIDE, STORED_MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=STORED_QUALITY, optimize=True)
    return buf.getvalue(), img.width, img.height


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(image: Image.Image, size: int = 8) -> str:
    """64-bit difference hash (16 hex chars). Near-identical photos differ in only a few bits."""
    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1])
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()
