"""In-memory stand-ins for Country Matcher's ports, shared by its tests (no database, network or neural networks)."""

from __future__ import annotations

import io
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
from PIL import Image

from app.country_matcher.application.ports import EvaluationRun, IndexedImage, SearchHit
from app.country_matcher.domain.models import EmbeddingInfo, ImageRecord, ModelName, NewImage, Split
from app.country_matcher.infrastructure.imaging import dhash, sha256_hex
from app.errors import NotFoundError

COLORS = {"AA": (220, 30, 30), "BB": (30, 220, 30), "CC": (30, 30, 220)}


def make_image(color: tuple[int, int, int], seed: int, size=(320, 240)) -> Image.Image:
    """A noisy picture around ``color``: the mean colour identifies it, the noise makes perceptual hashes differ."""
    rng = np.random.default_rng(seed)
    base = np.array(color, dtype=float)[None, None, :]
    pixels = np.clip(base + rng.normal(0, 60, (size[1], size[0], 3)), 0, 255).astype("uint8")
    return Image.fromarray(pixels)


def jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


class FakeEmbedder:
    """Embeds an image by its centred mean colour, so images of one 'country' are close. Counts what it embeds."""

    def __init__(self, model: ModelName, version: str = "fake-1", scale: float = 1.0):
        self.model, self.dimension, self.version, self.scale = model, 3, version, scale
        self.embedded = 0
        self.loads = 0

    def load(self) -> None:
        self.loads += 1

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(self.model, f"fake/{self.model.value}", self.version, self.dimension)

    def embed(self, images):
        self.embedded += len(images)
        vectors = np.stack([np.asarray(im.convert("RGB"), dtype=float).mean(axis=(0, 1)) - 128 for im in images]) * self.scale
        return (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)).astype(np.float32)


class Embedders:
    def __init__(self):
        self.by_model = {m: FakeEmbedder(m) for m in ModelName}

    def get(self, model: ModelName) -> FakeEmbedder:
        return self.by_model[model]

    def describe(self):
        return {m: {"model_name": e.info.model_name, "dimension": e.dimension, "loaded": True} for m, e in self.by_model.items()}


class MemoryStorage:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    async def put(self, path, data, content_type):
        self.files[path] = data

    async def get(self, path):
        if path not in self.files:
            raise NotFoundError(path)
        return self.files[path]

    async def exists(self, path):
        return path in self.files

    def url(self, path):
        return f"https://storage.test/{path}?sig=1"


class MemoryStore:
    def __init__(self):
        self.images: dict[uuid.UUID, ImageRecord] = {}
        self.vectors: dict[ModelName, dict[uuid.UUID, tuple[EmbeddingInfo, np.ndarray]]] = {m: {} for m in ModelName}
        self.evaluations: list[EvaluationRun] = []

    def factory(self):
        @asynccontextmanager
        async def open_store():
            yield self

        return open_store

    async def find_by_hash(self, image_hash):
        return next((i for i in self.images.values() if i.image_hash == image_hash), None)

    async def has_original_url(self, original_url):
        return any(i.original_url == original_url for i in self.images.values())

    async def add_image(self, image: NewImage):
        record = ImageRecord(id=uuid.uuid4(), created_at=datetime.now(UTC), **{**image.__dict__})
        if any(i.image_hash == record.image_hash or i.original_url == record.original_url for i in self.images.values()):
            raise ValueError("duplicate")
        self.images[record.id] = record
        return record

    async def get_image(self, image_id):
        return self.images.get(image_id)

    async def list_images(self, *, split=None):
        rows = [i for i in self.images.values() if split is None or i.split is split]
        return sorted(rows, key=lambda i: (i.country_code, i.created_at, str(i.id)))

    async def indexed_images(self, *, country_code, limit, offset):
        rows = [i for i in await self.list_images(split=Split.INDEX) if country_code in (None, i.country_code)]
        rows.sort(key=lambda i: (i.country_name, i.created_at, str(i.id)))
        page = [
            IndexedImage(i, tuple(m for m in ModelName if i.id in self.vectors[m])) for i in rows[offset : offset + limit]
        ]
        return page, len(rows)

    async def indexed_countries(self):
        counts: dict[tuple[str, str], int] = {}
        for i in await self.list_images(split=Split.INDEX):
            counts[(i.country_code, i.country_name)] = counts.get((i.country_code, i.country_name), 0) + 1
        return sorted(((c, n, k) for (c, n), k in counts.items()), key=lambda t: t[1])

    async def count_images(self, *, split=None):
        return len(await self.list_images(split=split))

    async def count_countries(self):
        return len({i.country_code for i in self.images.values()})

    async def images_missing_embedding(self, info):
        done = {iid for iid, (i, _) in self.vectors[info.model].items() if i.model_version == info.model_version}
        return [i for i in await self.list_images(split=Split.INDEX) if i.id not in done]

    async def save_embeddings(self, info, items):
        for image_id, vector in items:
            self.vectors[info.model][image_id] = (info, vector)

    async def search(self, info, query, k):
        hits = []
        for image_id, (i, vector) in self.vectors[info.model].items():
            image = self.images[image_id]
            if i.model_version == info.model_version and image.split is Split.INDEX:
                hits.append(SearchHit(image, float(np.dot(query, vector))))
        return sorted(hits, key=lambda h: (-h.similarity, str(h.image.id)))[:k]

    async def embedding_stats(self, model):
        rows = self.vectors[model]
        dim = {ModelName.CLIP: 512, ModelName.RESNET50: 2048}[model]
        return {"count": len(rows), "versions": sorted({i.model_version for i, _ in rows.values()}), "dimension": dim,
                "vector_bytes": len(rows) * dim * 4, "total_bytes": len(rows) * dim * 4}

    async def save_evaluation(self, results: dict[str, Any]):
        run = EvaluationRun(uuid.uuid4(), datetime.now(UTC), results)
        self.evaluations.append(run)
        return run

    async def latest_evaluation(self):
        return self.evaluations[-1] if self.evaluations else None


async def seed_dataset(store: MemoryStore, storage: MemoryStorage, *, per_country_index=6, per_country_test=2) -> None:
    """Three countries, each a colour; index and test images (different scene groups) already stored."""
    seed = 0
    for code, color in COLORS.items():
        for n in range(per_country_index + per_country_test):
            seed += 1
            image = make_image(color, seed)
            data = jpeg(image)
            split = Split.TEST if n >= per_country_index else Split.INDEX
            path = f"images/{code}/{seed}.jpg"
            await storage.put(path, data, "image/jpeg")
            await store.add_image(
                NewImage(
                    country_code=code, country_name=f"Country {code}", city=None, original_url=f"https://x/{code}/{seed}",
                    source_page_url=None, storage_path=path, source_license="CC BY 4.0", source_attribution="tester",
                    image_hash=sha256_hex(data), perceptual_hash=dhash(image), scene_group=f"{code}|cat|author{seed}",
                    width=image.width, height=image.height, split=split,
                )
            )
