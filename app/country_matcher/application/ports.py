"""What the application layer needs from the outside world (implemented in ``infrastructure``, faked in tests)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from PIL import Image

from app.country_matcher.domain.models import EmbeddingInfo, ImageRecord, ModelName, NewImage, Split


class ImageEmbedder(Protocol):
    """Turns images into L2-normalised feature vectors. Loaded once per process, then reused by every request."""

    model: ModelName
    dimension: int  # of the vectors, known before the weights are loaded

    def load(self) -> None:
        """Load the weights now (idempotent, thread-safe). Blocking: call from a worker thread."""

    @property
    def info(self) -> EmbeddingInfo:
        """Model name and weights version (loads the weights on first use)."""

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """``(len(images), dimension)`` float32 array, each row with unit length. Blocking: call from a worker thread."""


class ImageStorage(Protocol):
    async def put(self, path: str, data: bytes, content_type: str) -> None: ...
    async def get(self, path: str) -> bytes: ...
    async def exists(self, path: str) -> bool: ...
    def url(self, path: str) -> str:
        """A URL the browser can load the image from (public, or signed and short-lived). Never carries credentials."""


@dataclass(frozen=True)
class IndexedImage:
    """A searchable image and the models that have stored an embedding of it."""

    image: ImageRecord
    models: tuple[ModelName, ...]


@dataclass(frozen=True)
class SearchHit:
    image: ImageRecord
    similarity: float


@dataclass(frozen=True)
class EvaluationRun:
    id: uuid.UUID
    created_at: datetime
    results: dict[str, Any]


class CountryStore(Protocol):
    """Persistence: images, one embedding table per model, and the stored evaluation runs."""

    async def find_by_hash(self, image_hash: str) -> ImageRecord | None: ...
    async def has_original_url(self, original_url: str) -> bool: ...
    async def add_image(self, image: NewImage) -> ImageRecord: ...
    async def get_image(self, image_id: uuid.UUID) -> ImageRecord | None: ...
    async def list_images(self, *, split: Split | None = None) -> list[ImageRecord]: ...
    async def images_missing_embedding(self, info: EmbeddingInfo) -> list[ImageRecord]:
        """**Index** images with no embedding of this model *at this version* (a new version re-embeds everything).
        Held-out test images never get stored embeddings: they are only ever queries."""

    async def save_embeddings(self, info: EmbeddingInfo, items: Sequence[tuple[uuid.UUID, np.ndarray]]) -> None: ...
    async def search(self, info: EmbeddingInfo, query: np.ndarray, k: int) -> list[SearchHit]:
        """The ``k`` most cosine-similar **index** images among the embeddings made by this exact model version."""

    async def indexed_images(self, *, country_code: str | None, limit: int, offset: int) -> tuple[list[IndexedImage], int]:
        """A page of the **index** images (never the held-out test ones) with their embedding models, and the total."""

    async def indexed_countries(self) -> list[tuple[str, str, int]]:
        """``(code, name, number of index images)`` per country, by name."""

    async def count_images(self, *, split: Split | None = None) -> int: ...
    async def count_countries(self) -> int: ...
    async def embedding_stats(self, model: ModelName) -> dict[str, Any]:
        """``count``, ``versions`` (every model version present), ``dimension``, ``vector_bytes`` (what the vectors
        occupy) and ``total_bytes`` (table plus HNSW index) of the stored embeddings."""

    async def save_evaluation(self, results: dict[str, Any]) -> EvaluationRun: ...
    async def latest_evaluation(self) -> EvaluationRun | None: ...
