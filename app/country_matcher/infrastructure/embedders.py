"""The two image encoders: CLIP (image tower) and ResNet50 (penultimate layer).

Both return L2-normalised vectors, so the cosine similarity of two images is just their dot product. Weights load
lazily, once per process, under a lock; every later request reuses the same in-memory model.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence

import numpy as np
from PIL import Image

from app.config import Settings
from app.country_matcher.domain.models import EmbeddingInfo, ModelName
from app.errors import ExternalServiceError

log = logging.getLogger("country_matcher.embedders")

CLIP_DIMENSION = 512
RESNET_DIMENSION = 2048  # the pooled features before the 1000-class layer, never the ImageNet logits
BATCH_SIZE = 16


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class _LazyEmbedder:
    model: ModelName
    dimension: int
    model_name: str

    def __init__(self, device: str):
        self._requested_device = device
        self._lock = threading.Lock()
        self._loaded = False
        self._info: EmbeddingInfo | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._info = self._load()
            except Exception as exc:  # noqa: BLE001 - weights missing / offline / out of memory
                log.exception("could not load %s", self.model_name)
                raise ExternalServiceError(
                    f"The {self.model.value} model could not be loaded on the server.", details={"model": self.model.value}
                ) from exc
            self._loaded = True
            log.info("model loaded", extra={"model": self.model_name, "version": self._info.model_version})

    @property
    def info(self) -> EmbeddingInfo:
        self.load()
        assert self._info is not None
        return self._info

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.load()
        if not images:
            return np.zeros((0, self.dimension), dtype=np.float32)
        parts = [self._embed_batch(images[i : i + BATCH_SIZE]) for i in range(0, len(images), BATCH_SIZE)]
        return l2_normalise(np.concatenate(parts))

    def _load(self) -> EmbeddingInfo:
        raise NotImplementedError

    def _embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        raise NotImplementedError


class ClipEmbedder(_LazyEmbedder):
    model = ModelName.CLIP
    dimension = CLIP_DIMENSION

    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        self.model_name = model_name

    def _load(self) -> EmbeddingInfo:
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.device = resolve_device(self._requested_device)
        self._torch = torch
        # only the image tower and its projection: the text tower is never used, so it is not kept in memory
        self._net = CLIPVisionModelWithProjection.from_pretrained(self.model_name).eval().to(self.device)
        self._processor = CLIPImageProcessor.from_pretrained(self.model_name)
        dim = int(self._net.config.projection_dim)
        if dim != self.dimension:
            raise RuntimeError(f"{self.model_name} produces {dim}-d embeddings, expected {self.dimension}")
        return EmbeddingInfo(ModelName.CLIP, self.model_name, _hub_revision(self.model_name), dim)

    def _embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixels = self._processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        with self._torch.inference_mode():
            features = self._net(pixel_values=pixels).image_embeds  # the projected 512-d CLIP image embedding
            out = features.float().cpu().numpy().copy()  # own the bytes: no tensor storage stays reachable from the result
        del pixels, features
        return out


class ResNetEmbedder(_LazyEmbedder):
    model = ModelName.RESNET50
    dimension = RESNET_DIMENSION
    model_name = "torchvision/resnet50"
    WEIGHTS = "IMAGENET1K_V2"

    def _load(self) -> EmbeddingInfo:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        self.device = resolve_device(self._requested_device)
        self._torch = torch
        weights = ResNet50_Weights[self.WEIGHTS]
        net = resnet50(weights=weights)
        net.fc = torch.nn.Identity()  # drop the ImageNet classifier: the embedding is the 2048-d pooled feature
        self._net = net.eval().to(self.device)
        self._transform = weights.transforms()
        return EmbeddingInfo(ModelName.RESNET50, self.model_name, f"torchvision-{self.WEIGHTS}", self.dimension)

    def _embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch = self._torch.stack([self._transform(im) for im in images]).to(self.device)
        with self._torch.inference_mode():
            features = self._net(batch)
            out = features.flatten(1).float().cpu().numpy().copy()
        del batch, features
        return out


def _hub_revision(repo_id: str) -> str:
    """Commit hash of the cached Hugging Face snapshot (what the weights actually were), or ``unknown``."""
    try:
        from huggingface_hub import snapshot_download

        return os.path.basename(snapshot_download(repo_id, local_files_only=True))[:12]
    except Exception:  # noqa: BLE001
        return "unknown"


class EmbedderRegistry:
    """Owns the one instance of each encoder. Nothing is loaded until it is first needed."""

    def __init__(self, settings: Settings):
        self._embedders: dict[ModelName, _LazyEmbedder] = {
            ModelName.CLIP: ClipEmbedder(settings.country_matcher_clip_model, settings.country_matcher_device),
            ModelName.RESNET50: ResNetEmbedder(settings.country_matcher_device),
        }

    def get(self, model: ModelName) -> _LazyEmbedder:
        return self._embedders[model]

    def load_all(self) -> None:
        for e in self._embedders.values():
            e.load()

    def describe(self) -> dict[ModelName, dict]:
        return {m: {"model_name": e.model_name, "dimension": e.dimension, "loaded": e.loaded} for m, e in self._embedders.items()}
