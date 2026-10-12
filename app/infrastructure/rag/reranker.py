from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from app.domain.errors import ExternalServiceError
from app.domain.rag.ports import Reranker

log = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    """Local cross-encoder (sentence-transformers). Scores are raw logits, only comparable within a call."""

    def __init__(self, model_name: str):
        self._name = model_name
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._name

    def _load(self) -> Any:
        if self._model is None:
            import math

            from sentence_transformers import CrossEncoder

            log.info("loading reranker", extra={"model": self._name})
            # Prefer the classic pytorch .bin checkpoint: on some platform/torch/transformers combinations
            # (observed on macOS 12) memory-mapped safetensors yield NaN scores or SIGBUS. Fall back to the
            # default loader for models that only ship safetensors.
            try:
                model = CrossEncoder(self._name, model_kwargs={"use_safetensors": False})
            except (OSError, ValueError):
                model = CrossEncoder(self._name)
            probe = float(model.predict([("probe query", "probe document")], show_progress_bar=False)[0])
            if not math.isfinite(probe):
                raise ExternalServiceError(f"Reranker '{self._name}' produced non-finite scores on this platform.")
            self._model = model
        return self._model

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        def _run() -> list[float]:
            return [float(s) for s in self._load().predict([(query, d) for d in documents], show_progress_bar=False)]

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            raise ExternalServiceError(f"Reranker '{self._name}' failed: {str(exc)[:300]}") from exc


class RerankerRegistry:
    """Caches one loaded model per name; the reranker is selectable in the live configuration."""

    def __init__(self) -> None:
        self._cache: dict[str, Reranker] = {}

    def get(self, model_name: str) -> Reranker:
        if model_name not in self._cache:
            self._cache[model_name] = CrossEncoderReranker(model_name)
        return self._cache[model_name]
