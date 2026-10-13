"""In-memory vector index of the job postings: one L2-normalised matrix per embedding model.

The corpus is a few dozen jobs, so a numpy matrix with a dot product (= cosine similarity on unit vectors) is exact and
instant; no vector database is needed. Each embedding model is embedded lazily on first use and kept for the process.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence

import numpy as np

from app.myresume.domain.models import Job
from app.restaurant.domain.rag.value_objects import EmbeddingSpec
from app.restaurant.infrastructure.rag.embeddings import LangChainEmbedder

log = logging.getLogger(__name__)


class VectorIndex:
    def __init__(self, job_ids: list[str], matrix: np.ndarray):
        self.job_ids, self.matrix = job_ids, matrix

    def similarities(self, query_vec: Sequence[float]) -> np.ndarray:
        q = np.asarray(query_vec, dtype=float)
        norm = np.linalg.norm(q)
        return self.matrix @ (q / norm if norm else q)


class VectorIndexRegistry:
    def __init__(self, jobs: Sequence[Job], embedder_for: Callable[[EmbeddingSpec], LangChainEmbedder]):
        self._jobs, self._embedder_for = list(jobs), embedder_for
        self._indexes: dict[EmbeddingSpec, VectorIndex] = {}
        self._locks: dict[EmbeddingSpec, asyncio.Lock] = {}

    async def get(self, spec: EmbeddingSpec) -> VectorIndex:
        if spec in self._indexes:
            return self._indexes[spec]
        async with self._locks.setdefault(spec, asyncio.Lock()):
            if spec not in self._indexes:  # another request may have built it while this one waited
                vectors = await self._embedder_for(spec).embed_documents([j.text for j in self._jobs])
                matrix = np.asarray(vectors, dtype=float)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                self._indexes[spec] = VectorIndex([j.id for j in self._jobs], matrix / np.where(norms == 0, 1, norms))
                log.info("embedded %d jobs", len(self._jobs), extra={"model": spec.model})
        return self._indexes[spec]
