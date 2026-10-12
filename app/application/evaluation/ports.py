from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RagasInput:
    user_input: str
    retrieved_contexts: list[str]  # the exact chunks that were given to the LLM, in rank order
    reference_contexts: list[str]  # the index chunks that cover the dishes that should have been found
    response: str = ""  # the answer the customer saw (only needed by the LLM-based metrics)


# RAGAS metrics that need no LLM and no embeddings (string similarity between retrieved and reference contexts).
RAGAS_METRIC_KEYS = ("non_llm_context_precision", "non_llm_context_recall")

# RAGAS metrics that call the judge LLM (and, for answer relevancy, the embedding model). Only used for live
# evaluation, which is capped per client because each question costs roughly a dozen model calls.
LLM_RAGAS_METRIC_KEYS = ("faithfulness", "answer_relevancy", "llm_context_precision", "llm_context_recall")

LIVE_RAGAS_METRIC_KEYS = (*RAGAS_METRIC_KEYS, *LLM_RAGAS_METRIC_KEYS)


class RagasEvaluator(ABC):
    """Computes RAGAS metrics for a batch of samples (one result dict per sample, None = not computable)."""

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    async def evaluate(self, samples: list[RagasInput]) -> list[dict[str, float | None]]:
        """LLM-free: ``non_llm_context_precision`` and ``non_llm_context_recall`` per sample."""

    async def evaluate_llm(self, samples: list[RagasInput]) -> list[dict[str, float | None]]:
        """LLM-based: the ``LLM_RAGAS_METRIC_KEYS`` per sample. Adapters without an LLM return empty dicts."""
        return [{} for _ in samples]
