"""Adapter over the official RAGAS framework, limited to the metrics that need no LLM and no embeddings.

``NonLLMContextPrecisionWithReference`` and ``NonLLMContextRecall`` compare each retrieved chunk with each reference
chunk by Levenshtein similarity. Nothing here calls a model, so scores are deterministic, free and reproducible.
Faithfulness and answer relevancy have no LLM-free form, so they are intentionally not computed.

The reference chunks are the index's own chunks that cover the relevant dishes, so a retrieved chunk either *is* one of
them (similarity 1.0) or is a different chunk. RAGAS's default threshold of 0.5 cannot be used: every dish is rendered
from the same template, so two *different* dishes already score ~0.55 on average (up to ~0.74) and nearly everything
would count as a match. 0.9 accepts only (near-)identical text.
"""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from collections.abc import Callable
from typing import Any

from app.restaurant.application.evaluation.ports import LLM_RAGAS_METRIC_KEYS, RagasEvaluator, RagasInput

log = logging.getLogger(__name__)

# RAGAS metric name -> key stored in our results
_COLUMN_MAP = {
    "non_llm_context_precision_with_reference": "non_llm_context_precision",
    "non_llm_context_recall": "non_llm_context_recall",
}
MATCH_THRESHOLD = 0.9
_EMPTY: dict[str, float | None] = dict.fromkeys(_COLUMN_MAP.values())


def _clean(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 4)


_LLM_NONE: dict[str, float | None] = dict.fromkeys(LLM_RAGAS_METRIC_KEYS)
# RAGAS column -> our key, for the metrics that call the judge LLM
_LLM_COLUMN_MAP = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_without_reference": "llm_context_precision",
}
_LLM_RECALL_COLUMN = "context_recall"


class RagasFrameworkEvaluator(RagasEvaluator):
    """RAGAS metrics. The LLM-free ones (string similarity) always work; the LLM-based ones (faithfulness, answer
    relevancy, LLM context precision/recall) need the judge LLM and the embedding model, passed as lazy factories
    returning LangChain objects."""

    def __init__(
        self, llm_factory: Callable[[], Any] | None = None, embeddings_factory: Callable[[], Any] | None = None
    ):
        self._llm_factory, self._embeddings_factory = llm_factory, embeddings_factory

    @property
    def version(self) -> str:
        import ragas

        return ragas.__version__

    async def evaluate(self, samples: list[RagasInput]) -> list[dict[str, float | None]]:
        return await asyncio.to_thread(self._run, samples)  # string matching is CPU work; keep the event loop free

    def _run(self, samples: list[RagasInput]) -> list[dict[str, float | None]]:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        with warnings.catch_warnings():  # RAGAS 0.4 flags the (still supported) legacy import path
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

        out: list[dict[str, float | None]] = [dict(_EMPTY) for _ in samples]
        # No reference dishes -> nothing should have been found -> the metrics are undefined (not zero).
        # Nothing retrieved -> recall is 0, precision is undefined; RAGAS would raise on an empty list.
        scorable: list[int] = []
        for i, s in enumerate(samples):
            if not s.reference_contexts:
                continue
            if not s.retrieved_contexts:
                out[i]["non_llm_context_recall"] = 0.0
                continue
            scorable.append(i)
        if not scorable:
            return out

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=samples[i].user_input,
                    retrieved_contexts=samples[i].retrieved_contexts,
                    reference_contexts=samples[i].reference_contexts,
                )
                for i in scorable
            ]
        )
        result = evaluate(
            dataset,
            metrics=[
                NonLLMContextPrecisionWithReference(threshold=MATCH_THRESHOLD),
                NonLLMContextRecall(threshold=MATCH_THRESHOLD),
            ],
            raise_exceptions=True,  # no external service can fail here, so an error is a real bug: surface it
            show_progress=False,
            allow_nest_asyncio=False,
        )
        for i, row in zip(scorable, result.scores, strict=True):
            out[i] = {ours: _clean(row.get(theirs)) for theirs, ours in _COLUMN_MAP.items()}
        return out

    # ------------------------------------------------------------------ LLM-based metrics

    async def evaluate_llm(self, samples: list[RagasInput]) -> list[dict[str, float | None]]:
        if self._llm_factory is None or self._embeddings_factory is None:
            return [{} for _ in samples]
        try:
            return await asyncio.to_thread(self._run_llm, samples)
        except Exception as exc:  # noqa: BLE001 - quota / network trouble must not discard the LLM-free scores
            log.warning("LLM-based RAGAS metrics failed", extra={"error": str(exc)[:300]})
            return [dict(_LLM_NONE) for _ in samples]

    def _run_llm(self, samples: list[RagasInput]) -> list[dict[str, float | None]]:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.run_config import RunConfig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                LLMContextRecall,
                ResponseRelevancy,
            )

            llm = LangchainLLMWrapper(self._llm_factory())  # type: ignore[misc]
            embeddings = LangchainEmbeddingsWrapper(self._embeddings_factory())  # type: ignore[misc]

        out: list[dict[str, float | None]] = [dict(_LLM_NONE) for _ in samples]
        # Nothing retrieved: there is nothing to judge (and the reply is a fixed "not found"); recall is 0 if
        # something should have been found.
        scorable = [i for i, s in enumerate(samples) if s.retrieved_contexts and s.response.strip()]
        for i, s in enumerate(samples):
            if not s.retrieved_contexts and s.reference_contexts:
                out[i]["llm_context_recall"] = 0.0
        if not scorable:
            return out

        run_config = RunConfig(timeout=120, max_retries=3, max_workers=2)
        # One evaluate() call: RAGAS runs it on its own event loop, and async model clients cannot be reused on
        # another one. Samples without a reference get an empty one; their recall is discarded below.
        result = evaluate(
            EvaluationDataset(
                samples=[
                    SingleTurnSample(
                        user_input=samples[i].user_input, response=samples[i].response,
                        retrieved_contexts=samples[i].retrieved_contexts,
                        reference="\n\n".join(samples[i].reference_contexts),
                    )
                    for i in scorable
                ]
            ),
            # strictness=1: Gemini rejects the several-candidates-per-call mode answer relevancy uses by default
            metrics=[
                Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithoutReference(),
                LLMContextRecall(),
            ],
            llm=llm, embeddings=embeddings, run_config=run_config,
            raise_exceptions=False,  # one failed call leaves that metric empty instead of losing the rest
            show_progress=False, allow_nest_asyncio=False,
        )
        for i, row in zip(scorable, result.scores, strict=True):
            for theirs, ours in _LLM_COLUMN_MAP.items():
                out[i][ours] = _clean(row.get(theirs))
            if samples[i].reference_contexts:
                out[i]["llm_context_recall"] = _clean(row.get(_LLM_RECALL_COLUMN))
        return out
