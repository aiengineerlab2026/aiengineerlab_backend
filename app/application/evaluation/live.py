"""Live evaluation: scores the questions real customers asked, as they are answered.

There is no pre-written ground truth, so:
* **Relevant dishes** are judged by the judge LLM against the complete menu (one structured call per question).
  Precision/Recall@K are then computed exactly like in the benchmark, with the same deterministic functions.
* **RAGAS** runs its LLM-free metrics (non-LLM context precision and recall: the chunks given to the LLM compared by
  string similarity with the index chunks that cover the judged-relevant dishes) and its LLM-based ones
  (faithfulness, answer relevancy, LLM context precision and recall) with the judge LLM.

The relevance judgement and the LLM-based metrics use an LLM, so the numbers are estimates, not human labels - the
UI says so. Because that is ~a dozen model calls per question, each client IP may trigger a limited number.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.application import prompts
from app.application.evaluation.ports import LIVE_RAGAS_METRIC_KEYS, RagasEvaluator, RagasInput
from app.application.rag.jobs import UnitOfWork
from app.domain.errors import ServiceNotConfigured
from app.domain.evaluation.entities import LiveEvaluation
from app.domain.evaluation.metrics import mean_ignoring_none, retrieval_metrics
from app.domain.rag.entities import JobStatus
from app.domain.rag.ports import LLMClient
from app.domain.rag.ranking import rank_dishes
from app.domain.rag.value_objects import ChunkingConfig, RetrievalConfig
from app.domain.restaurant.entities import Dish
from app.domain.restaurant.filters import DishFilter

log = logging.getLogger(__name__)

RETRIEVAL_LIVE_KEYS = tuple(f"{m}_at_{k}" for k in (5, 10) for m in ("precision", "recall"))


class RelevantDishes(BaseModel):
    relevant_dish_ids: list[str] = Field(default_factory=list, description="Ids of every relevant dish, from the menu.")


def menu_table(dishes: list[Dish]) -> str:
    return "\n".join(
        f"{d.id} | {d.name} | {d.category} | {float(d.price):g} | {d.calories} | {d.protein_grams} | "
        f"{d.spice_level} | {'yes' if d.available else 'no'} | {','.join(d.allergens) or '-'} | "
        f"{','.join(d.tags) or '-'} | {', '.join(d.ingredients)}"
        for d in dishes
    )


def config_key(snapshot: dict) -> str:
    """Identity of a configuration: same index (chunking + embedding + dataset) and same retrieval settings."""
    return json.dumps({"index": snapshot.get("index_fingerprint"), "retrieval": snapshot.get("retrieval")}, sort_keys=True)


def summarize(items: list[LiveEvaluation]) -> dict[str, float | None]:
    """Mean of every metric over the items that have it (None-valued metrics are excluded, never zero-filled)."""
    done = [i for i in items if i.status == JobStatus.COMPLETED or i.retrieval_metrics or i.ragas_scores]
    out: dict[str, float | None] = {}
    for k in RETRIEVAL_LIVE_KEYS:
        out[k] = mean_ignoring_none(i.retrieval_metrics.get(k) for i in done)
    for k in LIVE_RAGAS_METRIC_KEYS:
        out[k] = mean_ignoring_none(i.ragas_scores.get(k) for i in done)
    lat = [i.latency_ms for i in done if i.latency_ms]
    out["avg_retrieval_latency_ms"] = round(sum(lat) / len(lat), 2) if lat else None
    return out


class LiveEvaluator:
    def __init__(self, uow: UnitOfWork, judge: LLMClient, ragas: RagasEvaluator):
        self._uow, self._judge, self._ragas = uow, judge, ragas

    async def _save(self, item: LiveEvaluation) -> None:
        async with self._uow() as s:  # short transaction per stage so the UI sees progress immediately
            await s.eval_repo.save_live(item)

    async def process(self, item: LiveEvaluation) -> None:
        item.judge_model = self._judge.model_name
        try:
            if not self._judge.is_configured:
                raise ServiceNotConfigured("Live evaluation needs a judge LLM (set LLM_API_KEY).")

            # 1) which dishes should a perfect answer contain?
            item.stage = "judging_relevance"
            await self._save(item)
            async with self._uow() as s:
                dishes = await s.menu_repo.list_dishes()
            valid = {d.id for d in dishes}
            verdict = await self._judge.generate_structured(
                prompts.RELEVANCE_JUDGE.system,
                prompts.RELEVANCE_JUDGE_USER.format(question=item.question, query=item.query, menu=menu_table(dishes)),
                RelevantDishes,
            )
            item.relevant_dish_ids = [i for i in dict.fromkeys(verdict.relevant_dish_ids) if i in valid]

            # 2) depth-10 retrieval with the same configuration and hard filters the customer got
            item.stage = "retrieval"
            await self._save(item)
            snap = item.config_snapshot
            chunking = ChunkingConfig.model_validate(snap["chunking"])
            retrieval = RetrievalConfig.model_validate(snap["retrieval"])
            deep = retrieval.model_copy(
                update={"top_k": max(retrieval.top_k, 10), "candidate_count": max(retrieval.candidate_count, 10)}
            )
            flt = DishFilter.from_dict(item.filters) if item.filters else None
            async with self._uow() as s:
                trace = await s.retrieval.retrieve(
                    item.query, chunking=chunking, retrieval=deep,
                    index_fingerprint=snap["index_fingerprint"], dish_filter=flt,
                )
            item.retrieved_dish_ids = rank_dishes([c.dish_ids for c in trace.chunks])
            item.retrieval_metrics = retrieval_metrics(item.retrieved_dish_ids, item.relevant_dish_ids)
            item.stage = "ragas"
            await self._save(item)  # retrieval metrics are visible while RAGAS is still running

            # 3) LLM-free RAGAS: chunks the customer's answer was built from vs. the judged-relevant dishes
            async with self._uow() as s:
                reference = await s.retrieval.reference_contexts(snap["index_fingerprint"], item.relevant_dish_ids)
            sample = RagasInput(item.question, item.contexts, reference, item.answer)
            scores = await self._ragas.evaluate([sample])
            item.ragas_scores = dict(scores[0]) if scores else {}
            item.stage = "ragas_llm"
            await self._save(item)  # the LLM-free scores are visible while the LLM-based ones are computed

            # 4) LLM-based RAGAS (faithfulness, answer relevancy, LLM context precision/recall)
            llm_scores = await self._ragas.evaluate_llm([sample])
            if llm_scores:
                item.ragas_scores.update(llm_scores[0])
            item.status, item.stage = JobStatus.COMPLETED, "done"
        except Exception as exc:  # noqa: BLE001 - one bad evaluation must never take the worker down
            log.warning("live evaluation failed", extra={"id": item.id, "error": str(exc)[:300]})
            item.status, item.stage = JobStatus.FAILED, "failed"
            item.error = f"{type(exc).__name__}: {exc}"[:1500]
        item.completed_at = datetime.now(UTC)
        await self._save(item)
        async with self._uow() as s:  # the conversation lives in the customer's browser; only the scores stay here
            await s.eval_repo.redact_live(item.id)
