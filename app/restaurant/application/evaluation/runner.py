"""Executes an evaluation run against the real configured pipeline (no mocks, no random values)."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from app.errors import ConflictError, ServiceNotConfigured
from app.restaurant.application import prompts
from app.restaurant.application.evaluation.ports import RAGAS_METRIC_KEYS, RagasEvaluator, RagasInput
from app.restaurant.application.rag.jobs import UnitOfWork, run_index_build
from app.restaurant.domain.evaluation.entities import EvaluationRun, QueryResult
from app.restaurant.domain.evaluation.metrics import K_VALUES, mean_ignoring_none, retrieval_metrics
from app.restaurant.domain.rag.entities import IndexStatus, JobStatus
from app.restaurant.domain.rag.ports import LLMClient
from app.restaurant.domain.rag.ranking import rank_dishes
from app.restaurant.domain.rag.value_objects import ChunkingConfig, RetrievalConfig

log = logging.getLogger(__name__)
RETRIEVAL_METRIC_KEYS = tuple(f"{m}_at_{k}" for k in K_VALUES for m in ("precision", "recall"))


class EvaluationRunner:
    def __init__(self, uow: UnitOfWork, llm: LLMClient, ragas: RagasEvaluator):
        self._uow, self._llm, self._ragas = uow, llm, ragas

    async def _progress(self, run: EvaluationRun, stage: str, done: int, total: int) -> None:
        run.progress = {"stage": stage, "done": done, "total": total}
        async with self._uow() as s:
            await s.eval_repo.update_progress(run.id, run.progress)

    async def execute(self, run: EvaluationRun) -> None:
        t0 = time.perf_counter()
        results: list[QueryResult] = []
        try:
            if not self._llm.is_configured:
                raise ServiceNotConfigured("LLM is not configured; evaluation needs it to generate the answers shown per question.")
            snap = run.config_snapshot
            chunking = ChunkingConfig.model_validate(snap["chunking"])
            retrieval = RetrievalConfig.model_validate(snap["retrieval"])
            fingerprint = snap["index_fingerprint"]

            await self._ensure_index(run, chunking)
            async with self._uow() as s:
                questions = await s.eval_repo.list_questions(run.dataset_id)
            limit = run.settings.get("question_limit")
            if limit:
                questions = questions[:limit]
            total = len(questions)

            # Rank metrics need the top-10 dishes; the LLM context only uses the configured top_k chunks.
            deep = retrieval.model_copy(update={"top_k": max(retrieval.top_k, 10), "candidate_count": max(retrieval.candidate_count, 10)})
            for i, q in enumerate(questions):
                await self._progress(run, "retrieval_and_generation", i, total)
                results.append(await self._run_question(run, q, chunking, deep, retrieval.top_k, fingerprint))
            await self._progress(run, "retrieval_and_generation", total, total)

            ok = [r for r in results if not r.error]
            if not ok:
                raise ConflictError("Every question failed; see per-query errors.")

            await self._progress(run, "ragas", 0, len(ok))
            ragas_error: str | None = None
            try:
                async with self._uow() as s:
                    samples = [
                        RagasInput(
                            r.query, r.contexts,
                            await s.retrieval.reference_contexts(fingerprint, r.relevant_dish_ids),
                        )
                        for r in ok
                    ]
                scores = await self._ragas.evaluate(samples)
                for r, sc in zip(ok, scores, strict=True):
                    r.ragas_scores = sc
            except Exception as exc:
                ragas_error = f"RAGAS evaluation failed: {type(exc).__name__}: {exc}"
                log.exception("ragas failed")
            await self._progress(run, "ragas", len(ok), len(ok))

            run.metrics = self._aggregate(results)
            run.duration_seconds = round(time.perf_counter() - t0, 2)
            run.metrics["duration_seconds"] = run.duration_seconds
            run.completed_at = datetime.now(UTC)
            if ragas_error:
                run.status, run.error = JobStatus.FAILED, ragas_error[:2000]
            else:
                run.status = JobStatus.COMPLETED
                run.progress = {"stage": "completed", "done": total, "total": total}
        except Exception as exc:
            log.exception("evaluation run failed", extra={"run_id": run.id})
            run.status, run.error = JobStatus.FAILED, f"{type(exc).__name__}: {exc}"[:2000]
            run.duration_seconds = round(time.perf_counter() - t0, 2)
            run.completed_at = datetime.now(UTC)
            if results:
                run.metrics = self._aggregate(results)
        async with self._uow() as s:
            if results:
                await s.eval_repo.add_query_results(results)
            await s.eval_repo.save_run(run)

    # ------------------------------------------------------------------

    async def _ensure_index(self, run: EvaluationRun, chunking: ChunkingConfig) -> None:
        async with self._uow() as s:
            index, _ = await s.indexing.request_index(chunking)
        if index.fingerprint != run.config_snapshot["index_fingerprint"]:
            raise ConflictError("The menu or embedding model changed since this run was queued; re-queue the run.")
        if index.status == IndexStatus.READY:
            return
        await self._progress(run, "indexing", 0, 0)
        async with self._uow() as s:
            await s.index_repo.update_status(index.id, IndexStatus.BUILDING, progress={"stage": "starting"})
        await run_index_build(self._uow, index)

    async def _run_question(self, run, q, chunking, deep: RetrievalConfig, context_k: int, fingerprint) -> QueryResult:
        query = q.retrieval_query()
        res = QueryResult(
            run_id=run.id, question_key=q.key, category=q.category, question=q.question, query=query,
            relevant_dish_ids=list(q.relevant_dish_ids), retrieved_dish_ids=[], contexts=[], answer="",
        )
        try:
            async with self._uow() as s:
                trace = await s.retrieval.retrieve(query, chunking=chunking, retrieval=deep, index_fingerprint=fingerprint)
            res.latency_ms = trace.latency_ms
            res.retrieved_dish_ids = rank_dishes([c.dish_ids for c in trace.chunks])  # dish-level attribution policy
            res.retrieval_metrics = retrieval_metrics(res.retrieved_dish_ids, q.relevant_dish_ids)
            context_chunks = trace.chunks[:context_k]
            res.contexts = [c.content for c in context_chunks]
            res.trace = {
                "query_variants": trace.query_variants, "final_score_type": trace.final_score_type,
                "latency_breakdown_ms": trace.latency_breakdown_ms, "context_chunk_ids": [c.chunk_id for c in context_chunks],
                "chunks": [
                    {"rank": c.rank, "chunk_id": c.chunk_id, "dish_ids": c.dish_ids, "scores": c.scores,
                     "in_context": c.rank <= context_k}
                    for c in trace.chunks
                ],
            }
            context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(res.contexts, start=1))
            res.answer = await self._llm.generate(
                prompts.EVAL_ANSWER.system, prompts.EVAL_ANSWER_USER.format(question=query, context=context)
            )
        except Exception as exc:  # noqa: BLE001 - one bad question must not sink the whole run
            log.warning("question failed", extra={"run_id": run.id, "question": q.key, "error": str(exc)[:300]})
            res.error = f"{type(exc).__name__}: {exc}"[:1000]
        return res

    @staticmethod
    def _aggregate(results: list[QueryResult]) -> dict[str, float | None]:
        m: dict[str, float | None] = {}
        for k in RETRIEVAL_METRIC_KEYS:
            m[k] = mean_ignoring_none(r.retrieval_metrics.get(k) for r in results if not r.error)
        for k in RAGAS_METRIC_KEYS:
            m[k] = mean_ignoring_none(r.ragas_scores.get(k) for r in results if not r.error)
        lat = [r.latency_ms for r in results if not r.error]
        m["avg_retrieval_latency_ms"] = round(sum(lat) / len(lat), 2) if lat else None
        m["questions_total"] = float(len(results))
        m["questions_failed"] = float(sum(1 for r in results if r.error))
        m["questions_scored_retrieval"] = float(sum(1 for r in results if not r.error and r.retrieval_metrics.get("precision_at_5") is not None))
        return m
