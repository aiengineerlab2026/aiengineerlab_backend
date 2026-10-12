from __future__ import annotations

import uuid
from typing import Any

from app.application.evaluation.ports import RAGAS_METRIC_KEYS, RagasEvaluator
from app.application.rag.config import ConfigService
from app.domain.errors import ConflictError, NotFoundError, ValidationFailed
from app.domain.evaluation.entities import EvalDataset, EvalQuestion, EvaluationRun, QueryResult
from app.domain.evaluation.repositories import EvaluationRepository
from app.domain.rag.entities import JobStatus
from app.domain.rag.ports import LLMClient

LOWER_IS_BETTER = {"avg_retrieval_latency_ms", "duration_seconds"}
COMPARE_METRICS = (
    "precision_at_5", "recall_at_5", "precision_at_10", "recall_at_10", *RAGAS_METRIC_KEYS,
    "avg_retrieval_latency_ms", "duration_seconds",
)


class EvaluationService:
    def __init__(
        self, repo: EvaluationRepository, config: ConfigService, llm: LLMClient,
        ragas: RagasEvaluator, dataset_name: str,
    ):
        self._repo, self._config, self._llm, self._ragas = repo, config, llm, ragas
        self._dataset_name = dataset_name

    async def dataset(self) -> EvalDataset:
        ds = await self._repo.get_dataset(self._dataset_name)
        if ds is None:
            raise NotFoundError("Evaluation dataset is not seeded. Run `python -m app.cli seed`.")
        return ds

    async def questions(self) -> list[EvalQuestion]:
        return await self._repo.list_questions((await self.dataset()).id)

    async def create_run(self, *, question_limit: int | None = None) -> EvaluationRun:
        if question_limit is not None and question_limit < 1:
            raise ValidationFailed("question_limit must be >= 1.")
        if not self._llm.is_configured:
            raise ConflictError(
                "Evaluation needs a configured LLM to generate the answers shown per question. Set LLM_API_KEY in backend/.env.",
                details={"code": "llm_not_configured"},
            )
        cfg = await self._config.get()  # the live configuration, frozen into this run
        index = await self._config.ensure_index()  # queue the index build if it's missing
        dataset = await self.dataset()
        total = dataset.question_count
        if question_limit:
            total = min(total, question_limit)
        run = EvaluationRun(
            id=str(uuid.uuid4()),
            dataset_id=dataset.id,
            status=JobStatus.PENDING,
            fingerprint=index.fingerprint,
            config_snapshot={
                "chunking": cfg.chunking.model_dump(mode="json"),
                "retrieval": cfg.retrieval.model_dump(mode="json"),
                "embedding": cfg.embedding.model_dump(mode="json"),
                "llm_model": cfg.llm_model,
                "dataset_version": f"{dataset.name}:{dataset.version}",
                "menu_dataset_version": dataset.menu_dataset_version,
                "index_fingerprint": index.fingerprint,
            },
            settings={
                "answer_model": self._llm.model_name,
                "ragas_version": self._ragas.version,
                "ragas_metrics": "non-llm-context-v1",  # string-similarity metrics: no judge LLM, no embeddings
                "k_values": [5, 10],
                "question_limit": question_limit,
                "question_count": total,
            },
            progress={"stage": "queued", "done": 0, "total": total},
        )
        await self._repo.add_run(run)
        return run

    async def get_run(self, run_id: str) -> EvaluationRun:
        run = await self._repo.get_run(run_id)
        if run is None:
            raise NotFoundError(f"Evaluation run '{run_id}' not found.")
        return run

    async def list_runs(self) -> list[EvaluationRun]:
        return await self._repo.list_runs()

    async def results(self, run_id: str) -> list[QueryResult]:
        await self.get_run(run_id)
        return await self._repo.list_query_results(run_id)

    async def compare(self, run_ids: list[str]) -> dict[str, Any]:
        if len(run_ids) < 2:
            raise ValidationFailed("Select at least two evaluation runs to compare.")
        runs = [await self.get_run(r) for r in dict.fromkeys(run_ids)]
        not_done = [r.id for r in runs if r.status != JobStatus.COMPLETED]
        if not_done:
            raise ConflictError("Only completed runs can be compared.", details={"run_ids": not_done})
        warnings: list[str] = []
        first = runs[0]
        for label, getter in (
            ("evaluation dataset", lambda r: r.dataset_id),
            ("question count", lambda r: r.settings.get("question_count")),
            ("RAGAS metric set", lambda r: r.settings.get("ragas_metrics")),
            ("answer model", lambda r: r.settings.get("answer_model")),
            ("index embedding model", lambda r: r.config_snapshot.get("embedding", {}).get("model")),
        ):
            if len({str(getter(r)) for r in runs}) > 1:
                warnings.append(f"Runs differ in {label}; scores are not strictly comparable.")
        best: dict[str, str] = {}
        for m in COMPARE_METRICS:
            vals = [(r.id, r.metrics.get(m)) for r in runs if r.metrics.get(m) is not None]
            if len(vals) >= 2:
                pick = min if m in LOWER_IS_BETTER else max
                best[m] = pick(vals, key=lambda kv: kv[1])[0]
        return {
            "comparable": not warnings,
            "warnings": warnings,
            "dataset_id": first.dataset_id,
            "runs": runs,
            "best_by_metric": best,  # highest per metric (lowest for latency/duration); not an overall winner
        }
