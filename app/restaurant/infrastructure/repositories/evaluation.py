from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.restaurant.domain.evaluation.entities import (
    EvalDataset,
    EvalQuestion,
    EvaluationRun,
    LiveEvaluation,
    QueryResult,
)
from app.restaurant.domain.evaluation.repositories import EvaluationRepository
from app.restaurant.domain.rag.entities import JobStatus
from app.restaurant.infrastructure.models import (
    EvaluationDatasetModel,
    EvaluationQueryResultModel,
    EvaluationQuestionModel,
    EvaluationRunModel,
    LiveEvaluationModel,
)


def _live(m: LiveEvaluationModel) -> LiveEvaluation:
    return LiveEvaluation(
        id=str(m.id), session_id=str(m.session_id), message_id=str(m.message_id),
        config_snapshot=dict(m.config_snapshot or {}), question=m.question, query=m.query,
        filters=dict(m.filters or {}), answer=m.answer, contexts=list(m.contexts or []),
        status=JobStatus(m.status), stage=m.stage, relevant_dish_ids=list(m.relevant_dish_ids or []),
        retrieved_dish_ids=list(m.retrieved_dish_ids or []), retrieval_metrics=dict(m.retrieval_metrics or {}),
        ragas_scores=dict(m.ragas_scores or {}), latency_ms=m.latency_ms, judge_model=m.judge_model,
        error=m.error, created_at=m.created_at, completed_at=m.completed_at,
    )


def _ds(m: EvaluationDatasetModel, count: int = 0) -> EvalDataset:
    return EvalDataset(str(m.id), m.name, m.version, m.menu_dataset_version, m.content_hash, m.description, count)


def _q(m: EvaluationQuestionModel) -> EvalQuestion:
    return EvalQuestion(
        id=str(m.id), key=m.key, category=m.category, question=m.question,
        relevant_dish_ids=tuple(m.relevant_dish_ids or ()), reference_answer=m.reference_answer,
        required_constraints=dict(m.required_constraints or {}), expected_behavior=m.expected_behavior,
        history=tuple(m.history or ()), position=m.position,
    )


def _run(m: EvaluationRunModel) -> EvaluationRun:
    return EvaluationRun(
        id=str(m.id), dataset_id=str(m.dataset_id), status=JobStatus(m.status), config_snapshot=dict(m.config_snapshot),
        fingerprint=m.fingerprint, settings=dict(m.settings or {}), progress=dict(m.progress or {}),
        metrics=dict(m.metrics or {}), error=m.error, created_at=m.created_at, started_at=m.started_at,
        completed_at=m.completed_at, duration_seconds=m.duration_seconds,
    )


def _qr(m: EvaluationQueryResultModel) -> QueryResult:
    return QueryResult(
        run_id=str(m.run_id), question_key=m.question_key, category=m.category, question=m.question,
        query=m.query, relevant_dish_ids=list(m.relevant_dish_ids or []),
        retrieved_dish_ids=list(m.retrieved_dish_ids or []), contexts=list(m.contexts or []),
        answer=m.answer, retrieval_metrics=dict(m.retrieval_metrics or {}),
        ragas_scores=dict(m.ragas_scores or {}), latency_ms=m.latency_ms, trace=dict(m.trace or {}),
        error=m.error,
    )


class SqlEvaluationRepository(EvaluationRepository):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def get_dataset(self, name: str | None = None, version: str | None = None) -> EvalDataset | None:
        stmt = select(EvaluationDatasetModel)
        if name:
            stmt = stmt.where(EvaluationDatasetModel.name == name)
        if version:
            stmt = stmt.where(EvaluationDatasetModel.version == version)
        m = (await self._s.execute(stmt.order_by(EvaluationDatasetModel.created_at.desc()).limit(1))).scalar_one_or_none()
        if m is None:
            return None
        n = len((await self._s.execute(select(EvaluationQuestionModel.id).where(EvaluationQuestionModel.dataset_id == m.id))).all())
        return _ds(m, n)

    async def get_dataset_by_id(self, dataset_id: str) -> EvalDataset | None:
        m = await self._s.get(EvaluationDatasetModel, uuid.UUID(dataset_id))
        return _ds(m) if m else None

    async def list_questions(self, dataset_id: str) -> list[EvalQuestion]:
        rows = (
            await self._s.execute(
                select(EvaluationQuestionModel)
                .where(EvaluationQuestionModel.dataset_id == uuid.UUID(dataset_id))
                .order_by(EvaluationQuestionModel.position)
            )
        ).scalars()
        return [_q(m) for m in rows]

    async def add_run(self, run: EvaluationRun) -> None:
        self._s.add(
            EvaluationRunModel(
                id=uuid.UUID(run.id),
                dataset_id=uuid.UUID(run.dataset_id), status=run.status.value, config_snapshot=run.config_snapshot,
                fingerprint=run.fingerprint, settings=run.settings, progress=run.progress,
            )
        )
        await self._s.flush()

    async def get_run(self, run_id: str) -> EvaluationRun | None:
        try:
            rid = uuid.UUID(run_id)
        except ValueError:
            return None
        m = await self._s.get(EvaluationRunModel, rid)
        if m is not None:
            await self._s.refresh(m)
        return _run(m) if m else None

    async def list_runs(self, limit: int = 100) -> list[EvaluationRun]:
        stmt = select(EvaluationRunModel).order_by(EvaluationRunModel.created_at.desc()).limit(limit)
        return [_run(m) for m in (await self._s.execute(stmt)).scalars()]

    async def save_run(self, run: EvaluationRun) -> None:
        await self._s.execute(
            update(EvaluationRunModel).where(EvaluationRunModel.id == uuid.UUID(run.id)).values(
                status=run.status.value, progress=run.progress, metrics=run.metrics, error=run.error,
                settings=run.settings, started_at=run.started_at, completed_at=run.completed_at,
                duration_seconds=run.duration_seconds,
            )
        )

    async def update_progress(self, run_id: str, progress: dict) -> None:
        await self._s.execute(
            update(EvaluationRunModel).where(EvaluationRunModel.id == uuid.UUID(run_id)).values(progress=progress)
        )

    async def add_query_results(self, results: list[QueryResult]) -> None:
        self._s.add_all(
            EvaluationQueryResultModel(
                run_id=uuid.UUID(r.run_id), question_key=r.question_key, category=r.category, question=r.question,
                query=r.query, relevant_dish_ids=r.relevant_dish_ids, retrieved_dish_ids=r.retrieved_dish_ids,
                contexts=r.contexts, answer=r.answer, retrieval_metrics=r.retrieval_metrics,
                ragas_scores=r.ragas_scores, latency_ms=r.latency_ms, trace=r.trace, error=r.error,
            )
            for r in results
        )
        await self._s.flush()

    async def list_query_results(self, run_id: str) -> list[QueryResult]:
        rows = (
            await self._s.execute(
                select(EvaluationQueryResultModel)
                .where(EvaluationQueryResultModel.run_id == uuid.UUID(run_id))
                .order_by(EvaluationQueryResultModel.question_key)
            )
        ).scalars()
        return [_qr(m) for m in rows]

    async def claim_pending_run(self) -> EvaluationRun | None:
        m = (
            await self._s.execute(
                select(EvaluationRunModel)
                .where(EvaluationRunModel.status == JobStatus.PENDING.value)
                .order_by(EvaluationRunModel.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if m is None:
            return None
        m.status = JobStatus.RUNNING.value
        m.started_at = datetime.now(UTC)
        m.progress = {"stage": "starting", "done": 0, "total": 0}
        await self._s.flush()
        return _run(m)

    async def requeue_interrupted(self) -> int:
        res = await self._s.execute(
            update(EvaluationRunModel)
            .where(EvaluationRunModel.status == JobStatus.RUNNING.value)
            .values(status=JobStatus.PENDING.value)
        )
        return res.rowcount or 0

    # ---- live evaluation -----------------------------------------------------------------------

    async def add_live(self, item: LiveEvaluation) -> None:
        self._s.add(
            LiveEvaluationModel(
                id=uuid.UUID(item.id), session_id=uuid.UUID(item.session_id), message_id=uuid.UUID(item.message_id),
                config_snapshot=item.config_snapshot, question=item.question,
                query=item.query, filters=item.filters, answer=item.answer, contexts=item.contexts,
                status=item.status.value, stage=item.stage, judge_model=item.judge_model,
                latency_ms=item.latency_ms, created_at=item.created_at,
            )
        )
        await self._s.flush()

    async def save_live(self, item: LiveEvaluation) -> None:
        await self._s.execute(
            update(LiveEvaluationModel).where(LiveEvaluationModel.id == uuid.UUID(item.id)).values(
                status=item.status.value, stage=item.stage, relevant_dish_ids=item.relevant_dish_ids,
                retrieved_dish_ids=item.retrieved_dish_ids, retrieval_metrics=item.retrieval_metrics,
                ragas_scores=item.ragas_scores, latency_ms=item.latency_ms, judge_model=item.judge_model,
                error=item.error, completed_at=item.completed_at,
            )
        )

    async def list_live(self, limit: int = 50) -> list[LiveEvaluation]:
        stmt = select(LiveEvaluationModel).order_by(LiveEvaluationModel.created_at.desc()).limit(limit)
        return [_live(m) for m in (await self._s.execute(stmt)).scalars()]

    async def list_live_by_message_ids(self, message_ids: list[str]) -> list[LiveEvaluation]:
        ids = [uuid.UUID(i) for i in message_ids]
        if not ids:
            return []
        stmt = select(LiveEvaluationModel).where(LiveEvaluationModel.message_id.in_(ids))
        return [_live(m) for m in (await self._s.execute(stmt)).scalars()]

    async def release_awaiting(self, message_ids: list[str]) -> int:
        ids = [uuid.UUID(i) for i in message_ids]
        if not ids:
            return 0
        res = await self._s.execute(
            update(LiveEvaluationModel)
            .where(LiveEvaluationModel.message_id.in_(ids), LiveEvaluationModel.status == JobStatus.AWAITING.value)
            .values(status=JobStatus.PENDING.value, stage="queued")
        )
        return res.rowcount or 0

    async def purge_awaiting(self, older_than: datetime) -> int:
        res = await self._s.execute(
            delete(LiveEvaluationModel).where(
                LiveEvaluationModel.status == JobStatus.AWAITING.value, LiveEvaluationModel.created_at < older_than
            )
        )
        return res.rowcount or 0

    async def redact_live(self, item_id: str) -> None:
        await self._s.execute(
            update(LiveEvaluationModel).where(LiveEvaluationModel.id == uuid.UUID(item_id))
            .values(question="", query="", answer="", contexts=[])
        )

    async def claim_pending_live(self) -> LiveEvaluation | None:
        m = (
            await self._s.execute(
                select(LiveEvaluationModel)
                .where(LiveEvaluationModel.status == JobStatus.PENDING.value)
                .order_by(LiveEvaluationModel.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if m is None:
            return None
        m.status = JobStatus.RUNNING.value
        m.stage = "judging_relevance"
        await self._s.flush()
        return _live(m)

    async def requeue_interrupted_live(self) -> int:
        res = await self._s.execute(
            update(LiveEvaluationModel)
            .where(LiveEvaluationModel.status == JobStatus.RUNNING.value)
            .values(status=JobStatus.PENDING.value, stage="queued")
        )
        return res.rowcount or 0

    async def clear_live(self) -> int:
        res = await self._s.execute(delete(LiveEvaluationModel))
        return res.rowcount or 0
