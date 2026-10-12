from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response, status

from app.application.evaluation.live import config_key, summarize
from app.application.evaluation.ports import RAGAS_METRIC_KEYS
from app.application.evaluation.runner import RETRIEVAL_METRIC_KEYS
from app.domain.errors import RateLimited, ValidationFailed
from app.domain.evaluation.metrics import mean_ignoring_none
from app.domain.rag.entities import JobStatus
from app.presentation.api.v1.schemas import (
    ChatUsageOut,
    CompareOut,
    CreateEvaluationIn,
    DatasetOut,
    EvaluationResultsOut,
    EvaluationRunOut,
    LiveEvaluationOut,
    LiveEvaluationsOut,
    LiveGroupOut,
    LiveSummaryOut,
    QueryResultOut,
    RunLiveIn,
    RunLiveOut,
)
from app.presentation.deps import ServicesDep, client_ip

router = APIRouter(prefix="/evaluations", tags=["evaluation"])


@router.post("", response_model=EvaluationRunOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Queue an evaluation run (executed by the background worker; poll GET /evaluations/{id})")
async def create_evaluation(body: CreateEvaluationIn, svc: ServicesDep) -> EvaluationRunOut:
    return EvaluationRunOut.of(await svc.evaluation.create_run(question_limit=body.question_limit))


@router.get("", response_model=list[EvaluationRunOut])
async def list_evaluations(svc: ServicesDep) -> list[EvaluationRunOut]:
    return [EvaluationRunOut.of(r) for r in await svc.evaluation.list_runs()]


@router.get(
    "/live", response_model=LiveEvaluationsOut,
    summary="Live evaluation of the questions real customers asked (no predefined dataset)",
)
async def live_evaluations(
    request: Request, svc: ServicesDep, limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> LiveEvaluationsOut:
    repo = svc.eval_repo
    # Questions still waiting for the customer to press "Evaluate" are not evaluations yet.
    scored = [i for i in await repo.list_live(limit=2000) if i.status != JobStatus.AWAITING]
    items, everything = scored[:limit], scored  # everything: for the running averages
    groups: dict[str, list] = defaultdict(list)
    for i in everything:  # newest first, so each group's snapshot is its most recent one
        groups[config_key(i.config_snapshot)].append(i)
    by_status = Counter(i.status.value for i in everything)
    return LiveEvaluationsOut(
        usage=_usage(request),
        enabled=request.app.state.infra.settings.live_evaluation_enabled,
        items=[LiveEvaluationOut.of(i) for i in items],
        summary=LiveSummaryOut(
            pending=by_status["pending"], running=by_status["running"], completed=by_status["completed"], failed=by_status["failed"],
            metrics=summarize(everything),
            by_config=[
                LiveGroupOut(config_snapshot=rows[0].config_snapshot, count=len(rows), metrics=summarize(rows))
                for rows in groups.values()
            ],
            judge_model=request.app.state.infra.judge_llm.model_name,
        ),
    )


def _usage(request: Request) -> ChatUsageOut:
    infra = request.app.state.infra
    u = infra.eval_limiter.peek(client_ip(request, infra.settings.trust_proxy_headers))
    return ChatUsageOut(
        enabled=infra.eval_limiter.enabled, limit=u.limit, used=u.used, remaining=u.remaining,
        window_seconds=infra.settings.eval_rate_limit_window_seconds, reset_in_seconds=u.reset_in_seconds,
    )


@router.post(
    "/live/run", response_model=RunLiveOut, status_code=status.HTTP_202_ACCEPTED,
    summary="Evaluate the given questions now (the \"Evaluate\" button); a limited number of runs per client",
)
async def run_live_evaluation(body: RunLiveIn, request: Request, svc: ServicesDep) -> RunLiveOut:
    infra = request.app.state.infra
    s = infra.settings
    if not s.live_evaluation_enabled:
        raise ValidationFailed("Evaluation is turned off (LIVE_EVALUATION_ENABLED=false).")
    rows = await svc.eval_repo.list_live_by_message_ids([str(m) for m in body.message_ids])
    waiting = [r.message_id for r in rows if r.status == JobStatus.AWAITING]
    if not waiting:  # nothing to score: do not use up a run
        raise ValidationFailed("There are no new questions to evaluate. Ask the assistant a search question first.")
    hit = infra.eval_limiter.hit(client_ip(request, s.trust_proxy_headers))
    if not hit.allowed:
        hours = max(1, round(hit.retry_after_seconds / 3600))
        raise RateLimited(
            f"Evaluation limit reached: you can run up to {s.eval_rate_limit_max} evaluations every "
            f"{max(1, s.eval_rate_limit_window_seconds // 3600)} hours. Try again in about {hours} hour(s).",
            details={
                "limit": s.eval_rate_limit_max, "window_seconds": s.eval_rate_limit_window_seconds,
                "retry_after_seconds": hit.retry_after_seconds,
            },
        )
    queued = await svc.eval_repo.release_awaiting(waiting)
    return RunLiveOut(queued=queued, usage=_usage(request))


@router.get(
    "/live/lookup", response_model=list[LiveEvaluationOut],
    summary="Scores of specific chat messages (the browser keeps the conversation, the server keeps the scores)",
)
async def live_lookup(
    svc: ServicesDep, message_id: Annotated[list[uuid.UUID], Query(max_length=100)] = [],  # noqa: B006
) -> list[LiveEvaluationOut]:
    items = await svc.eval_repo.list_live_by_message_ids([str(m) for m in message_id])
    return [LiveEvaluationOut.of(i) for i in items]


@router.delete("/live", status_code=status.HTTP_204_NO_CONTENT, summary="Delete all live evaluation records")
async def clear_live_evaluations(svc: ServicesDep) -> Response:
    await svc.eval_repo.clear_live()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/dataset", response_model=DatasetOut, summary="The evaluation dataset used by new runs")
async def dataset(svc: ServicesDep) -> DatasetOut:
    d = await svc.evaluation.dataset()
    return DatasetOut(id=d.id, name=d.name, version=d.version, menu_dataset_version=d.menu_dataset_version,
                      content_hash=d.content_hash, description=d.description, question_count=d.question_count)


@router.get("/compare", response_model=CompareOut)
async def compare(svc: ServicesDep, run_ids: Annotated[list[str], Query(min_length=2)]) -> CompareOut:
    out = await svc.evaluation.compare(run_ids)
    return CompareOut(
        comparable=out["comparable"], warnings=out["warnings"], dataset_id=out["dataset_id"],
        runs=[EvaluationRunOut.of(r) for r in out["runs"]], best_by_metric=out["best_by_metric"],
    )


@router.get("/{evaluation_id}", response_model=EvaluationRunOut)
async def get_evaluation(evaluation_id: str, svc: ServicesDep) -> EvaluationRunOut:
    return EvaluationRunOut.of(await svc.evaluation.get_run(evaluation_id))


@router.get("/{evaluation_id}/results", response_model=EvaluationResultsOut)
async def get_results(evaluation_id: str, svc: ServicesDep) -> EvaluationResultsOut:
    run = await svc.evaluation.get_run(evaluation_id)
    results = await svc.evaluation.results(evaluation_id)
    groups = defaultdict(list)
    for r in results:
        if not r.error:
            groups[r.category].append(r)
    by_category: dict[str, dict[str, float | None]] = {}
    for cat, rs in groups.items():
        row: dict[str, float | None] = {"count": float(len(rs))}
        for k in RETRIEVAL_METRIC_KEYS:
            row[k] = mean_ignoring_none(r.retrieval_metrics.get(k) for r in rs)
        for k in RAGAS_METRIC_KEYS:
            row[k] = mean_ignoring_none(r.ragas_scores.get(k) for r in rs)
        by_category[cat] = row
    return EvaluationResultsOut(
        run=EvaluationRunOut.of(run), results=[QueryResultOut.of(r) for r in results], by_category=by_category
    )
