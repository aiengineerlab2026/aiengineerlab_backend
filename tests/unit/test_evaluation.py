from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from conftest import ScriptedLLM, World  # type: ignore[import-not-found]

from app.application.evaluation.ports import RAGAS_METRIC_KEYS, RagasEvaluator, RagasInput
from app.application.evaluation.runner import EvaluationRunner
from app.application.evaluation.service import EvaluationService
from app.domain.errors import ConflictError, ValidationFailed
from app.domain.evaluation.entities import (
    EvalDataset,
    EvalQuestion,
    EvaluationRun,
    LiveEvaluation,
    QueryResult,
)
from app.domain.evaluation.metrics import retrieval_metrics
from app.domain.evaluation.repositories import EvaluationRepository
from app.domain.rag.entities import JobStatus
from app.domain.rag.ranking import rank_dishes
from app.domain.rag.value_objects import ChunkingConfig, ChunkingStrategy, RetrievalConfig
from app.infrastructure.seed.evaluation_data import (
    EVAL_DATASET_NAME,
    EVAL_DATASET_VERSION,
    build_dataset,
    dataset_content_hash,
)

# ------------------------------------------------------------------------------ dataset


def test_eval_dataset_is_versioned_sized_and_independent_of_retrieval(dishes):
    rows = build_dataset(dishes)
    assert 30 <= len(rows) <= 50
    assert rows == build_dataset(dishes) and dataset_content_hash(rows) == dataset_content_hash(build_dataset(dishes))
    assert len({r["key"] for r in rows}) == len(rows)
    assert EVAL_DATASET_VERSION
    ids = {d.id for d in dishes}
    for r in rows:
        assert set(r["relevant_dish_ids"]) <= ids and r["reference_answer"]
    categories = {r["category"] for r in rows}
    assert {"high_protein", "follow_up"} <= categories
    assert any(not r["relevant_dish_ids"] for r in rows), "dataset must contain no-match questions"
    assert not any(not d.available and d.id in r["relevant_dish_ids"] for r in rows for d in dishes)


# ------------------------------------------------------------------------------ fakes for the runner


class MemEvalRepo(EvaluationRepository):
    def __init__(self, dataset: EvalDataset, questions: list[EvalQuestion]) -> None:
        self.dataset, self.questions = dataset, questions
        self.runs: dict[str, EvaluationRun] = {}
        self.results: dict[str, list[QueryResult]] = {}
        self.live: dict[str, LiveEvaluation] = {}

    async def get_dataset(self, name=None, version=None):
        return self.dataset

    async def get_dataset_by_id(self, dataset_id):
        return self.dataset

    async def list_questions(self, dataset_id):
        return list(self.questions)

    async def add_run(self, run):
        self.runs[run.id] = run

    async def get_run(self, run_id):
        return self.runs.get(run_id)

    async def list_runs(self, limit=100):
        return list(self.runs.values())

    async def save_run(self, run):
        self.runs[run.id] = run

    async def update_progress(self, run_id, progress):
        self.runs[run_id].progress = progress

    async def add_query_results(self, results):
        for r in results:
            self.results.setdefault(r.run_id, []).append(r)

    async def list_query_results(self, run_id):
        return self.results.get(run_id, [])

    async def claim_pending_run(self):
        return None

    async def requeue_interrupted(self):
        return 0

    async def add_live(self, item):
        self.live[item.id] = item

    async def save_live(self, item):
        self.live[item.id] = item

    async def list_live(self, limit=50):
        return sorted(self.live.values(), key=lambda i: i.created_at, reverse=True)[:limit]

    async def list_live_by_message_ids(self, message_ids):
        return [i for i in self.live.values() if i.message_id in set(message_ids)]

    async def release_awaiting(self, message_ids):
        n = 0
        for i in self.live.values():
            if i.message_id in set(message_ids) and i.status == JobStatus.AWAITING:
                i.status, i.stage = JobStatus.PENDING, "queued"
                n += 1
        return n

    async def purge_awaiting(self, older_than):
        old = [k for k, i in self.live.items() if i.status == JobStatus.AWAITING and i.created_at < older_than]
        for k in old:
            del self.live[k]
        return len(old)

    async def redact_live(self, item_id):
        i = self.live[item_id]
        i.question = i.query = i.answer = ""
        i.contexts = []

    async def claim_pending_live(self):
        item = next((i for i in self.live.values() if i.status == JobStatus.PENDING), None)
        if item:
            item.status = JobStatus.RUNNING
        return item

    async def requeue_interrupted_live(self):
        return 0

    async def clear_live(self):
        n, self.live = len(self.live), {}
        return n


class FakeRagas(RagasEvaluator):
    """Deterministic stand-in for RAGAS: constant, recognisable scores; records its inputs."""

    version = "0.0-test"  # type: ignore[assignment]

    def __init__(self, *, fail: bool = False) -> None:
        self.fail, self.seen = fail, []

    async def evaluate(self, samples: list[RagasInput]):
        self.seen = samples
        if self.fail:
            raise RuntimeError("ragas unavailable")
        return [{"non_llm_context_precision": 0.4, "non_llm_context_recall": 0.6} for _ in samples]


def _questions(world: World, n: int | None = None) -> tuple[EvalDataset, list[EvalQuestion]]:
    rows = build_dataset(world.dishes)[:n]
    qs = [
        EvalQuestion(
            id=str(uuid.uuid4()), key=r["key"], category=r["category"], question=r["question"],
            relevant_dish_ids=tuple(r["relevant_dish_ids"]), reference_answer=r["reference_answer"],
            required_constraints=r["required_constraints"], expected_behavior=r["expected_behavior"],
            history=tuple(r["history"]), position=r["position"],
        )
        for r in rows
    ]
    ds = EvalDataset("ds1", EVAL_DATASET_NAME, EVAL_DATASET_VERSION, "menu-v1", "hash", "d", len(qs))
    return ds, qs


async def _setup(llm=None, *, ragas=None, n=8, retrieval=None):
    w = World(llm or ScriptedLLM(text="a grounded answer"))
    chunking = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    exp, _ = await w.config.update(chunking, retrieval or RetrievalConfig(top_k=5))
    await w.build_index(chunking)
    ds, qs = _questions(w, n)
    repo = MemEvalRepo(ds, qs)
    ragas = ragas or FakeRagas()

    @asynccontextmanager
    async def uow():
        yield SimpleNamespace(
            eval_repo=repo, retrieval=w.retrieval, indexing=w.indexing, index_repo=w.indexes,
        )

    service = EvaluationService(repo, w.config, w.llm, ragas, EVAL_DATASET_NAME)
    return w, exp, repo, service, EvaluationRunner(uow, w.llm, ragas), ragas


# ------------------------------------------------------------------------------ runner


async def test_run_computes_metrics_from_ground_truth_and_real_retrieval():
    w, exp, repo, service, runner, ragas = await _setup(n=8)
    run = await service.create_run()
    assert run.status == JobStatus.PENDING and run.settings["question_count"] == 8
    await runner.execute(run)
    run = await service.get_run(run.id)
    assert run.status == JobStatus.COMPLETED and run.error is None
    results = await service.results(run.id)
    assert len(results) == 8 and not any(r.error for r in results)

    # per-question retrieval metrics are exactly the deterministic functions of (retrieved, ground truth)
    for r in results:
        assert r.retrieval_metrics == retrieval_metrics(r.retrieved_dish_ids, r.relevant_dish_ids)
        assert r.retrieved_dish_ids == rank_dishes([c["dish_ids"] for c in r.trace["chunks"]])
    # aggregate = mean over defined values only
    defined = [r.retrieval_metrics["recall_at_5"] for r in results if r.retrieval_metrics["recall_at_5"] is not None]
    assert run.metrics["recall_at_5"] == pytest.approx(sum(defined) / len(defined))
    assert run.metrics["questions_total"] == 8 and run.metrics["questions_failed"] == 0
    assert run.metrics["non_llm_context_precision"] == pytest.approx(0.4)
    assert run.metrics["non_llm_context_recall"] == pytest.approx(0.6)
    assert run.metrics["duration_seconds"] is not None


async def test_ragas_receives_the_exact_contexts_and_the_index_chunks_covering_the_ground_truth():
    w, exp, repo, service, runner, ragas = await _setup(n=3, retrieval=RetrievalConfig(top_k=3))
    run = await service.create_run()
    await runner.execute(run)
    results = await service.results(run.id)
    assert len(ragas.seen) == 3
    for sample, r, q in zip(ragas.seen, results, repo.questions, strict=True):
        covering = [c.content for cs in w.store.chunks.values() for c, _ in cs if set(c.dish_ids) & set(q.relevant_dish_ids)]
        assert sample.reference_contexts == list(dict.fromkeys(covering))  # index chunks that cover the ground truth
        assert sample.retrieved_contexts == r.contexts and len(r.contexts) == 3  # only top_k chunks reach the LLM
        assert r.trace["chunks"][-1]["rank"] == 10 or len(r.trace["chunks"]) >= 3  # deeper list kept for rank metrics


async def test_follow_up_questions_are_retrieved_with_their_history():
    w, exp, repo, service, runner, _ = await _setup(n=40)
    follow = next(q for q in repo.questions if q.history)
    run = await service.create_run()
    await runner.execute(run)
    res = next(r for r in await service.results(run.id) if r.question_key == follow.key)
    assert res.query == follow.retrieval_query() and res.query != follow.question


async def test_question_limit_is_honoured():
    w, exp, repo, service, runner, _ = await _setup(n=20)
    run = await service.create_run(question_limit=4)
    await runner.execute(run)
    assert len(await service.results(run.id)) == 4
    with pytest.raises(ValidationFailed):
        await service.create_run(question_limit=0)


async def test_ragas_failure_fails_the_run_but_keeps_retrieval_results():
    w, exp, repo, service, runner, _ = await _setup(ragas=FakeRagas(fail=True), n=4)
    run = await service.create_run()
    await runner.execute(run)
    run = await service.get_run(run.id)
    assert run.status == JobStatus.FAILED and "ragas unavailable" in (run.error or "")
    assert run.metrics["recall_at_5"] is not None and run.metrics["non_llm_context_precision"] is None
    assert len(await service.results(run.id)) == 4


async def test_one_failing_question_does_not_sink_the_run():
    class FlakyLLM(ScriptedLLM):
        async def generate(self, system, user):
            if "protein" in user.lower() and not getattr(self, "failed", False):
                self.failed = True
                raise RuntimeError("provider timeout")
            return await super().generate(system, user)

    w, exp, repo, service, runner, _ = await _setup(FlakyLLM(text="ok"), n=6)
    run = await service.create_run()
    await runner.execute(run)
    run = await service.get_run(run.id)
    results = await service.results(run.id)
    failed = [r for r in results if r.error]
    assert run.status == JobStatus.COMPLETED and len(failed) == 1 and "provider timeout" in failed[0].error
    assert run.metrics["questions_failed"] == 1.0
    assert run.metrics["questions_total"] == 6.0


async def test_llm_is_required_to_start_a_run():
    w, exp, repo, service, runner, _ = await _setup(ScriptedLLM(configured=False))
    with pytest.raises(ConflictError) as e:
        await service.create_run()
    assert e.value.details["code"] == "llm_not_configured"


# ------------------------------------------------------------------------------ comparison


async def test_compare_flags_differences_and_picks_best_per_metric():
    w, exp, repo, service, runner, _ = await _setup(n=5)
    a = await service.create_run()
    await runner.execute(a)
    b = await service.create_run()
    await runner.execute(b)
    repo.runs[a.id].metrics.update({"precision_at_5": 0.2, "avg_retrieval_latency_ms": 50.0})
    repo.runs[b.id].metrics.update({"precision_at_5": 0.4, "avg_retrieval_latency_ms": 80.0})

    cmp = await service.compare([a.id, b.id])
    assert cmp["comparable"] and cmp["warnings"] == []
    assert cmp["best_by_metric"]["precision_at_5"] == b.id
    assert cmp["best_by_metric"]["avg_retrieval_latency_ms"] == a.id  # lower is better

    repo.runs[b.id].settings["ragas_metrics"] = "other-metric-set"
    cmp = await service.compare([a.id, b.id])
    assert not cmp["comparable"] and any("RAGAS metric set" in x for x in cmp["warnings"])


async def test_compare_validates_its_input():
    w, exp, repo, service, runner, _ = await _setup(n=3)
    a = await service.create_run()
    with pytest.raises(ValidationFailed):
        await service.compare([a.id])
    b = await service.create_run()
    with pytest.raises(ConflictError):  # still pending
        await service.compare([a.id, b.id])


def test_ragas_metric_keys_are_the_llm_free_metrics():
    assert set(RAGAS_METRIC_KEYS) == {"non_llm_context_precision", "non_llm_context_recall"}
