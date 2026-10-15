"""Ranking quality of one search, measured with ranx: Precision@K, Recall@K, NDCG@K and MRR.

A real resume has no hand-labelled relevant jobs, so the judgements are derived from the text itself: a job is *relevant*
when at least ``RELEVANT_COVERAGE`` of its required skills are named in the resume, and *highly relevant* from
``STRONG_COVERAGE``. That is a transparent proxy, not ground truth (and it is lexical, so it can favour BM25). Precision,
recall and MRR treat both grades as relevant; NDCG uses the grades (2 and 3). Nothing is searched again: the ranking being
judged is the one the user already saw, whichever method produced it. No model is involved.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import threading
import warnings
from collections.abc import Sequence

from pydantic import BaseModel

from app.errors import ValidationFailed
from app.myresume.domain.models import Job

# Both must be set before ranx is imported. numba (behind ranx) caches compiled code next to the library or in the home
# directory, and neither is writable everywhere (Cloud Run); matplotlib (which ranx imports) writes its font cache to
# MPLCONFIGDIR and prefers a headless backend. Importing ranx here, at start-up, also builds that font cache and loads
# scipy / pandas / numba once, instead of during the first evaluation request.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba-cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
from ranx import Qrels, Run, evaluate  # noqa: E402

RELEVANT_COVERAGE = 0.3
STRONG_COVERAGE = 0.5
QUERY = "resume"


class RelevantJob(BaseModel):
    job_id: str
    coverage: float  # share of the job's required skills named in the resume
    grade: int  # 2 relevant, 3 highly relevant
    rank: int | None  # its position in the evaluated ranking (None when the search did not return it)


class RankingEvaluation(BaseModel):
    k: int  # the ranking length: every metric is "@K" for this K
    precision: float | None
    recall: float | None
    ndcg: float | None
    mrr: float | None
    relevant: list[RelevantJob]  # the judgements, best coverage first
    relevant_coverage: float
    strong_coverage: float
    notices: list[str]


def skill_coverage(job: Job, resume_text: str) -> float:
    """Share of the job's required skills that appear (as whole words or phrases) in the resume."""
    text = resume_text.lower()
    found = sum(bool(re.search(rf"(?<![a-z0-9]){re.escape(s.lower())}(?![a-z0-9])", text)) for s in job.skills)
    return found / len(job.skills) if job.skills else 0.0


def _grade(coverage: float) -> int:
    return 3 if coverage >= STRONG_COVERAGE else 2 if coverage >= RELEVANT_COVERAGE else 0


# ranx's metric kernels are numba ``parallel=True`` functions, and ranx forces numba's "workqueue" threading layer when it
# is imported (overriding NUMBA_THREADING_LAYER). That layer aborts the whole process ("Concurrent access has been
# detected", SIGABRT) when two threads launch a parallel kernel at once, which is what concurrent requests do through
# ``asyncio.to_thread``. One evaluation is a few milliseconds, so they simply take turns.
_RANX_LOCK = threading.Lock()


def _ranx(qrels: dict[str, int], ranked: Sequence[str], k: int) -> dict[str, float]:
    # numba compiles its kernels once per document-id string *width* (numpy ``<U{n}`` arrays), about ten seconds each, so
    # job ids of different lengths ("accountant", "backend-python") would each trigger a compilation inside a request.
    # Fixed-width ids ("d0000") keep a single signature, the one ``warm_up`` already compiled.
    ids = {job_id: f"d{n:04d}" for n, job_id in enumerate(dict.fromkeys([*ranked, *qrels]))}
    run = Run.from_dict({QUERY: {ids[job_id]: float(len(ranked) - i) for i, job_id in enumerate(ranked)}})
    binary = Qrels.from_dict({QUERY: {ids[j]: 1 for j in qrels}})
    graded = Qrels.from_dict({QUERY: {ids[j]: grade for j, grade in qrels.items()}})
    with _RANX_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = evaluate(binary, run, [f"precision@{k}", f"recall@{k}", f"mrr@{k}"])
        graded_scores = evaluate(graded, run, f"ndcg@{k}")
    return {
        "precision": float(flat[f"precision@{k}"]), "recall": float(flat[f"recall@{k}"]),
        "mrr": float(flat[f"mrr@{k}"]), "ndcg": float(graded_scores),
    }


def warm_up() -> None:
    """Compile ranx's numba kernels (about ten seconds, once per process) with a dry run on mock data.

    Start-up awaits it (see ``lifespan``): the first evaluation a user asks for then takes milliseconds.
    """
    _ranx({"a": 2, "b": 3}, ["a", "c", "b"], 3)


class RankingEvaluationService:
    def __init__(self, jobs: Sequence[Job]):
        self._all, self._jobs = list(jobs), {j.id: j for j in jobs}

    def judgements(self, resume_text: str) -> list[tuple[str, float, int]]:
        """(job id, coverage, grade) of every relevant job for this resume, best coverage first."""
        scored = sorted(((skill_coverage(j, resume_text), j.id) for j in self._all), key=lambda t: -t[0])
        return [(i, c, _grade(c)) for c, i in scored if _grade(c) > 0]

    def jobs_by_id(self, job_ids: Sequence[str]) -> list[Job]:
        """The distinct jobs of a ranking, in order; raises ``ValidationFailed`` for an empty or unknown one."""
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            raise ValidationFailed("There is no ranking to evaluate: search for matching jobs first.")
        unknown = [j for j in ids if j not in self._jobs]
        if unknown:
            raise ValidationFailed(f"Unknown job '{unknown[0]}'.")
        return [self._jobs[j] for j in ids]

    async def run(self, resume_text: str, job_ids: Sequence[str]) -> RankingEvaluation:
        """Score the ranking ``job_ids`` (best first) against the jobs this resume makes relevant."""
        ranked = [j.id for j in self.jobs_by_id(job_ids)]
        rank_of = {job_id: i for i, job_id in enumerate(ranked, 1)}
        judged = self.judgements(resume_text)
        relevant = [RelevantJob(job_id=i, coverage=round(c, 3), grade=g, rank=rank_of.get(i)) for i, c, g in judged]
        scores: dict[str, float | None] = {"precision": None, "recall": None, "ndcg": None, "mrr": None}
        notices: list[str] = []
        if judged:
            scores = await asyncio.to_thread(_ranx, {i: g for i, _, g in judged}, ranked, len(ranked))
        else:
            notices.append(
                f"No job has {RELEVANT_COVERAGE:.0%} of its required skills named in the resume, so the metrics are not defined."
            )
        return RankingEvaluation(
            k=len(ranked), relevant=relevant, relevant_coverage=RELEVANT_COVERAGE, strong_coverage=STRONG_COVERAGE,
            notices=notices, **scores,
        )
