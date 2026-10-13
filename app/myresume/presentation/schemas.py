from __future__ import annotations

from pydantic import BaseModel, Field

from app.myresume.application.evaluation import RankingEvaluation
from app.myresume.domain.models import Job
from app.myresume.domain.retrieval import MatchResult, Method, RetrievalConfig, ScoredJob
from app.restaurant.domain.rag.value_objects import ModelRef
from app.restaurant.presentation.schemas import ModelOptionOut, ProviderOut


class JobOut(BaseModel):
    id: str
    title: str
    company: str
    location: str
    work_mode: str
    level: str
    summary: str
    skills: list[str]
    experience: str
    responsibilities: list[str]
    qualifications: list[str]
    nice_to_have: list[str]

    @classmethod
    def of(cls, job: Job) -> JobOut:
        return cls(**job.model_dump())


class MethodInfo(BaseModel):
    id: Method
    name: str
    description: str


class DefaultsOut(BaseModel):
    retrieval: RetrievalConfig
    methods: list[MethodInfo]
    spacy_model: str
    lemmatizer_available: bool
    jobs: int


class ModelsOut(BaseModel):
    providers: list[ProviderOut]
    embedding: list[ModelOptionOut]
    default_embedding: ModelRef


class MatchIn(BaseModel):
    resume_text: str = Field(min_length=1, max_length=20_000)
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: ModelRef | None = None


class MatchedJobOut(ScoredJob):
    job: JobOut


class MatchOut(BaseModel):
    method: Method
    results: list[MatchedJobOut]
    latency_ms: float
    embed_ms: float | None
    query_terms: int | None
    notices: list[str]
    embedding: ModelRef | None  # the model that embedded the resume (vector / hybrid)

    @classmethod
    def of(cls, res: MatchResult, jobs: dict[str, Job], embedding: ModelRef) -> MatchOut:
        return cls(
            method=res.config.method,
            results=[MatchedJobOut(**r.model_dump(), job=JobOut.of(jobs[r.job_id])) for r in res.results],
            latency_ms=res.latency_ms, embed_ms=res.embed_ms, query_terms=res.query_terms, notices=res.notices,
            embedding=embedding if res.config.method != Method.BM25 else None,
        )


class LimitUsage(BaseModel):
    """How many uses of a capped action (an upload) this client IP has left."""

    limit: int
    used: int
    remaining: int
    window_seconds: int
    reset_in_seconds: int  # until the oldest counted use expires (0 when none is counted)


class EvaluationIn(BaseModel):
    """The ranking to evaluate: the resume and the job ids a search returned for it, best first (all K are scored)."""

    resume_text: str = Field(min_length=1, max_length=20_000)
    job_ids: list[str] = Field(min_length=1, max_length=20)


class RelevantJobOut(BaseModel):
    job_id: str
    title: str
    company: str
    coverage: float
    grade: int
    rank: int | None


class EvaluationOut(BaseModel):
    k: int
    precision: float | None
    recall: float | None
    ndcg: float | None
    mrr: float | None
    relevant: list[RelevantJobOut]
    relevant_coverage: float
    strong_coverage: float
    notices: list[str]

    @classmethod
    def of(cls, res: RankingEvaluation, jobs: dict[str, Job]) -> EvaluationOut:
        return cls(
            **res.model_dump(exclude={"relevant"}),
            relevant=[
                RelevantJobOut(**r.model_dump(), title=jobs[r.job_id].title, company=jobs[r.job_id].company)
                for r in res.relevant
            ],
        )


class ExtractedResume(BaseModel):
    filename: str
    text: str
    characters: int
    truncated: bool  # the file had more text than the resume field accepts; the rest was cut
    usage: LimitUsage  # uploads this client IP has left
