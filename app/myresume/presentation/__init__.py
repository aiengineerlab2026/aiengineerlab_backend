"""Everything under ``/api/v1/myresume``: jobs, resume upload, retrieval (BM25 / vector / hybrid) and its ranx evaluation."""

import asyncio

from fastapi import APIRouter, Request, UploadFile

from app.myresume.domain.retrieval import METHOD_DESCRIPTIONS, METHOD_NAMES, Method, RetrievalConfig
from app.myresume.infrastructure.bm25 import SPACY_MODEL, lemmatizer_available
from app.myresume.infrastructure.data.jobs import JOBS, JOBS_BY_ID
from app.myresume.infrastructure.resume_files import MAX_BYTES, MAX_CHARS, extract_resume_text
from app.myresume.presentation.deps import InfraDep, count_use, refund_use, usage
from app.myresume.presentation.schemas import (
    DefaultsOut,
    EvaluationIn,
    EvaluationOut,
    ExtractedResume,
    JobOut,
    LimitUsage,
    MatchIn,
    MatchOut,
    MethodInfo,
    ModelsOut,
)
from app.restaurant.infrastructure.llm import catalog
from app.restaurant.presentation.schemas import ModelOptionOut, ProviderOut

router = APIRouter(prefix="/myresume", tags=["myresume"])


@router.get("/jobs", response_model=list[JobOut], summary="The job postings that are searched")
async def list_jobs() -> list[JobOut]:
    return [JobOut.of(j) for j in JOBS]


@router.get("/config/defaults", response_model=DefaultsOut, summary="Default retrieval settings and what the methods are")
async def defaults() -> DefaultsOut:
    return DefaultsOut(
        retrieval=RetrievalConfig(),
        methods=[MethodInfo(id=m, name=METHOD_NAMES[m], description=METHOD_DESCRIPTIONS[m]) for m in Method],
        spacy_model=SPACY_MODEL,
        lemmatizer_available=lemmatizer_available(),
        jobs=len(JOBS),
    )


def _options(items: list[catalog.ModelOption], infra) -> list[ModelOptionOut]:
    return [
        ModelOptionOut(
            provider=o.provider, model=o.model, label=o.label, note=o.note, dimensions=o.dimensions,
            quota_exhausted_for_seconds=infra.quota.remaining(o.provider, o.model),
        )
        for o in items
    ]


@router.get("/models", response_model=ModelsOut, summary="Embedding models that can be chosen (Gemini and Cloudflare Workers AI)")
async def models(infra: InfraDep) -> ModelsOut:
    s = infra.settings
    return ModelsOut(
        providers=[ProviderOut(**p) for p in catalog.providers(s)],
        embedding=_options(catalog.embedding_models(s), infra),
        default_embedding=infra.default_embedding,
    )


@router.get("/resume/usage", response_model=LimitUsage, summary="How many resumes this IP may still upload")
async def upload_quota(request: Request) -> LimitUsage:
    return usage(request, "upload")


@router.post(
    "/resume/extract", response_model=ExtractedResume,
    summary="Extract the text of an uploaded PDF or DOCX resume (nothing is stored); a few uploads per client IP",
)
async def extract_resume(file: UploadFile, request: Request) -> ExtractedResume:
    count_use(request, "upload", "Resume upload")
    try:
        data = await file.read(MAX_BYTES + 1)  # one byte more than allowed is enough to detect an oversized file
        text = await asyncio.to_thread(extract_resume_text, file.filename or "", data)
    except Exception:
        refund_use(request, "upload")  # an unreadable file is not a used upload
        raise
    return ExtractedResume(
        filename=file.filename or "resume", text=text[:MAX_CHARS], characters=min(len(text), MAX_CHARS),
        truncated=len(text) > MAX_CHARS, usage=usage(request, "upload"),
    )


@router.get("/match/vector/usage", response_model=LimitUsage, summary="How many vector searches this IP may still run")
async def vector_quota(request: Request) -> LimitUsage:
    return usage(request, "vector")


@router.post("/match", response_model=MatchOut, summary="Find the jobs that best match a resume with the chosen method")
async def match(body: MatchIn, request: Request, infra: InfraDep) -> MatchOut:
    spec = infra.resolve_embedding(body.embedding)
    if body.config.method == Method.VECTOR:  # the search that calls the embedding model is capped; BM25 and hybrid are not
        count_use(request, "vector", "Vector search")
    try:
        res = await infra.search.match(body.resume_text, body.config, spec)
    except Exception:
        if body.config.method == Method.VECTOR:
            refund_use(request, "vector")  # a search that failed is not a used one
        raise
    return MatchOut.of(res, JOBS_BY_ID, infra.default_embedding if body.embedding is None else body.embedding)


@router.post(
    "/evaluation", response_model=EvaluationOut,
    summary="Precision@K, Recall@K, NDCG@K and MRR (ranx) of the K jobs the last search returned; no new search, no model",
)
async def evaluate_ranking(body: EvaluationIn, infra: InfraDep) -> EvaluationOut:
    return EvaluationOut.of(await infra.evaluation.run(body.resume_text, body.job_ids), JOBS_BY_ID)
