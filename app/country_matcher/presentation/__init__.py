"""HTTP API of Country Matcher, under /api/v1/country-matcher.

Public, read-and-predict only. Dataset ingestion, indexing and evaluation are admin scripts (``python -m
app.country_matcher.cli``) and are deliberately not reachable over HTTP.
"""

from __future__ import annotations

import logging
import uuid
from itertools import zip_longest
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app.country_matcher.container import CountryMatcherInfra
from app.country_matcher.domain.models import Aggregation, ImageRecord, ModelName, Prediction, Split
from app.country_matcher.infrastructure.imaging import decode_image
from app.country_matcher.infrastructure.storage import LocalImageStorage
from app.country_matcher.presentation.schemas import (
    CompareResponse,
    CountryScoreOut,
    DatasetOut,
    EvaluationResponse,
    ExampleOut,
    ExamplesResponse,
    ImageOut,
    IndexedCountryOut,
    IndexedImageOut,
    IndexedImagesResponse,
    ModelIndexOut,
    ModelInfoOut,
    ModelsResponse,
    NeighborOut,
    PredictionOut,
    PredictResponse,
    QueryOut,
    TimingsOut,
)
from app.errors import NotFoundError, RateLimited, ValidationFailed
from app.restaurant.presentation.deps import client_ip

log = logging.getLogger("country_matcher.api")
router = APIRouter(prefix="/country-matcher", tags=["country-matcher"])

MODEL_CARDS = {
    ModelName.CLIP: (
        "CLIP",
        "OpenAI's CLIP image encoder (ViT-B/32), trained on 400M image-text pairs. Its 512-d embeddings capture what a "
        "picture is about, not only how it looks.",
    ),
    ModelName.RESNET50: (
        "ResNet50",
        "A convolutional network trained to classify 1000 ImageNet categories. Its 2048-d embeddings come from the layer "
        "before the classifier and capture visual structure and texture.",
    ),
}


def infra_of(request: Request) -> CountryMatcherInfra:
    return request.app.state.country_matcher


def _absolute(request: Request, url: str) -> str:
    """Local-storage URLs are relative to the API root; the browser needs them absolute."""
    return str(request.base_url).rstrip("/") + url if url.startswith("/") else url


def _image_out(request: Request, infra: CountryMatcherInfra, image: ImageRecord) -> ImageOut:
    return ImageOut.of(image, _absolute(request, infra.storage.url(image.storage_path)))


def _prediction_out(request: Request, infra: CountryMatcherInfra, p: Prediction) -> PredictionOut:
    assert p.country and p.timings and p.embedding

    def score(c) -> CountryScoreOut:
        return CountryScoreOut(
            country_code=c.country_code, country_name=c.country_name, votes=c.votes, weight=round(c.weight, 4), support=round(c.support, 4)
        )

    return PredictionOut(
        model=p.model, model_name=p.embedding.model_name, model_version=p.embedding.model_version,
        dimension=p.embedding.dimension, aggregation=p.aggregation, k=p.k, country=score(p.country),
        ranking=[score(c) for c in p.ranking],
        neighbors=[
            NeighborOut(rank=i, similarity=round(n.similarity, 4), image=_image_out(request, infra, n.image))
            for i, n in enumerate(p.neighbors, start=1)
        ],
        explanation=p.explanation,
        timings=TimingsOut(**{k: round(v, 1) for k, v in vars(p.timings).items()}),
    )


async def _query_image(
    infra: CountryMatcherInfra, file: UploadFile | None, example_id: uuid.UUID | None
) -> tuple[Image.Image, QueryOut]:
    """The image to analyse: an upload (validated) or one of the held-out example images."""
    if (file is None) == (example_id is None):
        raise ValidationFailed("Send either an image file or an example_id.")
    if example_id is not None:
        async with infra.open_store() as store:
            record = await store.get_image(example_id)
        if record is None or record.split is not Split.TEST:
            raise NotFoundError("Unknown example image.")
        image = decode_image(await infra.storage.get(record.storage_path), min_side=1)
        return image, QueryOut(example_id=record.id, actual_country_code=record.country_code, actual_country_name=record.country_name)
    assert file is not None
    limit = infra.settings.country_matcher_max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)  # never read more than the limit
    if len(data) > limit:
        raise ValidationFailed(f"The image is larger than {infra.settings.country_matcher_max_upload_mb} MB.")
    return decode_image(data), QueryOut()


def _spend(request: Request, infra: CountryMatcherInfra, cost: int) -> int:
    """Count ``cost`` predictions against this client's budget; ``RateLimited`` when it is used up."""
    key = client_ip(request, infra.settings.trust_proxy_headers)
    spent = 0
    for _ in range(cost):
        result = infra.predict_limiter.hit(key)
        if not result.allowed:
            for _ in range(spent):
                infra.predict_limiter.refund(key)
            raise RateLimited(
                "Prediction limit reached for now. This demo runs the models on a small server; please try again later.",
                details={"limit": infra.predict_limiter.max_events, "retry_after_seconds": result.retry_after_seconds},
            )
        spent += 1
    return spent


def _refund(request: Request, infra: CountryMatcherInfra, spent: int) -> None:
    key = client_ip(request, infra.settings.trust_proxy_headers)
    for _ in range(spent):
        infra.predict_limiter.refund(key)


def _check_k(infra: CountryMatcherInfra, k: int | None) -> int:
    k = infra.settings.country_matcher_default_k if k is None else k
    if not 1 <= k <= infra.settings.country_matcher_max_k:
        raise ValidationFailed(f"k must be between 1 and {infra.settings.country_matcher_max_k}.")
    return k


@router.get("/models", response_model=ModelsResponse, summary="The models, their index and the dataset size")
async def models(request: Request) -> ModelsResponse:
    infra = infra_of(request)
    async with infra.open_store() as store:
        stats = {m: await store.embedding_stats(m) for m in ModelName}
        dataset = DatasetOut(
            index_images=await store.count_images(split=Split.INDEX),
            test_images=await store.count_images(split=Split.TEST),
            countries=await store.count_countries(),
        )
    described = infra.embedders.describe()
    cards = [
        ModelInfoOut(
            id=m, label=MODEL_CARDS[m][0], model_name=described[m]["model_name"], dimension=described[m]["dimension"],
            description=MODEL_CARDS[m][1], loaded=described[m]["loaded"],
            index=ModelIndexOut(images=stats[m]["count"], versions=stats[m]["versions"]),
        )
        for m in ModelName
    ]
    return ModelsResponse(
        models=cards, dataset=dataset, default_k=infra.settings.country_matcher_default_k,
        max_k=infra.settings.country_matcher_max_k, max_upload_mb=infra.settings.country_matcher_max_upload_mb,
        aggregations=list(Aggregation), ready=all(c.index.images > 0 for c in cards) and dataset.index_images > 0,
    )


@router.get("/examples", response_model=ExamplesResponse, summary="Held-out images to try (their label is hidden)")
async def examples(request: Request, limit: Annotated[int, Query(ge=1, le=24)] = 10) -> ExamplesResponse:
    infra = infra_of(request)
    async with infra.open_store() as store:
        test = await store.list_images(split=Split.TEST)
    by_country: dict[str, list[ImageRecord]] = {}
    for r in test:
        by_country.setdefault(r.country_code, []).append(r)
    # One image per country first, then a second round, and so on: a varied set in a stable order.
    picked = [r for round_ in zip_longest(*by_country.values()) for r in round_ if r is not None][:limit]
    return ExamplesResponse(
        examples=[
            ExampleOut(
                id=r.id, url=_absolute(request, infra.storage.url(r.storage_path)), width=r.width, height=r.height,
                attribution=r.source_attribution, license=r.source_license, source_page_url=r.source_page_url,
            )
            for r in picked
        ]
    )


@router.get("/indexed-images", response_model=IndexedImagesResponse, summary="The searchable (indexed) images and the models that embedded them")
async def indexed_images(
    request: Request,
    country: Annotated[str | None, Query(pattern="^[A-Za-z]{2}$")] = None,
    limit: Annotated[int, Query(ge=1, le=60)] = 24,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> IndexedImagesResponse:
    """Read-only. Held-out test images are not listed: they are never part of the index."""
    infra = infra_of(request)
    async with infra.open_store() as store:
        page, total = await store.indexed_images(country_code=country.upper() if country else None, limit=limit, offset=offset)
        countries = await store.indexed_countries()
    return IndexedImagesResponse(
        total=total, offset=offset, limit=limit,
        countries=[IndexedCountryOut(code=c, name=n, images=k) for c, n, k in countries],
        images=[
            IndexedImageOut(**_image_out(request, infra, item.image).model_dump(), models=list(item.models)) for item in page
        ],
    )


@router.post("/predict", response_model=PredictResponse, summary="Estimate the country of one image with one model")
async def predict(
    request: Request,
    model: Annotated[ModelName, Form()] = ModelName.CLIP,
    k: Annotated[int | None, Form()] = None,
    aggregation: Annotated[Aggregation, Form()] = Aggregation.WEIGHTED,
    file: Annotated[UploadFile | None, File()] = None,
    example_id: Annotated[uuid.UUID | None, Form()] = None,
) -> PredictResponse:
    infra = infra_of(request)
    k = _check_k(infra, k)
    spent = _spend(request, infra, 1)
    try:
        image, query = await _query_image(infra, file, example_id)
        prediction = await infra.predictor.predict(image, model, k=k, aggregation=aggregation)
    except Exception:
        _refund(request, infra, spent)  # nothing was predicted: do not use up the visitor's budget
        raise
    return PredictResponse(query=query, prediction=_prediction_out(request, infra, prediction))


@router.post("/compare", response_model=CompareResponse, summary="Estimate the country with CLIP and ResNet50 side by side")
async def compare(
    request: Request,
    k: Annotated[int | None, Form()] = None,
    aggregation: Annotated[Aggregation, Form()] = Aggregation.WEIGHTED,
    file: Annotated[UploadFile | None, File()] = None,
    example_id: Annotated[uuid.UUID | None, Form()] = None,
) -> CompareResponse:
    infra = infra_of(request)
    k = _check_k(infra, k)
    spent = _spend(request, infra, len(ModelName))  # two networks run: it costs two predictions
    try:
        image, query = await _query_image(infra, file, example_id)
        results = await infra.predictor.compare(image, k=k, aggregation=aggregation)
    except Exception:
        _refund(request, infra, spent)
        raise
    return CompareResponse(query=query, predictions={m: _prediction_out(request, infra, p) for m, p in results.items()})


@router.get("/evaluation", response_model=EvaluationResponse, summary="The stored evaluation results (never recomputed here)")
async def evaluation(request: Request) -> EvaluationResponse:
    infra = infra_of(request)
    async with infra.open_store() as store:
        run = await store.latest_evaluation()
    if run is None:
        return EvaluationResponse(available=False)
    return EvaluationResponse(available=True, generated_at=run.created_at.isoformat(), results=run.results)


@router.get("/local-images/{path:path}", include_in_schema=False)
async def local_image(request: Request, path: str) -> FileResponse:
    """Serves dataset images in local-storage development mode only. With S3 this route does not exist."""
    storage = infra_of(request).storage
    if not isinstance(storage, LocalImageStorage):
        raise NotFoundError("Not found.")
    target: Path = storage.resolve(path)
    if not target.is_file():
        raise NotFoundError("Not found.")
    return FileResponse(target, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
