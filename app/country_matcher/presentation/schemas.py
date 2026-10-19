from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel

from app.country_matcher.domain.models import Aggregation, ImageRecord, ModelName

SUPPORT_NOTE = (
    "The support score is the share of the retrieved images' evidence (votes, or similarity weight) that backs this "
    "country. It describes agreement among neighbours; it is not a calibrated probability."
)


class ImageOut(BaseModel):
    id: uuid.UUID
    url: str  # public or signed, short-lived: never carries a secret
    country_code: str
    country_name: str
    city: str | None
    width: int
    height: int
    license: str
    attribution: str
    source_page_url: str | None
    original_url: str

    @classmethod
    def of(cls, image: ImageRecord, url: str) -> ImageOut:
        return cls(
            id=image.id, url=url, country_code=image.country_code, country_name=image.country_name, city=image.city,
            width=image.width, height=image.height, license=image.source_license, attribution=image.source_attribution,
            source_page_url=image.source_page_url, original_url=image.original_url,
        )


class NeighborOut(BaseModel):
    rank: int
    similarity: float
    image: ImageOut


class CountryScoreOut(BaseModel):
    country_code: str
    country_name: str
    votes: int
    weight: float
    support: float  # 0..1, see support_note


class TimingsOut(BaseModel):
    embed_ms: float
    search_ms: float
    total_ms: float


class PredictionOut(BaseModel):
    model: ModelName
    model_name: str
    model_version: str
    dimension: int
    aggregation: Aggregation
    k: int
    country: CountryScoreOut
    ranking: list[CountryScoreOut]
    neighbors: list[NeighborOut]
    explanation: str
    timings: TimingsOut
    support_note: str = SUPPORT_NOTE


class QueryOut(BaseModel):
    """What was analysed. For a built-in example the true label is revealed here, after the prediction."""

    example_id: uuid.UUID | None = None
    actual_country_code: str | None = None
    actual_country_name: str | None = None


class PredictResponse(BaseModel):
    query: QueryOut
    prediction: PredictionOut


class CompareResponse(BaseModel):
    query: QueryOut
    predictions: dict[ModelName, PredictionOut]


class ExampleOut(BaseModel):
    id: uuid.UUID
    url: str
    width: int
    height: int
    attribution: str
    license: str
    source_page_url: str | None


class ExamplesResponse(BaseModel):
    examples: list[ExampleOut]


class IndexedImageOut(ImageOut):
    models: list[ModelName]  # the models that have an embedding of this image in the index


class IndexedCountryOut(BaseModel):
    code: str
    name: str
    images: int


class IndexedImagesResponse(BaseModel):
    total: int  # matching the filter
    offset: int
    limit: int
    countries: list[IndexedCountryOut]
    images: list[IndexedImageOut]


class ModelIndexOut(BaseModel):
    images: int  # embeddings stored for this model
    versions: list[str]


class ModelInfoOut(BaseModel):
    id: ModelName
    label: str
    model_name: str
    dimension: int
    description: str
    loaded: bool
    index: ModelIndexOut


class DatasetOut(BaseModel):
    index_images: int
    test_images: int
    countries: int


class ModelsResponse(BaseModel):
    models: list[ModelInfoOut]
    dataset: DatasetOut
    default_k: int
    max_k: int
    max_upload_mb: int
    aggregations: list[Aggregation]
    ready: bool  # both models have a searchable index


class EvaluationResponse(BaseModel):
    available: bool
    generated_at: str | None = None
    results: dict[str, Any] | None = None
