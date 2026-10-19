"""Entities and value objects of Country Matcher (no I/O, no framework)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class ModelName(StrEnum):
    """The two image encoders being compared. The value is the public id used in URLs and stored rows."""

    CLIP = "clip"
    RESNET50 = "resnet50"


class Split(StrEnum):
    INDEX = "index"  # searchable: the labelled images a query is compared against
    TEST = "test"  # held out: never searched, only used as evaluation queries and as "try an example" images


class Aggregation(StrEnum):
    MAJORITY = "majority"  # one vote per neighbour
    WEIGHTED = "weighted"  # each neighbour votes with its cosine similarity (the default)


@dataclass(frozen=True)
class ImageRecord:
    id: uuid.UUID
    country_code: str  # ISO 3166-1 alpha-2
    country_name: str
    city: str | None
    original_url: str
    source_page_url: str | None
    storage_path: str
    source_license: str
    source_attribution: str
    image_hash: str  # sha256 of the stored (normalised) bytes: the exact-duplicate key
    perceptual_hash: str  # 64-bit difference hash as 16 hex chars: finds near-duplicates
    scene_group: str  # photos of one place / photographer: the unit that is never split across index and test
    width: int
    height: int
    split: Split
    created_at: datetime


@dataclass(frozen=True)
class NewImage:
    """An image about to be stored (the database assigns the id and timestamp)."""

    country_code: str
    country_name: str
    city: str | None
    original_url: str
    source_page_url: str | None
    storage_path: str
    source_license: str
    source_attribution: str
    image_hash: str
    perceptual_hash: str
    scene_group: str
    width: int
    height: int
    split: Split


@dataclass(frozen=True)
class EmbeddingInfo:
    """Which model (and which version of it) produced a stored embedding."""

    model: ModelName
    model_name: str  # e.g. openai/clip-vit-base-patch32
    model_version: str  # weights revision: embeddings of different versions are never mixed
    dimension: int


@dataclass(frozen=True)
class Neighbor:
    image: ImageRecord
    similarity: float  # cosine similarity of the (L2-normalised) embeddings, 1 = identical direction


@dataclass(frozen=True)
class CountryScore:
    country_code: str
    country_name: str
    votes: int  # how many of the K neighbours carry this label
    weight: float  # sum of those neighbours' similarities
    support: float  # share of the evidence (votes or similarity weight, per the aggregation) in 0..1. NOT a probability
    best_similarity: float
    image_ids: tuple[uuid.UUID, ...] = ()


@dataclass(frozen=True)
class Timings:
    embed_ms: float
    search_ms: float
    total_ms: float


@dataclass
class Prediction:
    model: ModelName
    aggregation: Aggregation
    k: int
    country: CountryScore | None  # None when the index is empty
    ranking: list[CountryScore] = field(default_factory=list)
    neighbors: list[Neighbor] = field(default_factory=list)
    explanation: str = ""
    timings: Timings | None = None
    embedding: EmbeddingInfo | None = None


@dataclass(frozen=True)
class Candidate:
    """A licensed image offered by a source, before it is downloaded."""

    country_code: str
    country_name: str
    city: str | None
    category: str  # the source category it came from (a place: used to keep scenes diverse)
    title: str
    download_url: str  # a reasonably sized rendition on an allowed host
    original_url: str  # the full-size file
    page_url: str | None  # the file's description page (credits, license text)
    license: str
    attribution: str
    author: str
    width: int
    height: int
