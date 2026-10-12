"""RAG Evaluation bounded context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.restaurant.domain.rag.entities import JobStatus


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    key: str  # stable human key, e.g. "q_007"
    category: str
    question: str
    relevant_dish_ids: tuple[str, ...]  # ground truth; NEVER derived from retrieval output
    reference_answer: str
    required_constraints: dict[str, Any]
    expected_behavior: str
    history: tuple[dict[str, str], ...] = ()  # prior turns for follow-up questions
    position: int = 0

    def retrieval_query(self) -> str:
        """Follow-ups are retrieved with the previous user turn prepended (documented behaviour)."""
        prior = [m["content"] for m in self.history if m.get("role") == "user"]
        return f"{prior[-1]} {self.question}" if prior else self.question


@dataclass(frozen=True)
class EvalDataset:
    id: str
    name: str
    version: str
    menu_dataset_version: str
    content_hash: str
    description: str
    question_count: int = 0


@dataclass
class QueryResult:
    run_id: str
    question_key: str
    category: str
    question: str
    query: str
    relevant_dish_ids: list[str]
    retrieved_dish_ids: list[str]
    contexts: list[str]
    answer: str
    retrieval_metrics: dict[str, float | None] = field(default_factory=dict)
    ragas_scores: dict[str, float | None] = field(default_factory=dict)
    latency_ms: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class LiveEvaluation:
    """One real customer question, scored automatically after the assistant answered it.

    There is no pre-written ground truth: ``relevant_dish_ids`` is judged by the judge LLM against the whole
    menu, and the RAGAS metrics are the reference-free ones (faithfulness, answer relevancy, context precision).
    """

    id: str
    session_id: str
    message_id: str  # the assistant message that answered the question
    config_snapshot: dict[str, Any]
    question: str  # exactly what the customer typed
    query: str  # the standalone search query that was retrieved with
    filters: dict[str, Any]  # hard filters the chat applied (replayed for the depth-10 retrieval)
    answer: str  # the reply the customer actually saw
    contexts: list[str]  # the chunks that were given to the LLM
    status: JobStatus = JobStatus.PENDING
    stage: str = "queued"  # (awaiting ->) queued -> judging_relevance -> retrieval -> ragas -> done
    relevant_dish_ids: list[str] = field(default_factory=list)  # judged by the LLM
    retrieved_dish_ids: list[str] = field(default_factory=list)  # depth-10 ranking, dish-level
    retrieval_metrics: dict[str, float | None] = field(default_factory=dict)
    ragas_scores: dict[str, float | None] = field(default_factory=dict)
    latency_ms: float = 0.0
    judge_model: str = ""
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    completed_at: datetime | None = None


@dataclass
class EvaluationRun:
    id: str
    dataset_id: str
    status: JobStatus
    config_snapshot: dict[str, Any]
    fingerprint: str
    settings: dict[str, Any] = field(default_factory=dict)  # judge model, embedding, K values ...
    progress: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
