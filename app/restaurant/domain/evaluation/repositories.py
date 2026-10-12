from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from app.restaurant.domain.evaluation.entities import (
    EvalDataset,
    EvalQuestion,
    EvaluationRun,
    LiveEvaluation,
    QueryResult,
)


class EvaluationRepository(ABC):
    @abstractmethod
    async def get_dataset(self, name: str | None = None, version: str | None = None) -> EvalDataset | None:
        """Latest version if ``version`` is omitted."""

    @abstractmethod
    async def get_dataset_by_id(self, dataset_id: str) -> EvalDataset | None: ...

    @abstractmethod
    async def list_questions(self, dataset_id: str) -> list[EvalQuestion]: ...

    @abstractmethod
    async def add_run(self, run: EvaluationRun) -> None: ...

    @abstractmethod
    async def get_run(self, run_id: str) -> EvaluationRun | None: ...

    @abstractmethod
    async def list_runs(self, limit: int = 100) -> list[EvaluationRun]: ...

    @abstractmethod
    async def save_run(self, run: EvaluationRun) -> None: ...

    @abstractmethod
    async def update_progress(self, run_id: str, progress: dict) -> None: ...

    @abstractmethod
    async def add_query_results(self, results: list[QueryResult]) -> None: ...

    @abstractmethod
    async def list_query_results(self, run_id: str) -> list[QueryResult]: ...

    @abstractmethod
    async def claim_pending_run(self) -> EvaluationRun | None: ...

    @abstractmethod
    async def requeue_interrupted(self) -> int: ...

    # ---- live evaluation of real customer questions ------------------------------------------

    @abstractmethod
    async def add_live(self, item: LiveEvaluation) -> None: ...

    @abstractmethod
    async def save_live(self, item: LiveEvaluation) -> None: ...

    @abstractmethod
    async def list_live(self, limit: int = 50) -> list[LiveEvaluation]:
        """Newest first."""

    @abstractmethod
    async def list_live_by_message_ids(self, message_ids: list[str]) -> list[LiveEvaluation]:
        """The evaluations of these assistant messages (the browser owns the conversation; the server owns scores)."""

    @abstractmethod
    async def release_awaiting(self, message_ids: list[str]) -> int:
        """Move the AWAITING items of these assistant messages to PENDING so the worker scores them. Returns how many."""

    @abstractmethod
    async def purge_awaiting(self, older_than: datetime) -> int:
        """Delete AWAITING items (and the customer text they hold) that nobody asked to evaluate in time."""

    @abstractmethod
    async def redact_live(self, item_id: str) -> None:
        """Blank the customer's text (question, query, answer, contexts) once scoring is over. Scores stay."""

    @abstractmethod
    async def claim_pending_live(self) -> LiveEvaluation | None:
        """Atomically move the oldest pending item to RUNNING (SKIP LOCKED)."""

    @abstractmethod
    async def requeue_interrupted_live(self) -> int: ...

    @abstractmethod
    async def clear_live(self) -> int: ...
