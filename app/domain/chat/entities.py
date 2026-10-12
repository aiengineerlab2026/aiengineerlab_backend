"""Chat & Conversation bounded context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class ChatSession:
    id: str
    state: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)

    # --- conversational state kept per demo session (no auth in the MVP) ---
    @property
    def pending_order_confirmation(self) -> bool:
        return bool(self.state.get("pending_order_confirmation"))

    @property
    def last_dish_ids(self) -> list[str]:
        return list(self.state.get("last_dish_ids", []))


@dataclass
class ChatMessage:
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_now)
