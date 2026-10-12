from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.chat.entities import ChatSession


class ChatRepository(ABC):
    @abstractmethod
    async def create_session(self) -> ChatSession: ...

    @abstractmethod
    async def get_session(self, session_id: str) -> ChatSession | None: ...

    @abstractmethod
    async def save_session(self, session: ChatSession) -> None: ...
