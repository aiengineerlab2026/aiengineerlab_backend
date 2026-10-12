from __future__ import annotations

import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.chat.entities import ChatSession
from app.domain.chat.repositories import ChatRepository
from app.infrastructure.db.models import ChatSessionModel


class SqlChatRepository(ChatRepository):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def create_session(self) -> ChatSession:
        m = ChatSessionModel(state={})
        self._s.add(m)
        await self._s.flush()
        await self._s.refresh(m)
        return ChatSession(id=str(m.id), state={}, created_at=m.created_at)

    async def get_session(self, session_id: str) -> ChatSession | None:
        try:
            sid = uuid.UUID(session_id)
        except ValueError:
            return None
        m = await self._s.get(ChatSessionModel, sid)
        return ChatSession(id=str(m.id), state=dict(m.state or {}), created_at=m.created_at) if m else None

    async def save_session(self, session: ChatSession) -> None:
        await self._s.execute(
            update(ChatSessionModel).where(ChatSessionModel.id == uuid.UUID(session.id)).values(state=session.state)
        )
