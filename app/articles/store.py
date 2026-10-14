"""Persistence of articles and of the admin password hash, behind a small port so tests can use memory."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.articles.models import AdminCredentialModel, ArticleModel


@dataclass(frozen=True)
class Article:
    id: uuid.UUID
    title: str
    content: str
    created_at: datetime
    updated_at: datetime


class AdminStore(Protocol):
    async def list_articles(self) -> list[Article]: ...
    async def get_article(self, article_id: uuid.UUID) -> Article | None: ...
    async def add_article(self, title: str, content: str) -> Article: ...
    async def update_article(self, article_id: uuid.UUID, title: str, content: str) -> Article | None: ...
    async def delete_article(self, article_id: uuid.UUID) -> bool: ...
    async def get_password_hash(self) -> str | None: ...
    async def set_password_hash(self, password_hash: str) -> None: ...


def _article(row: ArticleModel) -> Article:
    return Article(row.id, row.title, row.content, row.created_at, row.updated_at)


class SqlAdminStore:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_articles(self) -> list[Article]:
        rows = await self.session.scalars(select(ArticleModel).order_by(ArticleModel.created_at.desc()))
        return [_article(r) for r in rows]

    async def get_article(self, article_id: uuid.UUID) -> Article | None:
        row = await self.session.get(ArticleModel, article_id)
        return _article(row) if row else None

    async def add_article(self, title: str, content: str) -> Article:
        row = ArticleModel(title=title, content=content)
        self.session.add(row)
        await self.session.flush()
        await self.session.refresh(row)
        return _article(row)

    async def update_article(self, article_id: uuid.UUID, title: str, content: str) -> Article | None:
        row = await self.session.get(ArticleModel, article_id)
        if row is None:
            return None
        row.title, row.content = title, content
        await self.session.flush()
        await self.session.refresh(row)
        return _article(row)

    async def delete_article(self, article_id: uuid.UUID) -> bool:
        result = await self.session.execute(delete(ArticleModel).where(ArticleModel.id == article_id))
        return result.rowcount > 0

    async def get_password_hash(self) -> str | None:
        row = await self.session.get(AdminCredentialModel, 1)
        return row.password_hash if row else None

    async def set_password_hash(self, password_hash: str) -> None:
        row = await self.session.get(AdminCredentialModel, 1)
        if row is None:
            self.session.add(AdminCredentialModel(id=1, password_hash=password_hash))
        else:
            row.password_hash = password_hash
        await self.session.flush()
