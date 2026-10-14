from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class LoginIn(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class TokenOut(BaseModel):
    token: str
    username: str
    expires_in: int


class SessionOut(BaseModel):
    username: str


class PasswordChangeIn(BaseModel):
    current_password: str = Field(max_length=200)
    new_password: str = Field(min_length=8, max_length=200)


class ArticleIn(BaseModel):
    title: str = Field(max_length=200)
    content: str = Field(max_length=20_000_000)  # HTML from the rich text editor; pasted images are inside it as data URIs

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("The title cannot be empty.")
        return v


class ArticleSummaryOut(BaseModel):
    id: uuid.UUID
    title: str
    excerpt: str
    created_at: datetime
    updated_at: datetime


class ArticleOut(ArticleSummaryOut):
    content: str
