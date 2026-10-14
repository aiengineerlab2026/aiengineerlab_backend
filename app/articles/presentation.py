"""Articles API: public reading, and everything else behind the admin login (a signed bearer token)."""

from __future__ import annotations

import hmac
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.articles import security
from app.articles.container import AdminInfra
from app.articles.sanitize import clean_html, excerpt, plain_text
from app.articles.schemas import (
    ArticleIn,
    ArticleOut,
    ArticleSummaryOut,
    LoginIn,
    PasswordChangeIn,
    SessionOut,
    TokenOut,
)
from app.articles.store import AdminStore, Article
from app.errors import NotFoundError, RateLimited, Unauthorized, ValidationFailed
from app.restaurant.presentation.deps import client_ip

router = APIRouter(tags=["articles"])


def get_admin(request: Request) -> AdminInfra:
    return request.app.state.admin


AdminDep = Annotated[AdminInfra, Depends(get_admin)]


async def get_store(admin: AdminDep) -> AsyncIterator[AdminStore]:
    async with admin.store_factory() as store:  # committed right after the handler returns
        yield store


StoreDep = Annotated[AdminStore, Depends(get_store, scope="function")]


async def _password_matches(admin: AdminInfra, store: AdminStore, password: str) -> bool:
    stored = await store.get_password_hash()
    if stored is None:  # the password has never been changed: the configured default is still valid
        return hmac.compare_digest(password.encode(), admin.settings.admin_default_password.encode())
    return security.verify_password(password, stored)


async def _issue_token(admin: AdminInfra, store: AdminStore) -> TokenOut:
    s = admin.settings
    fingerprint = security.password_fingerprint(await store.get_password_hash())
    token = security.make_token(admin.secret, s.admin_username, fingerprint, s.admin_token_ttl_seconds)
    return TokenOut(token=token, username=s.admin_username, expires_in=s.admin_token_ttl_seconds)


async def require_admin(request: Request, admin: AdminDep, store: StoreDep) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    parsed = security.read_token(admin.secret, token.strip()) if scheme.lower() == "bearer" else None
    if parsed is None:
        raise Unauthorized("Please log in.")
    username, fingerprint = parsed
    current = security.password_fingerprint(await store.get_password_hash())
    if username != admin.settings.admin_username or not hmac.compare_digest(fingerprint, current):
        raise Unauthorized("Your session has expired. Please log in again.")
    return username


AdminUser = Annotated[str, Depends(require_admin)]


def _summary(a: Article) -> ArticleSummaryOut:
    return ArticleSummaryOut(
        id=a.id, title=a.title, excerpt=excerpt(a.content), created_at=a.created_at, updated_at=a.updated_at
    )


def _full(a: Article) -> ArticleOut:
    return ArticleOut(**_summary(a).model_dump(), content=a.content)


def _checked(body: ArticleIn) -> tuple[str, str]:
    content = clean_html(body.content)
    if not plain_text(content) and "<img" not in content:  # an article of only pictures is fine
        raise ValidationFailed("The article cannot be empty.")
    return body.title, content


# ----------------------------------------------------------------------------------------------------- admin


@router.post("/admin/login", response_model=TokenOut, summary="Log in as the admin")
async def login(body: LoginIn, request: Request, admin: AdminDep, store: StoreDep) -> TokenOut:
    key = client_ip(request, admin.settings.trust_proxy_headers)
    if admin.login_limiter.enabled and admin.login_limiter.peek(key).remaining == 0:
        wait = admin.login_limiter.peek(key).reset_in_seconds
        raise RateLimited("Too many failed attempts. Try again later.", details={"retry_after_seconds": wait})
    user_ok = hmac.compare_digest(body.username.encode(), admin.settings.admin_username.encode())
    password_ok = await _password_matches(admin, store, body.password)  # always checked, so timing does not reveal the user
    if not (user_ok and password_ok):
        admin.login_limiter.hit(key)
        raise Unauthorized("Wrong username or password.")
    return await _issue_token(admin, store)


@router.get("/admin/session", response_model=SessionOut, summary="Who is logged in (validates a stored token)")
async def session(username: AdminUser) -> SessionOut:
    return SessionOut(username=username)


@router.post("/admin/password", response_model=TokenOut, summary="Change the admin password")
async def change_password(body: PasswordChangeIn, admin: AdminDep, store: StoreDep, _: AdminUser) -> TokenOut:
    if not await _password_matches(admin, store, body.current_password):
        raise ValidationFailed("The current password is wrong.")
    if body.new_password == body.current_password:
        raise ValidationFailed("The new password must be different from the current one.")
    await store.set_password_hash(security.hash_password(body.new_password))
    return await _issue_token(admin, store)  # the old tokens stop working, so hand back a fresh one


# -------------------------------------------------------------------------------------------------- articles


@router.get("/articles", response_model=list[ArticleSummaryOut], summary="List articles, newest first")
async def list_articles(store: StoreDep) -> list[ArticleSummaryOut]:
    return [_summary(a) for a in await store.list_articles()]


@router.get("/articles/{article_id}", response_model=ArticleOut, summary="One article")
async def get_article(article_id: uuid.UUID, store: StoreDep) -> ArticleOut:
    article = await store.get_article(article_id)
    if article is None:
        raise NotFoundError("Article not found.")
    return _full(article)


@router.post("/articles", response_model=ArticleOut, status_code=201, summary="Add an article")
async def add_article(body: ArticleIn, store: StoreDep, _: AdminUser) -> ArticleOut:
    return _full(await store.add_article(*_checked(body)))


@router.put("/articles/{article_id}", response_model=ArticleOut, summary="Edit an article")
async def update_article(article_id: uuid.UUID, body: ArticleIn, store: StoreDep, _: AdminUser) -> ArticleOut:
    article = await store.update_article(article_id, *_checked(body))
    if article is None:
        raise NotFoundError("Article not found.")
    return _full(article)


@router.delete("/articles/{article_id}", status_code=204, summary="Delete an article")
async def delete_article(article_id: uuid.UUID, store: StoreDep, _: AdminUser) -> Response:
    if not await store.delete_article(article_id):
        raise NotFoundError("Article not found.")
    return Response(status_code=204)
