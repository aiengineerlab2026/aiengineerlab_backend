"""Articles and the admin login: sanitising, tokens, password change and the HTTP API (in-memory store, no database)."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.articles import presentation, security
from app.articles.container import AdminInfra
from app.articles.sanitize import clean_html, excerpt
from app.articles.store import Article
from app.config import Settings
from app.error_handlers import register_error_handlers


class MemoryStore:
    def __init__(self):
        self.articles: dict[uuid.UUID, Article] = {}
        self.password_hash: str | None = None

    async def list_articles(self):
        return sorted(self.articles.values(), key=lambda a: a.created_at, reverse=True)

    async def get_article(self, article_id):
        return self.articles.get(article_id)

    async def add_article(self, title, content):
        now = datetime.now(UTC)
        a = Article(uuid.uuid4(), title, content, now, now)
        self.articles[a.id] = a
        return a

    async def update_article(self, article_id, title, content):
        old = self.articles.get(article_id)
        if old is None:
            return None
        a = Article(old.id, title, content, old.created_at, datetime.now(UTC))
        self.articles[a.id] = a
        return a

    async def delete_article(self, article_id):
        return self.articles.pop(article_id, None) is not None

    async def get_password_hash(self):
        return self.password_hash

    async def set_password_hash(self, password_hash):
        self.password_hash = password_hash


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def client(store):
    @asynccontextmanager
    async def open_store():
        yield store

    app = FastAPI()
    app.state.admin = AdminInfra(Settings(admin_token_secret="test-secret", admin_login_max_failures=3), open_store)
    register_error_handlers(app)
    app.include_router(presentation.router, prefix="/api/v1")
    return TestClient(app)


def login(client, password="admin", username="admin"):
    return client.post("/api/v1/admin/login", json={"username": username, "password": password})


def auth(client, password="admin"):
    return {"Authorization": f"Bearer {login(client, password).json()['token']}"}


ARTICLE = {"title": "  Hello   RAG ", "content": "<h2>Intro</h2><p>First <strong>bold</strong> paragraph.</p>"}

# -------------------------------------------------------------------------------------------------- unit


def test_password_hash_roundtrip():
    h = security.hash_password("s3cret-pass")
    assert h != "s3cret-pass" and security.verify_password("s3cret-pass", h)
    assert not security.verify_password("other", h) and not security.verify_password("x", "garbage")


def test_token_is_signed_and_expires():
    t = security.make_token("k", "admin", "fp", 60, now=1000)
    assert security.read_token("k", t, now=1030) == ("admin", "fp")
    assert security.read_token("k", t, now=1100) is None  # expired
    assert security.read_token("other", t, now=1030) is None  # wrong secret
    assert security.read_token("k", t[:-2] + "xx", now=1030) is None  # tampered
    assert security.read_token("k", "", now=1030) is None and security.read_token("k", "a.b", now=1030) is None


def test_sanitiser_strips_scripts_handlers_and_bad_links_but_keeps_formatting():
    dirty = (
        '<p onclick="x()">Hi <strong>there</strong><script>alert(1)</script></p>'
        '<a href="javascript:alert(1)">bad</a><a href="https://ok.dev">ok</a><img src=x onerror=alert(1)>'
    )
    out = clean_html(dirty)
    assert "<script" not in out and "onclick" not in out and "javascript:" not in out and "onerror" not in out
    assert "<strong>there</strong>" in out and 'href="https://ok.dev"' in out and "noopener" in out


def test_excerpt_keeps_neighbouring_blocks_apart_and_truncates():
    assert excerpt("<p>one</p><p>two &amp; three</p>") == "one two & three"
    assert excerpt("<p>" + "word " * 100 + "</p>").endswith("…") and len(excerpt("<p>" + "word " * 100 + "</p>")) <= 200


# ------------------------------------------------------------------------------------------------- login


def test_default_credentials_log_in_and_wrong_ones_do_not(client):
    ok = login(client)
    assert ok.status_code == 200 and ok.json()["username"] == "admin" and ok.json()["token"]
    for bad in (login(client, "nope"), login(client, "admin", "root")):
        assert bad.status_code == 401 and bad.json()["error"]["code"] == "unauthorized"


def test_repeated_failures_are_blocked_then_even_the_right_password_waits(client):
    for _ in range(3):
        assert login(client, "nope").status_code == 401
    blocked = login(client, "admin")
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers


def test_session_requires_a_valid_token(client):
    assert client.get("/api/v1/admin/session").status_code == 401
    assert client.get("/api/v1/admin/session", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert client.get("/api/v1/admin/session", headers=auth(client)).json() == {"username": "admin"}


# --------------------------------------------------------------------------------------- change password


def test_change_password_replaces_the_default_and_logs_out_old_sessions(client, store):
    old = auth(client)
    r = client.post(
        "/api/v1/admin/password", headers=old, json={"current_password": "admin", "new_password": "a-better-pass"}
    )
    assert r.status_code == 200
    assert store.password_hash and "a-better-pass" not in store.password_hash  # only a hash is kept
    assert client.get("/api/v1/admin/session", headers=old).status_code == 401  # the old token is dead
    assert client.get("/api/v1/admin/session", headers={"Authorization": f"Bearer {r.json()['token']}"}).status_code == 200
    assert login(client, "admin").status_code == 401  # the default no longer works
    assert login(client, "a-better-pass").status_code == 200


def test_change_password_rejects_wrong_current_short_or_same(client):
    h = auth(client)
    url = "/api/v1/admin/password"
    assert client.post(url, headers=h, json={"current_password": "x", "new_password": "long-enough"}).status_code == 422
    assert client.post(url, headers=h, json={"current_password": "admin", "new_password": "short"}).status_code == 422
    same = client.post(url, headers=h, json={"current_password": "admin", "new_password": "admin"})
    assert same.status_code == 422
    assert client.post(url, json={"current_password": "admin", "new_password": "long-enough"}).status_code == 401


# ------------------------------------------------------------------------------------------------ articles


def test_articles_are_public_to_read_but_admin_only_to_change(client):
    assert client.get("/api/v1/articles").json() == []
    assert client.post("/api/v1/articles", json=ARTICLE).status_code == 401
    created = client.post("/api/v1/articles", json=ARTICLE, headers=auth(client))
    assert created.status_code == 201
    a = created.json()
    assert a["title"] == "Hello RAG" and "<strong>bold</strong>" in a["content"]

    listing = client.get("/api/v1/articles").json()  # no token needed
    assert [x["id"] for x in listing] == [a["id"]] and listing[0]["excerpt"] == "Intro First bold paragraph."
    assert "content" not in listing[0]
    assert client.get(f"/api/v1/articles/{a['id']}").json()["content"] == a["content"]

    assert client.put(f"/api/v1/articles/{a['id']}", json=ARTICLE).status_code == 401
    assert client.delete(f"/api/v1/articles/{a['id']}").status_code == 401


def test_edit_and_delete(client):
    h = auth(client)
    a = client.post("/api/v1/articles", json=ARTICLE, headers=h).json()
    upd = client.put(f"/api/v1/articles/{a['id']}", headers=h, json={"title": "New", "content": "<p>Changed</p>"})
    assert upd.status_code == 200 and upd.json()["title"] == "New" and upd.json()["content"] == "<p>Changed</p>"
    assert client.delete(f"/api/v1/articles/{a['id']}", headers=h).status_code == 204
    assert client.get(f"/api/v1/articles/{a['id']}").status_code == 404
    assert client.delete(f"/api/v1/articles/{a['id']}", headers=h).status_code == 404
    assert client.put(f"/api/v1/articles/{uuid.uuid4()}", headers=h, json=ARTICLE).status_code == 404


def test_stored_content_is_sanitised_and_empty_articles_are_rejected(client):
    h = auth(client)
    evil = {"title": "x", "content": "<p>safe</p><script>alert(1)</script>"}
    assert client.post("/api/v1/articles", json=evil, headers=h).json()["content"] == "<p>safe</p>"
    for body in ({"title": "  ", "content": "<p>x</p>"}, {"title": "x", "content": "<p> </p><br>"}):
        assert client.post("/api/v1/articles", json=body, headers=h).status_code == 422


PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


def test_sanitiser_keeps_pasted_images_but_not_unsafe_sources():
    assert f'src="{PIXEL}"' in clean_html(f'<p><img src="{PIXEL}" alt="x"></p>')
    assert 'src="https://cdn.dev/a.png"' in clean_html('<img src="https://cdn.dev/a.png">')
    for bad in (
        '<img src="javascript:alert(1)">',
        '<img src="data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=">',
        '<img src="data:text/html;base64,PHNjcmlwdD4=">',
        '<img src="http://insecure.dev/a.png">',
    ):
        assert "src=" not in clean_html(bad), bad
    assert "href=" not in clean_html(f'<a href="{PIXEL}">x</a>')  # data: is for pictures only, never for links
    assert "onerror" not in clean_html(f'<img src="{PIXEL}" onerror="alert(1)">')


def test_an_article_of_only_a_picture_is_accepted(client):
    body = {"title": "Chart", "content": f'<img src="{PIXEL}">'}
    r = client.post("/api/v1/articles", json=body, headers=auth(client))
    assert r.status_code == 201 and PIXEL in r.json()["content"] and r.json()["excerpt"] == ""


def test_resized_images_keep_numeric_sizes_only():
    out = clean_html(f'<img src="{PIXEL}" width="240" height="135.5">')
    assert 'width="240"' in out and 'height="135.5"' in out
    bad = clean_html(f'<img src="{PIXEL}" width="100%" height="expression(alert(1))">')
    assert "width=" not in bad and "height=" not in bad
