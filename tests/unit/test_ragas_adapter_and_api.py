"""RAGAS adapter wiring (framework call mocked) and the HTTP error envelope."""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.config import Settings, build_async_url
from app.error_handlers import register_error_handlers
from app.errors import (
    ConflictError,
    ExternalServiceError,
    NotFoundError,
    ServiceNotConfigured,
    ValidationFailed,
)
from app.restaurant.application.evaluation.ports import RagasInput
from app.restaurant.infrastructure.evaluation import ragas_evaluator as mod

# ------------------------------------------------------------------------------ RAGAS adapter


def test_clean_maps_nan_inf_and_garbage_to_none():
    assert mod._clean(0.123456) == 0.1235
    assert mod._clean(math.nan) is None and mod._clean(math.inf) is None
    assert mod._clean(None) is None and mod._clean("x") is None


DISH_A = "Dish: Grilled Lemon Herb Chicken (dish_025).\nCategory: Chicken.\nPrice: 15.50 USD."
DISH_B = "Dish: Margherita Pizza (dish_040).\nCategory: Pizza.\nPrice: 10.00 USD."


async def test_adapter_scores_with_real_ragas_and_needs_no_llm():
    """Runs the genuine RAGAS non-LLM metrics: no judge, no embeddings, no network."""
    out = await mod.RagasFrameworkEvaluator().evaluate([
        RagasInput("q1", [DISH_A, DISH_B], [DISH_A]),  # the relevant dish is retrieved first
        RagasInput("q2", [DISH_B], [DISH_A]),  # nothing relevant retrieved
        RagasInput("q3", [DISH_B, DISH_A], [DISH_A]),  # relevant dish retrieved second
    ])
    assert out[0] == {"non_llm_context_precision": 1.0, "non_llm_context_recall": 1.0}
    assert out[1] == {"non_llm_context_precision": 0.0, "non_llm_context_recall": 0.0}
    assert out[2]["non_llm_context_recall"] == 1.0 and out[2]["non_llm_context_precision"] == 0.5  # rank-aware


async def test_adapter_marks_undefined_cases_instead_of_scoring_them():
    out = await mod.RagasFrameworkEvaluator().evaluate([
        RagasInput("nothing should match", [DISH_A], []),  # no reference dishes -> undefined, not zero
        RagasInput("nothing retrieved", [], [DISH_A]),  # recall 0, precision undefined
    ])
    assert out[0] == {"non_llm_context_precision": None, "non_llm_context_recall": None}
    assert out[1] == {"non_llm_context_precision": None, "non_llm_context_recall": 0.0}
    assert await mod.RagasFrameworkEvaluator().evaluate([]) == []


async def test_only_llm_free_ragas_metrics_are_wired(monkeypatch):
    import ragas

    seen = {}

    class FakeResult:
        scores = [{"non_llm_context_precision_with_reference": 0.5, "non_llm_context_recall": 1.0}]

    def fake_evaluate(dataset, **kw):
        seen.update(kw)
        return FakeResult()

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    out = await mod.RagasFrameworkEvaluator().evaluate([RagasInput("q", [DISH_A], [DISH_A])])
    assert out == [{"non_llm_context_precision": 0.5, "non_llm_context_recall": 1.0}]
    assert [type(m).__name__ for m in seen["metrics"]] == ["NonLLMContextPrecisionWithReference", "NonLLMContextRecall"]
    assert all(m.threshold == mod.MATCH_THRESHOLD for m in seen["metrics"])
    assert "llm" not in seen and "embeddings" not in seen  # nothing for RAGAS to call a model with


def test_the_default_threshold_would_accept_different_dishes_so_a_strict_one_is_used():
    from rapidfuzz.distance import Levenshtein

    from app.restaurant.domain.canonical import render_dish
    from app.restaurant.infrastructure.seed.menu_data import build_menu

    dishes = build_menu()
    texts = [render_dish(d) for d in dishes]
    cross = [1 - Levenshtein.normalized_distance(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
    assert sum(c >= 0.5 for c in cross) / len(cross) > 0.9  # RAGAS's default 0.5 would call ~everything a match
    assert max(cross) < mod.MATCH_THRESHOLD  # ... but no two different dishes reach the strict threshold


# ------------------------------------------------------------------------------ error envelope


class Body(BaseModel):
    quantity: int = Field(ge=1, le=20)


def _app() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    @app.post("/validate")
    async def validate(body: Body):
        return {"ok": True}

    for path, exc in {
        "/nf": NotFoundError("no such thing"), "/conflict": ConflictError("busy", details={"a": 1}),
        "/invalid": ValidationFailed("bad"), "/noconf": ServiceNotConfigured("set a key"),
        "/ext": ExternalServiceError("provider down"), "/boom": RuntimeError("secret internals"),
    }.items():
        def make(e):
            async def h():
                raise e
            return h
        app.get(path)(make(exc))
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("path", "status", "code"),
    [("/nf", 404, "not_found"), ("/conflict", 409, "conflict"), ("/invalid", 422, "validation_failed"),
     ("/noconf", 503, "service_not_configured"), ("/ext", 502, "external_service_error"), ("/boom", 500, "internal_error")],
)
def test_domain_errors_map_to_status_and_envelope(path, status, code):
    r = _app().get(path)
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code


def test_unexpected_errors_do_not_leak_internals():
    assert "secret" not in _app().get("/boom").text


def test_request_validation_uses_the_same_envelope():
    r = _app().post("/validate", json={"quantity": 0})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["details"]["errors"][0]["loc"] == ["body", "quantity"]
    assert _app().post("/validate", json={"quantity": 3}).json() == {"ok": True}


# ------------------------------------------------------------------------------ settings


def test_database_url_is_rewritten_for_asyncpg_and_password_is_decoded():
    url = build_async_url("postgresql://postgres:p%40ss%2Fword@db.example.supabase.co:5432/postgres?sslmode=require")
    assert url.drivername == "postgresql+asyncpg" and url.password == "p@ss/word"
    assert "sslmode" not in url.query
    with pytest.raises(RuntimeError):
        build_async_url("")


def test_tls_is_on_for_remote_hosts_and_off_for_localhost():
    assert Settings(database_url="postgresql://u:p@localhost:5433/d").connect_args == {}
    assert Settings(database_url="postgresql://u:p@db.x.supabase.co:5432/d").connect_args == {"ssl": "require"}
    pooled = Settings(database_url="postgresql://u:p@pooler.supabase.com:6543/d").connect_args
    assert pooled["statement_cache_size"] == 0


def test_cors_origins_and_llm_flag_parse_from_env():
    s = Settings(cors_origins="http://a.test, http://b.test ,", llm_api_key="")
    assert s.cors_origin_list == ["http://a.test", "http://b.test"] and not s.llm_configured
    assert Settings(llm_api_key="k").llm_configured
    assert Settings(llm_base_url="http://localhost:11434/v1").llm_configured
