"""Opt-in end-to-end tests against the REAL stack (PostgreSQL + pgvector, local embedding model, real API).

    docker compose up -d db && alembic upgrade head && python -m app.cli seed
    RUN_INTEGRATION=1 pytest tests/integration

Nothing is mocked. The app has ONE live configuration, so the tests edit it through ``PUT /rag/config`` and the
module fixture puts the original back afterwards (the extra indexes are cheap cache entries).
Tests that need a real LLM additionally require ``LLM_API_KEY`` and are skipped otherwise.
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

if os.getenv("RUN_INTEGRATION") != "1":
    pytest.skip("set RUN_INTEGRATION=1 to run tests against the live database and models", allow_module_level=True)

API = "/api/v1"
ENTITY = {"strategy": "entity"}


@pytest.fixture(scope="module")
def client():
    from app.main import create_app

    with TestClient(create_app()) as c:  # runs the lifespan: DB check, in-process job worker, warm-up
        yield c


@pytest.fixture(scope="module", autouse=True)
def _restore_config(client):
    original = client.get(f"{API}/rag/config").json()
    yield
    client.put(f"{API}/rag/config", json={"chunking": original["chunking"], "retrieval": original["retrieval"]})


@pytest.fixture(scope="module")
def llm_ready(client) -> bool:
    return bool(client.get("/health").json()["llm_configured"])


def _session(client) -> str:
    r = client.post(f"{API}/chat/sessions")
    assert r.status_code == 201
    return r.json()["id"]


def _configure(client, chunking=None, retrieval=None) -> dict:
    r = client.put(f"{API}/rag/config", json={"chunking": chunking or ENTITY, "retrieval": retrieval or {"strategy": "vector"}})
    assert r.status_code == 200, r.text
    return r.json()


def _wait_ready(client, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        idx = client.get(f"{API}/rag/config/index-status").json()
        if idx["status"] == "ready":
            return idx
        assert idx["status"] in ("pending", "building"), idx
        time.sleep(1)
    pytest.fail("index did not become ready in time")


# ------------------------------------------------------------------------------ platform / menu


def test_health_and_app_registry(client):
    assert client.get("/health").json()["status"] == "ok"
    apps = client.get(f"{API}/apps").json()
    assert [a["slug"] if "slug" in a else a["id"] for a in apps] and any("Restaurant" in a["name"] for a in apps)


def test_menu_is_seeded_deterministically(client):
    menu = client.get(f"{API}/restaurant/menu").json()
    assert 50 <= len(menu["dishes"]) <= 100
    assert {c["name"] for c in client.get(f"{API}/restaurant/categories").json()} >= {"Burgers", "Pizza", "Desserts"}
    assert client.get(f"{API}/restaurant/menu/dish_001").json()["name"] == "Classic Beef Burger"
    assert client.get(f"{API}/restaurant/menu/nope").status_code == 404


# ------------------------------------------------------------------------------ cart / orders API


def test_cart_and_order_flow_and_validation(client):
    sid = _session(client)
    assert client.post(f"{API}/carts/{sid}/items", json={"dish_id": "dish_001", "quantity": 0}).status_code == 422
    assert client.post(f"{API}/carts/{sid}/items", json={"dish_id": "dish_001", "quantity": 21}).status_code == 422
    assert client.post(f"{API}/carts/{sid}/items", json={"dish_id": "missing", "quantity": 1}).status_code == 404
    assert client.post(f"{API}/carts/unknown/items", json={"dish_id": "dish_001", "quantity": 1}).status_code == 404

    cart = client.post(f"{API}/carts/{sid}/items", json={"dish_id": "dish_001", "quantity": 2}).json()
    item = cart["items"][0]
    assert float(cart["total"]) == 2 * float(item["unit_price"])
    cart = client.patch(f"{API}/carts/{sid}/items/{item['id']}", json={"quantity": 3}).json()
    assert cart["item_count"] == 3

    r = client.post(f"{API}/orders", json={"session_id": sid, "confirm": False})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_failed"
    assert client.get(f"{API}/carts/{sid}").json()["item_count"] == 3  # nothing was consumed

    order = client.post(f"{API}/orders", json={"session_id": sid, "confirm": True})
    assert order.status_code == 201
    assert client.get(f"{API}/orders/{order.json()['id']}").json()["total"] == order.json()["total"]
    assert client.get(f"{API}/carts/{sid}").json()["items"] == []
    assert client.post(f"{API}/orders", json={"session_id": sid, "confirm": True}).status_code == 409  # empty cart


# ------------------------------------------------------------------------------ real retrieval


def test_real_embeddings_retrieve_the_right_dish(client):
    _configure(client)
    _wait_ready(client)
    r = client.post(f"{API}/rag/retrieve", json={"query": "vegan falafel burger with tahini"})
    assert r.status_code == 200
    trace = r.json()["trace"]
    assert trace["chunks"][0]["dish_names"] == ["Veggie Falafel Burger"]
    assert trace["final_score_type"] == "cosine_similarity" and trace["latency_ms"] > 0
    assert trace["context"].startswith("[1]")


def test_hybrid_and_filters_on_the_real_database(client):
    _configure(client, retrieval={"strategy": "hybrid_rrf", "top_k": 5, "candidate_count": 20, "rrf_k": 60})
    _wait_ready(client)
    r = client.post(
        f"{API}/rag/retrieve",
        json={"query": "creamy cheesy pizza", "filters": {"exclude_allergens": ["dairy"]}},
    ).json()
    assert [s["name"] for s in r["trace"]["stages"]] == ["vector", "lexical", "rrf"]
    assert r["trace"]["stages"][1]["score_type"] == "ts_rank_cd"
    menu = {d["id"]: d for d in client.get(f"{API}/restaurant/menu").json()["dishes"]}
    assert r["dish_ids"] and all("dairy" not in menu[i]["allergens"] for i in r["dish_ids"])


def test_changing_retrieval_reuses_the_index_but_changing_chunking_does_not(client):
    recursive = {"strategy": "recursive", "chunk_size": 350, "chunk_overlap": 35}
    _configure(client)
    fp = _wait_ready(client)["fingerprint"]
    patched = _configure(client, retrieval={"strategy": "hybrid_rrf", "top_k": 8})
    assert patched["index_fingerprint"] == fp and patched["index"]["status"] == "ready"

    patched = _configure(client, chunking=recursive)
    assert patched["index_fingerprint"] != fp
    _wait_ready(client)  # built by the background worker
    assert _configure(client, chunking=recursive)["index"]["status"] == "ready"  # cache hit on the second request


def test_while_a_new_index_builds_retrieval_uses_the_previous_one(client):
    _configure(client)
    _wait_ready(client)
    _configure(client, chunking={"strategy": "fixed", "chunk_size": 777, "chunk_overlap": 11})
    # The worker may finish quickly. Either way the live config answers: from the previous index (with a warning)
    # while the new one builds, or from the new one once ready - never an error.
    r = client.post(f"{API}/rag/retrieve", json={"query": "burger"})
    assert r.status_code == 200, r.text
    trace = r.json()["trace"]
    assert trace["chunks"] and (trace["warnings"] == [] or "still" in trace["warnings"][0])


def test_invalid_config_is_rejected(client):
    bad = {"chunking": {"strategy": "fixed", "chunk_size": 100, "chunk_overlap": 100}}
    r = client.put(f"{API}/rag/config", json=bad)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert client.get(f"{API}/rag/experiments").status_code == 404  # the experiment concept is gone


def test_real_cross_encoder_reranker(client):
    _configure(client, retrieval={"strategy": "hybrid_rrf", "rerank_enabled": True, "top_k": 3, "candidate_count": 10})
    _wait_ready(client)
    r = client.post(f"{API}/rag/retrieve", json={"query": "which dish has peanut sauce"})
    if r.status_code == 502:
        pytest.skip(f"reranker model unavailable on this machine: {r.json()['error']['message']}")
    trace = r.json()["trace"]
    assert trace["stages"][-1]["name"] == "rerank" and trace["final_score_type"] == "cross_encoder_score"
    scores = [c["scores"]["cross_encoder_score"] for c in trace["chunks"]]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------------------ chat without an LLM


def test_chat_pipeline_enforces_allergens_with_the_real_index(client):
    _configure(client)
    _wait_ready(client)
    sid = _session(client)
    out = client.post(f"{API}/chat/sessions/{sid}/messages", json={"content": "I'm allergic to peanuts, recommend a chicken dish"})
    assert out.status_code == 200, out.text
    body = out.json()
    menu = {d["id"]: d for d in client.get(f"{API}/restaurant/menu").json()["dishes"]}
    meta = body["assistant_message"]["metadata"]
    assert "peanuts" in meta["filters"]["exclude_allergens"] and meta["trace"]["chunks"]
    assert all("peanuts" not in menu[i]["allergens"] for i in meta["recommended_dish_ids"])


def test_chat_order_requires_confirmation(client):
    _configure(client)
    _wait_ready(client)
    sid = _session(client)

    def send(text: str) -> dict:
        r = client.post(f"{API}/chat/sessions/{sid}/messages", json={"content": text})
        assert r.status_code == 200, r.text
        return r.json()

    assert send("add two classic beef burgers")["cart"]["item_count"] == 2
    assert send("place my order")["order"] is None
    done = send("yes")
    assert done["order"] is not None and done["cart"]["items"] == []


# ------------------------------------------------------------------------------ evaluation (needs a real LLM)


def test_evaluation_run_end_to_end(client, llm_ready):
    if not llm_ready:
        pytest.skip("LLM_API_KEY is not configured: answer generation needs a real model")
    _configure(client)
    _wait_ready(client)
    run = client.post(f"{API}/evaluations", json={"question_limit": 3}).json()
    deadline = time.time() + 600
    while time.time() < deadline:
        run = client.get(f"{API}/evaluations/{run['id']}").json()
        if run["status"] in ("completed", "failed"):
            break
        time.sleep(3)
    assert run["status"] == "completed", run.get("error")
    for key in ("precision_at_5", "recall_at_5", "non_llm_context_precision", "non_llm_context_recall"):
        assert key in run["metrics"]
    assert len(client.get(f"{API}/evaluations/{run['id']}/results").json()["results"]) == 3


def test_evaluation_without_an_llm_is_refused_clearly(client, llm_ready):
    if llm_ready:
        pytest.skip("an LLM is configured")
    r = client.post(f"{API}/evaluations", json={})
    assert r.status_code == 409 and r.json()["error"]["details"]["code"] == "llm_not_configured"
