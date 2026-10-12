"""Live evaluation: real customer questions are scored automatically, with no predefined dataset."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

from conftest import ScriptedLLM, World  # type: ignore[import-not-found]
from test_evaluation import FakeRagas, MemEvalRepo  # type: ignore[import-not-found]

from app.restaurant.application.chat.parser import ParsedTurn, SearchConstraints
from app.restaurant.application.chat.service import AnswerOut
from app.restaurant.application.evaluation.live import LiveEvaluator, RelevantDishes, menu_table, summarize
from app.restaurant.domain.evaluation.entities import LiveEvaluation
from app.restaurant.domain.filters import DishFilter
from app.restaurant.domain.rag.entities import JobStatus
from app.restaurant.domain.rag.ranking import rank_dishes
from app.restaurant.domain.rag.value_objects import ChunkingConfig, ChunkingStrategy, RetrievalConfig

QUESTION = "something spicy with chicken"


async def _world(judge_ids=None, *, ragas=None, retrieval=None):
    """A live configuration + built index; returns (world, config, repo, evaluator, item, ragas)."""
    w = World(ScriptedLLM(text="unused"))
    chunking = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    exp, _ = await w.config.update(chunking, retrieval or RetrievalConfig(top_k=5))
    await w.build_index(chunking)
    repo = MemEvalRepo(None, [])  # type: ignore[arg-type]
    ragas = ragas or FakeRagas()
    judge = ScriptedLLM(structured={"RelevantDishes": RelevantDishes(relevant_dish_ids=judge_ids or [])})

    @asynccontextmanager
    async def uow():
        yield SimpleNamespace(eval_repo=repo, retrieval=w.retrieval, menu_repo=w.menu)

    item = LiveEvaluation(
        id="live-1", session_id="s1", message_id="m1", config_snapshot={
            "chunking": exp.chunking.model_dump(mode="json"), "retrieval": exp.retrieval.model_dump(mode="json"),
            "index_fingerprint": exp.index_fingerprint,
        },
        question=QUESTION, query=QUESTION, filters={}, answer="Try the Spicy Peri-Peri Chicken.",
        contexts=["Dish: Spicy Peri-Peri Chicken (dish_028)."],
    )
    await repo.add_live(item)
    return w, exp, repo, LiveEvaluator(uow, judge, ragas), item, ragas, judge


async def _ranking(w, exp, query=QUESTION, k=10):
    deep = exp.retrieval.model_copy(update={"top_k": k, "candidate_count": k})
    trace = await w.retrieval.retrieve(
        query, chunking=exp.chunking, retrieval=deep, index_fingerprint=exp.index_fingerprint
    )
    return rank_dishes([c.dish_ids for c in trace.chunks])


async def test_metrics_come_from_the_judged_relevant_dishes_and_real_retrieval():
    w, exp, repo, ev, item, ragas, _ = await _world()
    ranked = await _ranking(w, exp)
    assert len(ranked) >= 5
    relevant = ranked[:2]  # the judge says the top two retrieved dishes are the relevant ones
    judge = ScriptedLLM(structured={"RelevantDishes": RelevantDishes(relevant_dish_ids=relevant)})
    ev._judge = judge  # noqa: SLF001

    await ev.process(item)

    saved = repo.live["live-1"]
    assert saved.status == JobStatus.COMPLETED and saved.stage == "done" and saved.error is None
    assert saved.relevant_dish_ids == relevant
    assert saved.retrieved_dish_ids[:10] == ranked[:10]
    assert saved.retrieval_metrics["recall_at_5"] == 1.0  # both relevant dishes are inside the top 5
    assert saved.retrieval_metrics["precision_at_5"] == 2 / 5
    assert saved.ragas_scores == {"non_llm_context_precision": 0.4, "non_llm_context_recall": 0.6}
    assert saved.completed_at is not None and saved.judge_model == "fake-llm"


async def test_ragas_gets_the_chunks_the_customer_was_answered_from_and_the_index_chunks_covering_the_judged_relevant_dishes_as_reference():
    w, exp, _, ev, item, ragas, _ = await _world()
    ranked = await _ranking(w, exp)
    ev._judge = ScriptedLLM(structured={"RelevantDishes": RelevantDishes(relevant_dish_ids=ranked[:2])})  # noqa: SLF001
    await ev.process(item)
    (sample,) = ragas.seen
    assert sample.user_input == QUESTION
    assert sample.retrieved_contexts == ["Dish: Spicy Peri-Peri Chicken (dish_028)."]
    covering = [c.content for cs in w.store.chunks.values() for c, _ in cs if set(c.dish_ids) & set(ranked[:2])]
    assert sample.reference_contexts == list(dict.fromkeys(covering)) and len(covering) == 2  # judged-relevant only


async def test_judge_ids_not_on_the_menu_are_dropped():
    w, exp, repo, ev, item, *_ = await _world()
    ranked = await _ranking(w, exp)
    ev._judge = ScriptedLLM(  # noqa: SLF001
        structured={"RelevantDishes": RelevantDishes(relevant_dish_ids=["dish_999", ranked[0], ranked[0], "nope"])}
    )
    await ev.process(item)
    assert repo.live["live-1"].relevant_dish_ids == [ranked[0]]


async def test_when_nothing_is_relevant_precision_and_recall_are_undefined_not_zero():
    _, _, repo, ev, item, *_ = await _world(judge_ids=[])
    await ev.process(item)
    saved = repo.live["live-1"]
    assert saved.status == JobStatus.COMPLETED and saved.relevant_dish_ids == []
    assert all(v is None for v in saved.retrieval_metrics.values())
    assert summarize([saved])["precision_at_5"] is None  # excluded from means, never zero-filled
    assert summarize([saved])["non_llm_context_recall"] == 0.6


async def test_ragas_failure_marks_failed_but_keeps_the_retrieval_metrics():
    w, exp, repo, ev, item, *_ = await _world(ragas=FakeRagas(fail=True))
    ranked = await _ranking(w, exp)
    ev._judge = ScriptedLLM(structured={"RelevantDishes": RelevantDishes(relevant_dish_ids=ranked[:1])})  # noqa: SLF001
    await ev.process(item)
    saved = repo.live["live-1"]
    assert saved.status == JobStatus.FAILED and "ragas unavailable" in (saved.error or "")
    assert saved.retrieval_metrics["recall_at_10"] == 1.0 and saved.ragas_scores == {}


async def test_a_metric_that_ragas_cannot_define_is_stored_as_none_and_the_item_still_completes():
    class _NoPrecision(FakeRagas):
        async def evaluate(self, samples):
            return [{"non_llm_context_precision": None, "non_llm_context_recall": 0.5} for _ in samples]

    _, _, repo, ev, item, *_ = await _world(ragas=_NoPrecision(), judge_ids=[])
    await ev.process(item)
    saved = repo.live["live-1"]
    assert saved.status == JobStatus.COMPLETED and saved.ragas_scores["non_llm_context_precision"] is None


async def test_unconfigured_judge_fails_with_a_clear_message_instead_of_inventing_scores():
    _, _, repo, ev, item, *_ = await _world()
    ev._judge = ScriptedLLM(configured=False)  # noqa: SLF001
    await ev.process(item)
    saved = repo.live["live-1"]
    assert saved.status == JobStatus.FAILED and "judge LLM" in (saved.error or "")
    assert saved.retrieval_metrics == {} and saved.ragas_scores == {}


async def test_the_same_hard_filters_the_customer_had_are_replayed_for_the_depth_10_retrieval():
    w, exp, repo, ev, item, *_ = await _world(judge_ids=[])
    item.filters = {"exclude_allergens": ["dairy"], "available_only": True}
    await ev.process(item)
    flt = DishFilter.from_dict(item.filters)
    assert flt.exclude_allergens == ("dairy",)
    by_id = {d.id: d for d in w.dishes}
    retrieved = repo.live["live-1"].retrieved_dish_ids
    assert retrieved and all(flt.matches(by_id[i]) for i in retrieved)


def test_dish_filter_round_trips_through_to_dict():
    flt = DishFilter(max_price=200.0, exclude_allergens=("peanuts",), dietary=("vegan",), available_only=True)
    assert DishFilter.from_dict(flt.to_dict()) == flt
    assert DishFilter.from_dict({}) == DishFilter(available_only=False)


def test_menu_table_lists_every_dish_with_the_facts_the_judge_needs(dishes):
    table = menu_table(dishes).splitlines()
    assert len(table) == len(dishes)
    d = dishes[0]
    assert table[0].startswith(f"{d.id} | {d.name} | {d.category} |")
    assert f"| {d.calories} |" in table[0]


# ------------------------------------------------------------------------------ queuing from the chat


def _llm():
    return ScriptedLLM(structured={
        "ParsedTurn": ParsedTurn(intent="search", search_query="chicken burger",
                                 constraints=SearchConstraints(exclude_allergens=["peanuts"], categories=["Burgers"])),
        "AnswerOut": AnswerOut(message="Try the Crispy Chicken Burger.", recommended_dish_ids=[]),
    })


async def _chat_world(llm, live=True):
    repo = MemEvalRepo(None, [])  # type: ignore[arg-type]
    w = World(llm, live_evals=repo if live else None)
    chunking = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    exp, _ = await w.config.update(chunking, RetrievalConfig(top_k=5))
    await w.build_index(chunking)
    return w, exp, repo, (await w.chat.create_session()).id


async def test_an_answered_search_question_is_recorded_but_not_scored_until_asked():
    w, exp, repo, sid = await _chat_world(_llm())
    turn = await w.chat.handle_message(sid, "a chicken burger, I can't have peanuts")
    (item,) = repo.live.values()
    assert item.status == JobStatus.AWAITING and item.stage == "awaiting"  # nothing is scored on its own
    assert item.question == "a chicken burger, I can't have peanuts"  # verbatim, not the rewritten query
    assert item.query == "chicken burger"
    assert item.message_id == turn.assistant_message.id and item.session_id == sid
    assert item.answer == turn.assistant_message.content
    assert item.contexts == [c.content for c in turn.trace.chunks] and item.contexts
    assert item.filters["exclude_allergens"] == ["peanuts"]
    assert item.config_snapshot["index_fingerprint"] == exp.index_fingerprint
    assert item.config_snapshot["retrieval"]["top_k"] == 5


async def test_only_search_turns_are_queued():
    llm = ScriptedLLM(structured={"ParsedTurn": ParsedTurn(intent="view_cart")})
    w, _, repo, sid = await _chat_world(llm)
    await w.chat.handle_message(sid, "show my cart")
    assert repo.live == {}


async def test_nothing_is_queued_without_an_llm_or_when_the_feature_is_off():
    w, _, repo, sid = await _chat_world(ScriptedLLM(configured=False))
    await w.chat.handle_message(sid, "something spicy with chicken")
    assert repo.live == {}

    w2, _, repo2, sid2 = await _chat_world(_llm(), live=False)
    await w2.chat.handle_message(sid2, "a chicken burger, I can't have peanuts")
    assert repo2.live == {}


async def test_a_failure_while_queuing_never_breaks_the_customers_chat_turn():
    w, _, repo, sid = await _chat_world(_llm())

    async def boom(_item):
        raise RuntimeError("db down")

    repo.add_live = boom  # type: ignore[method-assign]
    turn = await w.chat.handle_message(sid, "a chicken burger, I can't have peanuts")
    assert turn.assistant_message.content


def test_summary_averages_only_metrics_that_exist():
    a = LiveEvaluation("a", "s", "m1", {}, "q", "q", {}, "x", [], status=JobStatus.COMPLETED,
                       retrieval_metrics={"precision_at_5": 0.4}, ragas_scores={"non_llm_context_recall": 1.0}, latency_ms=10)
    b = LiveEvaluation("b", "s", "m2", {}, "q", "q", {}, "x", [], status=JobStatus.COMPLETED,
                       retrieval_metrics={"precision_at_5": None}, ragas_scores={"non_llm_context_recall": 0.5}, latency_ms=30)
    pending = LiveEvaluation("c", "s", "m3", {}, "q", "q", {}, "x", [])
    out = summarize([a, b, pending])
    assert out["precision_at_5"] == 0.4 and out["non_llm_context_recall"] == 0.75 and out["avg_retrieval_latency_ms"] == 20
    assert out["non_llm_context_precision"] is None


async def test_reference_contexts_are_the_index_chunks_covering_the_dishes():
    w, exp, *_ = await _world()
    one = await w.retrieval.reference_contexts(exp.index_fingerprint, ["dish_028"])
    assert len(one) == 1 and "dish_028" in one[0]  # entity chunking: one chunk per dish
    assert await w.retrieval.reference_contexts(exp.index_fingerprint, []) == []
    assert await w.retrieval.reference_contexts("no-such-fingerprint", ["dish_028"]) == []


async def test_reference_contexts_cover_every_dish_a_multi_dish_chunk_mentions():
    w = World(ScriptedLLM(text="unused"))
    chunking = ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=900, chunk_overlap=0)
    exp, _ = await w.config.update(chunking, RetrievalConfig(top_k=3))
    await w.build_index(chunking)
    chunks = await w.retrieval.reference_contexts(exp.index_fingerprint, ["dish_001"])
    expected = [c.content for cs in w.store.chunks.values() for c, _ in cs if "dish_001" in c.dish_ids]
    assert chunks and chunks == list(dict.fromkeys(expected))  # exactly the chunks whose provenance includes the dish


def test_questions_are_grouped_by_the_configuration_they_were_answered_with():
    from app.restaurant.application.evaluation.live import config_key

    base = {"index_fingerprint": "ix1", "retrieval": {"strategy": "vector", "top_k": 5}, "llm_model": "m"}
    assert config_key(base) == config_key({**base, "llm_model": "other"})  # only index + retrieval matter
    assert config_key(base) != config_key({**base, "retrieval": {"strategy": "vector", "top_k": 8}})
    assert config_key(base) != config_key({**base, "index_fingerprint": "ix2"})


async def test_a_question_nothing_was_retrieved_for_is_still_evaluated():
    llm = ScriptedLLM(structured={
        "ParsedTurn": ParsedTurn(intent="search", search_query="pizza", constraints=SearchConstraints(max_price=1)),
    })
    w, _, repo, sid = await _chat_world(llm)
    turn = await w.chat.handle_message(sid, "a pizza under $1")
    assert turn.assistant_message.content == "Sorry, I cannot find this item on our menu."
    (item,) = repo.live.values()
    assert item.contexts == []


async def test_the_customers_text_is_removed_from_the_server_once_scored():
    w, exp, repo, ev, item, ragas, _ = await _world()
    await ev.process(item)
    saved = repo.live["live-1"]
    assert saved.status == JobStatus.COMPLETED and saved.ragas_scores  # the scores stay
    assert saved.question == saved.query == saved.answer == "" and saved.contexts == []  # the conversation does not
    assert [i.id for i in await repo.list_live_by_message_ids(["m1"])] == ["live-1"]


async def test_pressing_evaluate_releases_only_the_waiting_questions_to_the_worker():
    w, _, repo, sid = await _chat_world(_llm())
    turn = await w.chat.handle_message(sid, "a chicken burger, I can't have peanuts")
    mid = turn.assistant_message.id
    assert await repo.claim_pending_live() is None  # the worker ignores waiting questions
    assert await repo.release_awaiting([mid]) == 1
    assert await repo.release_awaiting([mid]) == 0  # already released
    claimed = await repo.claim_pending_live()
    assert claimed is not None and claimed.message_id == mid


async def test_questions_nobody_evaluates_are_deleted_after_a_day():
    from datetime import UTC, datetime, timedelta

    w, _, repo, sid = await _chat_world(_llm())
    await w.chat.handle_message(sid, "a chicken burger, I can't have peanuts")
    (old,) = repo.live.values()
    old.created_at = datetime.now(UTC) - timedelta(hours=25)
    await w.chat.handle_message(sid, "a chicken burger, I can't have peanuts")  # recording purges the stale one
    assert old.id not in repo.live and len(repo.live) == 1
