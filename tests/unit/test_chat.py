from __future__ import annotations

import pytest
from conftest import ScriptedLLM, World  # type: ignore[import-not-found]

from app.application.chat.parser import ParsedTurn, RuleTurnParser, SearchConstraints, extract_constraints
from app.application.chat.resolver import resolve_dish, resolve_reference
from app.application.chat.service import AnswerOut
from app.domain.rag.value_objects import ChunkingConfig, ChunkingStrategy, RetrievalConfig

# ------------------------------------------------------------------------------ rule parser


async def _parse(text: str, pending: bool = False) -> ParsedTurn:
    return await RuleTurnParser().parse(text, [], pending)


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("add two classic beef burgers to my cart", "cart_update"),
        ("remove the margherita pizza", "cart_update"),
        ("change the burger quantity to 3", "cart_update"),
        ("show my cart", "view_cart"),
        ("place my order", "confirm_order"),
        ("checkout", "confirm_order"),
        ("yes", "affirm"),
        ("no", "decline"),
        ("tell me about the falafel burger", "dish_details"),
        ("hello", "greeting"),
        ("something high in protein under $13", "search"),
    ],
)
async def test_rule_parser_intents(text, intent):
    assert (await _parse(text)).intent == intent


async def test_rule_parser_extracts_cart_actions():
    t = await _parse("add 2 classic beef burgers and a margherita pizza to my cart")
    assert [(a.op, a.item, a.quantity) for a in t.cart_actions] == [
        ("add", "classic beef burgers", 2), ("add", "margherita pizza", 1),
    ]
    t = await _parse("replace the margherita pizza with a pepperoni pizza")
    assert [(a.op, a.item) for a in t.cart_actions] == [("remove", "margherita pizza"), ("add", "pepperoni pizza")]
    t = await _parse("set the falafel burger to three")
    assert (t.cart_actions[0].op, t.cart_actions[0].quantity) == ("set_quantity", 3)


async def test_rule_parser_keeps_menu_names_containing_and_whole(dishes):
    """Regression: "Shrimp and Avocado Power Bowl" used to be split into "Shrimp" + "Avocado Power Bowl"."""
    names = [d.name for d in dishes]
    t = await RuleTurnParser().parse("add the shrimp and avocado power bowl and a tiramisu", [], False, names)
    assert [(a.item.lower(), a.quantity) for a in t.cart_actions] == [("shrimp and avocado power bowl", 1), ("tiramisu", 1)]
    # without menu knowledge the old (naive) behaviour is unchanged
    naive = await RuleTurnParser().parse("add the shrimp and avocado power bowl", [], False)
    assert len(naive.cart_actions) == 2


def test_constraint_extraction():
    c = extract_constraints("high protein vegan pasta under $17, I'm allergic to peanuts and dairy, under 700 calories")
    assert c.max_price == 17 and c.max_calories == 700 and c.min_protein == 30
    assert set(c.exclude_allergens) == {"peanuts", "dairy"}
    assert c.dietary == ["vegan"] and c.categories == ["Pasta"]
    assert extract_constraints("gluten-free dessert").exclude_allergens == ["gluten"]
    assert extract_constraints("I have a tree nut allergy, avoid nuts").exclude_allergens == ["peanuts", "tree_nuts"]
    assert extract_constraints("surprise me") == SearchConstraints()
    assert extract_constraints("a burger for $12.50").max_price == 12.5
    assert extract_constraints("something for 8 dollars").max_price == 8
    assert extract_constraints("a table for 2 please").max_price is None


def test_dish_resolution(dishes):
    assert resolve_dish("classic beef burgers", dishes).dish.name == "Classic Beef Burger"
    amb = resolve_dish("burger", dishes)
    assert amb.status == "ambiguous" and len(amb.candidates) > 1
    assert resolve_dish("burger", dishes, prefer_ids=[amb.candidates[2].id]).dish == amb.candidates[2]
    assert resolve_dish("clasic beef burgr", dishes).status == "not_found"
    assert resolve_dish("sushi", dishes).status == "not_found"
    shown = [dishes[3].id, dishes[1].id]
    assert resolve_reference("it", dishes, shown).dish == dishes[3]
    assert resolve_reference("the second one", dishes, shown).dish == dishes[1]
    assert resolve_reference("it", dishes, []).status == "not_found"


# ------------------------------------------------------------------------------ chat service


async def _active(llm=None) -> tuple[World, str]:
    w = World(llm)
    chunking = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    await w.config.update(chunking, RetrievalConfig(top_k=5))
    await w.build_index(chunking)
    return w, (await w.chat.create_session()).id


async def test_search_enforces_hard_filters_deterministically():
    w, sid = await _active()
    turn = await w.chat.handle_message(sid, "I'm allergic to peanuts and dairy, show me something under $17")
    by_id = {d.id: d for d in w.dishes}
    assert turn.recommended_dish_ids
    for i in turn.recommended_dish_ids:
        d = by_id[i]
        assert not ({"peanuts", "dairy"} & set(d.allergens)) and float(d.price) <= 17 and d.available
    assert turn.trace is not None and turn.assistant_message.metadata["parser"] == "rules"
    assert "Allergen check" in turn.assistant_message.content
    assert set(turn.assistant_message.metadata["filters"]["exclude_allergens"]) == {"dairy", "peanuts"}


async def test_conversation_comes_from_the_client_and_is_not_stored():
    from app.domain.chat.entities import ChatMessage

    w, sid = await _active()
    history = [ChatMessage(session_id=sid, role="user", content="hi"), ChatMessage(session_id=sid, role="assistant", content="hello")]
    turn = await w.chat.handle_message(sid, "show my cart", history)
    assert turn.user_message.content == "show my cart" and turn.assistant_message.role == "assistant"
    assert not hasattr(w.chats, "messages") and not hasattr(w.chat, "list_messages")


async def test_impossible_request_says_no_match_instead_of_inventing_a_dish():
    w, sid = await _active()
    turn = await w.chat.handle_message(sid, "a pizza under $1")
    assert turn.recommended_dish_ids == []
    assert turn.assistant_message.content == "Sorry, I cannot find this item on our menu."


async def test_cart_operations_go_through_validated_services():
    w, sid = await _active()
    turn = await w.chat.handle_message(sid, "add two classic beef burgers to my cart")
    assert [(i.name, i.quantity) for i in turn.cart.items] == [("Classic Beef Burger", 2)]
    turn = await w.chat.handle_message(sid, "change the classic beef burger to 4")
    assert turn.cart.items[0].quantity == 4
    turn = await w.chat.handle_message(sid, "remove the classic beef burger")
    assert turn.cart.items == []
    turn = await w.chat.handle_message(sid, "add 99 classic beef burgers")  # over the quantity limit
    assert turn.cart.items == [] and turn.assistant_message.metadata["actions"][0]["status"] == "rejected"


async def test_ambiguous_and_unknown_dishes_do_not_touch_the_cart():
    w, sid = await _active()
    turn = await w.chat.handle_message(sid, "add a burger")
    assert turn.cart.items == [] and "Which one" in turn.assistant_message.content
    assert turn.recommended_dish_ids  # offered as options
    turn = await w.chat.handle_message(sid, "add a sushi platter")
    assert turn.cart.items == [] and "couldn't find" in turn.assistant_message.content


async def test_unavailable_dish_is_refused():
    w, sid = await _active()
    gone = next(d for d in w.dishes if not d.available)
    turn = await w.chat.handle_message(sid, f"add {gone.name}")
    assert turn.cart.items == []
    assert turn.assistant_message.metadata["actions"][0]["status"] == "rejected"


async def test_order_needs_explicit_two_step_confirmation():
    w, sid = await _active()
    await w.chat.handle_message(sid, "add a margherita pizza")
    turn = await w.chat.handle_message(sid, "place my order")
    assert turn.order is None and w.orders_repo.orders == {}
    assert (await w.chats.get_session(sid)).pending_order_confirmation
    assert "Reply \"yes\"" in turn.assistant_message.content

    turn = await w.chat.handle_message(sid, "yes")
    assert turn.order is not None and len(w.orders_repo.orders) == 1
    assert turn.cart.items == [] and not (await w.chats.get_session(sid)).pending_order_confirmation


async def test_declining_or_changing_topic_cancels_the_pending_order():
    w, sid = await _active()
    await w.chat.handle_message(sid, "add a margherita pizza")
    await w.chat.handle_message(sid, "place my order")
    turn = await w.chat.handle_message(sid, "no")
    assert turn.order is None and turn.cart.item_count == 1

    await w.chat.handle_message(sid, "place my order")
    await w.chat.handle_message(sid, "show my cart")  # unrelated message -> confirmation is dropped
    turn = await w.chat.handle_message(sid, "yes")
    assert turn.order is None and w.orders_repo.orders == {} and turn.cart.item_count == 1


async def test_yes_without_a_pending_confirmation_never_places_an_order():
    w, sid = await _active()
    await w.chat.handle_message(sid, "add a margherita pizza")
    turn = await w.chat.handle_message(sid, "yes")
    assert turn.order is None and w.orders_repo.orders == {}


async def test_empty_cart_cannot_be_ordered():
    w, sid = await _active()
    turn = await w.chat.handle_message(sid, "place my order")
    assert "empty" in turn.assistant_message.content
    assert not (await w.chats.get_session(sid)).pending_order_confirmation


async def test_follow_up_references_use_the_last_shown_dishes():
    w, sid = await _active()
    shown = await w.chat.handle_message(sid, "tell me about the falafel burger")
    assert shown.recommended_dish_ids
    turn = await w.chat.handle_message(sid, "add it")
    assert [i.dish_id for i in turn.cart.items] == shown.recommended_dish_ids[:1]


async def test_blank_and_unknown_session_are_rejected():
    from app.domain.errors import NotFoundError, ValidationFailed

    w, sid = await _active()
    with pytest.raises(ValidationFailed):
        await w.chat.handle_message(sid, "   ")
    with pytest.raises(NotFoundError):
        await w.chat.handle_message("missing", "hi")


# ------------------------------------------------------------------------------ LLM output is never trusted


def _llm_world_script(answer: str, ids: list[str]) -> ScriptedLLM:
    """LLM that understands "burger without peanuts" and then (mis)behaves in its answer."""
    return ScriptedLLM(structured={
        "ParsedTurn": ParsedTurn(
            intent="search", search_query="chicken burger",
            constraints=SearchConstraints(exclude_allergens=["peanuts"], categories=["Burgers"]),
        ),
        "AnswerOut": AnswerOut(message=answer, recommended_dish_ids=ids),
    })


async def test_llm_answer_naming_a_filtered_out_dish_is_replaced_by_a_safe_reply():
    w, _ = await _active()
    peanut = next(d for d in w.dishes if d.name == "Peanut Satay Chicken Burger")
    w2, sid = await _active(_llm_world_script(f"You'll love the {peanut.name}!", [peanut.id]))
    turn = await w2.chat.handle_message(sid, "a chicken burger, I can't have peanuts")
    by_id = {d.id: d for d in w2.dishes}
    meta = turn.assistant_message.metadata
    assert meta["parser"] == "llm" and meta["llm_used"] is False  # validation rejected the LLM text
    assert peanut.name not in turn.assistant_message.content
    assert turn.recommended_dish_ids, "falls back to the deterministic reply instead of returning nothing"
    assert all("peanuts" not in by_id[i].allergens and by_id[i].category == "Burgers" for i in turn.recommended_dish_ids)


async def test_llm_recommendations_outside_the_eligible_set_are_dropped():
    """The LLM may only recommend dishes from the deterministic eligible set (burgers without peanuts)."""
    w, _ = await _active()
    peanut = next(d for d in w.dishes if d.name == "Peanut Satay Chicken Burger")
    ok = next(d for d in w.dishes if d.name == "Crispy Chicken Burger")
    pizza = next(d for d in w.dishes if d.category == "Pizza")
    w2, sid = await _active(_llm_world_script("Here you go.", ["dish_999", peanut.id, pizza.id, ok.id]))
    turn = await w2.chat.handle_message(sid, "a chicken burger, I can't have peanuts")
    assert turn.assistant_message.metadata["llm_used"] is True
    assert turn.recommended_dish_ids == [ok.id]  # unknown, allergen-violating and off-category ids were dropped


async def test_when_the_free_llm_quota_is_used_up_the_customer_is_told_to_try_later():
    from app.domain.errors import LLMQuotaExceeded
    from tests.conftest import ScriptedLLM

    class OutOfQuota(ScriptedLLM):
        async def generate_structured(self, system, user, schema):
            raise LLMQuotaExceeded()

    w, sid = await _active(OutOfQuota())
    turn = await w.chat.handle_message(sid, "something spicy with chicken")
    assert "free version of Gemini" in turn.assistant_message.content and "try again" in turn.assistant_message.content
    assert turn.assistant_message.metadata["error"]["code"] == "llm_quota_exceeded"
    assert turn.recommended_dish_ids == [] and turn.trace is None


def test_provider_quota_errors_are_recognised():
    from app.infrastructure.llm.quota import is_quota_error as _quota_exhausted

    assert _quota_exhausted(RuntimeError("429 RESOURCE_EXHAUSTED: You exceeded your current quota"))
    assert not _quota_exhausted(RuntimeError("500 internal error"))


# ------------------------------------------------------------------------------ remembering earlier questions


async def test_rule_parser_remembers_the_previous_question_for_follow_ups():
    from app.domain.chat.entities import ChatMessage  # noqa: PLC0415

    history = [
        ChatMessage(session_id="s", role="user", content="show me pasta dishes"),
        ChatMessage(session_id="s", role="assistant", content="Here are some pasta dishes."),
    ]
    follow = await RuleTurnParser().parse("which of those are spicy?", history, False)
    assert follow.intent == "search"
    assert "pasta" in follow.search_query and "spicy" in follow.search_query
    assert "Pasta" in follow.constraints.categories and follow.constraints.spicy_only

    fresh = await RuleTurnParser().parse("I want a vegan dessert under $10 please", history, False)
    assert "pasta" not in fresh.search_query
