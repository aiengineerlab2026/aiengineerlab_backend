"""Chat orchestration: intent -> (RAG search | validated cart operations | order confirmation)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from app.errors import (
    DomainError,
    ExternalServiceError,
    LLMQuotaExceeded,
    NotFoundError,
    ValidationFailed,
)
from app.restaurant.application import prompts
from app.restaurant.application.chat.parser import ParsedTurn, TurnParser, to_dish_filter
from app.restaurant.application.chat.resolver import Resolution, resolve_reference
from app.restaurant.application.ordering.services import CartService, OrderService
from app.restaurant.application.rag.config import ConfigService
from app.restaurant.application.rag.retrieval import RetrievalService
from app.restaurant.domain.chat.entities import ChatMessage, ChatSession
from app.restaurant.domain.chat.repositories import ChatRepository
from app.restaurant.domain.entities import Dish
from app.restaurant.domain.evaluation.entities import LiveEvaluation
from app.restaurant.domain.evaluation.repositories import EvaluationRepository
from app.restaurant.domain.filters import DishFilter, apply_filter
from app.restaurant.domain.ordering.entities import Cart, Order
from app.restaurant.domain.rag.entities import JobStatus, RetrievalTrace
from app.restaurant.domain.rag.ports import LLMClient
from app.restaurant.domain.rag.ranking import rank_dishes
from app.restaurant.domain.rag.value_objects import EmbeddingSpec
from app.restaurant.domain.repositories import MenuRepository

log = logging.getLogger(__name__)
MAX_RECOMMENDATIONS = 4
AWAITING_TTL = timedelta(hours=24)
NOT_FOUND_REPLY = "Sorry, I cannot find this item on our menu."


class AnswerOut(BaseModel):
    message: str
    recommended_dish_ids: list[str] = Field(default_factory=list)


@dataclass
class TurnResult:
    user_message: ChatMessage
    assistant_message: ChatMessage
    cart: Cart
    recommended_dish_ids: list[str] = field(default_factory=list)
    trace: RetrievalTrace | None = None
    order: Order | None = None


def _money(v: Decimal | float) -> str:
    return f"${float(v):.2f}"


class ChatService:
    def __init__(
        self, chats: ChatRepository, menu: MenuRepository, carts: CartService, orders: OrderService,
        config: ConfigService, retrieval: RetrievalService, llm: LLMClient, parser: TurnParser,
        live_evals: EvaluationRepository | None = None,
    ):
        self._chats, self._menu, self._carts, self._orders = chats, menu, carts, orders
        self._config, self._retrieval, self._llm, self._parser = config, retrieval, llm, parser
        self._live_evals = live_evals  # when set, every search question is recorded, ready to be evaluated on demand

    # ------------------------------------------------------------------ sessions

    async def create_session(self) -> ChatSession:
        """Sessions only hold the cart/confirmation state. Messages live in the browser, never here."""
        return await self._chats.create_session()

    async def get_session(self, session_id: str) -> ChatSession:
        s = await self._chats.get_session(session_id)
        if s is None:
            raise NotFoundError(f"Session '{session_id}' not found.")
        return s

    # ------------------------------------------------------------------ main entry

    async def handle_message(
        self, session_id: str, text: str, history: list[ChatMessage] | None = None,
        embedding: EmbeddingSpec | None = None,
    ) -> TurnResult:
        """``history`` is the recent conversation as sent by the client; nothing is stored server-side.

        ``embedding`` is the embedding model this request searches with (its pre-built index); default when None."""
        text = text.strip()
        if not text:
            raise ValidationFailed("Message must not be empty.")
        session = await self.get_session(session_id)
        history = (history or [])[-8:]
        user_msg = ChatMessage(session_id=session_id, role="user", content=text)

        pending = session.pending_order_confirmation
        names = [d.name for d in await self._menu.list_dishes()]
        try:
            parsed, parser_name = await self._parser.parse(text, history, pending, names)
        except LLMQuotaExceeded as exc:
            return await self._quota_turn(session_id, user_msg, exc)
        meta: dict[str, Any] = {"intent": parsed.intent, "parser": parser_name}
        trace: RetrievalTrace | None = None
        order: Order | None = None
        recommended: list[str] = []
        try:
            if pending and parsed.intent in ("affirm", "confirm_order"):
                reply, order = await self._place_order(session)
                meta["order_id"] = order.id
            elif pending and parsed.intent == "decline":
                self._set_pending(session, False)
                reply = "No problem - I haven't placed anything. Your cart is unchanged."
            else:
                if pending:  # any other message cancels a half-finished confirmation
                    self._set_pending(session, False)
                reply, recommended, trace, extra = await self._dispatch(session, parsed, text, embedding, history)
                meta.update(extra)
        except LLMQuotaExceeded as exc:
            return await self._quota_turn(session_id, user_msg, exc)
        except DomainError as exc:
            reply = f"Sorry, I couldn't do that: {exc.message}"
            meta["error"] = {"code": exc.code, "message": exc.message}
        if recommended:
            session.state["last_dish_ids"] = recommended
        await self._chats.save_session(session)

        meta["recommended_dish_ids"] = recommended
        if trace is not None:
            meta["trace"] = trace.model_dump(mode="json")
        assistant = ChatMessage(session_id=session_id, role="assistant", content=reply, metadata=meta)
        await self._record_for_evaluation(session_id, text, assistant, trace, meta)
        cart = await self._carts.get_cart(session_id)
        return TurnResult(user_msg, assistant, cart, recommended, trace, order)

    async def _quota_turn(self, session_id: str, user_msg: ChatMessage, exc: LLMQuotaExceeded) -> TurnResult:
        """The provider's free quota is used up: say so plainly. Nothing changes in the cart; not evaluated."""
        meta = {"error": {"code": exc.code, "message": exc.message}, "recommended_dish_ids": []}
        assistant = ChatMessage(session_id=session_id, role="assistant", content=exc.message, metadata=meta)
        return TurnResult(user_msg, assistant, await self._carts.get_cart(session_id))

    async def _record_for_evaluation(
        self, session_id: str, question: str, assistant: ChatMessage, trace: RetrievalTrace | None,
        meta: dict[str, Any],
    ) -> None:
        """Remember every search question - including the ones where nothing was retrieved (they must score zero
        recall, not vanish) - so that the customer can have them scored later with the Evaluate button. Nothing is
        scored here, and a question nobody evaluates is deleted after a day. Never allowed to break the chat turn."""
        if self._live_evals is None or trace is None or meta.get("error") or not self._llm.is_configured:
            return
        try:
            await self._live_evals.purge_awaiting(datetime.now(UTC) - AWAITING_TTL)
            cfg = await self._config.get()
            await self._live_evals.add_live(
                LiveEvaluation(
                    id=str(uuid.uuid4()), session_id=session_id, message_id=assistant.id,
                    # What the answer was really produced with (the trace), not whatever is configured by now.
                    config_snapshot={
                        "chunking": trace.chunking.model_dump(mode="json"),
                        "retrieval": trace.retrieval.model_dump(mode="json"),
                        "embedding": meta.get("embedding") or cfg.embedding.model_dump(mode="json"),
                        "index_fingerprint": trace.index_fingerprint,
                        "llm_model": cfg.llm_model,
                    },
                    question=question, query=meta.get("query") or question, filters=meta.get("filters") or {},
                    answer=assistant.content, contexts=[c.content for c in trace.chunks],
                    latency_ms=trace.latency_ms, status=JobStatus.AWAITING, stage="awaiting",
                )
            )
        except Exception:  # noqa: BLE001
            log.warning("could not record the question for evaluation", exc_info=True)

    # ------------------------------------------------------------------ dispatch

    def _set_pending(self, session: ChatSession, value: bool) -> None:
        session.state["pending_order_confirmation"] = value

    async def _dispatch(
        self, session: ChatSession, parsed: ParsedTurn, text: str, embedding: EmbeddingSpec | None = None,
        history: list[ChatMessage] | None = None,
    ) -> tuple[str, list[str], RetrievalTrace | None, dict[str, Any]]:
        intent = parsed.intent
        if intent == "search":
            return await self._search(session, parsed, text, embedding, history)
        if intent == "cart_update":
            reply, recommended, actions = await self._cart_update(session, parsed)
            return reply, recommended, None, {"actions": actions}
        if intent == "view_cart":
            return await self._describe_cart(session.id), [], None, {}
        if intent == "confirm_order":
            return await self._request_confirmation(session), [], None, {}
        if intent == "dish_details":
            reply, ids = await self._dish_details(session, parsed.dish_ref or text)
            return reply, ids, None, {}
        if intent == "greeting":
            return "Hello! What are you in the mood for today?", [], None, {}
        if intent in ("affirm", "decline"):
            return "Okay! Tell me what you'd like to eat or add to your cart.", [], None, {}
        return "I can help with the menu, your cart and placing a demo order. What would you like to eat?", [], None, {}

    # ------------------------------------------------------------------ search (RAG)

    async def _search(
        self, session: ChatSession, parsed: ParsedTurn, text: str, embedding: EmbeddingSpec | None = None,
        history: list[ChatMessage] | None = None,
    ) -> tuple[str, list[str], RetrievalTrace, dict[str, Any]]:
        live = await self._config.serving(embedding)  # the configuration as it is right now, for this embedding model
        categories = [c.name for c in await self._menu.list_categories()]
        flt = to_dish_filter(parsed.constraints, categories)
        query = parsed.search_query or text
        trace = await self._retrieval.retrieve(
            query, chunking=live.chunking, retrieval=live.retrieval, index_fingerprint=live.index_fingerprint,
            dish_filter=flt, notice=live.warning,
        )
        ranked_ids = rank_dishes([c.dish_ids for c in trace.chunks])
        retrieved = await self._menu.get_dishes(ranked_ids)
        # Deterministic, dish-level enforcement - independent of what the chunk text or the LLM says.
        outcome = apply_filter(retrieved, flt)
        eligible = outcome.allowed
        extra = {
            "filters": flt.to_dict(),
            "retrieved_dish_ids": ranked_ids,
            "rejected": outcome.rejected,
            "query": query,
            "embedding": live.embedding.model_dump(mode="json") if live.embedding else None,
        }
        if not trace.chunks:  # retrieval came back empty: say so plainly, never improvise
            return NOT_FOUND_REPLY, [], trace, {**extra, "llm_used": False}
        if not eligible:
            return self._no_match_reply(flt, outcome.rejected), [], trace, {**extra, "llm_used": False}

        reply: str | None = None
        ids: list[str] = []
        llm_used = False
        if self._llm.is_configured:
            try:
                reply, ids = await self._llm_answer(text, flt, eligible, trace, history or [])
                llm_used = reply is not None
            except LLMQuotaExceeded:
                raise
            except ExternalServiceError as exc:
                log.warning("LLM answer failed; using deterministic reply", extra={"error": exc.message[:200]})
        if reply is None:
            ids = [d.id for d in eligible[:3]]
            reply = self._template_reply(eligible[:3], retrieval_note=not self._llm.is_configured)
        reply += self._allergen_note(flt)
        return reply, ids, trace, {**extra, "llm_used": llm_used}

    async def _llm_answer(
        self, question: str, flt: DishFilter, eligible: list[Dish], trace: RetrievalTrace,
        history: list[ChatMessage] | None = None,
    ) -> tuple[str | None, list[str]]:
        eligible_ids = [d.id for d in eligible]
        conversation = "\n".join(f"{m.role}: {m.content[:300]}" for m in (history or [])[-4:])
        user = prompts.ANSWER_USER.format(
            conversation=conversation or "(none)",
            question=question,
            filters=flt.to_dict() or "none",
            eligible=", ".join(eligible_ids),
            context=trace.context or "(no context retrieved)",
        )
        out = await self._llm.generate_structured(prompts.ANSWER.system, user, AnswerOut)
        valid = [i for i in dict.fromkeys(out.recommended_dish_ids) if i in set(eligible_ids)][:MAX_RECOMMENDATIONS]
        # Safety net: the text must not name a dish that failed the hard filters.
        lowered = out.message.lower()
        everyone = await self._menu.list_dishes()
        banned = [d for d in everyone if d.id not in set(eligible_ids) and d.name.lower() in lowered]
        if banned or not out.message.strip():
            log.warning("LLM answer rejected by validation", extra={"mentioned_ineligible": [d.id for d in banned]})
            return None, []
        return out.message.strip(), valid or eligible_ids[:3]

    @staticmethod
    def _template_reply(dishes: list[Dish], *, retrieval_note: bool) -> str:
        lines = [
            f"• {d.name} - {_money(d.price)}, {d.calories} kcal, {d.protein_grams} g protein"
            for d in dishes
        ]
        head = "Here are the best matches from our menu:" 
        tail = "\n(LLM not configured - this reply was composed deterministically from retrieved menu data.)" if retrieval_note else ""
        return head + "\n" + "\n".join(lines) + tail

    @staticmethod
    def _allergen_note(flt: DishFilter) -> str:
        if not flt.exclude_allergens:
            return ""
        names = ", ".join(a.replace("_", " ") for a in flt.exclude_allergens)
        return (
            f"\n\nAllergen check: none of the dishes above list {names} in our menu data. "
            "Please confirm with the staff if your allergy is severe."
        )

    @staticmethod
    def _no_match_reply(flt: DishFilter, rejected: dict[str, list[str]]) -> str:
        applied = flt.to_dict()
        applied.pop("available_only", None)
        base = "I couldn't find a dish on our menu that matches all of your requirements."
        if applied:
            base += " (Filters applied: " + ", ".join(f"{k}={v}" for k, v in applied.items()) + ".)"
        if rejected:
            base += f" {len(rejected)} retrieved dish(es) were excluded by your constraints."
        return base + " Would you like to relax one of the constraints?"

    # ------------------------------------------------------------------ cart

    async def _cart_update(self, session: ChatSession, parsed: ParsedTurn) -> tuple[str, list[str], list[dict]]:
        cart = await self._carts.get_cart(session.id)
        everyone = await self._menu.list_dishes()
        last = session.last_dish_ids
        messages: list[str] = []
        actions: list[dict] = []
        options: list[str] = []
        if not parsed.cart_actions:
            return "Tell me which dish to add or remove, for example \"add two grilled chicken bowls\".", [], []
        for act in parsed.cart_actions:
            in_cart = [d for d in everyone if cart.find_by_dish(d.id)]
            pool = in_cart if act.op in ("remove", "set_quantity") else [d for d in everyone]
            res: Resolution = resolve_reference(act.item, pool, last, prefer_ids=[d.id for d in in_cart])
            record = {"op": act.op, "item": act.item}
            if res.status == "not_found":
                hint = f" Did you mean {', '.join(d.name for d in res.candidates)}?" if res.candidates else ""
                where = "in your cart" if act.op != "add" else "on the menu"
                messages.append(f"I couldn't find \"{act.item}\" {where}.{hint}")
                actions.append({**record, "status": "not_found"})
                continue
            if res.status == "ambiguous":
                names = ", ".join(d.name for d in res.candidates[:6])
                messages.append(f"\"{act.item}\" matches several dishes ({names}). Which one do you mean?")
                options += [d.id for d in res.candidates[:6]]
                actions.append({**record, "status": "ambiguous", "candidates": [d.id for d in res.candidates]})
                continue
            dish = res.dish
            assert dish is not None
            try:
                qty = act.quantity or 1
                if act.op == "add":
                    cart, item = await self._carts.add_item(session.id, dish.id, qty)
                    messages.append(f"Added {qty} × {dish.name} ({_money(dish.price * qty)}).")
                elif act.op == "remove":
                    line = cart.find_by_dish(dish.id)
                    assert line is not None
                    cart = await self._carts.remove_item(session.id, line.id)
                    messages.append(f"Removed {dish.name} from your cart.")
                else:
                    line = cart.find_by_dish(dish.id)
                    assert line is not None
                    cart = await self._carts.update_quantity(session.id, line.id, qty)
                    messages.append(f"Updated {dish.name} to {qty}.")
                actions.append({**record, "status": "ok", "dish_id": dish.id, "quantity": qty})
            except DomainError as exc:
                messages.append(f"Couldn't {act.op.replace('_', ' ')} {dish.name}: {exc.message}")
                actions.append({**record, "status": "rejected", "dish_id": dish.id, "reason": exc.message})
        cart = await self._carts.get_cart(session.id)
        messages.append(self._cart_line(cart))
        return " ".join(messages), list(dict.fromkeys(options)), actions

    @staticmethod
    def _cart_line(cart: Cart) -> str:
        if not cart.items:
            return "Your cart is empty."
        return f"Cart: {cart.item_count} item(s), total {_money(cart.total)}."

    async def _describe_cart(self, session_id: str) -> str:
        cart = await self._carts.get_cart(session_id)
        if not cart.items:
            return "Your cart is empty. Ask me for a recommendation to get started!"
        lines = [f"• {i.quantity} × {i.name} - {_money(i.line_total)}" for i in cart.items]
        return "Here's your cart:\n" + "\n".join(lines) + f"\nTotal: {_money(cart.total)}."

    # ------------------------------------------------------------------ orders

    async def _request_confirmation(self, session: ChatSession) -> str:
        cart = await self._carts.get_cart(session.id)
        if not cart.items:
            return "Your cart is empty, so there's nothing to order yet."
        self._set_pending(session, True)
        lines = [f"• {i.quantity} × {i.name} - {_money(i.line_total)}" for i in cart.items]
        return (
            "Please review your demo order:\n" + "\n".join(lines) + f"\nTotal: {_money(cart.total)}.\n"
            "Reply \"yes\" to place the order, or \"no\" to cancel. (No real payment is taken.)"
        )

    async def _place_order(self, session: ChatSession) -> tuple[str, Order]:
        self._set_pending(session, False)
        order = await self._orders.place_order(session.id, confirmed=True)
        return (
            f"Order confirmed! 🎉 Order #{order.id[:8].upper()} - {len(order.items)} dish(es), total {_money(order.total)}. "
            "This is a demo order; no payment was taken.",
            order,
        )

    # ------------------------------------------------------------------ dish details (no LLM needed)

    async def _dish_details(self, session: ChatSession, ref: str) -> tuple[str, list[str]]:
        everyone = await self._menu.list_dishes()
        res = resolve_reference(ref, everyone, session.last_dish_ids)
        if res.status == "resolved" and res.dish:
            d = res.dish
            allergens = ", ".join(a.replace("_", " ") for a in d.allergens) or "none listed"
            return (
                f"{d.name} ({d.category}) - {_money(d.price)}. {d.description}\n"
                f"Ingredients: {', '.join(d.ingredients)}.\nAllergens: {allergens}.\n"
                f"Nutrition: {d.calories} kcal, {d.protein_grams} g protein."
                + ("" if d.available else "\nNote: currently unavailable."),
                [d.id],
            )
        if res.status == "ambiguous":
            return f"Several dishes match \"{ref}\": {', '.join(x.name for x in res.candidates[:6])}. Which one?", [x.id for x in res.candidates[:6]]
        hint = f" Did you mean {', '.join(x.name for x in res.candidates)}?" if res.candidates else ""
        return f"I couldn't find \"{ref}\" on the menu.{hint}", []
