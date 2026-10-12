"""Turn parsing: customer message -> structured intent + constraints + cart actions.

The parser only *interprets* the message. It never touches the cart or the database; all effects go
through validated application services. ``LLMTurnParser`` uses LangChain structured output (tool
calling); ``RuleTurnParser`` is a deterministic fallback used when no LLM is configured/available.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.errors import ExternalServiceError, LLMQuotaExceeded
from app.restaurant.application import prompts
from app.restaurant.domain.chat.entities import ChatMessage
from app.restaurant.domain.filters import ALLERGEN_VOCABULARY, DishFilter, normalize_allergens
from app.restaurant.domain.rag.ports import LLMClient

log = logging.getLogger(__name__)

Intent = Literal[
    "search", "cart_update", "view_cart", "confirm_order", "affirm", "decline", "dish_details", "greeting", "other"
]


class SearchConstraints(BaseModel):
    max_price: float | None = None
    min_price: float | None = None
    max_calories: int | None = None
    min_calories: int | None = None
    min_protein: int | None = None
    exclude_allergens: list[str] = Field(default_factory=list)
    exclude_ingredients: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    dietary: list[str] = Field(default_factory=list, description="vegetarian | vegan | gluten_free")
    spicy_only: bool = False


class CartAction(BaseModel):
    op: Literal["add", "remove", "set_quantity"]
    item: str = Field(description="Dish reference exactly as the customer said it")
    quantity: int | None = None


class ParsedTurn(BaseModel):
    intent: Intent = "other"
    search_query: str = ""
    dish_ref: str = ""
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    cart_actions: list[CartAction] = Field(default_factory=list)


def to_dish_filter(c: SearchConstraints, category_names: list[str], *, available_only: bool = True) -> DishFilter:
    """Sanitise LLM/rule output into a trusted DishFilter (unknown values are dropped, never guessed)."""
    cats = {n.lower(): n for n in category_names}
    allergens = tuple(a for a in normalize_allergens(c.exclude_allergens) if a in ALLERGEN_VOCABULARY)
    return DishFilter(
        max_price=c.max_price if c.max_price and c.max_price > 0 else None,
        min_price=c.min_price if c.min_price and c.min_price > 0 else None,
        max_calories=c.max_calories if c.max_calories and c.max_calories > 0 else None,
        min_calories=c.min_calories if c.min_calories and c.min_calories > 0 else None,
        min_protein=c.min_protein if c.min_protein and c.min_protein > 0 else None,
        exclude_allergens=allergens,
        exclude_ingredients=tuple(i.strip().lower() for i in c.exclude_ingredients if i.strip()),
        categories=tuple(cats[x.lower()] for x in c.categories if x.lower() in cats),
        dietary=tuple(d for d in c.dietary if d in ("vegetarian", "vegan", "gluten_free")),
        spicy_only=c.spicy_only,
        available_only=available_only,
    )


# ---------------------------------------------------------------------------------------------
# Rule-based parser
# ---------------------------------------------------------------------------------------------

_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "couple": 2}
_QTY = r"(\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
_TRAIL = re.compile(r"\b(to|in|into|from|on)\s+(my|the)\s+(cart|order|basket)\b|\bplease\b|\bfor me\b", re.IGNORECASE)
_CATEGORY_WORDS = {
    "burger": "Burgers", "pizza": "Pizza", "pasta": "Pasta", "salad": "Salads", "dessert": "Desserts",
    "sweet": "Desserts", "drink": "Drinks", "beverage": "Drinks", "juice": "Drinks", "smoothie": "Drinks",
}
_NEGATION = re.compile(
    r"(allergic to|allergy to|allergies to|without|\bno\b|free of|avoid|except|can'?t eat|cannot eat|don'?t eat|"
    r"intolerant to|intolerance to)\s+((?:[a-z\-']+[ ,]*(?:and |or )?){1,6})"
)


def _qty(token: str | None) -> int:
    if not token:
        return 1
    return int(token) if token.isdigit() else _NUM.get(token.lower(), 1)


def _clean_item(phrase: str) -> tuple[str, int | None]:
    phrase = _TRAIL.sub("", phrase).strip(" .,!?")
    m = re.match(rf"^{_QTY}\s*(?:x|×)?\s+(.*)$", phrase, re.IGNORECASE)
    if m:
        return m.group(2).strip(), _qty(m.group(1))
    return re.sub(r"^(the|some|my)\s+", "", phrase, flags=re.IGNORECASE).strip(), None


# Connectors inside a menu name ("Shrimp and Avocado Power Bowl") must not be read as list separators.
_PLACEHOLDERS = {"and": "\x01", "&": "\x02", ",": "\x03"}
_RESTORE = {"\x01": " and ", "\x02": " & ", "\x03": ", "}


def _protect(text: str, known_names: Sequence[str]) -> str:
    for name in known_names:
        if re.search(r"\band\b|&|,", name, re.IGNORECASE):
            safe = re.sub(
                r"\s*(\band\b|&|,)\s*", lambda m: _PLACEHOLDERS[m.group(1).lower()], name, flags=re.IGNORECASE
            )
            text = re.sub(re.escape(name), safe, text, flags=re.IGNORECASE)
    return text


def _split_items(text: str, known_names: Sequence[str] = ()) -> list[str]:
    parts = re.split(r",|\band\b|\bplus\b|&", _protect(text, known_names))
    out = []
    for p in parts:
        for k, v in _RESTORE.items():
            p = p.replace(k, v)
        if p.strip():
            out.append(p.strip())
    return out


def extract_constraints(text: str) -> SearchConstraints:
    t = text.lower()
    c = SearchConstraints()
    m = re.search(r"(?:under|below|less than|up to|at most|within|max(?:imum)?(?: of)?|<=?)\s*(\d{2,5})\s*(kcal|cal(?:ories)?)\b", t)
    if m:
        c.max_calories = int(m.group(1))
    m = re.search(r"(?:over|above|more than|at least|min(?:imum)?(?: of)?|>=?)\s*(\d{2,5})\s*(kcal|cal(?:ories)?)\b", t)
    if m:
        c.min_calories = int(m.group(1))
    # Prices are in USD: "$12", "12 dollars", "12.50 usd" are explicit; a bare number needs a "under/up to/only" cue.
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:usd|dollars?|bucks)\b", t)
    if m:
        c.max_price = float(m.group(1) or m.group(2))
    else:
        m = re.search(
            r"(?:under|below|less than|up to|at most|within|max(?:imum)?(?: of)?|<=?|for|only|budget (?:of|is))\s*(\d{2,3}(?:\.\d+)?)(?!\s*(?:kcal|cal|g\b|gram))",
            t,
        )
        if m and not re.search(rf"{re.escape(m.group(1))}\s*(kcal|cal|g\b|gram)", t):
            c.max_price = float(m.group(1))
    m = re.search(r"(\d{2,3})\s*g(?:rams?)?(?:\s+of)?\s+protein", t)
    if m:
        c.min_protein = int(m.group(1))
    elif re.search(r"high[- ]protein|protein[- ]rich|lots of protein", t):
        c.min_protein = 30
    if re.search(r"low[- ]cal(?:orie)?s?|light meal|\bdiet\b", t) and c.max_calories is None:
        c.max_calories = 500
    allergens: list[str] = []
    for m in _NEGATION.finditer(t):
        allergens += list(normalize_allergens(re.findall(r"[a-z\-']+", m.group(2))))
    for m in re.finditer(r"\b([a-z]+)[- ]free\b", t):
        allergens += list(normalize_allergens([m.group(1)]))
    c.exclude_allergens = list(dict.fromkeys(allergens))
    for word in ("vegan", "vegetarian"):
        if word in t:
            c.dietary.append(word)
    if re.search(r"\bspicy|\bfiery|\bhot and spicy", t) and not re.search(r"not spicy|non[- ]spicy|no spic", t):
        c.spicy_only = True
    for word, cat in _CATEGORY_WORDS.items():
        if re.search(rf"\b{word}s?\b", t) and cat not in c.categories:
            c.categories.append(cat)
    return c


_FOLLOW_UP = re.compile(
    r"^(and|also|but|what about|how about|which (of|ones?)|any(thing)?|make (it|them)|cheaper|spicier|less|more|only)\b"
    r"|\b(those|them|these|that one|ones)\b"
)


def _is_follow_up(text: str) -> bool:
    """A short or referring message ("which of those are vegan?") that only makes sense with the previous question."""
    t = text.lower().strip()
    return len(t.split()) <= 3 or bool(_FOLLOW_UP.search(t))


class RuleTurnParser:
    """Deterministic regex-based interpretation (no model call)."""

    name = "rules"

    async def parse(
        self, message: str, history: list[ChatMessage], pending_confirmation: bool, known_names: Sequence[str] = ()
    ) -> ParsedTurn:
        text = message.strip()
        t = text.lower().rstrip("!. ")
        words = t.split()
        if re.search(r"\b(confirm|place|submit|finish|complete)\b.*\border\b|\bcheck ?out\b|\bpay\b", t):
            return ParsedTurn(intent="confirm_order")
        if len(words) <= 4 and re.match(r"^(yes|yeah|yep|yup|sure|ok|okay|confirm|go ahead|do it|please do|y)\b", t):
            return ParsedTurn(intent="affirm")
        if len(words) <= 4 and re.match(r"^(no|nope|nah|cancel|never ?mind|not now|stop)\b", t):
            return ParsedTurn(intent="decline")
        replace = re.search(r"(?:remove|delete|take out|drop|swap|replace)\s+(?:the\s+)?(.+?)\s+(?:and\s+(?:replace|swap)\s+it\s+with|with|for|and\s+(?:add|get me))\s+(.+)", t)
        if replace:
            old, _ = _clean_item(replace.group(1))
            new, q = _clean_item(replace.group(2))
            return ParsedTurn(intent="cart_update", cart_actions=[CartAction(op="remove", item=old), CartAction(op="add", item=new, quantity=q or 1)])
        setq = re.search(rf"(?:change|set|update|make)\s+(?:the\s+)?(?:quantity\s+of\s+)?(.+?)\s+(?:quantity\s+)?(?:to|=)\s*{_QTY}\b", t)
        if setq:
            item, _ = _clean_item(setq.group(1))
            return ParsedTurn(intent="cart_update", cart_actions=[CartAction(op="set_quantity", item=item, quantity=_qty(setq.group(2)))])
        rm = re.search(r"(?:remove|delete|take out|drop)\s+(.+)", t)
        if rm:
            acts = []
            for p in _split_items(rm.group(1), known_names):
                item, _ = _clean_item(p)
                if item:
                    acts.append(CartAction(op="remove", item=item))
            if acts:
                return ParsedTurn(intent="cart_update", cart_actions=acts)
        if re.search(r"\b(show|view|see|what'?s in|display|check)\b.*\b(cart|basket|order)\b|^(my )?cart\??$", t):
            return ParsedTurn(intent="view_cart")
        add = re.search(r"(?:^|\b)(?:please\s+)?(?:add|order|put|include|i'?ll take|i will take|i'?d like to order|get me)\s+(.+)", t)
        if add:
            acts = []
            for p in _split_items(add.group(1), known_names):
                item, q = _clean_item(p)
                if item:
                    acts.append(CartAction(op="add", item=item, quantity=q or 1))
            if acts:
                return ParsedTurn(intent="cart_update", cart_actions=acts)
        det = re.search(r"(?:tell me about|details (?:of|about|on)|what'?s in|ingredients (?:of|in)|what is in|nutrition (?:of|for))\s+(?:the\s+)?(.+)", t)
        if det:
            return ParsedTurn(intent="dish_details", dish_ref=det.group(1).strip(" ?."))
        if re.match(r"^(hi|hello|hey|good (morning|evening|afternoon)|thanks|thank you)\b", t) and len(words) <= 5:
            return ParsedTurn(intent="greeting")
        previous = next((m.content for m in reversed(history) if m.role == "user"), None)
        if previous and _is_follow_up(text):  # remember the last question: "which of those are vegan?"
            merged = f"{previous.strip()}. {text}"
            return ParsedTurn(intent="search", search_query=merged, constraints=extract_constraints(merged))
        return ParsedTurn(intent="search", search_query=text, constraints=extract_constraints(text))


class LLMTurnParser:
    name = "llm"

    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def parse(
        self, message: str, history: list[ChatMessage], pending_confirmation: bool, known_names: Sequence[str] = ()
    ) -> ParsedTurn:
        convo = "\n".join(f"{m.role}: {m.content[:300]}" for m in history[-6:])
        user = (
            f"CONVERSATION SO FAR:\n{convo or '(none)'}\n\n"
            f"pending_order_confirmation: {str(pending_confirmation).lower()}\n\n"
            f"LATEST CUSTOMER MESSAGE:\n{message}"
        )
        parsed = await self._llm.generate_structured(prompts.INTENT.system, user, ParsedTurn)
        if parsed.intent == "search" and not parsed.search_query:
            parsed.search_query = message
        return parsed


class TurnParser:
    """LLM parser when available, otherwise (or on provider failure) the deterministic rule parser."""

    def __init__(self, llm: LLMClient):
        self._llm_parser = LLMTurnParser(llm)
        self._rules = RuleTurnParser()
        self._llm = llm

    async def parse(
        self, message: str, history: list[ChatMessage], pending: bool, known_names: Sequence[str] = ()
    ) -> tuple[ParsedTurn, str]:
        if self._llm.is_configured:
            try:
                return await self._llm_parser.parse(message, history, pending), "llm"
            except LLMQuotaExceeded:
                raise  # tell the customer instead of quietly degrading
            except ExternalServiceError as exc:
                log.warning("LLM intent parsing failed, using rule parser", extra={"error": exc.message[:200]})
        return await self._rules.parse(message, history, pending, known_names), "rules"
