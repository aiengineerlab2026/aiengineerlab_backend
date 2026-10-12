"""Orders & Cart bounded context.

The Cart aggregate owns every invariant on cart mutations. Callers (REST API *and* the chatbot)
go through ``CartService`` -> ``Cart`` so an LLM can never write to the database directly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from app.errors import ConflictError, NotFoundError, ValidationFailed
from app.restaurant.domain.entities import Dish

MAX_QUANTITY_PER_ITEM = 20


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class CartItem:
    id: str
    dish_id: str
    name: str
    unit_price: Decimal
    quantity: int
    category: str = ""

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity


@dataclass
class Cart:
    session_id: str
    currency: str = "USD"
    items: list[CartItem] = field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return sum((i.line_total for i in self.items), Decimal(0))

    @property
    def item_count(self) -> int:
        return sum(i.quantity for i in self.items)

    @staticmethod
    def _check_quantity(quantity: int) -> None:
        if not isinstance(quantity, int) or quantity < 1 or quantity > MAX_QUANTITY_PER_ITEM:
            raise ValidationFailed(f"Quantity must be between 1 and {MAX_QUANTITY_PER_ITEM}.")

    def find(self, item_id: str) -> CartItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise NotFoundError(f"Cart item '{item_id}' not found.")

    def find_by_dish(self, dish_id: str) -> CartItem | None:
        return next((i for i in self.items if i.dish_id == dish_id), None)

    def add(self, dish: Dish, quantity: int) -> CartItem:
        """Add a dish (merging quantity). Price/name always come from the server-side dish."""
        self._check_quantity(quantity)
        if not dish.available:
            raise ConflictError(f"'{dish.name}' is currently unavailable.")
        existing = self.find_by_dish(dish.id)
        if existing:
            self._check_quantity(existing.quantity + quantity)
            existing.quantity += quantity
            existing.unit_price = dish.price
            return existing
        item = CartItem(str(uuid.uuid4()), dish.id, dish.name, dish.price, quantity, dish.category)
        self.items.append(item)
        return item

    def set_quantity(self, item_id: str, quantity: int) -> CartItem:
        self._check_quantity(quantity)
        item = self.find(item_id)
        item.quantity = quantity
        return item

    def remove(self, item_id: str) -> CartItem:
        item = self.find(item_id)
        self.items.remove(item)
        return item

    def clear(self) -> None:
        self.items.clear()


@dataclass(frozen=True)
class OrderItem:
    dish_id: str
    name: str
    unit_price: Decimal
    quantity: int

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity


@dataclass(frozen=True)
class Order:
    id: str
    session_id: str
    items: tuple[OrderItem, ...]
    total: Decimal
    currency: str
    status: str = "confirmed"  # demo order: no payment
    created_at: datetime = field(default_factory=_now)

    @staticmethod
    def place(cart: Cart, *, confirmed: bool) -> Order:
        if not confirmed:
            raise ValidationFailed("Order must be explicitly confirmed by the customer.")
        if not cart.items:
            raise ConflictError("Cannot place an order with an empty cart.")
        items = tuple(OrderItem(i.dish_id, i.name, i.unit_price, i.quantity) for i in cart.items)
        return Order(str(uuid.uuid4()), cart.session_id, items, cart.total, cart.currency)
