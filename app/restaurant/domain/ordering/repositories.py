from __future__ import annotations

from abc import ABC, abstractmethod

from app.restaurant.domain.ordering.entities import Cart, Order


class CartRepository(ABC):
    @abstractmethod
    async def get(self, session_id: str) -> Cart:
        """Returns the cart for the session (empty if none exists yet)."""

    @abstractmethod
    async def save(self, cart: Cart) -> None: ...


class OrderRepository(ABC):
    @abstractmethod
    async def add(self, order: Order) -> None: ...

    @abstractmethod
    async def get(self, order_id: str) -> Order | None: ...
