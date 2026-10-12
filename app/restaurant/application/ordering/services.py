"""Cart and order use cases. The ONLY code path that mutates carts/orders (REST and chatbot alike)."""

from __future__ import annotations

from app.errors import ConflictError, NotFoundError
from app.restaurant.domain.chat.repositories import ChatRepository
from app.restaurant.domain.ordering.entities import Cart, CartItem, Order
from app.restaurant.domain.ordering.repositories import CartRepository, OrderRepository
from app.restaurant.domain.repositories import MenuRepository


class CartService:
    def __init__(self, carts: CartRepository, menu: MenuRepository, chats: ChatRepository):
        self._carts, self._menu, self._chats = carts, menu, chats

    async def _require_session(self, session_id: str) -> None:
        if await self._chats.get_session(session_id) is None:
            raise NotFoundError(f"Session '{session_id}' not found.")

    async def get_cart(self, session_id: str) -> Cart:
        await self._require_session(session_id)
        return await self._carts.get(session_id)

    async def add_item(self, session_id: str, dish_id: str, quantity: int) -> tuple[Cart, CartItem]:
        cart = await self.get_cart(session_id)
        dish = await self._menu.get_dish(dish_id)  # price/name/availability are always server-side
        if dish is None:
            raise NotFoundError(f"Dish '{dish_id}' not found.")
        item = cart.add(dish, quantity)
        await self._carts.save(cart)
        return cart, item

    async def update_quantity(self, session_id: str, item_id: str, quantity: int) -> Cart:
        cart = await self.get_cart(session_id)
        cart.set_quantity(item_id, quantity)
        await self._carts.save(cart)
        return cart

    async def remove_item(self, session_id: str, item_id: str) -> Cart:
        cart = await self.get_cart(session_id)
        cart.remove(item_id)
        await self._carts.save(cart)
        return cart


class OrderService:
    def __init__(self, carts: CartRepository, orders: OrderRepository, menu: MenuRepository, chats: ChatRepository):
        self._carts, self._orders, self._menu, self._chats = carts, orders, menu, chats

    async def place_order(self, session_id: str, *, confirmed: bool) -> Order:
        if await self._chats.get_session(session_id) is None:
            raise NotFoundError(f"Session '{session_id}' not found.")
        cart = await self._carts.get(session_id)
        # Re-validate against the current menu before committing the order.
        dishes = {d.id: d for d in await self._menu.get_dishes([i.dish_id for i in cart.items])}
        for item in cart.items:
            dish = dishes.get(item.dish_id)
            if dish is None or not dish.available:
                raise ConflictError(f"'{item.name}' is no longer available. Please remove it from your cart.")
            item.unit_price = dish.price
        order = Order.place(cart, confirmed=confirmed)
        await self._orders.add(order)
        cart.clear()
        await self._carts.save(cart)
        return order

    async def get_order(self, order_id: str) -> Order:
        order = await self._orders.get(order_id)
        if order is None:
            raise NotFoundError(f"Order '{order_id}' not found.")
        return order
