from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.ordering.entities import Cart, CartItem, Order, OrderItem
from app.domain.ordering.repositories import CartRepository, OrderRepository
from app.infrastructure.db.models import CartItemModel, CartModel, OrderItemModel, OrderModel


class SqlCartRepository(CartRepository):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def _load(self, session_id: str) -> CartModel | None:
        stmt = select(CartModel).where(CartModel.session_id == uuid.UUID(session_id))
        row = (await self._s.execute(stmt)).unique().scalar_one_or_none()
        if row is not None:
            await self._s.refresh(row, ["items"])
        return row

    async def get(self, session_id: str) -> Cart:
        model = await self._load(session_id)
        if model is None:
            return Cart(session_id=session_id)
        items = [
            CartItem(
                id=str(i.id),
                dish_id=i.dish_id,
                name=i.dish.name,
                unit_price=i.dish.price,  # always the current server-side price
                quantity=i.quantity,
                category=i.dish.category.name,
            )
            for i in model.items
        ]
        return Cart(session_id=session_id, currency=model.currency, items=items)

    async def save(self, cart: Cart) -> None:
        model = await self._load(cart.session_id)
        if model is None:
            model = CartModel(session_id=uuid.UUID(cart.session_id), currency=cart.currency, items=[])
            self._s.add(model)
        existing = {str(i.id): i for i in model.items}
        wanted = {i.id: i for i in cart.items}
        for item_id, row in list(existing.items()):
            if item_id not in wanted:
                model.items.remove(row)
        for item_id, item in wanted.items():
            if item_id in existing:
                existing[item_id].quantity = item.quantity
            else:
                model.items.append(
                    CartItemModel(id=uuid.UUID(item_id), dish_id=item.dish_id, quantity=item.quantity)
                )
        await self._s.flush()


class SqlOrderRepository(OrderRepository):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def add(self, order: Order) -> None:
        self._s.add(
            OrderModel(
                id=uuid.UUID(order.id),
                session_id=uuid.UUID(order.session_id),
                status=order.status,
                total=order.total,
                currency=order.currency,
                items=[
                    OrderItemModel(dish_id=i.dish_id, name=i.name, unit_price=i.unit_price, quantity=i.quantity)
                    for i in order.items
                ],
            )
        )
        await self._s.flush()

    async def get(self, order_id: str) -> Order | None:
        try:
            oid = uuid.UUID(order_id)
        except ValueError:
            return None
        m = await self._s.get(OrderModel, oid)
        if m is None:
            return None
        return Order(
            id=str(m.id),
            session_id=str(m.session_id),
            items=tuple(OrderItem(i.dish_id, i.name, i.unit_price, i.quantity) for i in m.items),
            total=m.total,
            currency=m.currency,
            status=m.status,
            created_at=m.created_at,
        )
