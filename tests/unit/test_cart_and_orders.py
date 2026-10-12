from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from app.domain.errors import ConflictError, NotFoundError, ValidationFailed
from app.domain.ordering.entities import MAX_QUANTITY_PER_ITEM, Cart, Order


def test_add_merges_quantities_and_uses_server_side_price(dishes):
    cart = Cart("s1")
    d = dishes[0]
    cart.add(d, 1)
    cart.add(d, 2)
    assert len(cart.items) == 1
    assert cart.items[0].quantity == 3
    assert cart.items[0].unit_price == d.price
    assert cart.total == d.price * 3
    assert cart.item_count == 3


@pytest.mark.parametrize("qty", [0, -1, MAX_QUANTITY_PER_ITEM + 1, 1.5, "2"])
def test_invalid_quantities_are_rejected(dishes, qty):
    with pytest.raises(ValidationFailed):
        Cart("s1").add(dishes[0], qty)


def test_merged_quantity_cannot_exceed_the_limit(dishes):
    cart = Cart("s1")
    cart.add(dishes[0], MAX_QUANTITY_PER_ITEM)
    with pytest.raises(ValidationFailed):
        cart.add(dishes[0], 1)
    assert cart.items[0].quantity == MAX_QUANTITY_PER_ITEM


def test_unavailable_dish_cannot_be_added(dishes):
    with pytest.raises(ConflictError):
        Cart("s1").add(replace(dishes[0], available=False), 1)


def test_set_quantity_and_remove(dishes):
    cart = Cart("s1")
    item = cart.add(dishes[0], 1)
    cart.set_quantity(item.id, 5)
    assert cart.items[0].quantity == 5
    with pytest.raises(ValidationFailed):
        cart.set_quantity(item.id, 0)
    cart.remove(item.id)
    assert cart.items == []
    with pytest.raises(NotFoundError):
        cart.remove(item.id)


def test_order_requires_explicit_confirmation_and_a_non_empty_cart(dishes):
    cart = Cart("s1")
    with pytest.raises(ValidationFailed):
        Order.place(cart, confirmed=False)
    with pytest.raises(ConflictError):
        Order.place(cart, confirmed=True)
    cart.add(dishes[0], 2)
    with pytest.raises(ValidationFailed):
        Order.place(cart, confirmed=False)
    order = Order.place(cart, confirmed=True)
    assert order.total == dishes[0].price * 2
    assert order.items[0].quantity == 2


# ------------------------------------------------------------------------------ services (in-memory repositories)


async def _session(world) -> str:
    return (await world.chats.create_session()).id


async def test_cart_service_roundtrip(world):
    sid = await _session(world)
    d = world.dishes[0]
    cart, item = await world.cart_service.add_item(sid, d.id, 2)
    assert cart.total == d.price * 2
    cart = await world.cart_service.update_quantity(sid, item.id, 3)
    assert cart.item_count == 3
    cart = await world.cart_service.remove_item(sid, item.id)
    assert cart.items == []
    assert (await world.cart_service.get_cart(sid)).items == []


async def test_cart_service_rejects_unknown_session_and_dish(world):
    with pytest.raises(NotFoundError):
        await world.cart_service.get_cart("nope")
    sid = await _session(world)
    with pytest.raises(NotFoundError):
        await world.cart_service.add_item(sid, "dish_999", 1)


async def test_place_order_clears_cart_and_is_retrievable(world):
    sid = await _session(world)
    await world.cart_service.add_item(sid, world.dishes[0].id, 2)
    order = await world.order_service.place_order(sid, confirmed=True)
    assert order.total == world.dishes[0].price * 2
    assert (await world.cart_service.get_cart(sid)).items == []
    assert (await world.order_service.get_order(order.id)).id == order.id
    with pytest.raises(ConflictError):  # nothing left to order
        await world.order_service.place_order(sid, confirmed=True)
    with pytest.raises(NotFoundError):
        await world.order_service.get_order("missing")


async def test_unconfirmed_order_is_rejected_and_cart_kept(world):
    sid = await _session(world)
    await world.cart_service.add_item(sid, world.dishes[0].id, 1)
    with pytest.raises(ValidationFailed):
        await world.order_service.place_order(sid, confirmed=False)
    assert (await world.cart_service.get_cart(sid)).item_count == 1


async def test_order_revalidates_availability_and_reprices(world):
    sid = await _session(world)
    d = world.dishes[0]
    await world.cart_service.add_item(sid, d.id, 1)
    world.menu.dishes[0] = replace(d, price=Decimal(999))
    order = await world.order_service.place_order(sid, confirmed=True)
    assert order.total == Decimal(999)  # current menu price, not the stale cart price

    await world.cart_service.add_item(sid, d.id, 1)
    world.menu.dishes[0] = replace(d, available=False)
    with pytest.raises(ConflictError):
        await world.order_service.place_order(sid, confirmed=True)
