from __future__ import annotations

from fastapi import APIRouter, status

from app.restaurant.presentation.deps import ServicesDep
from app.restaurant.presentation.schemas import AddCartItemIn, CartOut, UpdateCartItemIn

router = APIRouter(prefix="/carts", tags=["restaurant: cart"])


@router.get("/{session_id}", response_model=CartOut)
async def get_cart(session_id: str, svc: ServicesDep) -> CartOut:
    return CartOut.of(await svc.carts.get_cart(session_id))


@router.post("/{session_id}/items", response_model=CartOut, status_code=status.HTTP_201_CREATED)
async def add_cart_item(session_id: str, body: AddCartItemIn, svc: ServicesDep) -> CartOut:
    cart, _ = await svc.carts.add_item(session_id, body.dish_id, body.quantity)
    return CartOut.of(cart)


@router.patch("/{session_id}/items/{item_id}", response_model=CartOut)
async def update_cart_item(session_id: str, item_id: str, body: UpdateCartItemIn, svc: ServicesDep) -> CartOut:
    return CartOut.of(await svc.carts.update_quantity(session_id, item_id, body.quantity))


@router.delete("/{session_id}/items/{item_id}", response_model=CartOut)
async def delete_cart_item(session_id: str, item_id: str, svc: ServicesDep) -> CartOut:
    return CartOut.of(await svc.carts.remove_item(session_id, item_id))
