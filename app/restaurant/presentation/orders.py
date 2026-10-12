from __future__ import annotations

from fastapi import APIRouter, status

from app.restaurant.presentation.deps import ServicesDep
from app.restaurant.presentation.schemas import CreateOrderIn, OrderOut

router = APIRouter(prefix="/orders", tags=["restaurant: orders"])


@router.post("", response_model=OrderOut, status_code=status.HTTP_201_CREATED)
async def create_order(body: CreateOrderIn, svc: ServicesDep) -> OrderOut:
    return OrderOut.of(await svc.orders.place_order(body.session_id, confirmed=body.confirm))


@router.get("/{order_id}", response_model=OrderOut)
async def get_order(order_id: str, svc: ServicesDep) -> OrderOut:
    return OrderOut.of(await svc.orders.get_order(order_id))
