from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.domain.chat.entities import ChatMessage
from app.presentation.api.v1.schemas import (
    AddCartItemIn,
    CartOut,
    ChatUsageOut,
    CreateOrderIn,
    DishOut,
    MessageOut,
    OrderOut,
    SendMessageIn,
    SendMessageOut,
    SessionOut,
    UpdateCartItemIn,
)
from app.presentation.deps import InfraDep, ServicesDep, client_ip, enforce_chat_rate_limit

router = APIRouter(tags=["chat", "cart", "orders"])


# ------------------------------------------------------------------ chat


@router.post("/chat/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(svc: ServicesDep) -> SessionOut:
    return SessionOut.of(await svc.chat.create_session())


@router.get("/chat/usage", response_model=ChatUsageOut)
async def chat_usage(request: Request, infra: InfraDep) -> ChatUsageOut:
    """Messages this client (IP) has left in the current window. Does not count as a message."""
    s = infra.settings
    u = infra.chat_limiter.peek(client_ip(request, s.trust_proxy_headers))
    return ChatUsageOut(
        enabled=infra.chat_limiter.enabled,
        limit=u.limit,
        used=u.used,
        remaining=u.remaining,
        window_seconds=s.chat_rate_limit_window_seconds,
        reset_in_seconds=u.reset_in_seconds,
    )


@router.get("/chat/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: str, svc: ServicesDep) -> SessionOut:
    return SessionOut.of(await svc.chat.get_session(session_id))


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=SendMessageOut,
    dependencies=[Depends(enforce_chat_rate_limit)],  # 429 after CHAT_RATE_LIMIT_MAX messages per IP
)
async def send_message(session_id: str, body: SendMessageIn, svc: ServicesDep, infra: InfraDep) -> SendMessageOut:
    history = [ChatMessage(session_id=session_id, role=h.role, content=h.content) for h in body.history]
    embedding = infra.resolve_embedding(body.embedding)  # 422 when that model is unknown or not set up
    result = await svc.chat.handle_message(session_id, body.content, history, embedding)
    dishes = await svc.menu_repo.get_dishes(result.recommended_dish_ids)
    return SendMessageOut(
        user_message=MessageOut.of(result.user_message),
        assistant_message=MessageOut.of(result.assistant_message),
        cart=CartOut.of(result.cart),
        recommended_dishes=[DishOut.of(d) for d in dishes],
        order=OrderOut.of(result.order) if result.order else None,
    )


# ------------------------------------------------------------------ cart


@router.get("/carts/{session_id}", response_model=CartOut)
async def get_cart(session_id: str, svc: ServicesDep) -> CartOut:
    return CartOut.of(await svc.carts.get_cart(session_id))


@router.post("/carts/{session_id}/items", response_model=CartOut, status_code=status.HTTP_201_CREATED)
async def add_cart_item(session_id: str, body: AddCartItemIn, svc: ServicesDep) -> CartOut:
    cart, _ = await svc.carts.add_item(session_id, body.dish_id, body.quantity)
    return CartOut.of(cart)


@router.patch("/carts/{session_id}/items/{item_id}", response_model=CartOut)
async def update_cart_item(session_id: str, item_id: str, body: UpdateCartItemIn, svc: ServicesDep) -> CartOut:
    return CartOut.of(await svc.carts.update_quantity(session_id, item_id, body.quantity))


@router.delete("/carts/{session_id}/items/{item_id}", response_model=CartOut)
async def delete_cart_item(session_id: str, item_id: str, svc: ServicesDep) -> CartOut:
    return CartOut.of(await svc.carts.remove_item(session_id, item_id))


# ------------------------------------------------------------------ orders


@router.post("/orders", response_model=OrderOut, status_code=status.HTTP_201_CREATED)
async def create_order(body: CreateOrderIn, svc: ServicesDep) -> OrderOut:
    return OrderOut.of(await svc.orders.place_order(body.session_id, confirmed=body.confirm))


@router.get("/orders/{order_id}", response_model=OrderOut)
async def get_order(order_id: str, svc: ServicesDep) -> OrderOut:
    return OrderOut.of(await svc.orders.get_order(order_id))
