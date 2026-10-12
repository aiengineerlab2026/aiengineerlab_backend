from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.restaurant.domain.chat.entities import ChatMessage
from app.restaurant.presentation.deps import InfraDep, ServicesDep, client_ip, enforce_chat_rate_limit
from app.restaurant.presentation.schemas import (
    CartOut,
    ChatUsageOut,
    DishOut,
    MessageOut,
    OrderOut,
    SendMessageIn,
    SendMessageOut,
    SessionOut,
)

router = APIRouter(prefix="/chat", tags=["restaurant: chat"])


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(svc: ServicesDep) -> SessionOut:
    return SessionOut.of(await svc.chat.create_session())


@router.get("/usage", response_model=ChatUsageOut)
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


@router.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: str, svc: ServicesDep) -> SessionOut:
    return SessionOut.of(await svc.chat.get_session(session_id))


@router.post(
    "/sessions/{session_id}/messages",
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
