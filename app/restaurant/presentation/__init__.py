"""Everything under ``/api/v1/restaurant``: menu, chat, cart, orders, RAG and evaluation of the restaurant app."""

from fastapi import APIRouter

from app.restaurant.presentation import cart, chat, evaluation, menu, orders, rag

router = APIRouter(prefix="/restaurant")
for module in (menu, chat, cart, orders, rag, evaluation):
    router.include_router(module.router)
