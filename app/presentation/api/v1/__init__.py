from fastapi import APIRouter

from app.presentation.api.v1 import evaluation, ordering, rag, restaurant

api_router = APIRouter(prefix="/api/v1")
for module in (restaurant, ordering, rag, evaluation):
    api_router.include_router(module.router)
