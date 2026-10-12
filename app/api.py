from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.restaurant import presentation as restaurant


class AppInfo(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    route: str
    icon: str
    status: str = "active"


apps_router = APIRouter(tags=["apps"])

# Application registry: add an entry (and a frontend route) to register another RAG application.
APP_REGISTRY: list[AppInfo] = [
    AppInfo(
        id="restaurant",
        name="The Restaurant",
        description=(
            "An AI-powered restaurant ordering assistant where you can experiment with chunking, "
            "retrieval strategies, reranking, and RAG evaluation."
        ),
        tags=["LangChain", "RAG", "RAGAS"],
        route="/restaurant",
        icon="utensils",
        status="active",
    )
]


@apps_router.get("/apps", response_model=list[AppInfo], summary="List registered RAG applications")
async def list_apps() -> list[AppInfo]:
    return APP_REGISTRY


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(apps_router)
api_router.include_router(restaurant.router)
