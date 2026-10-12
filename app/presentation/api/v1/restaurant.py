from __future__ import annotations

from fastapi import APIRouter, Query

from app.presentation.api.v1.schemas import AppInfo, CategoryOut, DishOut, MenuOut, RestaurantOut
from app.presentation.deps import ServicesDep

router = APIRouter(tags=["restaurant"])

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
        route="/apps/restaurant",
        icon="utensils",
        status="active",
    )
]


@router.get("/apps", response_model=list[AppInfo], summary="List registered RAG applications")
async def list_apps() -> list[AppInfo]:
    return APP_REGISTRY


@router.get("/restaurant/menu", response_model=MenuOut)
async def get_menu(
    svc: ServicesDep,
    category: str | None = Query(None, description="Filter by category name"),
    available_only: bool = False,
) -> MenuOut:
    restaurant = await svc.menu.restaurant()
    dishes = await svc.menu.list_dishes(category, available_only)
    return MenuOut(restaurant=RestaurantOut.of(restaurant), dishes=[DishOut.of(d) for d in dishes])


@router.get("/restaurant/categories", response_model=list[CategoryOut])
async def get_categories(svc: ServicesDep) -> list[CategoryOut]:
    return [CategoryOut.of(c) for c in await svc.menu.categories()]


@router.get("/restaurant/menu/{dish_id}", response_model=DishOut)
async def get_dish(dish_id: str, svc: ServicesDep) -> DishOut:
    return DishOut.of(await svc.menu.get_dish(dish_id))
