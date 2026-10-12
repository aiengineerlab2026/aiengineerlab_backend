from __future__ import annotations

from fastapi import APIRouter, Query

from app.restaurant.presentation.deps import ServicesDep
from app.restaurant.presentation.schemas import CategoryOut, DishOut, MenuOut, RestaurantOut

router = APIRouter(tags=["restaurant: menu"])


@router.get("/menu", response_model=MenuOut)
async def get_menu(
    svc: ServicesDep,
    category: str | None = Query(None, description="Filter by category name"),
    available_only: bool = False,
) -> MenuOut:
    restaurant = await svc.menu.restaurant()
    dishes = await svc.menu.list_dishes(category, available_only)
    return MenuOut(restaurant=RestaurantOut.of(restaurant), dishes=[DishOut.of(d) for d in dishes])


@router.get("/categories", response_model=list[CategoryOut])
async def get_categories(svc: ServicesDep) -> list[CategoryOut]:
    return [CategoryOut.of(c) for c in await svc.menu.categories()]


@router.get("/menu/{dish_id}", response_model=DishOut)
async def get_dish(dish_id: str, svc: ServicesDep) -> DishOut:
    return DishOut.of(await svc.menu.get_dish(dish_id))
