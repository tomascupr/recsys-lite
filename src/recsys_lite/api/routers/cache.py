"""Router for cache management endpoints."""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from recsys_lite.api.dependencies import get_api_state, get_recommendation_service
from recsys_lite.api.services import RecommendationService
from recsys_lite.api.state import APIState
from recsys_lite.utils.logging import get_logger

logger = get_logger("api.routers.cache")

router = APIRouter()


class CacheStatsResponse(BaseModel):
    """Cache statistics response model."""

    cache_stats: Dict
    cache_enabled: bool


class CacheWarmResponse(BaseModel):
    """Cache warming response model."""

    users_warmed: int
    items_warmed: int
    message: str


class CacheClearResponse(BaseModel):
    """Cache clear response model."""

    cleared: bool
    message: str


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(state: APIState = Depends(get_api_state)) -> CacheStatsResponse:
    """Get cache statistics.

    Args:
        state: API state

    Returns:
        Cache statistics
    """
    cache_stats = state.cache_manager.get_stats()

    return CacheStatsResponse(cache_stats=cache_stats, cache_enabled=state.cache_manager.enabled)


@router.post("/cache/clear", response_model=CacheClearResponse)
async def clear_cache(state: APIState = Depends(get_api_state)) -> CacheClearResponse:
    """Clear all cache entries.

    Args:
        state: API state

    Returns:
        Cache clear response
    """
    try:
        state.cache_manager.clear_all()
        logger.info("Cache cleared successfully")

        return CacheClearResponse(cleared=True, message="Cache cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

        return CacheClearResponse(cleared=False, message=f"Failed to clear cache: {e}")


@router.post("/cache/warm", response_model=CacheWarmResponse)
async def warm_cache(
    popular_users: List[str] = Query(None, description="List of popular user IDs to warm"),
    popular_items: List[str] = Query(None, description="List of popular item IDs to warm"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    state: APIState = Depends(get_api_state),
) -> CacheWarmResponse:
    """Warm cache with popular users and items.

    Args:
        popular_users: List of popular user IDs
        popular_items: List of popular item IDs
        recommendation_service: Recommendation service
        state: API state

    Returns:
        Cache warming response
    """
    try:
        stats = state.cache_manager.warm_cache(
            popular_users=popular_users, popular_items=popular_items, recommendation_service=recommendation_service
        )

        message = f"Cache warmed: {stats['users_warmed']} users, {stats['items_warmed']} items"
        logger.info(message)

        return CacheWarmResponse(
            users_warmed=stats["users_warmed"], items_warmed=stats["items_warmed"], message=message
        )
    except Exception as e:
        logger.error(f"Failed to warm cache: {e}")

        return CacheWarmResponse(users_warmed=0, items_warmed=0, message=f"Failed to warm cache: {e}")


@router.post("/cache/invalidate/user")
async def invalidate_user_cache(
    user_id: Optional[str] = Query(None, description="User ID to invalidate (leave empty for all users)"),
    state: APIState = Depends(get_api_state),
) -> Dict[str, str]:
    """Invalidate user cache entries.

    Args:
        user_id: User ID to invalidate, or None for all users
        state: API state

    Returns:
        Status message
    """
    try:
        state.cache_manager.invalidate_user_cache(user_id)

        if user_id:
            message = f"Invalidated cache for user: {user_id}"
        else:
            message = "Invalidated cache for all users"

        logger.info(message)
        return {"status": "success", "message": message}
    except Exception as e:
        error_msg = f"Failed to invalidate user cache: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


@router.post("/cache/invalidate/item")
async def invalidate_item_cache(
    item_id: Optional[str] = Query(None, description="Item ID to invalidate (leave empty for all items)"),
    state: APIState = Depends(get_api_state),
) -> Dict[str, str]:
    """Invalidate item cache entries.

    Args:
        item_id: Item ID to invalidate, or None for all items
        state: API state

    Returns:
        Status message
    """
    try:
        state.cache_manager.invalidate_item_cache(item_id)

        if item_id:
            message = f"Invalidated cache for item: {item_id}"
        else:
            message = "Invalidated cache for all items"

        logger.info(message)
        return {"status": "success", "message": message}
    except Exception as e:
        error_msg = f"Failed to invalidate item cache: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


@router.post("/cache/invalidate/model")
async def invalidate_model_cache(
    model_version: Optional[str] = Query(None, description="New model version"),
    state: APIState = Depends(get_api_state),
) -> Dict[str, str]:
    """Invalidate model cache when model is updated.

    Args:
        model_version: New model version
        state: API state

    Returns:
        Status message
    """
    try:
        state.cache_manager.invalidate_model_cache(model_version)

        message = f"Invalidated model cache, new version: {model_version}"
        logger.info(message)
        return {"status": "success", "message": message}
    except Exception as e:
        error_msg = f"Failed to invalidate model cache: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
