"""Filtering utilities for recommendation API."""

from typing import List, Optional, Tuple

from recsys_lite.api.models import FilterInfo, ItemMetadata
from recsys_lite.api.services import RecommendationService
from recsys_lite.utils.logging import get_logger

logger = get_logger("api.filters")


def apply_recommendation_filters(
    recommendation_service: RecommendationService,
    item_ids: List[str],
    scores: List[float],
    item_metadata: List[ItemMetadata],
    categories: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    exclude_items: Optional[List[str]] = None,
    include_items: Optional[List[str]] = None,
    context_type: str = "recommendations",
    context_id: str = "",
) -> Tuple[List[str], List[float], List[ItemMetadata], Optional[FilterInfo]]:
    """Apply filters to recommendations if any filter parameters are provided.

    Args:
        recommendation_service: Recommendation service instance
        item_ids: List of item IDs
        scores: List of scores
        item_metadata: List of item metadata
        categories: Filter by categories
        brands: Filter by brands
        min_price: Minimum price filter
        max_price: Maximum price filter
        exclude_items: Item IDs to exclude
        include_items: Limit to these item IDs
        context_type: Type of context for logging (recommendations/similar_items)
        context_id: ID for logging context (user_id/item_id)

    Returns:
        Tuple of (filtered_item_ids, filtered_scores, filtered_item_metadata, filter_info)
    """
    # Check if any filters are provided
    has_filters = any([categories, brands, min_price is not None, max_price is not None, exclude_items, include_items])

    if not has_filters:
        return item_ids, scores, item_metadata, None

    logger.debug(
        f"Applying filters to {context_type} for {context_id}",
        extra={
            "context_id": context_id,
            "context_type": context_type,
            "categories": categories,
            "brands": brands,
            "min_price": min_price,
            "max_price": max_price,
            "exclude_items_count": len(exclude_items) if exclude_items else 0,
            "include_items_count": len(include_items) if include_items else 0,
        },
    )

    return recommendation_service.filter_recommendations(
        item_ids=item_ids,
        scores=scores,
        item_metadata=item_metadata,
        categories=categories,
        brands=brands,
        min_price=min_price,
        max_price=max_price,
        exclude_items=exclude_items,
        include_items=include_items,
    )


def prepare_similar_items_exclusions(item_id: str, exclude_items: Optional[List[str]] = None) -> List[str]:
    """Prepare exclusion list for similar items, ensuring the original item is excluded.

    Args:
        item_id: Original item ID to exclude
        exclude_items: Additional items to exclude

    Returns:
        Updated exclusion list including the original item
    """
    if exclude_items is None:
        return [item_id]
    elif item_id not in exclude_items:
        return exclude_items + [item_id]
    else:
        return exclude_items
