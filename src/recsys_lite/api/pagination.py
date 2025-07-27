"""Pagination utilities for recommendation API."""

from typing import List, Optional, Tuple

from recsys_lite.api.models import ItemMetadata, PaginationInfo
from recsys_lite.api.services import RecommendationService
from recsys_lite.utils.logging import get_logger

logger = get_logger("api.pagination")


def apply_pagination(
    recommendation_service: RecommendationService,
    item_ids: List[str],
    scores: List[float],
    item_metadata: List[ItemMetadata],
    page: int,
    page_size: int,
    context_type: str = "recommendations",
    context_id: str = "",
) -> Tuple[List[str], List[float], List[ItemMetadata], Optional[PaginationInfo]]:
    """Apply pagination to results if needed.

    Args:
        recommendation_service: Recommendation service instance
        item_ids: List of item IDs
        scores: List of scores
        item_metadata: List of item metadata
        page: Page number
        page_size: Page size
        context_type: Type of context for logging (recommendations/similar_items)
        context_id: ID for logging context (user_id/item_id)

    Returns:
        Tuple of (paginated_item_ids, paginated_scores, paginated_item_metadata, pagination_info)
    """
    # Check if pagination is needed
    if page <= 1 and len(item_ids) <= page_size:
        return item_ids, scores, item_metadata, None

    logger.debug(
        f"Applying pagination to {context_type} for {context_id}",
        extra={
            "context_id": context_id,
            "context_type": context_type,
            "page": page,
            "page_size": page_size,
            "total_items": len(item_ids),
        },
    )

    paginated_item_ids, paginated_scores, paginated_metadata, pagination_data = recommendation_service.paginate_results(
        item_ids=item_ids, scores=scores, item_metadata=item_metadata, page=page, page_size=page_size
    )

    # Convert pagination data to PaginationInfo model
    pagination_info = PaginationInfo(**pagination_data)

    return paginated_item_ids, paginated_scores, paginated_metadata, pagination_info
