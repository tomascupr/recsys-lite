"""Helper functions for recommendation processing."""

from typing import Any, Dict, List, Optional, Tuple

from recsys_lite.api.models import FilterInfo, ItemMetadata, PaginationInfo, Recommendation, RecommendationResponse
from recsys_lite.api.services import RecommendationService
from recsys_lite.api.state import APIState
from recsys_lite.api.validation import validate_request_parameters
from recsys_lite.utils.logging import get_logger

logger = get_logger("api.recommendation_helpers")


def validate_and_extract_parameters(
    user_id: Optional[str] = None,
    item_id: Optional[str] = None,
    k: Optional[int] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    categories: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    exclude_items: Optional[List[str]] = None,
    include_items: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate parameters and extract validated values.

    Args:
        user_id: User ID to validate
        item_id: Item ID to validate
        k: Number of recommendations
        page: Page number
        page_size: Page size
        min_price: Minimum price
        max_price: Maximum price
        categories: Categories filter
        brands: Brands filter
        exclude_items: Items to exclude
        include_items: Items to include

    Returns:
        Dictionary of validated parameters
    """
    # Validate all input parameters
    validated_params = validate_request_parameters(
        user_id=user_id,
        item_id=item_id,
        k=k,
        page=page,
        page_size=page_size,
        min_price=min_price,
        max_price=max_price,
        categories=categories,
        brands=brands,
        exclude_items=exclude_items,
        include_items=include_items,
    )

    # Extract validated parameters with defaults
    result = {
        "k": validated_params.get("k", k),
        "page": validated_params.get("page", page),
        "page_size": validated_params.get("page_size", page_size),
        "min_price": validated_params.get("min_price"),
        "max_price": validated_params.get("max_price"),
        "categories": validated_params.get("categories"),
        "brands": validated_params.get("brands"),
        "exclude_items": validated_params.get("exclude_items"),
        "include_items": validated_params.get("include_items"),
    }

    if user_id is not None:
        result["user_id"] = validated_params["user_id"]
    if item_id is not None:
        result["item_id"] = validated_params["item_id"]

    return result


def calculate_buffer_k(
    k: int,
    categories: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    exclude_items: Optional[List[str]] = None,
    include_items: Optional[List[str]] = None,
    buffer_factor: int = 3,
    max_buffer_k: int = 1000,
) -> int:
    """Calculate buffered k value for filtering.

    Args:
        k: Original number of recommendations
        categories: Categories filter
        brands: Brands filter
        min_price: Minimum price
        max_price: Maximum price
        exclude_items: Items to exclude
        include_items: Items to include
        buffer_factor: Multiplication factor for buffer
        max_buffer_k: Maximum buffer k value

    Returns:
        Buffered k value
    """
    # Check if we need buffering for filtering
    use_buffer = any([categories, brands, min_price, max_price, exclude_items, include_items])
    buffer_k = k * buffer_factor if use_buffer else k

    # Cap at a reasonable maximum
    return min(buffer_k, max_buffer_k)


def get_user_recommendations(
    recommendation_service: RecommendationService,
    user_id: str,
    k: int,
    use_faiss: bool,
    categories: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    exclude_items: Optional[List[str]] = None,
    include_items: Optional[List[str]] = None,
) -> Tuple[List[str], List[float], List[ItemMetadata]]:
    """Get recommendations for a user from the recommendation service.

    Args:
        recommendation_service: Recommendation service instance
        user_id: User ID
        k: Number of recommendations
        use_faiss: Whether to use Faiss index
        categories: Categories filter
        brands: Brands filter
        min_price: Minimum price
        max_price: Maximum price
        exclude_items: Items to exclude
        include_items: Items to include

    Returns:
        Tuple of (item_ids, scores, item_metadata)
    """
    logger.debug(
        f"Generating recommendations for user {user_id}", extra={"user_id": user_id, "k": k, "use_faiss": use_faiss}
    )

    return recommendation_service.recommend_for_user(
        user_id=user_id,
        k=k,
        use_faiss=use_faiss,
        categories=categories,
        brands=brands,
        min_price=min_price,
        max_price=max_price,
        exclude_items=exclude_items,
        include_items=include_items,
    )


def get_similar_items_recommendations(
    recommendation_service: RecommendationService,
    item_id: str,
    k: int,
    categories: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    exclude_items: Optional[List[str]] = None,
) -> Tuple[List[str], List[float], List[ItemMetadata]]:
    """Get similar items from the recommendation service.

    Args:
        recommendation_service: Recommendation service instance
        item_id: Item ID
        k: Number of similar items
        categories: Categories filter
        brands: Brands filter
        min_price: Minimum price
        max_price: Maximum price
        exclude_items: Items to exclude

    Returns:
        Tuple of (item_ids, scores, item_metadata)
    """
    logger.debug(f"Finding similar items for item {item_id}", extra={"item_id": item_id, "k": k})

    return recommendation_service.find_similar_items(
        item_id=item_id,
        k=k,
        categories=categories,
        brands=brands,
        min_price=min_price,
        max_price=max_price,
        exclude_items=exclude_items,
    )


def create_recommendation_objects(
    item_ids: List[str],
    scores: List[float],
    item_metadata: List[ItemMetadata],
) -> List[Recommendation]:
    """Create Recommendation objects from raw data.

    Args:
        item_ids: List of item IDs
        scores: List of scores
        item_metadata: List of item metadata

    Returns:
        List of Recommendation objects
    """
    return [
        Recommendation(
            item_id=item_id,
            score=float(score),
            title=metadata.get("title"),
            category=metadata.get("category"),
            brand=metadata.get("brand"),
            price=metadata.get("price"),
            img_url=metadata.get("img_url"),
        )
        for item_id, score, metadata in zip(item_ids, scores, item_metadata)
    ]


def create_recommendation_response(
    user_id: str,
    recommendations: List[Recommendation],
    pagination_info: Optional[PaginationInfo] = None,
    filter_info: Optional[FilterInfo] = None,
) -> RecommendationResponse:
    """Create a RecommendationResponse object.

    Args:
        user_id: User ID (or item ID for similar items)
        recommendations: List of recommendations
        pagination_info: Pagination information
        filter_info: Filter information

    Returns:
        RecommendationResponse object
    """
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        pagination=pagination_info,
        filter_info=filter_info,
    )


def update_metrics_and_log(
    state: APIState,
    recommendations: List[Recommendation],
    context_type: str,
    context_id: str,
    use_faiss: Optional[bool] = None,
    filter_info: Optional[FilterInfo] = None,
    pagination_info: Optional[PaginationInfo] = None,
) -> None:
    """Update metrics and log the results.

    Args:
        state: API state for metrics
        recommendations: List of recommendations
        context_type: Type of context (recommendations/similar_items)
        context_id: Context ID (user_id/item_id)
        use_faiss: Whether Faiss was used (for user recommendations)
        filter_info: Filter information
        pagination_info: Pagination information
    """
    # Update metrics
    state.increase_recommendation_count(len(recommendations))

    # Prepare logging extra data
    log_extra = {
        "context_id": context_id,
        "recommendation_count": len(recommendations),
        "filtered": filter_info is not None,
        "paginated": pagination_info is not None,
    }

    if use_faiss is not None:
        log_extra["use_faiss"] = use_faiss

    logger.info(f"Generated {len(recommendations)} {context_type} for {context_id}", extra=log_extra)
