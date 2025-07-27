"""Router for recommendation endpoints."""

from typing import List

from fastapi import APIRouter, Depends, Query

from recsys_lite.api.dependencies import get_api_state, get_recommendation_service
from recsys_lite.api.error_handling import handle_recommendation_errors
from recsys_lite.api.filters import apply_recommendation_filters, prepare_similar_items_exclusions
from recsys_lite.api.models import RecommendationResponse
from recsys_lite.api.pagination import apply_pagination
from recsys_lite.api.recommendation_helpers import (
    calculate_buffer_k,
    create_recommendation_objects,
    create_recommendation_response,
    get_similar_items_recommendations,
    get_user_recommendations,
    update_metrics_and_log,
    validate_and_extract_parameters,
)
from recsys_lite.api.services import RecommendationService
from recsys_lite.api.state import APIState
from recsys_lite.utils.logging import get_logger

logger = get_logger("api.routers.recommendations")

router = APIRouter()


@router.get("/recommend", response_model=RecommendationResponse)
async def recommend(
    user_id: str = Query(..., description="User ID to get recommendations for"),
    k: int = Query(10, description="Number of recommendations to return"),
    use_faiss: bool = Query(True, description="Whether to use Faiss index or direct model"),
    # Pagination parameters
    page: int = Query(1, description="Page number (1-based)", ge=1),
    page_size: int = Query(10, description="Number of items per page", ge=1, le=100),
    # Filtering parameters
    categories: List[str] = Query(None, description="Filter by categories"),
    brands: List[str] = Query(None, description="Filter by brands"),
    min_price: float = Query(None, description="Minimum price"),
    max_price: float = Query(None, description="Maximum price"),
    exclude_items: List[str] = Query(None, description="Item IDs to exclude"),
    include_items: List[str] = Query(None, description="Limit to these item IDs"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    state: APIState = Depends(get_api_state),
) -> RecommendationResponse:
    """Get recommendations for a user with pagination and filtering.

    Args:
        user_id: User ID to get recommendations for
        k: Number of recommendations to return
        use_faiss: Whether to use Faiss index (faster) or direct model predictions
        page: Page number (1-based)
        page_size: Number of items per page (max 100)
        categories: Filter by categories
        brands: Filter by brands
        min_price: Minimum price filter
        max_price: Maximum price filter
        exclude_items: Item IDs to exclude from recommendations
        include_items: Limit recommendations to these item IDs
        recommendation_service: Recommendation service from dependency injection
        state: API state for metrics

    Returns:
        Recommendation response with pagination and filter information

    Raises:
        UserNotFoundError: If user ID is not found
        ModelNotInitializedError: If recommendation system is not initialized
        VectorRetrievalError: If vector retrieval fails
    """
    try:
        # Validate and extract parameters
        params = validate_and_extract_parameters(
            user_id=user_id,
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

        # Calculate buffer k for filtering
        buffer_k = calculate_buffer_k(
            k=params["k"],
            categories=params["categories"],
            brands=params["brands"],
            min_price=params["min_price"],
            max_price=params["max_price"],
            exclude_items=params["exclude_items"],
            include_items=params["include_items"],
        )

        # Get recommendations
        item_ids, scores, item_metadata = get_user_recommendations(
            recommendation_service=recommendation_service,
            user_id=params["user_id"],
            k=buffer_k,
            use_faiss=use_faiss,
            categories=params["categories"],
            brands=params["brands"],
            min_price=params["min_price"],
            max_price=params["max_price"],
            exclude_items=params["exclude_items"],
            include_items=params["include_items"],
        )

        # Apply filtering
        item_ids, scores, item_metadata, filter_info = apply_recommendation_filters(
            recommendation_service=recommendation_service,
            item_ids=item_ids,
            scores=scores,
            item_metadata=item_metadata,
            categories=params["categories"],
            brands=params["brands"],
            min_price=params["min_price"],
            max_price=params["max_price"],
            exclude_items=params["exclude_items"],
            include_items=params["include_items"],
            context_type="recommendations",
            context_id=params["user_id"],
        )

        # Apply pagination
        item_ids, scores, item_metadata, pagination_info = apply_pagination(
            recommendation_service=recommendation_service,
            item_ids=item_ids,
            scores=scores,
            item_metadata=item_metadata,
            page=params["page"],
            page_size=params["page_size"],
            context_type="recommendations",
            context_id=params["user_id"],
        )

        # Create recommendation objects and response
        recommendations = create_recommendation_objects(item_ids, scores, item_metadata)

        # Update metrics and log
        update_metrics_and_log(
            state=state,
            recommendations=recommendations,
            context_type="recommendations",
            context_id=params["user_id"],
            use_faiss=use_faiss,
            filter_info=filter_info,
            pagination_info=pagination_info,
        )

        return create_recommendation_response(
            user_id=params["user_id"],
            recommendations=recommendations,
            pagination_info=pagination_info,
            filter_info=filter_info,
        )

    except Exception as e:
        handle_recommendation_errors(
            error=e,
            state=state,
            context_type="recommendations",
            context_data={"user_id": user_id, "k": k, "use_faiss": use_faiss, "page": page, "page_size": page_size},
        )


@router.get("/similar-items", response_model=RecommendationResponse)
async def similar_items(
    item_id: str = Query(..., description="Item ID to find similar items for"),
    k: int = Query(10, description="Number of similar items to return"),
    # Pagination parameters
    page: int = Query(1, description="Page number (1-based)", ge=1),
    page_size: int = Query(10, description="Number of items per page", ge=1, le=100),
    # Filtering parameters
    categories: List[str] = Query(None, description="Filter by categories"),
    brands: List[str] = Query(None, description="Filter by brands"),
    min_price: float = Query(None, description="Minimum price"),
    max_price: float = Query(None, description="Maximum price"),
    exclude_items: List[str] = Query(None, description="Item IDs to exclude"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    state: APIState = Depends(get_api_state),
) -> RecommendationResponse:
    """Get similar items with pagination and filtering.

    Args:
        item_id: Item ID to find similar items for
        k: Number of similar items to return
        page: Page number (1-based)
        page_size: Number of items per page (max 100)
        categories: Filter by categories
        brands: Filter by brands
        min_price: Minimum price filter
        max_price: Maximum price filter
        exclude_items: Item IDs to exclude from similar items
        recommendation_service: Recommendation service from dependency injection
        state: API state for metrics

    Returns:
        Recommendation response with pagination and filter information

    Raises:
        ItemNotFoundError: If item ID is not found
        ModelNotInitializedError: If recommendation system is not initialized
        VectorRetrievalError: If vector retrieval fails
    """
    try:
        # Validate and extract parameters
        params = validate_and_extract_parameters(
            item_id=item_id,
            k=k,
            page=page,
            page_size=page_size,
            min_price=min_price,
            max_price=max_price,
            categories=categories,
            brands=brands,
            exclude_items=exclude_items,
        )

        # Prepare exclusions (ensure original item is excluded)
        prepared_exclude_items = prepare_similar_items_exclusions(
            item_id=params["item_id"], exclude_items=params["exclude_items"]
        )

        # Calculate buffer k for filtering
        buffer_k = calculate_buffer_k(
            k=params["k"],
            categories=params["categories"],
            brands=params["brands"],
            min_price=params["min_price"],
            max_price=params["max_price"],
            exclude_items=prepared_exclude_items,
        )

        # Get similar items
        item_ids, scores, item_metadata = get_similar_items_recommendations(
            recommendation_service=recommendation_service,
            item_id=params["item_id"],
            k=buffer_k,
            categories=params["categories"],
            brands=params["brands"],
            min_price=params["min_price"],
            max_price=params["max_price"],
            exclude_items=prepared_exclude_items,
        )

        # Apply filtering
        item_ids, scores, item_metadata, filter_info = apply_recommendation_filters(
            recommendation_service=recommendation_service,
            item_ids=item_ids,
            scores=scores,
            item_metadata=item_metadata,
            categories=params["categories"],
            brands=params["brands"],
            min_price=params["min_price"],
            max_price=params["max_price"],
            exclude_items=prepared_exclude_items,
            include_items=None,
            context_type="similar_items",
            context_id=params["item_id"],
        )

        # Apply pagination
        item_ids, scores, item_metadata, pagination_info = apply_pagination(
            recommendation_service=recommendation_service,
            item_ids=item_ids,
            scores=scores,
            item_metadata=item_metadata,
            page=params["page"],
            page_size=params["page_size"],
            context_type="similar_items",
            context_id=params["item_id"],
        )

        # Create recommendation objects and response
        recommendations = create_recommendation_objects(item_ids, scores, item_metadata)

        # Update metrics and log
        update_metrics_and_log(
            state=state,
            recommendations=recommendations,
            context_type="similar_items",
            context_id=params["item_id"],
            filter_info=filter_info,
            pagination_info=pagination_info,
        )

        return create_recommendation_response(
            user_id=params["item_id"],  # Use item_id as user_id for similar items
            recommendations=recommendations,
            pagination_info=pagination_info,
            filter_info=filter_info,
        )

    except Exception as e:
        handle_recommendation_errors(
            error=e,
            state=state,
            context_type="similar_items",
            context_data={"item_id": item_id, "k": k, "page": page, "page_size": page_size},
        )
