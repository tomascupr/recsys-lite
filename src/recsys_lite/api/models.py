"""Pydantic models for the RecSys-Lite API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class FilterInfo(TypedDict, total=False):
    """Type definition for filter information."""

    original_count: int
    filtered_count: int
    filters_applied: Dict[str, Any]


class PaginationData(TypedDict):
    """Type definition for pagination data."""

    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class ItemMetadata(TypedDict, total=False):
    """Type definition for item metadata."""

    title: Optional[str]
    category: Optional[str]
    brand: Optional[str]
    price: Optional[float]
    img_url: Optional[str]


class ModelInfo(TypedDict, total=False):
    """Type definition for model information."""

    num_users: int
    num_items: int
    num_interactions: int
    embedding_dim: Optional[int]
    model_type: str


class Recommendation(BaseModel):
    """Recommendation model for API responses."""

    item_id: str
    score: float
    title: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    img_url: Optional[str] = None


class PaginationInfo(BaseModel):
    """Pagination information."""

    total: int = Field(..., description="Total number of available items")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class RecommendationResponse(BaseModel):
    """API response for recommendations."""

    user_id: str
    recommendations: List[Recommendation]
    pagination: Optional[PaginationInfo] = None
    filter_info: Optional[FilterInfo] = None


class FilterOptions(BaseModel):
    """Filter options for recommendations."""

    categories: Optional[List[str]] = Field(None, description="Filter by category names")
    brands: Optional[List[str]] = Field(None, description="Filter by brand names")
    min_price: Optional[float] = Field(None, description="Minimum price filter")
    max_price: Optional[float] = Field(None, description="Maximum price filter")
    exclude_items: Optional[List[str]] = Field(None, description="Item IDs to exclude")
    include_items: Optional[List[str]] = Field(None, description="Limit to these item IDs")


class MetricsResponse(BaseModel):
    """API response for metrics endpoint."""

    uptime_seconds: float
    request_count: int
    recommendation_count: int
    error_count: int
    recommendations_per_second: float
    model_type: Optional[str] = None
    model_info: Optional[ModelInfo] = None
