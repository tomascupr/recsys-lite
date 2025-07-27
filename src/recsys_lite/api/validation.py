"""Input validation utilities for RecSys-Lite API."""

from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

from recsys_lite.utils.logging import get_logger

logger = get_logger("api.validation")


class ValidationError(HTTPException):
    """Custom validation error with 400 status code."""

    def __init__(self, detail: str, field: Optional[str] = None):
        """Initialize validation error.

        Args:
            detail: Error detail message
            field: Field name that failed validation
        """
        full_detail = f"Validation error in field '{field}': {detail}" if field else f"Validation error: {detail}"
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=full_detail)


def validate_positive_integer(value: int, field_name: str, min_value: int = 1, max_value: Optional[int] = None) -> int:
    """Validate that an integer is positive and within bounds.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        min_value: Minimum allowed value (default: 1)
        max_value: Maximum allowed value (optional)

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if value < min_value:
        raise ValidationError(f"must be at least {min_value}", field_name)

    if max_value is not None and value > max_value:
        raise ValidationError(f"must be at most {max_value}", field_name)

    return value


def validate_k_parameter(k: int) -> int:
    """Validate the k parameter for recommendations.

    Args:
        k: Number of recommendations to return

    Returns:
        Validated k value

    Raises:
        ValidationError: If k is invalid
    """
    return validate_positive_integer(k, "k", min_value=1, max_value=1000)


def validate_page_parameter(page: int) -> int:
    """Validate the page parameter.

    Args:
        page: Page number (1-based)

    Returns:
        Validated page value

    Raises:
        ValidationError: If page is invalid
    """
    return validate_positive_integer(page, "page", min_value=1)


def validate_page_size_parameter(page_size: int) -> int:
    """Validate the page_size parameter.

    Args:
        page_size: Number of items per page

    Returns:
        Validated page_size value

    Raises:
        ValidationError: If page_size is invalid
    """
    return validate_positive_integer(page_size, "page_size", min_value=1, max_value=100)


def validate_user_id(user_id: str) -> str:
    """Validate user ID format.

    Args:
        user_id: User ID to validate

    Returns:
        Validated user ID

    Raises:
        ValidationError: If user_id is invalid
    """
    if not user_id or not user_id.strip():
        raise ValidationError("cannot be empty or whitespace only", "user_id")

    user_id = user_id.strip()

    # Check length constraints
    if len(user_id) > 255:
        raise ValidationError("cannot be longer than 255 characters", "user_id")

    # Check for potentially malicious patterns
    dangerous_chars = ["<", ">", "&", '"', "'", ";", "\\", "/"]
    if any(char in user_id for char in dangerous_chars):
        raise ValidationError("contains invalid characters", "user_id")

    return user_id


def validate_item_id(item_id: str) -> str:
    """Validate item ID format.

    Args:
        item_id: Item ID to validate

    Returns:
        Validated item ID

    Raises:
        ValidationError: If item_id is invalid
    """
    if not item_id or not item_id.strip():
        raise ValidationError("cannot be empty or whitespace only", "item_id")

    item_id = item_id.strip()

    # Check length constraints
    if len(item_id) > 255:
        raise ValidationError("cannot be longer than 255 characters", "item_id")

    # Check for potentially malicious patterns
    dangerous_chars = ["<", ">", "&", '"', "'", ";", "\\", "/"]
    if any(char in item_id for char in dangerous_chars):
        raise ValidationError("contains invalid characters", "item_id")

    return item_id


def validate_entity_index(entity_idx: int, entity_type: str, max_entities: Optional[int] = None) -> int:
    """Validate entity index bounds.

    Args:
        entity_idx: Entity index to validate
        entity_type: Type of entity (user/item) for error messages
        max_entities: Maximum number of entities (optional)

    Returns:
        Validated entity index

    Raises:
        ValidationError: If entity_idx is invalid
    """
    if entity_idx < 0:
        raise ValidationError(f"{entity_type} index cannot be negative", f"{entity_type}_idx")

    if max_entities is not None and entity_idx >= max_entities:
        raise ValidationError(
            f"{entity_type} index {entity_idx} is out of bounds (max: {max_entities - 1})", f"{entity_type}_idx"
        )

    return entity_idx


def validate_batch_size(batch_size: int) -> int:
    """Validate batch size parameter for update worker.

    Args:
        batch_size: Batch size to validate

    Returns:
        Validated batch size

    Raises:
        ValidationError: If batch_size is invalid
    """
    return validate_positive_integer(batch_size, "batch_size", min_value=1, max_value=10000)


def validate_interval(interval: int) -> int:
    """Validate interval parameter for update worker.

    Args:
        interval: Interval in seconds to validate

    Returns:
        Validated interval

    Raises:
        ValidationError: If interval is invalid
    """
    return validate_positive_integer(interval, "interval", min_value=1, max_value=86400)  # Max 24 hours


def validate_price_range(min_price: Optional[float], max_price: Optional[float]) -> None:
    """Validate price range parameters.

    Args:
        min_price: Minimum price (optional)
        max_price: Maximum price (optional)

    Raises:
        ValidationError: If price range is invalid
    """
    if min_price is not None and min_price < 0:
        raise ValidationError("minimum price cannot be negative", "min_price")

    if max_price is not None and max_price < 0:
        raise ValidationError("maximum price cannot be negative", "max_price")

    if min_price is not None and max_price is not None and min_price > max_price:
        raise ValidationError("minimum price cannot be greater than maximum price", "price_range")


def validate_filter_lists(
    categories: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    exclude_items: Optional[List[str]] = None,
    include_items: Optional[List[str]] = None,
) -> None:
    """Validate filter list parameters.

    Args:
        categories: List of categories to filter by
        brands: List of brands to filter by
        exclude_items: List of item IDs to exclude
        include_items: List of item IDs to include

    Raises:
        ValidationError: If any filter list is invalid
    """
    # Validate categories
    if categories is not None:
        if len(categories) > 50:
            raise ValidationError("cannot specify more than 50 categories", "categories")
        for category in categories:
            if not category or not category.strip():
                raise ValidationError("category names cannot be empty", "categories")

    # Validate brands
    if brands is not None:
        if len(brands) > 50:
            raise ValidationError("cannot specify more than 50 brands", "brands")
        for brand in brands:
            if not brand or not brand.strip():
                raise ValidationError("brand names cannot be empty", "brands")

    # Validate exclude_items
    if exclude_items is not None:
        if len(exclude_items) > 1000:
            raise ValidationError("cannot exclude more than 1000 items", "exclude_items")
        for item_id in exclude_items:
            validate_item_id(item_id)

    # Validate include_items
    if include_items is not None:
        if len(include_items) > 1000:
            raise ValidationError("cannot include more than 1000 items", "include_items")
        for item_id in include_items:
            validate_item_id(item_id)

    # Check for conflicts
    if exclude_items and include_items:
        overlapping = set(exclude_items) & set(include_items)
        if overlapping:
            raise ValidationError(
                f"items cannot be both included and excluded: {list(overlapping)[:5]}", "filter_conflict"
            )


def validate_request_parameters(
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
    """Validate all request parameters and return cleaned values.

    Args:
        user_id: User ID to validate
        item_id: Item ID to validate
        k: Number of recommendations to validate
        page: Page number to validate
        page_size: Page size to validate
        min_price: Minimum price to validate
        max_price: Maximum price to validate
        categories: Categories list to validate
        brands: Brands list to validate
        exclude_items: Exclude items list to validate
        include_items: Include items list to validate

    Returns:
        Dictionary of validated parameters

    Raises:
        ValidationError: If any parameter is invalid
    """
    validated = {}

    # Validate individual parameters
    if user_id is not None:
        validated["user_id"] = validate_user_id(user_id)

    if item_id is not None:
        validated["item_id"] = validate_item_id(item_id)

    if k is not None:
        validated["k"] = validate_k_parameter(k)

    if page is not None:
        validated["page"] = validate_page_parameter(page)

    if page_size is not None:
        validated["page_size"] = validate_page_size_parameter(page_size)

    # Validate price range
    validate_price_range(min_price, max_price)
    if min_price is not None:
        validated["min_price"] = min_price
    if max_price is not None:
        validated["max_price"] = max_price

    # Validate filter lists
    validate_filter_lists(categories, brands, exclude_items, include_items)
    if categories is not None:
        validated["categories"] = categories
    if brands is not None:
        validated["brands"] = brands
    if exclude_items is not None:
        validated["exclude_items"] = exclude_items
    if include_items is not None:
        validated["include_items"] = include_items

    return validated
