"""Tests for input validation utilities."""

import pytest
from fastapi import HTTPException

from recsys_lite.api.validation import (
    ValidationError,
    validate_batch_size,
    validate_entity_index,
    validate_filter_lists,
    validate_interval,
    validate_item_id,
    validate_k_parameter,
    validate_page_parameter,
    validate_page_size_parameter,
    validate_positive_integer,
    validate_price_range,
    validate_request_parameters,
    validate_user_id,
)


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_without_field(self):
        """Test ValidationError without field name."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("test error message")

        assert exc_info.value.status_code == 400
        assert "Validation error: test error message" in str(exc_info.value.detail)

    def test_validation_error_with_field(self):
        """Test ValidationError with field name."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("must be positive", "test_field")

        assert exc_info.value.status_code == 400
        assert "Validation error in field 'test_field': must be positive" in str(exc_info.value.detail)


class TestPositiveIntegerValidation:
    """Test positive integer validation."""

    def test_valid_positive_integer(self):
        """Test valid positive integer."""
        result = validate_positive_integer(10, "test_field")
        assert result == 10

    def test_invalid_zero(self):
        """Test zero value is invalid."""
        with pytest.raises(ValidationError):
            validate_positive_integer(0, "test_field")

    def test_invalid_negative(self):
        """Test negative value is invalid."""
        with pytest.raises(ValidationError):
            validate_positive_integer(-5, "test_field")

    def test_max_value_constraint(self):
        """Test maximum value constraint."""
        with pytest.raises(ValidationError):
            validate_positive_integer(15, "test_field", max_value=10)

    def test_custom_min_value(self):
        """Test custom minimum value."""
        result = validate_positive_integer(5, "test_field", min_value=5)
        assert result == 5

        with pytest.raises(ValidationError):
            validate_positive_integer(4, "test_field", min_value=5)


class TestKParameterValidation:
    """Test k parameter validation."""

    def test_valid_k_values(self):
        """Test valid k values."""
        assert validate_k_parameter(1) == 1
        assert validate_k_parameter(10) == 10
        assert validate_k_parameter(1000) == 1000

    def test_invalid_k_values(self):
        """Test invalid k values."""
        with pytest.raises(ValidationError):
            validate_k_parameter(0)

        with pytest.raises(ValidationError):
            validate_k_parameter(-1)

        with pytest.raises(ValidationError):
            validate_k_parameter(1001)


class TestPageParameterValidation:
    """Test page parameter validation."""

    def test_valid_page_values(self):
        """Test valid page values."""
        assert validate_page_parameter(1) == 1
        assert validate_page_parameter(100) == 100

    def test_invalid_page_values(self):
        """Test invalid page values."""
        with pytest.raises(ValidationError):
            validate_page_parameter(0)

        with pytest.raises(ValidationError):
            validate_page_parameter(-1)


class TestPageSizeParameterValidation:
    """Test page_size parameter validation."""

    def test_valid_page_size_values(self):
        """Test valid page_size values."""
        assert validate_page_size_parameter(1) == 1
        assert validate_page_size_parameter(50) == 50
        assert validate_page_size_parameter(100) == 100

    def test_invalid_page_size_values(self):
        """Test invalid page_size values."""
        with pytest.raises(ValidationError):
            validate_page_size_parameter(0)

        with pytest.raises(ValidationError):
            validate_page_size_parameter(-1)

        with pytest.raises(ValidationError):
            validate_page_size_parameter(101)


class TestUserIdValidation:
    """Test user ID validation."""

    def test_valid_user_ids(self):
        """Test valid user IDs."""
        assert validate_user_id("user123") == "user123"
        assert validate_user_id("  user123  ") == "user123"  # Trimmed
        assert validate_user_id("user-123_abc") == "user-123_abc"

    def test_invalid_user_ids(self):
        """Test invalid user IDs."""
        with pytest.raises(ValidationError):
            validate_user_id("")

        with pytest.raises(ValidationError):
            validate_user_id("   ")

        with pytest.raises(ValidationError):
            validate_user_id("user<script>")

        with pytest.raises(ValidationError):
            validate_user_id("a" * 256)  # Too long


class TestItemIdValidation:
    """Test item ID validation."""

    def test_valid_item_ids(self):
        """Test valid item IDs."""
        assert validate_item_id("item123") == "item123"
        assert validate_item_id("  item123  ") == "item123"  # Trimmed
        assert validate_item_id("item-123_abc") == "item-123_abc"

    def test_invalid_item_ids(self):
        """Test invalid item IDs."""
        with pytest.raises(ValidationError):
            validate_item_id("")

        with pytest.raises(ValidationError):
            validate_item_id("   ")

        with pytest.raises(ValidationError):
            validate_item_id("item<script>")

        with pytest.raises(ValidationError):
            validate_item_id("a" * 256)  # Too long


class TestEntityIndexValidation:
    """Test entity index validation."""

    def test_valid_entity_indices(self):
        """Test valid entity indices."""
        assert validate_entity_index(0, "user") == 0
        assert validate_entity_index(10, "item") == 10

    def test_invalid_negative_index(self):
        """Test invalid negative index."""
        with pytest.raises(ValidationError):
            validate_entity_index(-1, "user")

    def test_invalid_out_of_bounds_index(self):
        """Test invalid out of bounds index."""
        with pytest.raises(ValidationError):
            validate_entity_index(10, "user", max_entities=5)


class TestBatchSizeValidation:
    """Test batch size validation."""

    def test_valid_batch_sizes(self):
        """Test valid batch sizes."""
        assert validate_batch_size(1) == 1
        assert validate_batch_size(1000) == 1000
        assert validate_batch_size(10000) == 10000

    def test_invalid_batch_sizes(self):
        """Test invalid batch sizes."""
        with pytest.raises(ValidationError):
            validate_batch_size(0)

        with pytest.raises(ValidationError):
            validate_batch_size(-1)

        with pytest.raises(ValidationError):
            validate_batch_size(10001)


class TestIntervalValidation:
    """Test interval validation."""

    def test_valid_intervals(self):
        """Test valid intervals."""
        assert validate_interval(1) == 1
        assert validate_interval(3600) == 3600
        assert validate_interval(86400) == 86400

    def test_invalid_intervals(self):
        """Test invalid intervals."""
        with pytest.raises(ValidationError):
            validate_interval(0)

        with pytest.raises(ValidationError):
            validate_interval(-1)

        with pytest.raises(ValidationError):
            validate_interval(86401)  # More than 24 hours


class TestPriceRangeValidation:
    """Test price range validation."""

    def test_valid_price_ranges(self):
        """Test valid price ranges."""
        validate_price_range(None, None)  # No constraints
        validate_price_range(0, 100)
        validate_price_range(10.5, 99.99)
        validate_price_range(0, None)  # Only min
        validate_price_range(None, 100)  # Only max

    def test_invalid_negative_prices(self):
        """Test invalid negative prices."""
        with pytest.raises(ValidationError):
            validate_price_range(-10, 100)

        with pytest.raises(ValidationError):
            validate_price_range(0, -50)

    def test_invalid_min_greater_than_max(self):
        """Test invalid range where min > max."""
        with pytest.raises(ValidationError):
            validate_price_range(100, 50)


class TestFilterListsValidation:
    """Test filter lists validation."""

    def test_valid_filter_lists(self):
        """Test valid filter lists."""
        validate_filter_lists()  # No filters
        validate_filter_lists(categories=["electronics", "books"])
        validate_filter_lists(brands=["nike", "adidas"])
        validate_filter_lists(exclude_items=["item1", "item2"])
        validate_filter_lists(include_items=["item3", "item4"])

    def test_invalid_too_many_categories(self):
        """Test too many categories."""
        with pytest.raises(ValidationError):
            validate_filter_lists(categories=["cat" + str(i) for i in range(51)])

    def test_invalid_too_many_brands(self):
        """Test too many brands."""
        with pytest.raises(ValidationError):
            validate_filter_lists(brands=["brand" + str(i) for i in range(51)])

    def test_invalid_empty_category(self):
        """Test empty category name."""
        with pytest.raises(ValidationError):
            validate_filter_lists(categories=["electronics", ""])

    def test_invalid_conflicting_items(self):
        """Test conflicting include/exclude items."""
        with pytest.raises(ValidationError):
            validate_filter_lists(exclude_items=["item1", "item2"], include_items=["item2", "item3"])

    def test_invalid_too_many_exclude_items(self):
        """Test too many exclude items."""
        with pytest.raises(ValidationError):
            validate_filter_lists(exclude_items=["item" + str(i) for i in range(1001)])


class TestRequestParametersValidation:
    """Test complete request parameters validation."""

    def test_valid_complete_request(self):
        """Test valid complete request."""
        result = validate_request_parameters(
            user_id="user123",
            k=10,
            page=1,
            page_size=20,
            min_price=10.0,
            max_price=100.0,
            categories=["electronics"],
            brands=["nike"],
            exclude_items=["item1"],
            include_items=["item2", "item3"],
        )

        assert result["user_id"] == "user123"
        assert result["k"] == 10
        assert result["page"] == 1
        assert result["page_size"] == 20
        assert result["min_price"] == 10.0
        assert result["max_price"] == 100.0
        assert result["categories"] == ["electronics"]
        assert result["brands"] == ["nike"]
        assert result["exclude_items"] == ["item1"]
        assert result["include_items"] == ["item2", "item3"]

    def test_valid_partial_request(self):
        """Test valid partial request with some parameters."""
        result = validate_request_parameters(user_id="user123", k=5)

        assert result["user_id"] == "user123"
        assert result["k"] == 5
        assert "page" not in result
        assert "categories" not in result

    def test_invalid_request_propagates_error(self):
        """Test that invalid parameters propagate validation errors."""
        with pytest.raises(ValidationError):
            validate_request_parameters(user_id="user<script>")

        with pytest.raises(ValidationError):
            validate_request_parameters(k=0)

        with pytest.raises(ValidationError):
            validate_request_parameters(min_price=-10)
