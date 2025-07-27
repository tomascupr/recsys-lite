"""Error handling utilities for recommendation API."""

from typing import Any, Dict

from recsys_lite.api.errors import ItemNotFoundError, ModelNotInitializedError, UserNotFoundError, VectorRetrievalError
from recsys_lite.api.state import APIState
from recsys_lite.utils.logging import LogLevel, get_logger, log_exception

logger = get_logger("api.error_handling")


def handle_recommendation_errors(
    error: Exception,
    state: APIState,
    context_type: str,
    context_data: Dict[str, Any],
) -> None:
    """Handle errors that occur during recommendation generation.

    Args:
        error: The exception that occurred
        state: API state for metrics
        context_type: Type of operation (recommend/similar_items)
        context_data: Data about the operation for logging

    Raises:
        The original exception after logging and metrics update
    """
    # Known errors - re-raise to be handled by exception handlers
    if isinstance(error, (UserNotFoundError, ModelNotInitializedError, VectorRetrievalError, ItemNotFoundError)):
        raise error

    # Unexpected errors - log and update metrics
    log_exception(logger, f"Error generating {context_type}", error, level=LogLevel.ERROR, extra=context_data)
    state.increase_error_count()
    raise error
