"""Error handling for RecSys-Lite API."""

from typing import Optional, TypedDict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from recsys_lite.utils.logging import LogLevel, get_logger, log_exception

logger = get_logger("api")


class RecSysError(Exception):
    """Base exception for RecSys-Lite API errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred"

    def __init__(self, detail: Optional[str] = None):
        """Initialize exception.

        Args:
            detail: Error detail message
        """
        if detail:
            self.detail = detail
        super().__init__(self.detail)


class NotFoundError(RecSysError):
    """Error raised when a resource is not found."""

    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found"


class UserNotFoundError(NotFoundError):
    """Error raised when a user is not found."""

    detail = "User not found"

    def __init__(self, user_id: Optional[str] = None):
        """Initialize exception.

        Args:
            user_id: User ID that was not found
        """
        detail = f"User {user_id} not found" if user_id else self.detail
        super().__init__(detail)


class ItemNotFoundError(NotFoundError):
    """Error raised when an item is not found."""

    detail = "Item not found"

    def __init__(self, item_id: Optional[str] = None):
        """Initialize exception.

        Args:
            item_id: Item ID that was not found
        """
        detail = f"Item {item_id} not found" if item_id else self.detail
        super().__init__(detail)


class ServiceUnavailableError(RecSysError):
    """Error raised when a service is unavailable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Service unavailable"


class ModelNotInitializedError(ServiceUnavailableError):
    """Error raised when the model is not initialized."""

    detail = "Recommender system not initialized"


class VectorError(RecSysError):
    """Base class for vector-related errors."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Vector operation failed"


class VectorRetrievalError(VectorError):
    """Error raised when vector retrieval fails."""

    detail = "Failed to retrieve vector"

    def __init__(self, entity_type: str, entity_id: Optional[str] = None, reason: Optional[str] = None):
        """Initialize exception.

        Args:
            entity_type: Type of entity (user or item)
            entity_id: Entity ID that caused the error
            reason: Reason for the failure
        """
        detail_parts = [f"Failed to retrieve {entity_type} vector"]

        if entity_id:
            detail_parts.append(f"for {entity_id}")

        if reason:
            detail_parts.append(f": {reason}")

        self.detail = " ".join(detail_parts)
        super().__init__(self.detail)


def add_error_handlers(app: FastAPI) -> None:
    """Add error handlers to FastAPI app.

    Args:
        app: FastAPI application
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle validation errors.

        Args:
            request: FastAPI request
            exc: Validation exception

        Returns:
            JSON response with error details
        """
        log_exception(
            logger,
            "Validation error",
            exc,
            level=LogLevel.WARNING,
            extra={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RecSysError)
    async def recsys_exception_handler(request: Request, exc: RecSysError) -> JSONResponse:
        """Handle RecSys-Lite API errors.

        Args:
            request: FastAPI request
            exc: RecSys-Lite API exception

        Returns:
            JSON response with error details
        """
        # Use appropriate log level based on status code
        level = LogLevel.ERROR
        if exc.status_code < 500:
            level = LogLevel.WARNING

        log_exception(
            logger,
            "API error",
            exc,
            level=level,
            extra={"path": request.url.path, "method": request.method, "status_code": exc.status_code},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions.

        Args:
            request: FastAPI request
            exc: Exception

        Returns:
            JSON response with error details
        """
        log_exception(
            logger,
            "Unhandled exception",
            exc,
            level=LogLevel.ERROR,
            extra={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred"},
        )


class ErrorResponse(TypedDict):
    """Type definition for error response."""

    status_code: int
    detail: str
