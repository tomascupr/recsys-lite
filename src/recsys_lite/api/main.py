"""FastAPI service for RecSys-Lite."""

from pathlib import Path
from typing import Any, Callable, Union, cast

from fastapi import FastAPI, Request, Response

from recsys_lite.api.dependencies import get_api_state
from recsys_lite.api.errors import add_error_handlers
from recsys_lite.api.loaders import setup_recommendation_service
from recsys_lite.api.routers import cache, health, recommendations
from recsys_lite.utils.logging import LogLevel, configure_logging, get_logger

# Configure logging
configure_logging(level=LogLevel.INFO)
logger = get_logger("api")


def create_app(model_dir: Union[str, Path] = "model_artifacts/als") -> FastAPI:
    """Create FastAPI application.

    Args:
        model_dir: Path to model artifacts

    Returns:
        FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="RecSys-Lite API",
        description="Lightweight recommendation system for small e-commerce shops",
        version="0.1.0",
    )

    # Add error handlers
    add_error_handlers(app)

    # Add request middleware to count requests
    @app.middleware("http")
    async def count_requests(request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Count requests middleware.

        Args:
            request: FastAPI request
            call_next: Next middleware function

        Returns:
            Response
        """
        state = get_api_state()
        state.increase_request_count()
        response = await call_next(request)
        # Cast to Response - FastAPI's middleware uses Starlette Response
        return cast(Response, response)

    # Load model artifacts on startup
    @app.on_event("startup")
    async def startup_event() -> None:
        """Load model artifacts on startup."""
        logger.info(f"Loading model artifacts from {model_dir}")
        state = get_api_state()

        try:
            # Set up recommendation service
            rec_service = setup_recommendation_service(Path(model_dir))

            # Store service in app state
            state.recommendation_service = rec_service
            state.model = rec_service.model
            state.model_type = rec_service.model_type
            state.model_version = rec_service.model_version

            # Store model information
            state.user_mapping = rec_service.user_mapping
            state.item_mapping = rec_service.item_mapping
            state.reverse_user_mapping = {int(v): k for k, v in rec_service.user_mapping.items()}
            state.reverse_item_mapping = {int(v): k for k, v in rec_service.item_mapping.items()}

            logger.info(
                f"API initialized with model type: {rec_service.model_type}",
                extra={
                    "model_type": rec_service.model_type,
                    "user_count": len(state.user_mapping),
                    "item_count": len(state.item_mapping),
                },
            )
        except Exception as e:
            from recsys_lite.utils.logging import log_exception

            log_exception(
                logger, "Error loading model artifacts", e, level=LogLevel.ERROR, extra={"model_dir": str(model_dir)}
            )
            # Allow API to start without model for health checks
            # Recommendation endpoints will return appropriate errors

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(recommendations.router, tags=["recommendations"])
    app.include_router(cache.router, tags=["cache"])

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
