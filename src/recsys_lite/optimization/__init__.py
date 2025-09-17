"""Hyperparameter optimization module for RecSys-Lite."""

from recsys_lite.optimization.metrics import hr_at_k, ndcg_at_k

try:  # pragma: no cover - optional dependency
    from recsys_lite.optimization.optimizer import OptunaOptimizer
except ImportError:  # pragma: no cover - fallback when optuna missing

    class OptunaOptimizer:  # type: ignore[misc]
        """Placeholder optimizer that raises if Optuna is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise ImportError(
                "Optuna is not installed. Install the 'optuna' extra to use OptunaOptimizer."
            )


__all__ = ["OptunaOptimizer", "hr_at_k", "ndcg_at_k"]
