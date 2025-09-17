"""Optuna-based hyperparameter optimization for recommendation models."""

import os
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp

try:  # pragma: no cover - optional dependency
    import optuna
except ImportError:  # pragma: no cover - optional dependency missing
    optuna = None  # type: ignore[assignment]

from recsys_lite.optimization.metrics import hr_at_k, ndcg_at_k


class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna."""

    def __init__(
        self,
        model_class: Any,
        metric: str = "ndcg@20",
        direction: str = "maximize",
        n_trials: int = 20,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            model_class: Recommendation model class to optimize
            metric: Evaluation metric ('hr@k' or 'ndcg@k')
            direction: Optimization direction ('maximize' or 'minimize')
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            study_name: Optuna study name
            storage: Optuna storage URL
            seed: Random seed for reproducibility
        """
        if optuna is None:  # pragma: no cover - executed when dependency missing
            raise ImportError(
                "Optuna is not installed. Install the 'optuna' extra to use OptunaOptimizer."
            )
        self.model_class = model_class

        # Parse metric and k
        metric_parts = metric.split("@")
        self.metric_name = metric_parts[0].lower()
        self.k = int(metric_parts[1]) if len(metric_parts) > 1 else 10

        # Set metric function
        if self.metric_name == "hr":
            self.metric_func = hr_at_k
        elif self.metric_name == "ndcg":
            self.metric_func = ndcg_at_k
        else:
            raise ValueError(f"Unknown metric: {self.metric_name}")

        # Optuna settings
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"{model_class.__name__}_optimization"
        self.storage = storage
        self.seed = seed

        # Optuna study
        self.study = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None

    def optimize(
        self,
        train_data: sp.csr_matrix,
        valid_data: sp.csr_matrix,
        param_space: Dict[str, Any],
        fixed_params: Optional[Dict[str, Any]] = None,
        user_mapping: Optional[Dict[Union[int, str], int]] = None,
        item_mapping: Optional[Dict[Union[int, str], int]] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            train_data: Training data
            valid_data: Validation data
            param_space: Parameter space definition
            fixed_params: Fixed parameters (not optimized)
            user_mapping: Mapping from original user IDs to matrix indices
            item_mapping: Mapping from original item IDs to matrix indices

        Returns:
            Best parameters
        """
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.seed) if self.seed else None,
        )

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {}
            for param_name, param_spec in param_space.items():
                param_type = param_spec["type"]

                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                        step=param_spec.get("step", 1),
                        log=param_spec.get("log", False),
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(  # type: ignore
                        param_name,
                        float(param_spec["low"]),
                        float(param_spec["high"]),
                        step=param_spec.get("step"),
                        log=param_spec.get("log", False),
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_spec["choices"],
                    )

            # Add fixed parameters
            if fixed_params:
                params.update(fixed_params)

            # Create and train model
            model = self.model_class(**params)
            model.fit(train_data)

            # Evaluate model
            score = self._evaluate(model, train_data, valid_data, self.k, user_mapping, item_mapping)

            return score

        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Get best parameters and value
        self.best_params = self.study.best_params
        if fixed_params:
            self.best_params.update(fixed_params)
        self.best_value = float(self.study.best_value)

        return self.best_params

    def _evaluate(
        self,
        model: Any,
        train_data: sp.csr_matrix,
        valid_data: sp.csr_matrix,
        k: int,
        user_mapping: Optional[Dict[Union[int, str], int]] = None,
        item_mapping: Optional[Dict[Union[int, str], int]] = None,
    ) -> float:
        """Evaluate model on validation data.

        Args:
            model: Trained model
            train_data: Training data
            valid_data: Validation data
            k: Number of recommendations to consider
            user_mapping: Mapping from original user IDs to matrix indices
            item_mapping: Mapping from original item IDs to matrix indices

        Returns:
            Evaluation score
        """
        # Get all users with validation data
        valid_users = valid_data.nonzero()[0]

        # Calculate metric for each user
        scores = []
        for user_idx in valid_users:
            # Convert matrix index to original user ID if mapping provided
            if user_mapping:
                reverse_mapping = {v: k for k, v in user_mapping.items()}
                user_id = reverse_mapping.get(user_idx, user_idx)
            else:
                user_id = user_idx

            # Get ground truth items for this user
            ground_truth = valid_data[user_idx].nonzero()[1].tolist()

            # Skip if no validation items
            if not ground_truth:
                continue

            # Get recommendations
            try:
                recs, _ = model.recommend(user_id, train_data, n_items=k)
                recommended_items = recs.tolist()
            except (AttributeError, NotImplementedError):
                # Fallback if model doesn't have recommend method
                scores_array = np.zeros(train_data.shape[1])
                user_items = train_data[user_idx].toarray().flatten()

                # Get item factors
                if hasattr(model, "item_factors") and hasattr(model, "user_factors"):
                    user_factor = model.user_factors[user_idx]
                    scores_array = np.dot(user_factor, model.item_factors.T)

                # Filter out already seen items
                scores_array[user_items > 0] = -np.inf

                # Get top k items
                recommended_items = np.argsort(-scores_array)[:k].tolist()

            # Calculate metric
            score = self.metric_func(ground_truth, recommended_items, k=k)
            scores.append(score)

        # Return average score across all users
        if not scores:
            return 0.0
        # Convert numpy float to Python float for type checker
        return float(np.mean(scores))

    def get_best_model(
        self,
        train_data: sp.csr_matrix,
        save_path: Optional[str] = None,
    ) -> Any:
        """Train a model with the best parameters.

        Args:
            train_data: Training data
            save_path: Path to save model

        Returns:
            Trained model with best parameters
        """
        if not self.best_params:
            raise ValueError("No best parameters available. Run optimize() first.")

        # Create and train model with best parameters
        model = self.model_class(**self.best_params)
        model.fit(train_data)

        # Save model if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save_model(save_path)

        return model

    def get_trials_dataframe(self) -> Any:
        """Get trials as a DataFrame.

        Returns:
            DataFrame with trial information
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        return self.study.trials_dataframe()
