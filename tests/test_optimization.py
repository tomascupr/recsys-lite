"""Tests for hyperparameter optimization."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# Skip tests in CI environment due to dependency issues
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(is_ci, reason="Tests don't run in CI environment due to dependency issues")

from recsys_lite.optimization import OptunaOptimizer


class MockModel:
    """Mock model for testing OptunaOptimizer."""

    def __init__(self, param1=5, param2=0.1):
        """Initialize mock model."""
        self.param1 = param1
        self.param2 = param2
        self.is_fitted = False

        # Mock factors for generating recommendations
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_data):
        """Fit the model."""
        self.is_fitted = True
        n_users, n_items = train_data.shape

        # Create random factors for testing
        rng = np.random.RandomState(42)
        self.user_factors = rng.random((n_users, 10)).astype(np.float32)
        self.item_factors = rng.random((n_items, 10)).astype(np.float32)

    def recommend(self, user_id, user_items, n_items=10):
        """Generate recommendations."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Convert user_id to int if needed
        if isinstance(user_id, str):
            # Handle user IDs with 'U_' prefix by removing the prefix
            if user_id.startswith("U_"):
                user_id = int(user_id[2:])
            else:
                user_id = int(user_id)

        user_vector = self.user_factors[user_id]

        # Calculate scores
        scores = np.dot(user_vector, self.item_factors.T)

        # Filter out already seen items
        seen_items = user_items[user_id].nonzero()[1]
        scores[seen_items] = -np.inf

        # Get top items
        top_indices = np.argsort(-scores)[:n_items]
        top_scores = scores[top_indices]

        return top_indices, top_scores

    def get_item_factors(self):
        """Get item factors."""
        return self.item_factors

    def save_model(self, path):
        """Save model."""
        return True

    def load_model(self, path):
        """Load model."""
        return True


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_users = 20
    n_items = 50
    density = 0.1

    # Create sparse matrix
    rng = np.random.RandomState(42)
    train_data = sp.random(n_users, n_items, density=density, format="csr", random_state=rng)
    valid_data = sp.random(n_users, n_items, density=density / 2, format="csr", random_state=rng)

    # Make sure every user and item has at least one interaction
    for i in range(n_users):
        j = rng.randint(0, n_items)
        train_data[i, j] = 1.0

    for j in range(n_items):
        i = rng.randint(0, n_users)
        train_data[i, j] = 1.0

    # Create mappings
    user_mapping = {f"U_{i}": i for i in range(n_users)}
    item_mapping = {f"I_{i}": i for i in range(n_items)}

    return train_data, valid_data, user_mapping, item_mapping


def test_optuna_optimizer_initialization():
    """Test OptunaOptimizer initialization."""
    optimizer = OptunaOptimizer(
        model_class=MockModel,
        metric="ndcg@10",
        direction="maximize",
        n_trials=5,
    )

    assert optimizer.model_class == MockModel
    assert optimizer.metric_name == "ndcg"
    assert optimizer.k == 10
    assert optimizer.direction == "maximize"
    assert optimizer.n_trials == 5


def test_optuna_optimizer_optimize(sample_data):
    """Test OptunaOptimizer optimization."""
    train_data, valid_data, user_mapping, item_mapping = sample_data

    # Define parameter space
    param_space = {
        "param1": {"type": "int", "low": 5, "high": 20},
        "param2": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
    }

    # Create optimizer
    optimizer = OptunaOptimizer(
        model_class=MockModel,
        metric="hr@10",
        direction="maximize",
        n_trials=3,  # Use few trials for testing
    )

    # Run optimization
    best_params = optimizer.optimize(
        train_data=train_data,
        valid_data=valid_data,
        param_space=param_space,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )

    # Check that best parameters are returned
    assert "param1" in best_params
    assert "param2" in best_params
    assert optimizer.best_value is not None


def test_optuna_optimizer_get_best_model(sample_data):
    """Test getting best model from optimizer."""
    train_data, valid_data, user_mapping, item_mapping = sample_data

    # Create temporary directory for model saving
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "model.pkl"

        # Define parameter space
        param_space = {
            "param1": {"type": "int", "low": 5, "high": 20},
            "param2": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        }

        # Create optimizer
        optimizer = OptunaOptimizer(
            model_class=MockModel,
            metric="ndcg@10",
            direction="maximize",
            n_trials=3,  # Use few trials for testing
        )

        # Run optimization
        optimizer.optimize(
            train_data=train_data,
            valid_data=valid_data,
            param_space=param_space,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )

        # Get best model
        best_model = optimizer.get_best_model(train_data, save_path=save_path)

        # Check that best model is returned
        assert best_model is not None
        assert best_model.is_fitted
        assert best_model.user_factors is not None
        assert best_model.item_factors is not None
