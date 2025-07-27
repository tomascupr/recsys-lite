"""Helpers for tests to mock dependencies and provide common utilities."""

import json
import os
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import scipy.sparse as sp

# --- Environment helpers ---


def is_ci_environment() -> bool:
    """Check if tests are running in a CI environment."""
    return os.environ.get("CI", "false").lower() == "true"


# Check if dependencies are available
def is_dependency_available(module_name: str) -> bool:
    """Check if a dependency is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# --- Mock classes for dependencies ---


class MockModelType:
    """Mock for ModelType enum."""

    ALS = "als"
    BPR = "bpr"
    ITEM2VEC = "item2vec"
    LIGHTFM = "lightfm"
    GRU4REC = "gru4rec"


class MockMetricType:
    """Mock for MetricType enum."""

    HR_10 = "hr@10"
    HR_20 = "hr@20"
    NDCG_10 = "ndcg@10"
    NDCG_20 = "ndcg@20"


# --- Common test fixtures ---


def create_test_sparse_matrix(
    n_users: int = 10, n_items: int = 20, density: float = 0.1, seed: int = 42
) -> sp.csr_matrix:
    """Create a test sparse matrix with controlled randomness."""
    rng = np.random.RandomState(seed)
    interactions = sp.lil_matrix((n_users, n_items), dtype=np.float32)

    # Add interactions based on density
    n_interactions = int(n_users * n_items * density)
    for _ in range(n_interactions):
        user = rng.randint(0, n_users)
        item = rng.randint(0, n_items)
        interactions[user, item] = rng.randint(1, 5)  # Rating between 1-5

    # Convert to CSR for efficient operations
    return interactions.tocsr()


def create_test_directory_structure(base_dir: Path) -> Dict[str, Path]:
    """Create a test directory structure for model artifacts."""
    # Create required directories
    model_dir = base_dir / "model_artifacts" / "als"
    data_dir = base_dir / "data"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Create user mapping
    user_mapping = {"user1": 0, "user2": 1, "user3": 2}
    with open(model_dir / "user_mapping.json", "w") as f:
        json.dump(user_mapping, f)

    # Create item mapping
    item_mapping = {"item1": 0, "item2": 1, "item3": 2, "item4": 3}
    with open(model_dir / "item_mapping.json", "w") as f:
        json.dump(item_mapping, f)

    # Create dummy items data
    items_data = {
        "item1": {"title": "Item 1", "category": "Category A"},
        "item2": {"title": "Item 2", "category": "Category B"},
        "item3": {"title": "Item 3", "category": "Category A"},
        "item4": {"title": "Item 4", "category": "Category C"},
    }
    with open(data_dir / "items.json", "w") as f:
        json.dump(items_data, f)

    # Create directory for Faiss index
    (model_dir / "faiss_index").mkdir(exist_ok=True)

    # Create a fake pickled model (real models expect a directory where the pickle file will be saved)
    # This matches the structure expected by ALSModel.load_model which joins the path with "als_model.pkl"

    # Return the path structure
    return {
        "model_dir": model_dir,
        "data_dir": data_dir,
    }


def create_mock_model() -> MagicMock:
    """Create a mock model for testing."""
    mock = MagicMock()
    # Set up standard model methods
    mock.fit = MagicMock()
    mock.recommend = MagicMock(return_value=(np.array([1, 2, 3]), np.array([0.9, 0.8, 0.7])))
    mock.save_model = MagicMock()
    mock.load_model = MagicMock()

    # Set up model-specific attributes
    mock.user_factors = np.random.random((10, 5))
    mock.item_factors = np.random.random((20, 5))

    return mock


def create_mock_faiss_index() -> MagicMock:
    """Create a mock Faiss index for testing."""
    mock = MagicMock()
    # Set up search to return meaningful results
    mock.search = MagicMock(
        return_value=(
            np.array([[0.9, 0.8, 0.7]]),  # Distances
            np.array([[1, 2, 3]]),  # Indices
        )
    )
    mock.d = 10  # Dimension of vectors
    mock.add = MagicMock()

    return mock


# --- Context managers for patching ---


def patch_dependencies():
    """Create a context manager that patches all external dependencies.

    Usage:
        with patch_dependencies() as mocks:
            # Test code here
            # Access mocks through mocks dictionary
    """
    patches = []
    mocks = {}

    # Patch model classes
    model_classes = [
        "recsys_lite.models.ALSModel",
        "recsys_lite.models.BPRModel",
        "recsys_lite.models.Item2VecModel",
        "recsys_lite.models.LightFMModel",
        "recsys_lite.models.GRU4Rec",
    ]

    for cls in model_classes:
        mock = create_mock_model()
        patches.append(patch(cls, return_value=mock))
        mocks[cls.split(".")[-1]] = mock

    # Patch FaissIndexBuilder
    faiss_builder_mock = MagicMock()
    faiss_index_mock = create_mock_faiss_index()
    faiss_builder_mock.load.return_value = MagicMock(index=faiss_index_mock)
    patches.append(patch("recsys_lite.indexing.FaissIndexBuilder", faiss_builder_mock))
    mocks["FaissIndexBuilder"] = faiss_builder_mock
    mocks["faiss_index"] = faiss_index_mock

    # Return a combined context manager
    class CombinedContext:
        def __init__(self, patches, mocks):
            self.patches = patches
            self.mocks = mocks

        def __enter__(self):
            for p in self.patches:
                self.mocks[p.target.split(".")[-1]] = p.start()
            return self.mocks

        def __exit__(self, *args):
            for p in self.patches:
                p.stop()

    return CombinedContext(patches, mocks)
