"""Simplified test script for Update Worker."""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from recsys_lite.update.worker import UpdateWorker

# Add the project's src directory to the Python path
project_dir = Path(__file__).parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))


# Define the classes we need for testing
class MockModel:
    """Mock model for testing."""

    def __init__(self):
        """Initialize mock model."""
        self.user_factors = np.random.random((10, 5))
        self.item_factors = np.random.random((20, 5))

    def partial_fit_users(self, user_item_matrix, user_ids):
        """Mock implementation of partial_fit_users."""
        print(f"Called partial_fit_users with {user_item_matrix.shape[0]} users")
        return True

    def get_item_factors(self):
        """Mock implementation of get_item_factors."""
        return self.item_factors


class MockFaissIndex:
    """Mock Faiss index for testing."""

    def __init__(self):
        """Initialize mock index."""
        self.d = 5  # Dimensionality
        self.vectors = []

    def add(self, vectors):
        """Mock implementation of add."""
        print(f"Called add with {vectors.shape[0]} vectors")
        self.vectors.extend(vectors)
        return True


# Initialize the UpdateWorker class we imported

print("Testing UpdateWorker initialization...")

# Create a simple user-item matrix
user_item_matrix = sp.csr_matrix((5, 10))
user_ids = np.array(["user1", "user2", "user3", "user4", "user5"])
item_id_map = {i: f"item{i}" for i in range(10)}

# Initialize worker components
model = MockModel()
faiss_index = MockFaissIndex()

# Create a temporary worker
worker = UpdateWorker(
    db_path=Path("test_data/test.db"),
    model=model,
    faiss_index=faiss_index,
    item_id_map=item_id_map,
    interval=1,  # 1 second interval for testing
)

print("Worker initialized successfully!")

# Test the _update_item_vectors method
new_items = ["new_item1", "new_item2", "new_item3"]
print(f"\nTesting _update_item_vectors with {len(new_items)} new items...")
worker._update_item_vectors(new_items)

# Test the _update_user_factors method
user_ids = np.array([0, 1, 2])
matrix = sp.csr_matrix((3, 5))
print(f"\nTesting _update_user_factors with {len(user_ids)} users...")
worker._update_user_factors(matrix, user_ids)

print("\nAll methods tested successfully!")
