"""Test script for API functionality."""

import json
import shutil
import sys
from pathlib import Path

# Data and test client imports
import numpy as np
import scipy.sparse as sp
from fastapi.testclient import TestClient

# Set the path to the source directory
src_path = Path(__file__).parent.parent
print(f"Adding {src_path} to Python path")
sys.path.insert(0, str(src_path))


try:
    from src.recsys_lite.api.main import create_app
    from src.recsys_lite.indexing.faiss_index import FaissIndexBuilder
    from src.recsys_lite.models.als import ALSModel

    # (import removed – unused)

    print("Successfully imported project modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    sys.path.insert(0, str(src_path / "src"))
    from recsys_lite.api.main import create_app
    from recsys_lite.indexing.faiss_index import FaissIndexBuilder
    from recsys_lite.models.als import ALSModel

    print("Successfully imported modules with alternative path")

# Create a temporary directory for testing
from recsys_lite.api.dependencies import get_api_state
from recsys_lite.api.loaders import setup_recommendation_service

test_dir = Path("test_data")
# Clear any previous model artifacts
shutil.rmtree(test_dir / "model_artifacts", ignore_errors=True)
test_dir.mkdir(exist_ok=True)

# Prepare model artifacts directory
model_dir = test_dir / "model_artifacts" / "als"
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Creating test model artifacts in {model_dir}")

# Create sample data
users = ["user1", "user2", "user3", "user4", "user5"]
items = ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8"]

# Create user and item mappings
user_mapping = {user_id: i for i, user_id in enumerate(users)}
item_mapping = {item_id: i for i, item_id in enumerate(items)}

# Save mappings
with open(model_dir / "user_mapping.json", "w") as f:
    json.dump(user_mapping, f)

with open(model_dir / "item_mapping.json", "w") as f:
    json.dump(item_mapping, f)

# Create a test user-item matrix
user_item_matrix = sp.lil_matrix((len(user_mapping), len(item_mapping)), dtype=np.float32)

# Add some interactions
for i in range(len(users)):
    for _j in range(3):  # 3 interactions per user
        item_idx = np.random.randint(0, len(items))
        user_item_matrix[i, item_idx] = np.random.randint(1, 5)

user_item_matrix = user_item_matrix.tocsr()

# Create and train a test model
print("Training test model...")
model = ALSModel(factors=10, iterations=5)
model.fit(user_item_matrix)
print("Model trained")

# Save model
model.save_model(str(model_dir))

# Create and save a Faiss index
print("Creating Faiss index...")
item_vectors = model.get_item_factors()
index_builder = FaissIndexBuilder(vectors=item_vectors, ids=list(item_mapping.keys()), index_type="Flat")
index_builder.save(str(model_dir / "faiss_index"))
print("Faiss index saved")

# Create item data
item_data = {}
for item_id in items:
    item_data[item_id] = {
        "title": f"Test {item_id}",
        "category": "test",
        "brand": "test_brand",
        "price": np.random.randint(10, 100),
        "img_url": f"http://example.com/{item_id}.jpg",
    }

# Save item data
with open(test_dir / "data" / "items.json", "w") as f:
    json.dump(item_data, f)

# Test the API
print("\nTesting API...")
app = create_app(model_dir=model_dir)
# Manually initialize recommendation service (bypass FastAPI startup)
state = get_api_state()
service = setup_recommendation_service(model_dir)
state.recommendation_service = service
state.model_type = service.model_type
state.user_mapping = service.user_mapping
state.item_mapping = service.item_mapping
with TestClient(app) as client:
    # Test health endpoint
    print("Testing /health endpoint...")
    response = client.get("/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test metrics endpoint
    print("\nTesting /metrics endpoint...")
    response = client.get("/metrics")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test recommendation endpoint
    print("\nTesting /recommend endpoint...")
    response = client.get(f"/recommend?user_id={users[0]}&k=3")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Test similar items endpoint
    print("\nTesting /similar-items endpoint...")
    response = client.get(f"/similar-items?item_id={items[0]}&k=3")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    print("\nAPI test complete")
