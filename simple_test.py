"""
Simple test script for the recommendation system.

This script creates a simple model and runs a recommendation test directly.
"""

import json
import pickle
from pathlib import Path

import numpy as np

# Directory to save model artifacts
MODEL_DIR = Path("model_artifacts/als")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Creating simple test model...")

# Create simple user and item data
n_users = 10
n_items = 20
n_factors = 16

# Create random user and item factors
user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
item_factors = np.random.normal(0, 0.1, (n_items, n_factors))

# Create mappings
user_ids = [f"U_{str(i).zfill(2)}" for i in range(n_users)]
item_ids = [f"I_{str(i).zfill(4)}" for i in range(n_items)]

user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}

idx_to_user = {str(i): user_id for i, user_id in enumerate(user_ids)}
idx_to_item = {str(i): item_id for i, item_id in enumerate(item_ids)}

# Save mappings
with open(MODEL_DIR / "user_mapping.json", "w") as f:
    json.dump({"user_to_idx": user_to_idx, "idx_to_user": idx_to_user}, f)

with open(MODEL_DIR / "item_mapping.json", "w") as f:
    json.dump({"item_to_idx": item_to_idx, "idx_to_item": idx_to_item}, f)

# Save a simple model that uses the factors
model_data = {
    "factors": n_factors,
    "regularization": 0.01,
    "alpha": 1.0,
    "iterations": 20,
    "user_factors": user_factors,
    "item_factors": item_factors,
}

with open(MODEL_DIR / "als_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Save a simple Faiss index
faiss_data = {
    "vectors": item_factors,
    "ids": item_ids,
    "dim": n_factors,
    "id_to_index": item_to_idx,
    "index_to_id": idx_to_item,
}

with open(MODEL_DIR / "faiss_index.pkl", "wb") as f:
    pickle.dump(faiss_data, f)

print(f"Model artifacts saved to {MODEL_DIR}")
print("Done!")
