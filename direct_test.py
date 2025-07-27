"""
Direct test script for the recommendation API.
This script loads the model directly without using Docker containers.
"""

import json
import sys
from pathlib import Path

import scipy.sparse as sp

# Add project modules to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recsys_lite.models.als import ALSModel


def main():
    """Test recommendation functionality directly."""
    print("Testing recommendation functionality directly...")

    # Define paths
    model_dir = Path("model_artifacts/als")

    # Load model
    print(f"Loading model from {model_dir}...")
    model = ALSModel()
    model.load_model(str(model_dir))
    print("Model loaded successfully!")

    # Load user mapping
    with open(model_dir / "user_mapping.json", "r") as f:
        user_mapping = json.load(f)

    # Load item mapping
    with open(model_dir / "item_mapping.json", "r") as f:
        item_mapping = json.load(f)

    # Get user index for U_01
    user_to_idx = user_mapping["user_to_idx"]
    idx_to_item = item_mapping["idx_to_item"]

    user_id = "U_01"
    user_idx = user_to_idx.get(user_id)

    if user_idx is None:
        print(f"User {user_id} not found in mapping")
        return

    # Create a dummy user-item matrix with a single row for the user we're testing
    n_items = len(idx_to_item)

    data = []
    rows = []
    cols = []

    # Create a sparse matrix with just the row for the user we're querying
    user_items = sp.csr_matrix((data, (rows, cols)), shape=(1, n_items))

    # Get recommendations
    print(f"Getting recommendations for user {user_id} (index {user_idx})...")
    # ALS model requires user_id to be 0 for a single-row user_items matrix
    item_ids, scores = model.recommend(
        user_id=0,  # Always use 0 for a single-row user_items matrix
        user_items=user_items,
        n_items=5,
    )

    # Convert item indices to item IDs
    item_ids = [idx_to_item[str(idx)] for idx in item_ids]

    # Format results
    results = [{"item_id": item_id, "score": float(score)} for item_id, score in zip(item_ids, scores, strict=False)]

    print("Recommendations:")
    for i, rec in enumerate(results, 1):
        print(f"{i}. {rec['item_id']} (score: {rec['score']:.4f})")

    print("Direct test complete!")


if __name__ == "__main__":
    main()
