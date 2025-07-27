"""
Script to generate model artifacts from sample data.
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

# Add src to path so we can import recsys_lite
sys.path.insert(0, str(Path(__file__).parent))

# Fix imports with explicit paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
from recsys_lite.indexing.faiss_index import FaissIndexBuilder
from recsys_lite.models.als import ALSModel


def main():
    print("Generating model artifacts from sample data")

    # Define paths
    sample_data_dir = Path("data/sample_data")
    model_dir = Path("model_artifacts/als")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load sample data
    events_df = pd.read_parquet(sample_data_dir / "events.parquet")
    items_df = pd.read_csv(sample_data_dir / "items.csv")

    # Create user and item mappings
    user_ids = events_df["user_id"].unique()
    item_ids = items_df["item_id"].unique()

    user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}

    idx_to_user = {i: user_id for user_id, i in user_to_idx.items()}
    idx_to_item = {i: item_id for item_id, i in item_to_idx.items()}

    # Save mappings
    with open(model_dir / "user_mapping.json", "w") as f:
        json.dump({"user_to_idx": user_to_idx, "idx_to_user": idx_to_user}, f)

    with open(model_dir / "item_mapping.json", "w") as f:
        json.dump({"item_to_idx": item_to_idx, "idx_to_item": idx_to_item}, f)

    # Create interaction matrix
    rows = []
    cols = []
    data = []

    for _, row in events_df.iterrows():
        user_idx = user_to_idx.get(row["user_id"])
        item_idx = item_to_idx.get(row["item_id"])

        if user_idx is not None and item_idx is not None:
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(row["qty"])

    n_users = len(user_ids)
    n_items = len(item_ids)

    interactions = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    # Train ALS model
    print("Training ALS model...")
    model = ALSModel(factors=64, regularization=0.01, alpha=1.0, iterations=20)
    model.fit(interactions)

    # Save model
    model.save_model(str(model_dir))

    # Create FAISS index
    print("Creating FAISS index...")
    item_factors = model.item_factors

    # Create index builder with item factors
    index_builder = FaissIndexBuilder(
        vectors=item_factors,
        index_type="Flat",  # Use simple Flat index for small dataset
        metric="inner_product",
    )

    # Save index
    index_builder_filename = "faiss_index.pkl"
    with open(model_dir / index_builder_filename, "wb") as f:
        pickle.dump(index_builder, f)

    print(f"Model artifacts saved to {model_dir}")
    print("Done")


if __name__ == "__main__":
    main()
