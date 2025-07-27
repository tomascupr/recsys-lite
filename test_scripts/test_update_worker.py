"""Test script for Update Worker."""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Set the path to the source directory
src_path = Path(__file__).parent.parent
print(f"Adding {src_path} to Python path")
sys.path.insert(0, str(src_path))

try:
    from src.recsys_lite.indexing.faiss_index import FaissIndexBuilder
    from src.recsys_lite.models.als import ALSModel

    # (import removed – unused)
    from src.recsys_lite.update.worker import UpdateWorker

    print("Successfully imported project modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    sys.path.insert(0, str(src_path / "src"))
    from recsys_lite.indexing.faiss_index import FaissIndexBuilder
    from recsys_lite.models.als import ALSModel
    from recsys_lite.update.worker import UpdateWorker

    print("Successfully imported modules with alternative path")

# Create a temporary directory for testing
test_dir = Path("test_data")
test_dir.mkdir(exist_ok=True)
incremental_dir = test_dir / "incremental"
incremental_dir.mkdir(exist_ok=True)

# Create a small test database
db_path = test_dir / "test.db"

print(f"Creating test database at {db_path}")

# Create sample data
users = ["user1", "user2", "user3", "user4", "user5"]
items = ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8"]
timestamps = list(range(1000, 1100))

# Create initial events data
events_data = []
for _i in range(20):
    user_id = np.random.choice(users)
    item_id = np.random.choice(items)
    timestamp = np.random.choice(timestamps)
    quantity = np.random.randint(1, 5)
    events_data.append((user_id, item_id, timestamp, quantity))

events_df = pd.DataFrame(events_data, columns=["user_id", "item_id", "ts", "qty"])

# Initialize database
conn = duckdb.connect(str(db_path))
conn.execute("DROP TABLE IF EXISTS events")
conn.execute(
    """
    CREATE TABLE events (
        user_id VARCHAR,
        item_id VARCHAR,
        ts INTEGER,
        qty INTEGER
    )
"""
)

# Insert initial data
conn.execute(
    """
    INSERT INTO events SELECT * FROM events_df
"""
)
conn.close()

print("Database initialized with sample data")

# Create a simple test model
model = ALSModel(factors=10, iterations=5)

# Create user-item matrix from events
conn = duckdb.connect(str(db_path))
events_df = conn.execute("SELECT user_id, item_id, qty FROM events").fetchdf()
conn.close()

# Create user and item mappings
user_ids = events_df["user_id"].unique()
item_ids = events_df["item_id"].unique()

user_mapping = {user_id: i for i, user_id in enumerate(user_ids)}
item_mapping = {item_id: i for i, item_id in enumerate(item_ids)}

# Create user-item matrix
user_item_matrix = sp.lil_matrix((len(user_mapping), len(item_mapping)), dtype=np.float32)

for _, row in events_df.iterrows():
    user_idx = user_mapping[row["user_id"]]
    item_idx = item_mapping[row["item_id"]]
    user_item_matrix[user_idx, item_idx] = row["qty"]

user_item_matrix = user_item_matrix.tocsr()

# Train the model
print("Training test model...")
model.fit(user_item_matrix)
print("Model trained")

# Create a Faiss index
print("Creating Faiss index...")
item_vectors = model.get_item_factors()
index_builder = FaissIndexBuilder(vectors=item_vectors, ids=list(item_mapping.keys()), index_type="Flat")
print("Faiss index created")

# Test the UpdateWorker
print("\nTesting UpdateWorker...")

# Create a new incremental events file
print("Creating incremental events data...")
new_events_data = []
for _i in range(5):
    user_id = np.random.choice(users)
    item_id = np.random.choice(["item9", "item10"])  # New items
    timestamp = np.random.choice(range(1100, 1200))  # Newer timestamps
    quantity = np.random.randint(1, 5)
    new_events_data.append((user_id, item_id, timestamp, quantity))

new_events_df = pd.DataFrame(new_events_data, columns=["user_id", "item_id", "ts", "qty"])
new_events_file = incremental_dir / "new_events.parquet"
new_events_df.to_parquet(new_events_file)

print(f"Created incremental events file: {new_events_file}")

# Create reverse item mapping for the worker
item_id_map = {idx: item_id for item_id, idx in item_mapping.items()}

# Create the worker
worker = UpdateWorker(
    db_path=db_path,
    model=model,
    faiss_index=index_builder.index,
    item_id_map=item_id_map,
    interval=1,  # Use a short interval for testing
)

print("Testing _get_new_events method...")
user_item_matrix, user_ids, new_items = worker._get_new_events()

print(f"Retrieved {user_item_matrix.shape[0]} users, {user_item_matrix.shape[1]} items")
print(f"User IDs: {user_ids}")
print(f"New items: {new_items}")

if len(new_items) > 0:
    print("\nTesting _update_item_vectors method...")
    worker._update_item_vectors(new_items)
    print(f"Item ID map now has {len(worker.item_id_map)} items")

print("\nUpdate Worker test complete")
