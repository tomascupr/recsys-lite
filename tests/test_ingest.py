"""Tests for data ingestion module."""

import os
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest

# Skip tests in CI environment due to dependency issues
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(is_ci, reason="Tests don't run in CI environment due to dependency issues")

from recsys_lite.ingest import ingest_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample events data
        events_data = pd.DataFrame(
            {
                "user_id": ["U_01", "U_01", "U_02", "U_03"],
                "item_id": ["I_01", "I_02", "I_01", "I_03"],
                "ts": [1617235200, 1617235300, 1617235400, 1617235500],
                "qty": [1, 2, 1, 3],
            }
        )

        # Create sample items data
        items_data = pd.DataFrame(
            {
                "item_id": ["I_01", "I_02", "I_03"],
                "category": ["electronics", "books", "clothing"],
                "brand": ["brand1", "brand2", "brand3"],
                "price": [100.0, 20.0, 50.0],
                "img_url": [
                    "http://example.com/1.jpg",
                    "http://example.com/2.jpg",
                    "http://example.com/3.jpg",
                ],
            }
        )

        # Save data to temporary files
        events_path = Path(temp_dir) / "events.parquet"
        items_path = Path(temp_dir) / "items.csv"

        events_data.to_parquet(events_path)
        items_data.to_csv(items_path, index=False)

        # Create temporary database path
        db_path = Path(temp_dir) / "test.db"

        yield events_path, items_path, db_path


def test_ingest_data(sample_data):
    """Test data ingestion functionality."""
    events_path, items_path, db_path = sample_data

    # Ingest data
    ingest_data(events_path, items_path, db_path)

    # Check if database was created
    assert db_path.exists()

    # Connect to database and check tables
    conn = duckdb.connect(str(db_path))

    # Check events table
    events_df = conn.execute("SELECT * FROM events").fetchdf()
    assert len(events_df) == 4
    assert set(events_df.columns) == {"user_id", "item_id", "ts", "qty"}

    # Check items table
    items_df = conn.execute("SELECT * FROM items").fetchdf()
    assert len(items_df) == 3
    assert set(items_df.columns) == {"item_id", "category", "brand", "price", "img_url"}

    # Check data integrity
    user_counts = conn.execute("SELECT user_id, COUNT(*) FROM events GROUP BY user_id").fetchdf()
    assert len(user_counts) == 3
    assert user_counts[user_counts["user_id"] == "U_01"]["count_star()"].iloc[0] == 2

    # Close connection
    conn.close()
