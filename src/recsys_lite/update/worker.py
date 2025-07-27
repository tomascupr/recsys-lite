"""Worker for incremental model updates."""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import faiss
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.api.validation import validate_batch_size, validate_interval
from recsys_lite.utils.logging import get_logger

# Create module-level logger
logger = get_logger(__name__)


class UpdateWorker:
    """Worker for incremental model updates."""

    def __init__(
        self,
        db_path: Path,
        model: Any,
        faiss_index: faiss.Index,
        item_id_map: Dict[int, str],
        batch_size: int = 1000,
        interval: int = 60,
    ) -> None:
        """Initialize update worker.

        Args:
            db_path: Path to DuckDB database
            model: Recommendation model with partial_fit_users method
            faiss_index: Faiss index for similar item lookup
            item_id_map: Mapping from index to item ID
            batch_size: Maximum number of events to process per batch
            interval: Update interval in seconds
        """
        # Validate input parameters
        batch_size = validate_batch_size(batch_size)
        interval = validate_interval(interval)

        self.db_path = db_path
        self.model = model
        self.faiss_index = faiss_index
        self.item_id_map = item_id_map
        self.batch_size = batch_size
        self.interval = interval
        self.last_timestamp = 0

    def run(self) -> None:
        """Run the update worker loop."""
        while True:
            try:
                # Get new events
                user_item_matrix, user_ids, new_items = self._get_new_events()

                if user_ids.size > 0:
                    # Update user factors
                    self._update_user_factors(user_item_matrix, user_ids)

                if new_items:
                    # Update item factors and index
                    self._update_item_vectors(new_items)

                # Sleep until next update
                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in update worker: {e}", exc_info=True)
                time.sleep(self.interval)

    def _get_new_events(self) -> Tuple[sp.csr_matrix, NDArray[np.int_], List[str]]:
        """Get new events from the database and incremental parquet files.

        Returns:
            Tuple of user-item matrix, user IDs, and new items
        """
        conn = duckdb.connect(str(self.db_path))
        try:
            # Get events since last timestamp from database
            events_df = conn.execute(
                f"""
                SELECT user_id, item_id, qty, ts
                FROM events
                WHERE ts > {self.last_timestamp}
                ORDER BY ts
                LIMIT {self.batch_size}
                """
            ).fetchdf()

            # Also check for new incremental parquet files in data/incremental directory
            incremental_dir = Path(self.db_path).parent / "incremental"
            if incremental_dir.exists():
                for parquet_file in incremental_dir.glob("*.parquet"):
                    # Only process files that haven't been processed yet
                    file_modified_time = parquet_file.stat().st_mtime
                    if file_modified_time > self.last_timestamp:
                        # Load and append new events
                        try:
                            new_events = conn.execute(
                                f"""
                                SELECT user_id, item_id, qty, ts
                                FROM read_parquet('{parquet_file}')
                                WHERE ts > {self.last_timestamp}
                                ORDER BY ts
                                LIMIT {self.batch_size}
                                """
                            ).fetchdf()

                            if not new_events.empty:
                                # Append to existing events
                                events_df = events_df.append(new_events)
                        except Exception as e:
                            logger.error(f"Error reading incremental parquet file {parquet_file}: {e}", exc_info=True)

            # Update last timestamp
            if not events_df.empty:
                self.last_timestamp = events_df["ts"].max()
        finally:
            conn.close()

        if events_df.empty:
            return sp.csr_matrix((0, 0)), np.array([]), []

        # Extract unique users and items
        unique_users = set(events_df["user_id"])
        unique_items = set(events_df["item_id"])

        # Get existing items in our model to identify new items
        existing_items = set(self.item_id_map.values())
        new_items = list(unique_items - existing_items)

        # Create user and item ID mappings for the sparse matrix
        # This is a simple mapping for the incremental update
        user_mapping = {user_id: i for i, user_id in enumerate(unique_users)}
        item_mapping = {item_id: i for i, item_id in enumerate(unique_items)}

        # Create sparse user-item matrix
        n_users = len(user_mapping)
        n_items = len(item_mapping)

        # Prepare data for sparse matrix
        row_indices = [user_mapping[user] for user in events_df["user_id"]]
        col_indices = [item_mapping[item] for item in events_df["item_id"]]
        data = events_df["qty"].values

        # Create sparse matrix
        user_item_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_users, n_items))

        # Extract user IDs as numpy array (for partial_fit_users)
        user_ids = np.array(list(user_mapping.keys()))

        return user_item_matrix, user_ids, new_items

    def _update_user_factors(self, user_item_matrix: sp.csr_matrix, user_ids: NDArray[np.int_]) -> None:
        """Update user factors for new events.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_ids: IDs of users to update
        """
        self.model.partial_fit_users(user_item_matrix, user_ids)

    def _update_item_vectors(self, new_items: List[str]) -> None:
        """Update item vectors and Faiss index.

        Args:
            new_items: List of new item IDs
        """
        if not new_items:
            return

        logger.info(f"Updating item vectors for {len(new_items)} new items")

        try:
            # Different models use different methods to get item vectors
            if hasattr(self.model, "get_item_vectors_matrix"):
                # For Item2Vec model
                item_vectors = self.model.get_item_vectors_matrix(new_items)
            elif hasattr(self.model, "get_item_factors"):
                # For ALS, BPR models
                item_vectors = self.model.get_item_factors()
                # Need to filter to just the new items - this requires model-specific implementation
                # For simplicity, we're assuming item_vectors is already the right shape
            else:
                logger.warning("Model does not provide a method to get item vectors")
                return

            # Add to Faiss index
            if item_vectors.shape[0] > 0:
                self.faiss_index.add(item_vectors)

                # Update item ID map
                start_idx = len(self.item_id_map)
                for i, item_id in enumerate(new_items):
                    self.item_id_map[start_idx + i] = item_id

                logger.info(f"Successfully added {len(new_items)} new items to the Faiss index")
            else:
                logger.warning("No item vectors available to add")
        except Exception as e:
            logger.error(f"Error updating item vectors: {e}", exc_info=True)
            # Continue with worker instead of crashing
