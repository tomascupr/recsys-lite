"""Worker for incremental model updates."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.api.validation import validate_batch_size, validate_interval
from recsys_lite.utils.logging import get_logger

# Create module-level logger
logger = get_logger(__name__)


class UpdateWorker:
    """Worker responsible for incremental model updates."""

    def __init__(
        self,
        db_path: Path,
        model: Any,
        faiss_index: faiss.Index,
        user_mapping: Dict[str, int],
        item_mapping: Dict[str, int],
        model_dir: Path,
        item_id_map: Optional[Dict[int, str]] = None,
        batch_size: int = 1000,
        interval: int = 60,
        incremental_dir: Optional[Path] = None,
    ) -> None:
        """Initialise update worker.

        Args:
            db_path: Path to DuckDB database.
            model: Recommendation model with ``partial_fit_users`` support.
            faiss_index: Faiss index for similarity search.
            user_mapping: Mapping from external user IDs to model indices.
            item_mapping: Mapping from item IDs to model indices.
            model_dir: Directory containing persisted model artifacts.
            item_id_map: Existing mapping from Faiss index positions to item IDs.
            batch_size: Maximum number of events to process per batch.
            interval: Update interval in seconds.
            incremental_dir: Directory containing incremental parquet event files.
        """
        batch_size = validate_batch_size(batch_size)
        interval = validate_interval(interval)

        self.db_path = Path(db_path)
        self.model = model
        self.faiss_index = faiss_index
        self.model_dir = Path(model_dir)
        self.user_mapping: Dict[str, int] = {str(k): int(v) for k, v in user_mapping.items()}
        self.reverse_user_mapping: Dict[int, str] = {int(v): str(k) for k, v in self.user_mapping.items()}
        self.item_mapping: Dict[str, int] = {str(k): int(v) for k, v in item_mapping.items()}
        self.item_id_map: Dict[int, str] = (
            {int(k): str(v) for k, v in item_id_map.items()} if item_id_map is not None else {int(v): str(k) for k, v in item_mapping.items()}
        )
        self.batch_size = batch_size
        self.interval = interval
        self.last_timestamp = 0
        self.num_items = len(self.item_mapping)
        self.incremental_dir = Path(incremental_dir) if incremental_dir else self.db_path.parent / "incremental"

    def run(self) -> None:
        """Run the update loop, applying new events at the configured interval."""
        while True:
            try:
                user_item_matrix, user_ids, new_items = self._get_new_events()

                if user_ids.size > 0:
                    self._update_user_factors(user_item_matrix, user_ids)

                if new_items:
                    self._update_item_vectors(new_items)

                time.sleep(self.interval)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Error in update worker", exc_info=True, extra={"error": str(exc)})
                time.sleep(self.interval)

    def _get_new_events(self) -> Tuple[sp.csr_matrix, NDArray[np.int64], List[str]]:
        """Fetch new events and transform them into a sparse interaction matrix."""
        conn = duckdb.connect(str(self.db_path))
        try:
            events_df = conn.execute(
                f"""
                SELECT user_id, item_id, qty, ts
                FROM events
                WHERE ts > {self.last_timestamp}
                ORDER BY ts
                LIMIT {self.batch_size}
                """
            ).fetchdf()

            frames: List[pd.DataFrame] = [events_df]

            if self.incremental_dir.exists():
                for parquet_file in sorted(self.incremental_dir.glob("*.parquet")):
                    file_modified_time = parquet_file.stat().st_mtime
                    if file_modified_time <= self.last_timestamp:
                        continue

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
                            frames.append(new_events)
                    except Exception as exc:
                        logger.error(
                            "Error reading incremental parquet file",
                            exc_info=True,
                            extra={"file": str(parquet_file), "error": str(exc)},
                        )

            if len(frames) > 1:
                events_df = pd.concat(frames, ignore_index=True)

            if not events_df.empty:
                self.last_timestamp = events_df["ts"].max()
        finally:
            conn.close()

        if events_df.empty:
            empty_matrix = sp.csr_matrix((0, self.num_items), dtype=np.float32)
            return empty_matrix, np.array([], dtype=np.int64), []

        existing_items = set(self.item_id_map.values())
        new_items: List[str] = []
        row_lookup: Dict[int, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for _, event in events_df.iterrows():
            user_id = str(event["user_id"])
            item_id = str(event["item_id"])
            qty = float(event["qty"])

            item_index = self.item_mapping.get(item_id)
            if item_index is None:
                if item_id not in new_items:
                    new_items.append(item_id)
                continue

            if item_id not in existing_items and item_id not in new_items:
                new_items.append(item_id)

            user_index = self._ensure_user_index(user_id)
            row_idx = row_lookup.setdefault(user_index, len(row_lookup))

            rows.append(row_idx)
            cols.append(item_index)
            data.append(qty)

        if not rows:
            empty_matrix = sp.csr_matrix((0, self.num_items), dtype=np.float32)
            return empty_matrix, np.array([], dtype=np.int64), new_items

        user_item_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(row_lookup), self.num_items), dtype=np.float32)
        user_ids = np.fromiter(row_lookup.keys(), dtype=np.int64)

        return user_item_matrix, user_ids, new_items

    def _update_user_factors(self, user_item_matrix: sp.csr_matrix, user_ids: NDArray[np.int64]) -> None:
        """Update the model's user factors using new interactions."""
        self.model.partial_fit_users(user_item_matrix, user_ids)

    def _update_item_vectors(self, new_items: List[str]) -> None:
        """Add vectors for new items to the Faiss index and mapping."""
        if not new_items:
            return

        vectors_to_add: List[np.ndarray] = []
        added_item_ids: List[str] = []
        skipped_items: List[str] = []

        for item_id in new_items:
            item_key = str(item_id)
            item_index = self.item_mapping.get(item_key)
            if item_index is None:
                skipped_items.append(item_key)
                continue

            vector = self._get_item_vector(item_index)
            if vector is None or np.asarray(vector).size == 0:
                skipped_items.append(item_key)
                continue

            vector_array = np.asarray(vector, dtype=np.float32)
            if vector_array.ndim == 1:
                vector_array = vector_array.reshape(1, -1)

            vectors_to_add.append(vector_array)
            added_item_ids.append(item_key)

        if vectors_to_add:
            stacked = np.vstack(vectors_to_add).astype(np.float32)
            try:
                self.faiss_index.add(stacked)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Error updating item vectors", exc_info=True, extra={"error": str(exc)})
                return

            start_idx = len(self.item_id_map)
            for offset, item_id in enumerate(added_item_ids):
                self.item_id_map[start_idx + offset] = item_id

            logger.info(
                "Successfully added new items to Faiss index",
                extra={"count": len(added_item_ids)},
            )

        if skipped_items:
            logger.warning(
                "Skipped items without available vectors",
                extra={"items": skipped_items},
            )

    def _ensure_user_index(self, user_id: str) -> int:
        """Return the model index for ``user_id``, adding it if necessary."""
        if user_id in self.user_mapping:
            return self.user_mapping[user_id]

        new_index = len(self.user_mapping)
        self.user_mapping[user_id] = new_index
        self.reverse_user_mapping[new_index] = user_id
        self._persist_user_mapping()
        return new_index

    def _persist_user_mapping(self) -> None:
        """Persist the current user mapping to disk."""
        mapping_path = self.model_dir / "user_mapping.json"
        try:
            with mapping_path.open("w", encoding="utf-8") as fp:
                json.dump(self.user_mapping, fp)
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("Failed to persist user mapping", exc_info=True, extra={"error": str(exc)})

    def _get_item_vector(self, item_index: int) -> Optional[np.ndarray]:
        """Retrieve an item vector from the underlying model."""
        if hasattr(self.model, "get_item_vectors"):
            try:
                vectors = self.model.get_item_vectors([item_index])
                if isinstance(vectors, np.ndarray) and vectors.size:
                    return vectors[0]
            except Exception as exc:  # pragma: no cover - warning logged below
                logger.warning(
                    "get_item_vectors call failed",
                    extra={"item_index": item_index, "error": str(exc)},
                )

        if hasattr(self.model, "get_item_factors"):
            try:
                factors = self.model.get_item_factors()
                if factors is not None and item_index < len(factors):
                    return factors[item_index]
            except Exception as exc:  # pragma: no cover - warning logged below
                logger.warning(
                    "get_item_factors call failed",
                    extra={"item_index": item_index, "error": str(exc)},
                )

        return None
