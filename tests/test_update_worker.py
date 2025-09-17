"""Tests for the incremental update worker."""

import json
from pathlib import Path
from typing import List, Tuple

import duckdb
import numpy as np
import pytest
import scipy.sparse as sp

from recsys_lite.update.worker import UpdateWorker


class StubModel:
    """Simple model stub capturing partial updates."""

    def __init__(self, num_items: int) -> None:
        self.partial_calls: List[Tuple[sp.csr_matrix, np.ndarray]] = []
        # Deterministic item representations
        self.item_factors = np.eye(num_items, dtype=np.float32)

    def partial_fit_users(self, matrix: sp.csr_matrix, user_ids: np.ndarray) -> None:
        self.partial_calls.append((matrix.copy(), user_ids.copy()))

    def get_item_factors(self) -> np.ndarray:
        return self.item_factors

    def get_item_vectors(self, item_indices: List[int]) -> np.ndarray:
        indices = np.array(item_indices, dtype=int)
        return self.item_factors[indices]


class StubFaissIndex:
    """Minimal Faiss index stub that records added vectors."""

    def __init__(self) -> None:
        self.add_calls: List[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        self.add_calls.append(np.array(vectors))


@pytest.fixture()
def worker_setup(tmp_path: Path) -> Tuple[UpdateWorker, Path, StubModel, StubFaissIndex, Path]:
    db_path = tmp_path / "events.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE events (
            user_id VARCHAR,
            item_id VARCHAR,
            qty DOUBLE,
            ts BIGINT
        )
        """
    )
    conn.close()

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    user_map = {"user1": 0}
    item_map = {"itemA": 0, "itemB": 1}
    (model_dir / "user_mapping.json").write_text(json.dumps(user_map))
    (model_dir / "item_mapping.json").write_text(json.dumps(item_map))

    stub_model = StubModel(num_items=len(item_map))
    stub_index = StubFaissIndex()

    worker = UpdateWorker(
        db_path=db_path,
        model=stub_model,
        faiss_index=stub_index,
        user_mapping=user_map,
        item_mapping=item_map,
        model_dir=model_dir,
        item_id_map={0: "itemA"},
        batch_size=100,
        interval=1,
    )

    return worker, db_path, stub_model, stub_index, model_dir


def _insert_event(db_path: Path, user_id: str, item_id: str, qty: float, ts: int) -> None:
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?)",
        (user_id, item_id, qty, ts),
    )
    conn.close()


def test_update_worker_updates_existing_user(worker_setup: Tuple[UpdateWorker, Path, StubModel, StubFaissIndex, Path]) -> None:
    worker, db_path, stub_model, _stub_index, _model_dir = worker_setup

    _insert_event(db_path, "user1", "itemA", 3.0, 1)

    matrix, user_ids, new_items = worker._get_new_events()

    assert new_items == []
    assert user_ids.tolist() == [0]
    np.testing.assert_array_equal(matrix.toarray(), np.array([[3.0, 0.0]], dtype=np.float32))

    worker._update_user_factors(matrix, user_ids)
    assert len(stub_model.partial_calls) == 1
    call_matrix, call_ids = stub_model.partial_calls[0]
    np.testing.assert_array_equal(call_matrix.toarray(), np.array([[3.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(call_ids, np.array([0], dtype=np.int64))


def test_update_worker_adds_new_user_and_persists(worker_setup: Tuple[UpdateWorker, Path, StubModel, StubFaissIndex, Path]) -> None:
    worker, db_path, _stub_model, _stub_index, model_dir = worker_setup

    _insert_event(db_path, "user2", "itemA", 2.0, 2)

    _matrix, user_ids, _new_items = worker._get_new_events()

    assert user_ids.tolist() == [1]
    assert worker.user_mapping["user2"] == 1

    persisted = json.loads((model_dir / "user_mapping.json").read_text())
    assert "user2" in persisted
    assert persisted["user2"] == 1


def test_update_worker_adds_new_item_vectors(worker_setup: Tuple[UpdateWorker, Path, StubModel, StubFaissIndex, Path]) -> None:
    worker, db_path, _stub_model, stub_index, _model_dir = worker_setup

    _insert_event(db_path, "user1", "itemB", 1.0, 3)

    matrix, user_ids, new_items = worker._get_new_events()

    # The interaction matrix should include the new item column
    np.testing.assert_array_equal(matrix.toarray(), np.array([[0.0, 1.0]], dtype=np.float32))
    assert user_ids.tolist() == [0]
    assert "itemB" in new_items

    worker._update_item_vectors(new_items)

    assert len(stub_index.add_calls) == 1
    added = stub_index.add_calls[0]
    assert added.shape[0] == 1
    assert worker.item_id_map[1] == "itemB"
