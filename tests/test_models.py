"""Tests for recommendation models."""

import contextlib
import json
import os
import sys
import types
from typing import Optional

import numpy as np
import pytest
import scipy.sparse as sp

from recsys_lite.models import (
    ALSModel,
    BPRModel,
    EASEModel,
    Item2VecModel,
    LightFMModel,
    TextEmbeddingModel,
)

# Skip tests that require heavy dependencies in CI environment
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(is_ci, reason="Tests don't run in CI environment due to dependency issues")


def _install_text_embedding_stubs(monkeypatch, embedding_dim: int = 8) -> None:
    """Install lightweight torch and sentence-transformers stubs for testing."""

    class DummyTorch:
        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        cuda = _Cuda()

        @staticmethod
        def no_grad():
            return contextlib.nullcontext()

    class DummySentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self._dim = embedding_dim
            self.normalize_embeddings = True

        def get_sentence_embedding_dimension(self) -> int:
            return self._dim

        def to(self, device: str) -> "DummySentenceTransformer":
            return self

        def encode(
            self,
            texts: list[str],
            convert_to_numpy: bool = True,
            show_progress_bar: bool = False,
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
        ) -> np.ndarray:
            if not texts:
                return np.empty((0, self._dim), dtype=np.float32)
            base = np.arange(len(texts) * self._dim, dtype=np.float32).reshape(len(texts), self._dim)
            if normalize_embeddings:
                norms = np.linalg.norm(base, axis=1, keepdims=True)
                base = base / np.maximum(norms, 1e-12)
            return base

    monkeypatch.setitem(sys.modules, "torch", DummyTorch())
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=DummySentenceTransformer),
    )


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a small user-item matrix
    n_users = 10
    n_items = 20

    # Create interaction matrix with some interactions
    rng = np.random.RandomState(42)
    interactions = sp.lil_matrix((n_users, n_items), dtype=np.float32)

    # Add some interactions (about 10% density)
    for _ in range(20):
        user = rng.randint(0, n_users)
        item = rng.randint(0, n_items)
        interactions[user, item] = 1.0

    # Convert to CSR for efficient operations
    interactions = interactions.tocsr()

    # Create sessions for Item2Vec
    sessions = []
    for user in range(n_users):
        items = interactions[user].nonzero()[1].tolist()
        if items:
            # Convert to strings to match expected format
            items = [f"I_{item}" for item in items]
            sessions.append(items)

    return interactions, sessions


def test_als_model(sample_data):
    """Test ALS model."""
    interactions, _ = sample_data

    # Initialize model
    model = ALSModel(factors=10, regularization=0.01, alpha=1.0, iterations=5)

    # Fit model
    model.fit(interactions)

    # Check that factors were learned
    assert model.user_factors is not None
    assert model.item_factors is not None

    # Check dimensions
    assert model.user_factors.shape[0] == interactions.shape[0]
    assert model.item_factors.shape[0] == interactions.shape[1]

    # Test get_item_factors
    item_factors = model.get_item_factors()
    assert item_factors is not None
    assert item_factors.shape == (interactions.shape[1], 10)


def test_als_partial_fit(sample_data):
    """ALS partial_fit users/items update only targeted factors."""
    interactions, _ = sample_data
    model = ALSModel(factors=8, regularization=0.01, alpha=1.0, iterations=5)
    model.fit(interactions)

    original_user_factors = model.user_factors.copy()
    original_item_factors = model.item_factors.copy()

    updated_matrix = interactions.tolil()
    updated_matrix[0, :] = 0.0
    updated_matrix[0, 0] = 5.0
    updated_matrix = updated_matrix.tocsr()

    model.partial_fit_users(updated_matrix, np.array([0]))
    assert not np.allclose(model.user_factors[0], original_user_factors[0], atol=1e-6)
    np.testing.assert_allclose(model.user_factors[1], original_user_factors[1], rtol=1e-6, atol=1e-6)

    updated_items = interactions.tolil()
    updated_items[:, 0] = 0.0
    updated_items[0, 0] = 7.0
    updated_items = updated_items.tocsr()

    model.partial_fit_items(updated_items, np.array([0]))
    assert not np.allclose(model.item_factors[0], original_item_factors[0], atol=1e-6)
    np.testing.assert_allclose(model.item_factors[1], original_item_factors[1], rtol=1e-6, atol=1e-6)


def test_als_recommend_single_row(sample_data):
    """ALS recommend works with either full matrix or single-row input."""
    interactions, _ = sample_data
    model = ALSModel(factors=10, regularization=0.01, alpha=1.0, iterations=5)
    model.fit(interactions)

    user_idx = 3
    rec_full, score_full = model.recommend(user_idx, interactions, n_items=5)

    single_row = interactions[user_idx].reshape(1, -1)
    rec_single, score_single = model.recommend(user_idx, single_row, n_items=5)

    assert len(rec_full) == len(rec_single)
    np.testing.assert_allclose(score_full[: len(score_single)], score_single, rtol=1e-5, atol=1e-5)


def test_als_recommend_with_mapping(sample_data):
    """ALS recommend can resolve external string IDs via mapping."""
    interactions, _ = sample_data
    model = ALSModel(factors=10, regularization=0.01, alpha=1.0, iterations=5)
    user_mapping = {f"user-{i}": i for i in range(interactions.shape[0])}
    model.fit(interactions, user_mapping=user_mapping)

    user_id = "user-2"
    recs, scores = model.recommend(user_id, interactions, n_items=5)

    assert len(recs) == len(scores)
    assert all(0 <= rec < interactions.shape[1] for rec in recs)


def test_ease_model(sample_data):
    """Test EASE-R model."""
    interactions, _ = sample_data

    model = EASEModel(lambda_=0.5, topk=5)
    model.fit(interactions)

    assert model.item_weights is not None
    assert model.num_items == interactions.shape[1]

    # Recommendations should return valid indices and finite scores
    recs, scores = model.recommend(0, interactions, n_items=5)
    assert recs.shape == scores.shape
    assert len(recs) <= 5
    assert all(np.isfinite(scores))
    assert all(0 <= rec < interactions.shape[1] for rec in recs)

    # Persistence round-trip
    state = model._get_model_state()
    restored = EASEModel(lambda_=state["lambda_"])
    restored._set_model_state(state)
    recs2, scores2 = restored.recommend(0, interactions, n_items=5)
    np.testing.assert_allclose(scores[: len(scores2)], scores2, rtol=1e-5)


def test_bpr_model(sample_data):
    """Test BPR model."""
    interactions, _ = sample_data

    # Initialize model
    model = BPRModel(factors=10, learning_rate=0.01, regularization=0.01, iterations=5)

    # Fit model
    model.fit(interactions)

    # Check that factors were learned
    assert model.user_factors is not None
    assert model.item_factors is not None

    # Check dimensions
    assert model.user_factors.shape[0] == interactions.shape[0]
    assert model.item_factors.shape[0] == interactions.shape[1]

    # Test recommendations
    for user_id in range(min(3, interactions.shape[0])):
        recs, scores = model.recommend(user_id, interactions, n_items=5)

        # Check that correct number of recommendations was returned
        assert len(recs) == 5
        assert len(scores) == 5

        # Check that recommendations are valid item indices
        assert all(0 <= rec < interactions.shape[1] for rec in recs)

        # Check that scores are in descending order
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_item2vec_model(sample_data):
    """Test Item2Vec model."""
    _, sessions = sample_data

    if not sessions:
        pytest.skip("No sessions available for testing")

    # Initialize model
    model = Item2VecModel(vector_size=10, window=2, min_count=1, sg=1, epochs=5)

    # Ensure sessions are properly formatted for Item2Vec
    formatted_sessions = []
    for session in sessions:
        # Convert to strings since Item2Vec works with string IDs
        formatted_sessions.append([str(item) for item in session])

    # Fit model
    if formatted_sessions:
        model.fit(formatted_sessions)

    # Check that item vectors were learned
    assert model.item_vectors is not None

    # Check that vectors have correct dimensions
    for _item, vector in model.item_vectors.items():
        assert vector.shape == (10,)

    # Test get_item_vectors
    item_ids = list(model.item_vectors.keys())
    item_vectors = model.get_item_vectors(item_ids)
    assert item_vectors is not None
    assert isinstance(item_vectors, np.ndarray)

    # Test get_item_vectors_matrix
    item_ids = list(model.item_vectors.keys())[:3] if len(model.item_vectors) >= 3 else list(model.item_vectors.keys())
    matrix = model.get_item_vectors_matrix(item_ids)
    assert matrix.shape == (len(item_ids), 10)


def test_lightfm_model(sample_data):
    """Test LightFM model."""
    interactions, _ = sample_data

    # Initialize model
    model = LightFMModel(no_components=10, learning_rate=0.05, loss="warp", epochs=5)

    # Fit model
    model.fit(interactions)

    # Check that model attributes are set
    assert model.user_biases is not None
    assert model.item_biases is not None
    assert model.user_embeddings is not None
    assert model.item_embeddings is not None

    # Test predict
    user_ids = np.array([0])
    item_ids = np.array([0])
    scores = model.predict(user_ids, item_ids)

    assert scores.shape == (1,)

    # Test get_item_representations
    item_biases, item_embeddings = model.get_item_representations()

    assert item_biases.shape == (interactions.shape[1],)
    assert item_embeddings.shape == (interactions.shape[1], 10)


def test_lightfm_recommend_single_row(sample_data):
    """LightFM recommend works with single-row matrices."""
    interactions, _ = sample_data
    model = LightFMModel(no_components=8, epochs=10)
    model.fit(interactions)

    user_idx = 2
    full_items, full_scores = model.recommend(user_idx, interactions, n_items=5)
    single_row = interactions[user_idx].reshape(1, -1)
    row_items, row_scores = model.recommend(user_idx, single_row, n_items=5)

    assert len(full_items) == len(row_items)
    np.testing.assert_allclose(full_scores[: len(row_scores)], row_scores, rtol=1e-5, atol=1e-5)


def test_lightfm_recommend_with_mapping(sample_data):
    """LightFM recommend resolves external IDs via mapping."""
    interactions, _ = sample_data
    model = LightFMModel(no_components=8, epochs=10)
    user_mapping = {f"u-{i}": i for i in range(interactions.shape[0])}
    model.fit(interactions, user_mapping=user_mapping)

    items, scores = model.recommend("u-4", interactions, n_items=5)

    assert len(items) == len(scores)
    assert all(0 <= item < interactions.shape[1] for item in items)


def test_text_embedding_loads_cached_embeddings(tmp_path, monkeypatch):
    """TextEmbeddingModel loads cached embeddings when available and valid."""
    _install_text_embedding_stubs(monkeypatch, embedding_dim=4)

    cache_dir = tmp_path
    embeddings = np.ones((2, 4), dtype=np.float32)
    np.save(cache_dir / "text_embeddings.npy", embeddings)
    (cache_dir / "item_ids.json").write_text(json.dumps(["item-1", "item-2"]))

    model = TextEmbeddingModel(embedding_dim=4, cache_embeddings=True)
    model.fit(user_item_matrix=sp.csr_matrix((0, 0)), item_data={}, output_dir=cache_dir)

    np.testing.assert_allclose(model.item_embeddings, embeddings)
    assert model.item_ids == ["item-1", "item-2"]
    assert model.embedding_dim == 4


def test_text_embedding_regenerates_invalid_cache(tmp_path, monkeypatch):
    """Invalid cache triggers regeneration of embeddings."""
    _install_text_embedding_stubs(monkeypatch, embedding_dim=4)

    cache_dir = tmp_path
    np.save(cache_dir / "text_embeddings.npy", np.ones((1, 4), dtype=np.float32))
    (cache_dir / "item_ids.json").write_text(json.dumps(["item-1", "item-2"]))

    item_data = {
        "item-1": {"title": "Red Shirt", "description": "Comfortable cotton"},
        "item-2": {"title": "Blue Jeans", "description": "Slim fit"},
    }

    model = TextEmbeddingModel(embedding_dim=4, cache_embeddings=True)
    model.fit(user_item_matrix=sp.csr_matrix((0, 0)), item_data=item_data, output_dir=cache_dir)

    assert model.item_embeddings.shape == (2, 4)
    assert model.item_ids == ["item-1", "item-2"]
    cached = np.load(cache_dir / "text_embeddings.npy", allow_pickle=False)
    assert cached.shape == (2, 4)
