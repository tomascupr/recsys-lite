"""Tests for recommendation models."""

import os

import numpy as np
import pytest
import scipy.sparse as sp

from recsys_lite.models import ALSModel, BPRModel, Item2VecModel, LightFMModel

# Skip tests that require heavy dependencies in CI environment
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(is_ci, reason="Tests don't run in CI environment due to dependency issues")


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
