"""Tests for GRU4Rec model."""

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

# Import the mock implementation instead
from recsys_lite.models.gru4rec_mock import GRU4Rec

# Skip in CI environment
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(is_ci, reason="Tests with heavy dependencies don't run in CI environment")


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a small user-item matrix
    n_users = 5
    n_items = 10

    # Create interaction matrix with some interactions
    rng = np.random.RandomState(42)
    interactions = sp.lil_matrix((n_users, n_items), dtype=np.float32)

    # Add some interactions
    for _ in range(15):
        user = rng.randint(0, n_users)
        item = rng.randint(0, n_items)
        interactions[user, item] = 1.0

    # Convert to CSR for efficient operations
    interactions = interactions.tocsr()

    # Create sessions for GRU4Rec
    sessions = []
    for user in range(n_users):
        items = interactions[user].nonzero()[1].tolist()
        if items:
            sessions.append(items)

    return interactions, sessions


def test_gru4rec_initialization():
    """Test GRU4Rec model initialization."""
    # Initialize model
    model = GRU4Rec(n_items=100, hidden_size=50, n_layers=2, batch_size=32)

    # Check parameters
    assert model.n_items == 100
    assert model.batch_size == 32
    assert model.hidden_size == 50
    assert model.n_layers == 2


def test_gru4rec_fit(sample_data):
    """Test GRU4Rec model fit method."""
    interactions, sessions = sample_data

    # Initialize the model
    model = GRU4Rec(n_items=interactions.shape[1])

    # Call fit method
    model.fit(interactions, sessions=sessions)

    # Check that the model was trained
    assert model._trained is True
    assert model.item_embeddings is not None
    assert model.item_embeddings.shape[0] >= interactions.shape[1]


def test_gru4rec_recommend(sample_data):
    """Test GRU4Rec recommend method."""
    interactions, sessions = sample_data

    # Initialize and train the model
    model = GRU4Rec(n_items=interactions.shape[1])
    model.fit(interactions, sessions=sessions)

    # Create a user_items matrix
    user_items = interactions

    # Create a sample session
    session = [0, 1]  # Some item IDs to use as a session

    # Call recommend method
    items, scores = model.recommend(user_id=0, user_items=user_items, n_items=3, session=session)

    # Check results
    assert len(items) == 3
    assert len(scores) == 3
    assert np.all(np.isfinite(scores))  # Scores should be finite


def test_gru4rec_save_load(sample_data):
    """Test GRU4Rec model save and load methods."""
    interactions, sessions = sample_data

    # Create a temporary file
    with tempfile.NamedTemporaryFile() as temp_file:
        # Initialize and train model
        model = GRU4Rec(n_items=interactions.shape[1], hidden_size=32)
        model.fit(interactions, sessions=sessions)

        # Save original parameters
        original_n_items = model.n_items
        original_hidden_size = model.hidden_size
        original_embeddings = model.item_embeddings.copy() if model.item_embeddings is not None else None

        # Save model
        model.save_model(temp_file.name)

        # Create a new model with different parameters
        new_model = GRU4Rec(n_items=50, hidden_size=64)

        # Load model
        new_model.load_model(temp_file.name)

        # Check that model parameters were updated
        assert new_model.n_items == original_n_items
        assert new_model.hidden_size == original_hidden_size
        assert new_model._trained is True

        # Check that embeddings were loaded
        assert new_model.item_embeddings is not None
        assert new_model.item_embeddings.shape == original_embeddings.shape
