"""Mock version of LightFM model to use in CI where LightFM fails to build."""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.models.base import BaseRecommender, ModelRegistry


class LightFMModel(BaseRecommender):
    """Mock hybrid matrix factorization model for CI environment."""

    model_type = "lightfm"

    def __init__(
        self,
        no_components: int = 100,
        learning_rate: float = 0.05,
        loss: str = "warp",
        item_alpha: float = 0.0,
        user_alpha: float = 0.0,
        epochs: int = 50,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize mock LightFM model.

        Args:
            no_components: Number of latent factors
            learning_rate: Learning rate for SGD
            loss: Loss function to use ('warp', 'bpr', 'warp-kos', 'logistic')
            item_alpha: L2 penalty on item features
            user_alpha: L2 penalty on user features
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.loss = loss
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.epochs = epochs
        self.random_state = random_state
        self.user_biases: Optional[NDArray[np.float64]] = None
        self.item_biases: Optional[NDArray[np.float64]] = None
        self.user_embeddings: Optional[NDArray[np.float32]] = None
        self.item_embeddings: Optional[NDArray[np.float32]] = None
        self.user_features: Optional[sp.csr_matrix] = None
        self.item_features: Optional[sp.csr_matrix] = None
        self._user_mapping: Optional[Dict[Union[int, str], int]] = None
        self._item_mapping: Optional[Dict[Union[int, str], int]] = None

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the mock model on user-item interaction data.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        # Mock implementation - just create random embeddings
        n_users, n_items = user_item_matrix.shape
        self.user_biases = np.zeros(n_users, dtype=np.float64)
        self.item_biases = np.zeros(n_items, dtype=np.float64)
        self.user_embeddings = np.random.rand(n_users, self.no_components).astype(np.float32)
        self.item_embeddings = np.random.rand(n_items, self.no_components).astype(np.float32)
        self._user_mapping = kwargs.get("user_mapping", self._user_mapping)
        self._item_mapping = kwargs.get("item_mapping", self._item_mapping)

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            user_items: Sparse user-item interaction matrix
            n_items: Number of recommendations to return
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (item_ids, scores)
        """
        user_idx = self._resolve_user_index(user_id, kwargs.get("user_mapping"))

        # Get number of items
        if self.item_embeddings is None:
            raise ValueError("Model has not been trained, item_embeddings is None")
        n_items_total = self.item_embeddings.shape[0]

        filter_items = kwargs.get("filter_items")
        if filter_items is None and user_items is not None:
            if not sp.isspmatrix_csr(user_items):
                user_items = sp.csr_matrix(user_items)
            if user_items.shape[0] == 1:
                filter_items = user_items.indices
            elif user_idx < user_items.shape[0]:
                filter_items = user_items.getrow(user_idx).indices
            else:
                filter_items = np.array([], dtype=np.int64)
        if filter_items is None:
            filter_items = np.array([], dtype=np.int64)

        # Just return random recommendations
        rng_seed = (self.random_state or 0) + int(user_idx)
        rng = np.random.RandomState(rng_seed)
        candidate_indices = rng.permutation(n_items_total)
        mask = ~np.isin(candidate_indices, filter_items)
        candidate_indices = candidate_indices[mask][:n_items]
        top_items = candidate_indices.astype(np.int_)
        top_scores = rng.random(len(top_items)).astype(np.float32)

        return top_items, top_scores

    def get_item_factors(self) -> NDArray[np.float32]:
        """Get item factors matrix.

        Returns:
            Item factors matrix
        """
        if self.item_embeddings is None:
            return np.array([], dtype=np.float32)  # Return empty array if not initialized
        return np.asarray(self.item_embeddings, dtype=np.float32)

    def predict(self, user_ids: NDArray[np.int_], item_ids: NDArray[np.int_]) -> NDArray[np.float32]:
        """Predict scores for the given user-item pairs.

        Args:
            user_ids: User IDs
            item_ids: Item IDs

        Returns:
            Array of scores
        """
        if self.user_embeddings is None or self.item_embeddings is None:
            raise ValueError("Model has not been trained")

        # Generate random scores for the given user-item pairs
        scores = np.random.random(len(user_ids)).astype(np.float32)
        return scores

    def get_item_representations(self) -> Tuple[NDArray[np.float64], NDArray[np.float32]]:
        """Get item biases and embeddings.

        Returns:
            Tuple of (item_biases, item_embeddings)
        """
        if self.item_biases is None or self.item_embeddings is None:
            raise ValueError("Model has not been trained")

        return self.item_biases, self.item_embeddings

    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        os.makedirs(path, exist_ok=True)

        # Save model parameters
        model_state = {
            "no_components": self.no_components,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "item_alpha": self.item_alpha,
            "user_alpha": self.user_alpha,
            "epochs": self.epochs,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "user_embeddings": self.user_embeddings,
            "item_embeddings": self.item_embeddings,
        }

        with open(os.path.join(path, "lightfm_model.pkl"), "wb") as f:
            pickle.dump(model_state, f)

    def load_model(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        # Load model parameters
        with open(os.path.join(path, "lightfm_model.pkl"), "rb") as f:
            model_state = pickle.load(f)

        # Set model attributes
        self.no_components = model_state["no_components"]
        self.learning_rate = model_state["learning_rate"]
        self.loss = model_state["loss"]
        self.item_alpha = model_state["item_alpha"]
        self.user_alpha = model_state["user_alpha"]
        self.epochs = model_state["epochs"]
        self.user_biases = model_state["user_biases"]
        self.item_biases = model_state["item_biases"]
        self.user_embeddings = model_state["user_embeddings"]
        self.item_embeddings = model_state["item_embeddings"]

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "no_components": self.no_components,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "item_alpha": self.item_alpha,
            "user_alpha": self.user_alpha,
            "epochs": self.epochs,
            "random_state": self.random_state,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "user_embeddings": self.user_embeddings,
            "item_embeddings": self.item_embeddings,
        }

    def _resolve_user_index(
        self, user_id: Union[int, str], override_mapping: Optional[Dict[Union[int, str], int]]
    ) -> int:
        mapping = override_mapping or self._user_mapping
        if mapping is not None:
            if user_id in mapping:
                return int(mapping[user_id])
            raise KeyError(f"User ID {user_id!r} not present in mapping")

        if isinstance(user_id, str):
            if user_id.isdigit():
                return int(user_id)
            raise ValueError("user_id must be convertible to int when no mapping is supplied")

        return int(user_id)

    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data."""
        self.no_components = model_state.get("no_components", 100)
        self.learning_rate = model_state.get("learning_rate", 0.05)
        self.loss = model_state.get("loss", "warp")
        self.item_alpha = model_state.get("item_alpha", 0.0)
        self.user_alpha = model_state.get("user_alpha", 0.0)
        self.epochs = model_state.get("epochs", 50)
        self.random_state = model_state.get("random_state")
        self.user_biases = model_state.get("user_biases")
        self.item_biases = model_state.get("item_biases")
        self.user_embeddings = model_state.get("user_embeddings")
        self.item_embeddings = model_state.get("item_embeddings")

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> NDArray[np.float32]:
        """Get item vectors for specified items."""
        if self.item_embeddings is None:
            return np.array([], dtype=np.float32)

        # This is just a mock, so we'll return random vectors
        # Just generate random vectors of the correct shape regardless of input
        return np.random.rand(len(item_ids), self.no_components).astype(np.float32)


# Register the model
ModelRegistry.register("lightfm", LightFMModel)
