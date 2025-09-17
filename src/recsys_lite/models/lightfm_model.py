"""LightFM model implementation for hybrid matrix factorization."""

import os
import pickle
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from lightfm import LightFM

from recsys_lite.models.base import BaseRecommender, FloatArray, IntArray


class LightFMModel(BaseRecommender):
    """Hybrid matrix factorization model using LightFM."""

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
        """Initialize LightFM model.

        Args:
            no_components: Number of latent factors
            learning_rate: Learning rate for SGD
            loss: Loss function to use ('warp', 'bpr', 'warp-kos', 'logistic')
            item_alpha: L2 penalty on item features
            user_alpha: L2 penalty on user features
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.model = LightFM(
            no_components=no_components,
            learning_rate=learning_rate,
            loss=loss,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            random_state=random_state,
        )
        self.epochs = epochs
        self.user_biases = None
        self.item_biases = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_features = None
        self.item_features = None
        self._user_mapping: Optional[Mapping[Union[int, str], int]] = None
        self._item_mapping: Optional[Mapping[Union[int, str], int]] = None

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the LightFM model.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        # Get optional feature matrices from kwargs
        user_features = kwargs.get("user_features", None)
        item_features = kwargs.get("item_features", None)

        # Store features for later use
        self.user_features = user_features
        self.item_features = item_features
        self._user_mapping = kwargs.get("user_mapping", self._user_mapping)
        self._item_mapping = kwargs.get("item_mapping", self._item_mapping)

        self.model.fit(
            interactions=user_item_matrix,
            user_features=user_features,
            item_features=item_features,
            epochs=self.epochs,
            verbose=True,
        )

        # Store model parameters
        self.user_biases = self.model.user_biases
        self.item_biases = self.model.item_biases
        self.user_embeddings = self.model.user_embeddings
        self.item_embeddings = self.model.item_embeddings

    def predict(
        self,
        user_ids: IntArray,
        item_ids: IntArray,
        user_features: Optional[sp.csr_matrix] = None,
        item_features: Optional[sp.csr_matrix] = None,
    ) -> FloatArray:
        """Predict scores for user-item pairs.

        Args:
            user_ids: User IDs
            item_ids: Item IDs
            user_features: Sparse user features matrix
            item_features: Sparse item features matrix

        Returns:
            Array of prediction scores
        """
        result = self.model.predict(
            user_ids=user_ids,
            item_ids=item_ids,
            user_features=user_features,
            item_features=item_features,
        )
        # Ensure result is a numpy array to satisfy type checker
        return np.asarray(result)

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[IntArray, FloatArray]:
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

        # Get user and item features from kwargs or use stored ones
        user_features = kwargs.get("user_features", self.user_features)
        item_features = kwargs.get("item_features", self.item_features)

        # Get filter_items from kwargs or create from user_items
        filter_items = kwargs.get("filter_items", None)
        if filter_items is None and user_items is not None:
            # Get indices of items the user has already interacted with
            if not sp.isspmatrix_csr(user_items):
                user_items = sp.csr_matrix(user_items)
            if user_items.shape[0] == 1:
                filter_items = user_items.indices
            elif user_idx < user_items.shape[0]:
                filter_items = user_items.getrow(user_idx).indices

        # Get number of items
        if self.item_embeddings is None:
            raise ValueError("Model has not been trained, item_embeddings is None")
        else:
            n_items_total = self.item_embeddings.shape[0]

        # Get predictions for all items
        scores = self.predict(
            user_ids=np.full(n_items_total, user_idx, dtype=np.int32),
            item_ids=np.arange(n_items_total),
            user_features=user_features,
            item_features=item_features,
        )

        # Filter items if needed
        if filter_items is not None:
            scores[filter_items] = -np.inf

        # Get top N items
        top_items = np.argsort(-scores)[:n_items]
        top_scores = scores[top_items]

        valid_mask = np.isfinite(top_scores)
        if not np.any(valid_mask):
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        top_items = top_items[valid_mask]
        top_scores = top_scores[valid_mask]

        return top_items.astype(np.int_), top_scores.astype(np.float32)

    def get_item_factors(self) -> FloatArray:
        """Get item factors matrix.

        Returns:
            Item factors matrix
        """
        if self.model.item_embeddings is None:
            return np.array([], dtype=np.float32)  # Return empty array if not initialized
        return np.asarray(self.model.item_embeddings, dtype=np.float32)

    def get_item_representations(self, item_features: Optional[sp.csr_matrix] = None) -> Tuple[FloatArray, FloatArray]:
        """Get item biases and embeddings.

        Args:
            item_features: Sparse item features matrix

        Returns:
            Tuple of (item_biases, item_embeddings)
        """
        if item_features is not None:
            return (
                np.asarray(self.model.item_biases, dtype=np.float32),
                np.asarray(self.model.item_embeddings @ item_features.T, dtype=np.float32),
            )
        return (
            np.asarray(self.model.item_biases, dtype=np.float32),
            np.asarray(self.model.item_embeddings, dtype=np.float32),
        )

    def get_user_representations(self, user_features: Optional[sp.csr_matrix] = None) -> Tuple[FloatArray, FloatArray]:
        """Get user biases and embeddings.

        Args:
            user_features: Sparse user features matrix

        Returns:
            Tuple of (user_biases, user_embeddings)
        """
        if user_features is not None:
            return (
                np.asarray(self.model.user_biases, dtype=np.float32),
                np.asarray(self.model.user_embeddings @ user_features.T, dtype=np.float32),
            )
        return (
            np.asarray(self.model.user_biases, dtype=np.float32),
            np.asarray(self.model.user_embeddings, dtype=np.float32),
        )

    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        os.makedirs(path, exist_ok=True)

        # Save model parameters
        model_state = {
            "no_components": self.model.no_components,
            "learning_rate": self.model.learning_rate,
            "loss": self.model.loss,
            "item_alpha": self.model.item_alpha,
            "user_alpha": self.model.user_alpha,
            "epochs": self.epochs,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "user_embeddings": self.user_embeddings,
            "item_embeddings": self.item_embeddings,
            "user_mapping": dict(self._user_mapping) if self._user_mapping is not None else None,
            "item_mapping": dict(self._item_mapping) if self._item_mapping is not None else None,
        }

        with open(os.path.join(path, "lightfm_model.pkl"), "wb") as f:
            pickle.dump(model_state, f)

        # Save features if available
        user_features_path = os.path.join(path, "user_features.pkl")
        if self.user_features is not None:
            with open(user_features_path, "wb") as f:
                pickle.dump(self.user_features, f)

        item_features_path = os.path.join(path, "item_features.pkl")
        if self.item_features is not None:
            with open(item_features_path, "wb") as f:
                pickle.dump(self.item_features, f)

    def load_model(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        # Load model parameters
        with open(os.path.join(path, "lightfm_model.pkl"), "rb") as f:
            model_state = pickle.load(f)

        # Create a new model with the saved parameters
        self.model = LightFM(
            no_components=model_state["no_components"],
            learning_rate=model_state["learning_rate"],
            loss=model_state["loss"],
            item_alpha=model_state["item_alpha"],
            user_alpha=model_state["user_alpha"],
        )

        # Set model attributes
        self.epochs = model_state["epochs"]
        self.user_biases = model_state["user_biases"]
        self.item_biases = model_state["item_biases"]
        self.user_embeddings = model_state["user_embeddings"]
        self.item_embeddings = model_state["item_embeddings"]

        # Set model state
        self.model.user_biases = self.user_biases
        self.model.item_biases = self.item_biases
        self.model.user_embeddings = self.user_embeddings
        self.model.item_embeddings = self.item_embeddings

        # Load features if available
        try:
            with open(os.path.join(path, "user_features.pkl"), "rb") as f:
                self.user_features = pickle.load(f)
        except FileNotFoundError:
            self.user_features = None

        try:
            with open(os.path.join(path, "item_features.pkl"), "rb") as f:
                self.item_features = pickle.load(f)
        except FileNotFoundError:
            self.item_features = None

        self._user_mapping = model_state.get("user_mapping")
        self._item_mapping = model_state.get("item_mapping")

    def _resolve_user_index(
        self,
        user_id: Union[int, str],
        override_mapping: Optional[Mapping[Union[int, str], int]] = None,
    ) -> int:
        mapping = override_mapping or self._user_mapping
        if mapping is not None:
            try:
                return int(mapping[user_id])
            except KeyError as exc:
                raise KeyError(f"User ID {user_id!r} not present in mapping") from exc

        if isinstance(user_id, str):
            if user_id.isdigit():
                return int(user_id)
            raise ValueError("user_id must be convertible to int when no mapping is supplied")

        return int(user_id)
