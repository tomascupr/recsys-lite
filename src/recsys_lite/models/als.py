"""ALS model implementation using implicit library."""

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import implicit
import numpy as np
import scipy.sparse as sp

from recsys_lite.models.base import BaseRecommender, FactorizationModelMixin


class ALSModel(BaseRecommender, FactorizationModelMixin):
    """Alternating Least Squares model for collaborative filtering."""

    model_type = "als"

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        alpha: float = 1.0,
        iterations: int = 15,
    ) -> None:
        """Initialize ALS model.

        Args:
            factors: Number of latent factors
            regularization: Regularization factor
            alpha: Confidence scaling parameter
            iterations: Number of ALS iterations
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            calculate_training_loss=True,
            num_threads=0,  # Use all available cores
        )
        self.user_factors = None
        self.item_factors = None
        self._user_mapping: Optional[Mapping[Union[int, str], int]] = None

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the ALS model.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        if not sp.isspmatrix_csr(user_item_matrix):
            user_item_matrix = user_item_matrix.tocsr()

        user_mapping = kwargs.get("user_mapping")
        if user_mapping is not None:
            self._user_mapping = user_mapping

        self.model.fit(user_item_matrix)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        implicit_user_idx, implicit_user_items = self._prepare_user_matrix(user_idx, user_items)

        recommendations = self.model.recommend(
            userid=implicit_user_idx,
            user_items=implicit_user_items,
            N=n_items,
            filter_already_liked_items=True,
        )

        if isinstance(recommendations, tuple):
            item_ids, scores = recommendations
        else:
            item_ids = np.array([item_id for item_id, _ in recommendations])
            scores = np.array([score for _, score in recommendations])

        return item_ids, scores

    def partial_fit_users(self, user_item_matrix: sp.csr_matrix, user_ids: np.ndarray) -> None:
        """Update user factors for specified users.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_ids: IDs of users to update
        """
        if not sp.isspmatrix_csr(user_item_matrix):
            user_item_matrix = user_item_matrix.tocsr()

        user_ids_array = np.asarray(user_ids, dtype=np.int32)
        if user_ids_array.ndim != 1:
            raise ValueError("user_ids must be a 1-D array of user indices")

        user_subset = user_item_matrix[user_ids_array]
        self.model.partial_fit_users(user_ids_array, user_subset)
        self.user_factors = self.model.user_factors

    def partial_fit_items(self, user_item_matrix: sp.csr_matrix, item_ids: np.ndarray) -> None:
        """Update item factors for specified items."""

        if not sp.isspmatrix_csr(user_item_matrix):
            user_item_matrix = user_item_matrix.tocsr()

        item_ids_array = np.asarray(item_ids, dtype=np.int32)
        if item_ids_array.ndim != 1:
            raise ValueError("item_ids must be a 1-D array of item indices")

        item_subset = user_item_matrix[:, item_ids_array].T.tocsr()
        self.model.partial_fit_items(item_ids_array, item_subset)
        self.item_factors = self.model.item_factors

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization.

        Returns:
            Dictionary with model state
        """
        return {
            "factors": self.model.factors,
            "regularization": self.model.regularization,
            "alpha": self.model.alpha,
            "iterations": self.model.iterations,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_mapping": self._user_mapping,
        }

    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data.

        Args:
            model_state: Dictionary with model state
        """
        # Set model parameters
        self.model.factors = model_state["factors"]
        self.model.regularization = model_state["regularization"]
        self.model.alpha = model_state["alpha"]
        self.model.iterations = model_state["iterations"]

        # Set factors
        self.user_factors = model_state["user_factors"]
        self.item_factors = model_state["item_factors"]

        # Ensure model has these values too
        self.model.user_factors = self.user_factors
        self.model.item_factors = self.item_factors
        self._user_mapping = model_state.get("user_mapping")

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

    @staticmethod
    def _prepare_user_matrix(user_idx: int, user_items: sp.csr_matrix) -> Tuple[int, sp.csr_matrix]:
        if not sp.isspmatrix_csr(user_items):
            user_items = sp.csr_matrix(user_items)

        if user_items.shape[0] == 1:
            return 0, user_items

        if user_idx >= user_items.shape[0]:
            raise ValueError("user_idx is out of bounds for the provided user_items matrix")

        return 0, user_items.getrow(user_idx)
