"""Efficient implementation of the EASE-R recommender.

The EASE-R (Embarrassingly Shallow AutoEncoder) algorithm solves a
closed-form ridge regression of the item-item co-occurrence matrix.  It
provides a strong baseline for implicit-feedback recommendation tasks while
remaining CPU friendly.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

from .base import BaseRecommender, FactorizationModelMixin, ModelRegistry


class EASEModel(BaseRecommender, FactorizationModelMixin):
    """Closed-form EASE-R implementation with optional sparsification."""

    model_type = "ease"

    def __init__(self, lambda_: float = 0.5, topk: Optional[int] = 200) -> None:
        """Create a new EASE-R model.

        Args:
            lambda_: Ridge regularization strength.
            topk: Optional sparsification budget per item.  If ``None`` the full
                dense weight matrix is stored.  Values are clipped to ensure no
                more than ``n_items - 1`` neighbours are kept.
        """
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive for EASE-R")

        self.lambda_ = float(lambda_)
        self.topk = topk if topk is None or topk > 0 else None
        self.item_weights: Optional[sp.csr_matrix] = None
        self.num_items: int = 0

    # ------------------------------------------------------------------
    # BaseRecommender API
    # ------------------------------------------------------------------

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the EASE-R model on a user-item interaction matrix."""

        if not sp.isspmatrix_csr(user_item_matrix):
            user_item_matrix = user_item_matrix.tocsr()

        n_items = user_item_matrix.shape[1]
        if n_items == 0:
            raise ValueError("user_item_matrix must have at least one item")

        # Compute Gram matrix (item-item co-occurrence)
        gram = user_item_matrix.T @ user_item_matrix
        gram = gram.toarray().astype(np.float64, copy=False)

        # Add regularisation to the diagonal (lambda_ * I)
        diag_indices = np.arange(n_items)
        gram[diag_indices, diag_indices] += self.lambda_

        # Solve for the inverse via Cholesky for stability
        identity = np.eye(n_items, dtype=np.float64)
        try:
            chol_factor, lower = la.cho_factor(gram, overwrite_a=True, check_finite=False)
            inv_gram = la.cho_solve((chol_factor, lower), identity, check_finite=False)
        except la.LinAlgError:
            # Fall back to a generic solver if Cholesky fails (should be rare once regularised)
            inv_gram = la.solve(gram, identity, assume_a="sym")

        # Compute EASE weights: W = -B / diag(B); set diagonal to 0
        diag = np.diag(inv_gram)
        if np.any(np.isclose(diag, 0.0)):
            raise ValueError("EASE-R encountered zero diagonal entries in inverse gram matrix")

        weights = -inv_gram / diag[np.newaxis, :]
        np.fill_diagonal(weights, 0.0)

        # Optional sparsification to retain top-k contributors per item
        if self.topk is not None:
            k = min(self.topk, max(n_items - 1, 1))
            rows = []
            cols = []
            data = []
            for item_idx in range(n_items):
                column = weights[:, item_idx]
                # Skip the diagonal (already zero)
                if k < len(column):
                    top_idx = np.argpartition(column, -k)[-k:]
                    sorted_top_idx = top_idx[np.argsort(column[top_idx])[::-1]]
                else:
                    sorted_top_idx = np.argsort(column)[::-1]
                for neighbour in sorted_top_idx:
                    value = column[neighbour]
                    if value == 0.0:
                        continue
                    rows.append(neighbour)
                    cols.append(item_idx)
                    data.append(value)
            item_weights = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        else:
            item_weights = sp.csr_matrix(weights.astype(np.float32, copy=False))

        item_weights.eliminate_zeros()

        self.item_weights = item_weights
        self.num_items = n_items

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        """Generate top-N recommendations for a user."""

        if self.item_weights is None:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        if n_items <= 0:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        # Accept either a single-row CSR matrix or the full interaction matrix
        if user_items.shape[0] == 1:
            user_vector = user_items
        else:
            if isinstance(user_id, str):
                try:
                    user_idx = int(user_id)
                except ValueError as exc:  # pragma: no cover - defensive branch
                    raise ValueError("String user_id must be convertible to int when user_items has multiple rows") from exc
            else:
                user_idx = int(user_id)
            user_vector = user_items.getrow(user_idx)

        scores = user_vector @ self.item_weights
        if sp.isspmatrix(scores):
            scores = scores.toarray()
        scores = np.asarray(scores, dtype=np.float32).ravel()

        if self.num_items and scores.shape[0] < self.num_items:
            # Pad to expected size if necessary (should not happen but keeps code defensive)
            scores = np.pad(scores, (0, self.num_items - scores.shape[0]))

        interacted = set(user_vector.indices.tolist())
        if interacted:
            scores[list(interacted)] = -np.inf

        top_k = min(n_items, scores.shape[0])
        if top_k <= 0:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        sorted_scores = scores[sorted_indices]

        valid_mask = np.isfinite(sorted_scores)
        if not np.any(valid_mask):
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        sorted_indices = sorted_indices[valid_mask]
        sorted_scores = sorted_scores[valid_mask]

        return sorted_indices.astype(np.int_), sorted_scores.astype(np.float32)

    def get_item_vectors(self, item_ids: List[Union[int, str]]) -> NDArray[np.float32]:
        """Return item representations for ANN or similarity search."""

        if not item_ids:
            return np.empty((0, self.num_items), dtype=np.float32)

        if self.item_weights is None or self.num_items == 0:
            return np.zeros((len(item_ids), 0), dtype=np.float32)

        output = np.zeros((len(item_ids), self.num_items), dtype=np.float32)

        for pos, item in enumerate(item_ids):
            if isinstance(item, str):
                try:
                    internal_idx = int(item)
                except ValueError:
                    continue
            else:
                internal_idx = int(item)

            if internal_idx < 0 or internal_idx >= self.num_items:
                continue

            row = self.item_weights.getrow(internal_idx)
            if row.nnz == 0:
                continue
            output[pos] = row.toarray().astype(np.float32, copy=False)[0]

        return output

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict[str, Any]:
        return {
            "lambda_": self.lambda_,
            "topk": self.topk,
            "num_items": self.num_items,
            "item_weights": self.item_weights,
        }

    def _set_model_state(self, model_state: dict[str, Any]) -> None:
        self.lambda_ = float(model_state.get("lambda_", 0.5))
        self.topk = model_state.get("topk")
        self.num_items = int(model_state.get("num_items", 0))
        stored_weights = model_state.get("item_weights")
        if stored_weights is not None and not sp.isspmatrix_csr(stored_weights):
            stored_weights = sp.csr_matrix(stored_weights)
        self.item_weights = stored_weights


# Register model so it is available through the registry/CLI
ModelRegistry.register(EASEModel.model_type, EASEModel)
