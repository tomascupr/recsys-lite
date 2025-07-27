"""Base model class for RecSys-Lite."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, TypeVar, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

# Type variables for numpy arrays
T = TypeVar("T", bound=np.generic)
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int_]


class ModelPersistenceMixin:
    """Mixin for model persistence operations."""

    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_state = self._get_model_state()
        model_filename = f"{self._get_model_type()}_model.pkl"
        with open(save_path / model_filename, "wb") as f:
            pickle.dump(model_state, f)

    def load_model(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        load_path = Path(path)
        model_filename = f"{self._get_model_type()}_model.pkl"
        with open(load_path / model_filename, "rb") as f:
            model_state = pickle.load(f)
        self._set_model_state(model_state)

    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization.

        Returns:
            Dictionary with model state
        """
        pass

    @abstractmethod
    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data.

        Args:
            model_state: Dictionary with model state
        """
        pass

    @abstractmethod
    def _get_model_type(self) -> str:
        """Get model type identifier.

        Returns:
            Model type string
        """
        pass


class VectorProvider(Protocol):
    """Protocol for models that provide vector representations."""

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> FloatArray:
        """Get item vectors for specified items.

        Args:
            item_ids: List of item IDs

        Returns:
            Item vectors matrix
        """
        ...

    def get_user_vectors(self, user_ids: List[Union[str, int]]) -> FloatArray:
        """Get user vectors for specified users.

        Args:
            user_ids: List of user IDs

        Returns:
            User vectors matrix
        """
        ...


class FactorizationModelMixin:
    """Mixin for models with user and item factors."""

    user_factors: Optional[FloatArray] = None
    item_factors: Optional[FloatArray] = None

    def get_item_factors(self) -> FloatArray:
        """Get item factors matrix.

        Returns:
            Item factors matrix
        """
        if self.item_factors is None:
            return np.array([], dtype=np.float32)
        return self.item_factors

    def get_user_factors(self) -> FloatArray:
        """Get user factors matrix.

        Returns:
            User factors matrix
        """
        if self.user_factors is None:
            return np.array([], dtype=np.float32)
        return self.user_factors

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> FloatArray:
        """Get item vectors for specified items.

        Args:
            item_ids: List of item IDs (indices for factorization models)

        Returns:
            Item vectors matrix
        """
        if self.item_factors is None:
            return np.array([], dtype=np.float32)

        item_factor_length = self.item_factors.shape[0]
        indices = [int(item_id) for item_id in item_ids if int(item_id) < item_factor_length]
        if not indices:
            return np.array([], dtype=np.float32)

        return self.item_factors[indices]

    def get_user_vectors(self, user_ids: List[Union[str, int]]) -> FloatArray:
        """Get user vectors for specified users.

        Args:
            user_ids: List of user IDs (indices for factorization models)

        Returns:
            User vectors matrix
        """
        if self.user_factors is None:
            return np.array([], dtype=np.float32)

        # For factorization models, user_ids are typically indices
        user_factor_length = self.user_factors.shape[0]
        indices = [int(user_id) for user_id in user_ids if int(user_id) < user_factor_length]
        if not indices:
            return np.array([], dtype=np.float32)

        return self.user_factors[indices]


class BaseRecommender(ABC, ModelPersistenceMixin):
    """Abstract base class for recommendation models."""

    model_type: str = ""  # Override in subclasses

    def _get_model_type(self) -> str:
        """Get model type.

        Returns:
            Model type string
        """
        return self.model_type

    @abstractmethod
    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the model on user-item interaction data.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        pass

    @abstractmethod
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
        pass


class ModelRegistry:
    """Registry for recommendation models."""

    _registry: Dict[str, Type[BaseRecommender]] = {}

    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseRecommender]) -> None:
        """Register a model class.

        Args:
            model_type: Model type string
            model_class: Model class
        """
        cls._registry[model_type.lower()] = model_class

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseRecommender]:
        """Get model class by type.

        Args:
            model_type: Model type string

        Returns:
            Model class

        Raises:
            ValueError: If model type is not registered
        """
        model_class = cls._registry.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class

    @classmethod
    def create_model(cls, model_type: str, **kwargs: Any) -> BaseRecommender:
        """Create model instance by type.

        Args:
            model_type: Model type string
            **kwargs: Model parameters

        Returns:
            Model instance
        """
        model_class = cls.get_model_class(model_type)
        return model_class(**kwargs)

    @classmethod
    def load_model(cls, model_type: str, path: str) -> BaseRecommender:
        """Load model from disk.

        Args:
            model_type: Model type string
            path: Path to load model from

        Returns:
            Loaded model instance
        """
        model = cls.create_model(model_type)
        model.load_model(path)
        return model
