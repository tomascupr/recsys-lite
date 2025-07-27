"""Text embedding model using sentence-transformers with all-MiniLM-L6-v2.

This model provides content-based recommendations using state-of-the-art text embeddings
from the all-MiniLM-L6-v2 model. It generates embeddings for items based on their textual
description and provides recommendations based on semantic similarity.

Performance optimizations:
- ONNX runtime acceleration when available
- Batched processing for large catalogs
- Memory-efficient numpy arrays
- Vector caching to disk
- Sparse interaction handling
- Configurable context fields to maximize semantic understanding
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.models.base import BaseRecommender, ModelRegistry

logger = logging.getLogger("recsys-lite.models.text_embedding")


class TextEmbeddingModel(BaseRecommender):
    """Text embedding model using sentence-transformers for content-based recommendations."""

    model_type = "text_embedding"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        item_text_fields: Optional[List[str]] = None,
        field_weights: Optional[Dict[str, float]] = None,
        normalize_vectors: bool = True,
        cache_embeddings: bool = True,
        batch_size: int = 64,
        max_length: int = 512,
        embedding_dim: int = 384,  # Known dimension for all-MiniLM-L6-v2
    ) -> None:
        """Initialize text embedding model.

        Args:
            model_name: Pretrained model name or path
            item_text_fields: Item fields to use for text embedding
            field_weights: Weights for different fields to emphasize importance
            normalize_vectors: Whether to normalize vectors
            cache_embeddings: Whether to cache embeddings on disk
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length for text
            embedding_dim: Embedding dimension (known in advance for efficiency)
        """
        self.model_name = model_name
        self.item_text_fields = item_text_fields or ["title", "category", "brand", "description"]
        self.field_weights = field_weights or {
            "title": 2.0,
            "category": 1.0,
            "brand": 1.0,
            "description": 3.0,
        }
        self.normalize_vectors = normalize_vectors
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_ids: Optional[List[str]] = None
        self.id_to_idx: Dict[str, int] = {}

        # Flag for ONNX runtime
        self.using_onnx = False

        # Set environment variables for optimized performance
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        # Use MKL acceleration for numpy if available
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 4)

    def _prepare_item_text(self, item_data: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Prepare item text for embedding generation with weighted fields.

        Args:
            item_data: Dictionary of item metadata

        Returns:
            Tuple of (texts, item_ids)
        """
        texts = []
        item_ids = []

        for item_id, metadata in item_data.items():
            # Process text fields with weights
            text_parts = []

            for field in self.item_text_fields:
                if field in metadata and metadata[field]:
                    field_text = str(metadata[field])
                    weight = self.field_weights.get(field, 1.0)

                    # Repeat important fields to increase their weight in the embedding
                    # This is a simple form of field weighting without modifying the model
                    if weight > 1:
                        repetitions = int(weight)
                        text_parts.extend([field_text] * repetitions)
                    else:
                        text_parts.append(field_text)

            if text_parts:
                # Join with clear separator to maintain context boundaries
                item_text = " | ".join(text_parts)

                # Truncate if too long (for performance)
                if len(item_text) > self.max_length * 10:  # Rough character estimate
                    item_text = item_text[: self.max_length * 10]

                texts.append(item_text)
                item_ids.append(item_id)

        return texts, item_ids

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Generate item embeddings from text.

        Args:
            user_item_matrix: Ignored for this model
            **kwargs: Additional parameters
                - item_data: Dict mapping item_id to item metadata
                - output_dir: Optional directory to cache embeddings
        """
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "The sentence-transformers package is required. Install with: pip install sentence-transformers"
            )
            raise

        # Extract required fields
        item_data = kwargs.get("item_data", {})
        output_dir = kwargs.get("output_dir")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Try to load cached embeddings if available
        if self.cache_embeddings and output_dir:
            embedding_path = Path(output_dir) / "text_embeddings.npy"
            ids_path = Path(output_dir) / "item_ids.json"

            if embedding_path.exists() and ids_path.exists():
                try:
                    logger.info(f"Loading cached embeddings from {embedding_path}")
                    self.item_embeddings = np.load(str(embedding_path))

                    with open(ids_path, "r") as f:
                        self.item_ids = json.load(f)

                    # Create id to index mapping
                    if self.item_ids:
                        self.id_to_idx = {id_: i for i, id_ in enumerate(self.item_ids)}

                    item_count = 0 if self.item_ids is None else len(self.item_ids)
                    logger.info(f"Loaded embeddings for {item_count} items")
                    return
                except Exception as e:
                    logger.warning(f"Error loading cached embeddings: {e}")

        # Load model if not already loaded
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Get actual embedding dimension
            if self.model is not None:
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")

            # Performance optimization: Enable ONNX if available
            try:
                # Test if optimum and onnxruntime are available
                import importlib.util

                has_optimum = importlib.util.find_spec("optimum") is not None
                has_onnx = importlib.util.find_spec("onnxruntime") is not None

                if has_optimum and has_onnx:
                    from optimum.onnxruntime import ORTModelForFeatureExtraction

                    # Convert to ONNX for faster inference
                    if output_dir:
                        onnx_path = Path(output_dir) / "model.onnx"
                        if not onnx_path.exists() and self.model is not None:
                            logger.info("Converting model to ONNX format for faster inference...")
                            self.model.export_to_onnx(str(onnx_path))

                        # Load ONNX model for faster inference
                        if onnx_path.exists():
                            logger.info("Using ONNX acceleration for faster inference")
                            # Re-initialize with ONNX backend
                            self.model = SentenceTransformer(
                                modules=[ORTModelForFeatureExtraction.from_pretrained(str(onnx_path))]
                            )
                            self.using_onnx = True
                            logger.info("ONNX acceleration enabled successfully")
            except ImportError:
                logger.info("ONNX optimization not available. Using standard inference.")

        # Prepare text data
        texts, item_ids = self._prepare_item_text(item_data)

        if not texts:
            logger.warning("No text data found. Embeddings cannot be generated.")
            return

        logger.info(f"Generating embeddings for {len(texts)} items")

        # Generate embeddings in batches
        # Use device configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        if self.model is not None:
            self.model.to(device)

        # Generate embeddings in batches
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_size = len(batch_texts)

            logger.debug(f"Processing batch {i // self.batch_size + 1} with {batch_size} items")

            with torch.no_grad():
                if self.model is not None:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=len(texts) > 100,
                        device=device,
                        normalize_embeddings=self.normalize_vectors,
                    )
                else:
                    logger.error("Model is None. Unable to encode batch.")
                    continue
            all_embeddings.append(batch_embeddings)

        # Combine batches
        # Use float32 for memory efficiency
        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Normalize if requested and not already normalized by the model
        normalize_needed = True
        if self.model is not None and hasattr(self.model, "normalize_embeddings"):
            normalize_needed = not self.model.normalize_embeddings

        if self.normalize_vectors and normalize_needed:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        self.item_embeddings = embeddings
        self.item_ids = item_ids

        # Create id to index mapping
        self.id_to_idx = {id_: i for i, id_ in enumerate(item_ids)}

        # Cache embeddings if requested
        if self.cache_embeddings and output_dir:
            embedding_path = Path(output_dir) / "text_embeddings.npy"
            ids_path = Path(output_dir) / "item_ids.json"

            logger.info(f"Caching embeddings to {embedding_path}")
            np.save(str(embedding_path), embeddings)

            with open(ids_path, "w") as f:
                json.dump(item_ids, f)

        logger.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        """Generate recommendations based on user's item history.

        Args:
            user_id: User ID
            user_items: User-item interaction matrix
            n_items: Number of recommendations
            **kwargs: Additional parameters
                - item_mapping: Dict mapping item_id to index
                - reverse_item_mapping: Dict mapping index to item_id
        """
        if self.item_embeddings is None or self.item_ids is None:
            logger.warning("No item embeddings available for recommendations")
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        item_mapping = kwargs.get("item_mapping", {})
        reverse_item_mapping = kwargs.get("reverse_item_mapping", {})

        # Get items the user has interacted with
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)

        interacted_item_ids = []
        item_weights = {}

        if isinstance(user_id, int) and user_id < user_items.shape[0]:
            user_row = user_items[user_id]
            interacted_indices = user_row.indices
            interacted_values = user_row.data if hasattr(user_row, "data") else None

            # Get item IDs the user has interacted with
            for i, idx in enumerate(interacted_indices):
                item_id = reverse_item_mapping.get(int(idx))
                if item_id and item_id in self.id_to_idx:
                    interacted_item_ids.append(item_id)
                    # Store interaction strength if available
                    if interacted_values is not None:
                        item_weights[item_id] = float(interacted_values[i])

        if not interacted_item_ids:
            logger.debug(f"No interaction history for user {user_id}")
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        # Calculate user profile as weighted average of item embeddings
        item_indices = [self.id_to_idx[item_id] for item_id in interacted_item_ids if item_id in self.id_to_idx]

        if not item_indices:
            logger.debug(f"No matching items found in embeddings for user {user_id}")
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        # Use weighted average based on interaction frequency
        if item_weights:
            # Extract weights in the same order as item_indices
            weights = np.array([item_weights.get(self.item_ids[idx], 1.0) for idx in item_indices])
            weights = weights / weights.sum()  # Normalize

            # Apply weights to embeddings
            user_profile = np.zeros(self.embedding_dim, dtype=np.float32)
            for idx, weight in zip(item_indices, weights, strict=False):
                user_profile += weight * self.item_embeddings[idx]
        else:
            # Simple average if no weights available
            user_profile = np.mean(self.item_embeddings[item_indices], axis=0)

        # Normalize user profile
        if self.normalize_vectors:
            user_profile = user_profile / np.maximum(np.linalg.norm(user_profile), 1e-12)

        # Compute similarity scores
        # Optimized dot product for large matrices
        scores = np.dot(self.item_embeddings, user_profile)

        # Sort by similarity
        sorted_indices = np.argsort(-scores)

        # Filter out items the user has already interacted with
        filtered_indices = [idx for idx in sorted_indices if self.item_ids[idx] not in interacted_item_ids]

        # Get top N items
        top_indices = filtered_indices[:n_items]

        # Convert to item IDs in the global item mapping space
        top_item_indices = []
        top_scores = []

        for idx in top_indices:
            item_id = self.item_ids[idx]
            if item_id in item_mapping:
                top_item_indices.append(item_mapping[item_id])
                top_scores.append(float(scores[idx]))

        return np.array(top_item_indices, dtype=np.int_), np.array(top_scores, dtype=np.float32)

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> np.ndarray:
        """Get item vectors for specified items.

        Args:
            item_ids: List of item IDs

        Returns:
            Item vectors matrix
        """
        if self.item_embeddings is None or self.item_ids is None:
            return np.array([])

        indices = []
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.id_to_idx:
                indices.append(self.id_to_idx[item_id_str])

        if indices:
            return self.item_embeddings[indices]
        return np.array([])

    def encode_items(self, texts: List[str]) -> np.ndarray:
        """Encode new item texts using the loaded model.

        This is useful for encoding new items without retraining.

        Args:
            texts: List of text strings to encode

        Returns:
            Embeddings for the texts
        """
        if self.model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)

                # Get actual embedding dimension
                if self.model is not None:
                    self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except ImportError:
                logger.error(
                    "The sentence-transformers package is required. Install with: pip install sentence-transformers"
                )
                raise

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is not None:
            self.model.to(device)

        # Generate embeddings in batches
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            with torch.no_grad():
                if self.model is not None:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        device=device,
                        normalize_embeddings=self.normalize_vectors,
                    )
                else:
                    logger.error("Model is None. Unable to encode batch.")
                    continue
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Normalize if requested and not already normalized by the model
        normalize_needed = True
        if self.model is not None and hasattr(self.model, "normalize_embeddings"):
            normalize_needed = not self.model.normalize_embeddings

        if self.normalize_vectors and normalize_needed:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        return embeddings

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "model_name": self.model_name,
            "item_text_fields": self.item_text_fields,
            "field_weights": self.field_weights,
            "normalize_vectors": self.normalize_vectors,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "cache_embeddings": self.cache_embeddings,
            "item_embeddings": self.item_embeddings,
            "item_ids": self.item_ids,
            "using_onnx": self.using_onnx,
        }

    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data."""
        self.model_name = model_state["model_name"]
        self.item_text_fields = model_state["item_text_fields"]
        self.field_weights = model_state.get(
            "field_weights",
            {
                "title": 2.0,
                "category": 1.0,
                "brand": 1.0,
                "description": 3.0,
            },
        )
        self.normalize_vectors = model_state["normalize_vectors"]
        self.batch_size = model_state.get("batch_size", 64)
        self.max_length = model_state.get("max_length", 512)
        self.embedding_dim = model_state.get("embedding_dim", 384)
        self.cache_embeddings = model_state.get("cache_embeddings", True)
        self.item_embeddings = model_state.get("item_embeddings")
        self.item_ids = model_state.get("item_ids")
        self.using_onnx = model_state.get("using_onnx", False)

        # Recreate id to index mapping
        if self.item_ids is not None:
            self.id_to_idx = {id_: i for i, id_ in enumerate(self.item_ids)}


# Register the model
ModelRegistry.register("text_embedding", TextEmbeddingModel)
