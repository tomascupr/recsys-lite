"""Cache manager for RecSys-Lite."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from recsys_lite.api.models import ItemMetadata
from recsys_lite.cache.cache import CacheConfig, LRUCache, RedisCache, generate_cache_key
from recsys_lite.utils.logging import get_logger

logger = get_logger("cache.manager")


class CacheManager:
    """Manages caching for recommendations and vectors."""

    def __init__(self, config: CacheConfig):
        """Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.enabled = config.enabled

        # Initialize cache backend
        if config.redis_url or (config.redis_host and config.redis_port):
            logger.info("Initializing Redis cache")
            self._cache = RedisCache(config)
            # Fallback to LRU if Redis fails
            if not self._cache._is_available():
                logger.info("Redis unavailable, falling back to LRU cache")
                self._cache = LRUCache(config.max_size, config.default_ttl)
        else:
            logger.info("Initializing LRU cache")
            self._cache = LRUCache(config.max_size, config.default_ttl)

        # Cache hit/miss statistics
        self._total_hits = 0
        self._total_misses = 0
        self._start_time = time.time()

        # Key registries for targeted invalidation
        self._user_recommendation_keys: Dict[str, set[str]] = {}
        self._similar_item_keys: Dict[str, set[str]] = {}
        self._user_vector_keys: Dict[str, set[str]] = {}
        self._item_vector_keys: Dict[str, set[str]] = {}

    def _prune_registry(self, registry: Dict[str, set[str]]) -> None:
        stale_identifiers: List[str] = []
        for identifier, key_set in list(registry.items()):
            valid_keys = {cache_key for cache_key in key_set if self._cache.exists(cache_key)}
            if valid_keys:
                registry[identifier] = valid_keys
            else:
                stale_identifiers.append(identifier)

        for identifier in stale_identifiers:
            registry.pop(identifier, None)

    def _register_key(self, registry: Dict[str, set[str]], identifier: str, key: str) -> None:
        self._prune_registry(registry)
        if identifier not in registry:
            registry[identifier] = set()
        registry[identifier].add(key)

    def _collect_keys(self, registry: Dict[str, set[str]], identifier: Optional[str]) -> set[str]:
        self._prune_registry(registry)
        if identifier is None:
            keys: set[str] = set()
            for key_set in registry.values():
                keys.update(key_set)
            registry.clear()
            return keys

        key_set = registry.pop(identifier, set())
        return set(key_set)

    def _delete_keys(self, keys: set[str]) -> None:
        for cache_key in keys:
            self._cache.delete(cache_key)

    def _reset_registries(self) -> None:
        self._user_recommendation_keys.clear()
        self._similar_item_keys.clear()
        self._user_vector_keys.clear()
        self._item_vector_keys.clear()

    def _get_ttl_for_type(self, cache_type: str) -> int:
        """Get TTL for specific cache type.

        Args:
            cache_type: Type of cache (user_recommendations, similar_items, etc.)

        Returns:
            TTL in seconds
        """
        ttl_map = {
            "user_recommendations": self.config.user_recommendations_ttl,
            "similar_items": self.config.similar_items_ttl,
            "user_vectors": self.config.user_vectors_ttl,
            "item_vectors": self.config.item_vectors_ttl,
        }
        return ttl_map.get(cache_type, self.config.default_ttl)

    def _get_prefix_for_type(self, cache_type: str) -> str:
        """Get prefix for specific cache type.

        Args:
            cache_type: Type of cache

        Returns:
            Cache key prefix
        """
        prefix_map = {
            "user_recommendations": self.config.user_recommendations_prefix,
            "similar_items": self.config.similar_items_prefix,
            "user_vectors": self.config.user_vectors_prefix,
            "item_vectors": self.config.item_vectors_prefix,
        }
        return prefix_map.get(cache_type, "cache")

    def get_user_recommendations(
        self,
        user_id: str,
        k: int,
        use_faiss: bool = True,
        categories: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_items: Optional[List[str]] = None,
        include_items: Optional[List[str]] = None,
    ) -> Optional[Tuple[List[str], List[float], List[ItemMetadata]]]:
        """Get cached user recommendations.

        Args:
            user_id: User ID
            k: Number of recommendations
            use_faiss: Whether FAISS was used
            categories: Category filters
            brands: Brand filters
            min_price: Minimum price filter
            max_price: Maximum price filter
            exclude_items: Items to exclude
            include_items: Items to include

        Returns:
            Cached recommendations or None if not found
        """
        if not self.enabled:
            return None

        # Generate cache key including all parameters that affect results
        key = generate_cache_key(
            user_id,
            k,
            use_faiss,
            categories,
            brands,
            min_price,
            max_price,
            exclude_items,
            include_items,
            prefix=self._get_prefix_for_type("user_recommendations"),
        )

        start_time = time.time()
        result = self._cache.get(key)

        if result is not None:
            self._total_hits += 1
            logger.debug(
                f"Cache HIT for user recommendations: {user_id}",
                extra={
                    "user_id": user_id,
                    "cache_key": key,
                    "cache_time_ms": round((time.time() - start_time) * 1000, 2),
                },
            )
            return result

        self._total_misses += 1
        logger.debug(
            f"Cache MISS for user recommendations: {user_id}",
            extra={"user_id": user_id, "cache_key": key, "cache_time_ms": round((time.time() - start_time) * 1000, 2)},
        )
        return None

    def set_user_recommendations(
        self,
        user_id: str,
        k: int,
        use_faiss: bool,
        recommendations: Tuple[List[str], List[float], List[ItemMetadata]],
        categories: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_items: Optional[List[str]] = None,
        include_items: Optional[List[str]] = None,
    ) -> None:
        """Cache user recommendations.

        Args:
            user_id: User ID
            k: Number of recommendations
            use_faiss: Whether FAISS was used
            recommendations: Recommendations to cache
            categories: Category filters
            brands: Brand filters
            min_price: Minimum price filter
            max_price: Maximum price filter
            exclude_items: Items to exclude
            include_items: Items to include
        """
        if not self.enabled:
            return

        key = generate_cache_key(
            user_id,
            k,
            use_faiss,
            categories,
            brands,
            min_price,
            max_price,
            exclude_items,
            include_items,
            prefix=self._get_prefix_for_type("user_recommendations"),
        )

        ttl = self._get_ttl_for_type("user_recommendations")

        start_time = time.time()
        self._cache.set(key, recommendations, ttl)

        logger.debug(
            f"Cached user recommendations: {user_id}",
            extra={
                "user_id": user_id,
                "cache_key": key,
                "ttl": ttl,
                "recommendations_count": len(recommendations[0]) if recommendations[0] else 0,
                "cache_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        )

        self._register_key(self._user_recommendation_keys, user_id, key)

    def get_similar_items(
        self,
        item_id: str,
        k: int,
        categories: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_items: Optional[List[str]] = None,
    ) -> Optional[Tuple[List[str], List[float], List[ItemMetadata]]]:
        """Get cached similar items.

        Args:
            item_id: Item ID
            k: Number of similar items
            categories: Category filters
            brands: Brand filters
            min_price: Minimum price filter
            max_price: Maximum price filter
            exclude_items: Items to exclude

        Returns:
            Cached similar items or None if not found
        """
        if not self.enabled:
            return None

        key = generate_cache_key(
            item_id,
            k,
            categories,
            brands,
            min_price,
            max_price,
            exclude_items,
            prefix=self._get_prefix_for_type("similar_items"),
        )

        start_time = time.time()
        result = self._cache.get(key)

        if result is not None:
            self._total_hits += 1
            logger.debug(
                f"Cache HIT for similar items: {item_id}",
                extra={
                    "item_id": item_id,
                    "cache_key": key,
                    "cache_time_ms": round((time.time() - start_time) * 1000, 2),
                },
            )
            return result

        self._total_misses += 1
        logger.debug(
            f"Cache MISS for similar items: {item_id}",
            extra={"item_id": item_id, "cache_key": key, "cache_time_ms": round((time.time() - start_time) * 1000, 2)},
        )
        return None

    def set_similar_items(
        self,
        item_id: str,
        k: int,
        similar_items: Tuple[List[str], List[float], List[ItemMetadata]],
        categories: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_items: Optional[List[str]] = None,
    ) -> None:
        """Cache similar items.

        Args:
            item_id: Item ID
            k: Number of similar items
            similar_items: Similar items to cache
            categories: Category filters
            brands: Brand filters
            min_price: Minimum price filter
            max_price: Maximum price filter
            exclude_items: Items to exclude
        """
        if not self.enabled:
            return

        key = generate_cache_key(
            item_id,
            k,
            categories,
            brands,
            min_price,
            max_price,
            exclude_items,
            prefix=self._get_prefix_for_type("similar_items"),
        )

        ttl = self._get_ttl_for_type("similar_items")

        start_time = time.time()
        self._cache.set(key, similar_items, ttl)

        logger.debug(
            f"Cached similar items: {item_id}",
            extra={
                "item_id": item_id,
                "cache_key": key,
                "ttl": ttl,
                "similar_items_count": len(similar_items[0]) if similar_items[0] else 0,
                "cache_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        )

        self._register_key(self._similar_item_keys, item_id, key)

    def get_user_vector(self, user_idx: int, model_version: Optional[str] = None) -> Optional[NDArray[np.float32]]:
        """Get cached user vector.

        Args:
            user_idx: User index
            model_version: Model version for cache invalidation

        Returns:
            Cached user vector or None if not found
        """
        if not self.enabled:
            return None

        key = generate_cache_key(user_idx, model_version, prefix=self._get_prefix_for_type("user_vectors"))

        start_time = time.time()
        result = self._cache.get(key)

        if result is not None:
            self._total_hits += 1
            logger.debug(
                f"Cache HIT for user vector: {user_idx}",
                extra={
                    "user_idx": user_idx,
                    "cache_key": key,
                    "cache_time_ms": round((time.time() - start_time) * 1000, 2),
                },
            )
            return result

        self._total_misses += 1
        logger.debug(
            f"Cache MISS for user vector: {user_idx}",
            extra={
                "user_idx": user_idx,
                "cache_key": key,
                "cache_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        )
        return None

    def set_user_vector(self, user_idx: int, vector: NDArray[np.float32], model_version: Optional[str] = None) -> None:
        """Cache user vector.

        Args:
            user_idx: User index
            vector: User vector to cache
            model_version: Model version for cache invalidation
        """
        if not self.enabled:
            return

        key = generate_cache_key(user_idx, model_version, prefix=self._get_prefix_for_type("user_vectors"))

        ttl = self._get_ttl_for_type("user_vectors")

        start_time = time.time()
        self._cache.set(key, vector, ttl)

        logger.debug(
            f"Cached user vector: {user_idx}",
            extra={
                "user_idx": user_idx,
                "cache_key": key,
                "ttl": ttl,
                "vector_shape": vector.shape,
                "cache_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        )

        self._register_key(self._user_vector_keys, str(user_idx), key)

    def get_item_vector(self, item_idx: int, model_version: Optional[str] = None) -> Optional[NDArray[np.float32]]:
        """Get cached item vector.

        Args:
            item_idx: Item index
            model_version: Model version for cache invalidation

        Returns:
            Cached item vector or None if not found
        """
        if not self.enabled:
            return None

        key = generate_cache_key(item_idx, model_version, prefix=self._get_prefix_for_type("item_vectors"))

        start_time = time.time()
        result = self._cache.get(key)

        if result is not None:
            self._total_hits += 1
            logger.debug(
                f"Cache HIT for item vector: {item_idx}",
                extra={
                    "item_idx": item_idx,
                    "cache_key": key,
                    "cache_time_ms": round((time.time() - start_time) * 1000, 2),
                },
            )
            return result

        self._total_misses += 1
        logger.debug(
            f"Cache MISS for item vector: {item_idx}",
            extra={
                "item_idx": item_idx,
                "cache_key": key,
                "cache_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        )
        return None

    def set_item_vector(self, item_idx: int, vector: NDArray[np.float32], model_version: Optional[str] = None) -> None:
        """Cache item vector.

        Args:
            item_idx: Item index
            vector: Item vector to cache
            model_version: Model version for cache invalidation
        """
        if not self.enabled:
            return

        key = generate_cache_key(item_idx, model_version, prefix=self._get_prefix_for_type("item_vectors"))

        ttl = self._get_ttl_for_type("item_vectors")

        start_time = time.time()
        self._cache.set(key, vector, ttl)

        logger.debug(
            f"Cached item vector: {item_idx}",
            extra={
                "item_idx": item_idx,
                "cache_key": key,
                "ttl": ttl,
                "vector_shape": vector.shape,
                "cache_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        )

        self._register_key(self._item_vector_keys, str(item_idx), key)

    def invalidate_user_cache(self, user_id: Optional[str] = None) -> None:
        """Invalidate user-related cache entries.

        Args:
            user_id: Specific user ID to invalidate, or None for all users
        """
        if not self.enabled:
            return

        if user_id:
            keys = self._collect_keys(self._user_recommendation_keys, user_id)
            logger.info(f"Invalidating cache for user: {user_id}", extra={"keys": len(keys)})
            self._delete_keys(keys)
        else:
            keys = self._collect_keys(self._user_recommendation_keys, None)
            logger.info("Invalidating all user caches", extra={"keys": len(keys)})
            self._delete_keys(keys)

    def invalidate_item_cache(self, item_id: Optional[str] = None) -> None:
        """Invalidate item-related cache entries.

        Args:
            item_id: Specific item ID to invalidate, or None for all items
        """
        if not self.enabled:
            return

        if item_id:
            keys = self._collect_keys(self._similar_item_keys, item_id)
            logger.info(f"Invalidating cache for item: {item_id}", extra={"keys": len(keys)})
            self._delete_keys(keys)
        else:
            keys = self._collect_keys(self._similar_item_keys, None)
            logger.info("Invalidating all item caches", extra={"keys": len(keys)})
            self._delete_keys(keys)

    def invalidate_model_cache(self, model_version: Optional[str] = None) -> None:
        """Invalidate cache when model is updated.

        Args:
            model_version: New model version
        """
        if not self.enabled:
            return

        logger.info(f"Invalidating model cache, new version: {model_version}")
        # In a full implementation, this would clear all caches or update version keys
        self.clear_all()

    def clear_all(self) -> None:
        """Clear all cache entries."""
        if not self.enabled:
            return

        self._cache.clear()
        logger.info("Cleared all cache entries")
        self._reset_registries()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        backend_stats = self._cache.get_stats()

        uptime = time.time() - self._start_time
        total_requests = self._total_hits + self._total_misses
        overall_hit_rate = self._total_hits / total_requests if total_requests > 0 else 0.0

        return {
            "enabled": self.enabled,
            "backend": backend_stats,
            "overall": {
                "hits": self._total_hits,
                "misses": self._total_misses,
                "hit_rate": round(overall_hit_rate, 3),
                "total_requests": total_requests,
                "uptime_seconds": round(uptime, 2),
                "requests_per_second": round(total_requests / max(uptime, 1), 2),
            },
            "config": {
                "user_recommendations_ttl": self.config.user_recommendations_ttl,
                "similar_items_ttl": self.config.similar_items_ttl,
                "user_vectors_ttl": self.config.user_vectors_ttl,
                "item_vectors_ttl": self.config.item_vectors_ttl,
            },
        }

    def warm_cache(
        self,
        popular_users: Optional[List[str]] = None,
        popular_items: Optional[List[str]] = None,
        recommendation_service: Optional[Any] = None,
    ) -> Dict[str, int]:
        """Warm cache with popular users and items.

        Args:
            popular_users: List of popular user IDs to warm
            popular_items: List of popular item IDs to warm
            recommendation_service: Service to generate recommendations

        Returns:
            Dictionary with warming statistics
        """
        if not self.enabled or not recommendation_service:
            return {"users_warmed": 0, "items_warmed": 0}

        stats = {"users_warmed": 0, "items_warmed": 0}

        # Warm user recommendations
        if popular_users:
            logger.info(f"Warming cache for {len(popular_users)} users")
            for user_id in popular_users:
                try:
                    # Get and cache recommendations with default parameters
                    result = recommendation_service.recommend_for_user(user_id, k=10, use_faiss=True)
                    if result:
                        self.set_user_recommendations(user_id=user_id, k=10, use_faiss=True, recommendations=result)
                        stats["users_warmed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to warm cache for user {user_id}: {e}")

        # Warm similar items
        if popular_items:
            logger.info(f"Warming cache for {len(popular_items)} items")
            for item_id in popular_items:
                try:
                    result = recommendation_service.find_similar_items(item_id, k=10)
                    if result:
                        self.set_similar_items(item_id=item_id, k=10, similar_items=result)
                        stats["items_warmed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to warm cache for item {item_id}: {e}")

        logger.info(f"Cache warming completed: {stats}")
        return stats
