"""Tests for caching functionality."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from recsys_lite.api.services import EntityType, RecommendationService, VectorRetrievalService
from recsys_lite.cache import CacheConfig, CacheManager, LRUCache, RedisCache, generate_cache_key


class TestCacheKey:
    """Test cache key generation."""

    def test_generate_cache_key_simple(self):
        """Test simple cache key generation."""
        key = generate_cache_key("user123", 10, True, prefix="rec")
        assert key == "rec:user123:10:True"

    def test_generate_cache_key_with_lists(self):
        """Test cache key generation with lists."""
        key = generate_cache_key("user123", 10, ["electronics", "books"], prefix="rec")
        # Should include hash for complex types
        assert "rec:user123:10:" in key
        assert len(key) > 20  # Should include hash

    def test_generate_cache_key_with_numpy(self):
        """Test cache key generation with numpy arrays."""
        arr = np.array([1, 2, 3])
        key = generate_cache_key("user123", arr, prefix="vec")
        assert "vec:user123:np_" in key
        assert len(key) > 15


class TestLRUCache:
    """Test LRU cache implementation."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3, default_ttl=60)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test non-existent key
        assert cache.get("nonexistent") is None

        # Test exists
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(max_size=2, default_ttl=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(0.2)  # Wait for expiration
        assert cache.get("key1") is None

    def test_custom_ttl(self):
        """Test custom TTL per key."""
        cache = LRUCache(max_size=10, default_ttl=60)

        cache.set("key1", "value1", ttl=0.1)  # 100ms TTL
        assert cache.get("key1") == "value1"

        time.sleep(0.2)  # Wait for expiration
        assert cache.get("key1") is None

    def test_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=10, default_ttl=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10, default_ttl=60)

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Set and get
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestRedisCache:
    """Test Redis cache implementation."""

    @patch("redis.Redis")
    def test_redis_available(self, mock_redis):
        """Test Redis cache when Redis is available."""
        # Mock Redis client
        mock_client = Mock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.exists.return_value = False

        config = CacheConfig(redis_host="localhost", redis_port=6379)
        cache = RedisCache(config)

        assert cache._is_available() is True

        # Test operations
        cache.set("key1", "value1")
        mock_client.set.assert_called()

        cache.get("key1")
        mock_client.get.assert_called_with("key1")

        cache.exists("key1")
        mock_client.exists.assert_called_with("key1")

    def test_redis_unavailable(self):
        """Test Redis cache when Redis is unavailable."""
        config = CacheConfig(redis_host="nonexistent", redis_port=6379)
        cache = RedisCache(config)

        assert cache._is_available() is False

        # Operations should fail gracefully
        cache.set("key1", "value1")  # Should not raise
        assert cache.get("key1") is None
        assert cache.exists("key1") is False


class TestCacheManager:
    """Test cache manager."""

    def test_initialization_lru(self):
        """Test cache manager initialization with LRU cache."""
        config = CacheConfig(enabled=True)
        manager = CacheManager(config)

        assert manager.enabled is True
        assert isinstance(manager._cache, LRUCache)

    @patch("redis.Redis")
    def test_initialization_redis(self, mock_redis):
        """Test cache manager initialization with Redis cache."""
        mock_client = Mock()
        mock_redis.return_value = mock_client
        mock_client.ping.return_value = True

        config = CacheConfig(enabled=True, redis_host="localhost")
        manager = CacheManager(config)

        assert manager.enabled is True
        assert isinstance(manager._cache, RedisCache)

    def test_user_recommendations_cache(self):
        """Test user recommendations caching."""
        config = CacheConfig(enabled=True)
        manager = CacheManager(config)

        # Test cache miss
        result = manager.get_user_recommendations("user123", 10)
        assert result is None

        # Set cache
        recommendations = (["item1", "item2"], [0.9, 0.8], [{"title": "Item 1"}, {"title": "Item 2"}])
        manager.set_user_recommendations("user123", 10, True, recommendations)

        # Test cache hit
        cached_result = manager.get_user_recommendations("user123", 10)
        assert cached_result == recommendations

    def test_similar_items_cache(self):
        """Test similar items caching."""
        config = CacheConfig(enabled=True)
        manager = CacheManager(config)

        # Test cache miss
        result = manager.get_similar_items("item123", 10)
        assert result is None

        # Set cache
        similar_items = (["item2", "item3"], [0.9, 0.8], [{"title": "Item 2"}, {"title": "Item 3"}])
        manager.set_similar_items("item123", 10, similar_items)

        # Test cache hit
        cached_result = manager.get_similar_items("item123", 10)
        assert cached_result == similar_items

    def test_vector_cache(self):
        """Test vector caching."""
        config = CacheConfig(enabled=True)
        manager = CacheManager(config)

        # Test user vector
        user_vector = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # Cache miss
        assert manager.get_user_vector(123) is None

        # Set cache
        manager.set_user_vector(123, user_vector)

        # Cache hit
        cached_vector = manager.get_user_vector(123)
        assert cached_vector is not None
        np.testing.assert_array_equal(cached_vector, user_vector)

        # Test item vector
        item_vector = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)

        # Cache miss
        assert manager.get_item_vector(456) is None

        # Set cache
        manager.set_item_vector(456, item_vector)

        # Cache hit
        cached_vector = manager.get_item_vector(456)
        assert cached_vector is not None
        np.testing.assert_array_equal(cached_vector, item_vector)

    def test_cache_disabled(self):
        """Test behavior when cache is disabled."""
        config = CacheConfig(enabled=False)
        manager = CacheManager(config)

        assert manager.enabled is False

        # All operations should return None/do nothing
        assert manager.get_user_recommendations("user123", 10) is None
        manager.set_user_recommendations("user123", 10, True, ([], [], []))

        assert manager.get_user_vector(123) is None
        manager.set_user_vector(123, np.array([[1.0]], dtype=np.float32))

    def test_cache_stats(self):
        """Test cache statistics."""
        config = CacheConfig(enabled=True)
        manager = CacheManager(config)

        stats = manager.get_stats()
        assert "enabled" in stats
        assert "backend" in stats
        assert "overall" in stats
        assert "config" in stats

        assert stats["enabled"] is True
        assert stats["overall"]["hits"] == 0
        assert stats["overall"]["misses"] == 0

    def test_cache_warming(self):
        """Test cache warming functionality."""
        config = CacheConfig(enabled=True)
        manager = CacheManager(config)

        # Mock recommendation service
        mock_service = Mock()
        mock_service.recommend_for_user.return_value = (["item1"], [0.9], [{"title": "Item 1"}])
        mock_service.find_similar_items.return_value = (["item2"], [0.8], [{"title": "Item 2"}])

        # Test warming
        stats = manager.warm_cache(
            popular_users=["user1", "user2"], popular_items=["item1", "item2"], recommendation_service=mock_service
        )

        assert stats["users_warmed"] == 2
        assert stats["items_warmed"] == 2


class TestVectorService:
    """Test vector service with caching."""

    def test_vector_service_with_cache(self):
        """Test vector service with cache manager."""
        config = CacheConfig(enabled=True)
        cache_manager = CacheManager(config)
        from recsys_lite.api.services import VectorRetrievalService

        vector_service = VectorRetrievalService(cache_manager)

        # Mock model
        mock_model = Mock()
        mock_factors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_model.get_user_factors.return_value = mock_factors

        # First call - should cache the result
        vector1 = vector_service.get_user_vector(mock_model, 0, model_version="v1")

        # Second call - should hit cache
        vector2 = vector_service.get_user_vector(mock_model, 0, model_version="v1")

        # Vectors should be equal
        np.testing.assert_array_equal(vector1, vector2)

        # Model should only be called once
        mock_model.get_user_factors.assert_called_once()

    def test_vector_service_without_cache(self):
        """Test vector service without cache manager."""
        from recsys_lite.api.services import VectorRetrievalService

        vector_service = VectorRetrievalService(None)

        # Mock model
        mock_model = Mock()
        mock_factors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_model.get_user_factors.return_value = mock_factors

        # Multiple calls - should not cache
        vector_service.get_user_vector(mock_model, 0)
        vector_service.get_user_vector(mock_model, 0)

        # Model should be called twice
        assert mock_model.get_user_factors.call_count == 2


class TestRecommendationServiceCaching:
    """Test recommendation service with caching."""

    def test_recommendation_service_with_cache(self):
        """Test recommendation service with cache manager."""
        config = CacheConfig(enabled=True)
        cache_manager = CacheManager(config)

        # Mock components
        mock_model = Mock()
        mock_faiss_index = Mock()
        mock_faiss_index.d = 128
        mock_faiss_index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[1, 2]]))

        user_mapping = {"user123": 0}
        item_mapping = {"item1": 1, "item2": 2}
        reverse_item_mapping = {1: "item1", 2: "item2"}

        # Create service
        service = RecommendationService(
            model=mock_model,
            faiss_index=mock_faiss_index,
            model_type="test",
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            reverse_item_mapping=reverse_item_mapping,
            cache_manager=cache_manager,
            model_version="v1",
        )

        # Mock vector service
        service.vector_service = Mock()
        service.vector_service.get_vector.return_value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # First call - should cache the result
        result1 = service.recommend_for_user("user123", k=10)

        # Second call - should hit cache
        result2 = service.recommend_for_user("user123", k=10)

        # Results should be equal
        assert result1 == result2

        # Vector service should only be called once
        service.vector_service.get_vector.assert_called_once()

    def test_recommendation_service_cache_with_filters(self):
        """Test that filtered recommendations are not cached."""
        config = CacheConfig(enabled=True)
        cache_manager = CacheManager(config)

        # Mock components
        mock_model = Mock()
        mock_faiss_index = Mock()
        mock_faiss_index.d = 128
        mock_faiss_index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[1, 2]]))

        user_mapping = {"user123": 0}
        item_mapping = {"item1": 1, "item2": 2}
        reverse_item_mapping = {1: "item1", 2: "item2"}

        # Create service
        service = RecommendationService(
            model=mock_model,
            faiss_index=mock_faiss_index,
            model_type="test",
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            reverse_item_mapping=reverse_item_mapping,
            cache_manager=cache_manager,
            model_version="v1",
        )

        # Mock vector service
        service.vector_service = Mock()
        service.vector_service.get_vector.return_value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # Call with filters - should not cache
        service.recommend_for_user("user123", k=10, categories=["electronics"])
        service.recommend_for_user("user123", k=10, categories=["electronics"])

        # Vector service should be called twice (no caching)
        assert service.vector_service.get_vector.call_count == 2


@pytest.mark.integration
class TestCacheIntegration:
    """Integration tests for caching system."""

    def test_end_to_end_caching(self):
        """Test end-to-end caching workflow."""
        config = CacheConfig(enabled=True, user_recommendations_ttl=60)
        cache_manager = CacheManager(config)

        # Simulate recommendation workflow
        user_id = "user123"
        k = 10
        recommendations = (
            ["item1", "item2", "item3"],
            [0.9, 0.8, 0.7],
            [{"title": "Item 1"}, {"title": "Item 2"}, {"title": "Item 3"}],
        )

        # First request - cache miss
        cached_result = cache_manager.get_user_recommendations(user_id, k)
        assert cached_result is None

        # Store in cache
        cache_manager.set_user_recommendations(user_id, k, True, recommendations)

        # Second request - cache hit
        cached_result = cache_manager.get_user_recommendations(user_id, k)
        assert cached_result == recommendations

        # Check stats
        stats = cache_manager.get_stats()
        assert stats["overall"]["hits"] == 1
        assert stats["overall"]["misses"] == 1
        assert stats["overall"]["hit_rate"] == 0.5

    def test_cache_key_collision_avoidance(self):
        """Test that different parameter combinations don't collide."""
        config = CacheConfig(enabled=True)
        cache_manager = CacheManager(config)

        # Different parameter combinations should have different keys
        rec1 = (["item1"], [0.9], [{"title": "Item 1"}])
        rec2 = (["item2"], [0.8], [{"title": "Item 2"}])

        # Same user, different k
        cache_manager.set_user_recommendations("user123", 10, True, rec1)
        cache_manager.set_user_recommendations("user123", 20, True, rec2)

        result1 = cache_manager.get_user_recommendations("user123", 10)
        result2 = cache_manager.get_user_recommendations("user123", 20)

        assert result1 == rec1
        assert result2 == rec2
        assert result1 != result2
