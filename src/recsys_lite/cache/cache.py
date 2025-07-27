"""Cache implementations for RecSys-Lite."""

import hashlib
import json
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from recsys_lite.utils.logging import get_logger

logger = get_logger("cache")


@dataclass
class CacheConfig:
    """Configuration for caching system."""

    # General settings
    enabled: bool = True
    default_ttl: int = 3600  # 1 hour

    # LRU Cache settings
    max_size: int = 10000

    # Redis settings
    redis_url: Optional[str] = None
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_connection_timeout: int = 5
    redis_socket_timeout: int = 5

    # TTL settings for different cache types
    user_recommendations_ttl: int = 3600  # 1 hour
    similar_items_ttl: int = 3600  # 1 hour
    user_vectors_ttl: int = 86400  # 24 hours
    item_vectors_ttl: int = 86400  # 24 hours

    # Cache key prefixes
    user_recommendations_prefix: str = "rec:user"
    similar_items_prefix: str = "rec:similar"
    user_vectors_prefix: str = "vec:user"
    item_vectors_prefix: str = "vec:item"


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete key from cache.

        Args:
            key: Cache key to delete
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        pass


class LRUCache(CacheBackend):
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired.

        Args:
            entry: Cache entry

        Returns:
            True if expired, False otherwise
        """
        if entry.get("expires_at") is None:
            return False
        return time.time() > entry["expires_at"]

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if self._is_expired(entry):
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Calculate expiration time
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self.default_ttl > 0:
                expires_at = time.time() + self.default_ttl

            # Create entry
            entry = {"value": value, "expires_at": expires_at, "created_at": time.time()}

            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)

            # Evict oldest if necessary
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                return False

            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "type": "LRU",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(hit_rate, 3),
                "total_requests": total_requests,
            }


class RedisCache(CacheBackend):
    """Redis cache implementation."""

    def __init__(self, config: CacheConfig):
        """Initialize Redis cache.

        Args:
            config: Cache configuration
        """
        self.config = config
        self._client = None
        self._connection_error = False

        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0

        self._init_client()

    def _init_client(self) -> None:
        """Initialize Redis client."""
        try:
            import redis

            if self.config.redis_url:
                self._client = redis.from_url(
                    self.config.redis_url,
                    socket_timeout=self.config.redis_socket_timeout,
                    socket_connect_timeout=self.config.redis_connection_timeout,
                    ssl=self.config.redis_ssl,
                )
            else:
                self._client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    socket_timeout=self.config.redis_socket_timeout,
                    socket_connect_timeout=self.config.redis_connection_timeout,
                    ssl=self.config.redis_ssl,
                )

            # Test connection
            self._client.ping()
            self._connection_error = False
            logger.info("Redis cache connected successfully")

        except ImportError:
            logger.warning("Redis not installed, falling back to LRU cache")
            self._client = None
            self._connection_error = True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._client = None
            self._connection_error = True

    def _is_available(self) -> bool:
        """Check if Redis is available."""
        return self._client is not None and not self._connection_error

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage.

        Args:
            value: Value to serialize

        Returns:
            Serialized value as bytes
        """
        # Handle numpy arrays efficiently
        if isinstance(value, np.ndarray):
            return pickle.dumps(
                {"type": "numpy_array", "data": value.tobytes(), "dtype": str(value.dtype), "shape": value.shape}
            )

        # Handle lists of strings/floats efficiently
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, (str, int, float)) for x in value):
            return pickle.dumps({"type": "simple_list", "data": value})

        # Default pickle serialization
        return pickle.dumps({"type": "default", "data": value})

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis storage.

        Args:
            data: Serialized data

        Returns:
            Deserialized value
        """
        obj = pickle.loads(data)

        if obj["type"] == "numpy_array":
            return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        elif obj["type"] == "simple_list":
            return obj["data"]
        else:
            return obj["data"]

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._is_available():
            self._misses += 1
            return None

        try:
            data = self._client.get(key)
            if data is None:
                self._misses += 1
                return None

            self._hits += 1
            return self._deserialize_value(data)

        except Exception as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            self._errors += 1
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if not self._is_available():
            return

        try:
            serialized_value = self._serialize_value(value)

            if ttl is not None:
                self._client.setex(key, ttl, serialized_value)
            else:
                self._client.set(key, serialized_value)

        except Exception as e:
            logger.warning(f"Redis set error for key {key}: {e}")
            self._errors += 1

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if not self._is_available():
            return

        try:
            self._client.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete error for key {key}: {e}")
            self._errors += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        if not self._is_available():
            return

        try:
            self._client.flushdb()
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            self._errors += 1

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._is_available():
            return False

        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.warning(f"Redis exists error for key {key}: {e}")
            self._errors += 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "type": "Redis",
            "available": self._is_available(),
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "hit_rate": round(self._hits / (self._hits + self._misses), 3) if (self._hits + self._misses) > 0 else 0.0,
            "total_requests": self._hits + self._misses,
        }

        if self._is_available():
            try:
                info = self._client.info()
                stats.update(
                    {
                        "memory_used": info.get("used_memory_human", "Unknown"),
                        "connected_clients": info.get("connected_clients", 0),
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0),
                    }
                )
            except Exception:
                pass

        return stats


def generate_cache_key(*args: Any, prefix: str = "", separator: str = ":") -> str:
    """Generate a cache key from arguments.

    Args:
        *args: Arguments to include in the key
        prefix: Key prefix
        separator: Separator between key parts

    Returns:
        Generated cache key
    """
    key_parts = []

    if prefix:
        key_parts.append(prefix)

    for arg in args:
        if isinstance(arg, (list, tuple, dict)):
            # For complex types, use hash
            arg_str = hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:8]
        elif isinstance(arg, np.ndarray):
            # For numpy arrays, use shape and hash of first few elements
            shape_str = "x".join(map(str, arg.shape))
            sample = arg.flat[: min(10, arg.size)]
            sample_hash = hashlib.md5(sample.tobytes()).hexdigest()[:8]
            arg_str = f"np_{shape_str}_{sample_hash}"
        else:
            arg_str = str(arg)

        key_parts.append(arg_str)

    return separator.join(key_parts)
