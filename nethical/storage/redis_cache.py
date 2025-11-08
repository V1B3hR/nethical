"""
Redis-based caching layer for high-speed data access.

This module provides a Redis integration for caching frequently accessed data
such as risk profiles, ML model predictions, and policy configurations.
"""

import json
import logging
from typing import Any, Optional, Dict, List

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class RedisCache:
    """
    High-speed Redis cache for Nethical governance system.

    Features:
    - Automatic key expiration (TTL)
    - Connection pooling for performance
    - Fallback to in-memory cache when Redis unavailable
    - JSON serialization for complex objects
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 300,  # 5 minutes default
        max_connections: int = 50,
        socket_timeout: int = 5,
        enabled: bool = True,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (optional)
            default_ttl: Default time-to-live in seconds
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            enabled: Whether caching is enabled
        """
        self.enabled = enabled and REDIS_AVAILABLE
        self.default_ttl = default_ttl
        self._fallback_cache: Dict[str, Any] = {}

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install redis-py with: pip install redis")
            self.enabled = False
            return

        if not self.enabled:
            logger.info("Redis cache disabled by configuration")
            return

        try:
            # Create connection pool
            self.pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                decode_responses=True,
            )
            self.client = redis.Redis(connection_pool=self.pool)

            # Test connection
            self.client.ping()
            logger.info(f"Redis cache connected to {host}:{port}")

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using fallback cache.")
            self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return self._fallback_cache.get(key)

        try:
            value = self.client.get(key)
            if value is None:
                return None

            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return self._fallback_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            self._fallback_cache[key] = value
            return True

        ttl = ttl or self.default_ttl

        try:
            # Serialize to JSON if complex type
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value)

            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)

            return True

        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            self._fallback_cache[key] = value
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled:
            return self._fallback_cache.pop(key, None) is not None

        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return self._fallback_cache.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists, False otherwise
        """
        if not self.enabled:
            return key in self._fallback_cache

        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return key in self._fallback_cache

    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment integer value in cache.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value after increment
        """
        if not self.enabled:
            current = self._fallback_cache.get(key, 0)
            new_value = current + amount
            self._fallback_cache[key] = new_value
            return new_value

        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis increment error for key {key}: {e}")
            current = self._fallback_cache.get(key, 0)
            new_value = current + amount
            self._fallback_cache[key] = new_value
            return new_value

    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values
        """
        if not self.enabled:
            return {k: self._fallback_cache.get(k) for k in keys}

        try:
            values = self.client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value is None:
                    result[key] = None
                else:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = value
            return result

        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            return {k: self._fallback_cache.get(k) for k in keys}

    def set_multiple(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in cache.

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            self._fallback_cache.update(mapping)
            return True

        ttl = ttl or self.default_ttl

        try:
            # Serialize values
            serialized = {}
            for key, value in mapping.items():
                if not isinstance(value, (str, int, float, bool)):
                    serialized[key] = json.dumps(value)
                else:
                    serialized[key] = value

            # Use pipeline for atomic operation
            pipe = self.client.pipeline()
            pipe.mset(serialized)

            # Set TTL for each key if specified
            if ttl:
                for key in serialized.keys():
                    pipe.expire(key, ttl)

            pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            self._fallback_cache.update(mapping)
            return False

    def clear(self) -> bool:
        """
        Clear all keys from cache.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            self._fallback_cache.clear()
            return True

        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {"enabled": False, "type": "fallback", "keys": len(self._fallback_cache)}

        try:
            info = self.client.info("stats")
            return {
                "enabled": True,
                "type": "redis",
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"enabled": True, "type": "redis", "error": str(e)}

    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    def close(self):
        """Close Redis connection."""
        if self.enabled and hasattr(self, "client"):
            try:
                self.client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
