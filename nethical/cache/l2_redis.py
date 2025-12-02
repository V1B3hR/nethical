"""
L2 Redis Cache - Regional Redis Cache Client

Redis-based cache for regional deployment.
Target: <5ms latency within same region
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class L2Config:
    """
    L2 Redis configuration.

    Attributes:
        host: Redis host
        port: Redis port
        password: Redis password (optional)
        db: Redis database number
        ssl: Use TLS
        ttl_seconds: Default TTL
        max_connections: Maximum connections
        socket_timeout: Socket timeout in seconds
    """

    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    ttl_seconds: int = 300  # 5 minutes
    max_connections: int = 10
    socket_timeout: float = 2.0


class L2RedisCache:
    """
    Regional Redis cache client.

    Features:
    - Connection pooling
    - Automatic serialization
    - TTL management
    - Pub/sub for invalidation

    Target: <5ms latency
    """

    def __init__(self, config: Optional[L2Config] = None):
        """
        Initialize L2RedisCache.

        Args:
            config: Redis configuration
        """
        self.config = config or L2Config()
        self._client = None
        self._pool = None
        self._connected = False

        # Metrics
        self._hits = 0
        self._misses = 0
        self._errors = 0

        logger.info(
            f"L2RedisCache initialized: {self.config.host}:{self.config.port}"
        )

    def connect(self) -> bool:
        """
        Establish Redis connection.

        Returns:
            True if connected successfully
        """
        try:
            import redis

            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                ssl=self.config.ssl,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # Test connection
            self._client.ping()
            self._connected = True
            logger.info("L2RedisCache connected")
            return True

        except ImportError:
            logger.warning("Redis package not installed")
            return False
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._errors += 1
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self._connected:
            return None

        try:
            data = self._client.get(key)
            if data is None:
                self._misses += 1
                return None

            self._hits += 1
            return json.loads(data)

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._errors += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in Redis.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: TTL in seconds

        Returns:
            True if successful
        """
        if not self._connected:
            return False

        try:
            ttl = ttl if ttl is not None else self.config.ttl_seconds
            data = json.dumps(value)
            self._client.setex(key, ttl, data)
            return True

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self._errors += 1
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from Redis.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if not self._connected:
            return False

        try:
            return self._client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self._errors += 1
            return False

    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values.

        Args:
            keys: List of cache keys

        Returns:
            Dict of key -> value
        """
        if not self._connected:
            return {}

        try:
            values = self._client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = json.loads(value)
                    self._hits += 1
                else:
                    self._misses += 1
            return result

        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            self._errors += 1
            return {}

    def mset(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set multiple values.

        Args:
            items: Dict of key -> value
            ttl: TTL in seconds

        Returns:
            True if successful
        """
        if not self._connected:
            return False

        try:
            ttl = ttl if ttl is not None else self.config.ttl_seconds
            pipe = self._client.pipeline()
            for key, value in items.items():
                data = json.dumps(value)
                pipe.setex(key, ttl, data)
            pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            self._errors += 1
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate keys matching pattern.

        Args:
            pattern: Redis pattern (e.g., "policy:*")

        Returns:
            Number of keys deleted
        """
        if not self._connected:
            return 0

        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Redis invalidate error: {e}")
            self._errors += 1
            return 0

    def publish(self, channel: str, message: Any) -> int:
        """
        Publish message to channel.

        Args:
            channel: Pub/sub channel
            message: Message to publish

        Returns:
            Number of subscribers that received
        """
        if not self._connected:
            return 0

        try:
            data = json.dumps(message)
            return self._client.publish(channel, data)

        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            self._errors += 1
            return 0

    def subscribe(self, channel: str, callback: callable):
        """
        Subscribe to channel.

        Args:
            channel: Pub/sub channel
            callback: Function to call on message
        """
        if not self._connected:
            return

        try:
            pubsub = self._client.pubsub()
            pubsub.subscribe(**{channel: callback})

        except Exception as e:
            logger.error(f"Redis subscribe error: {e}")
            self._errors += 1

    def close(self):
        """Close Redis connection."""
        if self._pool:
            self._pool.disconnect()
            self._connected = False

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        total = self._hits + self._misses
        return {
            "connected": self._connected,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "errors": self._errors,
        }
