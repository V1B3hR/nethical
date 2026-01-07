"""Request Optimizer - Optimize request handling for ultra-low latency.

Features:
- LRU cache for repeated queries (10k entries)
- Dynamic batching (max 32, timeout 10ms)
- Request coalescing
"""

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from cachetools import LRUCache


@dataclass
class DynamicBatcherConfig:
    """Configuration for dynamic batching."""

    max_batch_size: int = 32
    timeout_ms: float = 10.0
    enable_batching: bool = True


class DynamicBatcher:
    """Dynamic batching for requests to optimize throughput."""

    def __init__(self, config: Optional[DynamicBatcherConfig] = None):
        """Initialize dynamic batcher.

        Args:
            config: Optional configuration
        """
        self.config = config or DynamicBatcherConfig()
        self._pending_requests: List[Tuple[Any, asyncio.Future]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None

    async def process_batch(
        self, processor: Callable[[List[Any]], Awaitable[List[Any]]]
    ) -> None:
        """Process pending batch.

        Args:
            processor: Function to process batch of requests
        """
        if not self._pending_requests:
            return

        # Extract requests
        requests = [req for req, _ in self._pending_requests]

        try:
            # Process batch
            results = await processor(requests)

            # Return results to waiting futures
            for (_, future), result in zip(self._pending_requests, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # Propagate error to all waiting futures
            for _, future in self._pending_requests:
                if not future.done():
                    future.set_exception(e)

        finally:
            self._pending_requests.clear()

    async def add_request(
        self, request: Any, processor: Callable[[List[Any]], Awaitable[List[Any]]]
    ) -> Any:
        """Add request to batch.

        Args:
            request: Request to add
            processor: Batch processor function

        Returns:
            Result for this request
        """
        if not self.config.enable_batching:
            # Bypass batching
            results = await processor([request])
            return results[0]

        async with self._batch_lock:
            # Create future for this request
            future: asyncio.Future = asyncio.Future()
            self._pending_requests.append((request, future))

            # Check if batch is full
            if len(self._pending_requests) >= self.config.max_batch_size:
                # Process immediately
                await self.process_batch(processor)
                return await future

            # Schedule batch processing
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._schedule_batch(processor))

        # Wait for result
        return await future

    async def _schedule_batch(
        self, processor: Callable[[List[Any]], Awaitable[List[Any]]]
    ) -> None:
        """Schedule batch processing after timeout.

        Args:
            processor: Batch processor function
        """
        await asyncio.sleep(self.config.timeout_ms / 1000.0)

        async with self._batch_lock:
            await self.process_batch(processor)


class RequestOptimizer:
    """Optimize request handling with caching and batching."""

    def __init__(
        self,
        cache_maxsize: int = 10000,
        batcher_config: Optional[DynamicBatcherConfig] = None,
    ):
        """Initialize request optimizer.

        Args:
            cache_maxsize: Maximum cache size
            batcher_config: Configuration for dynamic batcher
        """
        self.cache = LRUCache(maxsize=cache_maxsize)
        self.batcher = DynamicBatcher(batcher_config)

        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0
        self._batched_requests = 0

    def _compute_cache_key(self, request: Any) -> str:
        """Compute cache key for request.

        Args:
            request: Request data

        Returns:
            Cache key string
        """
        # Convert request to hashable string
        request_str = str(sorted(request.items()) if isinstance(request, dict) else request)
        return hashlib.md5(request_str.encode()).hexdigest()

    async def process(
        self,
        requests: List[Any],
        processor: Callable[[List[Any]], Awaitable[List[Any]]],
        enable_cache: bool = True,
        enable_batching: bool = True,
    ) -> List[Any]:
        """Process requests with caching and batching.

        Args:
            requests: List of requests
            processor: Function to process requests
            enable_cache: Whether to use cache
            enable_batching: Whether to use batching

        Returns:
            List of results
        """
        self._total_requests += len(requests)

        results = []

        # Separate cached and uncached requests
        uncached_requests = []
        uncached_indices = []

        for i, request in enumerate(requests):
            if enable_cache:
                cache_key = self._compute_cache_key(request)

                if cache_key in self.cache:
                    # Cache hit
                    self._cache_hits += 1
                    results.append(self.cache[cache_key])
                else:
                    # Cache miss
                    self._cache_misses += 1
                    uncached_requests.append(request)
                    uncached_indices.append(i)
                    results.append(None)  # Placeholder
            else:
                uncached_requests.append(request)
                uncached_indices.append(i)
                results.append(None)

        # Process uncached requests
        if uncached_requests:
            if enable_batching and len(uncached_requests) > 1:
                # Use batching
                self._batched_requests += len(uncached_requests)
                batch_results = await self._process_batch(uncached_requests, processor)
            else:
                # Process without batching
                batch_results = await processor(uncached_requests)

            # Update cache and results
            for i, request, result in zip(uncached_indices, uncached_requests, batch_results):
                results[i] = result

                if enable_cache:
                    cache_key = self._compute_cache_key(request)
                    self.cache[cache_key] = result

        return results

    async def _process_batch(
        self, requests: List[Any], processor: Callable[[List[Any]], Awaitable[List[Any]]]
    ) -> List[Any]:
        """Process batch of requests.

        Args:
            requests: List of requests
            processor: Batch processor function

        Returns:
            List of results
        """
        # Use dynamic batcher
        tasks = [self.batcher.add_request(req, processor) for req in requests]
        return await asyncio.gather(*tasks)

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        print("[RequestOptimizer] Cache cleared")

    def get_metrics(self) -> Dict[str, Any]:
        """Get optimizer metrics.

        Returns:
            Dictionary with metrics
        """
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0.0
        )

        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "batched_requests": self._batched_requests,
            "batching_rate": (
                self._batched_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }


class RequestCoalescer:
    """Coalesce duplicate concurrent requests."""

    def __init__(self):
        """Initialize request coalescer."""
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def coalesce(
        self, request: Any, processor: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Coalesce duplicate requests.

        If the same request is being processed, wait for existing result
        instead of processing again.

        Args:
            request: Request to process
            processor: Function to process request

        Returns:
            Result for request
        """
        # Compute request key
        request_str = str(sorted(request.items()) if isinstance(request, dict) else request)
        request_key = hashlib.md5(request_str.encode()).hexdigest()

        async with self._lock:
            # Check if request is already pending
            if request_key in self._pending:
                # Wait for existing request
                return await self._pending[request_key]

            # Create new future for this request
            future: asyncio.Future = asyncio.Future()
            self._pending[request_key] = future

        try:
            # Process request
            result = await processor(request)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            # Remove from pending
            async with self._lock:
                self._pending.pop(request_key, None)
