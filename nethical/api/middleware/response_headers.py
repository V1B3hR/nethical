"""Response headers middleware for API.

Provides standard response headers for latency tracking,
caching, and API versioning.

Implements Law 10 (Reasoning Transparency) through
observable response metadata.
"""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Current API version
API_VERSION = "2.0.0"


class ResponseHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding standard response headers.

    Headers added:
    - X-Nethical-Latency-Ms: Request processing latency
    - X-API-Version: Current API version
    - X-Cache-Status: Cache hit/miss status
    - Cache-Control: Caching directives
    """

    def __init__(self, app, api_version: str = API_VERSION):
        """Initialize middleware.

        Args:
            app: ASGI application
            api_version: API version string
        """
        super().__init__(app)
        self.api_version = api_version

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add response headers."""
        # Get start time from request state or now
        start_time = getattr(request.state, "start_time", time.perf_counter())

        # Process request
        response = await call_next(request)

        # Calculate latency
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Add standard headers
        response.headers["X-Nethical-Latency-Ms"] = str(latency_ms)
        response.headers["X-API-Version"] = self.api_version

        # Add cache status if available
        cache_status = getattr(request.state, "cache_status", None)
        if cache_status:
            response.headers["X-Cache-Status"] = cache_status

        # Set default cache control for governance decisions
        if "Cache-Control" not in response.headers:
            if request.url.path.startswith("/v2/evaluate"):
                # No caching for evaluation results
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
            elif request.url.path.startswith("/v2/policies"):
                # Short caching for policy lookups
                response.headers["Cache-Control"] = "private, max-age=60"
            elif request.url.path.startswith("/v2/metrics"):
                # Very short caching for metrics
                response.headers["Cache-Control"] = "no-cache"

        return response


def set_cache_status(request: Request, hit: bool) -> None:
    """Set cache status for the current request.

    Args:
        request: FastAPI request object
        hit: Whether the cache was hit
    """
    request.state.cache_status = "HIT" if hit else "MISS"
