"""Request context middleware for API.

Provides request ID propagation and context management.

Implements Law 15 (Audit Compliance) by ensuring all
requests are traceable through the system.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for request context management.

    Features:
    - Request ID generation/propagation
    - Start time tracking for latency measurement
    - Correlation ID support for distributed tracing

    Headers:
    - X-Request-ID: Unique request identifier
    - X-Correlation-ID: Alternative correlation header
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with context tracking."""
        # Generate or extract request ID
        request_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or str(uuid.uuid4())
        )

        # Store in request state
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()
        request.state.correlation_id = request.headers.get(
            "X-Correlation-ID", request_id
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response
        response.headers["X-Request-ID"] = request_id

        return response


def get_request_id(request: Request) -> str:
    """Get the request ID from request state.

    Args:
        request: FastAPI request object

    Returns:
        Request ID string
    """
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def get_start_time(request: Request) -> float:
    """Get the request start time from request state.

    Args:
        request: FastAPI request object

    Returns:
        Start time as perf_counter value
    """
    return getattr(request.state, "start_time", time.perf_counter())
