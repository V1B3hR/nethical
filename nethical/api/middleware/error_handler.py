"""Error handler middleware for API.

Provides structured error responses and error logging.

Implements Law 23 (Fail-Safe Design) by ensuring
graceful error handling.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for structured error handling.

    Features:
    - Consistent error response format
    - Error logging with request context
    - Safe error messages (no internal details leaked)
    - Fallback to blocking for safety-critical errors
    """

    def __init__(self, app, include_traceback: bool = False):
        """Initialize middleware.

        Args:
            app: ASGI application
            include_traceback: Whether to include traceback in responses (dev only)
        """
        super().__init__(app)
        self.include_traceback = include_traceback

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        try:
            return await call_next(request)
        except Exception as e:
            return await self._handle_error(request, e)

    async def _handle_error(self, request: Request, error: Exception) -> JSONResponse:
        """Handle an unhandled exception.

        Args:
            request: FastAPI request object
            error: The exception that occurred

        Returns:
            Structured JSON error response
        """
        request_id = getattr(request.state, "request_id", "unknown")

        # Log the error with context
        logger.error(
            "Unhandled error request_id=%s path=%s error=%s",
            request_id,
            request.url.path,
            str(error),
            exc_info=True,
        )

        # Determine error type and status code
        error_type = type(error).__name__
        status_code = 500

        # Safe error message (don't leak internals)
        safe_message = "An internal error occurred"
        if isinstance(error, ValueError):
            safe_message = "Invalid request data"
            status_code = 400
        elif isinstance(error, PermissionError):
            safe_message = "Permission denied"
            status_code = 403
        elif isinstance(error, FileNotFoundError):
            safe_message = "Resource not found"
            status_code = 404

        # Build error response
        error_response: dict[str, Any] = {
            "error": {
                "type": error_type,
                "message": safe_message,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "fundamental_law": "Law 23: Fail-Safe Design",
        }

        # Add traceback in development mode
        if self.include_traceback:
            error_response["error"]["traceback"] = traceback.format_exc()

        # Add safety-first decision for governance endpoints
        if "/evaluate" in request.url.path:
            error_response["decision"] = "BLOCK"
            error_response["reason"] = "Blocked for safety due to evaluation error"

        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers={
                "X-Request-ID": request_id,
                "X-Error-Type": error_type,
            },
        )


def create_error_response(
    error_type: str,
    message: str,
    request_id: str,
    status_code: int = 500,
    details: dict | None = None,
) -> JSONResponse:
    """Create a structured error response.

    Args:
        error_type: Type of error
        message: Human-readable error message
        request_id: Request identifier for tracing
        status_code: HTTP status code
        details: Optional additional details

    Returns:
        Structured JSON error response
    """
    content: dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    if details:
        content["error"]["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={"X-Request-ID": request_id},
    )
