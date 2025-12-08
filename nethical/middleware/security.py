"""
Security Headers Middleware

Adds security headers to all HTTP responses for enhanced protection.
"""

from typing import Any, Callable, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all HTTP responses.

    Security headers included:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking attacks
    - X-XSS-Protection: Legacy XSS protection for older browsers
    - Strict-Transport-Security: Enforces HTTPS connections
    - Content-Security-Policy: Controls resource loading
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    - Cache-Control: Controls caching behavior for sensitive data
    """

    DEFAULT_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }

    def __init__(
        self,
        app: Any,
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = False,
        content_security_policy: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        cache_control: str = "no-store, no-cache, must-revalidate, private",
    ) -> None:
        """
        Initialize the security headers middleware.

        Args:
            app: The ASGI application
            enable_hsts: Whether to enable HTTP Strict Transport Security
            hsts_max_age: Max age for HSTS in seconds (default: 1 year)
            hsts_include_subdomains: Include subdomains in HSTS
            hsts_preload: Enable HSTS preload
            content_security_policy: Custom CSP header value
            custom_headers: Additional custom headers to add
            cache_control: Cache-Control header value
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.content_security_policy = content_security_policy or self._default_csp()
        self.custom_headers = custom_headers or {}
        self.cache_control = cache_control

    def _default_csp(self) -> str:
        """Generate a secure default Content-Security-Policy."""
        directives = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        return "; ".join(directives)

    def _build_hsts_header(self) -> str:
        """Build the HSTS header value."""
        # Validate max-age is a positive integer
        if not isinstance(self.hsts_max_age, int) or self.hsts_max_age < 0:
            raise ValueError(
                f"hsts_max_age must be a non-negative integer, got {self.hsts_max_age}"
            )
        parts = [f"max-age={self.hsts_max_age}"]
        if self.hsts_include_subdomains:
            parts.append("includeSubDomains")
        if self.hsts_preload:
            parts.append("preload")
        return "; ".join(parts)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        """Process the request and add security headers to the response."""
        response = await call_next(request)

        # Add default security headers
        for header, value in self.DEFAULT_HEADERS.items():
            response.headers[header] = value

        # Add HSTS header if enabled
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = self._build_hsts_header()

        # Add Content-Security-Policy
        response.headers["Content-Security-Policy"] = self.content_security_policy

        # Add Cache-Control for sensitive endpoints
        if self._is_sensitive_endpoint(request.url.path):
            response.headers["Cache-Control"] = self.cache_control
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        # Add custom headers
        for header, value in self.custom_headers.items():
            response.headers[header] = value

        return response

    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if the endpoint handles sensitive data."""
        sensitive_patterns = [
            "/evaluate",
            "/status",
            "/metrics",
            "/health",
            "/api/",
        ]
        return any(pattern in path for pattern in sensitive_patterns)


def create_security_middleware(
    enable_hsts: bool = True,
    custom_csp: Optional[str] = None,
    additional_headers: Optional[Dict[str, str]] = None,
) -> type:
    """
    Factory function to create a configured SecurityHeadersMiddleware.

    Args:
        enable_hsts: Whether to enable HSTS
        custom_csp: Custom Content-Security-Policy
        additional_headers: Additional headers to include

    Returns:
        Configured middleware class
    """

    class ConfiguredSecurityMiddleware(SecurityHeadersMiddleware):
        def __init__(self, app: Any) -> None:
            super().__init__(
                app,
                enable_hsts=enable_hsts,
                content_security_policy=custom_csp,
                custom_headers=additional_headers,
            )

    return ConfiguredSecurityMiddleware
