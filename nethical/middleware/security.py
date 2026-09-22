# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Security Headers Middleware

Adds hardened, OWASP-aligned security headers to all HTTP responses, strips server
fingerprinting headers, enforces HTTPS transport constraints for HSTS per RFC 6797,
and prevents clickjacking, MIME-confusion, and cross-origin side-channel attacks.
"""

from __future__ import annotations

import logging
import re
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Set, Type

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Disallowed characters in HTTP headers to prevent CRLF response splitting
_CRLF_PATTERN = re.compile(r"[\r\n]")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds hardened security headers to all HTTP responses.
    
    Security headers included:
    - X-Content-Type-Options: nosniff (prevents MIME type sniffing)
    - X-Frame-Options: DENY (prevents clickjacking attacks)
    - X-XSS-Protection: 0 (OWASP modern standard; disables legacy buggy auditors)
    - Strict-Transport-Security: Enforces HTTPS connections (RFC 6797 compliant)
    - Content-Security-Policy: Modern restrictive policy (blocks object/frame embeddings)
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: Disables risky browser device access
    - Cross-Origin-Opener-Policy (COOP): same-origin (Spectre mitigation)
    - Cross-Origin-Embedder-Policy (COEP): require-corp
    - Cross-Origin-Resource-Policy (CORP): same-origin
    - Cache-Control: Controls caching behavior for sensitive endpoints
    - Server / X-Powered-By Stripping: Removes information disclosure fingerprints
    """

    # Immutable default headers dictionary to prevent class-level mutation
    _DEFAULT_HEADERS_DICT: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "0",  # Modern OWASP standard replaces deprecated "1; mode=block"
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    DEFAULT_HEADERS = MappingProxyType(_DEFAULT_HEADERS_DICT)

    # Core security headers protected from silent custom override
    RESTRICTED_SECURITY_HEADERS: Set[str] = {
        "x-content-type-options",
        "x-frame-options",
        "strict-transport-security",
        "content-security-policy",
        "cross-origin-opener-policy",
        "cross-origin-embedder-policy",
        "cross-origin-resource-policy",
    }

    # Headers stripped from outgoing responses to minimize server fingerprinting
    STRIP_HEADERS: Set[str] = {
        "server",
        "x-powered-by",
    }

    def __init__(
        self,
        app: Any,
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = False,
        trust_proxy: bool = True,
        content_security_policy: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        cache_control: str = "no-store, no-cache, must-revalidate, private",
        allow_security_override: bool = False,
    ) -> None:
        """
        Initialize the security headers middleware with fail-fast configuration validation.
        
        Args:
            app: The ASGI application
            enable_hsts: Whether to enable HTTP Strict Transport Security
            hsts_max_age: Max age for HSTS in seconds (default: 1 year, must be non-negative)
            hsts_include_subdomains: Include subdomains in HSTS
            hsts_preload: Enable HSTS preload
            trust_proxy: Whether to inspect X-Forwarded-Proto / X-Forwarded-SSL for HTTPS detection
            content_security_policy: Custom CSP header value
            custom_headers: Additional custom headers to add (validated against CRLF injection)
            cache_control: Cache-Control header value for sensitive data endpoints
            allow_security_override: If True, custom_headers may override restricted security headers
            
        Raises:
            ValueError: On invalid hsts_max_age or CRLF in custom headers
        """
        super().__init__(app)

        # Fail-fast HSTS validation in __init__
        if not isinstance(hsts_max_age, int) or hsts_max_age < 0:
            raise ValueError(f"hsts_max_age must be a non-negative integer, got {hsts_max_age}")

        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.trust_proxy = trust_proxy
        self.content_security_policy = content_security_policy or self._default_csp()
        self.cache_control = cache_control
        self.allow_security_override = allow_security_override

        # Validate and sanitize custom headers
        self.custom_headers: Dict[str, str] = {}
        if custom_headers:
            for k, v in custom_headers.items():
                if _CRLF_PATTERN.search(k) or _CRLF_PATTERN.search(v):
                    raise ValueError(f"CRLF characters detected in custom header: {k!r}: {v!r}")
                
                lower_k = k.lower()
                if lower_k in self.RESTRICTED_SECURITY_HEADERS and not self.allow_security_override:
                    logger.warning(
                        "Ignoring custom header '%s' to prevent weakening security headers. "
                        "Set allow_security_override=True to permit explicit override.",
                        k
                    )
                    continue
                self.custom_headers[k] = v

    def _default_csp(self) -> str:
        """Generate a secure, OWASP-aligned default Content-Security-Policy."""
        directives = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "object-src 'none'",
            "frame-src 'none'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        return "; ".join(directives)

    def _build_hsts_header(self) -> str:
        """Build the HSTS header value."""
        if not isinstance(self.hsts_max_age, int) or self.hsts_max_age < 0:
            raise ValueError(f"hsts_max_age must be a non-negative integer, got {self.hsts_max_age}")
        parts = [f"max-age={self.hsts_max_age}"]
        if self.hsts_include_subdomains:
            parts.append("includeSubDomains")
        if self.hsts_preload:
            parts.append("preload")
        return "; ".join(parts)

    def _is_https(self, request: Request) -> bool:
        """
        Check if the request was delivered over a secure HTTPS channel per RFC 6797.
        
        Inspects request scheme and trusted reverse proxy headers.
        """
        if request.url.scheme == "https":
            return True
        if self.trust_proxy:
            proto = request.headers.get("x-forwarded-proto", "").lower()
            if proto == "https":
                return True
            ssl_header = request.headers.get("x-forwarded-ssl", "").lower()
            if ssl_header in ("on", "1"):
                return True
        return False

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        """Process the request and apply security headers to the response."""
        response = await call_next(request)

        # 1. Strip server fingerprinting headers
        for strip_hdr in self.STRIP_HEADERS:
            if strip_hdr in response.headers:
                del response.headers[strip_hdr]

        # 2. Add default security headers
        for header, value in self.DEFAULT_HEADERS.items():
            response.headers[header] = value

        # 3. Add HSTS only if request is HTTPS (RFC 6797 Section 7.2 compliance)
        if self.enable_hsts and self._is_https(request):
            response.headers["Strict-Transport-Security"] = self._build_hsts_header()

        # 4. Add Content-Security-Policy
        response.headers["Content-Security-Policy"] = self.content_security_policy

        # 5. Add Cache-Control for sensitive endpoints
        if self._is_sensitive_endpoint(request.url.path):
            response.headers["Cache-Control"] = self.cache_control
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        # 6. Add validated custom headers
        for header, value in self.custom_headers.items():
            response.headers[header] = value

        return response

    def _is_sensitive_endpoint(self, path: str) -> bool:
        """
        Check if the endpoint handles sensitive governance or audit data.
        
        Anchors paths to segment boundaries to prevent false substring positives.
        """
        sensitive_patterns = [
            "/evaluate",
            "/status",
            "/metrics",
            "/health",
            "/api/",
        ]
        # Match if path equals or has pattern as a URL prefix/component
        for pattern in sensitive_patterns:
            if pattern.endswith("/"):
                if pattern in path or path.startswith(pattern.rstrip("/")):
                    return True
            else:
                if path == pattern or path.startswith(f"{pattern}/"):
                    return True
        return False


def create_security_middleware(
    enable_hsts: bool = True,
    custom_csp: Optional[str] = None,
    additional_headers: Optional[Dict[str, str]] = None,
    trust_proxy: bool = True,
    allow_security_override: bool = False,
) -> Type[SecurityHeadersMiddleware]:
    """
    Factory function to create a configured SecurityHeadersMiddleware class.
    
    Args:
        enable_hsts: Whether to enable HSTS
        custom_csp: Custom Content-Security-Policy
        additional_headers: Additional headers to include
        trust_proxy: Whether to trust X-Forwarded-Proto for HTTPS
        allow_security_override: Whether custom headers may override security defaults
    
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
                trust_proxy=trust_proxy,
                allow_security_override=allow_security_override,
            )

    return ConfiguredSecurityMiddleware
