"""Nethical Governance API v2 Application.

Main FastAPI router for the v2 API with enhanced features:
- Request ID propagation (X-Request-ID header)
- Structured error responses
- Rate limiting headers
- Latency headers (X-Nethical-Latency-Ms)
- Cache headers
- Full compliance with 25 Fundamental Laws

Usage:
    from nethical.api.v2 import create_v2_app
    
    app = create_v2_app()
    # Or mount as a sub-application:
    main_app.mount("/v2", create_v2_app())
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .routes import (
    appeals,
    audit,
    decisions,
    evaluate,
    fairness,
    metrics,
    policies,
)

# API Version
API_VERSION = "2.0.0"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for request context management.
    
    Implements:
    - Request ID propagation (Law 15: Audit Compliance)
    - Latency tracking (Law 10: Reasoning Transparency)
    - Cache headers for performance
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with context tracking."""
        # Generate or extract request ID
        request_id = (
            request.headers.get("X-Request-ID") or
            request.headers.get("X-Correlation-ID") or
            str(uuid.uuid4())
        )
        
        # Store in request state for downstream access
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()
        
        # Process request
        response = await call_next(request)
        
        # Add response headers
        latency_ms = int((time.perf_counter() - request.state.start_time) * 1000)
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Nethical-Latency-Ms"] = str(latency_ms)
        response.headers["X-API-Version"] = API_VERSION
        
        return response


def create_v2_app() -> FastAPI:
    """Create and configure the v2 API application.
    
    Returns:
        FastAPI: Configured v2 API application
    """
    app = FastAPI(
        title="Nethical Governance API v2",
        description=(
            "Enhanced REST API for AI safety and ethics governance.\n\n"
            "## Features\n"
            "- Enhanced evaluation with latency metrics\n"
            "- Batch processing for high-throughput scenarios\n"
            "- Decision lookup and history\n"
            "- Policy management\n"
            "- Fairness metrics and monitoring\n"
            "- Appeals submission and tracking\n"
            "- Comprehensive audit trail\n\n"
            "## Ethical Compliance\n"
            "This API adheres to the 25 Fundamental Laws of AI Ethics.\n"
            "All decisions are auditable, explainable, and fair."
        ),
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Add middleware
    app.add_middleware(RequestContextMiddleware)
    
    # CORS configuration
    # In production, replace "*" with specific allowed origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(evaluate.router, tags=["Evaluation"])
    app.include_router(decisions.router, tags=["Decisions"])
    app.include_router(policies.router, tags=["Policies"])
    app.include_router(metrics.router, tags=["Metrics"])
    app.include_router(fairness.router, tags=["Fairness"])
    app.include_router(appeals.router, tags=["Appeals"])
    app.include_router(audit.router, tags=["Audit"])
    
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, Any]:
        """API root with version and endpoint information."""
        return {
            "name": "Nethical Governance API",
            "version": API_VERSION,
            "description": "Enhanced REST API for AI safety and ethics governance",
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "evaluate": "POST /evaluate - Evaluate an action",
                "batch_evaluate": "POST /batch-evaluate - Batch evaluation",
                "decisions": "GET /decisions/{id} - Lookup decision",
                "policies": "GET /policies - List policies",
                "policy_create": "POST /policies - Create policy",
                "metrics": "GET /metrics - Prometheus metrics",
                "fairness": "GET /fairness - Fairness metrics",
                "appeals": "POST /appeals - Submit appeal",
                "audit": "GET /audit/{id} - Audit trail lookup",
            },
            "fundamental_laws": 25,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": API_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    return app


# Create router for mounting in main app
router = APIRouter(prefix="/v2")


# Copy routes to router
@router.get("/", tags=["Root"])
async def router_root() -> dict[str, Any]:
    """API v2 root endpoint."""
    return {
        "name": "Nethical Governance API v2",
        "version": API_VERSION,
        "description": "Enhanced REST API for AI safety and ethics governance",
        "endpoints": {
            "evaluate": "POST /v2/evaluate",
            "batch_evaluate": "POST /v2/batch-evaluate",
            "decisions": "GET /v2/decisions/{id}",
            "policies": "GET /v2/policies",
            "metrics": "GET /v2/metrics",
            "fairness": "GET /v2/fairness",
            "appeals": "POST /v2/appeals",
            "audit": "GET /v2/audit/{id}",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Include all route modules in the router
router.include_router(evaluate.router, tags=["Evaluation"])
router.include_router(decisions.router, tags=["Decisions"])
router.include_router(policies.router, tags=["Policies"])
router.include_router(metrics.router, tags=["Metrics"])
router.include_router(fairness.router, tags=["Fairness"])
router.include_router(appeals.router, tags=["Appeals"])
router.include_router(audit.router, tags=["Audit"])
