"""Nethical Governance API v1 Application.

Main FastAPI router for the v1 API with backend enhancements:
- RBAC with JWT authentication
- Agent management (CRUD)
- Policy management (CRUD)  
- Audit log access with Merkle tree verification
- Real-time threat notifications (WebSocket/SSE)

Usage:
    from nethical.api.v1 import create_v1_app
    
    app = create_v1_app()
    # Or mount as a sub-application:
    main_app.mount("/api/v1", create_v1_app())
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .routes import agents, audit, auth, policies, realtime

# API Version
API_VERSION = "1.0.0"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for request context management.
    
    Implements:
    - Request ID propagation
    - Latency tracking
    - Standard response headers
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


def create_v1_app() -> FastAPI:
    """Create and configure the v1 API application.
    
    Returns:
        FastAPI: Configured v1 API application
    """
    app = FastAPI(
        title="Nethical Governance API v1",
        description=(
            "Backend API for AI safety and ethics governance with full management capabilities.\n\n"
            "## Features\n"
            "- **RBAC**: Role-based access control with JWT authentication (admin, auditor, operator)\n"
            "- **Agent Management**: CRUD operations for AI agent configuration\n"
            "- **Policy Management**: CRUD operations for governance policies\n"
            "- **Audit Logs**: Read-only access with Merkle tree verification\n"
            "- **Real-time Threats**: WebSocket and SSE for live threat notifications\n\n"
            "## Authentication\n"
            "Use Bearer token authentication. Get a token from `/api/v1/auth/login`.\n\n"
            "## Roles\n"
            "- **admin**: Full access to all operations\n"
            "- **auditor**: Read-only access to logs and audit data\n"
            "- **operator**: Can evaluate risk, but cannot modify configuration\n\n"
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
    app.include_router(auth.router)
    app.include_router(agents.router)
    app.include_router(policies.router)
    app.include_router(audit.router)
    app.include_router(realtime.router)
    
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, Any]:
        """API root with version and endpoint information."""
        return {
            "name": "Nethical Governance API v1",
            "version": API_VERSION,
            "description": "Backend API for AI safety and ethics governance",
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "auth": {
                    "login": "POST /auth/login - Login and get access token",
                },
                "agents": {
                    "create": "POST /agents - Create new agent",
                    "list": "GET /agents - List all agents",
                    "get": "GET /agents/{id} - Get agent details",
                    "update": "PATCH /agents/{id} - Update agent",
                    "delete": "DELETE /agents/{id} - Delete agent",
                },
                "policies": {
                    "create": "POST /policies - Create new policy",
                    "list": "GET /policies - List all policies",
                    "get": "GET /policies/{id} - Get policy details",
                    "update": "PATCH /policies/{id} - Update policy",
                    "delete": "DELETE /policies/{id} - Delete policy",
                },
                "audit": {
                    "logs": "GET /audit/logs - Get audit logs",
                    "log": "GET /audit/logs/{id} - Get single log",
                    "merkle_tree": "GET /audit/merkle-tree - Get Merkle tree",
                    "verify": "POST /audit/verify - Verify Merkle proof",
                },
                "realtime": {
                    "websocket": "WS /ws/threats - WebSocket for real-time threats",
                    "sse": "GET /sse/threats - Server-Sent Events for threats",
                },
            },
            "authentication": "Bearer token (JWT)",
            "roles": ["admin", "auditor", "operator"],
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
router = APIRouter(prefix="/api/v1")


# Copy routes to router
@router.get("/", tags=["Root"])
async def router_root() -> dict[str, Any]:
    """API v1 root endpoint."""
    return {
        "name": "Nethical Governance API v1",
        "version": API_VERSION,
        "description": "Backend API for AI safety and ethics governance",
        "endpoints": {
            "auth": "POST /api/v1/auth/login",
            "agents": "GET /api/v1/agents",
            "policies": "GET /api/v1/policies",
            "audit": "GET /api/v1/audit/logs",
            "realtime_ws": "WS /api/v1/ws/threats",
            "realtime_sse": "GET /api/v1/sse/threats",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Include all route modules in the router
router.include_router(auth.router)
router.include_router(agents.router)
router.include_router(policies.router)
router.include_router(audit.router)
router.include_router(realtime.router)
