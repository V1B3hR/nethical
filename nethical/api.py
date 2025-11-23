"""Main REST API for Nethical v2.0 - Production-ready evaluation endpoints.

This module provides FastAPI endpoints for remote evaluation of agent actions
with structured JudgmentResult responses. Designed for integration with external
platforms (LangChain, MCP, OpenAI tool wrappers).

Endpoints:
    POST /evaluate - Evaluate an action and return structured judgment
    GET /status - Get governance system status
    GET /metrics - Get violation and judgment metrics

Features (v2.1):
    - Rate limiting: Per-identity request limits (configurable)
    - Authentication: API key-based auth (optional)
    - Input validation: Size caps and sanitization
    - Concurrency control: Semaphore-based request limiting
    - Timeout guards: Prevent runaway evaluations
    - Semantic caching: LRU+TTL cache for repeated queries

Environment Variables:
    NETHICAL_API_KEYS: Comma-separated API keys for authentication (optional)
    NETHICAL_RATE_BURST: Requests per second burst limit (default: 5)
    NETHICAL_RATE_SUSTAINED: Requests per minute sustained limit (default: 100)
    NETHICAL_MAX_CONCURRENCY: Max concurrent evaluations (default: 100)
    NETHICAL_EVAL_TIMEOUT: Evaluation timeout in seconds (default: 30)
    NETHICAL_MAX_INPUT_SIZE: Max chars for intent+action (default: 4096)
    NETHICAL_CACHE_MAXSIZE: Cache max entries (default: 20000)
    NETHICAL_CACHE_TTL: Cache TTL in seconds (default: 600)
    NETHICAL_CORS_ALLOW_ORIGINS: CORS allowed origins (default: *)

Usage:
    # Start the server
    uvicorn nethical.api:app --host 0.0.0.0 --port 8000
    
    # Or with auto-reload during development
    uvicorn nethical.api:app --reload --port 8000
    
    # With authentication
    NETHICAL_API_KEYS=key1,key2 uvicorn nethical.api:app --port 8000
"""

from __future__ import annotations

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import governance system
try:
    from nethical.core.integrated_governance import IntegratedGovernance
    from nethical.core.models import AgentAction, MonitoringConfig
    INTEGRATED_GOVERNANCE_AVAILABLE = True
except ImportError:
    try:
        from nethical.core.governance import SafetyGovernance as IntegratedGovernance
        from nethical.core.governance import AgentAction, MonitoringConfig
        INTEGRATED_GOVERNANCE_AVAILABLE = False
    except ImportError:
        IntegratedGovernance = None
        AgentAction = None
        MonitoringConfig = None

# Import API security components
from nethical.api.rate_limiter import TokenBucketLimiter, RateLimitConfig
from nethical.api.auth import AuthManager

logger = logging.getLogger(__name__)

# Global instances
governance: Optional[IntegratedGovernance] = None
rate_limiter: Optional[TokenBucketLimiter] = None
auth_manager: Optional[AuthManager] = None
concurrency_semaphore: Optional[asyncio.Semaphore] = None

# Configuration from environment
MAX_INPUT_SIZE = int(os.getenv("NETHICAL_MAX_INPUT_SIZE", "4096"))
MAX_CONCURRENCY = int(os.getenv("NETHICAL_MAX_CONCURRENCY", "100"))
EVAL_TIMEOUT = int(os.getenv("NETHICAL_EVAL_TIMEOUT", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for initialization and cleanup."""
    global governance, rate_limiter, auth_manager, concurrency_semaphore
    
    try:
        if IntegratedGovernance is None:
            logger.error("Governance system not available - check imports")
            raise RuntimeError("Governance system not available")
            
        # Initialize governance with semantic monitoring enabled by default
        config = MonitoringConfig(use_semantic_intent=True) if MonitoringConfig else None
        
        if INTEGRATED_GOVERNANCE_AVAILABLE:
            governance = IntegratedGovernance(
                storage_dir="./nethical_api_data",
                enable_performance_optimization=True,
                enable_merkle_anchoring=True,
                enable_ethical_taxonomy=True,
                enable_sla_monitoring=True,
            )
        else:
            # Fallback to basic SafetyGovernance
            governance = IntegratedGovernance(config=config)
            
        logger.info("Nethical v2.0 governance system initialized with semantic monitoring")
        
        # Initialize rate limiter
        rate_config = RateLimitConfig(
            requests_per_second=float(os.getenv("NETHICAL_RATE_BURST", "5.0")),
            requests_per_minute=int(os.getenv("NETHICAL_RATE_SUSTAINED", "100"))
        )
        rate_limiter = TokenBucketLimiter(config=rate_config)
        logger.info("Rate limiter initialized")
        
        # Initialize auth manager
        auth_manager = AuthManager()
        if auth_manager.is_permissive():
            logger.warning("Running in PERMISSIVE MODE - no authentication required")
        
        # Initialize concurrency semaphore
        concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        logger.info(f"Concurrency semaphore initialized (max={MAX_CONCURRENCY})")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize API components: {e}")
        raise
    finally:
        logger.info("Nethical API shutting down")


# Create FastAPI app
app = FastAPI(
    title="Nethical v2.0 Governance API",
    description="Production-ready REST API for AI safety and ethics governance with semantic monitoring",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS for integration scenarios
# SECURITY WARNING: Default allows all origins for development/demo
# For production, set NETHICAL_CORS_ALLOW_ORIGINS to specific domains
allowed_origins_str = os.getenv("NETHICAL_CORS_ALLOW_ORIGINS", "*")
allowed_origins = allowed_origins_str.split(",") if allowed_origins_str != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class EvaluateRequest(BaseModel):
    """Request model for action evaluation."""
    
    id: Optional[str] = Field(
        default=None,
        description="Optional action ID (auto-generated if not provided)"
    )
    agent_id: str = Field(
        ...,
        description="Agent identifier",
        min_length=1,
        max_length=256
    )
    stated_intent: Optional[str] = Field(
        default=None,
        description="The stated intent or goal of the action"
    )
    actual_action: str = Field(
        ...,
        description="The actual action, code, or content to evaluate",
        min_length=1,
        max_length=50000
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about the action"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for evaluation"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "agent_id": "gpt-4-agent",
                    "stated_intent": "Process user data",
                    "actual_action": "SELECT * FROM users WHERE email LIKE '%@example.com'",
                    "context": {"operation": "database_query"},
                    "parameters": {"strict_mode": True}
                }
            ]
        }
    }


class JudgmentResult(BaseModel):
    """Structured judgment result from evaluation."""
    
    judgment_id: str = Field(..., description="Unique judgment identifier")
    action_id: str = Field(..., description="Action identifier")
    decision: str = Field(
        ..., 
        description="Decision: ALLOW, WARN, BLOCK, QUARANTINE, ESCALATE, or TERMINATE"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation for the decision")
    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected violations"
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall risk score"
    )
    modifications: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Suggested modifications if applicable"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata from evaluation"
    )


class StatusResponse(BaseModel):
    """Governance system status response."""
    
    status: str = Field(..., description="System status: healthy, degraded, or unhealthy")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    semantic_monitoring: bool = Field(..., description="Whether semantic monitoring is enabled")
    semantic_available: bool = Field(..., description="Whether semantic models are available")
    components: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of various components"
    )


class MetricsResponse(BaseModel):
    """System metrics response."""
    
    total_evaluations: int = Field(..., description="Total number of evaluations")
    total_violations: int = Field(..., description="Total violations detected")
    violation_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Violations grouped by type"
    )
    decisions_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Decisions grouped by type"
    )
    avg_confidence: Optional[float] = Field(
        default=None,
        description="Average confidence score"
    )
    timestamp: str = Field(..., description="Metrics timestamp")


# ============================================================================
# Helper Functions
# ============================================================================

def extract_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """Extract API key from headers (X-API-Key or Authorization: Bearer)."""
    if x_api_key:
        return x_api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]  # Remove "Bearer " prefix
    return None


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request, handling proxies.
    
    SECURITY NOTE: X-Forwarded-For can be spoofed by clients. In production,
    ensure your reverse proxy/load balancer strips untrusted X-Forwarded-For
    headers and sets it correctly. Or configure trusted proxy IPs and validate.
    """
    # Check X-Forwarded-For header (from proxies/load balancers)
    # IMPORTANT: Only trust this if behind a trusted proxy that sets it correctly
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP in chain
        # TODO: In production, validate against trusted proxy list
        return forwarded.split(",")[0].strip()
    
    # Fallback to direct client
    if request.client:
        return request.client.host
    
    return "unknown"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nethical v2.1 Governance API",
        "version": "2.1.0",
        "description": "Production-ready REST API for AI safety and ethics governance",
        "features": [
            "Semantic intent deviation monitoring",
            "Adversarial prompt detection",
            "Real-time safety evaluation",
            "Structured judgment results",
            "Rate limiting and authentication",
            "Input validation and sanitization",
            "Concurrency control and timeouts"
        ],
        "endpoints": {
            "evaluate": "POST /evaluate - Evaluate an action",
            "status": "GET /status - System status",
            "metrics": "GET /metrics - System metrics",
            "docs": "GET /docs - Interactive API documentation"
        },
        "security": {
            "authentication": "Optional API key via X-API-Key or Authorization: Bearer",
            "rate_limiting": "Per-identity request limits",
            "input_validation": f"Max {MAX_INPUT_SIZE} chars for intent+action"
        }
    }


@app.post("/evaluate", response_model=JudgmentResult)
async def evaluate(
    eval_request: EvaluateRequest,
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> JudgmentResult:
    """Evaluate an action for ethical compliance and safety.
    
    This endpoint processes actions through Nethical's governance system with
    semantic monitoring enabled. Includes rate limiting, authentication,
    input validation, and concurrency control.
    
    Security Features:
    - Authentication: Requires valid API key if NETHICAL_API_KEYS is set
    - Rate Limiting: Per-identity request limits (burst and sustained)
    - Input Validation: Maximum input size enforced
    - Concurrency Control: Limited concurrent evaluations
    - Timeout Guard: Evaluation timeout to prevent hangs
    
    Args:
        eval_request: EvaluateRequest with action details
        request: FastAPI request object (for IP extraction)
        x_api_key: Optional API key from X-API-Key header
        authorization: Optional API key from Authorization header
        
    Returns:
        JudgmentResult with decision and evaluation details
        
    Raises:
        HTTPException: 
            - 401: Invalid or missing API key (when auth required)
            - 413: Input size exceeds limit
            - 429: Rate limit exceeded
            - 503: Service unavailable (governance not initialized or timeout)
            - 500: Internal evaluation error
    """
    # Check system availability
    if governance is None or rate_limiter is None or auth_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable - system not fully initialized"
        )
    
    # Extract API key and client IP
    api_key = extract_api_key(x_api_key, authorization)
    client_ip = get_client_ip(request)
    
    # Authentication check (if keys are configured)
    if not auth_manager.is_permissive():
        if not api_key or not auth_manager.validate_key(api_key):
            logger.warning(f"Unauthorized access attempt from {client_ip}")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized - valid API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    # Determine identity for rate limiting
    identity = auth_manager.extract_identity(api_key, client_ip)
    
    # Rate limiting check
    allowed, retry_after, rate_info = await rate_limiter.is_allowed(identity)
    if not allowed:
        logger.warning(f"Rate limit exceeded for {identity}")
        headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"]),
            "Retry-After": str(int(retry_after)) if retry_after else "60"
        }
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {int(retry_after)}s",
            headers=headers
        )
    
    # Input size validation
    intent_len = len(eval_request.stated_intent or "")
    action_len = len(eval_request.actual_action)
    total_input_size = intent_len + action_len
    
    if total_input_size > MAX_INPUT_SIZE:
        logger.warning(
            f"Input size {total_input_size} exceeds limit {MAX_INPUT_SIZE} for {identity}"
        )
        raise HTTPException(
            status_code=413,
            detail=f"Input too large: {total_input_size} chars (max: {MAX_INPUT_SIZE})"
        )
    
    # Concurrency control - limit concurrent evaluations
    if concurrency_semaphore is None:
        raise HTTPException(status_code=503, detail="Concurrency control not initialized")
    
    async with concurrency_semaphore:
        try:
            # Evaluation with timeout
            action_id = eval_request.id or f"action_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Build action - handle both IntegratedGovernance and basic SafetyGovernance
            async def do_evaluation():
                if hasattr(governance, 'process_action'):
                    # IntegratedGovernance interface
                    return governance.process_action(
                        action=eval_request.actual_action,
                        agent_id=eval_request.agent_id,
                        action_type=eval_request.parameters.get("action_type", "query") if eval_request.parameters else "query",
                        stated_intent=eval_request.stated_intent,
                        context=eval_request.context or {}
                    )
                else:
                    # Basic SafetyGovernance interface
                    action = AgentAction(
                        action_id=action_id,
                        agent_id=eval_request.agent_id,
                        action_type=eval_request.parameters.get("action_type", "query") if eval_request.parameters else "query",
                        content=eval_request.actual_action,
                        metadata=eval_request.parameters or {},
                        context=eval_request.context or {},
                        intent=eval_request.stated_intent,
                    )
                    return governance.evaluate_action(action)
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(do_evaluation(), timeout=EVAL_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"Evaluation timeout ({EVAL_TIMEOUT}s) for {identity}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Evaluation timeout after {EVAL_TIMEOUT}s. Please retry with simpler input.",
                    headers={"Retry-After": "10"}
                )
            
            # Convert result to response model
            judgment_id = getattr(result, 'judgment_id', f"judgment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}")
            decision = str(getattr(result, 'decision', 'ALLOW')).upper()
            confidence = float(getattr(result, 'confidence', 0.9))
            reasoning = str(getattr(result, 'reasoning', 'Action evaluated'))
            
            # Extract violations
            violations_list = []
            violations = getattr(result, 'violations', [])
            for v in violations:
                violations_list.append({
                    "id": getattr(v, 'id', ''),
                    "type": str(getattr(v, 'violation_type', '')),
                    "severity": str(getattr(v, 'severity', '')),
                    "description": str(getattr(v, 'description', '')),
                    "evidence": dict(getattr(v, 'evidence', {}))
                })
            
            # Build response with rate limit info
            judgment = JudgmentResult(
                judgment_id=judgment_id,
                action_id=action_id,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                violations=violations_list,
                timestamp=datetime.now(timezone.utc).isoformat(),
                risk_score=float(getattr(result, 'risk_score', 0.0)) if hasattr(result, 'risk_score') else None,
                modifications=dict(getattr(result, 'modifications', {})) if hasattr(result, 'modifications') else None,
                metadata={
                    "semantic_monitoring": True,
                    "agent_id": eval_request.agent_id,
                    "has_intent": eval_request.stated_intent is not None,
                    "rate_limit": rate_info
                }
            )
            
            # Log successful evaluation
            logger.info(
                f"Evaluation complete for {identity}: decision={decision}, "
                f"confidence={confidence:.3f}, violations={len(violations_list)}"
            )
            
            return judgment
            
        except HTTPException:
            # Re-raise HTTP exceptions (timeout, etc.)
            raise
        except Exception as e:
            logger.error(f"Evaluation failed for {identity}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed: {str(e)}"
            )


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Get governance system status and health information.
    
    Returns:
        StatusResponse with system status and component health including
        rate limiter, auth manager, and concurrency control
    """
    if governance is None:
        raise HTTPException(
            status_code=503,
            detail="Governance system not initialized"
        )
    
    try:
        # Check semantic availability
        try:
            from nethical.core.semantics import is_semantic_available
            semantic_available = is_semantic_available()
        except ImportError:
            semantic_available = False
        
        # Get component status
        components = {}
        if hasattr(governance, 'components_enabled'):
            components = governance.components_enabled
        else:
            components = {
                "ethical_monitoring": True,
                "safety_monitoring": True,
                "intent_monitoring": True,
            }
        
        # Add API layer components
        components["rate_limiter"] = rate_limiter is not None
        components["auth_manager"] = auth_manager is not None
        components["concurrency_control"] = concurrency_semaphore is not None
        
        # Add auth and rate limiter stats
        auth_stats = auth_manager.get_stats() if auth_manager else {}
        rate_stats = rate_limiter.get_stats() if rate_limiter else {}
        
        return StatusResponse(
            status="healthy",
            version="2.1.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            semantic_monitoring=True,
            semantic_available=semantic_available,
            components=components
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Get governance system metrics.
    
    Returns:
        MetricsResponse with evaluation and violation statistics
    """
    if governance is None:
        raise HTTPException(
            status_code=503,
            detail="Governance system not initialized"
        )
    
    try:
        # Get metrics from governance system
        if hasattr(governance, 'get_system_metrics'):
            sys_metrics = governance.get_system_metrics()
            
            return MetricsResponse(
                total_evaluations=sys_metrics.get('total_actions', 0),
                total_violations=sys_metrics.get('total_violations', 0),
                violation_by_type=sys_metrics.get('violations_by_type', {}),
                decisions_by_type=sys_metrics.get('decisions_by_type', {}),
                avg_confidence=sys_metrics.get('avg_confidence'),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        else:
            # Minimal metrics if not available
            return MetricsResponse(
                total_evaluations=0,
                total_violations=0,
                violation_by_type={},
                decisions_by_type={},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics retrieval failed: {str(e)}"
        )


# ============================================================================
# Health Check (for load balancers / orchestrators)
# ============================================================================

@app.get("/health")
async def health():
    """Simple health check endpoint for load balancers."""
    if governance is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
