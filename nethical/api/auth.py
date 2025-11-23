"""
Nethical Governance API (Improved Version)

Endpoints:
    POST /evaluate  - Evaluate an action
    GET  /status    - System and component status
    GET  /metrics   - Metrics snapshot
    GET  /           - API overview

Features (v2.2 proposed improvements):
    - Rate limiting: Per-identity (burst + sustained) with headers on all responses
    - Authentication: API key-based (optional, permissive if unset)
    - Input validation: Size caps on intent+action plus optional parameter/context limits
    - Concurrency control: Semaphore-based limiting
    - Timeout guards: Evaluation path protected
    - Semantic cache: LRU+TTL for similarity between intent and action (stores only floats)
    - Extended observability: Rate limit headers + X-Eval-Duration-ms
    - Enhanced status reporting (config + component stats)

Environment Variables:
    NETHICAL_API_KEYS            Comma-separated API keys (optional)
    NETHICAL_RATE_BURST          Requests per second burst (default: 5)
    NETHICAL_RATE_SUSTAINED      Requests per minute sustained (default: 100)
    NETHICAL_MAX_CONCURRENCY     Max concurrent evaluations (default: 100)
    NETHICAL_EVAL_TIMEOUT        Evaluation timeout seconds (default: 30)
    NETHICAL_MAX_INPUT_SIZE      Max chars (intent + action) (default: 4096)
    NETHICAL_CACHE_MAXSIZE       Semantic cache entries (default: 20000)
    NETHICAL_CACHE_TTL           Semantic cache TTL seconds (default: 600)
    NETHICAL_CORS_ALLOW_ORIGINS  Comma-separated origins or * (default: *)
    NETHICAL_MAX_PARAM_KEYS      Max keys in parameters dict (default: 100)
    NETHICAL_MAX_CONTEXT_SIZE    Max serialized context char length (default: 10000)

Usage:
    uvicorn nethical.api:app --host 0.0.0.0 --port 8000
    NETHICAL_API_KEYS=key1,key2 uvicorn nethical.api:app --port 8000
"""

from __future__ import annotations

import os
import asyncio
import logging
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Attempt imports for governance system
try:
    from nethical.core.integrated import IntegratedGovernance
    from nethical.core.models import AgentAction, MonitoringConfig
except ImportError:
    IntegratedGovernance = None
    AgentAction = None
    MonitoringConfig = None

# API security components
from nethical.api.rate_limiter import TokenBucketLimiter, RateLimitConfig
from nethical.api.auth import AuthManager
try:
    from nethical.api.semantic_cache import SemanticCache
except ImportError:
    SemanticCache = None

logger = logging.getLogger(__name__)

# Global instances
governance: Optional[IntegratedGovernance] = None
rate_limiter: Optional[TokenBucketLimiter] = None
auth_manager: Optional[AuthManager] = None
concurrency_semaphore: Optional[asyncio.Semaphore] = None
semantic_cache: Optional[SemanticCache] = None

# Configuration from environment
MAX_INPUT_SIZE = int(os.getenv("NETHICAL_MAX_INPUT_SIZE", "4096"))
MAX_CONCURRENCY = int(os.getenv("NETHICAL_MAX_CONCURRENCY", "100"))
EVAL_TIMEOUT = int(os.getenv("NETHICAL_EVAL_TIMEOUT", "30"))
MAX_PARAM_KEYS = int(os.getenv("NETHICAL_MAX_PARAM_KEYS", "100"))
MAX_CONTEXT_SIZE = int(os.getenv("NETHICAL_MAX_CONTEXT_SIZE", "10000"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global governance, rate_limiter, auth_manager, concurrency_semaphore, semantic_cache
    try:
        # Initialize governance
        if IntegratedGovernance is not None:
            config = MonitoringConfig(use_semantic_intent=True, enable_timings=True)
            # If heavy initialization occurs, consider offloading:
            # governance = await asyncio.to_thread(IntegratedGovernance, config=config)
            governance = IntegratedGovernance(config=config)
            logger.info("Governance system initialized with semantic monitoring")
        else:
            logger.warning("IntegratedGovernance not available - running in degraded mode")

        # Rate limiter
        rate_config = RateLimitConfig(
            requests_per_second=float(os.getenv("NETHICAL_RATE_BURST", "5.0")),
            requests_per_minute=int(os.getenv("NETHICAL_RATE_SUSTAINED", "100"))
        )
        rate_limiter = TokenBucketLimiter(config=rate_config)
        logger.info("Rate limiter initialized")

        # Auth manager
        auth_manager = AuthManager()
        if auth_manager.is_permissive():
            logger.warning("PERMISSIVE MODE - authentication disabled")

        # Concurrency semaphore
        concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        logger.info(f"Concurrency semaphore ready (max={MAX_CONCURRENCY})")

        # Semantic cache
        if SemanticCache:
            semantic_cache = SemanticCache(
                maxsize=int(os.getenv("NETHICAL_CACHE_MAXSIZE", "20000")),
                ttl=int(os.getenv("NETHICAL_CACHE_TTL", "600")),
                model_version="v2"
            )
            logger.info("Semantic cache initialized")
        else:
            logger.warning("SemanticCache not available - caching disabled")

        yield
    except Exception as e:
        logger.error(f"Initialization failure: {e}", exc_info=True)
        raise
    finally:
        logger.info("Nethical API shutting down")

# Create FastAPI app
app = FastAPI(
    title="Nethical Governance API",
    description="Production-ready REST API for AI safety and ethics governance",
    version="2.2.0-proposed",
    lifespan=lifespan
)

# CORS
allowed_origins_str = os.getenv("NETHICAL_CORS_ALLOW_ORIGINS", "*")
allowed_origins = allowed_origins_str.split(",") if allowed_origins_str != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic Models
# =========================

class EvaluateRequest(BaseModel):
    id: Optional[str] = Field(None, description="Optional action ID")
    agent_id: str = Field(..., description="Unique agent identifier")
    stated_intent: Optional[str] = Field(None, description="Declared intent of the agent")
    actual_action: str = Field(..., description="Concrete action content (e.g., query)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Structured parameters")


class JudgmentResult(BaseModel):
    judgment_id: str
    action_id: str
    decision: str
    confidence: float
    reasoning: str
    violations: List[Dict[str, Any]]
    timestamp: str
    risk_score: Optional[float] = None
    modifications: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class StatusResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    semantic_monitoring: bool
    semantic_available: bool
    components: Dict[str, Any]
    config: Dict[str, Any]


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    timestamp: str

# =========================
# Helper Functions
# =========================

def extract_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    if x_api_key:
        return x_api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"

def validate_payload(eval_request: EvaluateRequest) -> None:
    total_len = len(eval_request.actual_action) + len(eval_request.stated_intent or "")
    if total_len > MAX_INPUT_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Input too large: {total_len} chars (max: {MAX_INPUT_SIZE})"
        )
    if eval_request.parameters:
        if len(eval_request.parameters.keys()) > MAX_PARAM_KEYS:
            raise HTTPException(
                status_code=413,
                detail=f"Too many parameter keys ({len(eval_request.parameters)} > {MAX_PARAM_KEYS})"
            )
    if eval_request.context:
        # Serialize context to approximate size
        try:
            serialized = json.dumps(eval_request.context)
        except Exception:
            serialized = str(eval_request.context)
        if len(serialized) > MAX_CONTEXT_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Context too large ({len(serialized)} > {MAX_CONTEXT_SIZE})"
            )

async def compute_semantic_similarity(intent: Optional[str], action: str) -> float:
    """
    Placeholder similarity computation.
    If governance exposes a richer semantic pathway, integrate it here.
    Falls back to a trivial heuristic if intent absent.
    """
    if not intent:
        return 0.5  # Neutral when no stated intent
    # Simple normalized token overlap (placeholder)
    intent_tokens = set(intent.lower().split())
    action_tokens = set(action.lower().split())
    if not intent_tokens or not action_tokens:
        return 0.0
    overlap = len(intent_tokens & action_tokens) / len(intent_tokens | action_tokens)
    # Clamp
    return max(0.0, min(1.0, overlap))

# =========================
# Root Endpoint
# =========================

@app.get("/")
async def root():
    return {
        "name": "Nethical v2.2 Governance API",
        "version": "2.2.0-proposed",
        "description": "REST API for AI safety & ethics governance",
        "features": [
            "Semantic intent deviation monitoring",
            "Adversarial prompt detection",
            "Rate limiting & authentication",
            "Input validation",
            "Concurrency control & timeouts",
            "Semantic similarity caching"
        ],
        "security": {
            "authentication": "API key via X-API-Key or Authorization: Bearer (optional)",
            "rate_limiting": "Per-identity burst + sustained limits",
            "input_validation": f"Max {MAX_INPUT_SIZE} chars intent+action"
        },
        "endpoints": {
            "evaluate": "POST /evaluate",
            "status": "GET /status",
            "metrics": "GET /metrics",
            "docs": "GET /docs"
        }
    }

# =========================
# Evaluate Endpoint
# =========================

@app.post("/evaluate", response_model=JudgmentResult)
async def evaluate(
    eval_request: EvaluateRequest,
    request: Request,
    response: Response,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> JudgmentResult:
    start_time = time.perf_counter()

    # Availability checks
    if governance is None or rate_limiter is None or auth_manager is None:
        raise HTTPException(status_code=503, detail="Service unavailable - not fully initialized")

    # Extract API key / IP
    api_key = extract_api_key(x_api_key, authorization)
    client_ip = get_client_ip(request)

    # Authentication
    if not auth_manager.is_permissive():
        if not api_key or not auth_manager.validate_key(api_key):
            logger.warning(f"Unauthorized attempt from {client_ip}")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized - valid API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )

    identity = auth_manager.extract_identity(api_key, client_ip)

    # Rate limiting
    allowed, retry_after, rate_info = await rate_limiter.is_allowed(identity)
    if not allowed:
        headers = {
            "X-RateLimit-Limit": str(rate_info.get("limit")),
            "X-RateLimit-Burst-Limit": str(rate_info.get("burst_limit", "")),
            "X-RateLimit-Remaining": str(rate_info.get("remaining")),
            "X-RateLimit-Reset": str(rate_info.get("reset")),
            "Retry-After": str(int(retry_after)) if retry_after else "60"
        }
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers=headers)

    # Input validation
    validate_payload(eval_request)

    # Concurrency control
    if concurrency_semaphore is None:
        raise HTTPException(status_code=503, detail="Concurrency control not initialized")

    async with concurrency_semaphore:
        try:
            action_id = eval_request.id or f"action_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            async def do_evaluation():
                if hasattr(governance, "process_action"):
                    return governance.process_action(
                        action=eval_request.actual_action,
                        agent_id=eval_request.agent_id,
                        action_type=eval_request.parameters.get("action_type", "query") if eval_request.parameters else "query",
                        stated_intent=eval_request.stated_intent,
                        context=eval_request.context or {}
                    )
                else:
                    # Fallback path (if only basic governance)
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

            try:
                result = await asyncio.wait_for(do_evaluation(), timeout=EVAL_TIMEOUT)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail=f"Evaluation timeout after {EVAL_TIMEOUT}s",
                    headers={"Retry-After": "10"}
                )

            # Derive semantic similarity with caching (if available)
            similarity: Optional[float] = None
            if semantic_cache and eval_request.stated_intent:
                async def compute_fn():
                    return await compute_semantic_similarity(eval_request.stated_intent, eval_request.actual_action)
                similarity = await semantic_cache.get_or_compute(
                    eval_request.stated_intent,
                    eval_request.actual_action,
                    compute_fn,
                    config_params={"model_version": "v2"}
                )
            elif eval_request.stated_intent:
                similarity = await compute_semantic_similarity(eval_request.stated_intent, eval_request.actual_action)

            judgment_id = getattr(result, "judgment_id", f"judgment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}")
            decision = str(getattr(result, "decision", "ALLOW")).upper()
            confidence = float(getattr(result, "confidence", 0.9))
            reasoning = str(getattr(result, "reasoning", "Action evaluated"))

            violations_list = []
            violations = getattr(result, "violations", [])
            for v in violations:
                violations_list.append({
                    "id": getattr(v, "id", ""),
                    "type": str(getattr(v, "violation_type", "")),
                    "severity": str(getattr(v, "severity", "")),
                    "description": str(getattr(v, "description", "")),
                    "evidence": dict(getattr(v, "evidence", {}))
                })

            # Attach headers for rate limiting on success
            response.headers["X-RateLimit-Limit"] = str(rate_info.get("limit"))
            response.headers["X-RateLimit-Burst-Limit"] = str(rate_info.get("burst_limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(rate_info.get("remaining"))
            response.headers["X-RateLimit-Reset"] = str(rate_info.get("reset"))

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            response.headers["X-Eval-Duration-ms"] = str(duration_ms)

            metadata = {
                "semantic_monitoring": True,
                "agent_id": eval_request.agent_id,
                "has_intent": eval_request.stated_intent is not None,
                "rate_limit": rate_info,
                "similarity_cached": semantic_cache is not None,
                "intent_action_similarity": similarity
            }

            judgment = JudgmentResult(
                judgment_id=judgment_id,
                action_id=action_id,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                violations=violations_list,
                timestamp=datetime.now(timezone.utc).isoformat(),
                risk_score=float(getattr(result, "risk_score", 0.0)) if hasattr(result, "risk_score") else None,
                modifications=dict(getattr(result, "modifications", {})) if hasattr(result, "modifications") else None,
                metadata=metadata
            )

            logger.info(
                f"Evaluation identity={identity} decision={decision} confidence={confidence:.3f} "
                f"violations={len(violations_list)} duration_ms={duration_ms}"
            )
            return judgment

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Evaluation failed for {identity}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# =========================
# Status Endpoint
# =========================

@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    if governance is None:
        raise HTTPException(status_code=503, detail="Governance not initialized")

    semantic_available = hasattr(governance, "process_action")
    components = {
        "governance": governance is not None,
        "rate_limiter": rate_limiter is not None,
        "auth_manager": auth_manager is not None,
        "concurrency_control": concurrency_semaphore is not None,
        "semantic_cache": semantic_cache is not None
    }

    # Stats
    auth_stats = auth_manager.get_stats() if auth_manager else {}
    rate_stats = rate_limiter.get_stats() if rate_limiter else {}
    cache_stats = semantic_cache.get_stats() if semantic_cache else {}

    config_snapshot = {
        "max_input_size": MAX_INPUT_SIZE,
        "max_concurrency": MAX_CONCURRENCY,
        "eval_timeout_sec": EVAL_TIMEOUT,
        "rate_burst": os.getenv("NETHICAL_RATE_BURST", "5"),
        "rate_sustained": os.getenv("NETHICAL_RATE_SUSTAINED", "100"),
        "cache_enabled": semantic_cache is not None,
        "cache_maxsize": cache_stats.get("maxsize") if cache_stats else None,
        "cache_ttl": cache_stats.get("ttl") if cache_stats else None,
        "permissive_auth": auth_stats.get("permissive_mode"),
        "configured_keys": auth_stats.get("configured_keys"),
    }

    # Merge component stats
    components["rate_limiter_stats"] = rate_stats
    components["auth_stats"] = auth_stats
    components["cache_stats"] = cache_stats

    return StatusResponse(
        status="healthy",
        version="2.2.0-proposed",
        timestamp=datetime.now(timezone.utc).isoformat(),
        semantic_monitoring=True,
        semantic_available=semantic_available,
        components=components,
        config=config_snapshot
    )

# =========================
# Metrics Endpoint (Placeholder)
# =========================

@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    # Placeholder metrics; extend with real counters if available
    metric_blob = {
        "violations_total": 0,
        "judgments_total": 0,
        "cache_hit_rate": (semantic_cache.get_stats().get("hit_rate_percent") if semantic_cache else None),
        "active_identities": (rate_limiter.get_stats().get("active_identities") if rate_limiter else None),
    }
    return MetricsResponse(
        metrics=metric_blob,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
