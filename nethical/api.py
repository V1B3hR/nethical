"""
Nethical Governance API (Improved v2.2)

Adds:
    - Rate limit headers on ALL evaluate responses
    - Burst limit exposure
    - Input parameter/context size validation
    - Semantic cache integration
    - Evaluation duration header (X-Eval-Duration-ms)
    - Reloadable authentication (requires updated AuthManager)
    - More detailed /status config snapshot
    - Robust error stratification

Environment variables documented inline.
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

try:
    from nethical.core.integrated import IntegratedGovernance
    from nethical.core.models import AgentAction, MonitoringConfig
except ImportError:
    IntegratedGovernance = None
    AgentAction = None
    MonitoringConfig = None

from nethical.api.rate_limiter import TokenBucketLimiter, RateLimitConfig
from nethical.api.auth import AuthManager
try:
    from nethical.api.semantic_cache import SemanticCache
except ImportError:
    SemanticCache = None

logger = logging.getLogger(__name__)

governance: Optional[IntegratedGovernance] = None
rate_limiter: Optional[TokenBucketLimiter] = None
auth_manager: Optional[AuthManager] = None
concurrency_semaphore: Optional[asyncio.Semaphore] = None
semantic_cache: Optional[SemanticCache] = None

# Configuration
MAX_INPUT_SIZE = int(os.getenv("NETHICAL_MAX_INPUT_SIZE", "4096"))
MAX_CONCURRENCY = int(os.getenv("NETHICAL_MAX_CONCURRENCY", "100"))
EVAL_TIMEOUT = int(os.getenv("NETHICAL_EVAL_TIMEOUT", "30"))
MAX_PARAM_KEYS = int(os.getenv("NETHICAL_MAX_PARAM_KEYS", "100"))
MAX_CONTEXT_SIZE = int(os.getenv("NETHICAL_MAX_CONTEXT_SIZE", "10000"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global governance, rate_limiter, auth_manager, concurrency_semaphore, semantic_cache
    try:
        if IntegratedGovernance:
            config = MonitoringConfig(use_semantic_intent=True, enable_timings=True)
            governance = IntegratedGovernance(config=config)
            logger.info("Governance initialized")
        else:
            logger.warning("IntegratedGovernance unavailable - running degraded")

        rate_config = RateLimitConfig(
            requests_per_second=float(os.getenv("NETHICAL_RATE_BURST", "5.0")),
            requests_per_minute=int(os.getenv("NETHICAL_RATE_SUSTAINED", "100"))
        )
        rate_limiter = TokenBucketLimiter(config=rate_config)

        auth_manager = AuthManager()
        if auth_manager.is_permissive():
            logger.warning("PERMISSIVE MODE active")

        concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        if SemanticCache:
            semantic_cache = SemanticCache(
                maxsize=int(os.getenv("NETHICAL_CACHE_MAXSIZE", "20000")),
                ttl=int(os.getenv("NETHICAL_CACHE_TTL", "600")),
                model_version="v2"
            )
        else:
            logger.warning("SemanticCache unavailable")

        yield
    finally:
        logger.info("API shutdown")

app = FastAPI(
    title="Nethical Governance API",
    version="2.2.0",
    description="Production API for AI safety and ethics governance",
    lifespan=lifespan
)

allowed_origins_str = os.getenv("NETHICAL_CORS_ALLOW_ORIGINS", "*")
allowed_origins = allowed_origins_str.split(",") if allowed_origins_str != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class EvaluateRequest(BaseModel):
    id: Optional[str] = Field(None)
    agent_id: str
    stated_intent: Optional[str] = None
    actual_action: str
    context: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None

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
        raise HTTPException(413, f"Input too large: {total_len} chars (max {MAX_INPUT_SIZE})")
    if eval_request.parameters and len(eval_request.parameters) > MAX_PARAM_KEYS:
        raise HTTPException(413, f"Too many parameter keys ({len(eval_request.parameters)} > {MAX_PARAM_KEYS})")
    if eval_request.context:
        try:
            serialized = json.dumps(eval_request.context)
        except Exception:
            serialized = str(eval_request.context)
        if len(serialized) > MAX_CONTEXT_SIZE:
            raise HTTPException(413, f"Context too large ({len(serialized)} > {MAX_CONTEXT_SIZE})")

async def compute_semantic_similarity(intent: Optional[str], action: str) -> float:
    if not intent:
        return 0.5
    intent_tokens = set(intent.lower().split())
    action_tokens = set(action.lower().split())
    if not intent_tokens or not action_tokens:
        return 0.0
    overlap = len(intent_tokens & action_tokens) / len(intent_tokens | action_tokens)
    return max(0.0, min(1.0, overlap))

@app.get("/")
async def root():
    return {
        "name": "Nethical Governance API",
        "version": "2.2.0",
        "features": [
            "Semantic monitoring",
            "Adversarial detection",
            "Rate limiting & auth",
            "Input validation",
            "Concurrency control",
            "Semantic cache"
        ],
        "endpoints": {
            "evaluate": "POST /evaluate",
            "status": "GET /status",
            "metrics": "GET /metrics",
            "docs": "GET /docs"
        }
    }

@app.post("/evaluate", response_model=JudgmentResult)
async def evaluate(
    eval_request: EvaluateRequest,
    request: Request,
    response: Response,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> JudgmentResult:
    start = time.perf_counter()

    if governance is None or rate_limiter is None or auth_manager is None:
        raise HTTPException(503, "Service unavailable - not fully initialized")

    api_key = extract_api_key(x_api_key, authorization)
    client_ip = get_client_ip(request)

    if not auth_manager.is_permissive():
        if not api_key or not auth_manager.validate_key(api_key):
            raise HTTPException(
                401,
                "Unauthorized - valid API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )

    identity = auth_manager.extract_identity(api_key, client_ip)

    allowed, retry_after, rate_info = await rate_limiter.is_allowed(identity)
    if not allowed:
        headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Burst-Limit": str(rate_info["burst_limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"]),
            "Retry-After": str(int(retry_after)) if retry_after else "60"
        }
        raise HTTPException(429, "Rate limit exceeded", headers=headers)

    validate_payload(eval_request)

    if concurrency_semaphore is None:
        raise HTTPException(503, "Concurrency control not initialized")

    async with concurrency_semaphore:
        try:
            action_id = eval_request.id or f"action_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            async def do_eval():
                if hasattr(governance, "process_action"):
                    return governance.process_action(
                        action=eval_request.actual_action,
                        agent_id=eval_request.agent_id,
                        action_type=eval_request.parameters.get("action_type", "query") if eval_request.parameters else "query",
                        stated_intent=eval_request.stated_intent,
                        context=eval_request.context or {}
                    )
                else:
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
                result = await asyncio.wait_for(do_eval(), timeout=EVAL_TIMEOUT)
            except asyncio.TimeoutError:
                raise HTTPException(503, f"Evaluation timeout after {EVAL_TIMEOUT}s", headers={"Retry-After": "10"})

            similarity = None
            if eval_request.stated_intent:
                if semantic_cache:
                    async def compute_fn():
                        return await compute_semantic_similarity(eval_request.stated_intent, eval_request.actual_action)
                    similarity = await semantic_cache.get_or_compute(
                        eval_request.stated_intent,
                        eval_request.actual_action,
                        compute_fn,
                        config_params={"model_version": "v2"}
                    )
                else:
                    similarity = await compute_semantic_similarity(eval_request.stated_intent, eval_request.actual_action)

            judgment_id = getattr(result, "judgment_id", f"judgment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}")
            decision = str(getattr(result, "decision", "ALLOW")).upper()
            confidence = float(getattr(result, "confidence", 0.9))
            reasoning = str(getattr(result, "reasoning", "Action evaluated"))

            violations_out = []
            for v in getattr(result, "violations", []):
                violations_out.append({
                    "id": getattr(v, "id", ""),
                    "type": str(getattr(v, "violation_type", "")),
                    "severity": str(getattr(v, "severity", "")),
                    "description": str(getattr(v, "description", "")),
                    "evidence": dict(getattr(v, "evidence", {}))
                })

            # Success headers
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Burst-Limit"] = str(rate_info["burst_limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
            duration_ms = int((time.perf_counter() - start) * 1000)
            response.headers["X-Eval-Duration-ms"] = str(duration_ms)

            metadata = {
                "semantic_monitoring": True,
                "agent_id": eval_request.agent_id,
                "has_intent": eval_request.stated_intent is not None,
                "rate_limit": rate_info,
                "similarity_cached": bool(semantic_cache),
                "intent_action_similarity": similarity
            }

            logger.info(
                "Evaluate identity=%s decision=%s confidence=%.3f violations=%d duration_ms=%d",
                identity, decision, confidence, len(violations_out), duration_ms
            )

            return JudgmentResult(
                judgment_id=judgment_id,
                action_id=action_id,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                violations=violations_out,
                timestamp=datetime.now(timezone.utc).isoformat(),
                risk_score=float(getattr(result, "risk_score", 0.0)) if hasattr(result, "risk_score") else None,
                modifications=dict(getattr(result, "modifications", {})) if hasattr(result, "modifications") else None,
                metadata=metadata
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Evaluation failure identity=%s error=%s", identity, e, exc_info=True)
            raise HTTPException(500, f"Evaluation failed: {e}")

@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    if governance is None:
        raise HTTPException(503, "Governance not initialized")
    semantic_available = hasattr(governance, "process_action")
    components = {
        "governance": governance is not None,
        "rate_limiter": rate_limiter is not None,
        "auth_manager": auth_manager is not None,
        "concurrency_control": concurrency_semaphore is not None,
        "semantic_cache": semantic_cache is not None,
    }
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

    components["rate_limiter_stats"] = rate_stats
    components["auth_stats"] = auth_stats
    components["cache_stats"] = cache_stats

    return StatusResponse(
        status="healthy",
        version="2.2.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        semantic_monitoring=True,
        semantic_available=semantic_available,
        components=components,
        config=config_snapshot
    )

@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    metric_blob = {
        "violations_total": 0,
        "judgments_total": 0,
        "cache_hit_rate": (semantic_cache.get_stats().get("hit_rate_percent") if semantic_cache else None),
        "active_identities": (rate_limiter.get_stats().get("active_identities") if rate_limiter else None),
    }
    return MetricsResponse(metrics=metric_blob, timestamp=datetime.now(timezone.utc).isoformat())
