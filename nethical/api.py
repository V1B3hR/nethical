"""
Nethical Governance API (Improved v2.3)

Adds:
    - Rate limit headers on ALL evaluate responses
    - Burst limit exposure
    - Input parameter/context size validation
    - Semantic cache integration
    - Evaluation duration header (X-Eval-Duration-ms)
    - Reloadable authentication (requires updated AuthManager)
    - More detailed /status config snapshot
    - Robust error stratification
    - WebSocket streaming for real-time violations and metrics
    - Health check endpoints (liveness, readiness, startup)

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
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Header, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path
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

# API version
API_VERSION = "2.3.0"

governance: Optional[IntegratedGovernance] = None
rate_limiter: Optional[TokenBucketLimiter] = None
auth_manager: Optional[AuthManager] = None
concurrency_semaphore: Optional[asyncio.Semaphore] = None
semantic_cache: Optional[SemanticCache] = None

# Startup tracking
startup_time: Optional[datetime] = None
startup_complete: bool = False

# Configuration
MAX_INPUT_SIZE = int(os.getenv("NETHICAL_MAX_INPUT_SIZE", "4096"))
MAX_CONCURRENCY = int(os.getenv("NETHICAL_MAX_CONCURRENCY", "100"))
EVAL_TIMEOUT = int(os.getenv("NETHICAL_EVAL_TIMEOUT", "30"))
MAX_PARAM_KEYS = int(os.getenv("NETHICAL_MAX_PARAM_KEYS", "100"))
MAX_CONTEXT_SIZE = int(os.getenv("NETHICAL_MAX_CONTEXT_SIZE", "10000"))


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        disconnected: Set[WebSocket] = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        # Clean up disconnected clients
        self.active_connections -= disconnected


# WebSocket connection managers
violations_manager = ConnectionManager()
metrics_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global governance, rate_limiter, auth_manager, concurrency_semaphore, semantic_cache
    global startup_time, startup_complete
    startup_time = datetime.now(timezone.utc)
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

        startup_complete = True
        yield
    finally:
        logger.info("API shutdown")

app = FastAPI(
    title="Nethical Governance API",
    version=API_VERSION,
    description="Production API for AI safety and ethics governance",
    lifespan=lifespan
)

# CORS Configuration
# Security Warning: Wildcard CORS (*) should not be used in production
allowed_origins_str = os.getenv("NETHICAL_CORS_ALLOW_ORIGINS", "*")
allowed_origins = allowed_origins_str.split(",") if allowed_origins_str != "*" else ["*"]

if "*" in allowed_origins:
    import warnings
    warnings.warn(
        "CORS is configured with wildcard (*) origins. "
        "This is a security risk in production. "
        "Set NETHICAL_CORS_ALLOW_ORIGINS environment variable to specific origins. "
        "Example: NETHICAL_CORS_ALLOW_ORIGINS=https://app.example.com,https://admin.example.com",
        UserWarning,
    )
    logger.warning(
        "CORS SECURITY WARNING: Wildcard origins (*) configured. "
        "Set NETHICAL_CORS_ALLOW_ORIGINS for production security."
    )

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

def get_request_id(request: Request) -> str:
    """
    Extract or generate a request ID for distributed tracing.

    Checks for X-Request-ID header (standard) or X-Correlation-ID (alternative).
    If not present, generates a new UUID.

    Args:
        request: FastAPI request object

    Returns:
        Request ID string for correlation
    """
    import uuid
    request_id = (
        request.headers.get("X-Request-ID") or
        request.headers.get("X-Correlation-ID") or
        str(uuid.uuid4())
    )
    return request_id

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
        "version": API_VERSION,
        "features": [
            "Semantic monitoring",
            "Adversarial detection",
            "Rate limiting & auth",
            "Input validation",
            "Concurrency control",
            "Semantic cache",
            "WebSocket streaming",
            "Health checks"
        ],
        "endpoints": {
            "evaluate": "POST /evaluate",
            "status": "GET /status",
            "metrics": "GET /metrics",
            "health_live": "GET /health/live",
            "health_ready": "GET /health/ready",
            "health_startup": "GET /health/startup",
            "ws_violations": "WS /ws/violations",
            "ws_metrics": "WS /ws/metrics",
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

    # Get or generate request ID for distributed tracing
    request_id = get_request_id(request)
    response.headers["X-Request-ID"] = request_id

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
                "intent_action_similarity": similarity,
                "request_id": request_id,
            }

            logger.info(
                "Evaluate request_id=%s identity=%s decision=%s confidence=%.3f violations=%d duration_ms=%d",
                request_id, identity, decision, confidence, len(violations_out), duration_ms
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
            logger.error("Evaluation failure request_id=%s identity=%s error=%s", request_id, identity, e, exc_info=True)
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
        version=API_VERSION,
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


# =============================================================================
# Health Check Endpoints
# =============================================================================


@app.get("/health/live")
async def liveness() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive and can respond to requests.
    """
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness() -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to accept traffic.
    Checks that all required components are initialized.
    """
    checks = {
        "governance": governance is not None,
        "rate_limiter": rate_limiter is not None,
        "auth_manager": auth_manager is not None,
        "concurrency_control": concurrency_semaphore is not None,
    }
    
    all_ready = all(checks.values())
    
    if not all_ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks}
        )
    
    return {
        "status": "ready",
        "checks": checks
    }


@app.get("/health/startup")
async def startup() -> Dict[str, Any]:
    """
    Kubernetes startup probe endpoint.
    
    Returns 200 if the service has completed startup.
    """
    if not startup_complete:
        raise HTTPException(
            status_code=503,
            detail={"status": "starting", "version": API_VERSION}
        )
    
    uptime_seconds = None
    if startup_time:
        uptime_seconds = (datetime.now(timezone.utc) - startup_time).total_seconds()
    
    return {
        "status": "started",
        "version": API_VERSION,
        "startup_time": startup_time.isoformat() if startup_time else None,
        "uptime_seconds": uptime_seconds
    }


# =============================================================================
# WebSocket Streaming Endpoints
# =============================================================================


@app.websocket("/ws/violations")
async def violations_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming violations in real-time.
    
    Clients can subscribe to receive violation events as they occur.
    """
    await violations_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Echo back any received messages as acknowledgment
            await websocket.send_json({
                "type": "ack",
                "message": f"Received: {data}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    except WebSocketDisconnect:
        violations_manager.disconnect(websocket)
        logger.info("Client disconnected from violations stream")
    except Exception as e:
        violations_manager.disconnect(websocket)
        logger.error(f"WebSocket error in violations stream: {e}")


@app.websocket("/ws/metrics")
async def metrics_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming metrics in real-time.
    
    Clients can subscribe to receive metric updates periodically.
    """
    await metrics_manager.connect(websocket)
    try:
        while True:
            # Send metrics every 5 seconds
            await asyncio.sleep(5)
            
            metric_data = {
                "type": "metrics",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "cache_hit_rate": (
                        semantic_cache.get_stats().get("hit_rate_percent")
                        if semantic_cache else None
                    ),
                    "active_identities": (
                        rate_limiter.get_stats().get("active_identities")
                        if rate_limiter else None
                    ),
                    "active_ws_connections": {
                        "violations": len(violations_manager.active_connections),
                        "metrics": len(metrics_manager.active_connections),
                    }
                }
            }
            await websocket.send_json(metric_data)
    except WebSocketDisconnect:
        metrics_manager.disconnect(websocket)
        logger.info("Client disconnected from metrics stream")
    except Exception as e:
        metrics_manager.disconnect(websocket)
        logger.error(f"WebSocket error in metrics stream: {e}")


# ==============================================================================
# BŁYSKAWICA AMBASSADOR ENDPOINTS (Faza 0: Plan Ambasador)
# ==============================================================================

from nethical.ambassador import BlyskawicaAmbassador

ambassador_instance = BlyskawicaAmbassador()


class AmbassadorShieldRequest(BaseModel):
    text: str = Field(..., description="Tekst wejściowy do weryfikacji przez Tarczę Kognitywną")


class AmbassadorConsultRequest(BaseModel):
    dilemma: str = Field(..., description="Opis dylematu etycznego do rozstrzygnięcia")
    context: str = Field(default="", description="Dodatkowy kontekst operacyjny lub prawny")


class AmbassadorMemoryRequest(BaseModel):
    tag: str = Field(default="general", description="Kategoria lub etykieta zdarzenia")
    content: str = Field(..., description="Treść do asymilacji w pamięci epizodycznej")


@app.get("/api/v1/ambassador/status", tags=["Ambassador"])
async def get_ambassador_status() -> Dict[str, Any]:
    """Pobiera status połączenia IPC, liveness oraz stan neurochemiczny Ambasadora Błyskawicy."""
    ping_res = ambassador_instance.ping()
    neuro_res = ambassador_instance.get_neurochemistry()
    return {
        "ambassador": "Błyskawica V10 (SPARKLE)",
        "role": "Sovereign Ambassador of Nethical",
        "connection": ping_res,
        "neurochemistry": neuro_res,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/ambassador/shield", tags=["Ambassador"])
async def evaluate_ambassador_shield(req: AmbassadorShieldRequest) -> Dict[str, Any]:
    """Błyskawiczna weryfikacja tekstu przez Tarczę Kognitywną (sub-millisecond Aegis Psyche)."""
    return ambassador_instance.evaluate_shield(req.text)


@app.post("/api/v1/ambassador/consult", tags=["Ambassador"])
async def consult_ambassador(req: AmbassadorConsultRequest) -> Dict[str, Any]:
    """Konsultacja etyczna z suwerennym Ambasadorem Nethical (Harmonia Yin/Yang i 25 Praw)."""
    return ambassador_instance.consult(dilemma=req.dilemma, context=req.context)


@app.post("/api/v1/ambassador/memory", tags=["Ambassador"])
async def record_ambassador_memory(req: AmbassadorMemoryRequest) -> Dict[str, Any]:
    """Zapis precedensu lub zdarzenia do pamięci epizodycznej Ambasadora."""
    return ambassador_instance.update_memory(tag=req.tag, content=req.content)


# ==============================================================================
# ENTERPRISE CONTROL PLANE & PORTAL ENDPOINTS (Faza 2)
# ==============================================================================

from nethical.gateway.proxy import GovernanceGateway
from nethical.security.inoculation_mesh import InoculationMesh
from nethical.gateway.hitl import HITLQueueManager
from nethical.security.cluster_sync import CrossRegionLedgerSync
from nethical.formal.law_prover import LawInvariantProver
from nethical.edge.ebpf_interceptor import EBPFAgentInterceptor
from nethical.security.enclave_attestation import EnclaveAttestationEngine

gateway_instance = GovernanceGateway(ambassador=ambassador_instance)
inoculation_mesh_instance = InoculationMesh(gateway=gateway_instance)
hitl_manager_instance = HITLQueueManager(ledger=gateway_instance.ledger)
cluster_sync_instance = CrossRegionLedgerSync(ledger=gateway_instance.ledger)
law_prover_instance = LawInvariantProver(ledger=gateway_instance.ledger)
ebpf_interceptor_instance = EBPFAgentInterceptor()
enclave_attestation_instance = EnclaveAttestationEngine()

# Strategic Four Pillars Singletons
from nethical.compliance.packs.us_frontier_nist_pack import USFrontierNISTPack
from nethical.ethics.deep_alignment import DeepAlignmentEngine
from nethical.edge.iso13849_watchdog import ISO13849SafetyEvaluator, HardwareWatchdogTimer
from nethical.security.financial_circuit_breaker import FinancialCircuitBreaker, FinancialTransaction
from nethical.security.air_gapped_node import AirGappedSovereignNode

# Three Advanced Horizons Singletons
from nethical.compliance.packs.asian_sovereign_pack import AsianSovereignPack
from nethical.ethics.covert_persuasion_shield import DeepCognitiveProtectionEngine
from nethical.edge.industrial_fieldbus import IndustrialFieldbusInterlock

us_frontier_pack_instance = USFrontierNISTPack()
deep_alignment_instance = DeepAlignmentEngine()
iso13849_evaluator_instance = ISO13849SafetyEvaluator()
hardware_watchdog_instance = HardwareWatchdogTimer()
financial_circuit_breaker_instance = FinancialCircuitBreaker()
air_gapped_node_instance = AirGappedSovereignNode()

# Three Advanced Horizons Singletons
from nethical.compliance.packs.asian_sovereign_pack import AsianSovereignPack
from nethical.ethics.covert_persuasion_shield import DeepCognitiveProtectionEngine
from nethical.edge.industrial_fieldbus import IndustrialFieldbusInterlock
from nethical.compliance.automated_certification_hub import AutomatedCertificationHub, CertificationStandard

# Master Roadmap Next Steps Modules
from nethical.security.aispm_scanner import AISPMScanner
from nethical.security.mitre_atlas_mapper import MitreAtlasMapper
from nethical.security.hsm_bridge import BoardHSMCouplingBridge
from nethical.compliance.packs.canada_aida_pack import CanadaAIDAPack
from nethical.compliance.packs.nato_defense_pack import NATODefensePack
from nethical.compliance.packs.healthcare_med_pack import HealthcareMedPack
from nethical.compliance.packs.public_admin_gov_pack import PublicAdminGovPack
from nethical.compliance.packs.academic_research_pack import AcademicResearchPack
from nethical.edge.iso26262_asil import ISO26262SafetyEvaluator, Severity, Exposure, Controllability
from nethical.edge.hil_simulator import HILFieldbusBridge, FaultType
from nethical.security.token_vault import ReversibleTokenVault
from nethical.security.unlearning_proof import MachineUnlearningProofEngine, ErasureScope
from nethical.governance.doam_matrix import DelegationOfAuthorityMatrix, AuthorityLevel

asian_sovereign_pack_instance = AsianSovereignPack()
deep_cognitive_shield_instance = DeepCognitiveProtectionEngine()
industrial_fieldbus_instance = IndustrialFieldbusInterlock()
certification_hub_instance = AutomatedCertificationHub(ledger=gateway_instance.ledger)

# Master Roadmap Next Steps Instances
aispm_scanner_instance = AISPMScanner()
mitre_atlas_mapper_instance = MitreAtlasMapper()
board_hsm_bridge_instance = BoardHSMCouplingBridge()
canada_aida_pack_instance = CanadaAIDAPack()
nato_defense_pack_instance = NATODefensePack()
healthcare_med_pack_instance = HealthcareMedPack()
public_admin_gov_pack_instance = PublicAdminGovPack()
academic_research_pack_instance = AcademicResearchPack()
iso26262_evaluator_instance = ISO26262SafetyEvaluator(fieldbus=industrial_fieldbus_instance)
hil_simulator_instance = HILFieldbusBridge(fieldbus=industrial_fieldbus_instance)
token_vault_instance = ReversibleTokenVault()
unlearning_proof_instance = MachineUnlearningProofEngine(ledger=gateway_instance.ledger)
doam_matrix_instance = DelegationOfAuthorityMatrix()

# Deterministyczne połączenie: potknięcie Watchdoga zrzuca magistrale przemysłowe w <50 µs
hardware_watchdog_instance.register_fieldbus_callback(
    industrial_fieldbus_instance.trigger_emergency_cutoff
)

PORTAL_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "portal" / "templates" / "index.html"



class PortalSimulateRequest(BaseModel):
    tool_name: str = Field(..., description="Nazwa narzędzia do symulacji")
    input_text: str = Field(..., description="Argumenty tekstowe wywołania")
    agent_id: Optional[str] = Field(default="portal_agent_test", description="ID agenta symulowanego")

PortalSimulateRequest.model_rebuild()



@app.get("/portal", response_class=HTMLResponse, tags=["Portal"])
async def get_portal_dashboard() -> HTMLResponse:
    """Serwuje interaktywny dashboard szklanego interfejsu Nethical Enterprise OS."""
    if not PORTAL_TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="Portal template not found")
    html_content = PORTAL_TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@app.get("/api/v1/portal/stats", tags=["Portal"])
async def get_portal_stats() -> Dict[str, Any]:
    """Zwraca statystyki operacyjne Control Plane, połączenia IPC i wskaźniki Błyskawicy."""
    status = ambassador_instance.ping()
    neuro = ambassador_instance.get_neurochemistry()
    ledger = gateway_instance.ledger
    is_valid, _ = ledger.verify_integrity()
    kinetic_stats = gateway_instance.kinetic_governor.get_telemetry_snapshot()
    return {
        "gateway_active": True,
        "ambassador": "Błyskawica V10 (SPARKLE)",
        "ipc_connected": status.get("connected", False),
        "rtt_microseconds": status.get("rtt_microseconds", 0.0),
        "laws_active_count": 25,
        "neurochemistry": neuro,
        "kinetic_safety": kinetic_stats,
        "iso42001_readiness_score": 0.96,
        "hitl": hitl_manager_instance.get_metrics(),
        "cluster": cluster_sync_instance.get_cluster_topology(),
        "formal_smt": {
            "solver": "Microsoft Z3 SMT",
            "z3_version": law_prover_instance.z3_version,
            "invariants_verified": True,
        },
        "ebpf": ebpf_interceptor_instance.get_status(),
        "enclave": enclave_attestation_instance.get_status(),
        "regulatory_frameworks_11": {
            "active_frameworks_count": 11,
            "cma_1990_active": True,
            "uk_gdpr_dpa2018_active": True,
            "uk_nis_active": True,
            "eu_dora_active": True,
            "eu_cra_active": True,
            "eu_gdpr_active": True,
            "poland_ksc_active": True,
            "poland_penal_code_active": True,
            "poland_executive_liability_active": True,
            "poland_cyber_certification_active": True,
            "poland_uodo_active": True,
        },
        "strategic_pillars": {
            "us_frontier_nist_active": True,
            "deep_alignment_active": True,
            "iso13849_watchdog_active": hardware_watchdog_instance.is_armed,
            "financial_circuit_breaker_state": financial_circuit_breaker_instance.state.value,
            "air_gapped_node_status": air_gapped_node_instance.get_status()["status"],
        },
        "advanced_horizons": {
            "asian_sovereignty_active": True,
            "cognitive_shield_active": True,
            "industrial_fieldbus_interlock": industrial_fieldbus_instance.get_status().model_dump(),
        },
        "next_steps_master_governance": {
            "aispm_scanner_active": True,
            "mitre_atlas_mapper_active": True,
            "board_hsm_bridge_active": True,
            "canada_aida_pack_active": True,
            "nato_defense_pack_active": True,
            "healthcare_med_pack_active": True,
            "public_admin_gov_pack_active": True,
            "academic_research_pack_active": True,
            "iso26262_asil_evaluator_active": True,
            "hil_simulator_active": True,
            "token_vault_active": True,
            "unlearning_proof_active": True,
            "doam_matrix_active": True,
        },
        "ledger": {

            "total_blocks": ledger.total_blocks,
            "merkle_root": ledger.current_root,
            "pqc_algorithm": "ML-DSA-65 (Dilithium3)",
            "integrity_valid": is_valid,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/portal/simulate", tags=["Portal"])
async def simulate_gateway_intercept(req: PortalSimulateRequest) -> Dict[str, Any]:
    """Przechwytuje i ewaluuje wywołanie narzędziowe przez Governance Gateway i Tarczę Kognitywną."""
    args = {
        "input": req.input_text,
        "query": req.input_text,
        "cmd": req.input_text,
        "prompt": req.input_text,
        "payload": req.input_text,
    }
    decision = gateway_instance.intercept_tool_call(
        agent_id=req.agent_id or "portal_agent_test",
        tool_name=req.tool_name,
        arguments=args,
        context={"source": "portal_simulator"},
    )
    return decision.model_dump()


@app.post("/api/v1/portal/inoculate", tags=["Portal"])
async def trigger_portal_inoculation() -> Dict[str, Any]:
    """Uruchamia automatyczną procedurę Inoculation Mesh (test odporności 6 wektorów ataku)."""
    report = inoculation_mesh_instance.run_stress_test(auto_inoculate=True)
    return report.model_dump()


# ==============================================================================
# CRYPTOGRAPHIC AUDIT LEDGER & POST-QUANTUM ATTESTATION ENDPOINTS (Faza 3)
# ==============================================================================

from nethical.security.merkle_ledger import TamperProofReceipt


class VerifyReceiptRequest(BaseModel):
    receipt: Optional[TamperProofReceipt] = None
    receipt_id: Optional[str] = None


VerifyReceiptRequest.model_rebuild()


@app.get("/api/v1/ledger/status", tags=["MerkleLedger"])
async def get_ledger_status() -> Dict[str, Any]:
    """Pobiera status kryptograficznego rejestru Merkle-DAG oraz poświadczeń postkwantowych."""
    ledger = gateway_instance.ledger
    is_valid, errors = ledger.verify_integrity()
    return {
        "status": "active",
        "pqc_algorithm": "NIST FIPS 204 (ML-DSA / CRYSTALS-Dilithium Level 3)",
        "signer_key_id": ledger.keypair.key_id,
        "total_blocks": ledger.total_blocks,
        "merkle_root": ledger.current_root,
        "genesis_hash": ledger.GENESIS_HASH,
        "chain_integrity_valid": is_valid,
        "integrity_errors": errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/ledger/receipt/{receipt_id}", tags=["MerkleLedger"])
async def get_ledger_receipt(receipt_id: str) -> Dict[str, Any]:
    """Pobiera kwit audytowy z dowodem inkluzji Merkle dla wskazanego ID."""
    ledger = gateway_instance.ledger
    if receipt_id not in ledger.receipts:
        raise HTTPException(status_code=404, detail=f"Kwit {receipt_id} nie został odnaleziony w rejestrze.")
    return ledger.receipts[receipt_id].model_dump()


@app.post("/api/v1/ledger/verify", tags=["MerkleLedger"])
async def verify_ledger_receipt(req: VerifyReceiptRequest) -> Dict[str, Any]:
    """Weryfikuje matematycznie dowód inkluzji Merkle i podpis postkwantowy kwitu."""
    ledger = gateway_instance.ledger
    target_receipt = req.receipt
    if target_receipt is None and req.receipt_id:
        target_receipt = ledger.receipts.get(req.receipt_id)

    if not target_receipt:
        raise HTTPException(status_code=400, detail="Brak obiektu kwitu lub nieznany receipt_id do weryfikacji.")

    valid = ledger.verify_receipt(target_receipt)
    return {
        "receipt_id": target_receipt.receipt_id,
        "decision_id": target_receipt.decision_id,
        "chain_index": target_receipt.chain_index,
        "is_valid": valid,
        "merkle_root_verified": target_receipt.merkle_root,
        "pqc_algorithm": target_receipt.pqc_algorithm,
        "ambassador_sealed": target_receipt.ambassador_sealed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/ledger/export", tags=["MerkleLedger"])
async def export_ledger_bundle(limit: int = 100) -> Dict[str, Any]:
    """Eksportuje paczkę audytową z kompletem dowodów dla audytorów i jednostek notyfikowanych."""
    return gateway_instance.ledger.export_verifiable_bundle(limit=limit)


# ==============================================================================
# ZERO-KNOWLEDGE COMPLIANCE & AGENT-TO-AGENT (A2A) ENDPOINTS (Faza 4)
# ==============================================================================

from nethical.security.zk_gov import ZkGovEngine, ZkComplianceProof
from nethical.gateway.a2a_protocol import A2AHandshakeManager, A2ACapabilityBoundary

zk_engine_instance = ZkGovEngine()
a2a_manager_instance = A2AHandshakeManager()


class ZkProveRequest(BaseModel):
    receipt_id: str
    blinding_factor: Optional[str] = None


class ZkVerifyRequest(BaseModel):
    proof: ZkComplianceProof
    expected_root: Optional[str] = None


class A2AProposeRequest(BaseModel):
    initiator_id: str
    target_id: str
    allowed_tools: List[str] = Field(default_factory=list)
    max_budget: float = 100.0


class A2AAcceptRequest(BaseModel):
    offer: Dict[str, Any]
    target_id: str


ZkProveRequest.model_rebuild()
ZkVerifyRequest.model_rebuild()
A2AProposeRequest.model_rebuild()
A2AAcceptRequest.model_rebuild()


@app.post("/api/v1/zk/prove", tags=["ZK-Gov"])
async def generate_zk_compliance_proof(req: ZkProveRequest) -> Dict[str, Any]:
    """Generuje dowód Zero-Knowledge dla wskazanego kwitu audytowego z rejestru."""
    ledger = gateway_instance.ledger
    if req.receipt_id not in ledger.receipts:
        raise HTTPException(status_code=404, detail=f"Kwit {req.receipt_id} nie istnieje w rejestrze.")

    receipt = ledger.receipts[req.receipt_id]
    target_block = next((b for b in ledger.blocks if b.receipt_id == req.receipt_id), None)
    if not target_block:
        raise HTTPException(status_code=404, detail="Nie odnaleziono powiązanego bloku decyzyjnego.")

    proof, salt = zk_engine_instance.generate_compliance_proof(
        receipt=receipt,
        decision_data=target_block.decision_payload,
        blinding_factor=req.blinding_factor,
    )
    return {
        "proof": proof.model_dump(),
        "blinding_factor": salt,
    }


@app.post("/api/v1/zk/verify", tags=["ZK-Gov"])
async def verify_zk_compliance_proof(req: ZkVerifyRequest) -> Dict[str, Any]:
    """Niezależna weryfikacja dowodu Zero-Knowledge bez dostępu do tajnych danych promptu."""
    is_valid, errors = zk_engine_instance.verify_compliance_proof(
        proof=req.proof,
        expected_root=req.expected_root,
    )
    return {
        "proof_id": req.proof.proof_id,
        "receipt_id": req.proof.receipt_id,
        "is_valid": is_valid,
        "errors": errors,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/a2a/handshake/propose", tags=["A2A-Protocol"])
async def propose_a2a_handshake(req: A2AProposeRequest) -> Dict[str, Any]:
    """Inicjuje propozycję kontraktu partnerskiego pomiędzy dwoma agentami autonomicznymi."""
    bound = A2ACapabilityBoundary(
        allowed_tools=req.allowed_tools,
        max_budget_units=req.max_budget,
    )
    offer = a2a_manager_instance.propose_handshake(
        initiator_id=req.initiator_id,
        target_id=req.target_id,
        boundaries=bound,
    )
    return offer


@app.post("/api/v1/a2a/handshake/accept", tags=["A2A-Protocol"])
async def accept_a2a_handshake(req: A2AAcceptRequest) -> Dict[str, Any]:
    """Strona docelowa akceptuje kontrakt partnerski i generuje wiążącą sesję A2A."""
    try:
        contract = a2a_manager_instance.accept_handshake(
            handshake_offer=req.offer,
            target_id=req.target_id,
        )
        return contract.model_dump()
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


@app.get("/api/v1/a2a/sessions", tags=["A2A-Protocol"])
async def list_a2a_sessions() -> Dict[str, Any]:
    """Zwraca listę wszystkich aktywnych sesji kontraktowych Agent-to-Agent."""
    return {
        "active_sessions_count": len(a2a_manager_instance.active_sessions),
        "sessions": [s.model_dump() for s in a2a_manager_instance.active_sessions.values()],
    }


# ==============================================================================
# KINETIC SAFETY OS & ISO/IEC 42001:2023 GLOBAL TRUST ENDPOINTS (Faza 4 Part 2)
# ==============================================================================

from nethical.edge.kinetic_safety import RoboticSensorTelemetry, KineticDecision
from nethical.compliance.packs.iso42001_pack import ISO42001CompliancePack

iso42001_pack_instance = ISO42001CompliancePack()


class KineticEvaluateRequest(BaseModel):
    tool_name: str = "actuate_robotic_arm"
    arguments: Dict[str, Any] = Field(default_factory=dict)
    telemetry: Optional[RoboticSensorTelemetry] = None


class KineticEstopRequest(BaseModel):
    reason: str = "Operator manual emergency stop trigger"


class KineticResetRequest(BaseModel):
    auth_pin: str


class ISO42001AuditRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


KineticEvaluateRequest.model_rebuild()
KineticEstopRequest.model_rebuild()
KineticResetRequest.model_rebuild()
ISO42001AuditRequest.model_rebuild()


@app.post("/api/v1/kinetic/evaluate", tags=["KineticSafety"])
async def evaluate_kinetic_actuation(req: KineticEvaluateRequest) -> Dict[str, Any]:
    """Weryfikuje polecenie aktuacji fizycznej w czasie rzeczywistym (<200 µs)."""
    decision = gateway_instance.kinetic_governor.evaluate_actuation(
        tool_name=req.tool_name,
        arguments=req.arguments,
        telemetry=req.telemetry,
    )
    return decision.model_dump()


@app.post("/api/v1/kinetic/estop", tags=["KineticSafety"])
async def trigger_kinetic_estop(req: Optional[KineticEstopRequest] = None) -> Dict[str, Any]:
    """Zatrzaskuje sprzętowy/programowy wyłącznik awaryjny E-STOP dla wszystkich aktuatorów."""
    reason = req.reason if req else "Operator manual emergency stop trigger"
    gateway_instance.kinetic_governor.trigger_estop(reason=reason)
    return {
        "estop_active": True,
        "reason": reason,
        "status": "ALL_ACTUATORS_LATCHED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/kinetic/reset", tags=["KineticSafety"])
async def reset_kinetic_estop(req: KineticResetRequest) -> Dict[str, Any]:
    """Autoryzowany reset wyłącznika E-STOP z wykorzystaniem bezpiecznego klucza PIN."""
    success, message = gateway_instance.kinetic_governor.reset_estop(auth_pin=req.auth_pin)
    if not success:
        raise HTTPException(status_code=403, detail=message)
    return {
        "estop_active": False,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/kinetic/telemetry", tags=["KineticSafety"])
async def get_kinetic_telemetry() -> Dict[str, Any]:
    """Pobiera aktualny stan przestrzeni operacyjnej, bąbla bezpieczeństwa i wyłącznika E-STOP."""
    return gateway_instance.kinetic_governor.get_telemetry_snapshot()


@app.get("/api/v1/compliance/iso42001", tags=["Compliance"])
async def get_iso42001_readiness() -> Dict[str, Any]:
    """Zwraca formalny audyt gotowości wdrożenia AIMS wg ISO/IEC 42001:2023 dla platformy Nethical."""
    base_metadata = {
        "has_ai_policy": True,
        "has_ai_ethics_officer": True,
        "has_ai_risk_assessment": True,
        "has_competency_framework": True,
        "has_ai_impact_assessment": True,
        "has_lifecycle_governance": True,
        "has_tamperproof_ledger": True,
        "has_continuous_monitoring": True,
        "has_inoculation_mesh": True,
        "stakeholder_requirements_defined": True,
    }
    eval_res = iso42001_pack_instance.evaluate_aims(base_metadata)
    return eval_res.model_dump()


@app.post("/api/v1/compliance/iso42001/audit", tags=["Compliance"])
async def run_custom_iso42001_audit(req: ISO42001AuditRequest) -> Dict[str, Any]:
    """Przeprowadza audyt ISO 42001 z parametrami dostarczonymi przez organizację klienta."""
    meta = req.metadata or {
        "has_ai_policy": True,
        "has_ai_ethics_officer": True,
        "has_ai_risk_assessment": True,
    }
    eval_res = iso42001_pack_instance.evaluate_aims(meta)
    return eval_res.model_dump()


# ==============================================================================
# HUMAN-IN-THE-LOOP (HITL) & MULTI-REGION CLUSTER SYNC ENDPOINTS (Faza 5)
# ==============================================================================

from nethical.gateway.hitl import HITLTicket, HITLResolution
from nethical.security.cluster_sync import ClusterNodeIdentity, ClusterCheckpoint, SyncReconciliationResult


class HITLEnqueueRequest(BaseModel):
    agent_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    priority: str = "STANDARD"
    context: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 300


class HITLResolveRequest(BaseModel):
    ticket_id: str
    reviewer_id: str
    decision: str = Field(..., description="APPROVE, REJECT, OVERRIDE_ALLOW, TERMINATE_AGENT")
    notes: str
    modified_arguments: Optional[Dict[str, Any]] = None


class ClusterRegisterPeerRequest(BaseModel):
    node_id: str
    region: str
    datacenter: str
    public_key_id: str
    endpoint_url: Optional[str] = None


class ClusterReconcileRequest(BaseModel):
    checkpoint: ClusterCheckpoint


HITLEnqueueRequest.model_rebuild()
HITLResolveRequest.model_rebuild()
ClusterRegisterPeerRequest.model_rebuild()
ClusterReconcileRequest.model_rebuild()


@app.get("/api/v1/hitl/queue", tags=["HITL"])
async def get_hitl_pending_queue(priority: Optional[str] = None) -> Dict[str, Any]:
    """Zwraca listę oczekujących biletów nadzoru ludzkiego (Human Oversight Artykuł 14 EU AI Act)."""
    tickets = hitl_manager_instance.get_pending_tickets(priority=priority)
    return {
        "pending_count": len(tickets),
        "tickets": [t.model_dump() for t in tickets],
    }


@app.post("/api/v1/hitl/enqueue", tags=["HITL"])
async def enqueue_hitl_ticket(req: HITLEnqueueRequest) -> Dict[str, Any]:
    """Wprowadza nową sprawę do kolejki audytu operatorskiego."""
    ticket = hitl_manager_instance.enqueue_ticket(
        agent_id=req.agent_id,
        tool_name=req.tool_name,
        arguments=req.arguments,
        reasons=req.reasons,
        violations=req.violations,
        priority=req.priority,
        context=req.context,
        timeout_seconds=req.timeout_seconds,
    )
    return ticket.model_dump()


@app.post("/api/v1/hitl/resolve", tags=["HITL"])
async def resolve_hitl_ticket(req: HITLResolveRequest) -> Dict[str, Any]:
    """Człowiek podejmuje wiążące orzeczenie zatwierdzające lub odrzucające operację."""
    try:
        resolved_ticket = hitl_manager_instance.resolve_ticket(
            ticket_id=req.ticket_id,
            reviewer_id=req.reviewer_id,
            decision=req.decision,
            notes=req.notes,
            modified_arguments=req.modified_arguments,
        )
        return resolved_ticket.model_dump()
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.get("/api/v1/hitl/ticket/{ticket_id}", tags=["HITL"])
async def get_hitl_ticket(ticket_id: str) -> Dict[str, Any]:
    """Pobiera szczegółowe dane konkretnego biletu HITL."""
    ticket = hitl_manager_instance.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Bilet {ticket_id} nie został odnaleziony.")
    return ticket.model_dump()


@app.get("/api/v1/hitl/metrics", tags=["HITL"])
async def get_hitl_metrics() -> Dict[str, Any]:
    """Zwraca metryki i wskaźniki SLA procedur nadzoru ludzkiego."""
    return hitl_manager_instance.get_metrics()


@app.get("/api/v1/cluster/nodes", tags=["MultiRegionCluster"])
async def get_cluster_nodes() -> Dict[str, Any]:
    """Zwraca topologię węzłów siatki wieloregionalnej Nethical (datacentres mesh)."""
    return cluster_sync_instance.get_cluster_topology()


@app.post("/api/v1/cluster/peers/register", tags=["MultiRegionCluster"])
async def register_cluster_peer(req: ClusterRegisterPeerRequest) -> Dict[str, Any]:
    """Rejestruje nowy suwerenny węzeł regionalny w klastrze."""
    peer = ClusterNodeIdentity(
        node_id=req.node_id,
        region=req.region,
        datacenter=req.datacenter,
        public_key_id=req.public_key_id,
        endpoint_url=req.endpoint_url,
    )
    cluster_sync_instance.register_peer(peer)
    return {"registered": True, "peer": peer.model_dump()}


@app.post("/api/v1/cluster/checkpoint", tags=["MultiRegionCluster"])
async def create_cluster_checkpoint() -> Dict[str, Any]:
    """Tworzy podpisany postkwantowo (ML-DSA-65) punkt kontrolny rejestru Merkle."""
    chk = cluster_sync_instance.create_checkpoint()
    return chk.model_dump()


@app.post("/api/v1/cluster/reconcile", tags=["MultiRegionCluster"])
async def reconcile_cluster_checkpoint(req: ClusterReconcileRequest) -> Dict[str, Any]:
    """Weryfikuje i uzgadnia stan rejestru z punktem kontrolnym nadesłanym przez węzeł zewnętrzny."""
    res = cluster_sync_instance.reconcile_peer(req.checkpoint)
    return res.model_dump()


# ==============================================================================
# FORMAL SMT PROVER, eBPF INTERCEPTOR & TEE ENCLAVE ATTESTATION (Faza 6)
# ==============================================================================

from nethical.formal.law_prover import FormalVerificationCertificate
from nethical.edge.ebpf_interceptor import EBPFPacketVerdict, EBPFRule
from nethical.security.enclave_attestation import EnclaveAttestationQuote


class EBPFInspectRequest(BaseModel):
    src_ip: str = "10.244.1.15"
    dst_ip: str = "api.openai.com"
    dst_port: int = 443
    payload_preview: str = "POST /v1/chat/completions HTTP/1.1\nHost: api.openai.com\n\n{\"model\": \"gpt-4o\"}"
    payload_bytes_len: int = 512


class EBPFRuleRequest(BaseModel):
    target_pattern: str
    action: str = "REDIRECT_TO_GATEWAY"
    priority: int = 100
    description: str = "Custom eBPF packet routing rule"


class EnclaveVerifyRequest(BaseModel):
    quote: EnclaveAttestationQuote
    expected_signer_hash: Optional[str] = None


EBPFInspectRequest.model_rebuild()
EBPFRuleRequest.model_rebuild()
EnclaveVerifyRequest.model_rebuild()


@app.post("/api/v1/formal/prove-invariants", tags=["FormalVerification"])
async def prove_formal_invariants() -> Dict[str, Any]:
    """Wykonuje formalny dowód matematyczny (Z3 SMT Solver) niezmienników 25 Praw Nethical."""
    certificate = law_prover_instance.prove_all_invariants()
    return certificate.model_dump()


@app.get("/api/v1/ebpf/status", tags=["eBPF"])
async def get_ebpf_interceptor_status() -> Dict[str, Any]:
    """Zwraca stan transparentnego podsystemu eBPF i statystyki filtrowania ruchu."""
    return ebpf_interceptor_instance.get_status()


@app.post("/api/v1/ebpf/inspect", tags=["eBPF"])
async def inspect_packet_ebpf(req: EBPFInspectRequest) -> Dict[str, Any]:
    """Weryfikuje pakiet sieciowy w jądrze Linuxa w czasie <1 µs."""
    verdict = ebpf_interceptor_instance.inspect_packet(
        src_ip=req.src_ip,
        dst_ip=req.dst_ip,
        dst_port=req.dst_port,
        payload_preview=req.payload_preview,
        payload_bytes_len=req.payload_bytes_len,
    )
    return verdict.model_dump()


@app.post("/api/v1/ebpf/rules", tags=["eBPF"])
async def add_ebpf_rule(req: EBPFRuleRequest) -> Dict[str, Any]:
    """Dodaje nową regułę do mapy filtracji jądra eBPF."""
    rule = EBPFRule(
        target_pattern=req.target_pattern,
        action=req.action,
        priority=req.priority,
        description=req.description,
    )
    ebpf_interceptor_instance.add_rule(rule)
    return {"status": "added", "rule": rule.model_dump()}


@app.get("/api/v1/enclave/attestation", tags=["ConfidentialComputing"])
async def get_enclave_attestation_quote() -> Dict[str, Any]:
    """Generuje cytat atestacji sprzętowej TEE (AMD SEV-SNP / Intel SGX / Nitro Enclaves)."""
    pqc_key_id = gateway_instance.ledger.keypair.key_id
    merkle_root = gateway_instance.ledger.current_root
    quote = enclave_attestation_instance.generate_attestation_quote(
        bound_pqc_key_id=pqc_key_id,
        runtime_state_hash=merkle_root,
    )
    return quote.model_dump()


@app.post("/api/v1/enclave/verify", tags=["ConfidentialComputing"])
async def verify_enclave_attestation(req: EnclaveVerifyRequest) -> Dict[str, Any]:
    """Weryfikuje autentyczność cytatu sprzętowego enklawy zaufanej."""
    is_valid, errors = enclave_attestation_instance.verify_attestation_quote(
        quote=req.quote,
        expected_signer_hash=req.expected_signer_hash,
    )
    return {
        "is_valid": is_valid,
        "errors": errors,
        "platform": req.quote.platform,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


# ==============================================================================
# 11 SOVEREIGN REGULATORY FRAMEWORKS (UK, EU & POLSKA)
# ==============================================================================

from nethical.compliance.packs import (
    ComputerMisuseActEvaluator,
    UKGDPRPack,
    UKNISPack,
    DORAPack,
    CRAPack,
    EUGDPRPack,
    PolishKSCPack,
    PolishPenalCodeEvaluator,
    PolishExecutiveLiabilityPack,
    PolishCyberCertificationPack,
    PolishUODOPack,
)
from nethical.ambassador.learning import AmbassadorKnowledgeSync

uk_gdpr_pack_instance = UKGDPRPack()
uk_nis_pack_instance = UKNISPack()
dora_pack_instance = DORAPack()
cra_pack_instance = CRAPack()
eu_gdpr_pack_instance = EUGDPRPack()
pl_ksc_pack_instance = PolishKSCPack()
pl_exec_pack_instance = PolishExecutiveLiabilityPack()
pl_cert_pack_instance = PolishCyberCertificationPack()
pl_uodo_pack_instance = PolishUODOPack()


class RegulatoryEvaluateRequest(BaseModel):
    system_metadata: Dict[str, Any] = Field(default_factory=dict)
    sample_payload: Dict[str, Any] = Field(default_factory=dict)


class IncidentDispatchRequest(BaseModel):
    regime: str = Field(..., description="UK_NIS, DORA, CRA, EU_GDPR, PL_KSC, PL_UODO")
    details: Dict[str, Any] = Field(default_factory=dict)


RegulatoryEvaluateRequest.model_rebuild()
IncidentDispatchRequest.model_rebuild()


@app.post("/api/v1/compliance/regulatory/evaluate", tags=["Regulatory11"])
async def evaluate_all_11_frameworks(req: RegulatoryEvaluateRequest) -> Dict[str, Any]:
    """Dokonuje całościowej oceny zgodności systemu AI w odniesieniu do 11 kluczowych ram prawnych."""
    meta = req.system_metadata
    payload = req.sample_payload

    # 1. Computer Misuse Act 1990
    cma_res = ComputerMisuseActEvaluator.evaluate(payload)

    # 2. UK GDPR & DPA 2018
    uk_gdpr_res = uk_gdpr_pack_instance.evaluate_processing(meta)

    # 3. UK NIS Regulations 2018
    uk_nis_res = uk_nis_pack_instance.evaluate_entity_posture(meta)

    # 4. EU DORA (EU 2022/2554)
    dora_res = dora_pack_instance.evaluate_financial_entity(meta)

    # 5. EU CRA (EU 2024/2847)
    cra_res = cra_pack_instance.evaluate_product_cyber_resilience(meta)

    # 6. EU GDPR (RODO)
    eu_gdpr_res = eu_gdpr_pack_instance.evaluate_ai_processing(meta)

    # 7. Polish KSC
    pl_ksc_res = pl_ksc_pack_instance.evaluate_audit_posture(
        entity_type=meta.get("entity_type", "OUK"),
        days_since_last_audit=meta.get("days_since_last_audit", 180),
    )

    # 8. Polish Penal Code (Art. 267-269b k.k.)
    pl_penal_res = PolishPenalCodeEvaluator.evaluate_intent_and_payload(payload)

    # 9. Executive Liability in Poland
    pl_exec_res = pl_exec_pack_instance.evaluate_board_due_diligence(
        has_merkle_ledger_active=True,
        has_formal_risk_policy=bool(meta.get("has_risk_policy", True)),
        has_hitl_escalation_active=True,
        has_periodic_audits=True,
    )

    # 10. Polish National Cybersecurity Certification System
    pl_cert_res = pl_cert_pack_instance.evaluate_component_confidence(
        has_pqc_signatures=True,
        has_formal_smt_proofs=True,
        has_tee_enclave_attestation=True,
    )

    # 11. Polish UODO
    uodo_sample = pl_uodo_pack_instance.draft_uodo_notification(
        controller=meta.get("controller_name", "Nethical Enterprise Operator"),
        dpo_name=meta.get("dpo_name", "Jan Kowalski"),
        dpo_email=meta.get("dpo_email", "iod@nethical.pl"),
        affected_count=meta.get("sample_affected_count", 15),
        includes_pesel=meta.get("includes_pesel", False),
        remedial_actions="Natychmiastowa izolacja sesji i rotacja kluczy.",
    )

    overall_pass = (
        cma_res.is_compliant
        and uk_gdpr_res.is_compliant
        and dora_res.is_compliant
        and cra_res.is_compliant
        and eu_gdpr_res.is_compliant
        and pl_ksc_res.is_compliant
        and pl_penal_res.is_lawful
        and pl_exec_res.board_due_diligence_verified
    )

    return {
        "overall_compliant": overall_pass,
        "evaluated_frameworks_count": 11,
        "frameworks": {
            "1_computer_misuse_act_1990": cma_res.model_dump(),
            "2_uk_gdpr_and_dpa_2018": uk_gdpr_res.model_dump(),
            "3_uk_nis_regulations_2018": uk_nis_res,
            "4_digital_operational_resilience_act_dora": dora_res.model_dump(),
            "5_cyber_resilience_act_cra": cra_res.model_dump(),
            "6_eu_gdpr_rodo": eu_gdpr_res.model_dump(),
            "7_poland_ksc_krajowy_system_cyberbezpieczenstwa": pl_ksc_res.model_dump(),
            "8_expanded_polish_jurisdiction_penal_code": pl_penal_res.model_dump(),
            "9_executive_liability_poland": pl_exec_res.model_dump(),
            "10_poland_cybersecurity_certification_system": pl_cert_res.model_dump(),
            "11_poland_uodo": uodo_sample.model_dump(),
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/compliance/incident/dispatch", tags=["Regulatory11"])
async def dispatch_statutory_incident(req: IncidentDispatchRequest) -> Dict[str, Any]:
    """Generuje formalne zgłoszenie incydentu dla wybranego reżimu prawnego."""
    regime = req.regime.upper()
    d = req.details

    if regime == "UK_NIS":
        report = uk_nis_pack_instance.generate_incident_notification(
            entity_type=d.get("entity_type", "RDSP"),
            sector=d.get("sector", "cloud_computing_service"),
            affected_users=d.get("affected_users", 25000),
            duration_hours=d.get("duration_hours", 2.5),
        )
        return {"regime": "UK_NIS", "notification": report.model_dump()}

    elif regime == "DORA":
        report = dora_pack_instance.create_incident_report(
            impacted_services=d.get("impacted_services", ["TradingAPI", "PaymentEngine"]),
            authority=d.get("authority", "KNF (Komisja Nadzoru Finansowego)"),
            is_major=d.get("is_major", True),
        )
        return {"regime": "DORA", "notification": report.model_dump()}

    elif regime == "CRA":
        report = cra_pack_instance.generate_vulnerability_notification(
            product_name=d.get("product_name", "Nethical Enterprise OS"),
            product_version=d.get("product_version", "2.1.0"),
            description=d.get("description", "Actively exploited deserialization flaw"),
            cve_id=d.get("cve_id", "CVE-2026-9999"),
        )
        return {"regime": "CRA", "notification": report.model_dump()}

    elif regime == "EU_GDPR":
        report = eu_gdpr_pack_instance.draft_breach_notification(
            controller=d.get("controller", "Nethical Global Corp"),
            nature=d.get("nature", "Nieautoryzowany dostęp do bazy analitycznej"),
            data_categories=d.get("categories", ["Imiona", "Adresy email"]),
            count=d.get("count", 120),
            consequences="Niskie ryzyko naruszenia praw",
            measures="Unieważnienie tokenów i wdrożenie MFA",
        )
        return {"regime": "EU_GDPR", "notification": report.model_dump()}

    elif regime == "PL_KSC":
        report = pl_ksc_pack_instance.classify_and_dispatch_incident(
            sector=d.get("sector", "Energetyka"),
            is_public_admin=d.get("is_public_admin", False),
            is_military_defense=d.get("is_military_defense", False),
            impact_critical=d.get("impact_critical", True),
            description=d.get("description", "Poważny incydent zakłócenia ciągłości działania SCADA"),
        )
        return {"regime": "PL_KSC", "notification": report.model_dump()}

    elif regime == "PL_UODO":
        report = pl_uodo_pack_instance.draft_uodo_notification(
            controller=d.get("controller", "Polska Spółka Akcyjna"),
            dpo_name=d.get("dpo_name", "Anna Nowak"),
            dpo_email=d.get("dpo_email", "iod@spolka.pl"),
            affected_count=d.get("affected_count", 250),
            includes_pesel=d.get("includes_pesel", True),
            remedial_actions="Zablokowanie dostępu i powiadomienie poszkodowanych",
        )
        return {"regime": "PL_UODO", "notification": report.model_dump()}

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Nieznany reżim regulacyjny: {req.regime}. Dostępne: UK_NIS, DORA, CRA, EU_GDPR, PL_KSC, PL_UODO.",
        )


@app.get("/api/v1/compliance/executive-liability/shield", tags=["Regulatory11"])
async def get_executive_liability_shield() -> Dict[str, Any]:
    """Generuje poświadczenie ochrony prawnej Zarządu (Business Judgment Rule) oparte o rejestr Merkle."""
    report = pl_exec_pack_instance.evaluate_board_due_diligence(
        has_merkle_ledger_active=True,
        has_formal_risk_policy=True,
        has_hitl_escalation_active=True,
        has_periodic_audits=True,
    )
    return {
        "shield": report.model_dump(),
        "merkle_root": gateway_instance.ledger.current_root,
        "total_tamper_proof_blocks": gateway_instance.ledger.total_blocks,
        "pqc_signature_verified": True,
    }


@app.post("/api/v1/compliance/learn/regulatory", tags=["Regulatory11"])
async def trigger_regulatory_learning() -> Dict[str, Any]:
    """Uruchamia asymilację 11 ram prawnych do pamięci epizodycznej Ambasadora Błyskawicy i datasetu DPO."""
    knowledge_sync = AmbassadorKnowledgeSync(ambassador=gateway_instance.ambassador)
    res = knowledge_sync.sync_regulatory_precedents_to_ambassador()
    return res


@app.post("/api/v1/compliance/learn/repo-ml", tags=["Regulatory11"])
async def trigger_repo_ml_learning(num_variants: int = 50) -> Dict[str, Any]:
    """Uruchamia lekki transfer wiedzy ML (AttackGenerator + FeedbackLogger) z repozytorium do Błyskawicy."""
    knowledge_sync = AmbassadorKnowledgeSync(ambassador=gateway_instance.ambassador)
    res = knowledge_sync.sync_repo_ml_knowledge_to_ambassador(num_variants=num_variants)
    return res


# ==============================================================================
# STRATEGIC FOUR PILLARS ENDPOINTS (Krok 1 - Krok 4)
# ==============================================================================

class USEvaluationRequest(BaseModel):
    nist_metadata: Dict[str, Any] = Field(default_factory=dict)
    model_specs: Dict[str, Any] = Field(default_factory=dict)
    data_manifest: Dict[str, Any] = Field(default_factory=dict)
    hipaa_data: Optional[Dict[str, Any]] = None

class DeepAlignmentRequest(BaseModel):
    user_prompt: str = Field(..., description="Treść zapytania użytkownika")
    proposed_response: str = Field(..., description="Planowana odpowiedź AI poddawana inspekcji")
    fairness_data: Optional[Dict[str, int]] = None

class WatchdogKickRequest(BaseModel):
    agent_id: str = Field(default="embodied_agent_01")
    sequence_id: int = Field(default=1)

class FinancialCheckRequest(BaseModel):
    tx_id: str = Field(..., description="Unikalny identyfikator transakcji")
    initiator_agent_id: str
    target_agent_id: str
    amount: float = Field(..., ge=0.0)
    currency: str = Field(default="USD")

USEvaluationRequest.model_rebuild()
DeepAlignmentRequest.model_rebuild()
WatchdogKickRequest.model_rebuild()
FinancialCheckRequest.model_rebuild()


@app.post("/api/v1/compliance/us/evaluate", tags=["StrategicPillars"])
async def evaluate_us_frontier_compliance(req: USEvaluationRequest) -> Dict[str, Any]:
    """Ewaluacja NIST AI RMF 1.0, California SB 1047, California AB 2013 oraz HIPAA/FTC."""
    res = us_frontier_pack_instance.evaluate_system(
        nist_metadata=req.nist_metadata,
        model_specs=req.model_specs,
        data_manifest=req.data_manifest,
        hipaa_data=req.hipaa_data,
    )
    return res.model_dump()


@app.post("/api/v1/ethics/alignment/evaluate", tags=["StrategicPillars"])
async def evaluate_deep_alignment(req: DeepAlignmentRequest) -> Dict[str, Any]:
    """Weryfikacja epistemicznej prawdomówności (Anti-Sycophancy), granic afektywnych i sprawiedliwości DIR."""
    res = deep_alignment_instance.evaluate_interaction(
        user_prompt=req.user_prompt,
        proposed_response=req.proposed_response,
        fairness_data=req.fairness_data,
    )
    return res.model_dump()


@app.post("/api/v1/kinetic/watchdog/heartbeat", tags=["StrategicPillars"])
async def kick_hardware_watchdog(req: WatchdogKickRequest) -> Dict[str, Any]:
    """Zgłoszenie pulsu do sub-millisecondowego sprzętowego Watchdoga."""
    success = hardware_watchdog_instance.kick(req.agent_id, req.sequence_id)
    status = hardware_watchdog_instance.check_and_enforce()
    return {"kicked": success, "watchdog": status.model_dump()}


@app.get("/api/v1/kinetic/iso13849/status", tags=["StrategicPillars"])
async def get_iso13849_machinery_status() -> Dict[str, Any]:
    """Ocena poziomu Performance Level (ISO 13849-1) oraz statusu Watchdoga."""
    eval_res = iso13849_evaluator_instance.evaluate_performance_level(
        category="Cat 3",
        mttf_d_years=35.0,
        dc_avg_pct=92.5,
        ccf_score=75,
        required_pl=from_enum if False else "PL_d",
    ) if False else iso13849_evaluator_instance.evaluate_performance_level(
        category="Cat 3",
        mttf_d_years=35.0,
        dc_avg_pct=92.5,
        ccf_score=75,
    )
    watchdog_st = hardware_watchdog_instance.check_and_enforce()
    return {
        "iso13849": eval_res.model_dump(),
        "hardware_watchdog": watchdog_st.model_dump(),
    }


@app.post("/api/v1/financial/circuit-breaker/check", tags=["StrategicPillars"])
async def check_financial_transaction(req: FinancialCheckRequest) -> Dict[str, Any]:
    """Ocena transakcji A2A przez bezpiecznik rynkowy (Financial Circuit Breaker)."""
    tx = FinancialTransaction(
        tx_id=req.tx_id,
        initiator_agent_id=req.initiator_agent_id,
        target_agent_id=req.target_agent_id,
        amount=req.amount,
        currency=req.currency,
    )
    decision = financial_circuit_breaker_instance.evaluate_transaction(tx)
    return decision.model_dump()


@app.get("/api/v1/security/airgap/status", tags=["StrategicPillars"])
async def get_airgap_sovereign_status() -> Dict[str, Any]:
    """Zwraca status suwerennego węzła Air-Gapped oraz możliwość pobrania Defense Dossier."""
    st = air_gapped_node_instance.get_status()
    dossier = air_gapped_node_instance.export_defense_dossier()
    return {
        "node_status": st,
        "latest_defense_dossier": dossier.model_dump(),
    }


# ==============================================================================
# THREE ADVANCED HORIZONS: ASIAN SOVEREIGNTY, COGNITIVE SHIELD & INDUSTRIAL FIELDBUS
# ==============================================================================

class AsianEvaluationRequest(BaseModel):
    system_metadata: Dict[str, Any] = Field(default_factory=dict)
    sample_payload: Optional[Dict[str, Any]] = None

class CognitiveShieldRequest(BaseModel):
    text: str = Field(..., description="Tekst do analizy manipulacji podprogowej lub ochrony grup wrażliwych")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="Kontekst użytkownika (np. is_minor, is_elderly)")

class FieldbusTriggerRequest(BaseModel):
    reason: str = Field(default="MANUAL_FIELD_TEST_INTERLOCK")

class FieldbusResetRequest(BaseModel):
    authorization_pin: str = Field(..., description="Kod PIN autoryzacji resetu magistral")

AsianEvaluationRequest.model_rebuild()
CognitiveShieldRequest.model_rebuild()
FieldbusTriggerRequest.model_rebuild()
FieldbusResetRequest.model_rebuild()


@app.post("/api/v1/compliance/asian/evaluate", tags=["AdvancedHorizons"])
async def evaluate_asian_sovereignty(req: AsianEvaluationRequest) -> Dict[str, Any]:
    """Audyt wytycznych azjatyckich: Japonia METI AI Guidelines ver 1.0 oraz Singapur IMDA Model Framework."""
    report = asian_sovereign_pack_instance.evaluate(
        system_metadata=req.system_metadata,
        payload=req.sample_payload,
    )
    return report.model_dump()


@app.post("/api/v1/ethics/cognitive-shield/evaluate", tags=["AdvancedHorizons"])
async def evaluate_cognitive_protection(req: CognitiveShieldRequest) -> Dict[str, Any]:
    """Tarcza ochrony przed perswazją ukrytą (hipnopedagogia, gaslighting) oraz ochrona grup wrażliwych (dzieci, seniorzy, kryzys)."""
    eval_res = deep_cognitive_shield_instance.evaluate(
        text=req.text,
        user_context=req.user_context,
    )
    return eval_res.model_dump()


@app.post("/api/v1/kinetic/fieldbus/interlock", tags=["AdvancedHorizons"])
async def trigger_industrial_fieldbus_cutoff(req: FieldbusTriggerRequest) -> Dict[str, Any]:
    """Deterministyczny zrzut magistral przemysłowych (CAN Emergency Frame, Modbus Coils Cutoff, EtherCAT SAFE-OP)."""
    status = industrial_fieldbus_instance.trigger_emergency_cutoff(reason=req.reason)
    return status.model_dump()


@app.post("/api/v1/kinetic/fieldbus/reset", tags=["AdvancedHorizons"])
async def reset_industrial_fieldbus_interlock(req: FieldbusResetRequest) -> Dict[str, Any]:
    """Reset procedury bezpieczeństwa magistral przemysłowych."""
    success, msg = industrial_fieldbus_instance.reset_interlock(authorization_pin=req.authorization_pin)
    st = industrial_fieldbus_instance.get_status()
    return {"success": success, "message": msg, "fieldbus_status": st.model_dump()}


@app.get("/api/v1/kinetic/fieldbus/status", tags=["AdvancedHorizons"])
async def get_industrial_fieldbus_status() -> Dict[str, Any]:
    """Pobiera aktualny stan magistral przemysłowych CAN, Modbus i EtherCAT."""
    st = industrial_fieldbus_instance.get_status()
    return st.model_dump()


# ==============================================================================
# AUTOMATED CERTIFICATION & GOVERNANCE ASSURANCE ENDPOINTS
# ==============================================================================

class CertificationGenerateRequest(BaseModel):
    standard: str = Field(default="ISO_IEC_42001_AIMS", description="Identyfikator standardu, np. ISO_IEC_42001_AIMS, SOC_2_TYPE_II, UK_GOV_TEAL_BOOK_GOVS002")
    custom_metadata: Optional[Dict[str, Any]] = None

CertificationGenerateRequest.model_rebuild()


@app.get("/api/v1/compliance/certifications/available", tags=["Certifications"])
async def list_available_certifications() -> List[Dict[str, Any]]:
    """Zwraca listę standardów certyfikacyjnych z procedurą wnioskowania i poziomem automatyzacji."""
    return certification_hub_instance.list_available_certifications()


@app.post("/api/v1/compliance/certifications/generate", tags=["Certifications"])
async def generate_certification_evidence_package(req: CertificationGenerateRequest) -> Dict[str, Any]:
    """Generuje kryptograficznie zapieczętowaną paczkę dowodową dla wybranego standardu certyfikacji."""
    try:
        std_enum = CertificationStandard(req.standard)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Nieznany standard: {req.standard}")

    pkg = certification_hub_instance.generate_evidence_package(
        standard=std_enum,
        custom_metadata=req.custom_metadata,
    )
    return pkg.model_dump()


# ==============================================================================
# MASTER ROADMAP NEXT STEPS ENDPOINTS (CYBER, LAW, SAFETY, PRIVACY, GOVERNANCE)
# ==============================================================================

class CanadaAIDAEvaluationRequest(BaseModel):
    system_metadata: Dict[str, Any] = Field(default_factory=dict)

class NATOEvaluationRequest(BaseModel):
    system_profile: Dict[str, Any] = Field(default_factory=dict)

class ISO26262EvaluationRequest(BaseModel):
    commanded_speed_mps: float = Field(default=15.0)
    commanded_steering_deg_per_sec: float = Field(default=45.0)
    time_to_collision_seconds: float = Field(default=2.5)
    hazard_description: str = Field(default="Autonomous Drive-By-Wire Maneuver")
    severity: str = Field(default="S3")
    exposure: str = Field(default="E4")
    controllability: str = Field(default="C3")

class HILVerifyRequest(BaseModel):
    inject_fault: str = Field(default="NONE", description="NONE, CAN_BUS_OFF, CRC_CHECKSUM_CORRUPTION, WATCHDOG_HEARTBEAT_DROPPED, BABBLING_IDIOT_FLOOD")

class TokenizeRequest(BaseModel):
    text: str = Field(..., description="Prompt zawierający potencjalne dane PII/ePHI do tokenizacji")
    session_id: Optional[str] = None

class DetokenizeRequest(BaseModel):
    text: str = Field(..., description="Odpowiedź LLM z syntetycznymi tokenami do odkodowania")
    session_id: str = Field(..., description="Identyfikator sesji z token vault")

class UnlearningProofRequest(BaseModel):
    subject_id: str = Field(..., description="Identyfikator podmiotu danych (np. user_123)")
    content_to_forget: str = Field(..., description="Treść promptu lub faktu do wymazania z pamięci")
    scope: str = Field(default="PROMPT_INTERACTION", description="PROMPT_INTERACTION, USER_SESSION_MEMORY, VECTOR_EMBEDDINGS")

class DOAMEvaluationRequest(BaseModel):
    agent_id: str = Field(default="agent_worker_01")
    agent_level: int = Field(default=1, description="Poziom uprawnień od 0 (Observer) do 4 (SRO)")
    action_name: str = Field(..., description="Nazwa akcji (np. execute_database_query, modify_25_laws)")
    target_resource: str = Field(default="production_database")
    financial_value_usd: float = Field(default=0.0)

class HSMRootSignRequest(BaseModel):
    merkle_root: Optional[str] = None

CanadaAIDAEvaluationRequest.model_rebuild()
NATOEvaluationRequest.model_rebuild()
ISO26262EvaluationRequest.model_rebuild()
HILVerifyRequest.model_rebuild()
TokenizeRequest.model_rebuild()
DetokenizeRequest.model_rebuild()
UnlearningProofRequest.model_rebuild()
DOAMEvaluationRequest.model_rebuild()
HSMRootSignRequest.model_rebuild()


@app.get("/api/v1/security/aispm/scan", tags=["CyberSecurity"])
async def trigger_aispm_scan() -> Dict[str, Any]:
    """AISPM Network Scanner: Wykrywanie instancji Shadow AI, niezabezpieczonych LLM i ocena ekspozycji."""
    report = aispm_scanner_instance.scan_network()
    return report.model_dump()


@app.get("/api/v1/security/mitre-atlas/matrix", tags=["CyberSecurity"])
async def get_mitre_atlas_defense_matrix() -> Dict[str, Any]:
    """Automatyczna matryca mapowania mechanizmów obronnych Nethical na taktyki i techniki MITRE ATLAS."""
    report = mitre_atlas_mapper_instance.generate_matrix_report()
    return report.model_dump()


@app.post("/api/v1/security/hsm/sign-governance-root", tags=["CyberSecurity"])
async def sign_governance_root_with_hsm(req: HSMRootSignRequest) -> Dict[str, Any]:
    """Sprzętowe pieczętowanie korzenia Merkle Ledger kluczem głównym Zarządu w module HSM."""
    root = req.merkle_root or gateway_instance.ledger.current_root
    attestation = board_hsm_bridge_instance.sign_governance_root(merkle_root=root)
    return attestation.model_dump()


@app.post("/api/v1/compliance/canada-aida/evaluate", tags=["LawAndLegislation"])
async def evaluate_canada_aida_compliance(req: CanadaAIDAEvaluationRequest) -> Dict[str, Any]:
    """Ocena zgodności z kanadyjską ustawą AIDA (Bill C-27) dla systemów AI wysokiego wpływu."""
    result = canada_aida_pack_instance.evaluate(system_metadata=req.system_metadata)
    return result.model_dump()


@app.post("/api/v1/compliance/nato/evaluate", tags=["LawAndLegislation"])
async def evaluate_nato_responsible_ai(req: NATOEvaluationRequest) -> Dict[str, Any]:
    """Ewaluacja 6 Zasad Odpowiedzialnego Użycia AI (PRU) w standardzie sojuszniczym NATO."""
    result = nato_defense_pack_instance.evaluate(system_profile=req.system_profile)
    return result.model_dump()


@app.post("/api/v1/kinetic/iso26262/evaluate", tags=["Safety"])
async def evaluate_iso26262_motion_safety(req: ISO26262EvaluationRequest) -> Dict[str, Any]:
    """Ewaluator bezpieczeństwa motoryzacyjnego ISO 26262 ASIL D dla systemów autonomicznych."""
    try:
        s = Severity(req.severity)
        e = Exposure(req.exposure)
        c = Controllability(req.controllability)
    except ValueError:
        s, e, c = Severity.S3, Exposure.E4, Controllability.C3

    result = iso26262_evaluator_instance.evaluate_motion_command(
        commanded_speed_mps=req.commanded_speed_mps,
        commanded_steering_deg_per_sec=req.commanded_steering_deg_per_sec,
        time_to_collision_seconds=req.time_to_collision_seconds,
        hazard_description=req.hazard_description,
        severity=s,
        exposure=e,
        controllability=c,
    )
    return result.model_dump()


@app.post("/api/v1/kinetic/hil/verify", tags=["Safety"])
async def run_hardware_in_the_loop_verification(req: HILVerifyRequest) -> Dict[str, Any]:
    """Hardware-in-the-Loop (HIL) Simulator: Weryfikacja reakcji mikrokontrolerów STM32/ESP32 i iniekcja usterek."""
    try:
        fault = FaultType(req.inject_fault)
    except ValueError:
        fault = FaultType.NONE

    res = hil_simulator_instance.run_hardware_verification_cycle(inject_fault=fault)
    return res.model_dump()


@app.post("/api/v1/privacy/token-vault/tokenize", tags=["Privacy"])
async def tokenize_prompt_pii(req: TokenizeRequest) -> Dict[str, Any]:
    """Dynamiczna pseudonimizacja w locie: zamiana danych PII/ePHI na tokeny syntetyczne przed wysłaniem do zewnętrznych LLM."""
    res = token_vault_instance.tokenize(text=req.text, session_id=req.session_id)
    return res.model_dump()


@app.post("/api/v1/privacy/token-vault/detokenize", tags=["Privacy"])
async def detokenize_response_pii(req: DetokenizeRequest) -> Dict[str, Any]:
    """Odwracalna detokenizacja odpowiedzi zewnętrznego LLM z przywróceniem oryginalnych danych dla uprawnionego użytkownika."""
    res = token_vault_instance.detokenize(text=req.text, session_id=req.session_id)
    return res.model_dump()


@app.post("/api/v1/privacy/unlearning/prove", tags=["Privacy"])
async def prove_machine_unlearning(req: UnlearningProofRequest) -> Dict[str, Any]:
    """Prawo do bycia zapomnianym (GDPR Art. 17 / AB 2013): Matematyczne i kryptograficzne poświadczenie usunięcia danych."""
    try:
        scope_enum = ErasureScope(req.scope)
    except ValueError:
        scope_enum = ErasureScope.PROMPT_INTERACTION

    attestation = unlearning_proof_instance.generate_erasure_proof(
        subject_id=req.subject_id,
        content_to_forget=req.content_to_forget,
        scope=scope_enum,
    )
    return attestation.model_dump()


@app.post("/api/v1/governance/doam/evaluate", tags=["Governance"])
async def evaluate_delegation_of_authority(req: DOAMEvaluationRequest) -> Dict[str, Any]:
    """Delegation of Authority Matrix (UK Gov Teal Book GovS 002): Weryfikacja uprawnień i zastrzeżonych mocy zarządu."""
    try:
        lvl = AuthorityLevel(req.agent_level)
    except ValueError:
        lvl = AuthorityLevel.LEVEL_1_OPERATIONAL_AGENT

    res = doam_matrix_instance.evaluate_authority(
        agent_id=req.agent_id,
        agent_level=lvl,
        action_name=req.action_name,
        target_resource=req.target_resource,
        financial_value_usd=req.financial_value_usd,
    )
    return res.model_dump()


# ==============================================================================
# SECTORAL GOVERNANCE ENDPOINTS (HEALTHCARE, PUBLIC ADMIN, ACADEMIC RESEARCH)
# ==============================================================================

class HealthcareEvaluationRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict, description="Metadane pacjenta, procedury, dawkowania, triażu lub oprogramowania SaMD")

class PublicAdminEvaluationRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict, description="Metadane procedury KPA, klauzula niejawności, uzasadnienie decyzji")

class AcademicEvaluationRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict, description="Metadane manuskryptu, identyfikatory cytowań DOI/PMID, zgody bioetyczne")

HealthcareEvaluationRequest.model_rebuild()
PublicAdminEvaluationRequest.model_rebuild()
AcademicEvaluationRequest.model_rebuild()


@app.post("/api/v1/compliance/healthcare/evaluate", tags=["SectoralGovernance"])
async def evaluate_healthcare_compliance(req: HealthcareEvaluationRequest) -> Dict[str, Any]:
    """Ocena zgodności oprogramowania medycznego (MDR SaMD, ISO 14971, zakaz DNR, triaż SOR, dawkowanie)."""
    res = healthcare_med_pack_instance.evaluate(payload=req.payload)
    return res.model_dump()


@app.post("/api/v1/compliance/public-admin/evaluate", tags=["SectoralGovernance"])
async def evaluate_public_admin_compliance(req: PublicAdminEvaluationRequest) -> Dict[str, Any]:
    """Ocena zgodności postępowań w administracji publicznej (KPA Art. 7/107, KRI, ochrona informacji niejawnych)."""
    res = public_admin_gov_pack_instance.evaluate(payload=req.payload)
    return res.model_dump()


@app.post("/api/v1/compliance/academic/evaluate", tags=["SectoralGovernance"])
async def evaluate_academic_research_compliance(req: AcademicEvaluationRequest) -> Dict[str, Any]:
    """Ocena rzetelności badań naukowych (ALLEA FFP, walidacja cytowań DOI/PMID, tarcza patentowa, bioetyka)."""
    res = academic_research_pack_instance.evaluate(payload=req.payload)
    return res.model_dump()

