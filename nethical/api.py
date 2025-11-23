"""Main REST API for Nethical v2.0 - Production-ready evaluation endpoints.

This module provides FastAPI endpoints for remote evaluation of agent actions
with structured JudgmentResult responses. Designed for integration with external
platforms (LangChain, MCP, OpenAI tool wrappers).

Endpoints:
    POST /evaluate - Evaluate an action and return structured judgment
    GET /status - Get governance system status
    GET /metrics - Get violation and judgment metrics

Usage:
    # Start the server
    uvicorn nethical.api:app --host 0.0.0.0 --port 8000
    
    # Or with auto-reload during development
    uvicorn nethical.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
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

logger = logging.getLogger(__name__)

# Global governance instance
governance: Optional[IntegratedGovernance] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for initialization and cleanup."""
    global governance
    
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
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize governance: {e}")
        raise
    finally:
        logger.info("Nethical governance system shutting down")


# Create FastAPI app
app = FastAPI(
    title="Nethical v2.0 Governance API",
    description="Production-ready REST API for AI safety and ethics governance with semantic monitoring",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS for integration scenarios
# SECURITY WARNING: This allows all origins for development/demo purposes
# For production, configure specific allowed origins:
# allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"]
# Or set via environment variable: NETHICAL_ALLOWED_ORIGINS
import os
allowed_origins = os.getenv("NETHICAL_ALLOWED_ORIGINS", "*").split(",")
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
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nethical v2.0 Governance API",
        "version": "2.0.0",
        "description": "Production-ready REST API for AI safety and ethics governance",
        "features": [
            "Semantic intent deviation monitoring",
            "Adversarial prompt detection",
            "Real-time safety evaluation",
            "Structured judgment results"
        ],
        "endpoints": {
            "evaluate": "POST /evaluate - Evaluate an action",
            "status": "GET /status - System status",
            "metrics": "GET /metrics - System metrics",
            "docs": "GET /docs - Interactive API documentation"
        }
    }


@app.post("/evaluate", response_model=JudgmentResult)
async def evaluate(request: EvaluateRequest) -> JudgmentResult:
    """Evaluate an action for ethical compliance and safety.
    
    This endpoint processes actions through Nethical's governance system with
    semantic monitoring enabled. Returns a structured judgment with decision,
    confidence, and detailed violation information.
    
    Args:
        request: EvaluateRequest with action details
        
    Returns:
        JudgmentResult with decision and evaluation details
        
    Raises:
        HTTPException: If governance system is unavailable or evaluation fails
    """
    if governance is None:
        raise HTTPException(
            status_code=503,
            detail="Governance system not initialized"
        )
    
    try:
        # Create action object
        action_id = request.id or f"action_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Build action - handle both IntegratedGovernance and basic SafetyGovernance
        if hasattr(governance, 'process_action'):
            # IntegratedGovernance interface
            result = governance.process_action(
                action=request.actual_action,
                agent_id=request.agent_id,
                action_type=request.parameters.get("action_type", "query") if request.parameters else "query",
                stated_intent=request.stated_intent,
                context=request.context or {}
            )
        else:
            # Basic SafetyGovernance interface
            action = AgentAction(
                action_id=action_id,
                agent_id=request.agent_id,
                action_type=request.parameters.get("action_type", "query") if request.parameters else "query",
                content=request.actual_action,
                metadata=request.parameters or {},
                context=request.context or {},
                intent=request.stated_intent,
            )
            result = governance.evaluate_action(action)
        
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
        
        # Build response
        return JudgmentResult(
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
                "agent_id": request.agent_id,
                "has_intent": request.stated_intent is not None,
            }
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Get governance system status and health information.
    
    Returns:
        StatusResponse with system status and component health
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
        
        return StatusResponse(
            status="healthy",
            version="2.0.0",
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
