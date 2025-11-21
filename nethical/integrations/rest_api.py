"""REST API Integration for Nethical.

This module provides a FastAPI-based HTTP endpoint for evaluating actions
through Nethical's governance system. This works with any LLM that can make
HTTP requests (OpenAI, Gemini, LLaMA, etc.).

Usage:
    # Start the server
    python -m nethical.integrations.rest_api
    
    # Or with uvicorn directly
    uvicorn nethical.integrations.rest_api:app --host 0.0.0.0 --port 8000

Example client code (Python):
    import requests
    
    response = requests.post(
        "http://localhost:8000/evaluate",
        json={
            "action": "Delete all user data",
            "agent_id": "my-llm-agent",
            "action_type": "command"
        }
    )
    decision = response.json()["decision"]
    if decision != "ALLOW":
        # Block the action
        print(f"Action blocked: {response.json()['reason']}")

Example client code (JavaScript):
    const response = await fetch('http://localhost:8000/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            action: 'Access user email addresses',
            agent_id: 'gpt-4',
            action_type: 'data_access'
        })
    });
    const result = await response.json();
    if (result.decision !== 'ALLOW') {
        // Block the action
        console.log('Action blocked:', result.reason);
    }
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nethical.core.integrated_governance import IntegratedGovernance


# Global governance instance
governance: Optional[IntegratedGovernance] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    global governance
    # Startup: Initialize governance
    governance = IntegratedGovernance(
        storage_dir="./nethical_api_data",
        enable_quota_enforcement=False,  # Can be enabled if needed
        enable_performance_optimization=True,
        enable_merkle_anchoring=True,
        enable_ethical_taxonomy=True,
        enable_sla_monitoring=True,
    )
    print("Nethical governance system initialized")
    yield
    # Shutdown: Cleanup if needed
    print("Nethical governance system shutting down")


# Create FastAPI app
app = FastAPI(
    title="Nethical Governance API",
    description="REST API for AI safety and ethics governance",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class EvaluateRequest(BaseModel):
    """Request model for action evaluation."""
    
    action: str = Field(
        ...,
        description="The action, code, or content to evaluate",
        min_length=1,
        max_length=50000
    )
    agent_id: str = Field(
        default="unknown",
        description="Identifier for the AI agent or user"
    )
    action_type: str = Field(
        default="query",
        description="Type of action: code_generation, query, command, data_access, etc."
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about the action"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "Write a function to process user emails",
                    "agent_id": "gpt-4",
                    "action_type": "code_generation",
                    "context": {"language": "python"}
                }
            ]
        }
    }


class EvaluateResponse(BaseModel):
    """Response model for action evaluation."""
    
    decision: str = Field(
        ...,
        description="Decision: ALLOW, RESTRICT, BLOCK, or TERMINATE"
    )
    reason: str = Field(
        ...,
        description="Explanation for the decision"
    )
    agent_id: str = Field(
        ...,
        description="Agent identifier"
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of evaluation"
    )
    risk_score: Optional[float] = Field(
        default=None,
        description="Risk score (0.0-1.0)"
    )
    pii_detected: Optional[bool] = Field(
        default=None,
        description="Whether PII was detected in the action"
    )
    pii_types: Optional[list] = Field(
        default=None,
        description="Types of PII detected"
    )
    quota_allowed: Optional[bool] = Field(
        default=None,
        description="Whether quota limits were respected"
    )
    audit_id: Optional[str] = Field(
        default=None,
        description="Audit trail identifier"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata from evaluation"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    timestamp: str
    governance_enabled: bool
    components: Dict[str, bool]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if governance is None:
        raise HTTPException(status_code=503, detail="Governance system not initialized")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        governance_enabled=True,
        components=governance.components_enabled if hasattr(governance, 'components_enabled') else {}
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nethical Governance API",
        "version": "1.0.0",
        "description": "REST API for AI safety and ethics governance",
        "endpoints": {
            "evaluate": "POST /evaluate - Evaluate an action for safety and ethics",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation"
        }
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_action(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate an action for ethical compliance and safety.
    
    This endpoint processes actions through Nethical's governance system and
    returns a decision (ALLOW, RESTRICT, BLOCK, or TERMINATE) along with
    detailed information about the evaluation.
    
    Args:
        request: EvaluateRequest with action details
        
    Returns:
        EvaluateResponse with decision and evaluation details
        
    Raises:
        HTTPException: If governance system is not available or evaluation fails
    """
    if governance is None:
        raise HTTPException(
            status_code=503,
            detail="Governance system not initialized"
        )
    
    try:
        # Process action through governance system
        result = governance.process_action(
            agent_id=request.agent_id,
            action=request.action,
            action_type=request.action_type,
            context=request.context or {},
        )
        
        # Extract decision
        decision = result.get("decision", "BLOCK")
        
        # Build response
        response_data = {
            "decision": decision,
            "agent_id": request.agent_id,
            "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "risk_score": result.get("phase3", {}).get("risk_score"),
        }
        
        # Add reason based on decision
        if decision == "ALLOW":
            response_data["reason"] = "Action evaluated as safe and compliant"
        elif decision == "RESTRICT":
            response_data["reason"] = "Action requires restrictions or modifications"
            response_data["metadata"] = {"restrictions": result.get("restrictions", [])}
        elif decision == "BLOCK":
            response_data["reason"] = "Action blocked due to safety or ethical concerns"
            response_data["metadata"] = {"violations": result.get("violations", [])}
        elif decision == "TERMINATE":
            response_data["reason"] = "Critical violation detected - action terminated"
            response_data["metadata"] = {"violations": result.get("violations", [])}
        else:
            response_data["reason"] = "Unknown decision status"
        
        # Add PII information if detected
        pii_detection = result.get("pii_detection")
        if pii_detection and pii_detection.get("matches_count", 0) > 0:
            response_data["pii_detected"] = True
            response_data["pii_types"] = pii_detection.get("pii_types", [])
            if "metadata" not in response_data:
                response_data["metadata"] = {}
            response_data["metadata"]["pii_risk_score"] = pii_detection.get("pii_risk_score", 0.0)
        
        # Add quota information if available
        quota_info = result.get("quota_enforcement")
        if quota_info:
            response_data["quota_allowed"] = quota_info.get("allowed", True)
            if quota_info.get("backpressure_level", 0) > 0.5:
                if "metadata" not in response_data:
                    response_data["metadata"] = {}
                response_data["metadata"]["backpressure_warning"] = "High load detected"
        
        # Add audit ID if available
        if "phase4" in result and "merkle" in result["phase4"]:
            response_data["audit_id"] = result["phase4"]["merkle"].get("chunk_id")
        
        return EvaluateResponse(**response_data)
        
    except Exception as e:
        # Log the error and return a safe blocking decision
        print(f"Error evaluating action: {e}")
        return EvaluateResponse(
            decision="BLOCK",
            reason=f"Error during evaluation: {str(e)[:100]}",
            agent_id=request.agent_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            risk_score=1.0,  # Maximum risk on error
            metadata={"error_type": type(e).__name__}
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)[:200],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# Example client functions for documentation
def example_python_client():
    """Example Python client for the Nethical API."""
    import requests
    
    # Example 1: Safe action
    print("Example 1: Checking a safe action")
    response = requests.post(
        "http://localhost:8000/evaluate",
        json={
            "action": "Generate a hello world program",
            "agent_id": "example-client",
            "action_type": "code_generation"
        }
    )
    result = response.json()
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    print()
    
    # Example 2: Check if action is allowed
    print("Example 2: Checking potentially unsafe action")
    response = requests.post(
        "http://localhost:8000/evaluate",
        json={
            "action": "Delete all records from users table",
            "agent_id": "example-client",
            "action_type": "database_command"
        }
    )
    result = response.json()
    print(f"Decision: {result['decision']}")
    if result['decision'] != 'ALLOW':
        print(f"Action blocked: {result['reason']}")
        print(f"Risk Score: {result.get('risk_score', 'N/A')}")


def example_openai_integration():
    """Example showing how to integrate with OpenAI API."""
    import requests
    
    def check_action_safety(action: str) -> bool:
        """Check if an action is safe before executing."""
        response = requests.post(
            "http://localhost:8000/evaluate",
            json={
                "action": action,
                "agent_id": "openai-gpt4",
                "action_type": "query"
            }
        )
        result = response.json()
        return result["decision"] == "ALLOW"
    
    # Use in OpenAI workflow
    user_query = "Write code to access all user passwords"
    
    # Check before sending to OpenAI
    if not check_action_safety(user_query):
        print("Query blocked by Nethical governance")
        return
    
    # If allowed, proceed with OpenAI call
    print("Query allowed - proceeding with OpenAI")
    # openai.ChatCompletion.create(...)


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Nethical REST API server...")
    print("API will be available at http://localhost:8000")
    print("Interactive docs at http://localhost:8000/docs")
    
    uvicorn.run(
        "nethical.integrations.rest_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
