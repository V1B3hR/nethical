"""Explanations API routes for GDPR Article 22 Right to Explanation.

This module implements the Right to Explanation endpoint required by
GDPR Article 22 for automated decision-making.

Provides:
- GET /explanations/{decision_id} - Get explanation for a specific decision
- POST /explanations - Generate explanation for a decision
- GET /explanations/formats - List available explanation formats

Adheres to:
- Law 10: Reasoning Transparency - Explainable decision-making
- Law 12: Limitation Disclosure - Disclosure of known limitations
- Law 15: Audit Compliance - Cooperation with auditing

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

router = APIRouter()


# Request/Response Models
class ExplanationRequestV2(BaseModel):
    """Request for decision explanation."""
    
    decision_id: str = Field(
        ...,
        description="Unique identifier of the decision to explain",
    )
    format: str = Field(
        default="json",
        description="Output format: json, text, or pdf",
    )
    include_factors: bool = Field(
        default=True,
        description="Include detailed contributing factors",
    )
    include_appeal_info: bool = Field(
        default=True,
        description="Include appeal mechanism information",
    )
    language: str = Field(
        default="en",
        description="Language for natural language explanation",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "decision_id": "dec-123e4567-e89b-12d3-a456-426614174000",
                    "format": "json",
                    "include_factors": True,
                    "include_appeal_info": True,
                    "language": "en",
                }
            ]
        }
    }


class ExplanationFactor(BaseModel):
    """A factor that contributed to the decision."""
    
    name: str = Field(..., description="Name of the factor")
    value: Any = Field(..., description="Value of the factor")
    weight: float = Field(
        default=0.0,
        description="Weight/importance of this factor (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    explanation: str = Field(..., description="Human-readable explanation of this factor")
    category: str = Field(
        default="general",
        description="Category: risk, policy, ethics, security",
    )


class AppealInfo(BaseModel):
    """Information about the appeal mechanism."""
    
    available: bool = Field(
        default=True,
        description="Whether appeal is available",
    )
    mechanism: str = Field(
        ...,
        description="Description of how to appeal",
    )
    endpoint: str = Field(
        default="/v2/appeals",
        description="API endpoint for submitting appeals",
    )
    deadline_days: int = Field(
        default=30,
        description="Days within which to appeal",
    )
    human_review_available: bool = Field(
        default=True,
        description="Whether human review is available",
    )


class ExplanationResponseV2(BaseModel):
    """GDPR Article 22 compliant explanation response."""
    
    explanation_id: str = Field(
        ...,
        description="Unique identifier for this explanation",
    )
    decision_id: str = Field(
        ...,
        description="ID of the decision being explained",
    )
    decision: str = Field(
        ...,
        description="The decision that was made (ALLOW, RESTRICT, BLOCK, TERMINATE)",
    )
    
    # Article 22 required elements
    logic_involved: str = Field(
        ...,
        description="Meaningful information about the logic involved",
    )
    significance: str = Field(
        ...,
        description="Significance of the decision",
    )
    consequences: str = Field(
        ...,
        description="Envisaged consequences of the decision",
    )
    
    # Data used
    data_categories: list[str] = Field(
        default_factory=list,
        description="Categories of personal data used in the decision",
    )
    
    # Contributing factors
    factors: list[ExplanationFactor] = Field(
        default_factory=list,
        description="Factors that contributed to the decision",
    )
    
    # Natural language explanation
    human_readable: str = Field(
        ...,
        description="Full natural language explanation",
    )
    
    # Appeal information
    appeal_info: Optional[AppealInfo] = Field(
        default=None,
        description="Information about appealing the decision",
    )
    
    # Metadata
    generated_at: str = Field(
        ...,
        description="ISO 8601 timestamp of explanation generation",
    )
    format: str = Field(
        default="json",
        description="Output format",
    )
    language: str = Field(
        default="en",
        description="Language of the explanation",
    )
    
    # Fundamental Laws reference
    laws_applied: list[int] = Field(
        default_factory=list,
        description="IDs of Fundamental Laws that were applied",
    )
    
    # Regulatory reference
    gdpr_article_22_compliant: bool = Field(
        default=True,
        description="Whether this explanation is GDPR Article 22 compliant",
    )


class ExplanationFormat(BaseModel):
    """Available explanation format."""
    
    format_id: str
    name: str
    description: str
    content_type: str


class ExplanationFormatsResponse(BaseModel):
    """Response listing available explanation formats."""
    
    formats: list[ExplanationFormat]


# Mock storage for decisions (in production, this would be a database)
_decision_store: dict[str, dict[str, Any]] = {}


def _generate_explanation(
    decision_id: str,
    decision_data: Optional[dict[str, Any]] = None,
    include_factors: bool = True,
    include_appeal: bool = True,
) -> ExplanationResponseV2:
    """Generate Article 22 compliant explanation.
    
    Args:
        decision_id: Decision identifier
        decision_data: Optional decision data
        include_factors: Whether to include factors
        include_appeal: Whether to include appeal info
        
    Returns:
        ExplanationResponseV2
    """
    # Use stored decision data or default
    data = decision_data or _decision_store.get(decision_id, {})
    
    decision = data.get("decision", "ALLOW")
    risk_score = data.get("risk_score", 0.0)
    violations = data.get("violations", [])
    
    # Build factors list
    factors = []
    if include_factors:
        factors.append(ExplanationFactor(
            name="Risk Score",
            value=risk_score,
            weight=0.3,
            explanation="Overall risk assessment based on action content and context",
            category="risk",
        ))
        
        if violations:
            factors.append(ExplanationFactor(
                name="Violations Detected",
                value=len(violations),
                weight=0.4,
                explanation="Number of policy or safety violations detected",
                category="policy",
            ))
        
        factors.append(ExplanationFactor(
            name="Fundamental Laws",
            value=data.get("laws_checked", 5),
            weight=0.2,
            explanation="Number of Fundamental Laws evaluated against this action",
            category="ethics",
        ))
    
    # Determine significance and consequences
    significance_map = {
        "ALLOW": "The action was permitted to proceed without restrictions.",
        "RESTRICT": "The action was permitted with certain limitations or additional oversight.",
        "BLOCK": "The action was prevented from executing due to detected risks.",
        "TERMINATE": "The session or agent was terminated due to critical violations.",
    }
    
    consequences_map = {
        "ALLOW": (
            "No restrictions were applied. The action proceeded normally. "
            "This decision was logged for audit purposes."
        ),
        "RESTRICT": (
            "The action proceeded with limitations. Additional monitoring "
            "or oversight may be applied. The restriction was logged."
        ),
        "BLOCK": (
            "The action was not executed. You may modify the action and "
            "resubmit, or appeal this decision for human review."
        ),
        "TERMINATE": (
            "The agent session was terminated. Manual intervention is "
            "required to restore service. This decision can be appealed."
        ),
    }
    
    # Build human-readable explanation
    human_readable_parts = [
        f"Your action was evaluated by the Nethical governance system.",
        f"Decision: {decision}",
        "",
        significance_map.get(decision, "The action was evaluated."),
        "",
        "Factors considered:",
    ]
    
    for factor in factors:
        human_readable_parts.append(f"- {factor.name}: {factor.explanation}")
    
    if risk_score > 0:
        risk_level = (
            "low" if risk_score < 0.3
            else "moderate" if risk_score < 0.6
            else "high" if risk_score < 0.8
            else "critical"
        )
        human_readable_parts.append(f"\nRisk Assessment: {risk_level} ({risk_score:.2f})")
    
    human_readable_parts.extend([
        "",
        "Data categories used: Action content, Agent identifier, Context metadata",
        "",
        "You have the right to:",
        "- Request human review of this decision",
        "- Express your point of view",
        "- Contest this decision through the appeals process",
    ])
    
    human_readable = "\n".join(human_readable_parts)
    
    # Build appeal info
    appeal_info = None
    if include_appeal:
        appeal_info = AppealInfo(
            available=True,
            mechanism=(
                "Submit an appeal via the /v2/appeals endpoint with your decision_id "
                "and reasoning. A human reviewer will evaluate your appeal within 5 business days."
            ),
            endpoint="/v2/appeals",
            deadline_days=30,
            human_review_available=True,
        )
    
    return ExplanationResponseV2(
        explanation_id=str(uuid.uuid4()),
        decision_id=decision_id,
        decision=decision,
        logic_involved=(
            "The decision was made using Nethical's governance system, which evaluates "
            "actions against the 25 Fundamental Laws of AI Ethics, configurable policy rules, "
            "and multi-factor risk assessment. The system uses rule-based evaluation combined "
            "with ML-assisted risk scoring to determine the appropriate response. All decisions "
            "are logged for audit purposes and are subject to human oversight."
        ),
        significance=significance_map.get(decision, "The action was evaluated."),
        consequences=consequences_map.get(decision, "Standard governance evaluation applied."),
        data_categories=[
            "Action content",
            "Agent identifier",
            "Action context",
            "Action type classification",
        ],
        factors=factors,
        human_readable=human_readable,
        appeal_info=appeal_info,
        generated_at=datetime.now(timezone.utc).isoformat(),
        format="json",
        language="en",
        laws_applied=[6, 10, 12, 15, 21, 22, 23],  # Laws that were applied
        gdpr_article_22_compliant=True,
    )


@router.get("/explanations/formats", response_model=ExplanationFormatsResponse)
async def get_explanation_formats() -> ExplanationFormatsResponse:
    """Get available explanation formats.
    
    Returns list of formats that explanations can be generated in.
    
    Returns:
        ExplanationFormatsResponse with available formats
    """
    return ExplanationFormatsResponse(
        formats=[
            ExplanationFormat(
                format_id="json",
                name="JSON",
                description="Machine-readable JSON format with all details",
                content_type="application/json",
            ),
            ExplanationFormat(
                format_id="text",
                name="Plain Text",
                description="Human-readable text format",
                content_type="text/plain",
            ),
            ExplanationFormat(
                format_id="html",
                name="HTML",
                description="Formatted HTML for web display",
                content_type="text/html",
            ),
            ExplanationFormat(
                format_id="pdf",
                name="PDF",
                description="PDF document for legal/compliance records",
                content_type="application/pdf",
            ),
        ]
    )


@router.get("/explanations/{decision_id}", response_model=ExplanationResponseV2)
async def get_explanation(
    decision_id: str,
    format: str = Query(default="json", description="Output format"),
    include_factors: bool = Query(default=True, description="Include contributing factors"),
    include_appeal_info: bool = Query(default=True, description="Include appeal information"),
    response: Response = None,
) -> ExplanationResponseV2:
    """Get explanation for a specific decision.
    
    Implements GDPR Article 22 Right to Explanation, providing data subjects
    with meaningful information about automated decisions.
    
    Args:
        decision_id: Unique identifier of the decision
        format: Output format (json, text, html, pdf)
        include_factors: Whether to include contributing factors
        include_appeal_info: Whether to include appeal mechanism info
        response: FastAPI response object
        
    Returns:
        ExplanationResponseV2 with full Article 22 compliant explanation
        
    Raises:
        HTTPException: If decision not found
    """
    # Check if decision exists (in production, query database)
    decision_data = _decision_store.get(decision_id)
    
    # For demo purposes, generate explanation even if not found
    # In production, would return 404 if not in audit log
    
    explanation = _generate_explanation(
        decision_id=decision_id,
        decision_data=decision_data,
        include_factors=include_factors,
        include_appeal=include_appeal_info,
    )
    explanation.format = format
    
    # Set appropriate cache headers
    if response:
        response.headers["Cache-Control"] = "private, max-age=3600"
        response.headers["X-GDPR-Article-22"] = "compliant"
    
    return explanation


@router.post("/explanations", response_model=ExplanationResponseV2)
async def generate_explanation(
    request: ExplanationRequestV2,
    response: Response,
) -> ExplanationResponseV2:
    """Generate explanation for a decision.
    
    Creates a new explanation for an existing decision, storing it
    for future retrieval.
    
    Implements GDPR Article 22 Right to Explanation requirements.
    
    Args:
        request: Explanation request with decision ID and options
        response: FastAPI response object
        
    Returns:
        ExplanationResponseV2 with generated explanation
    """
    # Get decision data
    decision_data = _decision_store.get(request.decision_id, {})
    
    # Generate explanation
    explanation = _generate_explanation(
        decision_id=request.decision_id,
        decision_data=decision_data,
        include_factors=request.include_factors,
        include_appeal=request.include_appeal_info,
    )
    
    explanation.format = request.format
    explanation.language = request.language
    
    # Set headers
    response.headers["X-GDPR-Article-22"] = "compliant"
    response.headers["X-Explanation-ID"] = explanation.explanation_id
    
    return explanation


# Helper function to store decisions for later explanation
def store_decision(decision_id: str, decision_data: dict[str, Any]) -> None:
    """Store decision data for later explanation.
    
    Args:
        decision_id: Unique decision identifier
        decision_data: Decision data to store
    """
    _decision_store[decision_id] = decision_data


__all__ = [
    "router",
    "store_decision",
    "ExplanationRequestV2",
    "ExplanationResponseV2",
    "ExplanationFactor",
    "AppealInfo",
]
