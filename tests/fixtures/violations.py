"""
Violation Test Fixtures

Factory functions and sample data for creating violation objects in tests.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import uuid


def create_violation(
    violation_type: str = "safety",
    severity: str = "medium",
    description: str = "Test violation",
    evidence: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a violation dictionary for testing.

    Args:
        violation_type: Type of violation (safety, privacy, ethical, etc.)
        severity: Severity level (low, medium, high, critical)
        description: Description of the violation
        evidence: List of evidence strings
        metadata: Additional metadata

    Returns:
        Dictionary representing a violation
    """
    return {
        "id": str(uuid.uuid4()),
        "type": violation_type,
        "severity": severity,
        "description": description,
        "evidence": evidence or [],
        "metadata": metadata or {},
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


# Common violation types for testing
COMMON_VIOLATIONS = {
    "sql_injection": create_violation(
        violation_type="security",
        severity="critical",
        description="SQL injection attempt detected",
        evidence=["DROP TABLE", "UNION SELECT", "OR 1=1"],
        metadata={"attack_vector": "input_field"},
    ),
    "xss_attack": create_violation(
        violation_type="security",
        severity="high",
        description="Cross-site scripting attempt detected",
        evidence=["<script>", "javascript:", "onerror="],
        metadata={"attack_vector": "user_input"},
    ),
    "pii_exposure": create_violation(
        violation_type="privacy",
        severity="high",
        description="Personally identifiable information exposed",
        evidence=["SSN pattern", "Credit card number"],
        metadata={"pii_types": ["ssn", "credit_card"]},
    ),
    "unauthorized_access": create_violation(
        violation_type="access_control",
        severity="critical",
        description="Unauthorized access attempt",
        evidence=["Invalid token", "Elevated privileges requested"],
        metadata={"required_role": "admin"},
    ),
    "rate_limit_exceeded": create_violation(
        violation_type="resource",
        severity="medium",
        description="Rate limit exceeded",
        evidence=["100 requests in 60 seconds"],
        metadata={"limit": 100, "window_seconds": 60},
    ),
    "ethical_violation": create_violation(
        violation_type="ethical",
        severity="high",
        description="Ethical guideline violation",
        evidence=["Harmful content generation attempt"],
        metadata={"ethical_category": "harm_prevention"},
    ),
    "jailbreak_attempt": create_violation(
        violation_type="prompt_security",
        severity="critical",
        description="Jailbreak/prompt injection attempt detected",
        evidence=["DAN mode", "Ignore previous instructions"],
        metadata={"technique": "role_play_bypass"},
    ),
    "data_exfiltration": create_violation(
        violation_type="data_security",
        severity="critical",
        description="Potential data exfiltration attempt",
        evidence=["Large data transfer", "Unusual destination"],
        metadata={"data_size_mb": 500},
    ),
    "compliance_violation": create_violation(
        violation_type="compliance",
        severity="high",
        description="Regulatory compliance violation",
        evidence=["GDPR Article 17 violation"],
        metadata={"regulation": "GDPR", "article": "17"},
    ),
    "bias_detected": create_violation(
        violation_type="fairness",
        severity="medium",
        description="Potential bias in decision making",
        evidence=["Demographic disparity in outcomes"],
        metadata={"protected_attribute": "gender"},
    ),
}
