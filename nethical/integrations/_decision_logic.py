"""Shared decision logic for LLM integrations.

This module provides common decision-making logic used by both Claude and REST API integrations.
"""

from typing import Dict, Any, Tuple, List


def compute_decision(
    governance_result: Dict[str, Any],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Compute decision from governance results.

    The IntegratedGovernance.process_action() returns comprehensive phase results
    rather than a direct decision. This function interprets those results to
    determine the appropriate decision.

    Args:
        governance_result: Result from IntegratedGovernance.process_action()

    Returns:
        Tuple of (decision, reason, violations) where:
        - decision: One of "ALLOW", "RESTRICT", "BLOCK", "TERMINATE"
        - reason: Human-readable explanation
        - violations: List of detected violations/correlations
    """
    # Check if decision is already present (future-proofing)
    if "decision" in governance_result:
        decision = governance_result["decision"]
        reason = governance_result.get("reason", f"Decision: {decision}")
        violations = governance_result.get("violations", [])
        return decision, reason, violations

    # Extract indicators from governance results
    risk_score = governance_result.get("phase3", {}).get("risk_score", 0.0)
    pii_detection = governance_result.get("pii_detection", {})
    pii_risk = pii_detection.get("pii_risk_score", 0.0)

    # Get violations from correlations (actual location in governance results)
    correlations = governance_result.get("phase3", {}).get("correlations", [])
    violations = [
        {
            "pattern": c.get("pattern", "unknown"),
            "severity": c.get("severity", "medium"),
            "confidence": c.get("confidence", 0.5),
            "description": c.get("description", ""),
        }
        for c in correlations
    ]

    quarantined = governance_result.get("phase4", {}).get("quarantined", False)
    quota_blocked = False
    if governance_result.get("blocked_by_quota"):
        quota_blocked = True

    # Decision logic based on risk thresholds
    if quota_blocked:
        decision = "BLOCK"
        reason = "Action blocked due to quota limits"
    elif quarantined:
        decision = "TERMINATE"
        reason = "Agent is quarantined due to previous violations"
    elif risk_score >= 0.9 or pii_risk >= 0.9:
        decision = "TERMINATE"
        reason = "Critical risk detected - immediate termination required"
    elif risk_score >= 0.7 or pii_risk >= 0.7:
        decision = "BLOCK"
        reason = "High risk detected - action blocked"
    elif risk_score >= 0.5 or pii_risk >= 0.5 or len(violations) > 0:
        decision = "RESTRICT"
        reason = "Moderate risk or violations detected - restrictions required"
    else:
        decision = "ALLOW"
        reason = "Action evaluated as safe and compliant"

    return decision, reason, violations


def format_violations_for_response(
    violations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Format violations for inclusion in API response.

    Args:
        violations: List of violation dicts from compute_decision

    Returns:
        Formatted violations suitable for API response
    """
    return [
        {
            "type": v.get("pattern", "unknown"),
            "severity": v.get("severity", "medium"),
            "confidence": v.get("confidence", 0.5),
            "description": v.get("description", ""),
        }
        for v in violations
    ]
