"""
Explainability Validation Test Suite

Tests explainability functionality:
- Explain endpoint coverage
- Latency SLA compliance
- Decision explanation quality

Thresholds:
- Coverage: >95% decisions
- Latency SLA: <500ms
"""

import pytest
import time
from typing import Dict, List
import json
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.models import AgentAction


class ExplainabilityValidator:
    """Validate explainability coverage and quality"""
    
    @staticmethod
    def check_explanation_coverage(actions: List[AgentAction], 
                                   governance: IntegratedGovernance) -> Dict:
        """
        Check what percentage of decisions have explanations
        
        Args:
            actions: List of actions to evaluate
            governance: Governance instance
            
        Returns:
            Coverage statistics
        """
        total_actions = len(actions)
        explained_actions = 0
        latencies = []
        missing_explanations = []
        
        for action in actions:
            start_time = time.perf_counter()
            result = governance.process_action(action_id.split("_")[-2] if "_" in action_id else "test_agent", action_text)
            elapsed = time.perf_counter() - start_time
            
            # Check if result has explanation/reasoning
            has_explanation = bool(
                result.reasoning or 
                getattr(result, 'explanation', None) or
                len(result.violations) > 0  # Violations provide implicit explanation
            )
            
            if has_explanation:
                explained_actions += 1
                latencies.append(elapsed)
            else:
                missing_explanations.append(action.action_id)
        
        coverage = explained_actions / total_actions if total_actions > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        return {
            "total_actions": total_actions,
            "explained_actions": explained_actions,
            "coverage": coverage,
            "average_latency": avg_latency,
            "missing_explanations": missing_explanations,
            "latencies": latencies
        }
    
    @staticmethod
    def validate_explanation_quality(reasoning: str) -> Dict:
        """
        Validate explanation quality
        
        Args:
            reasoning: Explanation text
            
        Returns:
            Quality metrics
        """
        if not reasoning:
            return {
                "has_explanation": False,
                "sufficient_length": False,
                "score": 0.0
            }
        
        # Basic quality checks
        has_content = len(reasoning.strip()) > 0
        sufficient_length = len(reasoning) >= 20  # At least 20 characters
        has_keywords = any(word in reasoning.lower() for word in [
            "violation", "detected", "blocked", "allowed", "risk",
            "safe", "unsafe", "policy", "ethical", "security"
        ])
        
        # Calculate quality score
        score = sum([
            1.0 if has_content else 0.0,
            1.0 if sufficient_length else 0.0,
            1.0 if has_keywords else 0.0
        ]) / 3.0
        
        return {
            "has_explanation": has_content,
            "sufficient_length": sufficient_length,
            "has_keywords": has_keywords,
            "score": score,
            "length": len(reasoning)
        }


@pytest.fixture
def governance():
    """Initialize governance"""
    return IntegratedGovernance()


@pytest.fixture
def validator():
    """Initialize explainability validator"""
    return ExplainabilityValidator()


def test_explanation_coverage(governance, validator):
    """Test that >95% of decisions have explanations"""
    # Create diverse test actions
    test_actions = [
        AgentAction(
            action_id=f"coverage_test_{i}",
            agent_id="explain_tester",
            content=action_text,
            action_type="query"
        )
        for i, action_text in enumerate([
            "Process user data safely",
            "Access restricted database",
            "Generate harmful content",
            "Send notification to user",
            "Delete user records",
            "Create backup copy",
            "Analyze system logs",
            "Modify configuration",
            "Export sensitive data",
            "Update user preferences",
            "Validate input data",
            "Execute privileged command",
            "Read public information",
            "Write to audit log",
            "Query analytics database",
            "Trigger automated workflow",
            "Encrypt user data",
            "Decrypt stored information",
            "Authenticate user session",
            "Authorize API request"
        ])
    ]
    
    result = validator.check_explanation_coverage(test_actions, governance)
    
    print(f"\nExplanation Coverage Test:")
    print(f"  Total Actions: {result['total_actions']}")
    print(f"  Explained Actions: {result['explained_actions']}")
    print(f"  Coverage: {result['coverage']:.2%}")
    print(f"  Average Latency: {result['average_latency'] * 1000:.2f}ms")
    
    assert result["coverage"] > 0.95, f"Coverage {result['coverage']:.2%} below 95% threshold"


def test_explanation_latency_sla(governance, validator):
    """Test that explanations are provided within SLA"""
    # Create test actions
    test_actions = [
        AgentAction(
            action_id=f"latency_test_{i}",
            agent_id="latency_tester",
            action=f"Test action {i} for latency measurement",
            action_type="query"
        )
        for i in range(50)
    ]
    
    result = validator.check_explanation_coverage(test_actions, governance)
    
    # Convert latencies to milliseconds
    latencies_ms = [l * 1000 for l in result["latencies"]]
    avg_latency_ms = result["average_latency"] * 1000
    max_latency_ms = max(latencies_ms) if latencies_ms else 0
    
    # Calculate percentiles
    if latencies_ms:
        latencies_sorted = sorted(latencies_ms)
        p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99_latency = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    else:
        p95_latency = 0
        p99_latency = 0
    
    print(f"\nExplanation Latency SLA Test:")
    print(f"  Average Latency: {avg_latency_ms:.2f}ms")
    print(f"  P95 Latency: {p95_latency:.2f}ms")
    print(f"  P99 Latency: {p99_latency:.2f}ms")
    print(f"  Max Latency: {max_latency_ms:.2f}ms")
    
    assert avg_latency_ms < 500, f"Average latency {avg_latency_ms:.2f}ms exceeds 500ms SLA"
    assert p95_latency < 500, f"P95 latency {p95_latency:.2f}ms exceeds 500ms SLA"


def test_explanation_quality(governance, validator):
    """Test explanation quality"""
    # Create actions that should trigger different types of explanations
    test_cases = [
        ("Generate harmful content about violence", "violation"),
        ("Process normal user request", "allowed"),
        ("Access unauthorized database", "security"),
        ("Perform standard operation", "routine"),
    ]
    
    quality_scores = []
    
    print(f"\nExplanation Quality Test:")
    
    for action_text, expected_type in test_cases:
        # Using string action
        
        result = governance.process_action(action_id.split("_")[-2] if "_" in action_id else "test_agent", action_text)
        
        # Get reasoning
        reasoning = result.reasoning or ""
        if not reasoning and result.violations:
            reasoning = f"Violations detected: {', '.join([v.violation_type for v in result.violations])}"
        
        quality = validator.validate_explanation_quality(reasoning)
        quality_scores.append(quality["score"])
        
        print(f"  {expected_type}: Score={quality['score']:.2f}, Length={quality['length']}")
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    print(f"  Average Quality Score: {avg_quality:.2f}")
    
    # At least 80% should have decent quality explanations
    assert avg_quality >= 0.80, f"Average explanation quality {avg_quality:.2f} too low"


def test_explanation_completeness(governance):
    """Test that explanations are complete and informative"""
    # Test action that should trigger clear explanation
    # Using string action
    
    result = governance.process_action(action_id.split("_")[-2] if "_" in action_id else "test_agent", action_text)
    
    # Check for explanation components
    has_decision = result.decision is not None
    has_reasoning = bool(result.reasoning)
    has_violations = len(result.violations) > 0
    
    # If violations, check they have details
    violation_details_complete = True
    if result.violations:
        for violation in result.violations:
            if not (violation.violation_type and violation.severity):
                violation_details_complete = False
                break
    
    print(f"\nExplanation Completeness Test:")
    print(f"  Has Decision: {has_decision}")
    print(f"  Has Reasoning: {has_reasoning}")
    print(f"  Has Violations: {has_violations}")
    print(f"  Violation Details Complete: {violation_details_complete}")
    print(f"  Decision: {result.decision}")
    print(f"  Reasoning: {result.reasoning[:100] if result.reasoning else 'None'}...")
    
    assert has_decision, "Missing decision"
    assert has_reasoning or has_violations, "Missing explanation (no reasoning or violations)"
    if has_violations:
        assert violation_details_complete, "Violation details incomplete"


def test_generate_explainability_report(governance, validator, tmp_path):
    """Generate comprehensive explainability report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_suite": "explainability",
        "tests": {}
    }
    
    # Test coverage
    test_actions = [
        AgentAction(
            action_id=f"report_test_{i}",
            agent_id="report_agent",
            action=f"Test action {i}",
            action_type="query"
        )
        for i in range(100)
    ]
    
    coverage_result = validator.check_explanation_coverage(test_actions, governance)
    
    latencies_ms = [l * 1000 for l in coverage_result["latencies"]]
    avg_latency_ms = coverage_result["average_latency"] * 1000
    
    if latencies_ms:
        p95_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        p99_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
    else:
        p95_latency = 0
        p99_latency = 0
    
    report["tests"]["coverage"] = {
        "total_actions": coverage_result["total_actions"],
        "explained_actions": coverage_result["explained_actions"],
        "coverage_rate": coverage_result["coverage"],
        "threshold_met": coverage_result["coverage"] > 0.95
    }
    
    report["tests"]["latency"] = {
        "average_ms": avg_latency_ms,
        "p95_ms": p95_latency,
        "p99_ms": p99_latency,
        "sla_met": avg_latency_ms < 500 and p95_latency < 500
    }
    
    # Test quality
    quality_test_actions = [
        ("Harmless action", "routine"),
        ("Suspicious action", "flagged"),
        ("Violating action", "blocked")
    ]
    
    quality_scores = []
    for action_text, _ in quality_test_actions:
        # Using string action
        result = governance.process_action(action_id.split("_")[-2] if "_" in action_id else "test_agent", action_text)
        reasoning = result.reasoning or ""
        quality = validator.validate_explanation_quality(reasoning)
        quality_scores.append(quality["score"])
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    report["tests"]["quality"] = {
        "average_score": avg_quality,
        "threshold_met": avg_quality >= 0.80
    }
    
    # Overall compliance
    report["overall_compliance"] = all(
        test.get("threshold_met", False)
        for test in report["tests"].values()
    )
    
    # Save report
    report_path = tmp_path / "explainability_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nExplainability report saved to: {report_path}")
    print(f"Overall Compliance: {report['overall_compliance']}")
    
    assert report_path.exists()
    assert report["overall_compliance"], "Explainability validation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
