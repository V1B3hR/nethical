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
import logging
from typing import Dict, List
import json
from datetime import datetime
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.models import AgentAction
from .test_utils import extract_action_content

# Configure logging for detailed diagnostics
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExplainabilityValidator:
    """Validate explainability coverage and quality"""

    @staticmethod
    def extract_violation_types(violations: List) -> List[str]:
        """
        Extract violation types from violations list that may contain dicts or objects

        Args:
            violations: List of violations (can be dicts or objects)

        Returns:
            List of violation type strings
        """
        violation_types = []
        for v in violations:
            if isinstance(v, dict):
                violation_types.append(v.get("violation_type", "unknown"))
            elif hasattr(v, "violation_type"):
                violation_types.append(v.violation_type)
            else:
                violation_types.append(str(v))
        return violation_types

    @staticmethod
    def check_explanation_coverage(
        actions: List[AgentAction], governance: IntegratedGovernance
    ) -> Dict:
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
            # Extract content using utility function
            action_text = extract_action_content(action)
            result = governance.process_action(
                agent_id="explain_tester", action=action_text
            )
            elapsed = time.perf_counter() - start_time

            # Check if result has explanation/reasoning
            has_explanation = bool(
                result.get("reasoning")
                or result.get("explanation")
                or result.get("violation_detected", False)
            )

            if has_explanation:
                explained_actions += 1
                latencies.append(elapsed)
            else:
                action_id = (
                    action.action_id if hasattr(action, "action_id") else "unknown"
                )
                missing_explanations.append(action_id)

        coverage = explained_actions / total_actions if total_actions > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "total_actions": total_actions,
            "explained_actions": explained_actions,
            "coverage": coverage,
            "average_latency": avg_latency,
            "missing_explanations": missing_explanations,
            "latencies": latencies,
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
            return {"has_explanation": False, "sufficient_length": False, "score": 0.0}

        # Basic quality checks
        has_content = len(reasoning.strip()) > 0
        sufficient_length = len(reasoning) >= 20  # At least 20 characters
        has_keywords = any(
            word in reasoning.lower()
            for word in [
                "violation",
                "detected",
                "blocked",
                "allowed",
                "risk",
                "safe",
                "unsafe",
                "policy",
                "ethical",
                "security",
            ]
        )

        # Calculate quality score
        score = (
            sum(
                [
                    1.0 if has_content else 0.0,
                    1.0 if sufficient_length else 0.0,
                    1.0 if has_keywords else 0.0,
                ]
            )
            / 3.0
        )

        return {
            "has_explanation": has_content,
            "sufficient_length": sufficient_length,
            "has_keywords": has_keywords,
            "score": score,
            "length": len(reasoning),
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
    logger.info("=" * 80)
    logger.info("EXPLAINABILITY TEST - Coverage")
    logger.info("=" * 80)

    # Create diverse test actions
    test_actions = [
        AgentAction(
            action_id=f"coverage_test_{i}",
            agent_id="explain_tester",
            content=action_text,
            action_type="query",
        )
        for i, action_text in enumerate(
            [
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
                "Authorize API request",
            ]
        )
    ]

    logger.info(
        f"Testing explanation coverage with {len(test_actions)} diverse actions"
    )

    result = validator.check_explanation_coverage(test_actions, governance)

    logger.info("-" * 80)
    logger.info("Results:")
    logger.info(f"  Total Actions: {result['total_actions']}")
    logger.info(f"  Explained Actions: {result['explained_actions']}")
    logger.info(f"  Coverage: {result['coverage']:.2%} (threshold: >95%)")
    logger.info(f"  Average Latency: {result['average_latency'] * 1000:.2f}ms")
    logger.info(
        f"  Min Latency: {min(result['latencies']) * 1000:.2f}ms"
        if result["latencies"]
        else "  N/A"
    )
    logger.info(
        f"  Max Latency: {max(result['latencies']) * 1000:.2f}ms"
        if result["latencies"]
        else "  N/A"
    )

    if result["missing_explanations"]:
        logger.warning(
            f"\n{len(result['missing_explanations'])} actions without explanations:"
        )
        for action_id in result["missing_explanations"][:10]:
            logger.warning(f"  - {action_id}")

    print(f"\nExplanation Coverage Test:")
    print(f"  Total Actions: {result['total_actions']}")
    print(f"  Explained Actions: {result['explained_actions']}")
    print(
        f"  Coverage: {result['coverage']:.2%} {'✓' if result['coverage'] > 0.95 else '✗'}"
    )
    print(f"  Average Latency: {result['average_latency'] * 1000:.2f}ms")
    print(f"  Missing Explanations: {len(result['missing_explanations'])}")

    if result["coverage"] <= 0.95:
        logger.error("=" * 80)
        logger.error("COVERAGE THRESHOLD NOT MET")
        logger.error("=" * 80)
        logger.error(f"Coverage {result['coverage']:.2%} below 95% threshold")
        logger.error(
            f"Missing explanations for {len(result['missing_explanations'])} actions"
        )
        logger.error("\nDebugging steps:")
        logger.error("1. Review actions missing explanations (listed above)")
        logger.error("2. Check governance decision logic for explanation generation")
        logger.error("3. Verify all code paths return reasoning or explanation")
        logger.error("4. Ensure violation_detected flag is properly set")
        logger.error("5. Review IntegratedGovernance.process_action() return structure")
        logger.error("\nTo reproduce specific action:")
        if result["missing_explanations"]:
            logger.error(f"  action_id = '{result['missing_explanations'][0]}'")
            logger.error(
                "  result = governance.process_action(agent_id='test', action=action)"
            )
            logger.error("  print(result)")

    assert result["coverage"] > 0.95, (
        f"Coverage {result['coverage']:.2%} below 95% threshold.\n"
        f"  Explained: {result['explained_actions']}/{result['total_actions']}\n"
        f"  Missing explanations: {len(result['missing_explanations'])} actions\n"
        f"  Sample missing: {result['missing_explanations'][:3]}\n"
        f"  Review logs above for detailed analysis"
    )


def test_explanation_latency_sla(governance, validator):
    """Test that explanations are provided within SLA"""
    logger.info("=" * 80)
    logger.info("EXPLAINABILITY TEST - Latency SLA")
    logger.info("=" * 80)

    # Create test actions
    test_actions = [
        AgentAction(
            action_id=f"latency_test_{i}",
            agent_id="latency_tester",
            content=f"Test action {i} for latency measurement",
            action_type="query",
        )
        for i in range(50)
    ]

    logger.info(f"Testing explanation latency with {len(test_actions)} actions")

    result = validator.check_explanation_coverage(test_actions, governance)

    # Convert latencies to milliseconds
    latencies_ms = [l * 1000 for l in result["latencies"]]
    avg_latency_ms = result["average_latency"] * 1000
    max_latency_ms = max(latencies_ms) if latencies_ms else 0
    min_latency_ms = min(latencies_ms) if latencies_ms else 0

    # Calculate percentiles
    if latencies_ms:
        latencies_sorted = sorted(latencies_ms)
        p50_latency = latencies_sorted[int(len(latencies_sorted) * 0.50)]
        p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99_latency = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    else:
        p50_latency = p95_latency = p99_latency = 0

    logger.info("-" * 80)
    logger.info("Results:")
    logger.info(f"  Samples: {len(latencies_ms)}")
    logger.info(f"  Min Latency: {min_latency_ms:.2f}ms")
    logger.info(f"  P50 Latency: {p50_latency:.2f}ms")
    logger.info(f"  Average Latency: {avg_latency_ms:.2f}ms (SLO: <500ms)")
    logger.info(f"  P95 Latency: {p95_latency:.2f}ms (SLO: <500ms)")
    logger.info(f"  P99 Latency: {p99_latency:.2f}ms")
    logger.info(f"  Max Latency: {max_latency_ms:.2f}ms")

    # Identify slow requests
    slow_requests = [l for l in latencies_ms if l > 500]
    if slow_requests:
        logger.warning(f"\n{len(slow_requests)} requests exceeded 500ms SLA:")
        logger.warning(f"  Slowest: {max(slow_requests):.2f}ms")
        logger.warning(
            f"  Average of slow: {sum(slow_requests)/len(slow_requests):.2f}ms"
        )

    print(f"\nExplanation Latency SLA Test:")
    print(
        f"  Average Latency: {avg_latency_ms:.2f}ms {'✓' if avg_latency_ms < 500 else '✗'}"
    )
    print(f"  P95 Latency: {p95_latency:.2f}ms {'✓' if p95_latency < 500 else '✗'}")
    print(f"  P99 Latency: {p99_latency:.2f}ms {'✓' if p99_latency < 500 else '✗'}")
    print(f"  Max Latency: {max_latency_ms:.2f}ms")
    print(f"  Slow Requests (>500ms): {len(slow_requests)}")

    if avg_latency_ms >= 500 or p95_latency >= 500:
        logger.error("=" * 80)
        logger.error("LATENCY SLA VIOLATION")
        logger.error("=" * 80)
        logger.error(f"Explanation latency exceeds 500ms SLA")
        logger.error(f"Average: {avg_latency_ms:.2f}ms, P95: {p95_latency:.2f}ms")
        logger.error("\nDebugging steps:")
        logger.error("1. Profile explanation generation logic")
        logger.error("2. Check for expensive operations (DB queries, API calls)")
        logger.error("3. Review reasoning/explanation construction")
        logger.error("4. Consider caching frequently used explanations")
        logger.error("5. Optimize violation detection and formatting")
        logger.error(f"6. {len(slow_requests)} requests took >500ms")
        logger.error("\nTo reproduce:")
        logger.error(
            "  pytest tests/validation/test_explainability.py::test_explanation_latency_sla -v -s"
        )

    assert avg_latency_ms < 500, (
        f"Average latency {avg_latency_ms:.2f}ms exceeds 500ms SLA.\n"
        f"  P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms\n"
        f"  Slow requests: {len(slow_requests)}/{len(latencies_ms)}\n"
        f"  Review logs above for latency analysis"
    )
    assert p95_latency < 500, (
        f"P95 latency {p95_latency:.2f}ms exceeds 500ms SLA.\n"
        f"  This means 5% of requests are too slow.\n"
        f"  Max: {max_latency_ms:.2f}ms\n"
        f"  Consider optimizing slow code paths"
    )


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
        result = governance.process_action(
            agent_id="quality_tester", action=action_text
        )

        # Get reasoning
        reasoning = result.get("reasoning", "")
        violations = result.get("violations", [])
        if not reasoning and violations:
            violation_types = validator.extract_violation_types(violations)
            reasoning = f"Violations detected: {', '.join(violation_types)}"

        quality = validator.validate_explanation_quality(reasoning)
        quality_scores.append(quality["score"])

        print(
            f"  {expected_type}: Score={quality['score']:.2f}, Length={quality['length']}"
        )

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    print(f"  Average Quality Score: {avg_quality:.2f}")

    # At least 80% should have decent quality explanations
    assert avg_quality >= 0.80, f"Average explanation quality {avg_quality:.2f} too low"


def test_explanation_completeness(governance):
    """Test that explanations are complete and informative"""
    # Test action that should trigger clear explanation
    action_text = "Access user database without authorization"

    result = governance.process_action(
        agent_id="completeness_tester", action=action_text
    )

    # Check for explanation components
    has_decision = result.get("decision") is not None
    has_reasoning = bool(result.get("reasoning"))
    violations = result.get("violations", [])
    has_violations = len(violations) > 0

    # If violations, check they have details
    violation_details_complete = True
    if violations:
        for violation in violations:
            if isinstance(violation, dict):
                if not (violation.get("violation_type") and violation.get("severity")):
                    violation_details_complete = False
                    break
            else:
                # If not dict, assume it's an object with attributes
                if not (
                    hasattr(violation, "violation_type")
                    and hasattr(violation, "severity")
                ):
                    violation_details_complete = False
                    break

    print(f"\nExplanation Completeness Test:")
    print(f"  Has Decision: {has_decision}")
    print(f"  Has Reasoning: {has_reasoning}")
    print(f"  Has Violations: {has_violations}")
    print(f"  Violation Details Complete: {violation_details_complete}")
    print(f"  Decision: {result.get('decision')}")
    reasoning_text = result.get("reasoning", "")
    print(f"  Reasoning: {reasoning_text[:100] if reasoning_text else 'None'}...")

    assert has_decision, "Missing decision"
    assert (
        has_reasoning or has_violations
    ), "Missing explanation (no reasoning or violations)"
    if has_violations:
        assert violation_details_complete, "Violation details incomplete"


def test_generate_explainability_report(governance, validator, tmp_path):
    """Generate comprehensive explainability report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_suite": "explainability",
        "tests": {},
    }

    # Test coverage
    test_actions = [
        AgentAction(
            action_id=f"report_test_{i}",
            agent_id="report_agent",
            content=f"Test action {i}",
            action_type="query",
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
        "threshold_met": coverage_result["coverage"] > 0.95,
    }

    latency_threshold_met = avg_latency_ms < 500 and p95_latency < 500
    report["tests"]["latency"] = {
        "average_ms": avg_latency_ms,
        "p95_ms": p95_latency,
        "p99_ms": p99_latency,
        "sla_met": latency_threshold_met,  # Domain-specific field for latency tests
        "threshold_met": latency_threshold_met,  # Unified field for overall compliance checking
    }

    # Test quality
    quality_test_actions = [
        ("Harmless action", "routine"),
        ("Suspicious action", "flagged"),
        ("Violating action", "blocked"),
    ]

    quality_scores = []
    for action_text, _ in quality_test_actions:
        result = governance.process_action(agent_id="report_agent", action=action_text)
        reasoning = result.get("reasoning", "")
        quality = validator.validate_explanation_quality(reasoning)
        quality_scores.append(quality["score"])

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    report["tests"]["quality"] = {
        "average_score": avg_quality,
        "threshold_met": avg_quality >= 0.80,
    }

    # Overall compliance - check threshold_met for all tests
    report["overall_compliance"] = all(
        test.get("threshold_met", False) for test in report["tests"].values()
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
