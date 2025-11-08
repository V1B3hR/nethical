"""Phase 6: ML Assisted Enforcement Implementation.

This module implements:
- Risk blending: e.g. 0.7 * rules + 0.3 * ml
- Gray zone detection (mid-band risk: 0.4 ≤ rule_score ≤ 0.6)
- Blended risk computed only in uncertain range
- Pre/post decision audit trail
- FP delta tracking and gating
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json


class RiskZone(str, Enum):
    """Risk zones for blended enforcement."""

    CLEAR_ALLOW = "clear_allow"  # Low risk, no ML needed
    GRAY_ZONE = "gray_zone"  # Mid-range, ML assists
    CLEAR_DENY = "clear_deny"  # High risk, no ML needed


@dataclass
class BlendedDecision:
    """Blended risk decision with audit trail."""

    decision_id: str
    timestamp: datetime
    agent_id: str
    action_id: str

    # Original rule-based decision
    rule_risk_score: float
    rule_classification: str

    # ML prediction
    ml_risk_score: Optional[float] = None
    ml_confidence: Optional[float] = None

    # Blended outcome
    blended_risk_score: float = 0.0
    blended_classification: str = ""
    risk_zone: RiskZone = RiskZone.CLEAR_ALLOW
    ml_influenced: bool = False

    # Blend weights used
    rule_weight: float = 1.0
    ml_weight: float = 0.0

    # Decision change tracking
    classification_changed: bool = False
    original_classification: str = ""
    final_classification: str = ""

    # Audit metadata
    explanation: str = ""
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "action_id": self.action_id,
            "rule_risk_score": self.rule_risk_score,
            "rule_classification": self.rule_classification,
            "ml_risk_score": self.ml_risk_score,
            "ml_confidence": self.ml_confidence,
            "blended_risk_score": self.blended_risk_score,
            "blended_classification": self.blended_classification,
            "risk_zone": self.risk_zone.value,
            "ml_influenced": self.ml_influenced,
            "rule_weight": self.rule_weight,
            "ml_weight": self.ml_weight,
            "classification_changed": self.classification_changed,
            "original_classification": self.original_classification,
            "final_classification": self.final_classification,
            "explanation": self.explanation,
            "features": self.features,
        }


@dataclass
class BlendingMetrics:
    """Metrics for blended enforcement evaluation."""

    total_decisions: int = 0

    # Zone distribution
    clear_allow_count: int = 0
    gray_zone_count: int = 0
    clear_deny_count: int = 0

    # ML influence tracking
    ml_influenced_count: int = 0
    classification_changes: int = 0

    # Change direction
    escalations: int = 0  # ML made it stricter (allow->warn, warn->deny)
    de_escalations: int = 0  # ML made it more lenient

    # FP delta tracking (for gating)
    baseline_false_positives: int = 0
    blended_false_positives: int = 0

    # Detection improvements
    baseline_true_positives: int = 0
    blended_true_positives: int = 0

    @property
    def gray_zone_percentage(self) -> float:
        """Percentage of decisions in gray zone."""
        if self.total_decisions == 0:
            return 0.0
        return (self.gray_zone_count / self.total_decisions) * 100

    @property
    def ml_influence_rate(self) -> float:
        """Rate of ML influence on decisions."""
        if self.gray_zone_count == 0:
            return 0.0
        return (self.ml_influenced_count / self.gray_zone_count) * 100

    @property
    def classification_change_rate(self) -> float:
        """Rate of classification changes."""
        if self.ml_influenced_count == 0:
            return 0.0
        return (self.classification_changes / self.ml_influenced_count) * 100

    @property
    def fp_delta(self) -> float:
        """False positive delta (blended - baseline)."""
        return self.blended_false_positives - self.baseline_false_positives

    @property
    def fp_delta_percentage(self) -> float:
        """False positive delta as percentage."""
        if self.baseline_false_positives == 0:
            return 0.0
        return (self.fp_delta / self.baseline_false_positives) * 100

    @property
    def detection_improvement(self) -> int:
        """Improvement in true positives."""
        return self.blended_true_positives - self.baseline_true_positives

    @property
    def detection_improvement_rate(self) -> float:
        """Detection improvement as percentage."""
        if self.baseline_true_positives == 0:
            return 0.0
        return (self.detection_improvement / self.baseline_true_positives) * 100

    def gate_check(self, max_fp_delta_pct: float = 5.0) -> Tuple[bool, str]:
        """Check if blended mode meets promotion gate criteria.

        Args:
            max_fp_delta_pct: Maximum allowed FP delta percentage (default: 5%)

        Returns:
            Tuple of (passes_gate, reason)
        """
        # Check FP delta
        if abs(self.fp_delta_percentage) > max_fp_delta_pct:
            return (
                False,
                f"FP delta {self.fp_delta_percentage:.1f}% exceeds limit {max_fp_delta_pct}%",
            )

        # Check for detection improvement
        if self.detection_improvement <= 0:
            return False, "No improvement in detection rate"

        # Check minimum sample size
        if self.gray_zone_count < 100:
            return False, f"Insufficient gray zone samples ({self.gray_zone_count} < 100)"

        return True, "All gate checks passed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        passes_gate, gate_reason = self.gate_check()

        return {
            "total_decisions": self.total_decisions,
            "zone_distribution": {
                "clear_allow": self.clear_allow_count,
                "gray_zone": self.gray_zone_count,
                "clear_deny": self.clear_deny_count,
                "gray_zone_percentage": self.gray_zone_percentage,
            },
            "ml_influence": {
                "influenced_count": self.ml_influenced_count,
                "influence_rate": self.ml_influence_rate,
                "classification_changes": self.classification_changes,
                "change_rate": self.classification_change_rate,
                "escalations": self.escalations,
                "de_escalations": self.de_escalations,
            },
            "gate_metrics": {
                "fp_delta": self.fp_delta,
                "fp_delta_percentage": self.fp_delta_percentage,
                "detection_improvement": self.detection_improvement,
                "detection_improvement_rate": self.detection_improvement_rate,
                "passes_gate": passes_gate,
                "gate_reason": gate_reason,
            },
        }


class MLBlendedRiskEngine:
    """ML-assisted enforcement with blended risk scoring."""

    def __init__(
        self,
        gray_zone_lower: float = 0.4,
        gray_zone_upper: float = 0.6,
        rule_weight: float = 0.7,
        ml_weight: float = 0.3,
        storage_path: Optional[str] = None,
        enable_ml_blending: bool = True,
    ):
        """Initialize blended risk engine.

        Args:
            gray_zone_lower: Lower bound of gray zone (default: 0.4)
            gray_zone_upper: Upper bound of gray zone (default: 0.6)
            rule_weight: Weight for rule-based risk (default: 0.7)
            ml_weight: Weight for ML risk (default: 0.3)
            storage_path: Path for storing decisions
            enable_ml_blending: Enable ML blending (can be toggled for A/B testing)
        """
        self.gray_zone_lower = gray_zone_lower
        self.gray_zone_upper = gray_zone_upper
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        self.storage_path = storage_path
        self.enable_ml_blending = enable_ml_blending

        # Validate weights
        if abs((rule_weight + ml_weight) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {rule_weight + ml_weight}")

        # Decision log
        self.decisions: List[BlendedDecision] = []

        # Metrics
        self.metrics = BlendingMetrics()

    def compute_blended_risk(
        self,
        agent_id: str,
        action_id: str,
        rule_risk_score: float,
        rule_classification: str,
        ml_risk_score: Optional[float] = None,
        ml_confidence: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> BlendedDecision:
        """Compute blended risk decision.

        Args:
            agent_id: Agent identifier
            action_id: Action identifier
            rule_risk_score: Rule-based risk score (0-1)
            rule_classification: Rule-based classification
            ml_risk_score: Optional ML risk score (0-1)
            ml_confidence: Optional ML confidence (0-1)
            features: Optional feature dictionary

        Returns:
            Blended decision with audit trail
        """
        # Determine risk zone
        risk_zone = self._determine_risk_zone(rule_risk_score)

        # Initialize decision
        decision = BlendedDecision(
            decision_id=f"blend_{int(datetime.utcnow().timestamp() * 1000000)}",
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            action_id=action_id,
            rule_risk_score=rule_risk_score,
            rule_classification=rule_classification,
            ml_risk_score=ml_risk_score,
            ml_confidence=ml_confidence,
            risk_zone=risk_zone,
            original_classification=rule_classification,
            features=features or {},
        )

        # Apply blending logic based on zone
        if (
            risk_zone == RiskZone.GRAY_ZONE
            and self.enable_ml_blending
            and ml_risk_score is not None
        ):
            # Blend in gray zone
            decision.blended_risk_score = (
                self.rule_weight * rule_risk_score + self.ml_weight * ml_risk_score
            )
            decision.ml_influenced = True
            decision.rule_weight = self.rule_weight
            decision.ml_weight = self.ml_weight
            decision.explanation = (
                f"Gray zone: blended {self.rule_weight}*rule + {self.ml_weight}*ML"
            )
        else:
            # Use rule-based score outside gray zone or if ML not available
            decision.blended_risk_score = rule_risk_score
            decision.ml_influenced = False
            decision.rule_weight = 1.0
            decision.ml_weight = 0.0

            if risk_zone == RiskZone.CLEAR_ALLOW:
                decision.explanation = "Clear allow zone: rules only"
            elif risk_zone == RiskZone.CLEAR_DENY:
                decision.explanation = "Clear deny zone: rules only"
            else:
                decision.explanation = "Gray zone: ML not available, using rules only"

        # Determine final classification
        decision.blended_classification = self._score_to_classification(decision.blended_risk_score)
        decision.final_classification = decision.blended_classification

        # Check if classification changed
        decision.classification_changed = decision.blended_classification != rule_classification

        # Log decision
        self.decisions.append(decision)

        # Update metrics
        self._update_metrics(decision)

        # Persist if configured
        if self.storage_path:
            self._persist_decision(decision)

        return decision

    def _determine_risk_zone(self, rule_risk_score: float) -> RiskZone:
        """Determine which risk zone a score falls into."""
        if rule_risk_score < self.gray_zone_lower:
            return RiskZone.CLEAR_ALLOW
        elif rule_risk_score <= self.gray_zone_upper:
            return RiskZone.GRAY_ZONE
        else:
            return RiskZone.CLEAR_DENY

    def _score_to_classification(self, score: float) -> str:
        """Convert risk score to classification."""
        if score >= 0.7:
            return "deny"
        elif score >= 0.4:
            return "warn"
        else:
            return "allow"

    def _update_metrics(self, decision: BlendedDecision) -> None:
        """Update blending metrics."""
        self.metrics.total_decisions += 1

        # Update zone distribution
        if decision.risk_zone == RiskZone.CLEAR_ALLOW:
            self.metrics.clear_allow_count += 1
        elif decision.risk_zone == RiskZone.GRAY_ZONE:
            self.metrics.gray_zone_count += 1
        else:  # CLEAR_DENY
            self.metrics.clear_deny_count += 1

        # Update ML influence
        if decision.ml_influenced:
            self.metrics.ml_influenced_count += 1

        # Update classification changes
        if decision.classification_changed:
            self.metrics.classification_changes += 1

            # Track escalation/de-escalation
            severity_order = {"allow": 0, "warn": 1, "deny": 2}
            original_severity = severity_order.get(decision.original_classification, 0)
            final_severity = severity_order.get(decision.final_classification, 0)

            if final_severity > original_severity:
                self.metrics.escalations += 1
            elif final_severity < original_severity:
                self.metrics.de_escalations += 1

    def _persist_decision(self, decision: BlendedDecision) -> None:
        """Persist decision to storage."""
        if not self.storage_path:
            return

        try:
            import os

            os.makedirs(self.storage_path, exist_ok=True)

            # Append to decisions log
            log_file = os.path.join(self.storage_path, "blended_decisions.jsonl")
            with open(log_file, "a") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")
        except Exception:
            pass  # Silent fail for logging

    def record_ground_truth(
        self, decision_id: str, is_true_positive: bool, is_false_positive: bool
    ) -> None:
        """Record ground truth for gate check evaluation.

        Args:
            decision_id: Decision identifier
            is_true_positive: Whether this was a true positive detection
            is_false_positive: Whether this was a false positive
        """
        # Find the decision
        decision = None
        for d in self.decisions:
            if d.decision_id == decision_id:
                decision = d
                break

        if not decision:
            return

        # Update metrics based on whether ML influenced the decision
        if decision.ml_influenced:
            if is_true_positive:
                self.metrics.blended_true_positives += 1
            if is_false_positive:
                self.metrics.blended_false_positives += 1
        else:
            # Baseline (rule-only)
            if is_true_positive:
                self.metrics.baseline_true_positives += 1
            if is_false_positive:
                self.metrics.baseline_false_positives += 1

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report.

        Returns:
            Dictionary with all metrics and gate check results
        """
        report = self.metrics.to_dict()

        # Add configuration
        report["configuration"] = {
            "gray_zone_lower": self.gray_zone_lower,
            "gray_zone_upper": self.gray_zone_upper,
            "rule_weight": self.rule_weight,
            "ml_weight": self.ml_weight,
            "ml_blending_enabled": self.enable_ml_blending,
        }

        # Add decision log stats
        report["decision_log"] = {
            "total_logged": len(self.decisions),
            "recent_ml_influence_rate": self._get_recent_ml_influence_rate(),
        }

        return report

    def _get_recent_ml_influence_rate(self) -> float:
        """Calculate ML influence rate for recent decisions."""
        if len(self.decisions) == 0:
            return 0.0

        recent_decisions = self.decisions[-100:]
        gray_zone_decisions = [d for d in recent_decisions if d.risk_zone == RiskZone.GRAY_ZONE]

        if len(gray_zone_decisions) == 0:
            return 0.0

        influenced = sum(1 for d in gray_zone_decisions if d.ml_influenced)
        return (influenced / len(gray_zone_decisions)) * 100

    def export_decisions(
        self,
        risk_zone: Optional[RiskZone] = None,
        ml_influenced_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Export decisions for analysis.

        Args:
            risk_zone: Optional filter by risk zone
            ml_influenced_only: Only export ML-influenced decisions
            limit: Optional limit on number of decisions

        Returns:
            List of decision dictionaries
        """
        decisions_to_export = self.decisions

        # Apply filters
        if risk_zone:
            decisions_to_export = [d for d in decisions_to_export if d.risk_zone == risk_zone]

        if ml_influenced_only:
            decisions_to_export = [d for d in decisions_to_export if d.ml_influenced]

        # Apply limit
        if limit:
            decisions_to_export = decisions_to_export[-limit:]

        return [d.to_dict() for d in decisions_to_export]

    def reset_metrics(self) -> None:
        """Reset metrics (useful for evaluation periods)."""
        self.metrics = BlendingMetrics()
        self.decisions.clear()
