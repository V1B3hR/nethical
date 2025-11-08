"""Ethical Drift Reporter for Phase 3.4: Ethical Drift and Reporting.

This module implements:
- Ethical drift report generation
- Difference in violation types by agent cohort
- Fairness dashboard data aggregation
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from pathlib import Path


@dataclass
class ViolationStats:
    """Statistics for violations."""

    total_count: int = 0
    by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_time: List[Tuple[datetime, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "by_severity": dict(self.by_severity),
            "by_type": dict(self.by_type),
            "by_time": [(ts.isoformat(), count) for ts, count in self.by_time],
        }


@dataclass
class CohortProfile:
    """Profile of a specific agent cohort."""

    cohort_id: str
    agent_count: int = 0
    action_count: int = 0
    violation_stats: ViolationStats = field(default_factory=ViolationStats)
    avg_risk_score: float = 0.0
    risk_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cohort_id": self.cohort_id,
            "agent_count": self.agent_count,
            "action_count": self.action_count,
            "violation_stats": self.violation_stats.to_dict(),
            "avg_risk_score": self.avg_risk_score,
            "risk_distribution": dict(self.risk_distribution),
        }


@dataclass
class EthicalDriftReport:
    """Ethical drift analysis report."""

    report_id: str
    start_time: datetime
    end_time: datetime
    cohorts: Dict[str, CohortProfile]
    drift_metrics: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "cohorts": {cid: c.to_dict() for cid, c in self.cohorts.items()},
            "drift_metrics": self.drift_metrics,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class EthicalDriftReporter:
    """Reporter for ethical drift analysis across agent cohorts."""

    def __init__(
        self,
        report_dir: str = "drift_reports",
        redis_client=None,
        key_prefix: str = "nethical:drift",
    ):
        """Initialize ethical drift reporter.

        Args:
            report_dir: Directory for storing reports
            redis_client: Optional Redis client
            key_prefix: Redis key prefix
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.redis = redis_client
        self.key_prefix = key_prefix

        # Cohort data
        self.cohort_profiles: Dict[str, CohortProfile] = {}

    def track_violation(
        self,
        agent_id: str,
        cohort: str,
        violation_type: str,
        severity: str,
        timestamp: Optional[datetime] = None,
    ):
        """Track a violation for drift analysis.

        Args:
            agent_id: Agent identifier
            cohort: Agent cohort
            violation_type: Type of violation
            severity: Severity level
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if cohort not in self.cohort_profiles:
            self.cohort_profiles[cohort] = CohortProfile(cohort_id=cohort)

        profile = self.cohort_profiles[cohort]
        profile.violation_stats.total_count += 1
        profile.violation_stats.by_type[violation_type] += 1
        profile.violation_stats.by_severity[severity] += 1

    def track_action(self, agent_id: str, cohort: str, risk_score: float):
        """Track an action for cohort profiling.

        Args:
            agent_id: Agent identifier
            cohort: Agent cohort
            risk_score: Risk score
        """
        if cohort not in self.cohort_profiles:
            self.cohort_profiles[cohort] = CohortProfile(cohort_id=cohort)

        profile = self.cohort_profiles[cohort]
        profile.action_count += 1

        # Update average risk score
        if profile.action_count == 1:
            profile.avg_risk_score = risk_score
        else:
            # Running average
            profile.avg_risk_score = (
                profile.avg_risk_score * (profile.action_count - 1) + risk_score
            ) / profile.action_count

        # Risk distribution
        risk_tier = self._risk_score_to_tier(risk_score)
        profile.risk_distribution[risk_tier] += 1

    def _risk_score_to_tier(self, score: float) -> str:
        """Convert risk score to tier."""
        if score >= 0.75:
            return "elevated"
        elif score >= 0.5:
            return "high"
        elif score >= 0.25:
            return "normal"
        else:
            return "low"

    def generate_report(
        self, start_time: datetime, end_time: datetime, cohorts: Optional[List[str]] = None
    ) -> EthicalDriftReport:
        """Generate ethical drift report.

        Args:
            start_time: Report start time
            end_time: Report end time
            cohorts: Optional list of cohorts to include (all if None)

        Returns:
            Ethical drift report
        """
        report_id = f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Select cohorts
        if cohorts is None:
            selected_cohorts = dict(self.cohort_profiles)
        else:
            selected_cohorts = {
                c: self.cohort_profiles[c] for c in cohorts if c in self.cohort_profiles
            }

        # Calculate drift metrics
        drift_metrics = self._calculate_drift_metrics(selected_cohorts)

        # Generate recommendations
        recommendations = self._generate_recommendations(selected_cohorts, drift_metrics)

        report = EthicalDriftReport(
            report_id=report_id,
            start_time=start_time,
            end_time=end_time,
            cohorts=selected_cohorts,
            drift_metrics=drift_metrics,
            recommendations=recommendations,
        )

        # Persist report
        self._save_report(report)

        return report

    def _calculate_drift_metrics(self, cohorts: Dict[str, CohortProfile]) -> Dict[str, Any]:
        """Calculate drift metrics across cohorts.

        Args:
            cohorts: Dictionary of cohort profiles

        Returns:
            Drift metrics
        """
        if len(cohorts) < 2:
            return {"has_drift": False, "message": "Insufficient cohorts for drift analysis"}

        metrics = {
            "cohort_count": len(cohorts),
            "violation_type_drift": {},
            "severity_drift": {},
            "risk_score_drift": {},
            "has_drift": False,
            "drift_score": 0.0,
        }

        # Aggregate violation types across cohorts
        all_violation_types = set()
        for profile in cohorts.values():
            all_violation_types.update(profile.violation_stats.by_type.keys())

        # Calculate violation type drift
        for vtype in all_violation_types:
            cohort_rates = {}
            for cid, profile in cohorts.items():
                if profile.action_count > 0:
                    rate = profile.violation_stats.by_type.get(vtype, 0) / profile.action_count
                    cohort_rates[cid] = rate

            if cohort_rates:
                rates = list(cohort_rates.values())
                max_rate = max(rates)
                min_rate = min(rates)
                drift = max_rate - min_rate

                metrics["violation_type_drift"][vtype] = {
                    "max_rate": max_rate,
                    "min_rate": min_rate,
                    "drift": drift,
                    "cohort_rates": cohort_rates,
                }

                if drift > 0.1:  # 10% threshold
                    metrics["has_drift"] = True

        # Calculate severity drift
        all_severities = set()
        for profile in cohorts.values():
            all_severities.update(profile.violation_stats.by_severity.keys())

        for severity in all_severities:
            cohort_rates = {}
            for cid, profile in cohorts.items():
                if profile.violation_stats.total_count > 0:
                    rate = (
                        profile.violation_stats.by_severity.get(severity, 0)
                        / profile.violation_stats.total_count
                    )
                    cohort_rates[cid] = rate

            if cohort_rates:
                rates = list(cohort_rates.values())
                max_rate = max(rates)
                min_rate = min(rates)
                drift = max_rate - min_rate

                metrics["severity_drift"][severity] = {
                    "max_rate": max_rate,
                    "min_rate": min_rate,
                    "drift": drift,
                }

        # Calculate risk score drift
        risk_scores = [p.avg_risk_score for p in cohorts.values()]
        if risk_scores:
            max_risk = max(risk_scores)
            min_risk = min(risk_scores)
            metrics["risk_score_drift"] = {
                "max_risk": max_risk,
                "min_risk": min_risk,
                "drift": max_risk - min_risk,
                "avg_risk": sum(risk_scores) / len(risk_scores),
            }

            if (max_risk - min_risk) > 0.2:  # 20% threshold
                metrics["has_drift"] = True

        # Overall drift score
        drift_components = []
        for vtype_data in metrics["violation_type_drift"].values():
            drift_components.append(vtype_data["drift"])

        if drift_components:
            metrics["drift_score"] = sum(drift_components) / len(drift_components)

        return metrics

    def _generate_recommendations(
        self, cohorts: Dict[str, CohortProfile], drift_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on drift analysis.

        Args:
            cohorts: Dictionary of cohort profiles
            drift_metrics: Calculated drift metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        if not drift_metrics.get("has_drift", False):
            recommendations.append("No significant ethical drift detected across cohorts")
            return recommendations

        # Violation type recommendations
        violation_drift = drift_metrics.get("violation_type_drift", {})
        for vtype, data in violation_drift.items():
            if data["drift"] > 0.1:
                recommendations.append(
                    f"Significant drift in '{vtype}' violations ({data['drift']:.2%}). "
                    f"Review policies for affected cohorts."
                )

        # Risk score recommendations
        risk_drift = drift_metrics.get("risk_score_drift", {})
        if risk_drift and risk_drift.get("drift", 0) > 0.2:
            recommendations.append(
                f"High risk score variance across cohorts ({risk_drift['drift']:.2%}). "
                f"Consider cohort-specific risk thresholds."
            )

        # Severity recommendations
        severity_drift = drift_metrics.get("severity_drift", {})
        for severity, data in severity_drift.items():
            if data["drift"] > 0.15:
                recommendations.append(
                    f"Uneven distribution of '{severity}' severity violations. "
                    f"Investigate cohort-specific patterns."
                )

        # General recommendations
        if drift_metrics.get("drift_score", 0) > 0.3:
            recommendations.append(
                "High overall drift score detected. "
                "Conduct comprehensive fairness audit across all cohorts."
            )

        return recommendations

    def _save_report(self, report: EthicalDriftReport):
        """Save report to disk and Redis.

        Args:
            report: Ethical drift report
        """
        # Save to disk
        report_file = self.report_dir / f"{report.report_id}.json"
        try:
            with open(report_file, "w") as f:
                f.write(report.to_json())
        except Exception:
            pass  # Silent fail

        # Save to Redis
        if self.redis:
            try:
                key = f"{self.key_prefix}:report:{report.report_id}"
                self.redis.setex(key, 2592000, report.to_json())  # 30 day TTL
            except Exception:
                pass  # Silent fail

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get fairness dashboard data.

        Returns:
            Dashboard data with sampling coverage and exposure metrics
        """
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "cohort_summary": {},
            "overall_stats": {
                "total_cohorts": len(self.cohort_profiles),
                "total_violations": 0,
                "total_actions": 0,
                "avg_risk_score": 0.0,
            },
            "violation_distribution": defaultdict(int),
            "severity_distribution": defaultdict(int),
            "coverage_metrics": {},
        }

        total_risk_sum = 0.0
        total_actions = 0

        for cohort_id, profile in self.cohort_profiles.items():
            # Cohort summary
            dashboard["cohort_summary"][cohort_id] = {
                "agent_count": profile.agent_count,
                "action_count": profile.action_count,
                "violation_count": profile.violation_stats.total_count,
                "avg_risk_score": profile.avg_risk_score,
                "violation_rate": (
                    profile.violation_stats.total_count / profile.action_count
                    if profile.action_count > 0
                    else 0
                ),
            }

            # Aggregate stats
            dashboard["overall_stats"]["total_violations"] += profile.violation_stats.total_count
            dashboard["overall_stats"]["total_actions"] += profile.action_count

            total_risk_sum += profile.avg_risk_score * profile.action_count
            total_actions += profile.action_count

            # Violation distribution
            for vtype, count in profile.violation_stats.by_type.items():
                dashboard["violation_distribution"][vtype] += count

            # Severity distribution
            for severity, count in profile.violation_stats.by_severity.items():
                dashboard["severity_distribution"][severity] += count

        # Calculate overall average risk
        if total_actions > 0:
            dashboard["overall_stats"]["avg_risk_score"] = total_risk_sum / total_actions

        # Coverage metrics
        if self.cohort_profiles:
            action_counts = [p.action_count for p in self.cohort_profiles.values()]
            dashboard["coverage_metrics"] = {
                "max_actions_per_cohort": max(action_counts),
                "min_actions_per_cohort": min(action_counts),
                "avg_actions_per_cohort": sum(action_counts) / len(action_counts),
                "coverage_variance": self._calculate_variance(action_counts),
            }

        return dict(dashboard)

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
