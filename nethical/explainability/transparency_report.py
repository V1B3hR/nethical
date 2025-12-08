"""
Transparency Report Generator - Creates comprehensive transparency reports.

This module generates detailed transparency reports for auditing, compliance,
and public disclosure purposes. Reports include statistics, trends, and
explanations of governance decisions over time.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class TransparencyReport:
    """A comprehensive transparency report."""

    report_id: str
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any]
    decision_breakdown: Dict[str, int]
    violation_trends: Dict[str, List[int]]
    policy_effectiveness: Dict[str, float]
    key_insights: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransparencyReportGenerator:
    """
    Generates transparency reports for AI governance decisions.

    This class analyzes historical governance data and generates
    comprehensive reports that provide transparency into how the
    system makes decisions, what violations were detected, and
    how effective policies are.
    """

    def __init__(self, include_sensitive_data: bool = False):
        """
        Initialize the transparency report generator.

        Args:
            include_sensitive_data: Whether to include sensitive details
                                   (for internal reports only)
        """
        self.include_sensitive_data = include_sensitive_data

    def generate_report(
        self,
        decisions: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
        period_days: int = 30,
    ) -> TransparencyReport:
        """
        Generate a comprehensive transparency report.

        Args:
            decisions: List of governance decisions
            violations: List of detected violations
            policies: List of active policies
            period_days: Number of days to include in report

        Returns:
            A TransparencyReport with complete analysis
        """
        from datetime import timezone

        now = datetime.now(timezone.utc)
        period_start = now - timedelta(days=period_days)

        # Filter data to period
        period_decisions = [
            d
            for d in decisions
            if self._parse_timestamp(d.get("timestamp")) >= period_start
        ]
        period_violations = [
            v
            for v in violations
            if self._parse_timestamp(v.get("timestamp")) >= period_start
        ]

        # Generate report components
        summary = self._generate_summary(period_decisions, period_violations)
        decision_breakdown = self._analyze_decision_breakdown(period_decisions)
        violation_trends = self._analyze_violation_trends(
            period_violations, period_days
        )
        policy_effectiveness = self._analyze_policy_effectiveness(
            period_decisions, period_violations, policies
        )
        key_insights = self._extract_key_insights(
            summary, decision_breakdown, violation_trends
        )
        recommendations = self._generate_recommendations(
            summary, decision_breakdown, policy_effectiveness
        )

        report_id = f"TR-{now.strftime('%Y%m%d-%H%M%S')}"

        return TransparencyReport(
            report_id=report_id,
            period_start=period_start,
            period_end=now,
            summary=summary,
            decision_breakdown=decision_breakdown,
            violation_trends=violation_trends,
            policy_effectiveness=policy_effectiveness,
            key_insights=key_insights,
            recommendations=recommendations,
            metadata={
                "generated_at": now.isoformat(),
                "period_days": period_days,
                "total_decisions": len(period_decisions),
                "total_violations": len(period_violations),
            },
        )

    def _generate_summary(
        self, decisions: List[Dict[str, Any]], violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate high-level summary statistics."""
        total_decisions = len(decisions)
        total_violations = len(violations)

        if total_decisions == 0:
            return {
                "total_decisions": 0,
                "total_violations": 0,
                "block_rate": 0.0,
                "violation_rate": 0.0,
            }

        # Count decision types
        decision_types = defaultdict(int)
        for decision in decisions:
            dtype = decision.get("decision", "UNKNOWN")
            decision_types[dtype] += 1

        # Count violation severities
        violation_severities = defaultdict(int)
        for violation in violations:
            severity = violation.get("severity", "medium")
            violation_severities[severity] += 1

        block_rate = decision_types.get("BLOCK", 0) / total_decisions * 100
        violation_rate = total_violations / total_decisions * 100

        return {
            "total_decisions": total_decisions,
            "total_violations": total_violations,
            "decision_types": dict(decision_types),
            "violation_severities": dict(violation_severities),
            "block_rate": block_rate,
            "violation_rate": violation_rate,
            "allow_rate": decision_types.get("ALLOW", 0) / total_decisions * 100,
            "restrict_rate": decision_types.get("RESTRICT", 0) / total_decisions * 100,
        }

    def _analyze_decision_breakdown(
        self, decisions: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze breakdown of decisions by various categories."""
        breakdown = {
            "by_decision": defaultdict(int),
            "by_category": defaultdict(int),
            "by_agent": defaultdict(int),
            "by_hour": defaultdict(int),
        }

        for decision in decisions:
            # By decision type
            dtype = decision.get("decision", "UNKNOWN")
            breakdown["by_decision"][dtype] += 1

            # By category
            category = decision.get("category", "uncategorized")
            breakdown["by_category"][category] += 1

            # By agent (anonymize if needed)
            agent_id = decision.get("agent_id", "unknown")
            if not self.include_sensitive_data:
                agent_id = f"agent_{hash(agent_id) % 1000}"
            breakdown["by_agent"][agent_id] += 1

            # By hour of day
            timestamp = self._parse_timestamp(decision.get("timestamp"))
            hour = timestamp.hour if timestamp else 0
            breakdown["by_hour"][hour] += 1

        # Convert to regular dicts
        return {key: dict(value) for key, value in breakdown.items()}

    def _analyze_violation_trends(
        self, violations: List[Dict[str, Any]], period_days: int
    ) -> Dict[str, List[int]]:
        """Analyze violation trends over time."""
        # Initialize daily counts
        daily_counts = {
            "total": [0] * period_days,
            "by_type": defaultdict(lambda: [0] * period_days),
            "by_severity": defaultdict(lambda: [0] * period_days),
        }

        now = datetime.now()
        period_start = now - timedelta(days=period_days)

        for violation in violations:
            timestamp = self._parse_timestamp(violation.get("timestamp"))
            if not timestamp or timestamp < period_start:
                continue

            # Calculate day index
            day_index = (timestamp - period_start).days
            if day_index < 0 or day_index >= period_days:
                continue

            # Increment counts
            daily_counts["total"][day_index] += 1

            vtype = violation.get("type", "unknown")
            daily_counts["by_type"][vtype][day_index] += 1

            severity = violation.get("severity", "medium")
            daily_counts["by_severity"][severity][day_index] += 1

        # Convert nested defaultdicts to regular dicts
        result = {"total": daily_counts["total"]}
        result["by_type"] = dict(daily_counts["by_type"])
        result["by_severity"] = dict(daily_counts["by_severity"])

        return result

    def _analyze_policy_effectiveness(
        self,
        decisions: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Analyze effectiveness of each policy."""
        effectiveness = {}

        for policy in policies:
            policy_name = policy.get("name", "unnamed")

            # Count how many times this policy triggered
            triggers = sum(
                1 for d in decisions if policy_name in d.get("matched_policies", [])
            )

            # Count how many violations it prevented
            prevented = sum(
                1
                for d in decisions
                if policy_name in d.get("matched_policies", [])
                and d.get("decision") in ["BLOCK", "RESTRICT"]
            )

            # Calculate effectiveness score
            if triggers > 0:
                effectiveness[policy_name] = prevented / triggers
            else:
                effectiveness[policy_name] = 0.0

        return effectiveness

    def _extract_key_insights(
        self,
        summary: Dict[str, Any],
        decision_breakdown: Dict[str, Any],
        violation_trends: Dict[str, List[int]],
    ) -> List[str]:
        """Extract key insights from the data."""
        insights = []

        # Block rate insight
        block_rate = summary.get("block_rate", 0)
        if block_rate > 20:
            insights.append(
                f"High block rate ({block_rate:.1f}%) may indicate overly strict policies "
                "or increased threat activity"
            )
        elif block_rate < 1:
            insights.append(
                f"Very low block rate ({block_rate:.1f}%) suggests either effective "
                "prevention or potentially under-detection"
            )

        # Violation trends
        total_violations = violation_trends.get("total", [])
        if len(total_violations) >= 7:
            recent_week = sum(total_violations[-7:])
            previous_week = (
                sum(total_violations[-14:-7])
                if len(total_violations) >= 14
                else recent_week
            )

            if previous_week > 0:
                change = ((recent_week - previous_week) / previous_week) * 100
                if change > 20:
                    insights.append(
                        f"Violations increased by {change:.1f}% in the last week, "
                        "indicating increased threat activity or policy changes"
                    )
                elif change < -20:
                    insights.append(
                        f"Violations decreased by {abs(change):.1f}% in the last week, "
                        "suggesting improved compliance or threat reduction"
                    )

        # Decision distribution
        decision_types = summary.get("decision_types", {})
        if decision_types:
            most_common = max(decision_types.items(), key=lambda x: x[1])
            insights.append(
                f"Most common decision: {most_common[0]} "
                f"({most_common[1] / summary['total_decisions'] * 100:.1f}% of all decisions)"
            )

        # Time patterns
        by_hour = decision_breakdown.get("by_hour", {})
        if by_hour:
            peak_hour = max(by_hour.items(), key=lambda x: x[1])
            insights.append(
                f"Peak activity occurs at {peak_hour[0]:02d}:00 with "
                f"{peak_hour[1]} decisions"
            )

        return insights

    def _generate_recommendations(
        self,
        summary: Dict[str, Any],
        decision_breakdown: Dict[str, Any],
        policy_effectiveness: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Policy effectiveness recommendations
        ineffective_policies = [
            name for name, score in policy_effectiveness.items() if score < 0.3
        ]
        if ineffective_policies:
            recommendations.append(
                f"Review {len(ineffective_policies)} low-effectiveness policies: "
                f"{', '.join(ineffective_policies[:3])}"
            )

        # Block rate recommendations
        block_rate = summary.get("block_rate", 0)
        if block_rate > 30:
            recommendations.append(
                "Consider reviewing policies - high block rate may impact user experience"
            )
        elif block_rate < 0.5:
            recommendations.append(
                "Low block rate detected - verify detection systems are functioning properly"
            )

        # Violation severity recommendations
        violation_severities = summary.get("violation_severities", {})
        critical_violations = violation_severities.get("critical", 0)
        if critical_violations > 10:
            recommendations.append(
                f"Address {critical_violations} critical violations immediately"
            )

        # General recommendations
        recommendations.append(
            "Continue monitoring trends and adjust policies as needed"
        )
        recommendations.append("Review this report with stakeholders monthly")

        return recommendations

    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        from datetime import timezone

        if isinstance(timestamp, datetime):
            # Ensure timezone-aware
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp
        elif isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                # Ensure timezone-aware
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, AttributeError):
                return None
        elif isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (ValueError, OSError):
                return None
        return None

    def to_json(self, report: TransparencyReport) -> str:
        """Convert report to JSON format."""
        data = {
            "report_id": report.report_id,
            "period": {
                "start": report.period_start.isoformat(),
                "end": report.period_end.isoformat(),
            },
            "summary": report.summary,
            "decision_breakdown": report.decision_breakdown,
            "violation_trends": report.violation_trends,
            "policy_effectiveness": report.policy_effectiveness,
            "key_insights": report.key_insights,
            "recommendations": report.recommendations,
            "metadata": report.metadata,
        }
        return json.dumps(data, indent=2)

    def to_markdown(self, report: TransparencyReport) -> str:
        """Convert report to markdown format."""
        md = f"# Transparency Report: {report.report_id}\n\n"
        md += f"**Period:** {report.period_start.date()} to {report.period_end.date()}\n\n"

        # Summary
        md += "## Summary\n\n"
        summary = report.summary
        md += f"- **Total Decisions:** {summary.get('total_decisions', 0):,}\n"
        md += f"- **Total Violations:** {summary.get('total_violations', 0):,}\n"
        md += f"- **Block Rate:** {summary.get('block_rate', 0):.1f}%\n"
        md += f"- **Violation Rate:** {summary.get('violation_rate', 0):.1f}%\n\n"

        # Key Insights
        md += "## Key Insights\n\n"
        for insight in report.key_insights:
            md += f"- {insight}\n"
        md += "\n"

        # Recommendations
        md += "## Recommendations\n\n"
        for rec in report.recommendations:
            md += f"- {rec}\n"
        md += "\n"

        # Policy Effectiveness
        md += "## Policy Effectiveness\n\n"
        md += "| Policy | Effectiveness |\n"
        md += "|--------|---------------|\n"
        for policy, score in sorted(
            report.policy_effectiveness.items(), key=lambda x: x[1], reverse=True
        ):
            md += f"| {policy} | {score:.1%} |\n"
        md += "\n"

        return md
