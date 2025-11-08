"""Phase 4 Integration Module.

.. deprecated::
   This module is deprecated and maintained only for backward compatibility.
   Please use :class:`nethical.core.integrated_governance.IntegratedGovernance` instead,
   which provides a unified interface for all phases (3, 4, 5-7, 8-9).

This module integrates all Phase 4 components:
- Merkle Anchoring
- Policy Diff Auditing
- Quarantine Mode
- Ethical Taxonomy
- SLA Monitoring

Migration Guide:
    Old::
        from nethical.core.phase4_integration import Phase4IntegratedGovernance
        governance = Phase4IntegratedGovernance(
            storage_dir="./data",
            enable_merkle_anchoring=True
        )

    New::
        from nethical.core.integrated_governance import IntegratedGovernance
        governance = IntegratedGovernance(
            storage_dir="./data",
            enable_merkle_anchoring=True,
            enable_quarantine=True,
            enable_ethical_taxonomy=True,
            enable_sla_monitoring=True
        )
"""

from typing import Dict, Optional, Any
from datetime import datetime
import time
import warnings

from .audit_merkle import MerkleAnchor
from .policy_diff import PolicyDiffAuditor
from .quarantine import QuarantineManager, QuarantineReason
from .ethical_taxonomy import EthicalTaxonomy
from .sla_monitor import SLAMonitor


class Phase4IntegratedGovernance:
    """Integrated governance system with all Phase 4 features.

    .. deprecated::
       This class is deprecated. Use :class:`~nethical.core.integrated_governance.IntegratedGovernance` instead.
       Phase4IntegratedGovernance is maintained for backward compatibility only.
    """

    def __init__(
        self,
        storage_dir: str = "./phase4_data",
        enable_merkle_anchoring: bool = True,
        enable_quarantine: bool = True,
        enable_ethical_taxonomy: bool = True,
        enable_sla_monitoring: bool = True,
        s3_bucket: Optional[str] = None,
        taxonomy_path: str = "ethics_taxonomy.json",
    ):
        """Initialize Phase 4 integrated governance.

        .. deprecated::
           Use IntegratedGovernance instead for unified access to all phases.

        Args:
            storage_dir: Base storage directory
            enable_merkle_anchoring: Enable Merkle anchoring
            enable_quarantine: Enable quarantine mode
            enable_ethical_taxonomy: Enable ethical taxonomy
            enable_sla_monitoring: Enable SLA monitoring
            s3_bucket: Optional S3 bucket for Merkle anchoring
            taxonomy_path: Path to ethical taxonomy config
        """
        warnings.warn(
            "Phase4IntegratedGovernance is deprecated and will be removed in a future version. "
            "Use IntegratedGovernance from nethical.core.integrated_governance instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.storage_dir = storage_dir

        # Initialize components
        self.merkle_anchor = None
        if enable_merkle_anchoring:
            self.merkle_anchor = MerkleAnchor(
                storage_path=f"{storage_dir}/audit_logs", s3_bucket=s3_bucket
            )

        self.policy_auditor = PolicyDiffAuditor(storage_path=f"{storage_dir}/policy_history")

        self.quarantine_manager = None
        if enable_quarantine:
            self.quarantine_manager = QuarantineManager()

        self.ethical_taxonomy = None
        if enable_ethical_taxonomy:
            self.ethical_taxonomy = EthicalTaxonomy(taxonomy_path=taxonomy_path)

        self.sla_monitor = None
        if enable_sla_monitoring:
            self.sla_monitor = SLAMonitor()

        # Component flags
        self.components_enabled = {
            "merkle_anchoring": enable_merkle_anchoring,
            "policy_auditing": True,
            "quarantine": enable_quarantine,
            "ethical_taxonomy": enable_ethical_taxonomy,
            "sla_monitoring": enable_sla_monitoring,
        }

    def process_action(
        self,
        agent_id: str,
        action: Any,
        cohort: Optional[str] = None,
        violation_detected: bool = False,
        violation_type: Optional[str] = None,
        violation_severity: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process action through all Phase 4 components.

        Args:
            agent_id: Agent identifier
            action: Action object
            cohort: Agent cohort
            violation_detected: Whether violation was detected
            violation_type: Type of violation
            violation_severity: Severity level
            context: Additional context

        Returns:
            Processing results dictionary
        """
        start_time = time.time()

        results = {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "cohort": cohort,
        }

        # 1. Check quarantine status
        if self.quarantine_manager and cohort:
            quarantine_status = self.quarantine_manager.get_quarantine_status(cohort)
            results["quarantine_status"] = quarantine_status

            # Check if agent is quarantined
            if quarantine_status.get("is_quarantined"):
                results["action_allowed"] = False
                results["reason"] = "cohort_quarantined"
                return results

        results["action_allowed"] = True

        # 2. Tag with ethical dimensions
        if self.ethical_taxonomy and violation_detected and violation_type:
            tagging = self.ethical_taxonomy.create_tagging(
                violation_type=violation_type, context=context
            )
            results["ethical_tags"] = {
                "primary_dimension": tagging.primary_dimension,
                "dimensions": {tag.dimension: tag.score for tag in tagging.tags},
            }

        # 3. Audit to Merkle tree
        if self.merkle_anchor:
            event_data = {
                "event_id": f"evt_{int(time.time() * 1000000)}",
                "agent_id": agent_id,
                "cohort": cohort,
                "timestamp": datetime.utcnow().isoformat(),
                "action_type": str(type(action).__name__),
                "violation_detected": violation_detected,
                "violation_type": violation_type,
                "ethical_tags": results.get("ethical_tags"),
            }

            self.merkle_anchor.add_event(event_data)

            # Get current chunk info
            if self.merkle_anchor.current_chunk:
                results["audit"] = {
                    "chunk_id": self.merkle_anchor.current_chunk.chunk_id,
                    "event_count": self.merkle_anchor.current_chunk.event_count,
                }

        # 4. Track latency for SLA
        if self.sla_monitor:
            latency_ms = (time.time() - start_time) * 1000
            self.sla_monitor.record_latency(latency_ms)
            results["latency_ms"] = latency_ms

        return results

    def quarantine_cohort(
        self, cohort: str, reason: str = "manual", duration_hours: float = 24.0
    ) -> Dict[str, Any]:
        """Quarantine an agent cohort.

        Args:
            cohort: Cohort to quarantine
            reason: Reason for quarantine
            duration_hours: Quarantine duration

        Returns:
            Quarantine record
        """
        if not self.quarantine_manager:
            return {"error": "Quarantine not enabled"}

        # Map reason string to enum
        reason_enum = QuarantineReason.MANUAL_OVERRIDE
        if reason == "anomaly":
            reason_enum = QuarantineReason.ANOMALY_DETECTED
        elif reason == "attack":
            reason_enum = QuarantineReason.COORDINATED_ATTACK

        record = self.quarantine_manager.quarantine_cohort(
            cohort=cohort, reason=reason_enum, duration_hours=duration_hours
        )

        return {
            "cohort": record.cohort,
            "status": record.status.value,
            "activated_at": record.activated_at.isoformat() if record.activated_at else None,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "activation_time_ms": record.activation_time_ms,
            "affected_agents": len(record.affected_agents),
        }

    def release_cohort(self, cohort: str) -> bool:
        """Release cohort from quarantine.

        Args:
            cohort: Cohort to release

        Returns:
            True if released
        """
        if not self.quarantine_manager:
            return False

        return self.quarantine_manager.release_cohort(cohort)

    def compare_policies(
        self, old_policy: Dict[str, Any], new_policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two policy versions.

        Args:
            old_policy: Old policy
            new_policy: New policy

        Returns:
            Diff result
        """
        diff_result = self.policy_auditor.compare_policies(
            old_policy=old_policy, new_policy=new_policy
        )

        return {
            "risk_score": diff_result.risk_score,
            "risk_level": diff_result.risk_level.value,
            "summary": diff_result.summary,
            "recommendations": diff_result.recommendations,
            "changes": [
                {
                    "path": c.path,
                    "type": c.change_type.value,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "risk_level": c.risk_level.value,
                }
                for c in diff_result.changes
            ],
        }

    def finalize_audit_chunk(self) -> Optional[str]:
        """Finalize current audit chunk and get Merkle root.

        Returns:
            Merkle root hash or None
        """
        if not self.merkle_anchor:
            return None

        if self.merkle_anchor.current_chunk.event_count == 0:
            return None

        return self.merkle_anchor.finalize_chunk()

    def verify_audit_segment(self, chunk_id: str) -> bool:
        """Verify integrity of audit segment.

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if valid
        """
        if not self.merkle_anchor:
            return False

        return self.merkle_anchor.verify_chunk(chunk_id)

    def get_sla_report(self) -> Dict[str, Any]:
        """Get SLA compliance report.

        Returns:
            SLA report
        """
        if not self.sla_monitor:
            return {"error": "SLA monitoring not enabled"}

        return self.sla_monitor.get_sla_report()

    def get_ethical_coverage(self) -> Dict[str, Any]:
        """Get ethical taxonomy coverage report.

        Returns:
            Coverage report
        """
        if not self.ethical_taxonomy:
            return {"error": "Ethical taxonomy not enabled"}

        return self.ethical_taxonomy.get_coverage_report()

    def simulate_quarantine(self, cohort: str) -> Dict[str, Any]:
        """Simulate quarantine response for testing.

        Args:
            cohort: Cohort to test

        Returns:
            Simulation results
        """
        if not self.quarantine_manager:
            return {"error": "Quarantine not enabled"}

        return self.quarantine_manager.simulate_attack_response(
            cohort=cohort, attack_type="synthetic"
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.

        Returns:
            System status dictionary
        """
        status = {"timestamp": datetime.utcnow().isoformat(), "components": {}}

        # Merkle anchor status
        if self.merkle_anchor:
            stats = self.merkle_anchor.get_statistics()
            status["components"]["merkle_anchor"] = {
                "enabled": True,
                "total_chunks": stats["total_chunks"],
                "total_events": stats["total_events"],
                "current_chunk_events": stats["current_chunk_events"],
                "anchored_chunks": stats["anchored_chunks"],
            }
        else:
            status["components"]["merkle_anchor"] = {"enabled": False}

        # Policy auditor status
        status["components"]["policy_auditor"] = {
            "enabled": True,
            "version_count": len(self.policy_auditor.version_history),
        }

        # Quarantine status
        if self.quarantine_manager:
            stats = self.quarantine_manager.get_statistics()
            status["components"]["quarantine"] = {
                "enabled": True,
                "active_quarantines": stats["active_quarantines"],
                "total_quarantines": stats["total_quarantines"],
                "avg_activation_time_ms": stats["avg_activation_time_ms"],
            }
        else:
            status["components"]["quarantine"] = {"enabled": False}

        # Ethical taxonomy status
        if self.ethical_taxonomy:
            coverage = self.ethical_taxonomy.get_coverage_stats()
            status["components"]["ethical_taxonomy"] = {
                "enabled": True,
                "coverage_percentage": coverage["coverage_percentage"],
                "meets_target": coverage["meets_target"],
                "total_violation_types": coverage["total_violation_types"],
            }
        else:
            status["components"]["ethical_taxonomy"] = {"enabled": False}

        # SLA monitor status
        if self.sla_monitor:
            report = self.sla_monitor.get_sla_report()
            status["components"]["sla_monitor"] = {
                "enabled": True,
                "status": report["overall_status"],
                "sla_met": report["sla_met"],
                "p95_latency_ms": report["p95_latency_ms"],
            }
        else:
            status["components"]["sla_monitor"] = {"enabled": False}

        return status

    def export_phase4_report(self) -> str:
        """Export comprehensive Phase 4 report.

        Returns:
            Report as markdown
        """
        lines = []
        lines.append("# Phase 4: Integrity & Ethics Operationalization - Report")
        lines.append("")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")

        # System status
        status = self.get_system_status()
        lines.append("## System Status")
        lines.append("")
        for component, info in status["components"].items():
            enabled = "✅" if info.get("enabled") else "❌"
            lines.append(f"- {enabled} **{component.replace('_', ' ').title()}**")
            for key, value in info.items():
                if key != "enabled":
                    lines.append(f"  - {key}: {value}")
        lines.append("")

        # SLA Report
        if self.sla_monitor:
            lines.append("## SLA Performance")
            lines.append("")
            report = self.get_sla_report()
            lines.append(f"- Status: {report['overall_status']}")
            lines.append(
                f"- P95 Latency: {report['p95_latency_ms']:.1f}ms (target: {report['p95_target_ms']}ms)"
            )
            lines.append(f"- SLA Met: {'✅' if report['sla_met'] else '❌'}")
            lines.append("")

        # Ethical Coverage
        if self.ethical_taxonomy:
            lines.append("## Ethical Taxonomy Coverage")
            lines.append("")
            coverage = self.get_ethical_coverage()
            lines.append(f"- Coverage: {coverage['coverage_percentage']:.1f}%")
            lines.append(f"- Target: {coverage['target_percentage']:.1f}%")
            lines.append(f"- Meets Target: {'✅' if coverage['meets_target'] else '❌'}")
            lines.append("")

        # Quarantine Summary
        if self.quarantine_manager:
            lines.append("## Quarantine System")
            lines.append("")
            stats = self.quarantine_manager.get_statistics()
            lines.append(f"- Active Quarantines: {stats['active_quarantines']}")
            lines.append(f"- Avg Activation Time: {stats['avg_activation_time_ms']:.1f}ms")
            lines.append(f"- Target: <{stats['target_activation_time_s'] * 1000}ms")
            lines.append("")

        return "\n".join(lines)
