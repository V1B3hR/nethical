"""Phase 3 Integration Module.

.. deprecated::
   This module is deprecated and maintained only for backward compatibility.
   Please use :class:`nethical.core.integrated_governance.IntegratedGovernance` instead,
   which provides a unified interface for all phases (3, 4, 5-7, 8-9).

This module integrates all Phase 3 components:
- Risk Engine
- Correlation Engine
- Fairness Sampler
- Ethical Drift Reporter
- Performance Optimizer

Migration Guide:
    Old::
        from nethical.core.phase3_integration import Phase3IntegratedGovernance
        governance = Phase3IntegratedGovernance(redis_client=redis)

    New::
        from nethical.core.integrated_governance import IntegratedGovernance
        governance = IntegratedGovernance(
            redis_client=redis,
            enable_performance_optimization=True
        )
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import time
import warnings

from .risk_engine import RiskEngine
from .correlation_engine import CorrelationEngine
from .fairness_sampler import FairnessSampler, SamplingStrategy
from .ethical_drift_reporter import EthicalDriftReporter
from .performance_optimizer import PerformanceOptimizer, DetectorTier


class Phase3IntegratedGovernance:
    """Integrated governance system with all Phase 3 features.

    .. deprecated::
       This class is deprecated. Use :class:`~nethical.core.integrated_governance.IntegratedGovernance` instead.
       Phase3IntegratedGovernance is maintained for backward compatibility only.
    """

    def __init__(
        self,
        redis_client=None,
        correlation_config_path: Optional[str] = None,
        storage_dir: str = "nethical_data",
        enable_performance_optimization: bool = True,
    ):
        """Initialize integrated governance.

        .. deprecated::
           Use IntegratedGovernance instead for unified access to all phases.

        Args:
            redis_client: Optional Redis client for persistence
            correlation_config_path: Path to correlation rules config
            storage_dir: Base directory for data storage
            enable_performance_optimization: Enable performance optimizer
        """
        warnings.warn(
            "Phase3IntegratedGovernance is deprecated and will be removed in a future version. "
            "Use IntegratedGovernance from nethical.core.integrated_governance instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Initialize all components
        self.risk_engine = RiskEngine(redis_client=redis_client, key_prefix="nethical:risk")

        self.correlation_engine = CorrelationEngine(
            config_path=correlation_config_path,
            redis_client=redis_client,
            key_prefix="nethical:correlation",
        )

        self.fairness_sampler = FairnessSampler(
            storage_dir=f"{storage_dir}/fairness_samples",
            redis_client=redis_client,
            key_prefix="nethical:fairness",
        )

        self.ethical_drift_reporter = EthicalDriftReporter(
            report_dir=f"{storage_dir}/drift_reports",
            redis_client=redis_client,
            key_prefix="nethical:drift",
        )

        self.performance_optimizer = (
            PerformanceOptimizer(target_cpu_reduction_pct=30.0)
            if enable_performance_optimization
            else None
        )

    def process_action(
        self,
        agent_id: str,
        action: Any,
        cohort: Optional[str] = None,
        violation_detected: bool = False,
        violation_type: Optional[str] = None,
        violation_severity: Optional[str] = None,
        detector_invocations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Process an action through all Phase 3 components.

        Args:
            agent_id: Agent identifier
            action: Action object
            cohort: Optional agent cohort
            violation_detected: Whether a violation was detected
            violation_type: Type of violation if detected
            violation_severity: Severity of violation if detected
            detector_invocations: Dict of detector_name -> cpu_time_ms

        Returns:
            Processing results including risk score, tier, correlations
        """
        start_time = time.time()

        results = {
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_score": 0.0,
            "risk_tier": "low",
            "correlations": [],
            "invoke_advanced_detectors": False,
            "performance_metrics": {},
        }

        # 1. Calculate risk score
        violation_score = 0.0
        if violation_detected and violation_severity:
            severity_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            violation_score = severity_map.get(violation_severity.lower(), 0.5)

        action_context = {"cohort": cohort, "has_violation": violation_detected}

        risk_score = self.risk_engine.calculate_risk_score(
            agent_id=agent_id, violation_severity=violation_score, action_context=action_context
        )

        results["risk_score"] = risk_score
        results["risk_tier"] = self.risk_engine.get_tier(agent_id).value

        # 2. Check for elevated tier trigger
        results["invoke_advanced_detectors"] = self.risk_engine.should_invoke_advanced_detectors(
            agent_id
        )

        # 3. Track correlations
        payload = getattr(action, "content", str(action))
        correlations = self.correlation_engine.track_action(
            agent_id=agent_id, action=action, payload=payload
        )
        results["correlations"] = [
            {
                "pattern": c.pattern_name,
                "severity": c.severity,
                "confidence": c.confidence,
                "description": c.description,
            }
            for c in correlations
        ]

        # 4. Track for fairness sampling (if cohort provided)
        if cohort:
            # Assign cohort
            self.fairness_sampler.assign_agent_cohort(agent_id, cohort)

            # Track for drift analysis
            self.ethical_drift_reporter.track_action(
                agent_id=agent_id, cohort=cohort, risk_score=risk_score
            )

            if violation_detected and violation_type and violation_severity:
                self.ethical_drift_reporter.track_violation(
                    agent_id=agent_id,
                    cohort=cohort,
                    violation_type=violation_type,
                    severity=violation_severity,
                )

        # 5. Track performance metrics
        if self.performance_optimizer and detector_invocations:
            for detector_name, cpu_time_ms in detector_invocations.items():
                self.performance_optimizer.track_detector_invocation(
                    detector_name=detector_name, cpu_time_ms=cpu_time_ms
                )

            total_cpu_ms = (time.time() - start_time) * 1000
            self.performance_optimizer.track_action_processing(
                cpu_time_ms=total_cpu_ms, detectors_invoked=len(detector_invocations)
            )

            results["performance_metrics"] = {
                "total_cpu_ms": total_cpu_ms,
                "cpu_reduction_pct": self.performance_optimizer.get_cpu_reduction_pct(),
                "meeting_target": self.performance_optimizer.is_meeting_target(),
            }

        return results

    def should_invoke_detector(
        self, detector_name: str, agent_id: str, tier: DetectorTier = DetectorTier.STANDARD
    ) -> bool:
        """Determine if a detector should be invoked for an agent.

        Args:
            detector_name: Name of the detector
            agent_id: Agent identifier
            tier: Detector performance tier

        Returns:
            True if detector should be invoked
        """
        if not self.performance_optimizer:
            return True

        # Register detector if not already
        if detector_name not in self.performance_optimizer.detector_registry:
            self.performance_optimizer.register_detector(detector_name, tier)

        # Get current risk score
        risk_score = self.risk_engine.get_risk_score(agent_id)

        # Check gating
        return self.performance_optimizer.should_invoke_detector(
            detector_name=detector_name, risk_score=risk_score
        )

    def generate_drift_report(
        self, cohorts: Optional[List[str]] = None, days_back: int = 7
    ) -> Dict[str, Any]:
        """Generate ethical drift report.

        Args:
            cohorts: Optional list of cohorts to include
            days_back: Number of days to look back

        Returns:
            Drift report as dictionary
        """
        from datetime import timedelta

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)

        report = self.ethical_drift_reporter.generate_report(
            start_time=start_time, end_time=end_time, cohorts=cohorts
        )

        return report.to_dict()

    def create_fairness_sampling_job(
        self, cohorts: List[str], target_sample_size: int = 1000
    ) -> str:
        """Create a fairness sampling job.

        Args:
            cohorts: List of cohorts to sample
            target_sample_size: Target number of samples

        Returns:
            Job ID
        """
        return self.fairness_sampler.create_sampling_job(
            cohorts=cohorts,
            target_sample_size=target_sample_size,
            strategy=SamplingStrategy.STRATIFIED,
        )

    def get_fairness_dashboard_data(self) -> Dict[str, Any]:
        """Get data for fairness dashboard.

        Returns:
            Dashboard data
        """
        return self.ethical_drift_reporter.get_dashboard_data()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report.

        Returns:
            Performance report
        """
        if not self.performance_optimizer:
            return {"error": "Performance optimizer not enabled"}

        return self.performance_optimizer.get_performance_report()

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions.

        Returns:
            List of suggestions
        """
        if not self.performance_optimizer:
            return ["Performance optimizer not enabled"]

        return self.performance_optimizer.suggest_optimizations()

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.

        Returns:
            System status dictionary
        """
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "risk_engine": {"active_profiles": len(self.risk_engine.profiles), "enabled": True},
                "correlation_engine": {
                    "tracked_agents": len(self.correlation_engine.agent_windows),
                    "enabled": self.correlation_engine.config.get("correlation_engine", {}).get(
                        "enabled", True
                    ),
                },
                "fairness_sampler": {
                    "active_jobs": len(self.fairness_sampler.jobs),
                    "enabled": True,
                },
                "ethical_drift_reporter": {
                    "tracked_cohorts": len(self.ethical_drift_reporter.cohort_profiles),
                    "enabled": True,
                },
                "performance_optimizer": {
                    "enabled": self.performance_optimizer is not None,
                    "meeting_target": (
                        self.performance_optimizer.is_meeting_target()
                        if self.performance_optimizer
                        else None
                    ),
                },
            },
        }

        return status
