"""Phase 5-7 Integration Module.

.. deprecated::
   This module is deprecated and maintained only for backward compatibility.
   Please use :class:`nethical.core.integrated_governance.IntegratedGovernance` instead,
   which provides a unified interface for all phases (3, 4, 5-7, 8-9).

This module integrates all Phase 5, 6, and 7 components:
- Phase 5: ML Shadow Mode (MLShadowClassifier)
- Phase 6: ML Assisted Enforcement (MLBlendedRiskEngine)
- Phase 7: Anomaly & Drift Detection (AnomalyDriftMonitor)

Migration Guide:
    Old::
        from nethical.core.phase567_integration import Phase567IntegratedGovernance
        governance = Phase567IntegratedGovernance(
            storage_dir="./data",
            enable_shadow_mode=True
        )

    New::
        from nethical.core.integrated_governance import IntegratedGovernance
        governance = IntegratedGovernance(
            storage_dir="./data",
            enable_shadow_mode=True,
            enable_ml_blending=True,
            enable_anomaly_detection=True
        )
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import warnings

from .ml_shadow import MLShadowClassifier, MLModelType
from .ml_blended_risk import MLBlendedRiskEngine
from .anomaly_detector import AnomalyDriftMonitor


class Phase567IntegratedGovernance:
    """Integrated governance system with all Phase 5-7 ML and anomaly detection features.

    .. deprecated::
       This class is deprecated. Use :class:`~nethical.core.integrated_governance.IntegratedGovernance` instead.
       Phase567IntegratedGovernance is maintained for backward compatibility only.
    """

    def __init__(
        self,
        storage_dir: str = "./phase567_data",
        enable_shadow_mode: bool = True,
        enable_ml_blending: bool = True,
        enable_anomaly_detection: bool = True,
        # Shadow classifier params
        shadow_model_type: MLModelType = MLModelType.HEURISTIC,
        shadow_score_threshold: float = 0.1,
        # Blending params
        gray_zone_lower: float = 0.4,
        gray_zone_upper: float = 0.6,
        rule_weight: float = 0.7,
        ml_weight: float = 0.3,
        # Anomaly detection params
        sequence_n: int = 3,
        psi_threshold: float = 0.2,
        anomaly_sequence_threshold: float = 0.7,
        anomaly_drift_threshold: float = 0.3,
    ):
        """Initialize Phase 5-7 integrated governance.

        Args:
            storage_dir: Base storage directory
            enable_shadow_mode: Enable ML shadow mode (Phase 5)
            enable_ml_blending: Enable ML-assisted blending (Phase 6)
            enable_anomaly_detection: Enable anomaly detection (Phase 7)
            shadow_model_type: Type of shadow model to use
            shadow_score_threshold: Agreement threshold for shadow model
            gray_zone_lower: Lower bound of gray zone for blending
            gray_zone_upper: Upper bound of gray zone for blending
            rule_weight: Weight for rule-based score in blending
            ml_weight: Weight for ML score in blending
            sequence_n: N-gram size for sequence anomaly detection
            psi_threshold: PSI threshold for drift detection
            anomaly_sequence_threshold: Threshold for sequence anomalies
            anomaly_drift_threshold: Threshold for drift alerts
        """
        warnings.warn(
            "Phase567IntegratedGovernance is deprecated and will be removed in a future version. "
            "Use IntegratedGovernance from nethical.core.integrated_governance instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.storage_dir = storage_dir

        # Phase 5: ML Shadow Classifier
        self.shadow_classifier = None
        if enable_shadow_mode:
            self.shadow_classifier = MLShadowClassifier(
                model_type=shadow_model_type,
                score_agreement_threshold=shadow_score_threshold,
                storage_path=f"{storage_dir}/shadow_logs",
            )

        # Phase 6: ML Blended Risk Engine
        self.blended_engine = None
        if enable_ml_blending:
            self.blended_engine = MLBlendedRiskEngine(
                gray_zone_lower=gray_zone_lower,
                gray_zone_upper=gray_zone_upper,
                rule_weight=rule_weight,
                ml_weight=ml_weight,
                storage_path=f"{storage_dir}/blended_logs",
            )

        # Phase 7: Anomaly Drift Monitor
        self.anomaly_monitor = None
        if enable_anomaly_detection:
            self.anomaly_monitor = AnomalyDriftMonitor(
                sequence_n=sequence_n,
                psi_threshold=psi_threshold,
                kl_threshold=anomaly_drift_threshold,
                storage_path=f"{storage_dir}/anomaly_logs",
            )

        # Component flags
        self.components_enabled = {
            "shadow_mode": enable_shadow_mode,
            "ml_blending": enable_ml_blending,
            "anomaly_detection": enable_anomaly_detection,
        }

    def process_action(
        self,
        agent_id: str,
        action_id: str,
        action_type: str,
        features: Dict[str, float],
        rule_risk_score: float,
        rule_classification: str,
        cohort: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process action through all Phase 5-7 components.

        Args:
            agent_id: Agent identifier
            action_id: Action identifier
            action_type: Type of action
            features: Feature dictionary for ML
            rule_risk_score: Risk score from rule-based system
            rule_classification: Classification from rules (allow/warn/deny)
            cohort: Optional agent cohort
            ml_confidence: Optional ML confidence score
            context: Additional context

        Returns:
            Processing results dictionary including shadow prediction, blended decision, and anomaly alerts
        """
        start_time = time.time()

        results = {
            "agent_id": agent_id,
            "action_id": action_id,
            "action_type": action_type,
            "timestamp": datetime.utcnow().isoformat(),
            "rule_risk_score": rule_risk_score,
            "rule_classification": rule_classification,
            "cohort": cohort,
        }

        # Phase 5: Shadow prediction (passive)
        shadow_prediction = None
        if self.shadow_classifier:
            shadow_prediction = self.shadow_classifier.predict(
                agent_id=agent_id,
                action_id=action_id,
                features=features,
                rule_risk_score=rule_risk_score,
                rule_classification=rule_classification,
            )

            results["shadow"] = {
                "ml_risk_score": shadow_prediction.ml_risk_score,
                "ml_classification": shadow_prediction.ml_classification,
                "ml_confidence": shadow_prediction.ml_confidence,
                "scores_agree": shadow_prediction.scores_agree,
                "classifications_agree": shadow_prediction.classifications_agree,
            }

        # Phase 6: Blended risk (active if enabled)
        blended_decision = None
        final_risk_score = rule_risk_score
        final_classification = rule_classification

        if self.blended_engine and shadow_prediction:
            blended_decision = self.blended_engine.compute_blended_risk(
                agent_id=agent_id,
                action_id=action_id,
                rule_risk_score=rule_risk_score,
                rule_classification=rule_classification,
                ml_risk_score=shadow_prediction.ml_risk_score,
                ml_confidence=ml_confidence or shadow_prediction.ml_confidence,
                features=features,
            )

            final_risk_score = blended_decision.blended_risk_score
            final_classification = blended_decision.final_classification

            results["blended"] = {
                "blended_risk_score": blended_decision.blended_risk_score,
                "final_classification": blended_decision.final_classification,
                "risk_zone": blended_decision.risk_zone.value,
                "ml_influenced": blended_decision.ml_influenced,
                "blended_classification": blended_decision.blended_classification,
                "classification_changed": blended_decision.classification_changed,
            }

        # Phase 7: Anomaly detection
        anomaly_alert = None
        if self.anomaly_monitor:
            anomaly_alert = self.anomaly_monitor.record_action(
                agent_id=agent_id,
                action_type=action_type,
                risk_score=final_risk_score,
                cohort=cohort or "default",
            )

            if anomaly_alert:
                results["anomaly_alert"] = {
                    "anomaly_type": anomaly_alert.anomaly_type.value,
                    "severity": anomaly_alert.severity.value,
                    "anomaly_score": anomaly_alert.anomaly_score,
                    "threshold": anomaly_alert.threshold,
                    "quarantine_recommended": anomaly_alert.quarantine_recommended,
                    "message": anomaly_alert.description,
                }
            else:
                results["anomaly_alert"] = None

            # Also check for behavioral anomalies
            behavioral_alert = self.anomaly_monitor.check_behavioral_anomaly(agent_id)
            if behavioral_alert:
                results["behavioral_anomaly"] = {
                    "anomaly_type": behavioral_alert.anomaly_type.value,
                    "severity": behavioral_alert.severity.value,
                    "anomaly_score": behavioral_alert.anomaly_score,
                    "quarantine_recommended": behavioral_alert.quarantine_recommended,
                }

            # Check for drift in cohort
            if cohort:
                drift_alert = self.anomaly_monitor.check_drift(cohort=cohort)
                if drift_alert:
                    results["drift_alert"] = {
                        "anomaly_type": drift_alert.anomaly_type.value,
                        "severity": drift_alert.severity.value,
                        "anomaly_score": drift_alert.anomaly_score,
                        "quarantine_recommended": drift_alert.quarantine_recommended,
                    }

        # Overall decision
        results["final_decision"] = {
            "risk_score": final_risk_score,
            "classification": final_classification,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

        return results

    def set_baseline_distribution(self, risk_scores: List[float], cohort: str = "default") -> bool:
        """Set baseline distribution for drift detection.

        Args:
            risk_scores: List of historical risk scores
            cohort: Cohort identifier (not currently used by AnomalyDriftMonitor)

        Returns:
            True if successful
        """
        if not self.anomaly_monitor:
            return False

        self.anomaly_monitor.set_baseline_distribution(risk_scores)
        return True

    def get_shadow_metrics(self) -> Dict[str, Any]:
        """Get ML shadow mode metrics.

        Returns:
            Shadow metrics report
        """
        if not self.shadow_classifier:
            return {"error": "Shadow mode not enabled"}

        return self.shadow_classifier.get_metrics_report()

    def get_blending_metrics(self) -> Dict[str, Any]:
        """Get ML blending metrics.

        Returns:
            Blending metrics report
        """
        if not self.blended_engine:
            return {"error": "ML blending not enabled"}

        return self.blended_engine.get_metrics_report()

    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics.

        Returns:
            Anomaly statistics
        """
        if not self.anomaly_monitor:
            return {"error": "Anomaly detection not enabled"}

        return self.anomaly_monitor.get_statistics()

    def get_drift_report(self, cohort: str = "default") -> Dict[str, Any]:
        """Get drift analysis report for a cohort.

        Args:
            cohort: Cohort identifier

        Returns:
            Drift report
        """
        if not self.anomaly_monitor:
            return {"error": "Anomaly detection not enabled"}

        metrics = self.anomaly_monitor.get_drift_metrics(cohort)
        if not metrics:
            return {"error": f"No drift metrics for cohort {cohort}"}

        return {
            "cohort": cohort,
            "psi_score": metrics.psi_score,
            "kl_divergence": metrics.kl_divergence,
            "drift_detected": metrics.drift_detected,
            "baseline_size": metrics.baseline_size,
            "current_size": metrics.current_size,
            "timestamp": metrics.timestamp.isoformat(),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status for Phase 5-7.

        Returns:
            System status dictionary
        """
        status = {"timestamp": datetime.utcnow().isoformat(), "components": {}}

        # Shadow classifier status
        if self.shadow_classifier:
            metrics = self.shadow_classifier.get_metrics_report()
            status["components"]["shadow_classifier"] = {
                "enabled": True,
                "total_predictions": metrics.get("total_predictions", 0),
                "f1_score": metrics.get("f1_score", 0.0),
                "agreement_rate": metrics.get("classification_agreement_rate", 0.0),
            }
        else:
            status["components"]["shadow_classifier"] = {"enabled": False}

        # Blended engine status
        if self.blended_engine:
            report = self.blended_engine.get_metrics_report()
            status["components"]["blended_engine"] = {
                "enabled": True,
                "total_decisions": report.get("total_decisions", 0),
                "ml_influenced_count": report.get("ml_influenced_count", 0),
                "ml_influence_rate": report.get("ml_influence_rate", 0.0),
                "classification_change_rate": report.get("classification_change_rate", 0.0),
            }
        else:
            status["components"]["blended_engine"] = {"enabled": False}

        # Anomaly monitor status
        if self.anomaly_monitor:
            stats = self.anomaly_monitor.get_statistics()
            status["components"]["anomaly_monitor"] = {
                "enabled": True,
                "total_alerts": stats["alerts"]["total"],
                "tracked_agents": stats.get("tracked_agents", 0),
                "alert_rate": stats["alerts"]["total"] / max(stats.get("tracked_agents", 1), 1),
            }
        else:
            status["components"]["anomaly_monitor"] = {"enabled": False}

        return status

    def export_phase567_report(self) -> str:
        """Export comprehensive Phase 5-7 report.

        Returns:
            Report as markdown
        """
        lines = []
        lines.append("# Phase 5-7: ML & Anomaly Detection - Report")
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
                    if isinstance(value, float):
                        lines.append(f"  - {key}: {value:.3f}")
                    else:
                        lines.append(f"  - {key}: {value}")
        lines.append("")

        # Shadow metrics
        if self.shadow_classifier:
            lines.append("## Phase 5: Shadow Mode Metrics")
            lines.append("")
            metrics = self.get_shadow_metrics()
            lines.append(f"- Total Predictions: {metrics.get('total_predictions', 0)}")
            lines.append(f"- F1 Score: {metrics.get('f1_score', 0.0):.3f}")
            lines.append(f"- Precision: {metrics.get('precision', 0.0):.3f}")
            lines.append(f"- Recall: {metrics.get('recall', 0.0):.3f}")
            lines.append(
                f"- Agreement Rate: {metrics.get('classification_agreement_rate', 0.0)*100:.1f}%"
            )
            lines.append("")

        # Blending metrics
        if self.blended_engine:
            lines.append("## Phase 6: ML Blending Metrics")
            lines.append("")
            report = self.get_blending_metrics()
            lines.append(f"- Total Decisions: {report.get('total_decisions', 0)}")
            lines.append(f"- ML Influenced: {report.get('ml_influenced_count', 0)}")
            lines.append(f"- ML Influence Rate: {report.get('ml_influence_rate', 0.0)*100:.1f}%")
            lines.append(f"- Classification Changes: {report.get('classification_changes', 0)}")
            lines.append(f"- Change Rate: {report.get('classification_change_rate', 0.0)*100:.1f}%")
            lines.append("")

        # Anomaly statistics
        if self.anomaly_monitor:
            lines.append("## Phase 7: Anomaly Detection")
            lines.append("")
            stats = self.get_anomaly_statistics()
            lines.append(f"- Total Alerts: {stats['alerts']['total']}")
            lines.append(f"- Tracked Agents: {stats.get('tracked_agents', 0)}")
            lines.append(f"- Sequence Anomalies: {stats['alerts']['by_type'].get('sequence', 0)}")
            lines.append(
                f"- Distributional Anomalies: {stats['alerts']['by_type'].get('distributional', 0)}"
            )
            lines.append(
                f"- Critical Severity: {stats['alerts']['by_severity'].get('critical', 0)}"
            )
            lines.append("")

        return "\n".join(lines)
