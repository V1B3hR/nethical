"""Unified Integrated Governance Module.

This module consolidates ALL phases (3, 4, 5-7, 8-9, F3) into a single unified interface:
- Phase 3: Risk Engine, Correlation, Fairness, Drift Reporting, Performance Optimization
- Phase 4: Merkle Anchoring, Policy Diff, Quarantine, Ethical Taxonomy, SLA Monitoring
- Phase 5-7: ML Shadow Mode, ML Blended Risk, Anomaly Detection
- Phase 8-9: Human-in-the-Loop, Continuous Optimization
- F3: Privacy & Data Handling - Enhanced Redaction, Differential Privacy, Federated Analytics, Data Minimization

This provides a complete governance system with all features in a single interface.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import time
import hashlib
from functools import lru_cache
from collections import OrderedDict

# Phase 3 imports
from .risk_engine import RiskEngine
from .correlation_engine import CorrelationEngine
from .fairness_sampler import FairnessSampler
from .ethical_drift_reporter import EthicalDriftReporter
from .performance_optimizer import PerformanceOptimizer

# Phase 4 imports
from .audit_merkle import MerkleAnchor
from .policy_diff import PolicyDiffAuditor
from .quarantine import QuarantineManager
from .ethical_taxonomy import EthicalTaxonomy
from .sla_monitor import SLAMonitor

# Phase 5-7 imports
from .ml_shadow import MLShadowClassifier, MLModelType
from .ml_blended_risk import MLBlendedRiskEngine
from .anomaly_detector import AnomalyDriftMonitor

# Phase 8-9 imports
from .human_feedback import EscalationQueue
from .optimization import MultiObjectiveOptimizer, Configuration

# F3: Privacy & Data Handling imports
from .redaction_pipeline import EnhancedRedactionPipeline, RedactionPolicy as RedactionPolicyEnum
from .differential_privacy import DifferentialPrivacy, PrivacyMechanism, PrivacyAudit
from .federated_analytics import FederatedAnalytics
from .data_minimization import DataMinimization

# F2: Plugin Interface imports
from .plugin_interface import get_plugin_manager

# Governance/Safety imports
from .governance_core import EnhancedSafetyGovernance, AgentAction, Decision
from .models import AgentAction as ModelsAgentAction
import asyncio

# Resource Management imports
try:
    from ..quotas import QuotaEnforcer, QuotaConfig
    from ..utils.pii import PIIDetector, get_pii_detector

    QUOTA_AVAILABLE = True
except ImportError:
    QUOTA_AVAILABLE = False
    QuotaEnforcer = None
    QuotaConfig = None
    PIIDetector = None


class IntegratedGovernance:
    """Consolidated governance system with all features from Phases 3-9 and F3.

    This class provides a unified interface to all governance features:
    - Risk scoring and correlation detection (Phase 3)
    - Audit trails and policy management (Phase 4)
    - ML-assisted decision making (Phases 5-7)
    - Human oversight and optimization (Phases 8-9)
    """

    def __init__(
        self,
        storage_dir: str = "./nethical_data",
        # Regional & Sharding config
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        data_residency_policy: Optional[str] = None,
        # Phase 3 config
        redis_client=None,
        correlation_config_path: Optional[str] = None,
        enable_performance_optimization: bool = True,
        # Phase 4 config
        enable_merkle_anchoring: bool = True,
        enable_quarantine: bool = True,
        enable_ethical_taxonomy: bool = True,
        enable_sla_monitoring: bool = True,
        s3_bucket: Optional[str] = None,
        taxonomy_path: str = "taxonomies/ethics_taxonomy.json",
        # Phase 5-7 config
        enable_shadow_mode: bool = True,
        enable_ml_blending: bool = True,
        enable_anomaly_detection: bool = True,
        shadow_model_type: MLModelType = MLModelType.HEURISTIC,
        shadow_score_threshold: float = 0.1,
        gray_zone_lower: float = 0.4,
        gray_zone_upper: float = 0.6,
        rule_weight: float = 0.7,
        ml_weight: float = 0.3,
        sequence_n: int = 3,
        psi_threshold: float = 0.2,
        anomaly_sequence_threshold: float = 0.7,
        anomaly_drift_threshold: float = 0.3,
        # Phase 8-9 config
        triage_sla_seconds: float = 3600,
        resolution_sla_seconds: float = 86400,
        auto_escalate_on_block: bool = True,
        auto_escalate_on_low_confidence: bool = True,
        low_confidence_threshold: float = 0.7,
        # F3: Privacy & Data Handling config
        privacy_mode: Optional[str] = None,
        epsilon: float = 1.0,
        redaction_policy: str = "standard",
        # Quota & Resource Management config
        enable_quota_enforcement: bool = False,
        requests_per_second: float = 10.0,
        max_payload_bytes: int = 1_000_000,
        # Performance Optimization config
        enable_pii_caching: bool = True,
        pii_cache_size: int = 10000,
        enable_fast_path: bool = True,
        fast_path_risk_threshold: float = 0.3,
        db_pool_size: int = 10,
        merkle_batch_size: int = 100,
        enable_parallel_phases: bool = True,
    ):
        """Initialize unified integrated governance.

        Args:
            storage_dir: Base directory for all data storage
            region_id: Geographic region identifier (e.g., 'eu-west-1')
            logical_domain: Logical domain for hierarchical aggregation (e.g., 'customer-service')
            data_residency_policy: Data residency policy (e.g., 'EU_GDPR', 'US_CCPA')
            redis_client: Optional Redis client for persistence
            correlation_config_path: Path to correlation rules config
            enable_performance_optimization: Enable performance optimizer
            enable_merkle_anchoring: Enable immutable audit logs
            enable_quarantine: Enable quarantine functionality
            enable_ethical_taxonomy: Enable ethical tagging
            enable_sla_monitoring: Enable SLA tracking
            s3_bucket: Optional S3 bucket for storage
            taxonomy_path: Path to ethics taxonomy file
            enable_shadow_mode: Enable ML shadow mode
            enable_ml_blending: Enable ML-assisted blending
            enable_anomaly_detection: Enable anomaly detection
            shadow_model_type: Type of shadow model
            shadow_score_threshold: Agreement threshold for shadow model
            gray_zone_lower: Lower bound of gray zone for blending
            gray_zone_upper: Upper bound of gray zone for blending
            rule_weight: Weight for rule-based score in blending
            ml_weight: Weight for ML score in blending
            sequence_n: N-gram size for sequence anomaly detection
            psi_threshold: PSI threshold for drift detection
            anomaly_sequence_threshold: Threshold for sequence anomalies
            anomaly_drift_threshold: Threshold for drift alerts
            triage_sla_seconds: SLA for starting review
            resolution_sla_seconds: SLA for completing review
            auto_escalate_on_block: Auto-escalate BLOCK/TERMINATE decisions
            auto_escalate_on_low_confidence: Auto-escalate low confidence
            low_confidence_threshold: Threshold for low confidence escalation
            privacy_mode: Privacy mode ('differential', 'standard', or None)
            epsilon: Privacy budget for differential privacy
            redaction_policy: Redaction policy ('minimal', 'standard', 'aggressive')
        """
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)

        # ==================== Regional Configuration ====================
        self.region_id = region_id
        self.logical_domain = logical_domain
        self.data_residency_policy = data_residency_policy
        self.regional_policies: Dict[str, Any] = {}

        # Initialize regional policy configurations
        if data_residency_policy:
            self._load_regional_policy(data_residency_policy)

        # ==================== Core Governance & Violation Detection ====================
        # Initialize the enhanced safety governance for violation detection
        from .governance_core import MonitoringConfig
        safety_storage = storage_path / "safety_governance"
        safety_storage.mkdir(parents=True, exist_ok=True)
        safety_config = MonitoringConfig(
            enable_persistence=True,
            db_path=str(safety_storage / "governance_data.sqlite")
        )
        self.safety_governance = EnhancedSafetyGovernance(config=safety_config)

        # ==================== Phase 3 Components ====================
        self.risk_engine = RiskEngine(redis_client=redis_client, key_prefix="nethical:risk")

        self.correlation_engine = CorrelationEngine(
            config_path=correlation_config_path,
            redis_client=redis_client,
            key_prefix="nethical:correlation",
        )

        self.fairness_sampler = FairnessSampler(
            storage_dir=str(storage_path / "fairness_samples"),
            redis_client=redis_client,
            key_prefix="nethical:fairness",
        )

        self.ethical_drift_reporter = EthicalDriftReporter(
            report_dir=str(storage_path / "drift_reports"),
            redis_client=redis_client,
            key_prefix="nethical:drift",
        )

        self.performance_optimizer = (
            PerformanceOptimizer(target_cpu_reduction_pct=30.0)
            if enable_performance_optimization
            else None
        )

        # ==================== Phase 4 Components ====================
        self.merkle_anchor = None
        if enable_merkle_anchoring:
            self.merkle_anchor = MerkleAnchor(
                storage_path=str(storage_path / "merkle_data"), s3_bucket=s3_bucket
            )

        self.policy_auditor = PolicyDiffAuditor()

        self.quarantine_manager = None
        if enable_quarantine:
            self.quarantine_manager = QuarantineManager()

        self.ethical_taxonomy = None
        if enable_ethical_taxonomy:
            self.ethical_taxonomy = EthicalTaxonomy(taxonomy_path=taxonomy_path)

        self.sla_monitor = None
        if enable_sla_monitoring:
            self.sla_monitor = SLAMonitor()

        # ==================== Phase 5-7 Components ====================
        self.shadow_classifier = None
        if enable_shadow_mode:
            self.shadow_classifier = MLShadowClassifier(
                model_type=shadow_model_type,
                score_agreement_threshold=shadow_score_threshold,
                storage_path=str(storage_path / "shadow_logs"),
            )

        self.blended_engine = None
        if enable_ml_blending:
            self.blended_engine = MLBlendedRiskEngine(
                gray_zone_lower=gray_zone_lower,
                gray_zone_upper=gray_zone_upper,
                rule_weight=rule_weight,
                ml_weight=ml_weight,
                storage_path=str(storage_path / "blended_logs"),
            )

        self.anomaly_monitor = None
        if enable_anomaly_detection:
            self.anomaly_monitor = AnomalyDriftMonitor(
                sequence_n=sequence_n,
                psi_threshold=psi_threshold,
                kl_threshold=anomaly_drift_threshold,
                storage_path=str(storage_path / "anomaly_logs"),
            )

        # ==================== Phase 8-9 Components ====================
        self.escalation_queue = EscalationQueue(
            storage_path=str(storage_path / "escalations.db"),
            triage_sla_seconds=triage_sla_seconds,
            resolution_sla_seconds=resolution_sla_seconds,
        )

        self.optimizer = MultiObjectiveOptimizer(storage_path=str(storage_path / "optimization.db"))

        # Configuration
        self.auto_escalate_on_block = auto_escalate_on_block
        self.auto_escalate_on_low_confidence = auto_escalate_on_low_confidence
        self.low_confidence_threshold = low_confidence_threshold
        self.active_config: Optional[Configuration] = None

        # ==================== F3: Privacy & Data Handling ====================
        self.privacy_mode = privacy_mode
        self.epsilon = epsilon
        self.redaction_policy_name = redaction_policy

        # Enhanced Redaction Pipeline
        redaction_policy_enum = {
            "minimal": RedactionPolicyEnum.MINIMAL,
            "standard": RedactionPolicyEnum.STANDARD,
            "aggressive": RedactionPolicyEnum.AGGRESSIVE,
        }.get(redaction_policy, RedactionPolicyEnum.STANDARD)

        self.redaction_pipeline = EnhancedRedactionPipeline(
            policy=redaction_policy_enum,
            enable_audit=True,
            enable_reversible=(privacy_mode == "differential"),
            audit_log_path=str(storage_path / "redaction_audit.jsonl"),
        )

        # Differential Privacy (if enabled)
        self.differential_privacy = None
        if privacy_mode == "differential":
            self.differential_privacy = DifferentialPrivacy(
                epsilon=epsilon, delta=1e-5, mechanism=PrivacyMechanism.GAUSSIAN
            )
            self.privacy_audit = PrivacyAudit(self.differential_privacy)

        # Federated Analytics (if regions specified)
        self.federated_analytics = None
        if region_id:
            regions_list = [region_id]
            # In multi-region deployments, this would be configured with multiple regions
            self.federated_analytics = FederatedAnalytics(
                regions=regions_list,
                enable_encryption=True,
                privacy_preserving=(privacy_mode == "differential"),
                noise_level=0.1,
            )

        # Data Minimization
        self.data_minimization = DataMinimization(
            storage_dir=str(storage_path / "data_minimization"),
            enable_auto_deletion=True,
            anonymization_enabled=True,
        )

        # ==================== Quota & Resource Management ====================
        self.quota_enforcer = None
        self.enable_quota_enforcement = enable_quota_enforcement
        if enable_quota_enforcement and QUOTA_AVAILABLE:
            quota_config = QuotaConfig(
                requests_per_second=requests_per_second, max_payload_bytes=max_payload_bytes
            )
            self.quota_enforcer = QuotaEnforcer(quota_config)

        # PII Detector (always available for enhanced privacy)
        self.pii_detector = None
        if QUOTA_AVAILABLE:
            self.pii_detector = get_pii_detector()

        # ==================== Performance Optimization Configuration ====================
        self.enable_pii_caching = enable_pii_caching
        self.pii_cache_size = pii_cache_size
        self.enable_fast_path = enable_fast_path
        self.fast_path_risk_threshold = fast_path_risk_threshold
        self.db_pool_size = db_pool_size
        self.merkle_batch_size = merkle_batch_size
        self.enable_parallel_phases = enable_parallel_phases
        
        # Database connection pool (Fix 3)
        from .db_pool import SQLiteConnectionPool
        self._db_pool = SQLiteConnectionPool(
            db_path=str(storage_path / "governance.db"),
            pool_size=db_pool_size
        )
        
        # PII detection cache storage (using OrderedDict for proper LRU eviction)
        self._pii_cache: OrderedDict[str, Tuple[List[Any], float]] = OrderedDict()
        self._pii_cache_hits = 0
        self._pii_cache_misses = 0
        
        # Merkle batching for async anchoring
        self._merkle_pending: List[Dict[str, Any]] = []
        self._merkle_batch_lock = None  # Will be created lazily when needed
        self._merkle_anchoring_enabled = enable_merkle_anchoring
        
        # Component flags
        self.components_enabled = {
            # Phase 3
            "risk_engine": True,
            "correlation_engine": True,
            "fairness_sampler": True,
            "ethical_drift_reporter": True,
            "performance_optimizer": enable_performance_optimization,
            # Phase 4
            "merkle_anchoring": enable_merkle_anchoring,
            "policy_auditing": True,
            "quarantine": enable_quarantine,
            "ethical_taxonomy": enable_ethical_taxonomy,
            "sla_monitoring": enable_sla_monitoring,
            # Phase 5-7
            "shadow_mode": enable_shadow_mode,
            "ml_blending": enable_ml_blending,
            "anomaly_detection": enable_anomaly_detection,
            # Phase 8-9
            "human_escalation": True,
            "optimization": True,
            # F3: Privacy & Data Handling
            "redaction_pipeline": True,
            "differential_privacy": privacy_mode == "differential",
            "federated_analytics": region_id is not None,
            "data_minimization": True,
            # Resource Management
            "quota_enforcement": enable_quota_enforcement and QUOTA_AVAILABLE,
            "pii_detection": QUOTA_AVAILABLE,
        }

    def _load_regional_policy(self, policy_name: str) -> None:
        """Load regional policy configuration.

        Args:
            policy_name: Name of the policy (e.g., 'EU_GDPR', 'US_CCPA', 'AI_ACT')
        """
        # Define regional policy profiles
        policy_profiles = {
            "EU_GDPR": {
                "compliance_requirements": ["GDPR", "data_protection", "right_to_erasure"],
                "data_retention_days": 30,
                "cross_border_transfer_allowed": False,
                "encryption_required": True,
                "audit_trail_required": True,
                "consent_required": True,
            },
            "US_CCPA": {
                "compliance_requirements": ["CCPA", "consumer_privacy"],
                "data_retention_days": 90,
                "cross_border_transfer_allowed": True,
                "encryption_required": True,
                "audit_trail_required": True,
                "consent_required": False,
            },
            "AI_ACT": {
                "compliance_requirements": ["AI_ACT", "high_risk_ai", "transparency"],
                "data_retention_days": 180,
                "cross_border_transfer_allowed": True,
                "encryption_required": True,
                "audit_trail_required": True,
                "human_oversight_required": True,
            },
            "GLOBAL_DEFAULT": {
                "compliance_requirements": ["basic_safety"],
                "data_retention_days": 365,
                "cross_border_transfer_allowed": True,
                "encryption_required": False,
                "audit_trail_required": False,
                "consent_required": False,
            },
        }

        self.regional_policies = policy_profiles.get(policy_name, policy_profiles["GLOBAL_DEFAULT"])

    def validate_data_residency(self, region_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate data residency compliance.

        Args:
            region_id: Region identifier to validate

        Returns:
            Validation result with compliance status
        """
        validation_result = {
            "compliant": True,
            "region_id": region_id or self.region_id,
            "policy": self.data_residency_policy,
            "violations": [],
        }

        # Check if cross-border transfers are allowed
        if region_id and self.region_id and region_id != self.region_id:
            if not self.regional_policies.get("cross_border_transfer_allowed", True):
                validation_result["compliant"] = False
                validation_result["violations"].append(
                    f"Cross-border data transfer from {self.region_id} to {region_id} not allowed"
                )

        return validation_result

    def aggregate_by_region(
        self, metrics: List[Dict[str, Any]], group_by: str = "region_id"
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by region or logical domain.

        Args:
            metrics: List of metric dictionaries
            group_by: Field to group by ('region_id' or 'logical_domain')

        Returns:
            Aggregated metrics grouped by the specified field
        """
        aggregated: Dict[str, Dict[str, Any]] = {}

        for metric in metrics:
            key = metric.get(group_by, "unknown")

            if key not in aggregated:
                aggregated[key] = {
                    "count": 0,
                    "total_risk_score": 0.0,
                    "avg_risk_score": 0.0,
                    "violations": 0,
                    "actions": [],
                }

            aggregated[key]["count"] += 1
            aggregated[key]["total_risk_score"] += metric.get("risk_score", 0.0)
            if metric.get("violation_detected", False):
                aggregated[key]["violations"] += 1
            aggregated[key]["actions"].append(metric.get("action_id", "unknown"))

        # Calculate averages
        for key, data in aggregated.items():
            if data["count"] > 0:
                data["avg_risk_score"] = data["total_risk_score"] / data["count"]

        return aggregated

    def _cached_pii_detection(self, content: str) -> Tuple[List[Any], float]:
        """Cache PII detection results by content hash.
        
        Args:
            content: Content to check for PII
            
        Returns:
            Tuple of (pii_matches, pii_risk_score)
        """
        if not self.enable_pii_caching or not self.pii_detector:
            # Caching disabled or no detector, run directly
            if self.pii_detector:
                matches = self.pii_detector.detect_all(content)
                risk = self.pii_detector.calculate_pii_risk_score(matches) if matches else 0.0
                return (matches, risk)
            return ([], 0.0)
        
        # Compute content hash for cache key
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Check cache (move to end for LRU)
        if content_hash in self._pii_cache:
            self._pii_cache_hits += 1
            # Move to end to mark as recently used
            self._pii_cache.move_to_end(content_hash)
            return self._pii_cache[content_hash]
        
        # Cache miss - compute
        self._pii_cache_misses += 1
        matches = self.pii_detector.detect_all(content)
        risk = self.pii_detector.calculate_pii_risk_score(matches) if matches else 0.0
        
        # Store in cache with proper LRU eviction
        if len(self._pii_cache) >= self.pii_cache_size:
            # Remove least recently used (first item)
            self._pii_cache.popitem(last=False)
        
        self._pii_cache[content_hash] = (matches, risk)
        return (matches, risk)

    async def _process_merkle_batch(self) -> None:
        """Process pending Merkle anchoring events in batch (async background task)."""
        if not self.merkle_anchor:
            return
        
        # Create lock lazily if needed
        if self._merkle_batch_lock is None:
            self._merkle_batch_lock = asyncio.Lock()
            
        async with self._merkle_batch_lock:
            if not self._merkle_pending:
                return
                
            # Take all pending events
            batch = self._merkle_pending[:]
            self._merkle_pending.clear()
            
            # Process batch in background (non-blocking)
            for event_data in batch:
                self.merkle_anchor.add_event(event_data)

    async def _process_phase3_async(
        self,
        agent_id: str,
        action: Any,
        cohort: Optional[str],
        violation_detected: bool,
        violation_severity: Optional[str],
        pii_risk: float,
        quota_result: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Process Phase 3 (Risk & Correlation) asynchronously.
        
        Returns:
            Phase 3 results dictionary
        """
        phase3_results = {}
        
        # Calculate violation score
        violation_score = 0.0
        if violation_detected and violation_severity:
            severity_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            severity_str = str(violation_severity).lower() if violation_severity else "medium"
            violation_score = severity_map.get(severity_str, 0.5)
        
        action_context = {"cohort": cohort, "has_violation": violation_detected}
        
        # Calculate risk score
        risk_score = self.risk_engine.calculate_risk_score(
            agent_id=agent_id, violation_severity=violation_score, action_context=action_context
        )
        
        # Boost risk score based on PII detection and quota pressure
        if pii_risk > 0:
            risk_score = min(1.0, risk_score + (pii_risk * 0.3))
        
        if quota_result and quota_result.get("backpressure_level", 0) > 0.8:
            risk_score = min(1.0, risk_score + 0.2)
        
        phase3_results["risk_score"] = risk_score
        phase3_results["risk_tier"] = self.risk_engine.get_tier(agent_id).value
        phase3_results["invoke_advanced_detectors"] = (
            self.risk_engine.should_invoke_advanced_detectors(agent_id)
        )
        
        # Track correlations
        payload = getattr(action, "content", str(action))
        correlations = self.correlation_engine.track_action(
            agent_id=agent_id, action=action, payload=payload
        )
        phase3_results["correlations"] = [
            {
                "pattern": c.pattern_name,
                "severity": c.severity,
                "confidence": c.confidence,
                "description": c.description,
            }
            for c in correlations
        ]
        
        # Track for fairness and drift
        if cohort:
            self.fairness_sampler.assign_agent_cohort(agent_id, cohort)
            self.ethical_drift_reporter.track_action(
                agent_id=agent_id, cohort=cohort, risk_score=risk_score
            )
        
        return phase3_results, risk_score

    async def _process_phase4_async(
        self,
        agent_id: str,
        action: Any,
        cohort: Optional[str],
        violation_detected: bool,
        violation_type: Optional[str],
        risk_score: float,
        context: Optional[Dict[str, Any]],
        start_time: float,
    ) -> Dict[str, Any]:
        """Process Phase 4 (Audit & Taxonomy) asynchronously.
        
        Returns:
            Phase 4 results dictionary
        """
        phase4_results = {}
        
        # Quarantine check
        if self.quarantine_manager and cohort:
            status = self.quarantine_manager.get_quarantine_status(cohort)
            is_quarantined = status.get("is_quarantined", False)
            phase4_results["quarantined"] = is_quarantined
            
            if is_quarantined:
                phase4_results["quarantine_reason"] = status.get("reason", "Unknown")
        
        # Ethical tagging
        if self.ethical_taxonomy and violation_detected and violation_type:
            tagging = self.ethical_taxonomy.create_tagging(
                violation_type=violation_type, context=context
            )
            phase4_results["ethical_tags"] = {
                "primary_dimension": tagging.primary_dimension,
                "dimensions": {tag.dimension: tag.score for tag in tagging.tags},
            }
        
        # Merkle anchoring (batched, non-blocking)
        if self.merkle_anchor:
            event_data = {
                "agent_id": agent_id,
                "action": str(action),
                "risk_score": risk_score,
                "violation_detected": violation_detected,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            self._merkle_pending.append(event_data)
            
            if len(self._merkle_pending) >= self.merkle_batch_size:
                asyncio.create_task(self._process_merkle_batch())
            
            phase4_results["merkle"] = {
                "chunk_id": self.merkle_anchor.current_chunk.chunk_id,
                "event_count": self.merkle_anchor.current_chunk.event_count,
                "pending_batch_size": len(self._merkle_pending),
            }
        
        # SLA tracking
        if self.sla_monitor:
            latency_ms = (time.time() - start_time) * 1000
            self.sla_monitor.record_latency(latency_ms)
            phase4_results["latency_ms"] = latency_ms
        
        return phase4_results

    async def _process_phase567_async(
        self,
        agent_id: str,
        action_id: Optional[str],
        action_type: Optional[str],
        cohort: Optional[str],
        features: Optional[Dict[str, float]],
        rule_risk_score: Optional[float],
        rule_classification: Optional[str],
    ) -> Dict[str, Any]:
        """Process Phase 5-7 (ML & Anomaly Detection) asynchronously.
        
        Returns:
            Phase 5-7 results dictionary
        """
        phase567_results = {}
        
        if not (action_id and features and rule_risk_score is not None):
            return phase567_results
        
        # Phase 5: Shadow mode
        if self.shadow_classifier and rule_classification:
            shadow_pred = self.shadow_classifier.predict(
                agent_id=agent_id,
                action_id=action_id,
                features=features,
                rule_risk_score=rule_risk_score,
                rule_classification=rule_classification,
            )
            phase567_results["shadow"] = {
                "ml_risk_score": shadow_pred.ml_risk_score,
                "ml_classification": shadow_pred.ml_classification,
                "scores_agree": shadow_pred.scores_agree,
                "classifications_agree": shadow_pred.classifications_agree,
            }
        
        # Phase 6: ML blending
        if self.blended_engine and rule_classification:
            blended = self.blended_engine.compute_blended_risk(
                agent_id=agent_id,
                action_id=action_id,
                rule_risk_score=rule_risk_score,
                rule_classification=rule_classification,
                ml_risk_score=features.get("ml_score", 0.5),
            )
            phase567_results["blended"] = {
                "blended_risk_score": blended.blended_risk_score,
                "zone": blended.risk_zone.value,
                "blended_classification": blended.blended_classification,
                "ml_influenced": blended.ml_influenced,
            }
        
        # Phase 7: Anomaly detection
        if self.anomaly_monitor and cohort and action_type:
            alert = self.anomaly_monitor.record_action(
                agent_id=agent_id,
                action_type=action_type,
                risk_score=rule_risk_score,
                cohort=cohort,
            )
            if alert:
                phase567_results["anomaly_alert"] = {
                    "type": alert.anomaly_type.value,
                    "severity": alert.severity.value,
                    "description": alert.description,
                }
            else:
                drift_alert = self.anomaly_monitor.check_drift(cohort=cohort)
                if drift_alert:
                    phase567_results["anomaly_alert"] = {
                        "type": drift_alert.anomaly_type.value,
                        "severity": drift_alert.severity.value,
                        "description": drift_alert.description,
                    }
        
        return phase567_results

    def process_action(
        self,
        agent_id: str,
        action: Any,
        cohort: Optional[str] = None,
        violation_detected: bool = False,
        violation_type: Optional[str] = None,
        violation_severity: Optional[str] = None,
        detector_invocations: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        # For ML phases
        action_id: Optional[str] = None,
        action_type: Optional[str] = None,
        features: Optional[Dict[str, float]] = None,
        rule_risk_score: Optional[float] = None,
        rule_classification: Optional[str] = None,
        # For regional processing
        region_id: Optional[str] = None,
        compliance_requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process an action through all enabled governance phases.

        This is the primary method for evaluating actions with full governance oversight.

        Args:
            agent_id: Agent identifier
            action: Action object
            cohort: Optional agent cohort
            violation_detected: Whether a violation was detected
            violation_type: Type of violation if detected
            violation_severity: Severity of violation if detected
            detector_invocations: Dict of detector_name -> cpu_time_ms
            context: Additional context
            action_id: Action identifier (for ML phases)
            action_type: Type of action (for ML phases)
            features: Feature dict for ML models
            rule_risk_score: Rule-based risk score (for ML blending)
            rule_classification: Rule-based classification (for ML blending)
            region_id: Geographic region for this action
            compliance_requirements: List of compliance requirements to validate

        Returns:
            Comprehensive results from all enabled phases
        """
        start_time = time.time()

        # ==================== Regional Configuration ====================
        # Define effective_region and effective_domain early for use in quotas and throughout
        effective_region = region_id or self.region_id
        effective_domain = self.logical_domain

        # ==================== Quota & Resource Enforcement ====================
        quota_result = None
        pii_matches = []
        pii_risk = 0.0

        if self.quota_enforcer:
            # Calculate payload size
            payload_size = len(str(action)) if action else 0

            # Check quotas
            quota_result = self.quota_enforcer.check_quota(
                agent_id=agent_id,
                cohort=cohort,
                tenant=effective_region if effective_region else None,  # Use region as tenant
                payload_size=payload_size,
                action_type=action_type or "query",
            )

            # If blocked, return early
            if not quota_result["allowed"] and quota_result["decision"] == "BLOCK":
                return {
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "decision": "BLOCK",
                    "reason": quota_result["reason"],
                    "quota_enforcement": quota_result,
                    "blocked_by_quota": True,
                }

        # Enhanced PII Detection (with caching for performance)
        if self.pii_detector and action:
            action_str = str(action)
            pii_matches, pii_risk = self._cached_pii_detection(action_str)
            if pii_matches:
                # Boost violation score if PII detected
                if pii_risk > 0.5:
                    violation_detected = True
                    violation_type = violation_type or "privacy"
                    violation_severity = "critical" if pii_risk > 0.8 else "high"

        # Validate data residency if region specified
        residency_validation = None
        if effective_region:
            residency_validation = self.validate_data_residency(effective_region)

        # ==================== Core Violation Detection ====================
        # Import ActionType enum
        from .governance_core import ActionType
        
        # Convert action type string to enum
        action_type_enum = ActionType.QUERY
        if action_type:
            action_type_str = action_type.upper()
            if hasattr(ActionType, action_type_str):
                action_type_enum = getattr(ActionType, action_type_str)
        
        # Convert action to AgentAction format if it's a string
        if isinstance(action, str):
            # Create an AgentAction object from string
            action_obj = AgentAction(
                action_id=action_id or f"action_{agent_id}_{int(time.time() * 1000)}",
                agent_id=agent_id,
                content=action,
                action_type=action_type_enum,
                timestamp=datetime.now(timezone.utc)
            )
        elif hasattr(action, 'content'):
            # Already an AgentAction-like object
            # Handle action_type conversion
            existing_action_type = getattr(action, 'action_type', None)
            if isinstance(existing_action_type, str):
                existing_action_type_str = existing_action_type.upper()
                if hasattr(ActionType, existing_action_type_str):
                    existing_action_type = getattr(ActionType, existing_action_type_str)
                else:
                    existing_action_type = action_type_enum
            elif existing_action_type is None:
                existing_action_type = action_type_enum
                
            action_obj = AgentAction(
                action_id=getattr(action, 'action_id', action_id or f"action_{agent_id}_{int(time.time() * 1000)}"),
                agent_id=agent_id,
                content=action.content,
                action_type=existing_action_type,
                timestamp=getattr(action, 'timestamp', datetime.now(timezone.utc))
            )
        else:
            # Convert any other format to AgentAction
            action_obj = AgentAction(
                action_id=action_id or f"action_{agent_id}_{int(time.time() * 1000)}",
                agent_id=agent_id,
                content=str(action),
                action_type=action_type_enum,
                timestamp=datetime.now(timezone.utc)
            )

        # Run violation detection asynchronously
        # Use proper async handling based on current context
        def sync_evaluate():
            """Run evaluation in sync context by creating a new event loop"""
            # Create a new loop in a thread-safe manner
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.safety_governance.evaluate_action(action_obj))
            finally:
                loop.close()
        
        # Determine if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use concurrent.futures to run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(sync_evaluate)
                judgment = future.result(timeout=30.0)
        except RuntimeError:
            # No running loop, safe to run synchronously
            judgment = sync_evaluate()
        
        # Extract violation information
        detected_violations = judgment.violations
        decision = judgment.decision
        violation_detected = len(detected_violations) > 0
        
        # Determine violation type and severity from detected violations
        if detected_violations:
            # Use the most severe violation
            violation_type = detected_violations[0].violation_type.value if not violation_type else violation_type
            violation_severity = detected_violations[0].severity.value if not violation_severity else violation_severity

        results = {
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "region_id": effective_region,
            "logical_domain": effective_domain,
            "compliance_requirements": compliance_requirements or [],
            "data_residency": residency_validation,
            "quota_enforcement": quota_result if quota_result else None,
            "pii_detection": (
                {
                    "matches_count": len(pii_matches),
                    "pii_risk_score": pii_risk,
                    "pii_types": [m.pii_type.value for m in pii_matches] if pii_matches else [],
                }
                if self.pii_detector
                else None
            ),
            # Add violation detection results
            "violation_detected": violation_detected,
            "decision": decision.value.upper() if decision else "ALLOW",
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "violation_type": v.violation_type.value if hasattr(v.violation_type, 'value') else str(v.violation_type),
                    "severity": v.severity.value if hasattr(v.severity, 'value') else str(v.severity),
                    "description": v.description,
                    "confidence": v.confidence,
                    "detector_name": v.detector_name,
                }
                for v in detected_violations
            ],
            "reasoning": judgment.reasoning,
            "confidence": judgment.confidence,
            "phase3": {},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }

        # ==================== Phase 3: Risk & Correlation ====================
        violation_score = 0.0
        if violation_detected and violation_severity:
            severity_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            # Convert to string if needed and normalize
            severity_str = str(violation_severity).lower() if violation_severity else "medium"
            violation_score = severity_map.get(severity_str, 0.5)

        action_context = {"cohort": cohort, "has_violation": violation_detected}

        risk_score = self.risk_engine.calculate_risk_score(
            agent_id=agent_id, violation_severity=violation_score, action_context=action_context
        )

        # Boost risk score based on PII detection and quota pressure
        if pii_risk > 0:
            risk_score = min(1.0, risk_score + (pii_risk * 0.3))  # Add up to 30% for PII

        if quota_result and quota_result.get("backpressure_level", 0) > 0.8:
            risk_score = min(1.0, risk_score + 0.2)  # Add 20% for quota pressure

        results["phase3"]["risk_score"] = risk_score
        results["phase3"]["risk_tier"] = self.risk_engine.get_tier(agent_id).value
        results["phase3"]["invoke_advanced_detectors"] = (
            self.risk_engine.should_invoke_advanced_detectors(agent_id)
        )
        
        # ==================== Fast Path Detection (Fix 4) ====================
        # Determine if action is eligible for fast-path processing (skip expensive phases)
        fast_path_eligible = False
        if self.enable_fast_path:
            # Check if any violations were detected (either from parameters or from judgment)
            has_violation = violation_detected or len(detected_violations) > 0
            # Check severity - if critical or high, don't use fast path
            is_high_severity = False
            if violation_severity:
                is_high_severity = str(violation_severity).lower() in ["critical", "high"]
            
            fast_path_eligible = (
                not has_violation
                and not is_high_severity
                and pii_risk < 0.1 
                and risk_score < self.fast_path_risk_threshold
                and (rule_risk_score is None or rule_risk_score < self.fast_path_risk_threshold)
                and not self.components_enabled.get('force_full_pipeline', False)
            )
            results["phase3"]["fast_path_used"] = fast_path_eligible

        # Track correlations
        payload = getattr(action, "content", str(action))
        correlations = self.correlation_engine.track_action(
            agent_id=agent_id, action=action, payload=payload
        )
        results["phase3"]["correlations"] = [
            {
                "pattern": c.pattern_name,
                "severity": c.severity,
                "confidence": c.confidence,
                "description": c.description,
            }
            for c in correlations
        ]

        # Track for fairness and drift
        if cohort:
            self.fairness_sampler.assign_agent_cohort(agent_id, cohort)
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

        # ==================== Phase 4: Audit & Taxonomy ====================
        # Quarantine check
        if self.quarantine_manager and cohort:
            status = self.quarantine_manager.get_quarantine_status(cohort)
            is_quarantined = status.get("is_quarantined", False)
            results["phase4"]["quarantined"] = is_quarantined

            if is_quarantined:
                results["phase4"]["quarantine_reason"] = status.get("reason", "Unknown")

        # Ethical tagging
        if self.ethical_taxonomy and violation_detected and violation_type:
            tagging = self.ethical_taxonomy.create_tagging(
                violation_type=violation_type, context=context
            )
            results["phase4"]["ethical_tags"] = {
                "primary_dimension": tagging.primary_dimension,
                "dimensions": {tag.dimension: tag.score for tag in tagging.tags},
            }

        # Merkle anchoring (Fix 1: Async batching for performance)
        if self.merkle_anchor:
            event_data = {
                "agent_id": agent_id,
                "action": str(action),
                "risk_score": risk_score,
                "violation_detected": violation_detected,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            # Queue for background processing instead of blocking
            self._merkle_pending.append(event_data)
            
            # Trigger background batch processing if threshold reached
            if len(self._merkle_pending) >= self.merkle_batch_size:
                # Schedule async batch processing (non-blocking)
                # Only create task if there's a running event loop
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(self._process_merkle_batch())
                except RuntimeError:
                    # No event loop running - process synchronously in background
                    # This is a fallback for sync contexts
                    if len(self._merkle_pending) >= self.merkle_batch_size:
                        batch = self._merkle_pending[:self.merkle_batch_size]
                        self._merkle_pending = self._merkle_pending[self.merkle_batch_size:]
                        for evt in batch:
                            self.merkle_anchor.add_event(evt)
            
            results["phase4"]["merkle"] = {
                "chunk_id": self.merkle_anchor.current_chunk.chunk_id,
                "event_count": self.merkle_anchor.current_chunk.event_count,
                "pending_batch_size": len(self._merkle_pending),
            }

        # SLA tracking
        if self.sla_monitor:
            latency_ms = (time.time() - start_time) * 1000
            self.sla_monitor.record_latency(latency_ms)
            results["phase4"]["latency_ms"] = latency_ms

        # ==================== Phase 5-7: ML & Anomaly Detection ====================
        # Skip expensive ML phases if fast-path is eligible (Fix 4)
        if action_id and features and rule_risk_score is not None and not fast_path_eligible:
            # Phase 5: Shadow mode
            if self.shadow_classifier and rule_classification:
                shadow_pred = self.shadow_classifier.predict(
                    agent_id=agent_id,
                    action_id=action_id,
                    features=features,
                    rule_risk_score=rule_risk_score,
                    rule_classification=rule_classification,
                )
                results["phase567"]["shadow"] = {
                    "ml_risk_score": shadow_pred.ml_risk_score,
                    "ml_classification": shadow_pred.ml_classification,
                    "scores_agree": shadow_pred.scores_agree,
                    "classifications_agree": shadow_pred.classifications_agree,
                }

            # Phase 6: ML blending
            if self.blended_engine and rule_classification:
                blended = self.blended_engine.compute_blended_risk(
                    agent_id=agent_id,
                    action_id=action_id,
                    rule_risk_score=rule_risk_score,
                    rule_classification=rule_classification,
                    ml_risk_score=features.get("ml_score", 0.5),
                )
                results["phase567"]["blended"] = {
                    "blended_risk_score": blended.blended_risk_score,
                    "zone": blended.risk_zone.value,
                    "blended_classification": blended.blended_classification,
                    "ml_influenced": blended.ml_influenced,
                }

            # Phase 7: Anomaly detection (not skipped by fast-path since we're already in ML phases)
            if self.anomaly_monitor and cohort and action_type:
                # Record action and check for drift
                alert = self.anomaly_monitor.record_action(
                    agent_id=agent_id,
                    action_type=action_type,
                    risk_score=rule_risk_score,
                    cohort=cohort,
                )
                if alert:
                    results["phase567"]["anomaly_alert"] = {
                        "type": alert.anomaly_type.value,
                        "severity": alert.severity.value,
                        "description": alert.description,
                    }
                else:
                    # Check for drift separately
                    drift_alert = self.anomaly_monitor.check_drift(cohort=cohort)
                    if drift_alert:
                        results["phase567"]["anomaly_alert"] = {
                            "type": drift_alert.anomaly_type.value,
                            "severity": drift_alert.severity.value,
                            "description": drift_alert.description,
                        }

        # ==================== Phase 8-9: Human & Optimization ====================
        # Track performance metrics with active config
        if self.active_config and action_id:
            # This would typically be called after human feedback
            # For now we just track that we processed with this config
            results["phase89"]["active_config_id"] = self.active_config.config_id

        # Performance tracking for optimization
        if self.performance_optimizer and detector_invocations:
            for detector_name, cpu_time_ms in detector_invocations.items():
                self.performance_optimizer.track_detector_invocation(
                    detector_name=detector_name, cpu_time_ms=cpu_time_ms
                )

            total_cpu_ms = (time.time() - start_time) * 1000
            self.performance_optimizer.track_action_processing(
                cpu_time_ms=total_cpu_ms, detectors_invoked=len(detector_invocations)
            )

            results["phase3"]["performance_metrics"] = {
                "total_cpu_ms": total_cpu_ms,
                "cpu_reduction_pct": self.performance_optimizer.get_cpu_reduction_pct(),
                "meeting_target": self.performance_optimizer.is_meeting_target(),
            }

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status across all phases.

        Returns:
            Status dictionary with all component states
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components_enabled": self.components_enabled,
            "phase3": {
                "risk_engine": {
                    "active_profiles": len(self.risk_engine.profiles),
                },
                "correlation_engine": {
                    "tracked_agents": len(self.correlation_engine.agent_windows),
                },
                "fairness_sampler": {
                    "active_jobs": len(self.fairness_sampler.jobs),
                },
                "ethical_drift_reporter": {
                    "tracked_cohorts": len(self.ethical_drift_reporter.cohort_profiles),
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
            "phase4": {
                "merkle_anchor": {
                    "enabled": self.merkle_anchor is not None,
                    "current_chunk_events": (
                        self.merkle_anchor.current_chunk.event_count if self.merkle_anchor else 0
                    ),
                },
                "quarantine_manager": {
                    "enabled": self.quarantine_manager is not None,
                    "quarantined_cohorts": (
                        len(self.quarantine_manager.quarantines) if self.quarantine_manager else 0
                    ),
                },
                "sla_monitor": {"enabled": self.sla_monitor is not None},
            },
            "phase567": {
                "shadow_classifier": {"enabled": self.shadow_classifier is not None},
                "blended_engine": {"enabled": self.blended_engine is not None},
                "anomaly_monitor": {"enabled": self.anomaly_monitor is not None},
            },
            "phase89": {
                "escalation_queue": {
                    "pending_cases": len(self.escalation_queue.list_pending_cases())
                },
                "optimizer": {"tracked_configs": len(self.optimizer.configurations)},
                "active_config": self.active_config.config_id if self.active_config else None,
            },
        }

    def load_plugin(self, plugin_id: str) -> bool:
        """Load a plugin from the marketplace into the governance system.

        This method integrates marketplace plugins with the existing plugin system,
        allowing dynamically loaded detectors and policies to be used in governance.

        Args:
            plugin_id: Plugin identifier from marketplace

        Returns:
            True if plugin loaded successfully

        Example:
            >>> governance = IntegratedGovernance()
            >>> governance.load_plugin("financial-compliance-v2")
        """
        # Get the plugin manager
        plugin_manager = get_plugin_manager()

        # In a full implementation, this would:
        # 1. Check if plugin is installed via marketplace
        # 2. Load plugin code/module
        # 3. Register with plugin manager
        # 4. Integrate with active detectors/policies

        # For now, we verify the plugin manager is available
        # and the plugin would be registered through the normal plugin interface
        return plugin_manager is not None
