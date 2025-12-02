"""
Local Governor - Core Edge Governance Engine

Ultra-low latency governance for edge deployment.
Designed for autonomous vehicles, robots, and real-time systems.

Target: <10ms p99 latency
Mode: Offline-first with sync

Features:
- In-memory policy cache (no I/O)
- Pre-compiled decision rules
- Local risk scoring
- Safe default fallbacks
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Decision types for edge governance."""

    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"
    BLOCK = "BLOCK"
    TERMINATE = "TERMINATE"


@dataclass
class EdgeDecision:
    """
    Edge governance decision result.

    Attributes:
        decision: The decision type (ALLOW, RESTRICT, BLOCK, TERMINATE)
        risk_score: Calculated risk score (0.0-1.0)
        latency_ms: Time taken to make decision in milliseconds
        violations: List of detected violations
        from_cache: Whether decision came from cache
        confidence: Confidence level (0.0-1.0)
        context_hash: Hash of context used for decision
        metadata: Additional metadata
    """

    decision: DecisionType
    risk_score: float
    latency_ms: float
    violations: List[str] = field(default_factory=list)
    from_cache: bool = False
    confidence: float = 1.0
    context_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class EdgeGovernor:
    """
    Ultra-low latency governance for edge deployment.

    Designed for autonomous vehicles, robots, and real-time systems.

    Target: <10ms p99 latency
    Mode: Offline-first with sync

    Features:
    - In-memory policy cache (no I/O)
    - Pre-compiled decision rules
    - Local risk scoring
    - Safe default fallbacks
    """

    # Risk score thresholds for decision types
    RISK_THRESHOLD_ALLOW = 0.3
    RISK_THRESHOLD_RESTRICT = 0.6
    RISK_THRESHOLD_BLOCK = 0.8

    def __init__(
        self,
        agent_id: str,
        policy_cache: Optional["PolicyCache"] = None,
        fast_detector: Optional["FastDetector"] = None,
        safe_defaults: Optional["SafeDefaults"] = None,
        predictive_engine: Optional["PredictiveEngine"] = None,
        offline_fallback: Optional["OfflineFallback"] = None,
        circuit_breaker: Optional["CircuitBreaker"] = None,
        max_latency_ms: float = 10.0,
    ):
        """
        Initialize EdgeGovernor.

        Args:
            agent_id: Unique identifier for this agent
            policy_cache: In-memory policy cache
            fast_detector: Lightweight detector for edge
            safe_defaults: Safe default decisions
            predictive_engine: Pre-computation engine
            offline_fallback: Offline mode handler
            circuit_breaker: Latency circuit breaker
            max_latency_ms: Maximum allowed latency in milliseconds
        """
        self.agent_id = agent_id
        self.max_latency_ms = max_latency_ms

        # Import here to avoid circular imports
        from .policy_cache import PolicyCache
        from .fast_detector import FastDetector
        from .safe_defaults import SafeDefaults
        from .predictive_engine import PredictiveEngine
        from .offline_fallback import OfflineFallback
        from .circuit_breaker import CircuitBreaker

        self.policy_cache = policy_cache or PolicyCache()
        self.fast_detector = fast_detector or FastDetector()
        self.safe_defaults = safe_defaults or SafeDefaults()
        self.predictive_engine = predictive_engine or PredictiveEngine()
        self.offline_fallback = offline_fallback or OfflineFallback()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            max_latency_ms=max_latency_ms
        )

        # Decision history for pattern learning
        self._decision_history: List[EdgeDecision] = []
        self._max_history = 1000

        # Performance metrics
        self._total_decisions = 0
        self._cache_hits = 0
        self._latency_samples: List[float] = []

        logger.info(f"EdgeGovernor initialized for agent {agent_id}")

    def evaluate(
        self,
        action: str,
        action_type: str,
        context: Optional[Dict[str, Any]] = None,
        require_cache: bool = False,
    ) -> EdgeDecision:
        """
        Evaluate an action with ultra-low latency.

        Target: <5ms p50, <10ms p99

        Args:
            action: The action to evaluate
            action_type: Type of action (code_generation, data_access, etc.)
            context: Additional context for evaluation
            require_cache: If True, only return cached decisions

        Returns:
            EdgeDecision with result and metrics
        """
        start_time = time.perf_counter()
        context = context or {}

        try:
            # Check circuit breaker
            if not self.circuit_breaker.can_process():
                return self._make_safe_decision(
                    action, context, start_time, reason="circuit_open"
                )

            # Compute context fingerprint for cache lookup
            from .context_fingerprint import compute_fingerprint

            context_hash = compute_fingerprint(action, action_type, context)

            # Try predictive cache first (0ms for cache hits)
            cached_decision = self.predictive_engine.get_cached_decision(context_hash)
            if cached_decision is not None:
                self._cache_hits += 1
                latency_ms = (time.perf_counter() - start_time) * 1000
                cached_decision.latency_ms = latency_ms
                cached_decision.from_cache = True
                self._record_decision(cached_decision)
                return cached_decision

            # If require_cache is True and no cache hit, return safe default
            if require_cache:
                return self._make_safe_decision(
                    action, context, start_time, reason="cache_miss"
                )

            # Fast detection path
            detection_result = self.fast_detector.detect(action, action_type, context)

            # Calculate risk score
            risk_score = self._calculate_risk_score(detection_result)

            # Determine decision
            decision_type = self._determine_decision(risk_score, detection_result)

            # Build result
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Calculate confidence from detection result
            confidence = 1.0
            if detection_result and detection_result.confidences and len(detection_result.confidences) > 0:
                confidence = sum(detection_result.confidences) / len(detection_result.confidences)
            
            decision = EdgeDecision(
                decision=decision_type,
                risk_score=risk_score,
                latency_ms=latency_ms,
                violations=detection_result.violations if detection_result else [],
                from_cache=False,
                confidence=confidence,
                context_hash=context_hash,
                metadata={
                    "agent_id": self.agent_id,
                    "action_type": action_type,
                    "detection_categories": (
                        detection_result.categories if detection_result else []
                    ),
                },
            )

            # Cache for future use
            self.predictive_engine.cache_decision(context_hash, decision)

            # Record for pattern learning
            self._record_decision(decision)

            # Update circuit breaker
            self.circuit_breaker.record_latency(latency_ms)

            return decision

        except Exception as e:
            logger.error(f"EdgeGovernor evaluation error: {e}")
            return self._make_safe_decision(
                action, context, start_time, reason=f"error:{str(e)}"
            )

    def batch_evaluate(
        self, actions: List[Dict[str, Any]], parallel: bool = True
    ) -> List[EdgeDecision]:
        """
        Evaluate multiple actions in batch.

        Args:
            actions: List of action dicts with 'action', 'action_type', 'context'
            parallel: Whether to process in parallel (if available)

        Returns:
            List of EdgeDecision results
        """
        results = []
        for action_dict in actions:
            result = self.evaluate(
                action=action_dict.get("action", ""),
                action_type=action_dict.get("action_type", "unknown"),
                context=action_dict.get("context"),
            )
            results.append(result)
        return results

    def _calculate_risk_score(self, detection_result: Optional["DetectionResult"]) -> float:
        """Calculate risk score from detection result."""
        if detection_result is None:
            return 0.0

        # Use JIT-optimized calculation if available
        try:
            from ..core.jit_optimizations import calculate_risk_score_jit, NUMBA_AVAILABLE

            if NUMBA_AVAILABLE and detection_result.severities:
                severities = np.array(detection_result.severities, dtype=np.float64)
                confidences = np.array(detection_result.confidences, dtype=np.float64)
                return calculate_risk_score_jit(severities, confidences)
        except ImportError:
            pass

        # Fallback to simple calculation
        if not detection_result.severities:
            return 0.0

        weighted_sum = 0.0
        for sev, conf in zip(detection_result.severities, detection_result.confidences):
            weighted_sum += (sev / 5.0) * conf

        return min(1.0, weighted_sum / len(detection_result.severities))

    def _determine_decision(
        self, risk_score: float, detection_result: Optional["DetectionResult"]
    ) -> DecisionType:
        """Determine decision type based on risk score."""
        # Check for critical violations
        if detection_result and detection_result.has_critical:
            return DecisionType.TERMINATE

        # Use risk thresholds
        if risk_score >= self.RISK_THRESHOLD_BLOCK:
            return DecisionType.BLOCK
        elif risk_score >= self.RISK_THRESHOLD_RESTRICT:
            return DecisionType.RESTRICT
        elif risk_score >= self.RISK_THRESHOLD_ALLOW:
            return DecisionType.RESTRICT  # Borderline cases get restricted
        else:
            return DecisionType.ALLOW

    def _make_safe_decision(
        self,
        action: str,
        context: Dict[str, Any],
        start_time: float,
        reason: str,
    ) -> EdgeDecision:
        """Make a safe default decision."""
        default = self.safe_defaults.get_default(action, context)
        latency_ms = (time.perf_counter() - start_time) * 1000

        decision = EdgeDecision(
            decision=default.decision,
            risk_score=default.risk_score,
            latency_ms=latency_ms,
            violations=[],
            from_cache=False,
            confidence=default.confidence,
            context_hash="",
            metadata={"safe_default": True, "reason": reason},
        )

        self._record_decision(decision)
        return decision

    def _record_decision(self, decision: EdgeDecision):
        """Record decision for metrics and pattern learning."""
        self._total_decisions += 1
        self._latency_samples.append(decision.latency_ms)

        # Trim latency samples
        if len(self._latency_samples) > 10000:
            self._latency_samples = self._latency_samples[-5000:]

        # Add to history for pattern learning
        self._decision_history.append(decision)
        if len(self._decision_history) > self._max_history:
            self._decision_history.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self._latency_samples:
            return {
                "total_decisions": 0,
                "cache_hit_rate": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }

        sorted_latencies = sorted(self._latency_samples)
        n = len(sorted_latencies)

        return {
            "total_decisions": self._total_decisions,
            "cache_hit_rate": (
                self._cache_hits / self._total_decisions
                if self._total_decisions > 0
                else 0.0
            ),
            "p50_latency_ms": sorted_latencies[int(n * 0.50)],
            "p95_latency_ms": sorted_latencies[int(n * 0.95)],
            "p99_latency_ms": sorted_latencies[int(n * 0.99)],
            "avg_latency_ms": sum(sorted_latencies) / n,
            "min_latency_ms": min(sorted_latencies),
            "max_latency_ms": max(sorted_latencies),
        }

    def warmup(self, common_actions: Optional[List[Dict[str, Any]]] = None):
        """
        Warmup the governor by pre-computing common decisions.

        Args:
            common_actions: List of common action patterns to pre-compute
        """
        logger.info(f"Warming up EdgeGovernor for agent {self.agent_id}")

        # Warmup JIT compilation
        try:
            from ..core.jit_optimizations import NUMBA_AVAILABLE

            if NUMBA_AVAILABLE:
                # Trigger JIT compilation
                test_severities = np.array([1.0, 2.0, 3.0])
                test_confidences = np.array([0.8, 0.9, 0.7])
                from ..core.jit_optimizations import calculate_risk_score_jit

                _ = calculate_risk_score_jit(test_severities, test_confidences)
                logger.info("JIT warmup complete")
        except ImportError:
            pass

        # Pre-compute common action decisions
        if common_actions:
            for action_dict in common_actions:
                _ = self.evaluate(
                    action=action_dict.get("action", ""),
                    action_type=action_dict.get("action_type", "unknown"),
                    context=action_dict.get("context"),
                )
            logger.info(f"Pre-computed {len(common_actions)} common decisions")


# Import dependencies for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .policy_cache import PolicyCache
    from .fast_detector import FastDetector, DetectionResult
    from .safe_defaults import SafeDefaults
    from .predictive_engine import PredictiveEngine
    from .offline_fallback import OfflineFallback
    from .circuit_breaker import CircuitBreaker
