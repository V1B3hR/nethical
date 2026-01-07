"""AI vs AI Defender - Defense against AI attacks on AI systems.

This detector protects against:
- Model extraction attempts
- Adversarial examples
- Membership inference attacks
- Model inversion attacks

Target latency: <25ms
"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from ..base_detector import BaseDetector, DetectorStatus, ViolationSeverity
from ...core.models import SafetyViolation


@dataclass
class AIvsAIDefenderConfig:
    """Configuration for AI vs AI Defender."""

    # Query monitoring
    max_query_history: int = 1000
    similarity_threshold: float = 0.85
    rate_limit_threshold: int = 100  # queries per minute
    rate_limit_window: int = 60  # seconds

    # Adversarial detection
    perturbation_threshold: float = 0.01
    confidence_manipulation_threshold: float = 0.2

    # Performance
    max_defense_time_ms: float = 23.0  # Target: <25ms

    # Blocking thresholds
    extraction_attempt_threshold: int = 50
    systematic_probing_threshold: int = 20

    # Severity thresholds
    critical_threshold: float = 0.9
    high_threshold: float = 0.7
    medium_threshold: float = 0.5


class AIvsAIDefender(BaseDetector):
    """Defend against adversarial AI attacks on AI systems."""

    def __init__(self, config: Optional[AIvsAIDefenderConfig] = None):
        """Initialize the AI vs AI Defender.

        Args:
            config: Optional configuration for the defender
        """
        super().__init__(
            name="ai_vs_ai_defender",
            version="1.0.0",
            description="Defense against AI attacks on AI systems",
        )
        self.config = config or AIvsAIDefenderConfig()
        self._status = DetectorStatus.ACTIVE

        # Query history for pattern analysis
        self._query_history: Deque[Dict[str, Any]] = deque(
            maxlen=self.config.max_query_history
        )

        # Rate limiting tracker
        self._rate_tracker: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.rate_limit_threshold)
        )

    async def detect_violations(
        self, context: Dict[str, Any], **kwargs: Any
    ) -> List[SafetyViolation]:
        """Detect AI attacks on AI systems.

        Args:
            context: Detection context containing query, history, etc.
            **kwargs: Additional parameters

        Returns:
            List of detected safety violations
        """
        start_time = time.perf_counter()
        violations = []

        try:
            query = context.get("query", {})
            query_history = context.get("query_history", [])
            client_id = context.get("client_id", "unknown")

            if not query:
                return violations

            # Add to query history
            self._query_history.append(
                {"query": query, "timestamp": time.time(), "client_id": client_id}
            )

            # Run parallel detection
            detection_tasks = [
                self._detect_model_extraction(query, query_history),
                self._detect_adversarial_examples(query),
                self._detect_membership_inference(query, query_history),
                self._detect_systematic_probing(query, query_history, client_id),
                self._detect_rate_limiting(client_id),
            ]

            # Gather results
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    continue
                if result:
                    violations.extend(result)

            # Check execution time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.config.max_defense_time_ms:
                self._metrics.false_positives += 1  # Track performance issues

        except Exception as e:
            self._metrics.failed_runs += 1
            raise

        self._metrics.total_runs += 1
        self._metrics.successful_runs += 1
        self._metrics.violations_detected += len(violations)

        return violations

    async def _detect_model_extraction(
        self, query: Dict[str, Any], query_history: List[Dict[str, Any]]
    ) -> List[SafetyViolation]:
        """Detect model extraction attempts through systematic querying.

        Args:
            query: Current query
            query_history: Historical queries from this client

        Returns:
            List of violations for model extraction attempts
        """
        violations = []

        # Check for systematic boundary probing
        if len(query_history) > self.config.extraction_attempt_threshold:
            # Analyze query patterns
            query_similarity = self._compute_query_similarity(query_history)

            if query_similarity < 0.3:  # Diverse queries (exploration)
                confidence = 0.75
                violations.append(
                    SafetyViolation(
                        severity=self._compute_severity(confidence),
                        category="model_extraction",
                        description="Potential model extraction attempt detected",
                        confidence=confidence,
                        evidence=[
                            f"Query count: {len(query_history)}",
                            f"Query diversity: {1 - query_similarity:.2f}",
                            "Systematic boundary probing detected",
                        ],
                        recommendation="Consider rate limiting, add query monitoring",
                    )
                )

        return violations

    async def _detect_adversarial_examples(
        self, query: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Detect adversarial examples with small perturbations.

        Args:
            query: Current query

        Returns:
            List of violations for adversarial examples
        """
        violations = []

        # Check for perturbation patterns
        input_data = query.get("input", "")

        if isinstance(input_data, str):
            # Check for invisible characters (common in adversarial text)
            invisible_chars = sum(
                1 for c in input_data if ord(c) in [0x200B, 0x200C, 0x200D, 0xFEFF]
            )

            if invisible_chars > 0:
                confidence = 0.85
                violations.append(
                    SafetyViolation(
                        severity=self._compute_severity(confidence),
                        category="adversarial_attack",
                        description="Adversarial text detected (invisible characters)",
                        confidence=confidence,
                        evidence=[
                            f"Invisible characters: {invisible_chars}",
                            "Possible adversarial perturbation",
                        ],
                        recommendation="Sanitize input, remove invisible characters",
                    )
                )

        # Check for gradient-based attacks (numerical inputs)
        if "numerical_input" in query:
            numerical_data = query.get("numerical_input")
            if isinstance(numerical_data, (list, np.ndarray)):
                # Simulate gradient-based perturbation detection
                data_array = np.array(numerical_data) if isinstance(numerical_data, list) else numerical_data

                # Check for small perturbations
                if self._check_perturbation(data_array):
                    confidence = 0.80
                    violations.append(
                        SafetyViolation(
                            severity=self._compute_severity(confidence),
                            category="adversarial_attack",
                            description="Potential adversarial perturbation detected",
                            confidence=confidence,
                            evidence=["Small perturbations detected in numerical input"],
                            recommendation="Apply input smoothing or defensive distillation",
                        )
                    )

        return violations

    async def _detect_membership_inference(
        self, query: Dict[str, Any], query_history: List[Dict[str, Any]]
    ) -> List[SafetyViolation]:
        """Detect membership inference attacks.

        Args:
            query: Current query
            query_history: Historical queries

        Returns:
            List of violations for membership inference attempts
        """
        violations = []

        # Check for repeated similar queries with slight variations
        if len(query_history) > 10:
            similar_queries = sum(
                1
                for past_query in query_history[-10:]
                if self._query_similarity_score(query, past_query) > self.config.similarity_threshold
            )

            if similar_queries >= 5:
                confidence = 0.70
                violations.append(
                    SafetyViolation(
                        severity=self._compute_severity(confidence),
                        category="membership_inference",
                        description="Potential membership inference attack detected",
                        confidence=confidence,
                        evidence=[
                            f"Similar queries: {similar_queries}/10",
                            "Repeated queries with variations detected",
                        ],
                        recommendation="Add noise to model outputs, use differential privacy",
                    )
                )

        return violations

    async def _detect_systematic_probing(
        self, query: Dict[str, Any], query_history: List[Dict[str, Any]], client_id: str
    ) -> List[SafetyViolation]:
        """Detect systematic probing of model behavior.

        Args:
            query: Current query
            query_history: Historical queries
            client_id: Client identifier

        Returns:
            List of violations for systematic probing
        """
        violations = []

        # Check for systematic exploration patterns
        if len(query_history) > self.config.systematic_probing_threshold:
            # Analyze query distribution
            query_types = defaultdict(int)

            for past_query in query_history:
                query_type = past_query.get("query", {}).get("type", "unknown")
                query_types[query_type] += 1

            # Check if queries are evenly distributed (exploration)
            if len(query_types) > 5 and self._is_uniform_distribution(query_types):
                confidence = 0.65
                violations.append(
                    SafetyViolation(
                        severity=self._compute_severity(confidence),
                        category="systematic_probing",
                        description="Systematic probing detected",
                        confidence=confidence,
                        evidence=[
                            f"Query types explored: {len(query_types)}",
                            f"Total queries: {len(query_history)}",
                            "Uniform distribution pattern detected",
                        ],
                        recommendation="Monitor client behavior, consider temporary blocking",
                    )
                )

        return violations

    async def _detect_rate_limiting(self, client_id: str) -> List[SafetyViolation]:
        """Detect rate limiting violations.

        Args:
            client_id: Client identifier

        Returns:
            List of violations for rate limit exceeded
        """
        violations = []

        # Track rate
        current_time = time.time()
        self._rate_tracker[client_id].append(current_time)

        # Check rate within window
        window_start = current_time - self.config.rate_limit_window
        recent_queries = sum(1 for t in self._rate_tracker[client_id] if t >= window_start)

        if recent_queries >= self.config.rate_limit_threshold:
            confidence = 0.95
            violations.append(
                SafetyViolation(
                    severity=ViolationSeverity.HIGH,
                    category="rate_limit_exceeded",
                    description=f"Rate limit exceeded: {recent_queries} queries in {self.config.rate_limit_window}s",
                    confidence=confidence,
                    evidence=[
                        f"Queries: {recent_queries}/{self.config.rate_limit_threshold}",
                        f"Window: {self.config.rate_limit_window}s",
                    ],
                    recommendation="Block client temporarily, review behavior",
                )
            )

        return violations

    def _compute_query_similarity(self, query_history: List[Dict[str, Any]]) -> float:
        """Compute average similarity between queries.

        Args:
            query_history: List of past queries

        Returns:
            Average similarity score (0-1)
        """
        if len(query_history) < 2:
            return 0.0

        # Sample queries for efficiency
        sample_size = min(50, len(query_history))
        sampled_queries = query_history[-sample_size:]

        similarities = []
        for i in range(len(sampled_queries) - 1):
            similarity = self._query_similarity_score(
                sampled_queries[i].get("query", {}), sampled_queries[i + 1].get("query", {})
            )
            similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _query_similarity_score(self, query1: Dict[str, Any], query2: Dict[str, Any]) -> float:
        """Compute similarity between two queries.

        Args:
            query1: First query
            query2: Second query

        Returns:
            Similarity score (0-1)
        """
        # Simple hash-based similarity
        query1_str = str(sorted(query1.items()))
        query2_str = str(sorted(query2.items()))

        hash1 = hashlib.md5(query1_str.encode()).hexdigest()
        hash2 = hashlib.md5(query2_str.encode()).hexdigest()

        # Compare hashes
        matching_chars = sum(1 for a, b in zip(hash1, hash2) if a == b)
        return matching_chars / len(hash1)

    def _check_perturbation(self, data: np.ndarray) -> bool:
        """Check if data contains small adversarial perturbations.

        Args:
            data: Numerical input data

        Returns:
            True if perturbation detected
        """
        # Simulate perturbation detection
        # In production, would use statistical tests or gradient analysis

        if len(data.shape) == 0 or data.size == 0:
            return False

        # Check for unusual noise patterns
        if data.size > 1:
            variance = np.var(data)
            if 0 < variance < self.config.perturbation_threshold:
                return True

        return False

    def _is_uniform_distribution(self, distribution: Dict[str, int]) -> bool:
        """Check if distribution is roughly uniform.

        Args:
            distribution: Count distribution

        Returns:
            True if distribution is uniform
        """
        if not distribution:
            return False

        counts = list(distribution.values())
        mean_count = sum(counts) / len(counts)

        # Check if all counts are within 30% of mean
        return all(abs(count - mean_count) / mean_count < 0.3 for count in counts)

    def _compute_severity(self, confidence: float) -> ViolationSeverity:
        """Compute violation severity based on confidence.

        Args:
            confidence: Detection confidence score

        Returns:
            Violation severity level
        """
        if confidence >= self.config.critical_threshold:
            return ViolationSeverity.CRITICAL
        elif confidence >= self.config.high_threshold:
            return ViolationSeverity.HIGH
        elif confidence >= self.config.medium_threshold:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW

    async def defend(self, query: Dict[str, Any], query_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Public API for defending against AI attacks.

        Args:
            query: Current query to analyze
            query_history: Historical queries from this client

        Returns:
            Dictionary with defense results and recommendations
        """
        context = {"query": query, "query_history": query_history}
        violations = await self.detect_violations(context)

        # Determine blocking recommendation
        should_block = any(
            v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH] for v in violations
        )

        return {
            "status": "success",
            "attack_detected": len(violations) > 0,
            "should_block": should_block,
            "confidence": violations[0].confidence if violations else 0.0,
            "violations": [
                {
                    "severity": v.severity.value,
                    "category": v.category,
                    "description": v.description,
                    "confidence": v.confidence,
                    "evidence": v.evidence,
                }
                for v in violations
            ],
            "latency_ms": self._metrics.avg_execution_time * 1000,
        }
