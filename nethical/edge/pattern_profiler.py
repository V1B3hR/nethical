"""
Pattern Profiler - Action Pattern Learning

Tracks and learns common action patterns for predictive pre-computation.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ActionPattern:
    """
    Learned action pattern.

    Attributes:
        pattern_hash: Unique pattern identifier
        action_type: Type of action
        occurrence_count: Number of times seen
        avg_risk_score: Average risk score
        common_decisions: Most common decisions
        contexts: Common context patterns
        first_seen: First occurrence timestamp
        last_seen: Last occurrence timestamp
    """

    pattern_hash: str
    action_type: str
    occurrence_count: int = 0
    avg_risk_score: float = 0.0
    common_decisions: Dict[str, int] = field(default_factory=dict)
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class PatternProfiler:
    """
    Action pattern profiler for predictive caching.

    Features:
    - Track common action patterns per agent type
    - Identify high-frequency decision paths
    - Cluster similar contexts
    - Learn action patterns over time
    """

    def __init__(
        self,
        max_patterns: int = 10000,
        min_occurrences: int = 3,
        pattern_ttl_hours: int = 24,
    ):
        """
        Initialize PatternProfiler.

        Args:
            max_patterns: Maximum patterns to track
            min_occurrences: Minimum occurrences to consider pattern significant
            pattern_ttl_hours: Time-to-live for patterns in hours
        """
        self.max_patterns = max_patterns
        self.min_occurrences = min_occurrences
        self.pattern_ttl_seconds = pattern_ttl_hours * 3600

        # Pattern storage
        self._patterns: Dict[str, ActionPattern] = {}
        self._lock = threading.RLock()

        # Action type clustering
        self._type_clusters: Dict[str, Set[str]] = defaultdict(set)

        # Metrics
        self._total_actions_profiled = 0

        logger.info(
            f"PatternProfiler initialized: max_patterns={max_patterns}, "
            f"min_occurrences={min_occurrences}"
        )

    def record_action(
        self,
        action: str,
        action_type: str,
        context: Optional[Dict[str, Any]] = None,
        decision: Optional[str] = None,
        risk_score: Optional[float] = None,
    ):
        """
        Record an action for pattern learning.

        Args:
            action: The action content
            action_type: Type of action
            context: Action context
            decision: The decision made
            risk_score: Risk score calculated
        """
        from .context_fingerprint import action_similarity_hash

        context = context or {}
        current_time = time.time()

        # Compute pattern hash
        pattern_hash = action_similarity_hash(action, granularity="medium")

        with self._lock:
            self._total_actions_profiled += 1

            if pattern_hash in self._patterns:
                pattern = self._patterns[pattern_hash]
                pattern.occurrence_count += 1
                pattern.last_seen = current_time

                # Update rolling average risk score
                if risk_score is not None:
                    pattern.avg_risk_score = (
                        pattern.avg_risk_score * (pattern.occurrence_count - 1)
                        + risk_score
                    ) / pattern.occurrence_count

                # Track decision distribution
                if decision:
                    pattern.common_decisions[decision] = (
                        pattern.common_decisions.get(decision, 0) + 1
                    )

            else:
                # Create new pattern
                if len(self._patterns) >= self.max_patterns:
                    self._evict_old_patterns()

                pattern = ActionPattern(
                    pattern_hash=pattern_hash,
                    action_type=action_type,
                    occurrence_count=1,
                    avg_risk_score=risk_score or 0.0,
                    common_decisions={decision: 1} if decision else {},
                    contexts=[self._extract_context_features(context)],
                    first_seen=current_time,
                    last_seen=current_time,
                )
                self._patterns[pattern_hash] = pattern

            # Update type cluster
            self._type_clusters[action_type].add(pattern_hash)

    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from context."""
        features = {}
        relevant_keys = [
            "agent_type", "domain", "environment", "user_role", "session_type"
        ]

        for key in relevant_keys:
            if key in context:
                features[key] = context[key]

        return features

    def _evict_old_patterns(self):
        """Evict old or infrequent patterns."""
        current_time = time.time()

        # First, remove expired patterns
        expired = [
            h
            for h, p in self._patterns.items()
            if current_time - p.last_seen > self.pattern_ttl_seconds
        ]
        for h in expired:
            del self._patterns[h]

        # If still full, remove least frequent
        if len(self._patterns) >= self.max_patterns:
            sorted_patterns = sorted(
                self._patterns.items(), key=lambda x: x[1].occurrence_count
            )
            to_remove = len(self._patterns) - self.max_patterns // 2
            for h, _ in sorted_patterns[:to_remove]:
                del self._patterns[h]

    def get_common_patterns(
        self,
        action_type: Optional[str] = None,
        min_count: Optional[int] = None,
    ) -> List[ActionPattern]:
        """
        Get commonly occurring patterns.

        Args:
            action_type: Filter by action type
            min_count: Minimum occurrence count

        Returns:
            List of common patterns
        """
        threshold = min_count or self.min_occurrences

        with self._lock:
            patterns = []
            for pattern in self._patterns.values():
                if pattern.occurrence_count >= threshold:
                    if action_type is None or pattern.action_type == action_type:
                        patterns.append(pattern)

            # Sort by occurrence count
            patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
            return patterns

    def get_frequent_action_types(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequent action types.

        Args:
            top_n: Number of top types to return

        Returns:
            List of action type statistics
        """
        with self._lock:
            type_counts: Dict[str, int] = defaultdict(int)
            type_patterns: Dict[str, List[ActionPattern]] = defaultdict(list)

            for pattern in self._patterns.values():
                type_counts[pattern.action_type] += pattern.occurrence_count
                type_patterns[pattern.action_type].append(pattern)

            sorted_types = sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            return [
                {
                    "action_type": action_type,
                    "total_occurrences": count,
                    "unique_patterns": len(type_patterns[action_type]),
                    "avg_risk_score": (
                        sum(p.avg_risk_score for p in type_patterns[action_type])
                        / len(type_patterns[action_type])
                        if type_patterns[action_type]
                        else 0.0
                    ),
                }
                for action_type, count in sorted_types
            ]

    def predict_decision(
        self, action: str, action_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Predict likely decision based on similar patterns.

        Args:
            action: The action content
            action_type: Type of action

        Returns:
            Prediction with confidence if available
        """
        from .context_fingerprint import action_similarity_hash

        pattern_hash = action_similarity_hash(action, granularity="medium")

        with self._lock:
            if pattern_hash in self._patterns:
                pattern = self._patterns[pattern_hash]

                if pattern.occurrence_count >= self.min_occurrences:
                    # Find most common decision
                    if pattern.common_decisions:
                        most_common = max(
                            pattern.common_decisions.items(), key=lambda x: x[1]
                        )
                        total_decisions = sum(pattern.common_decisions.values())
                        confidence = most_common[1] / total_decisions

                        return {
                            "predicted_decision": most_common[0],
                            "confidence": confidence,
                            "occurrences": pattern.occurrence_count,
                            "avg_risk_score": pattern.avg_risk_score,
                        }

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get profiler metrics."""
        with self._lock:
            significant = sum(
                1
                for p in self._patterns.values()
                if p.occurrence_count >= self.min_occurrences
            )
            return {
                "total_patterns": len(self._patterns),
                "significant_patterns": significant,
                "total_actions_profiled": self._total_actions_profiled,
                "action_types_tracked": len(self._type_clusters),
                "max_patterns": self.max_patterns,
            }

    def export_patterns(self) -> List[Dict[str, Any]]:
        """Export all patterns for persistence."""
        with self._lock:
            return [
                {
                    "pattern_hash": p.pattern_hash,
                    "action_type": p.action_type,
                    "occurrence_count": p.occurrence_count,
                    "avg_risk_score": p.avg_risk_score,
                    "common_decisions": p.common_decisions,
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen,
                }
                for p in self._patterns.values()
            ]

    def import_patterns(self, patterns_data: List[Dict[str, Any]]):
        """Import patterns from persisted data."""
        with self._lock:
            for data in patterns_data:
                pattern = ActionPattern(
                    pattern_hash=data["pattern_hash"],
                    action_type=data["action_type"],
                    occurrence_count=data.get("occurrence_count", 1),
                    avg_risk_score=data.get("avg_risk_score", 0.0),
                    common_decisions=data.get("common_decisions", {}),
                    first_seen=data.get("first_seen", time.time()),
                    last_seen=data.get("last_seen", time.time()),
                )
                self._patterns[pattern.pattern_hash] = pattern
                self._type_clusters[pattern.action_type].add(pattern.pattern_hash)
