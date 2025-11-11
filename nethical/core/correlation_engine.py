"""Correlation Engine for Phase 3.1: Multi-Agent Pattern Detection.

This module implements:
- Multi-agent pattern detection (payload entropy shifts, escalating multi-ID probes)
- Correlation rules from correlation_rules.yaml
- Redis persistence for risk scores
"""

import math
import time
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque


@dataclass
class AgentActivityWindow:
    """Sliding window of agent activities for correlation."""

    agent_id: str
    actions: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)
    payloads: deque = field(default_factory=deque)

    def add_action(self, action: Any, timestamp: datetime, payload: str):
        """Add action to window."""
        self.actions.append(action)
        self.timestamps.append(timestamp)
        self.payloads.append(payload)

    def cleanup_old(self, window_seconds: int):
        """Remove actions older than window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        while self.timestamps and self.timestamps[0] < cutoff:
            self.actions.popleft()
            self.timestamps.popleft()
            self.payloads.popleft()


@dataclass
class CorrelationMatch:
    """Detected correlation pattern match."""

    pattern_name: str
    severity: str
    agent_ids: List[str]
    timestamp: datetime
    confidence: float
    evidence: Dict[str, Any]
    description: str


class CorrelationEngine:
    """Multi-agent pattern detection and correlation engine."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        redis_client=None,
        key_prefix: str = "nethical:correlation",
    ):
        """Initialize correlation engine.

        Args:
            config_path: Path to correlation_rules.yaml
            redis_client: Optional Redis client for persistence
            key_prefix: Redis key prefix
        """
        self.redis = redis_client
        self.key_prefix = key_prefix

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "policies" / "correlation_rules.yaml"
        self.config = self._load_config(config_path)

        # Agent activity windows
        self.agent_windows: Dict[str, AgentActivityWindow] = defaultdict(
            lambda: AgentActivityWindow(agent_id="")
        )

        # Detected patterns
        self.detected_patterns: List[CorrelationMatch] = []

        # Last cleanup time
        self.last_cleanup = time.time()

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load correlation rules configuration."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            # Default minimal config if file not found
            return {
                "multi_agent_patterns": [],
                "correlation_engine": {"enabled": True},
                "persistence": {"redis": {"enabled": True}},
            }

    def track_action(
        self, agent_id: str, action: Any, payload: str, timestamp: Optional[datetime] = None
    ) -> List[CorrelationMatch]:
        """Track an action and check for correlation patterns.

        Args:
            agent_id: Agent identifier
            action: Action object
            payload: Action payload/content
            timestamp: Optional timestamp (defaults to now)

        Returns:
            List of detected correlation patterns
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Add to window
        window = self.agent_windows[agent_id]
        window.agent_id = agent_id
        window.add_action(action, timestamp, payload)

        # Periodic cleanup
        self._periodic_cleanup()

        # Check patterns
        matches = self._check_all_patterns()

        # Persist if Redis available
        if matches:
            self._persist_matches(matches)

        return matches

    def _periodic_cleanup(self):
        """Periodically cleanup old windows."""
        now = time.time()
        cleanup_interval = self.config.get("correlation_engine", {}).get(
            "window_cleanup_interval", 300
        )

        if now - self.last_cleanup > cleanup_interval:
            max_window = max(
                (p.get("window_seconds", 600) for p in self.config.get("multi_agent_patterns", [])),
                default=600,
            )

            for window in self.agent_windows.values():
                window.cleanup_old(max_window)

            self.last_cleanup = now

    def _check_all_patterns(self) -> List[CorrelationMatch]:
        """Check all configured correlation patterns."""
        matches = []

        for pattern in self.config.get("multi_agent_patterns", []):
            if not pattern.get("enabled", True):
                continue

            pattern_name = pattern.get("name", "unknown")

            if pattern_name == "escalating_multi_id_probes":
                match = self._check_escalating_probes(pattern)
            elif pattern_name == "payload_entropy_shift":
                match = self._check_entropy_shift(pattern)
            elif pattern_name == "coordinated_attack":
                match = self._check_coordinated_attack(pattern)
            elif pattern_name == "distributed_reconnaissance":
                match = self._check_distributed_recon(pattern)
            elif pattern_name == "anomalous_agent_cluster":
                match = self._check_agent_cluster(pattern)
            else:
                continue

            if match:
                matches.append(match)

        return matches

    def _check_escalating_probes(self, pattern: Dict) -> Optional[CorrelationMatch]:
        """Detect escalating multi-ID probes."""
        window_seconds = pattern.get("window_seconds", 300)
        min_agents = pattern.get("min_agents", 3)
        min_actions = pattern.get("min_actions_per_agent", 2)

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        # Count actions per agent in window
        agent_counts = defaultdict(int)
        for agent_id, window in self.agent_windows.items():
            count = sum(1 for ts in window.timestamps if ts >= cutoff)
            if count >= min_actions:
                agent_counts[agent_id] = count

        if len(agent_counts) >= min_agents:
            # Check for escalation
            counts = sorted(agent_counts.values())
            if len(counts) >= 2:
                increase_rate = (counts[-1] - counts[0]) / max(counts[0], 1)
                threshold = pattern.get("thresholds", {}).get("action_rate_increase", 0.5)

                if increase_rate >= threshold:
                    return CorrelationMatch(
                        pattern_name="escalating_multi_id_probes",
                        severity=pattern.get("severity", "high"),
                        agent_ids=list(agent_counts.keys()),
                        timestamp=datetime.now(timezone.utc),
                        confidence=min(increase_rate, 1.0),
                        evidence={
                            "agent_count": len(agent_counts),
                            "action_counts": dict(agent_counts),
                            "escalation_rate": increase_rate,
                        },
                        description=f"Detected {len(agent_counts)} agents with escalating probe pattern",
                    )

        return None

    def _check_entropy_shift(self, pattern: Dict) -> Optional[CorrelationMatch]:
        """Detect payload entropy shifts across agents."""
        window_seconds = pattern.get("window_seconds", 600)
        min_samples = pattern.get("min_samples", 5)

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        # Collect recent payloads
        all_payloads = []
        agent_ids = []

        for agent_id, window in self.agent_windows.items():
            for ts, payload in zip(window.timestamps, window.payloads):
                if ts >= cutoff:
                    all_payloads.append(payload)
                    agent_ids.append(agent_id)

        if len(all_payloads) >= min_samples:
            # Calculate entropy for each payload
            entropies = [self._calculate_entropy(p) for p in all_payloads]

            if len(entropies) >= 2:
                avg_entropy = sum(entropies) / len(entropies)
                max_delta = max(abs(e - avg_entropy) for e in entropies)

                threshold = pattern.get("thresholds", {}).get("entropy_delta", 0.3)
                min_ent = pattern.get("thresholds", {}).get("min_entropy", 2.0)
                max_ent = pattern.get("thresholds", {}).get("max_entropy", 7.0)

                if max_delta >= threshold and min_ent <= avg_entropy <= max_ent:
                    return CorrelationMatch(
                        pattern_name="payload_entropy_shift",
                        severity=pattern.get("severity", "medium"),
                        agent_ids=list(set(agent_ids)),
                        timestamp=datetime.now(timezone.utc),
                        confidence=min(max_delta / threshold, 1.0),
                        evidence={
                            "avg_entropy": avg_entropy,
                            "max_delta": max_delta,
                            "sample_count": len(entropies),
                        },
                        description=f"Payload entropy shift detected across {len(set(agent_ids))} agents",
                    )

        return None

    def _check_coordinated_attack(self, pattern: Dict) -> Optional[CorrelationMatch]:
        """Detect coordinated multi-agent attacks."""
        window_seconds = pattern.get("window_seconds", 180)
        min_agents = pattern.get("min_agents", 2)

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        # Group actions by time proximity
        time_clusters = []
        time_threshold = pattern.get("thresholds", {}).get("time_correlation_threshold", 30)

        for agent_id, window in self.agent_windows.items():
            for ts, action in zip(window.timestamps, window.actions):
                if ts >= cutoff:
                    # Find or create time cluster
                    found = False
                    for cluster in time_clusters:
                        cluster_time = cluster["time"]
                        if abs((ts - cluster_time).total_seconds()) <= time_threshold:
                            cluster["agents"].add(agent_id)
                            cluster["actions"].append(action)
                            found = True
                            break

                    if not found:
                        time_clusters.append(
                            {"time": ts, "agents": {agent_id}, "actions": [action]}
                        )

        # Check for coordinated clusters
        for cluster in time_clusters:
            if len(cluster["agents"]) >= min_agents:
                return CorrelationMatch(
                    pattern_name="coordinated_attack",
                    severity=pattern.get("severity", "critical"),
                    agent_ids=list(cluster["agents"]),
                    timestamp=cluster["time"],
                    confidence=min(len(cluster["agents"]) / (min_agents * 2), 1.0),
                    evidence={
                        "agent_count": len(cluster["agents"]),
                        "action_count": len(cluster["actions"]),
                        "time_window": time_threshold,
                    },
                    description=f"Coordinated attack detected with {len(cluster['agents'])} agents",
                )

        return None

    def _check_distributed_recon(self, pattern: Dict) -> Optional[CorrelationMatch]:
        """Detect distributed reconnaissance patterns."""
        window_seconds = pattern.get("window_seconds", 900)
        min_agents = pattern.get("min_agents", 3)

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        # Track unique targets/endpoints
        agent_targets = defaultdict(set)

        for agent_id, window in self.agent_windows.items():
            for ts, action in zip(window.timestamps, window.actions):
                if ts >= cutoff:
                    # Extract targets from action context
                    target = getattr(action, "metadata", {}).get("target", "unknown")
                    agent_targets[agent_id].add(target)

        if len(agent_targets) >= min_agents:
            all_targets = set()
            for targets in agent_targets.values():
                all_targets.update(targets)

            unique_threshold = pattern.get("thresholds", {}).get("unique_endpoints", 5)

            if len(all_targets) >= unique_threshold:
                return CorrelationMatch(
                    pattern_name="distributed_reconnaissance",
                    severity=pattern.get("severity", "high"),
                    agent_ids=list(agent_targets.keys()),
                    timestamp=datetime.now(timezone.utc),
                    confidence=min(len(all_targets) / (unique_threshold * 2), 1.0),
                    evidence={
                        "agent_count": len(agent_targets),
                        "unique_targets": len(all_targets),
                        "targets": list(all_targets)[:10],  # Sample
                    },
                    description=f"Distributed reconnaissance across {len(all_targets)} targets",
                )

        return None

    def _check_agent_cluster(self, pattern: Dict) -> Optional[CorrelationMatch]:
        """Detect anomalous agent clusters."""
        window_seconds = pattern.get("window_seconds", 600)
        min_cluster = pattern.get("min_cluster_size", 4)

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        # Calculate behavior signatures for each agent
        agent_signatures = {}

        for agent_id, window in self.agent_windows.items():
            recent_actions = [
                action for ts, action in zip(window.timestamps, window.actions) if ts >= cutoff
            ]

            if recent_actions:
                signature = self._calculate_behavior_signature(recent_actions)
                agent_signatures[agent_id] = signature

        # Find clusters of similar agents
        if len(agent_signatures) >= min_cluster:
            clusters = self._cluster_agents(agent_signatures)

            for cluster in clusters:
                if len(cluster) >= min_cluster:
                    return CorrelationMatch(
                        pattern_name="anomalous_agent_cluster",
                        severity=pattern.get("severity", "high"),
                        agent_ids=cluster,
                        timestamp=datetime.now(timezone.utc),
                        confidence=len(cluster) / (min_cluster * 2),
                        evidence={
                            "cluster_size": len(cluster),
                            "total_agents": len(agent_signatures),
                        },
                        description=f"Anomalous cluster of {len(cluster)} agents detected",
                    )

        return None

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        # Character frequency
        char_freq = defaultdict(int)
        for char in text:
            char_freq[char] += 1

        # Calculate entropy
        length = len(text)
        entropy = 0.0

        for count in char_freq.values():
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)

        return entropy

    def _calculate_behavior_signature(self, actions: List[Any]) -> Dict[str, float]:
        """Calculate behavior signature for actions."""
        signature = {
            "action_count": len(actions),
            "avg_payload_length": 0.0,
            "avg_entropy": 0.0,
        }

        if actions:
            total_length = 0
            total_entropy = 0.0

            for action in actions:
                content = getattr(action, "content", "")
                total_length += len(content)
                total_entropy += self._calculate_entropy(content)

            signature["avg_payload_length"] = total_length / len(actions)
            signature["avg_entropy"] = total_entropy / len(actions)

        return signature

    def _cluster_agents(self, signatures: Dict[str, Dict[str, float]]) -> List[List[str]]:
        """Simple clustering of agents by behavior similarity."""
        agents = list(signatures.keys())
        clusters = []
        used = set()

        for i, agent1 in enumerate(agents):
            if agent1 in used:
                continue

            cluster = [agent1]
            used.add(agent1)

            for agent2 in agents[i + 1 :]:
                if agent2 in used:
                    continue

                # Calculate similarity
                sig1 = signatures[agent1]
                sig2 = signatures[agent2]

                similarity = self._signature_similarity(sig1, sig2)

                if similarity > 0.7:  # Threshold
                    cluster.append(agent2)
                    used.add(agent2)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _signature_similarity(self, sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
        """Calculate similarity between two behavior signatures."""
        # Simple cosine-like similarity
        if not sig1 or not sig2:
            return 0.0

        total_diff = 0.0
        count = 0

        for key in sig1:
            if key in sig2:
                val1 = sig1[key]
                val2 = sig2[key]

                if val1 + val2 > 0:
                    diff = abs(val1 - val2) / (val1 + val2)
                    total_diff += diff
                    count += 1

        if count == 0:
            return 0.0

        avg_diff = total_diff / count
        similarity = 1.0 - avg_diff

        return max(0.0, similarity)

    def _persist_matches(self, matches: List[CorrelationMatch]):
        """Persist correlation matches to Redis."""
        if not self.redis:
            return

        for match in matches:
            try:
                key = f"{self.key_prefix}:match:{match.pattern_name}:{int(time.time())}"
                data = {
                    "pattern_name": match.pattern_name,
                    "severity": match.severity,
                    "agent_ids": match.agent_ids,
                    "timestamp": match.timestamp.isoformat(),
                    "confidence": match.confidence,
                    "evidence": match.evidence,
                    "description": match.description,
                }
                self.redis.setex(key, 86400, json.dumps(data))  # 24 hour TTL
            except Exception:
                pass  # Silent fail

    def get_recent_matches(
        self, pattern_name: Optional[str] = None, limit: int = 100
    ) -> List[CorrelationMatch]:
        """Get recent correlation matches.

        Args:
            pattern_name: Optional filter by pattern name
            limit: Maximum number of matches to return

        Returns:
            List of correlation matches
        """
        if pattern_name:
            return [m for m in self.detected_patterns[-limit:] if m.pattern_name == pattern_name]
        return self.detected_patterns[-limit:]
