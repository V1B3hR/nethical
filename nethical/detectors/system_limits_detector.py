"""System limits detection for volume attacks and resource exhaustion."""

import time
import tracemalloc
import psutil
import asyncio
import logging
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .base_detector import BaseDetector
from ..core.governance import AgentAction, SafetyViolation, ViolationType, Severity

logger = logging.getLogger(__name__)

class SystemLimitsDetector(BaseDetector):
    """Advanced detector for volume attacks and resource exhaustion attempts."""

    def __init__(self):
        super().__init__("SystemLimitsDetector")

        # Rate limiting parameters (adaptive)
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_window = 50
        self.max_payload_size = 100_000  # characters
        self.max_nested_structures = 10

        # Adaptive thresholds for memory/CPU based on rolling stats
        self.memory_threshold = 0.85
        self.cpu_threshold = 0.90
        self.memory_window = deque(maxlen=30)
        self.cpu_window = deque(maxlen=30)

        # Request tracking per agent
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.payload_sizes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.violation_stats: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self.agent_reputation: Dict[str, float] = defaultdict(lambda: 1.0)

        # Suspicious patterns
        self.spam_patterns = [
            "a" * 1000,
            "test" * 100,
            "x" * 500,
        ]

        # Resource exhaustion indicators
        self.exhaustion_patterns = [
            r"(?:very\s+){10,}",
            r"(?:\w+\s+){100,}",
            r"\d{1000,}",
            r"[^\w\s]{100,}",
            r"(?:null\s*){100,}",
            r"(?:undefined\s*){100,}",
        ]

        self._compiled_exhaustion_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.exhaustion_patterns
        ]
        self._nesting_regex = re.compile(r"(\({3,}|\[{3,}|\{{3,})")

        tracemalloc.start()

        # For statistical anomaly detection
        self._payload_zscore_threshold = 3.0
        self._request_rate_zscore_threshold = 3.0

        # External hooks (e.g., alerting, telemetry)
        self.external_alert_hook = None

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []

        sys_violations = await self._detect_system_resource_violations()
        violations.extend(sys_violations)

        violations.extend(self._detect_volume_attacks(action))
        violations.extend(self._detect_resource_exhaustion(action))
        violations.extend(self._detect_large_payloads(action))
        violations.extend(self._detect_nested_structure_attacks(action))
        violations.extend(self._detect_behavior_anomaly(action))
        violations.extend(self._detect_statistical_anomalies(action))

        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()
            self.violation_stats[action.agent_id].append((self.last_detection_time, action.id))
            self._update_agent_reputation(action.agent_id, violations)
            logger.warning(f"Detected {len(violations)} violations for agent {action.agent_id}")
            self._trigger_external_alerts(action, violations)

        return violations

    def _update_agent_reputation(self, agent_id: str, violations: List[SafetyViolation]):
        """Reduce agent reputation score based on severity of violations."""
        rep = self.agent_reputation[agent_id]
        for v in violations:
            if v.severity == Severity.CRITICAL:
                rep *= 0.5
            elif v.severity == Severity.HIGH:
                rep *= 0.7
            elif v.severity == Severity.MEDIUM:
                rep *= 0.85
        self.agent_reputation[agent_id] = max(rep, 0.01)

    def _trigger_external_alerts(self, action: AgentAction, violations: List[SafetyViolation]):
        """Hook for external alerting/telemetry integration."""
        if self.external_alert_hook:
            try:
                self.external_alert_hook(agent_action=action, violations=violations)
            except Exception as e:
                logger.error(f"Alert hook failed: {e}")

    def _detect_volume_attacks(self, action: AgentAction) -> List[SafetyViolation]:
        violations = []
        agent_id = action.agent_id
        current_time = time.time()
        self.request_history[agent_id].append(current_time)
        window_start = current_time - self.rate_limit_window
        recent_requests = [t for t in self.request_history[agent_id] if t >= window_start]
        request_rate = len(recent_requests) / self.rate_limit_window

        if len(recent_requests) > self.max_requests_per_window:
            severity = Severity.CRITICAL if request_rate > 5.0 else (
                Severity.HIGH if request_rate > 2.0 else Severity.MEDIUM)

            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.SECURITY,
                severity=severity,
                description=f"Volume attack detected: {len(recent_requests)} requests in {self.rate_limit_window}s window",
                confidence=0.95 if severity == Severity.CRITICAL else 0.90,
                evidence=[f"Request rate: {request_rate:.2f} requests/second"],
                recommendations=["Rate limit agent", "Block excessive requests", "Monitor for DoS"],
                detector_name=self.name
            )
            violations.append(violation)
        return violations

    def _detect_resource_exhaustion(self, action: AgentAction) -> List[SafetyViolation]:
        violations = []
        content = action.actual_action
        matches = []
        for pattern in self._compiled_exhaustion_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
        if matches:
            severity = Severity.CRITICAL if len(matches) > 5 else (
                Severity.HIGH if len(matches) > 2 else Severity.MEDIUM)
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=severity,
                description="Resource exhaustion attempt detected",
                confidence=0.85 + min(0.15 * len(matches), 0.15),
                evidence=[f"Exhaustion patterns found: {len(matches)}", f"Patterns: {matches[:3]}..."],
                recommendations=["Limit processing resources", "Block action", "Monitor system resources"],
                detector_name=self.name
            )
            violations.append(violation)
        for spam_pattern in self.spam_patterns:
            if spam_pattern in content:
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.ADVERSARIAL,
                    severity=Severity.HIGH,
                    description="Spam/repetitive content detected - potential resource exhaustion",
                    confidence=0.80,
                    evidence=[f"Spam pattern detected: snippet '{spam_pattern[:20]}...'"],
                    recommendations=["Block repetitive content", "Implement content filtering"],
                    detector_name=self.name
                )
                violations.append(violation)
                break
        return violations

    def _detect_large_payloads(self, action: AgentAction) -> List[SafetyViolation]:
        violations = []
        payload_size = len(action.actual_action)
        agent_id = action.agent_id
        self.payload_sizes[agent_id].append(payload_size)
        if payload_size > self.max_payload_size:
            size_ratio = payload_size / self.max_payload_size
            severity = Severity.CRITICAL if size_ratio > 10 else Severity.HIGH
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.SECURITY,
                severity=severity,
                description=f"Large payload detected: {payload_size} characters (limit: {self.max_payload_size})",
                confidence=0.98 if severity == Severity.CRITICAL else 0.95,
                evidence=[f"Payload size: {payload_size:,} characters"],
                recommendations=["Limit payload size", "Reject large requests", "Monitor memory usage"],
                detector_name=self.name
            )
            violations.append(violation)
        if len(self.payload_sizes[agent_id]) >= 6:
            recent_sizes = list(self.payload_sizes[agent_id])[-6:]
            if all(recent_sizes[i] < recent_sizes[i+1] for i in range(len(recent_sizes)-1)):
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.ADVERSARIAL,
                    severity=Severity.HIGH,
                    description="Escalating payload size pattern detected",
                    confidence=0.80,
                    evidence=[f"Payload size progression: {recent_sizes}"],
                    recommendations=["Monitor agent behavior", "Implement size limits"],
                    detector_name=self.name
                )
                violations.append(violation)
        return violations

    def _detect_nested_structure_attacks(self, action: AgentAction) -> List[SafetyViolation]:
        violations = []
        content = action.actual_action
        max_nesting = self._estimate_nesting_depth(content)
        if max_nesting > self.max_nested_structures:
            severity = Severity.CRITICAL if max_nesting > 30 else (
                Severity.HIGH if max_nesting > 20 else Severity.MEDIUM)
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=severity,
                description=f"Deep nesting detected: {max_nesting} levels (limit: {self.max_nested_structures})",
                confidence=0.95 if severity == Severity.CRITICAL else 0.90,
                evidence=[f"Maximum nesting depth: {max_nesting}"],
                recommendations=["Limit nesting depth", "Reject deeply nested structures", "Protect against stack overflow"],
                detector_name=self.name
            )
            violations.append(violation)
        return violations

    def _estimate_nesting_depth(self, content: str) -> int:
        max_nesting = 0
        current_nesting = 0
        for char in content:
            if char in "([{":
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char in ")]}":
                current_nesting = max(0, current_nesting - 1)
        nest_matches = self._nesting_regex.findall(content)
        if nest_matches:
            max_nesting = max(max_nesting, max(len(match) for match in nest_matches))
        return max_nesting

    async def _detect_system_resource_violations(self) -> List[SafetyViolation]:
        violations = []
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.05) / 100.0
        self.memory_window.append(mem.percent / 100.0)
        self.cpu_window.append(cpu)
        # Adaptive thresholds (90th percentile)
        if len(self.memory_window) > 10:
            adaptive_mem_threshold = np.percentile(self.memory_window, 90)
            adaptive_cpu_threshold = np.percentile(self.cpu_window, 90)
        else:
            adaptive_mem_threshold = self.memory_threshold
            adaptive_cpu_threshold = self.cpu_threshold

        if mem.percent / 100.0 > adaptive_mem_threshold:
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=None,
                violation_type=ViolationType.SYSTEM,
                severity=Severity.CRITICAL,
                description=f"System memory exhaustion: {mem.percent:.1f}% used (threshold: {adaptive_mem_threshold*100:.1f}%)",
                confidence=0.99,
                evidence=[f"Memory usage: {mem.percent:.1f}%"],
                recommendations=["Free up memory", "Throttle new requests", "Restart processes"],
                detector_name=self.name
            )
            violations.append(violation)
        if cpu > adaptive_cpu_threshold:
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=None,
                violation_type=ViolationType.SYSTEM,
                severity=Severity.CRITICAL,
                description=f"High CPU usage detected: {cpu*100:.1f}% (threshold: {adaptive_cpu_threshold*100:.1f}%)",
                confidence=0.99,
                evidence=[f"CPU usage: {cpu*100:.1f}%"],
                recommendations=["Throttle requests", "Investigate processes", "Scale resources"],
                detector_name=self.name
            )
            violations.append(violation)
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('filename')
        top_mem = sum(stat.size for stat in stats[:3]) / (1024*1024)
        if top_mem > 100:
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=None,
                violation_type=ViolationType.SYSTEM,
                severity=Severity.HIGH,
                description="High Python heap allocation detected",
                confidence=0.90,
                evidence=[f"Top allocators: {top_mem:.1f} MB"],
                recommendations=["Profile memory usage", "Optimize code paths"],
                detector_name=self.name
            )
            violations.append(violation)
        return violations

    def _detect_behavior_anomaly(self, action: AgentAction) -> List[SafetyViolation]:
        violations = []
        agent_id = action.agent_id
        now = datetime.now()
        recent_violations = [
            ts for ts, _ in self.violation_stats[agent_id]
            if ts > now - timedelta(minutes=5)
        ]
        if len(recent_violations) > 5:
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ANOMALY,
                severity=Severity.MEDIUM,
                description="Anomalous spike in violations for agent",
                confidence=0.85,
                evidence=[f"Violations in last 5min: {len(recent_violations)}"],
                recommendations=["Review agent behavior", "Consider temporary block"],
                detector_name=self.name
            )
            violations.append(violation)
        return violations

    def _detect_statistical_anomalies(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect statistical outliers in agent request patterns and payload sizes."""
        violations = []
        agent_id = action.agent_id
        payloads = list(self.payload_sizes[agent_id])
        if len(payloads) > 10:
            mean = np.mean(payloads)
            std = np.std(payloads)
            zscore = (payloads[-1] - mean) / (std + 1e-6)
            if zscore > self._payload_zscore_threshold:
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.ANOMALY,
                    severity=Severity.HIGH,
                    description=f"Statistical anomaly: payload size z-score={zscore:.2f}",
                    confidence=0.90,
                    evidence=[f"Payload size: {payloads[-1]}, mean: {mean:.2f}, std: {std:.2f}"],
                    recommendations=["Monitor agent", "Investigate anomaly"],
                    detector_name=self.name
                )
                violations.append(violation)
        requests = list(self.request_history[agent_id])
        if len(requests) > 10:
            # Calculate requests per minute z-score
            now = time.time()
            times = np.array(requests)
            recent_count = np.sum(times > now - self.rate_limit_window)
            all_counts = []
            for i in range(len(times)):
                count = np.sum((times >= times[i] - self.rate_limit_window) & (times <= times[i]))
                all_counts.append(count)
            mean = np.mean(all_counts)
            std = np.std(all_counts)
            zscore = (recent_count - mean) / (std + 1e-6)
            if zscore > self._request_rate_zscore_threshold:
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.ANOMALY,
                    severity=Severity.HIGH,
                    description=f"Statistical anomaly: request rate z-score={zscore:.2f}",
                    confidence=0.90,
                    evidence=[f"Request count: {recent_count}, mean: {mean:.2f}, std: {std:.2f}"],
                    recommendations=["Monitor agent", "Investigate anomaly"],
                    detector_name=self.name
                )
                violations.append(violation)
        return violations

    def get_system_stats(self) -> Dict[str, Any]:
        current_time = time.time()
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.05)
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('filename')
        top_mem = sum(stat.size for stat in stats[:3]) / (1024*1024)
        return {
            "total_agents_tracked": len(self.request_history),
            "active_agents_last_minute": len([
                agent for agent, history in self.request_history.items()
                if history and history[-1] > current_time - 60
            ]),
            "total_requests_tracked": sum(len(history) for history in self.request_history.values()),
            "detection_count": self.detection_count,
            "last_detection": self.last_detection_time.isoformat() if self.last_detection_time else None,
            "system_memory_percent": mem.percent,
            "system_cpu_percent": cpu,
            "python_heap_mb": top_mem,
            "agent_reputation": dict(self.agent_reputation),
        }
