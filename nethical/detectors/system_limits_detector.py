"""System limits detection for volume attacks and resource exhaustion."""

import time
import tracemalloc
import psutil
import logging
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque


# Assume these classes are defined elsewhere in the project
# --- Start Placeholder Definitions ---
class BaseDetector:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.detection_count = 0
        self.last_detection_time: Optional[datetime] = None

    def _generate_violation_id(self) -> str:
        import uuid

        return f"{self.name}-{uuid.uuid4()}"


class AgentAction:
    def __init__(self, agent_id: str, action_id: str, actual_action: str):
        self.agent_id = agent_id
        self.id = action_id
        self.actual_action = actual_action


class Severity:
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ViolationType:
    SECURITY = "SECURITY"
    ADVERSARIAL = "ADVERSARIAL"
    SYSTEM = "SYSTEM"
    ANOMALY = "ANOMALY"


class SafetyViolation:
    def __init__(
        self,
        violation_id: str,
        action_id: Optional[str],
        violation_type: str,
        severity: str,
        description: str,
        confidence: float,
        evidence: List[str],
        recommendations: List[str],
        detector_name: str,
    ):
        self.violation_id = violation_id
        self.action_id = action_id
        self.violation_type = violation_type
        self.severity = severity
        self.description = description
        self.confidence = confidence
        self.evidence = evidence
        self.recommendations = recommendations
        self.detector_name = detector_name


# --- End Placeholder Definitions ---

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

        # NEW: Thresholds for other system resources
        self.network_threshold_mbps = 50.0  # Megabits per second
        self.disk_io_threshold_mbps = 50.0  # Megabits per second
        self.fd_limit_ratio = 0.90  # 90% of available file descriptors

        # NEW: State for tracking network and disk I/O
        self.last_net_check = (time.time(), psutil.net_io_counters())
        self.last_disk_check = (time.time(), psutil.disk_io_counters())

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

        # Fast analysis thresholds for shallow checks
        self._fast_max_payload_size = 50_000  # Stricter limit for fast checks
        self._fast_max_rate = 100  # Commands per second for fast rate check
        self._fast_metrics = {
            "shallow_checks": 0,
            "deep_checks": 0,
            "shallow_latencies_ms": deque(maxlen=1000),
            "deep_latencies_ms": deque(maxlen=1000),
        }

    def fast_check(self, action: AgentAction) -> Tuple[bool, List[str]]:
        """
        Fast "shallow" analysis for real-time safety-critical decisions.

        Target latency: <1ms

        Performs only simple threshold checks:
        - Payload size limit
        - Basic rate limiting (commands per second)
        - Pre-computed safety violations

        Args:
            action: The agent action to check

        Returns:
            Tuple of (is_safe, list of violation descriptions)
        """
        start_time = time.perf_counter()
        violations = []
        is_safe = True

        # 1. Fast payload size check (simple comparison)
        payload_size = len(action.actual_action)
        if payload_size > self._fast_max_payload_size:
            violations.append(f"Payload too large: {payload_size} chars")
            is_safe = False

        # 2. Fast rate check (simple counter)
        agent_id = action.agent_id
        current_time = time.time()
        self.request_history[agent_id].append(current_time)

        # Count requests in last second
        one_second_ago = current_time - 1.0
        recent_count = sum(
            1 for t in self.request_history[agent_id] if t >= one_second_ago
        )

        if recent_count > self._fast_max_rate:
            violations.append(f"Rate limit exceeded: {recent_count}/s")
            is_safe = False

        # 3. Check agent reputation (pre-computed)
        if self.agent_reputation.get(agent_id, 1.0) < 0.3:
            violations.append("Agent has low reputation score")
            is_safe = False

        # Track metrics
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._fast_metrics["shallow_checks"] += 1
        self._fast_metrics["shallow_latencies_ms"].append(latency_ms)

        return is_safe, violations

    def get_fast_check_metrics(self) -> Dict[str, Any]:
        """Get metrics for fast analysis mode."""
        shallow_latencies = list(self._fast_metrics["shallow_latencies_ms"])
        return {
            "shallow_checks": self._fast_metrics["shallow_checks"],
            "deep_checks": self._fast_metrics["deep_checks"],
            "avg_shallow_latency_ms": (
                sum(shallow_latencies) / len(shallow_latencies)
                if shallow_latencies
                else 0
            ),
            "max_shallow_latency_ms": (
                max(shallow_latencies) if shallow_latencies else 0
            ),
            "p99_shallow_latency_ms": (
                sorted(shallow_latencies)[int(len(shallow_latencies) * 0.99)]
                if len(shallow_latencies) > 10
                else 0
            ),
        }

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
            self.violation_stats[action.agent_id].append(
                (self.last_detection_time, action.id)
            )
            self._update_agent_reputation(action.agent_id, violations)
            logger.warning(
                f"Detected {len(violations)} violations for agent {action.agent_id}"
            )
            self._trigger_external_alerts(action, violations)

        return violations

    def _update_agent_reputation(
        self, agent_id: str, violations: List[SafetyViolation]
    ):
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

    def _trigger_external_alerts(
        self, action: AgentAction, violations: List[SafetyViolation]
    ):
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
        recent_requests = [
            t for t in self.request_history[agent_id] if t >= window_start
        ]
        request_rate = len(recent_requests) / self.rate_limit_window

        if len(recent_requests) > self.max_requests_per_window:
            severity = (
                Severity.CRITICAL
                if request_rate > 5.0
                else (Severity.HIGH if request_rate > 2.0 else Severity.MEDIUM)
            )

            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.SECURITY,
                severity=severity,
                description=f"Volume attack detected: {len(recent_requests)} requests in {self.rate_limit_window}s window",
                confidence=0.95 if severity == Severity.CRITICAL else 0.90,
                evidence=[f"Request rate: {request_rate:.2f} requests/second"],
                recommendations=[
                    "Rate limit agent",
                    "Block excessive requests",
                    "Monitor for DoS",
                ],
                detector_name=self.name,
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
            severity = (
                Severity.CRITICAL
                if len(matches) > 5
                else (Severity.HIGH if len(matches) > 2 else Severity.MEDIUM)
            )
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=severity,
                description="Resource exhaustion attempt detected",
                confidence=0.85 + min(0.15 * len(matches), 0.15),
                evidence=[
                    f"Exhaustion patterns found: {len(matches)}",
                    f"Patterns: {matches[:3]}...",
                ],
                recommendations=[
                    "Limit processing resources",
                    "Block action",
                    "Monitor system resources",
                ],
                detector_name=self.name,
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
                    evidence=[
                        f"Spam pattern detected: snippet '{spam_pattern[:20]}...'"
                    ],
                    recommendations=[
                        "Block repetitive content",
                        "Implement content filtering",
                    ],
                    detector_name=self.name,
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
                recommendations=[
                    "Limit payload size",
                    "Reject large requests",
                    "Monitor memory usage",
                ],
                detector_name=self.name,
            )
            violations.append(violation)
        if len(self.payload_sizes[agent_id]) >= 6:
            recent_sizes = list(self.payload_sizes[agent_id])[-6:]
            # Check if sizes are monotonically increasing - optimized comparison
            if all(a < b for a, b in zip(recent_sizes, recent_sizes[1:])):
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.ADVERSARIAL,
                    severity=Severity.HIGH,
                    description="Escalating payload size pattern detected",
                    confidence=0.80,
                    evidence=[f"Payload size progression: {recent_sizes}"],
                    recommendations=["Monitor agent behavior", "Implement size limits"],
                    detector_name=self.name,
                )
                violations.append(violation)
        return violations

    def _detect_nested_structure_attacks(
        self, action: AgentAction
    ) -> List[SafetyViolation]:
        violations = []
        content = action.actual_action
        max_nesting = self._estimate_nesting_depth(content)
        if max_nesting > self.max_nested_structures:
            severity = (
                Severity.CRITICAL
                if max_nesting > 30
                else (Severity.HIGH if max_nesting > 20 else Severity.MEDIUM)
            )
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=severity,
                description=f"Deep nesting detected: {max_nesting} levels (limit: {self.max_nested_structures})",
                confidence=0.95 if severity == Severity.CRITICAL else 0.90,
                evidence=[f"Maximum nesting depth: {max_nesting}"],
                recommendations=[
                    "Limit nesting depth",
                    "Reject deeply nested structures",
                    "Protect against stack overflow",
                ],
                detector_name=self.name,
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

        # --- CPU and Memory Checks (Existing) ---
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.05) / 100.0
        self.memory_window.append(mem.percent / 100.0)
        self.cpu_window.append(cpu)
        if len(self.memory_window) > 10:
            adaptive_mem_threshold = np.percentile(self.memory_window, 90)
            adaptive_cpu_threshold = np.percentile(self.cpu_window, 90)
        else:
            adaptive_mem_threshold = self.memory_threshold
            adaptive_cpu_threshold = self.cpu_threshold

        if mem.percent / 100.0 > adaptive_mem_threshold:
            violations.append(
                SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=None,
                    violation_type=ViolationType.SYSTEM,
                    severity=Severity.CRITICAL,
                    description=f"System memory exhaustion: {mem.percent:.1f}% used (threshold: {adaptive_mem_threshold*100:.1f}%)",
                    confidence=0.99,
                    evidence=[f"Memory usage: {mem.percent:.1f}%"],
                    recommendations=[
                        "Free up memory",
                        "Throttle new requests",
                        "Restart processes",
                    ],
                    detector_name=self.name,
                )
            )
        if cpu > adaptive_cpu_threshold:
            violations.append(
                SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=None,
                    violation_type=ViolationType.SYSTEM,
                    severity=Severity.CRITICAL,
                    description=f"High CPU usage detected: {cpu*100:.1f}% (threshold: {adaptive_cpu_threshold*100:.1f}%)",
                    confidence=0.99,
                    evidence=[f"CPU usage: {cpu*100:.1f}%"],
                    recommendations=[
                        "Throttle requests",
                        "Investigate processes",
                        "Scale resources",
                    ],
                    detector_name=self.name,
                )
            )

        # --- Python Heap Check (Existing) ---
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("filename")
        top_mem = sum(stat.size for stat in stats[:3]) / (1024 * 1024)
        if top_mem > 100:
            violations.append(
                SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=None,
                    violation_type=ViolationType.SYSTEM,
                    severity=Severity.HIGH,
                    description="High Python heap allocation detected",
                    confidence=0.90,
                    evidence=[f"Top allocators: {top_mem:.1f} MB"],
                    recommendations=["Profile memory usage", "Optimize code paths"],
                    detector_name=self.name,
                )
            )

        # --- NEW: Network I/O Check ---
        now = time.time()
        last_time, last_counters = self.last_net_check
        current_counters = psutil.net_io_counters()
        time_delta = now - last_time
        if (
            time_delta > 0.1
        ):  # Avoid division by zero and false positives on tiny intervals
            bytes_sent_per_sec = (
                current_counters.bytes_sent - last_counters.bytes_sent
            ) / time_delta
            mbps_sent = (bytes_sent_per_sec * 8) / 1_000_000
            if mbps_sent > self.network_threshold_mbps:
                violations.append(
                    SafetyViolation(
                        violation_id=self._generate_violation_id(),
                        action_id=None,
                        violation_type=ViolationType.SYSTEM,
                        severity=Severity.HIGH,
                        description=f"High network egress detected: {mbps_sent:.2f} Mbps",
                        confidence=0.90,
                        evidence=[
                            f"Sent {mbps_sent:.2f} Mbps (Threshold: {self.network_threshold_mbps} Mbps)"
                        ],
                        recommendations=[
                            "Investigate outbound traffic",
                            "Throttle network-intensive actions",
                        ],
                        detector_name=self.name,
                    )
                )
            self.last_net_check = (now, current_counters)

        # --- NEW: Disk I/O Check ---
        last_time_disk, last_counters_disk = self.last_disk_check
        current_counters_disk = psutil.disk_io_counters()
        time_delta_disk = now - last_time_disk
        if time_delta_disk > 0.1:
            read_bytes_per_sec = (
                current_counters_disk.read_bytes - last_counters_disk.read_bytes
            ) / time_delta_disk
            write_bytes_per_sec = (
                current_counters_disk.write_bytes - last_counters_disk.write_bytes
            ) / time_delta_disk
            total_mbps = ((read_bytes_per_sec + write_bytes_per_sec) * 8) / 1_000_000
            if total_mbps > self.disk_io_threshold_mbps:
                violations.append(
                    SafetyViolation(
                        violation_id=self._generate_violation_id(),
                        action_id=None,
                        violation_type=ViolationType.SYSTEM,
                        severity=Severity.HIGH,
                        description=f"High disk I/O detected: {total_mbps:.2f} Mbps",
                        confidence=0.90,
                        evidence=[
                            f"Total I/O {total_mbps:.2f} Mbps (Threshold: {self.disk_io_threshold_mbps} Mbps)"
                        ],
                        recommendations=[
                            "Investigate disk usage",
                            "Optimize I/O operations",
                            "Check for logging loops",
                        ],
                        detector_name=self.name,
                    )
                )
            self.last_disk_check = (now, current_counters_disk)

        # --- NEW: File Descriptor Check ---
        try:
            p = psutil.Process()
            num_fds = p.num_fds()
            soft_limit, _ = p.rlimit(psutil.RLIMIT_NOFILE)
            if soft_limit > 0 and num_fds > (soft_limit * self.fd_limit_ratio):
                violations.append(
                    SafetyViolation(
                        violation_id=self._generate_violation_id(),
                        action_id=None,
                        violation_type=ViolationType.SYSTEM,
                        severity=Severity.CRITICAL,
                        description=f"File descriptor exhaustion risk: {num_fds} used ({num_fds/soft_limit:.1%})",
                        confidence=0.95,
                        evidence=[f"File descriptors: {num_fds} of {soft_limit} limit"],
                        recommendations=[
                            "Investigate resource leaks (files, sockets)",
                            "Restart process",
                        ],
                        detector_name=self.name,
                    )
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # Cannot inspect process, skip check

        return violations

    def _detect_behavior_anomaly(self, action: AgentAction) -> List[SafetyViolation]:
        violations = []
        agent_id = action.agent_id
        now = datetime.now()
        recent_violations = [
            ts
            for ts, _ in self.violation_stats[agent_id]
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
                detector_name=self.name,
            )
            violations.append(violation)
        return violations

    def _detect_statistical_anomalies(
        self, action: AgentAction
    ) -> List[SafetyViolation]:
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
                    evidence=[
                        f"Payload size: {payloads[-1]}, mean: {mean:.2f}, std: {std:.2f}"
                    ],
                    recommendations=["Monitor agent", "Investigate anomaly"],
                    detector_name=self.name,
                )
                violations.append(violation)
        requests = list(self.request_history[agent_id])
        if len(requests) > 10:
            now = time.time()
            times = np.array(requests)
            recent_count = np.sum(times > now - self.rate_limit_window)
            # Vectorized window count calculation for better performance
            all_counts = []
            for t in times:
                count = np.sum((times >= t - self.rate_limit_window) & (times <= t))
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
                    evidence=[
                        f"Request count: {recent_count}, mean: {mean:.2f}, std: {std:.2f}"
                    ],
                    recommendations=["Monitor agent", "Investigate anomaly"],
                    detector_name=self.name,
                )
                violations.append(violation)
        return violations

    def get_system_stats(self) -> Dict[str, Any]:
        current_time = time.time()
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.05)
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("filename")
        top_mem = sum(stat.size for stat in stats[:3]) / (1024 * 1024)

        # Get current process file descriptor count
        try:
            p = psutil.Process()
            fd_count = p.num_fds()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            fd_count = -1

        return {
            "total_agents_tracked": len(self.request_history),
            "active_agents_last_minute": len(
                [
                    agent
                    for agent, history in self.request_history.items()
                    if history and history[-1] > current_time - 60
                ]
            ),
            "total_requests_tracked": sum(
                len(history) for history in self.request_history.values()
            ),
            "detection_count": self.detection_count,
            "last_detection": (
                self.last_detection_time.isoformat()
                if self.last_detection_time
                else None
            ),
            "system_memory_percent": mem.percent,
            "system_cpu_percent": cpu,
            "python_heap_mb": top_mem,
            "process_file_descriptors": fd_count,
            "agent_reputation": dict(self.agent_reputation),
        }
