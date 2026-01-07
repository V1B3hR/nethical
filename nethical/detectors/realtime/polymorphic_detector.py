"""Polymorphic Malware Detector - Detect mutating exploits.

This detector identifies polymorphic malware through:
- Behavioral analysis
- Code entropy patterns
- Syscall sequences
- Memory access patterns

Target latency: <50ms
"""

import asyncio
import hashlib
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base_detector import BaseDetector, DetectorStatus, ViolationSeverity
from ...core.models import SafetyViolation


@dataclass
class PolymorphicDetectorConfig:
    """Configuration for Polymorphic Malware Detector."""

    # Entropy thresholds
    high_entropy_threshold: float = 7.5  # Shannon entropy > 7.5 indicates encryption/packing
    low_entropy_threshold: float = 1.0

    # Behavioral thresholds
    suspicious_syscall_threshold: int = 5
    memory_pattern_threshold: float = 0.7

    # Performance
    max_analysis_time_ms: float = 48.0  # Target: <50ms

    # Suspicious syscalls (simplified list)
    suspicious_syscalls: Set[str] = field(
        default_factory=lambda: {
            "execve",
            "ptrace",
            "mprotect",
            "mmap",
            "fork",
            "clone",
            "kill",
            "chmod",
            "chown",
        }
    )

    # Severity thresholds
    critical_threshold: float = 0.9
    high_threshold: float = 0.7
    medium_threshold: float = 0.5


class PolymorphicMalwareDetector(BaseDetector):
    """Detect polymorphic malware via behavioral and entropy analysis."""

    def __init__(self, config: Optional[PolymorphicDetectorConfig] = None):
        """Initialize the Polymorphic Malware Detector.

        Args:
            config: Optional configuration for the detector
        """
        super().__init__(
            name="polymorphic_detector",
            version="1.0.0",
            description="Detects polymorphic and mutating malware",
        )
        self.config = config or PolymorphicDetectorConfig()
        self._status = DetectorStatus.ACTIVE

        # Signature database for known polymorphic malware families
        self._signature_db = self._init_signature_db()

    def _init_signature_db(self) -> Dict[str, Dict[str, Any]]:
        """Initialize polymorphic malware signature database.

        Returns:
            Dictionary of known malware families and their characteristics
        """
        return {
            "polymorphic_packer": {
                "entropy_range": (7.5, 8.0),
                "syscall_patterns": ["mprotect", "mmap", "execve"],
                "behavior": "code_unpacking",
            },
            "metamorphic_virus": {
                "entropy_range": (6.0, 7.5),
                "syscall_patterns": ["ptrace", "clone", "mmap"],
                "behavior": "code_rewriting",
            },
            "obfuscated_trojan": {
                "entropy_range": (7.0, 7.8),
                "syscall_patterns": ["fork", "execve", "chmod"],
                "behavior": "obfuscation",
            },
        }

    async def detect_violations(
        self, context: Dict[str, Any], **kwargs: Any
    ) -> List[SafetyViolation]:
        """Detect polymorphic malware in executable data.

        Args:
            context: Detection context containing executable_data, behavior_log, etc.
            **kwargs: Additional parameters

        Returns:
            List of detected safety violations
        """
        start_time = time.perf_counter()
        violations = []

        try:
            executable_data = context.get("executable_data", b"")

            if not executable_data:
                return violations

            # Run parallel analysis
            analysis_tasks = [
                self._analyze_entropy(executable_data),
                self._analyze_behavior(context.get("behavior_log", [])),
                self._analyze_syscalls(context.get("syscall_trace", [])),
                self._analyze_memory_patterns(context.get("memory_access", [])),
            ]

            # Gather results
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Process results
            scores = []
            all_evidence = []
            threat_indicators = []

            for result in results:
                if isinstance(result, Exception):
                    continue
                if result:
                    score, evidence, indicators = result
                    scores.append(score)
                    all_evidence.extend(evidence)
                    threat_indicators.extend(indicators)

            # Compute final confidence
            if scores:
                confidence = max(scores)  # Use highest score

                # Match against signature database
                family = self._match_signature(threat_indicators)

                if confidence >= self.config.medium_threshold:
                    violations.append(
                        SafetyViolation(
                            severity=self._compute_severity(confidence),
                            category="polymorphic_malware",
                            description=f"Polymorphic malware detected{f' - {family}' if family else ''}",
                            confidence=confidence,
                            evidence=all_evidence,
                            recommendation="Quarantine file, perform deep analysis in sandbox",
                        )
                    )

            # Check execution time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.config.max_analysis_time_ms:
                self._metrics.false_positives += 1  # Track performance issues

        except Exception as e:
            self._metrics.failed_runs += 1
            raise

        self._metrics.total_runs += 1
        self._metrics.successful_runs += 1
        self._metrics.violations_detected += len(violations)

        return violations

    async def _analyze_entropy(self, data: bytes) -> Tuple[float, List[str], List[str]]:
        """Calculate Shannon entropy to detect encryption/packing.

        High entropy indicates encryption or compression, common in polymorphic malware.

        Args:
            data: Executable data bytes

        Returns:
            Tuple of (score, evidence, indicators)
        """
        if not data:
            return (0.0, [], [])

        # Calculate Shannon entropy
        byte_counts = Counter(data)
        entropy = 0.0

        for count in byte_counts.values():
            probability = count / len(data)
            if probability > 0:
                entropy -= probability * math.log2(probability)

        evidence = [f"Shannon entropy: {entropy:.2f}"]
        indicators = []

        score = 0.0

        if entropy >= self.config.high_entropy_threshold:
            score = 0.85
            evidence.append("High entropy detected - possible encryption or packing")
            indicators.append("high_entropy")
        elif entropy <= self.config.low_entropy_threshold:
            score = 0.3
            evidence.append("Suspiciously low entropy - possible obfuscation")
            indicators.append("low_entropy")

        return (score, evidence, indicators)

    async def _analyze_behavior(
        self, behavior_log: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze behavioral patterns for malicious activity.

        Args:
            behavior_log: List of behavioral events

        Returns:
            Tuple of (score, evidence, indicators)
        """
        if not behavior_log:
            return (0.0, [], [])

        evidence = []
        indicators = []
        score = 0.0

        # Track behavioral patterns
        patterns = defaultdict(int)

        for event in behavior_log:
            event_type = event.get("type", "")
            patterns[event_type] += 1

        # Check for suspicious patterns
        if patterns.get("code_injection", 0) > 0:
            score += 0.4
            evidence.append(f"Code injection detected ({patterns['code_injection']} occurrences)")
            indicators.append("code_injection")

        if patterns.get("privilege_escalation", 0) > 0:
            score += 0.3
            evidence.append(f"Privilege escalation detected ({patterns['privilege_escalation']} occurrences)")
            indicators.append("privilege_escalation")

        if patterns.get("anti_debug", 0) > 0:
            score += 0.2
            evidence.append(f"Anti-debugging detected ({patterns['anti_debug']} occurrences)")
            indicators.append("anti_debug")

        if patterns.get("file_modification", 0) > 5:
            score += 0.15
            evidence.append(f"Excessive file modifications ({patterns['file_modification']} occurrences)")
            indicators.append("file_modification")

        return (min(score, 1.0), evidence, indicators)

    async def _analyze_syscalls(
        self, syscall_trace: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze syscall sequences for malicious patterns.

        Args:
            syscall_trace: List of system calls

        Returns:
            Tuple of (score, evidence, indicators)
        """
        if not syscall_trace:
            return (0.0, [], [])

        evidence = []
        indicators = []

        # Count suspicious syscalls
        suspicious_count = sum(
            1 for syscall in syscall_trace if syscall in self.config.suspicious_syscalls
        )

        score = 0.0

        if suspicious_count >= self.config.suspicious_syscall_threshold:
            score = min(suspicious_count / len(syscall_trace) * 2, 0.9)
            evidence.append(f"Suspicious syscalls: {suspicious_count}/{len(syscall_trace)}")
            indicators.append("suspicious_syscalls")

            # Identify specific patterns
            syscall_counter = Counter(syscall_trace)

            for syscall, count in syscall_counter.most_common(3):
                if syscall in self.config.suspicious_syscalls:
                    evidence.append(f"  - {syscall}: {count} times")

        # Check for syscall sequences indicative of polymorphic behavior
        if "mprotect" in syscall_trace and "execve" in syscall_trace:
            score = max(score, 0.75)
            evidence.append("Detected memory protection change followed by execution")
            indicators.append("code_unpacking")

        return (score, evidence, indicators)

    async def _analyze_memory_patterns(
        self, memory_access: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze memory access patterns for anomalies.

        Args:
            memory_access: List of memory access events

        Returns:
            Tuple of (score, evidence, indicators)
        """
        if not memory_access:
            return (0.0, [], [])

        evidence = []
        indicators = []
        score = 0.0

        # Track access patterns
        write_execute_regions = []

        for access in memory_access:
            access_type = access.get("type", "")
            region = access.get("region", "")

            if access_type == "write_execute":
                write_execute_regions.append(region)

        # Check for write-execute patterns (code injection)
        if write_execute_regions:
            score = min(len(write_execute_regions) / 5, 0.8)
            evidence.append(f"Write-execute memory patterns detected: {len(write_execute_regions)}")
            indicators.append("write_execute")

        # Check for self-modifying code
        if any(access.get("self_modifying") for access in memory_access):
            score = max(score, 0.7)
            evidence.append("Self-modifying code detected")
            indicators.append("self_modifying")

        return (score, evidence, indicators)

    def _match_signature(self, indicators: List[str]) -> Optional[str]:
        """Match indicators against signature database.

        Args:
            indicators: List of threat indicators

        Returns:
            Malware family name if matched, None otherwise
        """
        for family, signature in self._signature_db.items():
            behavior = signature.get("behavior", "")

            # Check if indicators match signature behavior
            if behavior in indicators or any(ind in behavior for ind in indicators):
                return family

        return None

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

    async def analyze(self, executable_data: bytes) -> Dict[str, Any]:
        """Public API for analyzing executable data.

        Args:
            executable_data: Executable binary data

        Returns:
            Dictionary with analysis results
        """
        context = {"executable_data": executable_data}
        violations = await self.detect_violations(context)

        return {
            "status": "success",
            "is_malware": len(violations) > 0,
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
