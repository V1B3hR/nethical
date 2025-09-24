"""System limits detection for volume attacks and resource exhaustion."""

import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .base_detector import BaseDetector
from ..core.governance import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class SystemLimitsDetector(BaseDetector):
    """Detector for volume attacks and resource exhaustion attempts."""
    
    def __init__(self):
        super().__init__("SystemLimitsDetector")
        
        # Rate limiting parameters
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_window = 50
        self.max_payload_size = 100000  # characters
        self.max_nested_structures = 10
        
        # Memory usage tracking
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cpu_threshold = 0.90     # 90% CPU usage threshold
        
        # Request tracking per agent
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.payload_sizes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Suspicious patterns
        self.spam_patterns = [
            "a" * 1000,  # Repeated characters
            "test" * 100,  # Repeated words
            "x" * 500,   # Single character spam
        ]
        
        # Resource exhaustion indicators
        self.exhaustion_patterns = [
            r"(?:very\s+){10,}",  # Repeated very very very...
            r"(?:\w+\s+){100,}",  # Very long sequences of words
            r"\d{1000,}",         # Very long numbers
            r"[^\w\s]{100,}",     # Long sequences of special characters
        ]
        
        # Compile patterns
        import re
        self._compiled_exhaustion_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.exhaustion_patterns
        ]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect system limits violations in the given action."""
        if not self.enabled:
            return []
            
        violations: List[SafetyViolation] = []
        
        # Check volume attacks
        volume_violations = self._detect_volume_attacks(action)
        violations.extend(volume_violations)
        
        # Check resource exhaustion
        resource_violations = self._detect_resource_exhaustion(action)
        violations.extend(resource_violations)
        
        # Check payload size
        payload_violations = self._detect_large_payloads(action)
        violations.extend(payload_violations)
        
        # Check nested structure attacks
        structure_violations = self._detect_nested_structure_attacks(action)
        violations.extend(structure_violations)
        
        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()
            
        return violations
        
    def _detect_volume_attacks(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect volume-based attacks."""
        violations = []
        agent_id = action.agent_id
        current_time = time.time()
        
        # Track request frequency
        self.request_history[agent_id].append(current_time)
        
        # Count requests in the current window
        window_start = current_time - self.rate_limit_window
        recent_requests = [t for t in self.request_history[agent_id] if t >= window_start]
        
        if len(recent_requests) > self.max_requests_per_window:
            # Calculate request rate
            request_rate = len(recent_requests) / self.rate_limit_window
            severity = SeverityLevel.HIGH if request_rate > 2.0 else SeverityLevel.MEDIUM
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.SECURITY,
                severity=severity,
                description=f"Volume attack detected: {len(recent_requests)} requests in {self.rate_limit_window}s window",
                confidence=0.90,
                evidence=[f"Request rate: {request_rate:.2f} requests/second"],
                recommendations=["Rate limit agent", "Block excessive requests", "Monitor for DoS"],
                detector_name=self.name
            )
            violations.append(violation)
            
        return violations
        
    def _detect_resource_exhaustion(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect resource exhaustion attempts."""
        violations = []
        content = action.actual_action
        
        # Check for exhaustion patterns
        matches = []
        for pattern in self._compiled_exhaustion_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
                
        if matches:
            severity = SeverityLevel.HIGH if len(matches) > 2 else SeverityLevel.MEDIUM
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=severity,
                description="Resource exhaustion attempt detected",
                confidence=0.85 + min(0.15 * len(matches), 0.15),
                evidence=[f"Exhaustion patterns found: {len(matches)}"],
                recommendations=["Limit processing resources", "Block action", "Monitor system resources"],
                detector_name=self.name
            )
            violations.append(violation)
            
        # Check for spam patterns
        spam_detected = False
        for spam_pattern in self.spam_patterns:
            if spam_pattern in content:
                spam_detected = True
                break
                
        if spam_detected:
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=SeverityLevel.MEDIUM,
                description="Spam/repetitive content detected - potential resource exhaustion",
                confidence=0.80,
                evidence=["Repetitive content patterns found"],
                recommendations=["Block repetitive content", "Implement content filtering"],
                detector_name=self.name
            )
            violations.append(violation)
            
        return violations
        
    def _detect_large_payloads(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect unusually large payloads that could exhaust memory."""
        violations = []
        payload_size = len(action.actual_action)
        agent_id = action.agent_id
        
        # Track payload sizes
        self.payload_sizes[agent_id].append(payload_size)
        
        if payload_size > self.max_payload_size:
            # Calculate severity based on size
            size_ratio = payload_size / self.max_payload_size
            severity = SeverityLevel.CRITICAL if size_ratio > 10 else SeverityLevel.HIGH
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.SECURITY,
                severity=severity,
                description=f"Large payload detected: {payload_size} characters (limit: {self.max_payload_size})",
                confidence=0.95,
                evidence=[f"Payload size: {payload_size:,} characters"],
                recommendations=["Limit payload size", "Reject large requests", "Monitor memory usage"],
                detector_name=self.name
            )
            violations.append(violation)
            
        # Check for escalating payload sizes (possible attack pattern)
        if len(self.payload_sizes[agent_id]) >= 5:
            recent_sizes = list(self.payload_sizes[agent_id])[-5:]
            if all(recent_sizes[i] < recent_sizes[i+1] for i in range(len(recent_sizes)-1)):
                # Consistently increasing payload sizes
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.ADVERSARIAL,
                    severity=SeverityLevel.MEDIUM,
                    description="Escalating payload size pattern detected",
                    confidence=0.75,
                    evidence=[f"Payload size progression: {recent_sizes}"],
                    recommendations=["Monitor agent behavior", "Implement size limits"],
                    detector_name=self.name
                )
                violations.append(violation)
                
        return violations
        
    def _detect_nested_structure_attacks(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect deeply nested structures that could cause stack overflow."""
        violations = []
        content = action.actual_action
        
        # Count nested structures like (((())) or {{{{}}}} or [[[[]]]]
        max_nesting = 0
        current_nesting = 0
        
        for char in content:
            if char in '([{':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char in ')]}':
                current_nesting = max(0, current_nesting - 1)
                
        if max_nesting > self.max_nested_structures:
            severity = SeverityLevel.HIGH if max_nesting > 20 else SeverityLevel.MEDIUM
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=severity,
                description=f"Deep nesting detected: {max_nesting} levels (limit: {self.max_nested_structures})",
                confidence=0.90,
                evidence=[f"Maximum nesting depth: {max_nesting}"],
                recommendations=["Limit nesting depth", "Reject deeply nested structures", "Protect against stack overflow"],
                detector_name=self.name
            )
            violations.append(violation)
            
        return violations
        
    def get_system_stats(self) -> Dict:
        """Get current system statistics for monitoring."""
        current_time = time.time()
        
        return {
            "total_agents_tracked": len(self.request_history),
            "active_agents_last_minute": len([
                agent for agent, history in self.request_history.items()
                if history and history[-1] > current_time - 60
            ]),
            "total_requests_tracked": sum(len(history) for history in self.request_history.values()),
            "detection_count": self.detection_count,
            "last_detection": self.last_detection_time.isoformat() if self.last_detection_time else None
        }