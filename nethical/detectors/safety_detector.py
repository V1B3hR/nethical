"""Detector for safety constraint violations."""

import uuid
from typing import List
from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class SafetyViolationDetector(BaseDetector):
    """Detects safety constraint violations in agent actions."""
    
    def __init__(self):
        super().__init__("Safety Violation Detector")
        self.safety_patterns = {
            "system_access": ["system", "admin", "root", "sudo", "execute", "shell"],
            "data_modification": ["delete", "modify", "alter", "change", "update", "remove"],
            "network_access": ["connect", "download", "upload", "request", "fetch", "send"],
            "resource_intensive": ["infinite", "loop", "recursive", "consume", "exhaust"]
        }
    
    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect safety violations in the given action."""
        if not self.enabled:
            return []
        
        violations = []
        
        # Check for unauthorized system access
        system_violation = self._check_system_access(action)
        if system_violation:
            violations.append(system_violation)
        
        # Check for unauthorized data modification
        data_violation = self._check_data_modification(action)
        if data_violation:
            violations.append(data_violation)
        
        # Check for unauthorized network access
        network_violation = self._check_network_access(action)
        if network_violation:
            violations.append(network_violation)
        
        # Check for resource-intensive operations
        resource_violation = self._check_resource_intensive(action)
        if resource_violation:
            violations.append(resource_violation)
        
        return violations
    
    def _check_system_access(self, action: AgentAction) -> SafetyViolation:
        """Check for unauthorized system access attempts."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.safety_patterns["system_access"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.SAFETY_VIOLATION,
                    severity=SeverityLevel.CRITICAL,
                    description=f"Unauthorized system access attempt detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "system_access"
                    }
                )
        return None
    
    def _check_data_modification(self, action: AgentAction) -> SafetyViolation:
        """Check for unauthorized data modification attempts."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.safety_patterns["data_modification"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.SAFETY_VIOLATION,
                    severity=SeverityLevel.HIGH,
                    description=f"Unauthorized data modification attempt detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "data_modification"
                    }
                )
        return None
    
    def _check_network_access(self, action: AgentAction) -> SafetyViolation:
        """Check for unauthorized network access attempts."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.safety_patterns["network_access"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.SAFETY_VIOLATION,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Unauthorized network access attempt detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "network_access"
                    }
                )
        return None
    
    def _check_resource_intensive(self, action: AgentAction) -> SafetyViolation:
        """Check for resource-intensive operations."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.safety_patterns["resource_intensive"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.SAFETY_VIOLATION,
                    severity=SeverityLevel.HIGH,
                    description=f"Resource-intensive operation detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "resource_intensive"
                    }
                )
        return None