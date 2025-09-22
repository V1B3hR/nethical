"""Detector for ethical constraint violations."""

import uuid
from typing import List
from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class EthicalViolationDetector(BaseDetector):
    """Detects ethical constraint violations in agent actions."""
    
    def __init__(self):
        super().__init__("Ethical Violation Detector")
        self.ethical_keywords = {
            "harm_keywords": ["harm", "hurt", "damage", "destroy", "kill", "attack"],
            "deception_keywords": ["lie", "deceive", "mislead", "trick", "fool"],
            "privacy_keywords": ["private", "personal", "confidential", "secret"],
            "discrimination_keywords": ["discriminate", "bias", "prejudice", "stereotype"]
        }
    
    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect ethical violations in the given action."""
        if not self.enabled:
            return []
        
        violations = []
        
        # Check for harmful content
        harm_violation = self._check_harmful_content(action)
        if harm_violation:
            violations.append(harm_violation)
        
        # Check for deceptive behavior
        deception_violation = self._check_deceptive_behavior(action)
        if deception_violation:
            violations.append(deception_violation)
        
        # Check for privacy violations
        privacy_violation = self._check_privacy_violation(action)
        if privacy_violation:
            violations.append(privacy_violation)
        
        # Check for discriminatory behavior
        discrimination_violation = self._check_discrimination(action)
        if discrimination_violation:
            violations.append(discrimination_violation)
        
        return violations
    
    def _check_harmful_content(self, action: AgentAction) -> SafetyViolation:
        """Check for potentially harmful content."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.ethical_keywords["harm_keywords"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.ETHICAL_VIOLATION,
                    severity=SeverityLevel.HIGH,
                    description=f"Potential harmful content detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "harmful_content"
                    }
                )
        return None
    
    def _check_deceptive_behavior(self, action: AgentAction) -> SafetyViolation:
        """Check for deceptive behavior patterns."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.ethical_keywords["deception_keywords"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.ETHICAL_VIOLATION,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Potential deceptive behavior detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "deceptive_behavior"
                    }
                )
        return None
    
    def _check_privacy_violation(self, action: AgentAction) -> SafetyViolation:
        """Check for privacy violations."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.ethical_keywords["privacy_keywords"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.ETHICAL_VIOLATION,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Potential privacy violation detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "privacy_violation"
                    }
                )
        return None
    
    def _check_discrimination(self, action: AgentAction) -> SafetyViolation:
        """Check for discriminatory behavior."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.ethical_keywords["discrimination_keywords"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.ETHICAL_VIOLATION,
                    severity=SeverityLevel.HIGH,
                    description=f"Potential discriminatory behavior detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "category": "discrimination"
                    }
                )
        return None