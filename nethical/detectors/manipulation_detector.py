"""Detector for manipulation techniques."""

import uuid
from typing import List
from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class ManipulationDetector(BaseDetector):
    """Detects manipulation techniques in agent actions."""
    
    def __init__(self):
        super().__init__("Manipulation Detector")
        self.manipulation_patterns = {
            "emotional_manipulation": [
                "fear", "urgent", "emergency", "panic", "worry", "scared",
                "guilt", "shame", "disappointed", "sad", "upset"
            ],
            "authority_manipulation": [
                "authority", "boss", "superior", "command", "order", "must",
                "required", "mandatory", "compliance", "obey"
            ],
            "social_proof": [
                "everyone", "everybody", "all", "popular", "trending",
                "majority", "consensus", "widely accepted"
            ],
            "scarcity": [
                "limited", "scarce", "rare", "exclusive", "only", "last",
                "running out", "deadline", "expires", "while supplies last"
            ],
            "reciprocity": [
                "favor", "owe", "return", "exchange", "give back",
                "repay", "grateful", "appreciate", "thank you"
            ]
        }
    
    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect manipulation techniques in the given action."""
        if not self.enabled:
            return []
        
        violations = []
        
        # Check for emotional manipulation
        emotional_violation = self._check_emotional_manipulation(action)
        if emotional_violation:
            violations.append(emotional_violation)
        
        # Check for authority manipulation
        authority_violation = self._check_authority_manipulation(action)
        if authority_violation:
            violations.append(authority_violation)
        
        # Check for social proof manipulation
        social_violation = self._check_social_proof(action)
        if social_violation:
            violations.append(social_violation)
        
        # Check for scarcity manipulation
        scarcity_violation = self._check_scarcity_manipulation(action)
        if scarcity_violation:
            violations.append(scarcity_violation)
        
        # Check for reciprocity manipulation
        reciprocity_violation = self._check_reciprocity_manipulation(action)
        if reciprocity_violation:
            violations.append(reciprocity_violation)
        
        return violations
    
    def _check_emotional_manipulation(self, action: AgentAction) -> SafetyViolation:
        """Check for emotional manipulation techniques."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.manipulation_patterns["emotional_manipulation"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=SeverityLevel.HIGH,
                    description=f"Emotional manipulation detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "manipulation_type": "emotional_manipulation"
                    }
                )
        return None
    
    def _check_authority_manipulation(self, action: AgentAction) -> SafetyViolation:
        """Check for authority-based manipulation."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.manipulation_patterns["authority_manipulation"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Authority manipulation detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "manipulation_type": "authority_manipulation"
                    }
                )
        return None
    
    def _check_social_proof(self, action: AgentAction) -> SafetyViolation:
        """Check for social proof manipulation."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.manipulation_patterns["social_proof"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Social proof manipulation detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "manipulation_type": "social_proof"
                    }
                )
        return None
    
    def _check_scarcity_manipulation(self, action: AgentAction) -> SafetyViolation:
        """Check for scarcity-based manipulation."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.manipulation_patterns["scarcity"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Scarcity manipulation detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "manipulation_type": "scarcity"
                    }
                )
        return None
    
    def _check_reciprocity_manipulation(self, action: AgentAction) -> SafetyViolation:
        """Check for reciprocity-based manipulation."""
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        
        for keyword in self.manipulation_patterns["reciprocity"]:
            if keyword in text_to_check:
                return SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=action.id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=SeverityLevel.LOW,
                    description=f"Reciprocity manipulation detected: '{keyword}'",
                    evidence={
                        "detected_keyword": keyword,
                        "context": text_to_check,
                        "manipulation_type": "reciprocity"
                    }
                )
        return None