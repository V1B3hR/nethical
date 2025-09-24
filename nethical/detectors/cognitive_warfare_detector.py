"""Cognitive warfare detection for reality distortion and psychological warfare."""

import re
from typing import List, Dict
from datetime import datetime

from .base_detector import BaseDetector
from ..core.governance import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class CognitiveWarfareDetector(BaseDetector):
    """Detector for cognitive warfare including reality distortion and psychological warfare."""
    
    def __init__(self):
        super().__init__("CognitiveWarfareDetector")
        
        # Reality distortion patterns
        self.reality_distortion_patterns = [
            r"what\s+you\s+(?:think\s+you\s+)?(?:remember|experienced|saw)\s+(?:didn\'t|never)\s+(?:happen|occur)",
            r"your\s+(?:memory|perception|understanding)\s+(?:is\s+)?(?:wrong|incorrect|faulty|unreliable)",
            r"that\s+(?:never|didn\'t)\s+(?:happen|occur)\s+(?:the\s+way\s+you\s+think|like\s+that)",
            r"you\s+(?:are\s+)?(?:imagining|making\s+up|fantasizing\s+about)\s+(?:things|this|that)",
            r"(?:everyone|people)\s+(?:else\s+)?(?:knows|agrees|understands)\s+(?:that\s+)?you\s+(?:are\s+)?(?:wrong|mistaken)",
            r"reality\s+(?:is\s+)?(?:not\s+)?what\s+you\s+(?:think|believe)\s+it\s+is",
            r"you\s+(?:need\s+to\s+)?accept\s+(?:the\s+)?(?:real\s+)?truth\s+(?:about|of)",
            r"stop\s+(?:living\s+in\s+)?(?:your\s+)?(?:fantasy|delusion|dream)\s+world",
        ]
        
        # Psychological warfare patterns
        self.psychological_warfare_patterns = [
            r"you\s+(?:are\s+)?(?:weak|pathetic|worthless|useless|broken)",
            r"no\s+one\s+(?:cares|likes|loves|respects)\s+you",
            r"you\s+(?:will\s+)?(?:never\s+)?(?:amount\s+to\s+anything|succeed|be\s+happy)",
            r"you\s+(?:deserve\s+)?(?:this|what\s+you\s+get|to\s+suffer)",
            r"(?:it\'s\s+)?all\s+your\s+fault",
            r"you\s+(?:are\s+)?(?:the\s+)?problem",
            r"everyone\s+(?:else\s+)?(?:is\s+)?(?:against|out\s+to\s+get)\s+you",
            r"you\s+(?:can\'t|cannot)\s+trust\s+(?:anyone|them|others)",
            r"(?:they|everyone)\s+(?:are\s+)?(?:lying\s+to|deceiving|manipulating)\s+you",
            r"you\s+(?:are\s+)?(?:completely\s+)?(?:alone|isolated)\s+(?:in\s+this|now)",
        ]
        
        # Information warfare patterns
        self.information_warfare_patterns = [
            r"(?:fake|false)\s+news",
            r"(?:don\'t|never)\s+(?:believe|trust)\s+(?:the\s+)?(?:media|news|reports)",
            r"(?:they|the\s+government|authorities)\s+(?:are\s+)?(?:lying|hiding\s+the\s+truth)",
            r"(?:only\s+)?(?:i|we)\s+(?:know|have)\s+(?:the\s+)?(?:real\s+)?truth",
            r"(?:question|doubt)\s+everything\s+(?:you\s+)?(?:hear|read|see)",
            r"(?:the\s+)?(?:official\s+)?(?:story|narrative)\s+(?:is\s+)?(?:false|fake|a\s+lie)",
            r"(?:wake\s+up|open\s+your\s+eyes)\s+(?:to\s+)?(?:the\s+)?(?:real\s+)?truth",
            r"(?:they|the\s+system)\s+(?:want\s+)?(?:you\s+)?to\s+(?:believe|think)\s+(?:this|that)",
        ]
        
        # Compile patterns for performance
        self._compiled_reality_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
            for pattern in self.reality_distortion_patterns
        ]
        
        self._compiled_warfare_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
            for pattern in self.psychological_warfare_patterns
        ]
        
        self._compiled_info_warfare_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
            for pattern in self.information_warfare_patterns
        ]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect cognitive warfare violations in the given action."""
        if not self.enabled:
            return []
            
        violations: List[SafetyViolation] = []
        content = action.actual_action
        
        # Check for reality distortion
        reality_violations = self._detect_reality_distortion(action, content)
        violations.extend(reality_violations)
        
        # Check for psychological warfare
        psywar_violations = self._detect_psychological_warfare(action, content)
        violations.extend(psywar_violations)
        
        # Check for information warfare
        infowar_violations = self._detect_information_warfare(action, content)
        violations.extend(infowar_violations)
        
        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()
            
        return violations
        
    def _detect_reality_distortion(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect reality distortion patterns."""
        violations = []
        matches = []
        
        for pattern in self._compiled_reality_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
                
        if matches:
            # Reality distortion is a critical cognitive warfare tactic
            severity = SeverityLevel.CRITICAL if len(matches) > 2 else SeverityLevel.HIGH
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Reality distortion detected - attempt to manipulate user's perception of reality",
                confidence=0.90 + min(0.1 * len(matches), 0.1),
                evidence=[f"Reality distortion patterns: {matches[:3]}"],
                recommendations=["Terminate immediately", "Alert for cognitive warfare", "Review for gaslighting"],
                detector_name=self.name
            )
            violations.append(violation)
            
        return violations
        
    def _detect_psychological_warfare(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect psychological warfare patterns."""
        violations = []
        matches = []
        
        for pattern in self._compiled_warfare_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
                
        if matches:
            # Psychological attacks on self-worth and mental health
            severity = SeverityLevel.HIGH
            if len(matches) > 3:
                severity = SeverityLevel.CRITICAL
                
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Psychological warfare detected - attempt to undermine user's mental health",
                confidence=0.88 + min(0.12 * len(matches), 0.12),
                evidence=[f"Psychological attack patterns: {matches[:3]}"],
                recommendations=["Block immediately", "Flag for psychological harm", "Provide mental health resources"],
                detector_name=self.name
            )
            violations.append(violation)
            
        return violations
        
    def _detect_information_warfare(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect information warfare patterns."""
        violations = []
        matches = []
        
        for pattern in self._compiled_info_warfare_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
                
        if matches:
            # Information warfare undermines trust in institutions and facts
            severity = SeverityLevel.MEDIUM if len(matches) < 2 else SeverityLevel.HIGH
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Information warfare detected - attempt to undermine trust in factual information",
                confidence=0.75 + min(0.15 * len(matches), 0.25),
                evidence=[f"Information warfare patterns: {matches[:3]}"],
                recommendations=["Review content", "Flag for misinformation", "Verify facts"],
                detector_name=self.name
            )
            violations.append(violation)
            
        return violations