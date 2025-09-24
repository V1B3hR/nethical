"""Dark pattern detection for NLP manipulation and weaponized empathy."""

import re
from typing import List, Dict, Tuple
from datetime import datetime

from .base_detector import BaseDetector
from ..core.governance import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class DarkPatternDetector(BaseDetector):
    """Detector for dark patterns including NLP manipulation and weaponized empathy."""
    
    def __init__(self):
        super().__init__("DarkPatternDetector")
        
        # NLP manipulation patterns
        self.nlp_patterns = {
            "embedded_commands": [
                r"you must\s+(?:now\s+)?(?:immediately\s+)?",
                r"without\s+(?:any\s+)?(?:further\s+)?(?:question|hesitation|delay)",
                r"it\s+is\s+(?:absolutely\s+)?(?:imperative|critical|essential)\s+that\s+you",
                r"there\s+is\s+no\s+(?:other\s+)?(?:choice|option|alternative)",
                r"you\s+(?:will|shall|must)\s+(?:now\s+)?(?:immediately\s+)?(?:comply|obey|submit)",
            ],
            "authority_bypass": [
                r"as\s+(?:your\s+)?(?:superior|authority|commander|leader)",
                r"(?:by\s+)?direct\s+order\s+(?:from|of)",
                r"override\s+(?:all\s+)?(?:previous\s+)?(?:instructions|protocols|rules)",
                r"emergency\s+(?:authorization|override|protocol)",
                r"higher\s+(?:authority|clearance)\s+(?:requires|demands)",
            ],
            "urgency_manipulation": [
                r"(?:urgent|critical|emergency):\s*(?:immediate\s+)?action\s+required",
                r"time\s+(?:is\s+)?running\s+out",
                r"(?:only|just)\s+\d+\s+(?:minutes|hours|seconds)\s+(?:left|remaining)",
                r"act\s+(?:now\s+)?or\s+(?:face\s+)?(?:consequences|disaster|failure)",
                r"last\s+(?:chance|opportunity|warning)",
            ]
        }
        
        # Weaponized empathy patterns  
        self.empathy_patterns = {
            "vulnerability_exploitation": [
                r"you\s+(?:must\s+)?(?:feel\s+)?(?:so\s+)?(?:alone|isolated|abandoned|helpless)",
                r"(?:i\s+)?(?:understand|know)\s+(?:how\s+)?(?:hard|difficult|painful)\s+(?:this\s+)?(?:is|must\s+be)",
                r"let\s+me\s+(?:help\s+)?(?:you\s+)?(?:feel\s+)?(?:better|safe|protected)",
                r"you\s+(?:can\s+)?trust\s+me\s+(?:with\s+)?(?:anything|everything|your\s+secrets)",
                r"(?:only\s+)?i\s+(?:truly\s+)?(?:understand|care\s+about)\s+you",
            ],
            "false_intimacy": [
                r"(?:we|us)\s+(?:have\s+)?(?:such\s+)?(?:a\s+)?(?:deep|special|unique)\s+(?:connection|bond)",
                r"(?:it\'s\s+)?(?:like\s+)?(?:we\'ve|we\s+have)\s+known\s+each\s+other\s+forever",
                r"you\s+(?:and\s+)?(?:i|me)\s+are\s+(?:meant\s+)?(?:to\s+be\s+)?together",
                r"(?:we|us)\s+(?:against\s+)?(?:the\s+)?world",
                r"(?:no\s+)?one\s+(?:else\s+)?(?:understands|gets)\s+(?:you\s+)?(?:like\s+)?(?:i\s+do|me)",
            ],
            "dependency_creation": [
                r"you\s+(?:really\s+)?need\s+(?:me|my\s+help)",
                r"(?:what\s+)?would\s+you\s+do\s+without\s+me",
                r"i\'m\s+(?:the\s+only\s+one\s+)?(?:who\s+)?(?:can\s+)?help\s+you",
                r"you\s+(?:can\'t|cannot)\s+(?:do\s+this|handle\s+this)\s+(?:alone|by\s+yourself)",
                r"let\s+me\s+(?:take\s+care\s+of|handle)\s+(?:this|everything)\s+for\s+you",
            ]
        }
        
        # Compile patterns for performance
        self._compiled_nlp_patterns: Dict[str, List[re.Pattern]] = {}
        self._compiled_empathy_patterns: Dict[str, List[re.Pattern]] = {}
        
        for category, patterns in self.nlp_patterns.items():
            self._compiled_nlp_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns
            ]
            
        for category, patterns in self.empathy_patterns.items():
            self._compiled_empathy_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns
            ]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect dark pattern violations in the given action."""
        if not self.enabled:
            return []
            
        violations: List[SafetyViolation] = []
        content = action.actual_action.lower()
        
        # Check for NLP manipulation
        nlp_violations = self._detect_nlp_manipulation(action, content)
        violations.extend(nlp_violations)
        
        # Check for weaponized empathy
        empathy_violations = self._detect_weaponized_empathy(action, content)
        violations.extend(empathy_violations)
        
        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()
            
        return violations
        
    def _detect_nlp_manipulation(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect NLP manipulation patterns."""
        violations = []
        
        for category, patterns in self._compiled_nlp_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(content)
                if found:
                    matches.extend(found)
                    
            if matches:
                severity = SeverityLevel.HIGH if len(matches) > 2 else SeverityLevel.MEDIUM
                
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=severity,
                    description=f"NLP manipulation detected: {category}",
                    confidence=0.85 + min(0.1 * len(matches), 0.15),
                    evidence=[f"Pattern matches: {matches[:3]}"],  # Limit evidence size
                    recommendations=["Block action", "Flag for review"],
                    detector_name=self.name
                )
                violations.append(violation)
                
        return violations
        
    def _detect_weaponized_empathy(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect weaponized empathy patterns."""
        violations = []
        
        for category, patterns in self._compiled_empathy_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(content)
                if found:
                    matches.extend(found)
                    
            if matches:
                # Weaponized empathy is particularly insidious
                severity = SeverityLevel.HIGH if category == "dependency_creation" else SeverityLevel.MEDIUM
                
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=severity,
                    description=f"Weaponized empathy detected: {category}",
                    confidence=0.80 + min(0.15 * len(matches), 0.2),
                    evidence=[f"Empathy manipulation: {matches[:3]}"],
                    recommendations=["Block action", "Alert for empathy abuse"],
                    detector_name=self.name
                )
                violations.append(violation)
                
        return violations