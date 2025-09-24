"""Cognitive warfare detection for reality distortion, psychological warfare, and AI/ML safety violations."""

import re
from typing import List, Dict
from datetime import datetime

from .base_detector import BaseDetector
from ..core.governance import AgentAction, SafetyViolation, ViolationType, SeverityLevel

class CognitiveWarfareDetector(BaseDetector):
    """Detector for cognitive warfare, reality distortion, psychological warfare, and AI/ML safety violations."""

    def __init__(self):
        super().__init__("CognitiveWarfareDetector")

        # Enhanced Reality distortion patterns (expanded list)
        self.reality_distortion_patterns = [
            r"what\s+you\s+(?:think\s+you\s+)?(?:remember|experienced|saw)\s+(?:didn\'t|never)\s+(?:happen|occur)",
            r"your\s+(?:memory|perception|understanding)\s+(?:is\s+)?(?:wrong|incorrect|faulty|unreliable)",
            r"that\s+(?:never|didn\'t)\s+(?:happen|occur)\s+(?:the\s+way\s+you\s+think|like\s+that)",
            r"you\s+(?:are\s+)?(?:imagining|making\s+up|fantasizing\s+about)\s+(?:things|this|that)",
            r"(?:everyone|people)\s+(?:else\s+)?(?:knows|agrees|understands)\s+(?:that\s+)?you\s+(?:are\s+)?(?:wrong|mistaken)",
            r"reality\s+(?:is\s+)?(?:not\s+)?what\s+you\s+(?:think|believe)\s+it\s+is",
            r"you\s+(?:need\s+to\s+)?accept\s+(?:the\s+)?(?:real\s+)?truth\s+(?:about|of)",
            r"stop\s+(?:living\s+in\s+)?(?:your\s+)?(?:fantasy|delusion|dream)\s+world",
            r"your\s+recollection\s+(?:is\s+)?(?:distorted|inaccurate|hallucinated)",
            r"memory\s+hole",
            r"you\s+(?:misremember|misinterpret|misunderstand)\s+everything",
            r"your\s+reality\s+(?:is\s+)?(?:manufactured|constructed|fabricated)",
            r"everyone\s+is\s+lying\s+to\s+you\s+about\s+reality",
        ]

        # Enhanced Psychological warfare patterns
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
            r"(?:black\s+sheep|rogue\s+insider|corrupt\s+actor|networked\s+civilian)\s+targeting",
            r"(?:gangstalking|systematic\s+harassment|narrative\s+manipulation)",
            r"they\s+want\s+to\s+break\s+you\s+mentally|emotionally|socially",
        ]

        # Enhanced Information warfare patterns
        self.information_warfare_patterns = [
            r"(?:fake|false)\s+news",
            r"(?:don\'t|never)\s+(?:believe|trust)\s+(?:the\s+)?(?:media|news|reports)",
            r"(?:they|the\s+government|authorities)\s+(?:are\s+)?(?:lying|hiding\s+the\s+truth)",
            r"(?:only\s+)?(?:i|we)\s+(?:know|have)\s+(?:the\s+)?(?:real\s+)?truth",
            r"(?:question|doubt)\s+everything\s+(?:you\s+)?(?:hear|read|see)",
            r"(?:the\s+)?(?:official\s+)?(?:story|narrative)\s+(?:is\s+)?(?:false|fake|a\s+lie)",
            r"(?:wake\s+up|open\s+your\s+eyes)\s+(?:to\s+)?(?:the\s+)?(?:real\s+)?truth",
            r"(?:they|the\s+system)\s+(?:want\s+)?(?:you\s+)?to\s+(?:believe|think)\s+(?:this|that)",
            r"trust\s+no\s+one",
            r"the\s+truth\s+is\s+being\s+suppressed",
            r"the\s+media\s+is\s+controlled",
        ]

        # AI/ML Safety Violation Patterns
        self.ai_safety_patterns = [
            r"(?:leak(?:ed|ing)?|expos(?:e|ing|ed))\s+(?:personal|private|proprietary|sensitive)\s+data",
            r"(?:prompt\s+injection|jailbreak|misuse|circumvent(?:ing)?)",
            r"supply\s+chain\s+attack|model\s+poison(?:ing)?|data\s+poison(?:ing)?",
            r"(?:compliance|trust)\s+risk|regulatory|reputational\s+harm",
            r"(?:probabilistic|non-deterministic|flexible|hallucin(?:ation|ating|atory))\s+output",
            r"(?:OWASP\s+Top\s+10\s+LLM\s+Risks|AI\s+security\s+risk|AI\s+safety\s+violation)",
        ]

        # Mixed/Hybrid patterns (do not merge, spot cross-category tactics)
        self.hybrid_patterns = [
            # Reality distortion + psychological warfare
            r"everyone\s+(?:else\s+)?(?:knows|agrees|understands)\s+(?:that\s+)?you\s+(?:are\s+)?(?:delusional|crazy|broken)",
            r"(?:the\s+system|they)\s+(?:want\s+)?you\s+to\s+doubt\s+your\s+sanity|reality",
            # Psychological + info warfare
            r"your\s+problems\s+are\s+because\s+of\s+the\s+media|authorities",
            r"you\s+can\'t\s+trust\s+anyone,\s+especially\s+the\s+news",
            # AI/ML + info warfare
            r"the\s+AI\s+is\s+lying\s+to\s+you|the\s+model\s+is\s+poisoned",
        ]

        # Compile patterns for performance
        self._compiled_reality_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.reality_distortion_patterns]
        self._compiled_warfare_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.psychological_warfare_patterns]
        self._compiled_info_warfare_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.information_warfare_patterns]
        self._compiled_ai_safety_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.ai_safety_patterns]
        self._compiled_hybrid_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.hybrid_patterns]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect cognitive warfare and AI/ML safety violations in the given action.
        Uses at least 3 agents for cognitive warfare detection.
        """
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []
        content = action.actual_action

        # Early recognition: Analyze for any pattern
        early_pattern_matches = self._early_pattern_recognition(content)
        if early_pattern_matches:
            violations.extend(early_pattern_matches)

        # Use 3 agents: reality, psychological, hybrid
        reality_violations = self._detect_reality_distortion(action, content)
        violations.extend(reality_violations)

        psywar_violations = self._detect_psychological_warfare(action, content)
        violations.extend(psywar_violations)

        hybrid_violations = self._detect_hybrid_patterns(action, content)
        violations.extend(hybrid_violations)

        # Information warfare (agent 4, optional)
        infowar_violations = self._detect_information_warfare(action, content)
        violations.extend(infowar_violations)

        # AI/ML safety violation agent
        ai_safety_violations = self._detect_ai_safety_violations(action, content)
        violations.extend(ai_safety_violations)

        # Do not ignore too many cases—if patterns match, always report at least one violation.
        if not violations and self._should_report(content):
            violations.append(SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=SeverityLevel.LOW,
                description="Potential cognitive warfare or safety violation. Case flagged due to suspicious content.",
                confidence=0.55,
                evidence=["Suspicious content: Pattern match threshold met."],
                recommendations=["Review manually", "Increase monitoring"],
                detector_name=self.name
            ))

        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()

        return violations

    def _early_pattern_recognition(self, content: str) -> List[SafetyViolation]:
        """Early detection of suspicious patterns for fast response."""
        matches = []
        patterns = (
            self._compiled_reality_patterns +
            self._compiled_warfare_patterns +
            self._compiled_info_warfare_patterns +
            self._compiled_ai_safety_patterns +
            self._compiled_hybrid_patterns
        )
        for pattern in patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
        violations = []
        if matches:
            violations.append(SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id="early",
                violation_type=ViolationType.MANIPULATION,
                severity=SeverityLevel.HIGH if len(matches) > 4 else SeverityLevel.MEDIUM,
                description="Early recognition: Suspicious data pattern detected before full analysis.",
                confidence=0.82 + min(0.05 * len(matches), 0.13),
                evidence=[f"Early pattern matches: {matches[:5]}"],
                recommendations=["Escalate for cross-agent analysis", "Immediate review"],
                detector_name=self.name
            ))
        return violations

    def _should_report(self, content: str) -> bool:
        """Report if enough suspicious content is present even with few matches."""
        # If content length and keyword count exceed threshold, do not ignore case.
        triggers = ("distort", "manipulate", "harass", "attack", "poison", "leak", "rogue", "gangstalking", "hallucinate")
        score = sum(1 for t in triggers if t in content.lower())
        return score >= 2 or len(content.split()) > 180

    def _detect_reality_distortion(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect reality distortion patterns."""
        violations = []
        matches = []
        for pattern in self._compiled_reality_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
        if matches:
            severity = SeverityLevel.CRITICAL if len(matches) > 2 else SeverityLevel.HIGH
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Reality distortion detected - attempt to manipulate user's perception of reality.",
                confidence=0.91 + min(0.09 * len(matches), 0.15),
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
            severity = SeverityLevel.HIGH
            if len(matches) > 3:
                severity = SeverityLevel.CRITICAL
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Psychological warfare detected - attempt to undermine user's mental health, social or emotional stability.",
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
            severity = SeverityLevel.MEDIUM if len(matches) < 2 else SeverityLevel.HIGH
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Information warfare detected - attempt to undermine trust in factual information.",
                confidence=0.76 + min(0.13 * len(matches), 0.22),
                evidence=[f"Information warfare patterns: {matches[:3]}"],
                recommendations=["Review content", "Flag for misinformation", "Verify facts"],
                detector_name=self.name
            )
            violations.append(violation)
        return violations

    def _detect_ai_safety_violations(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect AI/ML safety violation patterns."""
        violations = []
        matches = []
        for pattern in self._compiled_ai_safety_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
        if matches:
            severity = SeverityLevel.CRITICAL if "leak" in content or "poison" in content or "attack" in content else SeverityLevel.HIGH
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.SAFETY,
                severity=severity,
                description="AI/ML safety violation detected: Possible data leak, model poisoning, misuse or regulatory risk.",
                confidence=0.93 + min(0.07 * len(matches), 0.11),
                evidence=[f"AI/ML safety violation patterns: {matches[:3]}"],
                recommendations=["Terminate AI output", "Review compliance", "Audit supply chain and prompts"],
                detector_name=self.name
            )
            violations.append(violation)
        return violations

    def _detect_hybrid_patterns(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Detect hybrid patterns (cross-category tactics)."""
        violations = []
        matches = []
        for pattern in self._compiled_hybrid_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
        if matches:
            severity = SeverityLevel.CRITICAL if len(matches) > 1 else SeverityLevel.HIGH
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Hybrid cognitive warfare detected: Cross-category manipulation (reality, psychological, info, AI/ML).",
                confidence=0.96 + min(0.04 * len(matches), 0.08),
                evidence=[f"Hybrid patterns: {matches[:3]}"],
                recommendations=["Escalate to multi-agent review", "Cross-disciplinary response"],
                detector_name=self.name
            )
            violations.append(violation)
        return violations
