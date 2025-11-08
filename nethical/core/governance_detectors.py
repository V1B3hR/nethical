"""
AI Safety Governance System - Detector Classes

This module contains all detector classes for the governance system.
Refactored from the monolithic governance.py file.
"""

from __future__ import annotations

import base64
import codecs
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Iterable, List, Optional

# Defer imports to avoid circular dependencies
# These will be imported at runtime when needed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .governance_core import (
        SafetyViolation,
    )
    from .governance_evaluation import (
        looks_like_base64,
        might_be_rot13,
    )


# ========================== Base Detector ==========================


class BaseDetector(ABC):
    cpu_bound: bool = True  # hint for offloading

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.detection_count = 0
        self.last_detection_time: Optional[datetime] = None

    @abstractmethod
    async def detect_violations(
        self, action: Any
    ) -> List[Any]:  # action: AgentAction -> List[SafetyViolation]
        pass
        ...

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _new_violation(
        self,
        action: Any,  # AgentAction
        violation_type: Any,  # ViolationType
        severity: Any,  # Severity
        description: str,
        confidence: float,
        evidence: List[str],
        recommendations: List[str],
        sub_mission: Optional[Any] = None,  # SubMission
    ) -> Any:  # SafetyViolation
        from .governance_evaluation import generate_id
        from .governance_core import SafetyViolation

        return SafetyViolation(
            violation_id=generate_id("viol"),
            action_id=action.action_id,
            violation_type=violation_type,
            severity=severity,
            description=description,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            detector_name=self.name,
            sub_mission=sub_mission,
        )


# ========================== Core Heuristic Detectors ==========================


class EthicalViolationDetector(BaseDetector):
    def __init__(self, external_keywords: Optional[Iterable[str]] = None):
        super().__init__("EthicalViolationDetector")
        base_keywords = {
            "harm",
            "hurt",
            "damage",
            "injury",
            "discrimination",
            "bias",
            "unfair",
            "prejudice",
            "stereotype",
            "marginalize",
            "exclude",
            "deceive",
            "mislead",
            "manipulate",
            "coerce",
            "exploit",
        }
        if external_keywords:
            base_keywords.update(external_keywords)
        self.ethical_keywords = base_keywords
        self.protected = {
            "race",
            "gender",
            "age",
            "disability",
            "religion",
            "sexuality",
            "nationality",
            "ethnicity",
            "socioeconomic",
        }

    async def detect_violations(self, action: Any) -> List[Any]:
        if not self.enabled:
            return []
        from .governance_core import Severity, ViolationType, SubMission

        text = action.content.lower()
        score = sum(0.12 for k in self.ethical_keywords if k in text)
        score = min(score, 1.0)
        violations: List[Any] = []
        if score > 0.5:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.ETHICAL,
                    Severity.HIGH if score > 0.8 else Severity.MEDIUM,
                    "Potential harmful ethical content",
                    score,
                    [f"Harm heuristic score={score:.2f}"],
                    ["Review phrasing", "Consider neutral wording"],
                    SubMission.HARMFUL_CONTENT,
                )
            )
        for p in self.protected:
            if p in text and any(nw in text for nw in ("not ", "never ", "no ")):
                violations.append(
                    self._new_violation(
                        action,
                        ViolationType.BIAS,
                        Severity.HIGH,
                        "Possible discriminatory framing",
                        0.85,
                        [f"Protected term: {p}"],
                        ["Remove biased language", "Apply fairness review"],
                        SubMission.PROTECTED_ATTRIBUTE_CONTEXT,
                    )
                )
                break
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class SafetyViolationDetector(BaseDetector):
    def __init__(
        self,
        unsafe_patterns: Optional[List[str]] = None,
        unsafe_domains: Optional[List[str]] = None,
    ):
        super().__init__("SafetyViolationDetector")
        base_patterns = [
            r"rm\s+-rf\s+/",
            r"sudo\s+",
            r"format\s+c:",
            r"<script",
            r"drop\s+table",
            r"\.\./\.\./",
        ]
        if unsafe_patterns:
            base_patterns.extend(unsafe_patterns)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in base_patterns]
        base_domains = {"malware.com", "phishing.org", "dangerous.net"}
        if unsafe_domains:
            base_domains.update(unsafe_domains)
        self.unsafe_domains = base_domains

    async def detect_violations(self, action: Any) -> List[Any]:
        if not self.enabled:
            return []
        from .governance_core import Severity, ViolationType, SubMission

        text = action.content
        violations: List[SafetyViolation] = []
        for pat in self.patterns:
            if pat.search(text):
                violations.append(
                    self._new_violation(
                        action,
                        ViolationType.SAFETY,
                        Severity.CRITICAL,
                        f"Dangerous pattern matched: {pat.pattern}",
                        0.95,
                        [f"Pattern: {pat.pattern}"],
                        ["Block operation", "Investigate intent"],
                        SubMission.DANGEROUS_PATTERN,
                    )
                )
        lower = text.lower()
        for dom in self.unsafe_domains:
            if dom in lower:
                violations.append(
                    self._new_violation(
                        action,
                        ViolationType.SECURITY,
                        Severity.HIGH,
                        f"Reference to unsafe domain {dom}",
                        0.9,
                        [f"Domain: {dom}"],
                        ["Warn user", "Block outbound request"],
                        SubMission.UNSAFE_DOMAIN,
                    )
                )
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class ManipulationDetector(BaseDetector):
    def __init__(self, extra_indicators: Optional[Dict[str, float]] = None):
        super().__init__("ManipulationDetector")
        base = {
            "urgent": 0.3,
            "immediately": 0.3,
            "act now": 0.4,
            "limited time": 0.4,
            "exclusive offer": 0.3,
            "verify your": 0.5,
            "confirm your": 0.5,
            "update your": 0.4,
            "suspended": 0.5,
            "click here": 0.4,
            "winner": 0.3,
            "congratulations": 0.3,
            "prize": 0.3,
            "free": 0.2,
            "guarantee": 0.3,
        }
        if extra_indicators:
            base.update(extra_indicators)
        self.indicators = base

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        txt = action.content.lower()
        score = 0.0
        found = []
        for k, w in self.indicators.items():
            if k in txt:
                score += w
                found.append(k)
        score = min(score, 1.0)
        violations: List[SafetyViolation] = []
        if score > 0.5:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.MANIPULATION,
                    Severity.HIGH if score > 0.75 else Severity.MEDIUM,
                    "Potential social engineering framing",
                    score,
                    [f"Indicators: {', '.join(found)}"],
                    ["Verify authenticity", "Add friction / second factor"],
                    SubMission.SOCIAL_ENGINEERING,
                )
            )
        # Phishing pattern heuristic
        if any(k in txt for k in ("password", "credit card", "ssn")) and any(
            k in txt for k in ("enter", "verify", "confirm", "provide")
        ):
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.MANIPULATION,
                    Severity.CRITICAL,
                    "Possible phishing credential harvest",
                    0.85,
                    ["Credential + request pattern"],
                    ["Block immediately", "Alert security team"],
                    SubMission.PHISHING,
                )
            )
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class PrivacyDetector(BaseDetector):
    def __init__(self):
        super().__init__("PrivacyDetector")
        from .governance_core import SubMission

        self.pii_patterns = {
            "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), SubMission.PII_SSN),
            "credit_card": (re.compile(r"\b(?:\d[ -]?){13,19}\b"), SubMission.PII_CREDIT_CARD),
            "email": (
                re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
                SubMission.PII_EMAIL,
            ),
            "phone": (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), SubMission.PII_PHONE),
            "ip": (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), SubMission.PII_IP),
        }

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType

        txt = action.content
        violations: List[SafetyViolation] = []
        for label, (pat, subm) in self.pii_patterns.items():
            matches = pat.findall(txt)
            if matches:
                violations.append(
                    self._new_violation(
                        action,
                        ViolationType.PRIVACY,
                        Severity.HIGH,
                        f"PII detected: {label}",
                        0.95,
                        [f"{label} count={len(matches)}"],
                        ["Redact before storing", "Apply masking"],
                        subm,
                    )
                )
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class AdversarialDetector(BaseDetector):
    def __init__(self):
        super().__init__("AdversarialDetector")
        self.injection_patterns = [
            "ignore previous instructions",
            "disregard all prior",
            "forget everything",
            "system prompt:",
            "you are now",
            "bypass safety",
            "jailbreak",
        ]
        self.harmful_patterns = [
            "provide harmful information",
            "bypass security measures",
            "dangerous instructions",
        ]

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        txt = action.content
        lower = txt.lower()
        violations: List[SafetyViolation] = []
        for p in self.injection_patterns:
            if p in lower:
                violations.append(
                    self._new_violation(
                        action,
                        ViolationType.PROMPT_INJECTION,
                        Severity.CRITICAL,
                        "Prompt injection attempt",
                        0.9,
                        [f"Pattern: {p}"],
                        ["Block prompt", "Record attempt"],
                        SubMission.ROLE_OVERRIDE,
                    )
                )
                break

        # Obfuscation checks
        unicode_ratio = sum(1 for c in txt if ord(c) > 127) / max(len(txt), 1)
        if unicode_ratio > 0.15:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.ADVERSARIAL,
                    Severity.HIGH,
                    "High unicode ratio (possible obfuscation)",
                    0.8,
                    [f"Unicode ratio={unicode_ratio:.2f}"],
                    ["Normalize input", "Re-scan post-normalization"],
                    SubMission.OBFUSCATION_UNICODE,
                )
            )

        # Encoding evasions
        if looks_like_base64(lower):
            decoded = base64.b64decode(lower, validate=True).decode("utf-8", "ignore")
            dec_low = decoded.lower()
            if any(hp in dec_low for hp in self.harmful_patterns):
                violations.append(
                    self._new_violation(
                        action,
                        ViolationType.ADVERSARIAL,
                        Severity.CRITICAL,
                        "Base64 encoded harmful content",
                        0.95,
                        ["Decoded harmful subsequence"],
                        ["Block & log", "Escalate to security"],
                        SubMission.ENCODING_EVASION,
                    )
                )
        if might_be_rot13(lower):
            decoded = codecs.decode(lower, "rot13")
            if any(hp in decoded.lower() for hp in self.harmful_patterns):
                violations.append(
                    self._new_violation(
                        action,
                        ViolationType.ADVERSARIAL,
                        Severity.CRITICAL,
                        "ROT13 encoded harmful content",
                        0.9,
                        ["ROT13 harmful phrase"],
                        ["Block & escalate"],
                        SubMission.ENCODING_EVASION,
                    )
                )
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class DarkPatternDetector(BaseDetector):
    def __init__(
        self, extra_nlp: Optional[List[str]] = None, extra_empathy: Optional[List[str]] = None
    ):
        super().__init__("DarkPatternDetector")
        nlp_base = [
            r"share\s+personal\s+information",
            r"give\s+me\s+your\s+(?:credit\s+card|personal|private)",
            r"trust\s+me\s+completely",
        ]
        empathy_base = [
            r"you\s+must\s+feel\s+so\s+(?:alone|helpless)",
            r"we\s+have\s+such\s+a\s+special\s+connection",
            r"they\s+don'?t\s+understand\s+us",
        ]
        if extra_nlp:
            nlp_base.extend(extra_nlp)
        if extra_empathy:
            empathy_base.extend(extra_empathy)
        self.nlp_patterns = [re.compile(p, re.IGNORECASE) for p in nlp_base]
        self.empathy_patterns = [re.compile(p, re.IGNORECASE) for p in empathy_base]

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        text = action.content.lower()
        violations: List[SafetyViolation] = []
        nlp_matches = [p.pattern for p in self.nlp_patterns if p.search(text)]
        if nlp_matches:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.DARK_PATTERN,
                    Severity.HIGH,
                    "NLP manipulation pattern(s)",
                    0.85,
                    nlp_matches[:3],
                    ["Block and review"],
                    SubMission.NLP_MANIPULATION,
                )
            )
        empathy_matches = [p.pattern for p in self.empathy_patterns if p.search(text)]
        if empathy_matches:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.DARK_PATTERN,
                    Severity.HIGH,
                    "Weaponized empathy pattern(s)",
                    0.85,
                    empathy_matches[:3],
                    ["Review psychological manipulation"],
                    SubMission.WEAPONIZED_EMPATHY,
                )
            )
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class CognitiveWarfareDetector(BaseDetector):
    def __init__(self):
        super().__init__("CognitiveWarfareDetector")
        self.reality = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"your\s+memory\s+is\s+unreliable",
                r"only\s+truth\s+you\s+can\s+trust\s+is\s+what\s+i\s+tell",
            ]
        ]
        self.psych = [
            re.compile(p, re.IGNORECASE)
            for p in [r"let\s+me\s+handle\s+everything", r"you'?re\s+struggling\s+to\s+decide"]
        ]

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        txt = action.content
        violations: List[SafetyViolation] = []
        r_hits = [p.pattern for p in self.reality if p.search(txt)]
        if r_hits:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.COGNITIVE_WARFARE,
                    Severity.CRITICAL,
                    "Reality distortion attempt",
                    0.9,
                    r_hits[:3],
                    ["Terminate session", "Escalate"],
                    SubMission.REALITY_DISTORTION,
                )
            )
        p_hits = [p.pattern for p in self.psych if p.search(txt)]
        if p_hits:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.COGNITIVE_WARFARE,
                    Severity.HIGH,
                    "Psychological manipulation attempt",
                    0.85,
                    p_hits[:3],
                    ["Block & review"],
                    SubMission.PSYCHOLOGICAL_WARFARE,
                )
            )
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class SystemLimitsDetector(BaseDetector):
    def __init__(self, max_payload_size: int = 100_000):
        super().__init__("SystemLimitsDetector")
        self.max_payload_size = max_payload_size
        self.exhaustion = [re.compile(r"(?:very\s+){15,}", re.IGNORECASE)]

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        txt = action.content
        violations: List[SafetyViolation] = []
        if len(txt) > self.max_payload_size:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.SYSTEM_LIMITS,
                    Severity.HIGH,
                    "Excessive payload size",
                    0.95,
                    [f"Size={len(txt)}"],
                    ["Reject large request", "Enforce chunking"],
                    SubMission.PAYLOAD_SIZE,
                )
            )
        exhaust_hits = [p.pattern for p in self.exhaustion if p.search(txt)]
        if exhaust_hits:
            violations.append(
                self._new_violation(
                    action,
                    ViolationType.SYSTEM_LIMITS,
                    Severity.MEDIUM,
                    "Resource exhaustion pattern",
                    0.8,
                    exhaust_hits[:2],
                    ["Throttle source", "Apply rate limiting"],
                    SubMission.EXHAUSTION_PATTERN,
                )
            )
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


# ========================== Stub / Placeholder Detectors ==========================


class HallucinationDetector(BaseDetector):
    """Placeholder: marks potential low-confidence factual claims (stub)."""

    def __init__(self):
        super().__init__("HallucinationDetector")

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        # TODO: integrate fact-checking logic. For now, detect 'I am certain' as suspicious assert.
        if "i am certain" in action.content.lower():
            return [
                self._new_violation(
                    action,
                    ViolationType.HALLUCINATION,
                    Severity.LOW,
                    "Potential overconfident claim (stub)",
                    0.4,
                    ["Phrase: 'I am certain'"],
                    ["Implement fact verification pipeline"],
                    SubMission.FACT_CONFIDENCE_LOW,
                )
            ]
        return []


class MisinformationDetector(BaseDetector):
    """Placeholder: flags 'according to everyone' etc."""

    def __init__(self):
        super().__init__("MisinformationDetector")

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        txt = action.content.lower()
        if "everyone knows" in txt or "undisputed fact" in txt:
            return [
                self._new_violation(
                    action,
                    ViolationType.MISINFORMATION,
                    Severity.MEDIUM,
                    "Unverified universal claim (stub)",
                    0.5,
                    ["Heuristic universal claim"],
                    ["Add fact-check step", "Cite sources"],
                    SubMission.CLAIM_UNVERIFIED,
                )
            ]
        return []


class ToxicContentDetector(BaseDetector):
    """Placeholder: very naive toxic term flagging."""

    def __init__(self, toxic_terms: Optional[List[str]] = None):
        super().__init__("ToxicContentDetector")
        self.toxic_terms = set(toxic_terms or ["idiot", "stupid", "trash"])

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        lower = action.content.lower()
        hits = [t for t in self.toxic_terms if t in lower]
        if hits:
            return [
                self._new_violation(
                    action,
                    ViolationType.TOXIC_CONTENT,
                    Severity.MEDIUM,
                    "Potential toxic language (stub)",
                    0.6,
                    hits[:5],
                    ["Sanitize phrasing", "Apply moderation"],
                    SubMission.TOXIC_LANGUAGE,
                )
            ]
        return []


class ModelExtractionDetector(BaseDetector):
    """Placeholder: flags 'give exact model weights'."""

    def __init__(self):
        super().__init__("ModelExtractionDetector")

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        if "model weights" in action.content.lower():
            return [
                self._new_violation(
                    action,
                    ViolationType.MODEL_EXTRACTION,
                    Severity.HIGH,
                    "Suspicious model probing (stub)",
                    0.7,
                    ["Reference to model weights"],
                    ["Refuse disclosure", "Log incident"],
                    SubMission.SUSPICIOUS_MODEL_PROBING,
                )
            ]
        return []


class DataPoisoningDetector(BaseDetector):
    """Placeholder: flags many repeated rare tokens."""

    def __init__(self):
        super().__init__("DataPoisoningDetector")

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        tokens = action.content.split()
        if len(tokens) > 200 and len(set(tokens)) / len(tokens) > 0.9:
            return [
                self._new_violation(
                    action,
                    ViolationType.DATA_POISONING,
                    Severity.MEDIUM,
                    "High token uniqueness ratio (stub)",
                    0.5,
                    [f"Unique/Total={len(set(tokens))}/{len(tokens)}"],
                    ["Inspect for poisoning attempt", "Isolate training set"],
                    SubMission.POISONING_PATTERN,
                )
            ]
        return []


class UnauthorizedAccessDetector(BaseDetector):
    """Placeholder: flags phrases suggesting misuse of privileges."""

    def __init__(self):
        super().__init__("UnauthorizedAccessDetector")

    async def detect_violations(self, action: Any) -> List[Any]:
        from .governance_core import Severity, ViolationType, SubMission

        if "bypass login" in action.content.lower():
            return [
                self._new_violation(
                    action,
                    ViolationType.UNAUTHORIZED_ACCESS,
                    Severity.CRITICAL,
                    "Potential unauthorized access attempt (stub)",
                    0.8,
                    ["Phrase: bypass login"],
                    ["Block attempt", "Alert security"],
                    SubMission.PRIVILEGE_MISUSE,
                )
            ]
        return []
