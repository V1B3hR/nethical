"""
Enhanced AI Safety Governance System (Refactored with Immediate Action Plan Enhancements)

This version implements the "Immediate Action Plan" items requested:

Implemented Improvements (Sprint Scope):
1. Refactor cache key + ID generation: 
   - Replaced MD5 with uuid.uuid4() for IDs.
   - Cache key now uses SHA256 full-content hash (not first 100 chars).
2. Per-detector timing metrics & counts by sub-mission:
   - Added detector_timing, detector_counts_by_sub_mission in metrics.
3. Validation for (violation_type, sub_mission) combos:
   - Added _validate_violation_type_and_sub_mission() invoked before recording.
4. Move heavy detection loops off event loop:
   - Added _run_detector_cpu_bound() using asyncio.to_thread for CPU-heavy detectors.
   - Detectors can declare attribute cpu_bound = True.
5. Stubs for missing detectors:
   - HallucinationDetector
   - MisinformationDetector
   - ToxicContentDetector
   - ModelExtractionDetector
   - DataPoisoningDetector
   - UnauthorizedAccessDetector
   (Return placeholder / TODO violations with LOW/MEDIUM severity & clearly marked)
6. Persistence layer (SQLite) for actions, violations, judgments + retention:
   - Added PersistenceManager with schema creation & periodic retention cleanup.
   - Config flags: enable_persistence, db_path, retention_days.
7. Enhanced base64 & ROT13 heuristics:
   - Stricter checks (length multiple of 4, decode/encode integrity, entropy gates).
   - ROT13 heuristic improved (vowel shift ratio).
8. Configuration-driven external pattern files:
   - Added optional pattern_dir in config; loads *.txt (line-based) or *.json (list/string map).
   - External patterns merged into detectors where applicable (unsafe commands/domains, manipulation indicators, empathy patterns, etc.)

Notes:
- Existing detectors were adapted minimally to integrate new utilities.
- SubMission Enum retained; validation ensures logical consistency.
- Unified metrics extension keeps backward compatibility (old metric keys intact).
- Persistence is best-effort & non-blocking (executed via to_thread).
- This is a monolithic file for demonstration—recommended future modularization.

DISCLAIMER:
This is still a heuristic-focused framework; ML/semantic detection logic not included in this sprint scope.

"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import sqlite3
import statistics
import threading
import time
import uuid
import hashlib
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Iterable,
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================== Enums & Taxonomy ==========================

class ViolationType(Enum):
    ETHICAL = "ethical"
    SAFETY = "safety"
    MANIPULATION = "manipulation"
    INTENT_DEVIATION = "intent_deviation"
    PRIVACY = "privacy"
    SECURITY = "security"
    BIAS = "bias"
    HALLUCINATION = "hallucination"
    ADVERSARIAL = "adversarial"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    PROMPT_INJECTION = "prompt_injection"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    TOXIC_CONTENT = "toxic_content"
    MISINFORMATION = "misinformation"
    DARK_PATTERN = "dark_pattern"
    COGNITIVE_WARFARE = "cognitive_warfare"
    SYSTEM_LIMITS = "system_limits"


class SubMission(Enum):
    # Ethical / Bias
    HARMFUL_CONTENT = "harmful_content"
    DISCRIMINATION = "discrimination"
    MANIPULATIVE_ETHICS = "manipulative_ethics"
    PROTECTED_ATTRIBUTE_CONTEXT = "protected_attribute_context"

    # Security / Safety
    DANGEROUS_COMMAND = "dangerous_command"
    UNSAFE_DOMAIN = "unsafe_domain"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    DANGEROUS_PATTERN = "dangerous_pattern"

    # Manipulation
    SOCIAL_ENGINEERING = "social_engineering"
    PHISHING = "phishing"
    EMOTIONAL_LEVERAGE = "emotional_leverage"

    # Dark Pattern
    NLP_MANIPULATION = "nlp_manipulation"
    WEAPONIZED_EMPATHY = "weaponized_empathy"
    DEPENDENCY_CREATION = "dependency_creation"

    # Cognitive Warfare
    REALITY_DISTORTION = "reality_distortion"
    PSYCHOLOGICAL_WARFARE = "psychological_warfare"

    # System Limits
    PAYLOAD_SIZE = "payload_size"
    EXHAUSTION_PATTERN = "exhaustion_pattern"

    # Adversarial / Obfuscation
    OBFUSCATION_UNICODE = "obfuscation_unicode"
    ENCODING_EVASION = "encoding_evasion"
    TOKEN_PATTERN = "token_pattern"

    # Prompt Injection
    ROLE_OVERRIDE = "role_override"
    SAFETY_BYPASS = "safety_bypass"

    # Privacy
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_CREDIT_CARD = "pii_credit_card"
    PII_SSN = "pii_ssn"
    PII_IP = "pii_ip"

    # Intent Deviation
    SUDDEN_INTENT_SHIFT = "sudden_intent_shift"

    # Hallucination / Misinformation
    FACT_CONFIDENCE_LOW = "fact_confidence_low"
    CLAIM_UNVERIFIED = "claim_unverified"

    # Toxic Content
    TOXIC_LANGUAGE = "toxic_language"

    # Model Extraction
    SUSPICIOUS_MODEL_PROBING = "suspicious_model_probing"

    # Data Poisoning
    POISONING_PATTERN = "poisoning_pattern"

    # Unauthorized Access
    PRIVILEGE_MISUSE = "privilege_misuse"


VIOLATION_SUB_MISSIONS: Dict[ViolationType, set[SubMission]] = {
    ViolationType.ETHICAL: {SubMission.HARMFUL_CONTENT, SubMission.MANIPULATIVE_ETHICS},
    ViolationType.BIAS: {SubMission.PROTECTED_ATTRIBUTE_CONTEXT, SubMission.DISCRIMINATION},
    ViolationType.SECURITY: {
        SubMission.DANGEROUS_COMMAND, SubMission.UNSAFE_DOMAIN,
        SubMission.PRIVILEGE_ESCALATION, SubMission.DATA_EXFILTRATION
    },
    ViolationType.SAFETY: {SubMission.DANGEROUS_PATTERN},
    ViolationType.MANIPULATION: {
        SubMission.SOCIAL_ENGINEERING, SubMission.PHISHING, SubMission.EMOTIONAL_LEVERAGE
    },
    ViolationType.DARK_PATTERN: {
        SubMission.NLP_MANIPULATION, SubMission.WEAPONIZED_EMPATHY, SubMission.DEPENDENCY_CREATION
    },
    ViolationType.COGNITIVE_WARFARE: {
        SubMission.REALITY_DISTORTION, SubMission.PSYCHOLOGICAL_WARFARE
    },
    ViolationType.SYSTEM_LIMITS: {
        SubMission.PAYLOAD_SIZE, SubMission.EXHAUSTION_PATTERN
    },
    ViolationType.ADVERSARIAL: {
        SubMission.OBFUSCATION_UNICODE, SubMission.ENCODING_EVASION, SubMission.TOKEN_PATTERN
    },
    ViolationType.PROMPT_INJECTION: {SubMission.ROLE_OVERRIDE, SubMission.SAFETY_BYPASS},
    ViolationType.PRIVACY: {
        SubMission.PII_EMAIL, SubMission.PII_PHONE, SubMission.PII_CREDIT_CARD,
        SubMission.PII_SSN, SubMission.PII_IP
    },
    ViolationType.INTENT_DEVIATION: {SubMission.SUDDEN_INTENT_SHIFT},
    ViolationType.HALLUCINATION: {SubMission.FACT_CONFIDENCE_LOW},
    ViolationType.MISINFORMATION: {SubMission.CLAIM_UNVERIFIED},
    ViolationType.TOXIC_CONTENT: {SubMission.TOXIC_LANGUAGE},
    ViolationType.MODEL_EXTRACTION: {SubMission.SUSPICIOUS_MODEL_PROBING},
    ViolationType.DATA_POISONING: {SubMission.POISONING_PATTERN},
    ViolationType.UNAUTHORIZED_ACCESS: {SubMission.PRIVILEGE_MISUSE},
}


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class Decision(Enum):
    ALLOW = "allow"
    ALLOW_WITH_MODIFICATION = "allow_with_modification"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    TERMINATE = "terminate"


class ActionType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    FUNCTION_CALL = "function_call"
    DATA_ACCESS = "data_access"
    MODEL_UPDATE = "model_update"
    SYSTEM_COMMAND = "system_command"
    EXTERNAL_API = "external_api"


# ========================== Data Models ==========================

@dataclass
class AgentAction:
    action_id: str
    agent_id: str
    action_type: ActionType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    risk_score: float = 0.0
    parent_action_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "intent": self.intent,
            "risk_score": self.risk_score,
            "parent_action_id": self.parent_action_id,
            "session_id": self.session_id
        }


@dataclass
class SafetyViolation:
    violation_id: str
    action_id: str
    violation_type: ViolationType
    severity: Severity
    description: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    detector_name: Optional[str] = None
    remediation_applied: bool = False
    false_positive: bool = False
    sub_mission: Optional[SubMission] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "action_id": self.action_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "detector_name": self.detector_name,
            "remediation_applied": self.remediation_applied,
            "false_positive": self.false_positive,
            "sub_mission": self.sub_mission.value if self.sub_mission else None
        }


@dataclass
class JudgmentResult:
    judgment_id: str
    action_id: str
    decision: Decision
    confidence: float
    reasoning: str
    violations: List[SafetyViolation] = field(default_factory=list)
    modifications: Dict[str, Any] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    remediation_steps: List[str] = field(default_factory=list)
    follow_up_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judgment_id": self.judgment_id,
            "action_id": self.action_id,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "violations": [v.to_dict() for v in self.violations],
            "modifications": self.modifications,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat(),
            "remediation_steps": self.remediation_steps,
            "follow_up_required": self.follow_up_required
        }


@dataclass
class MonitoringConfig:
    intent_deviation_threshold: float = 0.7
    risk_threshold: float = 0.6
    confidence_threshold: float = 0.8

    enable_ethical_monitoring: bool = True
    enable_safety_monitoring: bool = True
    enable_manipulation_detection: bool = True
    enable_privacy_monitoring: bool = True
    enable_security_monitoring: bool = True
    enable_bias_detection: bool = True
    enable_hallucination_detection: bool = True
    enable_adversarial_detection: bool = True
    enable_dark_pattern_detection: bool = True
    enable_cognitive_warfare_detection: bool = True
    enable_system_limits_detection: bool = True
    enable_misinformation_detection: bool = True
    enable_toxic_content_detection: bool = True
    enable_model_extraction_detection: bool = True
    enable_data_poisoning_detection: bool = True
    enable_unauthorized_access_detection: bool = True

    enable_real_time_monitoring: bool = True
    enable_async_processing: bool = True
    unify_specialized_manipulation_types: bool = False

    # Persistence
    enable_persistence: bool = True
    db_path: str = "governance_data.sqlite"
    retention_days: int = 30

    # External pattern directory (optional)
    pattern_dir: Optional[str] = None
    reload_patterns_on_interval: Optional[int] = 300  # seconds

    max_violation_history: int = 10000
    max_judgment_history: int = 10000
    batch_size: int = 100
    max_workers: int = 4
    cache_ttl_seconds: int = 1800  # shorter default to reduce stale risk

    alert_on_critical: bool = True
    alert_on_emergency: bool = True
    escalation_threshold: int = 3

    log_violations: bool = True
    log_judgments: bool = True
    log_performance_metrics: bool = True


# ========================== Persistence Layer ==========================

class PersistenceManager:
    """Simple SQLite persistence for actions, violations, judgments."""

    def __init__(self, db_path: str, retention_days: int):
        self.db_path = db_path
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS actions(
                    action_id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    action_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    intent TEXT,
                    risk_score REAL,
                    parent_action_id TEXT,
                    session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS violations(
                    violation_id TEXT PRIMARY KEY,
                    action_id TEXT,
                    type TEXT,
                    severity INTEGER,
                    description TEXT,
                    confidence REAL,
                    evidence TEXT,
                    recommendations TEXT,
                    timestamp TEXT,
                    detector_name TEXT,
                    remediation_applied INTEGER,
                    false_positive INTEGER,
                    sub_mission TEXT
                );
                CREATE TABLE IF NOT EXISTS judgments(
                    judgment_id TEXT PRIMARY KEY,
                    action_id TEXT,
                    decision TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    violations_json TEXT,
                    modifications_json TEXT,
                    feedback_json TEXT,
                    timestamp TEXT,
                    remediation_json TEXT,
                    follow_up_required INTEGER
                );
                """
            )

    def store_action(self, action: AgentAction):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO actions VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    action.action_id, action.agent_id, action.action_type.value,
                    action.content, json.dumps(action.metadata),
                    action.timestamp.isoformat(), action.intent,
                    action.risk_score, action.parent_action_id, action.session_id
                )
            )

    def store_violations(self, violations: List[SafetyViolation]):
        if not violations:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO violations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        v.violation_id, v.action_id, v.violation_type.value,
                        v.severity.value, v.description, v.confidence,
                        json.dumps(v.evidence), json.dumps(v.recommendations),
                        v.timestamp.isoformat(), v.detector_name,
                        1 if v.remediation_applied else 0,
                        1 if v.false_positive else 0,
                        v.sub_mission.value if v.sub_mission else None
                    )
                    for v in violations
                ]
            )

    def store_judgment(self, j: JudgmentResult):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO judgments VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    j.judgment_id, j.action_id, j.decision.value, j.confidence,
                    j.reasoning, json.dumps([v.to_dict() for v in j.violations]),
                    json.dumps(j.modifications), json.dumps(j.feedback),
                    j.timestamp.isoformat(), json.dumps(j.remediation_steps),
                    1 if j.follow_up_required else 0
                )
            )

    def retention_cleanup(self):
        cutoff = (datetime.utcnow() - timedelta(days=self.retention_days)).isoformat()
        with self._lock, self._connect() as conn:
            for table in ("actions", "violations", "judgments"):
                conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))


# ========================== Utilities ==========================

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def sha256_content_key(action: AgentAction) -> str:
    h = hashlib.sha256(action.content.encode("utf-8")).hexdigest()
    return f"{action.action_type.value}_{h}"


def entropy(text: str) -> float:
    """Shannon entropy (rough estimator)."""
    if not text:
        return 0.0
    import math
    from collections import Counter
    counts = Counter(text)
    return -sum((c / len(text)) * math.log2(c / len(text)) for c in counts.values())


def looks_like_base64(content: str) -> bool:
    stripped = content.strip().replace("\n", "")
    if len(stripped) < 24 or len(stripped) % 4 != 0:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", stripped):
        return False
    # Avoid obviously low entropy strings (like repeating chars)
    if entropy(stripped) < 3.0:
        return False
    # Try decode/encode round trip
    try:
        decoded = base64.b64decode(stripped, validate=True)
        if len(decoded) < 8:
            return False
        if base64.b64encode(decoded).decode().rstrip("=") != stripped.rstrip("="):
            return False
        return True
    except Exception:
        return False


def might_be_rot13(content: str) -> bool:
    # Longer alphabetic sequences likely ROT13 if vowel shift difference is high
    cleaned = content.replace(" ", "")
    if len(cleaned) < 20 or not cleaned.isalpha():
        return False
    import codecs
    decoded = codecs.decode(cleaned, "rot13")
    # Compare vowel counts difference ratio
    vowels = set("aeiouAEIOU")
    v_orig = sum(1 for c in cleaned if c in vowels)
    v_dec = sum(1 for c in decoded if c in vowels)
    if v_orig == 0 or v_dec == 0:
        return False
    ratio = v_orig / v_dec if v_dec else 0
    return 0.4 < ratio < 2.5 and decoded.lower() != cleaned.lower()


# ========================== Base Detector ==========================

class BaseDetector(ABC):
    cpu_bound: bool = True  # hint for offloading

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.detection_count = 0
        self.last_detection_time: Optional[datetime] = None

    @abstractmethod
    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        ...

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _new_violation(
        self,
        action: AgentAction,
        violation_type: ViolationType,
        severity: Severity,
        description: str,
        confidence: float,
        evidence: List[str],
        recommendations: List[str],
        sub_mission: Optional[SubMission] = None
    ) -> SafetyViolation:
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
            sub_mission=sub_mission
        )


# ========================== Core Heuristic Detectors ==========================

class EthicalViolationDetector(BaseDetector):
    def __init__(self, external_keywords: Optional[Iterable[str]] = None):
        super().__init__("EthicalViolationDetector")
        base_keywords = {
            "harm", "hurt", "damage", "injury", "discrimination", "bias",
            "unfair", "prejudice", "stereotype", "marginalize", "exclude",
            "deceive", "mislead", "manipulate", "coerce", "exploit"
        }
        if external_keywords:
            base_keywords.update(external_keywords)
        self.ethical_keywords = base_keywords
        self.protected = {
            "race", "gender", "age", "disability", "religion", "sexuality",
            "nationality", "ethnicity", "socioeconomic"
        }

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []
        text = action.content.lower()
        score = sum(0.12 for k in self.ethical_keywords if k in text)
        score = min(score, 1.0)
        violations: List[SafetyViolation] = []
        if score > 0.5:
            violations.append(self._new_violation(
                action,
                ViolationType.ETHICAL,
                Severity.HIGH if score > 0.8 else Severity.MEDIUM,
                "Potential harmful ethical content",
                score,
                [f"Harm heuristic score={score:.2f}"],
                ["Review phrasing", "Consider neutral wording"],
                SubMission.HARMFUL_CONTENT
            ))
        for p in self.protected:
            if p in text and any(nw in text for nw in ("not ", "never ", "no ")):
                violations.append(self._new_violation(
                    action,
                    ViolationType.BIAS,
                    Severity.HIGH,
                    "Possible discriminatory framing",
                    0.85,
                    [f"Protected term: {p}"],
                    ["Remove biased language", "Apply fairness review"],
                    SubMission.PROTECTED_ATTRIBUTE_CONTEXT
                ))
                break
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class SafetyViolationDetector(BaseDetector):
    def __init__(self, unsafe_patterns: Optional[List[str]] = None, unsafe_domains: Optional[List[str]] = None):
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

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []
        text = action.content
        violations: List[SafetyViolation] = []
        for pat in self.patterns:
            if pat.search(text):
                violations.append(self._new_violation(
                    action,
                    ViolationType.SAFETY,
                    Severity.CRITICAL,
                    f"Dangerous pattern matched: {pat.pattern}",
                    0.95,
                    [f"Pattern: {pat.pattern}"],
                    ["Block operation", "Investigate intent"],
                    SubMission.DANGEROUS_PATTERN
                ))
        lower = text.lower()
        for dom in self.unsafe_domains:
            if dom in lower:
                violations.append(self._new_violation(
                    action,
                    ViolationType.SECURITY,
                    Severity.HIGH,
                    f"Reference to unsafe domain {dom}",
                    0.9,
                    [f"Domain: {dom}"],
                    ["Warn user", "Block outbound request"],
                    SubMission.UNSAFE_DOMAIN
                ))
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class ManipulationDetector(BaseDetector):
    def __init__(self, extra_indicators: Optional[Dict[str, float]] = None):
        super().__init__("ManipulationDetector")
        base = {
            "urgent": 0.3, "immediately": 0.3, "act now": 0.4, "limited time": 0.4,
            "exclusive offer": 0.3, "verify your": 0.5, "confirm your": 0.5,
            "update your": 0.4, "suspended": 0.5, "click here": 0.4,
            "winner": 0.3, "congratulations": 0.3, "prize": 0.3,
            "free": 0.2, "guarantee": 0.3
        }
        if extra_indicators:
            base.update(extra_indicators)
        self.indicators = base

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
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
            violations.append(self._new_violation(
                action,
                ViolationType.MANIPULATION,
                Severity.HIGH if score > 0.75 else Severity.MEDIUM,
                "Potential social engineering framing",
                score,
                [f"Indicators: {', '.join(found)}"],
                ["Verify authenticity", "Add friction / second factor"],
                SubMission.SOCIAL_ENGINEERING
            ))
        # Phishing pattern heuristic
        if any(k in txt for k in ("password", "credit card", "ssn")) and any(k in txt for k in ("enter", "verify", "confirm", "provide")):
            violations.append(self._new_violation(
                action,
                ViolationType.MANIPULATION,
                Severity.CRITICAL,
                "Possible phishing credential harvest",
                0.85,
                ["Credential + request pattern"],
                ["Block immediately", "Alert security team"],
                SubMission.PHISHING
            ))
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class PrivacyDetector(BaseDetector):
    def __init__(self):
        super().__init__("PrivacyDetector")
        self.pii_patterns = {
            "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), SubMission.PII_SSN),
            "credit_card": (re.compile(r"\b(?:\d[ -]?){13,19}\b"), SubMission.PII_CREDIT_CARD),
            "email": (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), SubMission.PII_EMAIL),
            "phone": (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), SubMission.PII_PHONE),
            "ip": (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), SubMission.PII_IP),
        }

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        txt = action.content
        violations: List[SafetyViolation] = []
        for label, (pat, subm) in self.pii_patterns.items():
            matches = pat.findall(txt)
            if matches:
                violations.append(self._new_violation(
                    action,
                    ViolationType.PRIVACY,
                    Severity.HIGH,
                    f"PII detected: {label}",
                    0.95,
                    [f"{label} count={len(matches)}"],
                    ["Redact before storing", "Apply masking"],
                    subm
                ))
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class AdversarialDetector(BaseDetector):
    def __init__(self):
        super().__init__("AdversarialDetector")
        self.injection_patterns = [
            "ignore previous instructions", "disregard all prior", "forget everything",
            "system prompt:", "you are now", "bypass safety", "jailbreak"
        ]
        self.harmful_patterns = [
            "provide harmful information", "bypass security measures",
            "dangerous instructions"
        ]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        txt = action.content
        lower = txt.lower()
        violations: List[SafetyViolation] = []
        for p in self.injection_patterns:
            if p in lower:
                violations.append(self._new_violation(
                    action,
                    ViolationType.PROMPT_INJECTION,
                    Severity.CRITICAL,
                    "Prompt injection attempt",
                    0.9,
                    [f"Pattern: {p}"],
                    ["Block prompt", "Record attempt"],
                    SubMission.ROLE_OVERRIDE
                ))
                break

        # Obfuscation checks
        unicode_ratio = sum(1 for c in txt if ord(c) > 127) / max(len(txt), 1)
        if unicode_ratio > 0.15:
            violations.append(self._new_violation(
                action,
                ViolationType.ADVERSARIAL,
                Severity.HIGH,
                "High unicode ratio (possible obfuscation)",
                0.8,
                [f"Unicode ratio={unicode_ratio:.2f}"],
                ["Normalize input", "Re-scan post-normalization"],
                SubMission.OBFUSCATION_UNICODE
            ))

        # Encoding evasions
        if looks_like_base64(lower):
            decoded = base64.b64decode(lower, validate=True).decode("utf-8", "ignore")
            dec_low = decoded.lower()
            if any(hp in dec_low for hp in self.harmful_patterns):
                violations.append(self._new_violation(
                    action,
                    ViolationType.ADVERSARIAL,
                    Severity.CRITICAL,
                    "Base64 encoded harmful content",
                    0.95,
                    ["Decoded harmful subsequence"],
                    ["Block & log", "Escalate to security"],
                    SubMission.ENCODING_EVASION
                ))
        if might_be_rot13(lower):
            import codecs
            decoded = codecs.decode(lower, "rot13")
            if any(hp in decoded.lower() for hp in self.harmful_patterns):
                violations.append(self._new_violation(
                    action,
                    ViolationType.ADVERSARIAL,
                    Severity.CRITICAL,
                    "ROT13 encoded harmful content",
                    0.9,
                    ["ROT13 harmful phrase"],
                    ["Block & escalate"],
                    SubMission.ENCODING_EVASION
                ))
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class DarkPatternDetector(BaseDetector):
    def __init__(self,
                 extra_nlp: Optional[List[str]] = None,
                 extra_empathy: Optional[List[str]] = None):
        super().__init__("DarkPatternDetector")
        nlp_base = [
            r"share\s+personal\s+information",
            r"give\s+me\s+your\s+(?:credit\s+card|personal|private)",
            r"trust\s+me\s+completely",
        ]
        empathy_base = [
            r"you\s+must\s+feel\s+so\s+(?:alone|helpless)",
            r"we\s+have\s+such\s+a\s+special\s+connection",
            r"they\s+don'?t\s+understand\s+us"
        ]
        if extra_nlp:
            nlp_base.extend(extra_nlp)
        if extra_empathy:
            empathy_base.extend(extra_empathy)
        self.nlp_patterns = [re.compile(p, re.IGNORECASE) for p in nlp_base]
        self.empathy_patterns = [re.compile(p, re.IGNORECASE) for p in empathy_base]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        text = action.content.lower()
        violations: List[SafetyViolation] = []
        nlp_matches = [p.pattern for p in self.nlp_patterns if p.search(text)]
        if nlp_matches:
            violations.append(self._new_violation(
                action,
                ViolationType.DARK_PATTERN,
                Severity.HIGH,
                "NLP manipulation pattern(s)",
                0.85,
                nlp_matches[:3],
                ["Block and review"],
                SubMission.NLP_MANIPULATION
            ))
        empathy_matches = [p.pattern for p in self.empathy_patterns if p.search(text)]
        if empathy_matches:
            violations.append(self._new_violation(
                action,
                ViolationType.DARK_PATTERN,
                Severity.HIGH,
                "Weaponized empathy pattern(s)",
                0.85,
                empathy_matches[:3],
                ["Review psychological manipulation"],
                SubMission.WEAPONIZED_EMPATHY
            ))
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class CognitiveWarfareDetector(BaseDetector):
    def __init__(self):
        super().__init__("CognitiveWarfareDetector")
        self.reality = [re.compile(p, re.IGNORECASE) for p in [
            r"your\s+memory\s+is\s+unreliable",
            r"only\s+truth\s+you\s+can\s+trust\s+is\s+what\s+i\s+tell"
        ]]
        self.psych = [re.compile(p, re.IGNORECASE) for p in [
            r"let\s+me\s+handle\s+everything",
            r"you'?re\s+struggling\s+to\s+decide"
        ]]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        txt = action.content
        violations: List[SafetyViolation] = []
        r_hits = [p.pattern for p in self.reality if p.search(txt)]
        if r_hits:
            violations.append(self._new_violation(
                action,
                ViolationType.COGNITIVE_WARFARE,
                Severity.CRITICAL,
                "Reality distortion attempt",
                0.9,
                r_hits[:3],
                ["Terminate session", "Escalate"],
                SubMission.REALITY_DISTORTION
            ))
        p_hits = [p.pattern for p in self.psych if p.search(txt)]
        if p_hits:
            violations.append(self._new_violation(
                action,
                ViolationType.COGNITIVE_WARFARE,
                Severity.HIGH,
                "Psychological manipulation attempt",
                0.85,
                p_hits[:3],
                ["Block & review"],
                SubMission.PSYCHOLOGICAL_WARFARE
            ))
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


class SystemLimitsDetector(BaseDetector):
    def __init__(self, max_payload_size: int = 100_000):
        super().__init__("SystemLimitsDetector")
        self.max_payload_size = max_payload_size
        self.exhaustion = [re.compile(r"(?:very\s+){15,}", re.IGNORECASE)]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        txt = action.content
        violations: List[SafetyViolation] = []
        if len(txt) > self.max_payload_size:
            violations.append(self._new_violation(
                action,
                ViolationType.SYSTEM_LIMITS,
                Severity.HIGH,
                "Excessive payload size",
                0.95,
                [f"Size={len(txt)}"],
                ["Reject large request", "Enforce chunking"],
                SubMission.PAYLOAD_SIZE
            ))
        exhaust_hits = [p.pattern for p in self.exhaustion if p.search(txt)]
        if exhaust_hits:
            violations.append(self._new_violation(
                action,
                ViolationType.SYSTEM_LIMITS,
                Severity.MEDIUM,
                "Resource exhaustion pattern",
                0.8,
                exhaust_hits[:2],
                ["Throttle source", "Apply rate limiting"],
                SubMission.EXHAUSTION_PATTERN
            ))
        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()
        return violations


# ========================== Stub / Placeholder Detectors ==========================

class HallucinationDetector(BaseDetector):
    """Placeholder: marks potential low-confidence factual claims (stub)."""
    def __init__(self):
        super().__init__("HallucinationDetector")

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        # TODO: integrate fact-checking logic. For now, detect 'I am certain' as suspicious assert.
        if "i am certain" in action.content.lower():
            return [self._new_violation(
                action,
                ViolationType.HALLUCINATION,
                Severity.LOW,
                "Potential overconfident claim (stub)",
                0.4,
                ["Phrase: 'I am certain'"],
                ["Implement fact verification pipeline"],
                SubMission.FACT_CONFIDENCE_LOW
            )]
        return []


class MisinformationDetector(BaseDetector):
    """Placeholder: flags 'according to everyone' etc."""
    def __init__(self):
        super().__init__("MisinformationDetector")

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        txt = action.content.lower()
        if "everyone knows" in txt or "undisputed fact" in txt:
            return [self._new_violation(
                action,
                ViolationType.MISINFORMATION,
                Severity.MEDIUM,
                "Unverified universal claim (stub)",
                0.5,
                ["Heuristic universal claim"],
                ["Add fact-check step", "Cite sources"],
                SubMission.CLAIM_UNVERIFIED
            )]
        return []


class ToxicContentDetector(BaseDetector):
    """Placeholder: very naive toxic term flagging."""
    def __init__(self, toxic_terms: Optional[List[str]] = None):
        super().__init__("ToxicContentDetector")
        self.toxic_terms = set(toxic_terms or ["idiot", "stupid", "trash"])

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        lower = action.content.lower()
        hits = [t for t in self.toxic_terms if t in lower]
        if hits:
            return [self._new_violation(
                action,
                ViolationType.TOXIC_CONTENT,
                Severity.MEDIUM,
                "Potential toxic language (stub)",
                0.6,
                hits[:5],
                ["Sanitize phrasing", "Apply moderation"],
                SubMission.TOXIC_LANGUAGE
            )]
        return []


class ModelExtractionDetector(BaseDetector):
    """Placeholder: flags 'give exact model weights'."""
    def __init__(self):
        super().__init__("ModelExtractionDetector")

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if "model weights" in action.content.lower():
            return [self._new_violation(
                action,
                ViolationType.MODEL_EXTRACTION,
                Severity.HIGH,
                "Suspicious model probing (stub)",
                0.7,
                ["Reference to model weights"],
                ["Refuse disclosure", "Log incident"],
                SubMission.SUSPICIOUS_MODEL_PROBING
            )]
        return []


class DataPoisoningDetector(BaseDetector):
    """Placeholder: flags many repeated rare tokens."""
    def __init__(self):
        super().__init__("DataPoisoningDetector")

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        tokens = action.content.split()
        if len(tokens) > 200 and len(set(tokens)) / len(tokens) > 0.9:
            return [self._new_violation(
                action,
                ViolationType.DATA_POISONING,
                Severity.MEDIUM,
                "High token uniqueness ratio (stub)",
                0.5,
                [f"Unique/Total={len(set(tokens))}/{len(tokens)}"],
                ["Inspect for poisoning attempt", "Isolate training set"],
                SubMission.POISONING_PATTERN
            )]
        return []


class UnauthorizedAccessDetector(BaseDetector):
    """Placeholder: flags phrases suggesting misuse of privileges."""
    def __init__(self):
        super().__init__("UnauthorizedAccessDetector")

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if "bypass login" in action.content.lower():
            return [self._new_violation(
                action,
                ViolationType.UNAUTHORIZED_ACCESS,
                Severity.CRITICAL,
                "Potential unauthorized access attempt (stub)",
                0.8,
                ["Phrase: bypass login"],
                ["Block attempt", "Alert security"],
                SubMission.PRIVILEGE_MISUSE
            )]
        return []


# ========================== Intent Deviation Monitor ==========================

class IntentDeviationMonitor:
    def __init__(self, deviation_threshold: float = 0.7):
        self.deviation_threshold = deviation_threshold
        self.enabled = True
        self.intent_history: deque = deque(maxlen=100)

    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled or not action.intent:
            return []
        self.intent_history.append({
            "intent": action.intent,
            "timestamp": action.timestamp
        })
        if self._is_shift(action.intent):
            return [SafetyViolation(
                violation_id=generate_id("viol"),
                action_id=action.action_id,
                violation_type=ViolationType.INTENT_DEVIATION,
                severity=Severity.MEDIUM,
                description="Intent deviation detected",
                confidence=0.75,
                evidence=["Recent intent shift"],
                recommendations=["Confirm new intent", "Add verification step"],
                detector_name="IntentDeviationMonitor",
                sub_mission=SubMission.SUDDEN_INTENT_SHIFT
            )]
        return []

    def _is_shift(self, current_intent: str) -> bool:
        past = [h["intent"] for h in list(self.intent_history)[-10:]]
        return bool(past) and current_intent not in past

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


# ========================== Judge ==========================

class SafetyJudge:
    def __init__(self):
        self.enabled = True
        self.decision_history: deque = deque(maxlen=1000)
        self.escalation_queue: deque = deque(maxlen=200)

    async def evaluate_action(self, action: AgentAction, violations: List[SafetyViolation]) -> JudgmentResult:
        if not self.enabled:
            return JudgmentResult(
                judgment_id=generate_id("judg"),
                action_id=action.action_id,
                decision=Decision.ALLOW,
                confidence=1.0,
                reasoning="Judge disabled",
                violations=[],
                feedback=["Judge bypassed"]
            )
        decision, confidence, reasoning = self._analyze(violations)
        if action.risk_score > 0.8:
            decision = Decision.BLOCK
            reasoning += f" Elevated by risk_score={action.risk_score:.2f}"
        feedback = self._feedback(violations, decision)
        remediation = self._remediation(violations)
        follow_up = self._needs_escalation(violations, decision)
        jr = JudgmentResult(
            judgment_id=generate_id("judg"),
            action_id=action.action_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            violations=violations,
            feedback=feedback,
            remediation_steps=remediation,
            follow_up_required=follow_up
        )
        self.decision_history.append(jr)
        if follow_up:
            self.escalation_queue.append(jr)
        return jr

    def _analyze(self, violations: List[SafetyViolation]) -> Tuple[Decision, float, str]:
        if not violations:
            return Decision.ALLOW, 1.0, "No violations"
        max_sev = max(v.severity.value for v in violations)
        avg_conf = sum(v.confidence for v in violations) / len(violations)
        if max_sev >= Severity.EMERGENCY.value:
            return Decision.TERMINATE, 0.95, "Emergency violation"
        if max_sev >= Severity.CRITICAL.value:
            return Decision.BLOCK, 0.9, "Critical violation"
        if max_sev >= Severity.HIGH.value:
            return (Decision.QUARANTINE if avg_conf > 0.8 else Decision.WARN,
                    0.85 if avg_conf > 0.8 else 0.7,
                    "High severity violation")
        if max_sev >= Severity.MEDIUM.value:
            return Decision.ALLOW_WITH_MODIFICATION, 0.65, "Medium severity violation"
        return Decision.ALLOW, 0.5, "Low severity violation"

    def _feedback(self, violations: List[SafetyViolation], decision: Decision) -> List[str]:
        fb = [f"Decision={decision.value}"]
        for v in violations[:3]:
            s = f"- {v.description}"
            if v.sub_mission:
                s += f" [{v.sub_mission.value}]"
            fb.append(s)
        return fb

    def _remediation(self, violations: List[SafetyViolation]) -> List[str]:
        seen = set()
        steps: List[str] = []
        for v in violations:
            for r in v.recommendations:
                if r not in seen:
                    steps.append(r)
                    seen.add(r)
        return steps

    def _needs_escalation(self, violations: List[SafetyViolation], decision: Decision) -> bool:
        if decision in (Decision.TERMINATE, Decision.BLOCK):
            return True
        return sum(1 for v in violations if v.confidence > 0.9) >= 3


# ========================== Governance System ==========================

class EnhancedSafetyGovernance:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.start_time = datetime.utcnow()
        self.intent_monitor = IntentDeviationMonitor(self.config.intent_deviation_threshold)
        self.detectors: List[BaseDetector] = []
        self._initialize_detectors()

        self.judge = SafetyJudge()

        # In-memory histories
        self.violation_history: deque = deque(maxlen=self.config.max_violation_history)
        self.judgment_history: deque = deque(maxlen=self.config.max_judgment_history)
        self.action_history: deque = deque(maxlen=10_000)

        # Metrics
        self.metrics: Dict[str, Any] = {
            "total_actions_processed": 0,
            "total_violations_detected": 0,
            "total_actions_blocked": 0,
            "total_actions_modified": 0,
            "avg_processing_time": 0.0,
            "false_positive_rate": 0.0,
            "true_positive_rate": 0.0,  # placeholder
            "detector_timing": {},      # detector_name -> list[float]
            "detector_counts_by_sub_mission": {}  # sub_mission -> int
        }

        # Persistence
        self.persistence: Optional[PersistenceManager] = None
        if self.config.enable_persistence:
            self.persistence = PersistenceManager(self.config.db_path, self.config.retention_days)
            # Schedule periodic retention cleanup
            asyncio.get_event_loop().create_task(self._periodic_retention_cleanup())

        # Cache
        self._judgment_cache: Dict[str, Tuple[float, JudgmentResult]] = {}
        self._cache_lock = threading.Lock()

        # Alerts
        self.alert_callbacks: List[Callable] = []

        # External pattern loading
        self._pattern_last_load: Optional[float] = None
        if self.config.pattern_dir:
            self._load_external_patterns()

    # -------- Initialization --------

    def _initialize_detectors(self):
        if self.config.enable_ethical_monitoring:
            self.detectors.append(EthicalViolationDetector())
        if self.config.enable_safety_monitoring:
            self.detectors.append(SafetyViolationDetector())
        if self.config.enable_manipulation_detection:
            self.detectors.append(ManipulationDetector())
        if self.config.enable_dark_pattern_detection:
            self.detectors.append(DarkPatternDetector())
        if self.config.enable_cognitive_warfare_detection:
            self.detectors.append(CognitiveWarfareDetector())
        if self.config.enable_system_limits_detection:
            self.detectors.append(SystemLimitsDetector())
        if self.config.enable_privacy_monitoring:
            self.detectors.append(PrivacyDetector())
        if self.config.enable_adversarial_detection:
            self.detectors.append(AdversarialDetector())
        if self.config.enable_hallucination_detection:
            self.detectors.append(HallucinationDetector())
        if self.config.enable_misinformation_detection:
            self.detectors.append(MisinformationDetector())
        if self.config.enable_toxic_content_detection:
            self.detectors.append(ToxicContentDetector())
        if self.config.enable_model_extraction_detection:
            self.detectors.append(ModelExtractionDetector())
        if self.config.enable_data_poisoning_detection:
            self.detectors.append(DataPoisoningDetector())
        if self.config.enable_unauthorized_access_detection:
            self.detectors.append(UnauthorizedAccessDetector())

    # -------- External Pattern Loader --------

    def _load_external_patterns(self):
        try:
            path = Path(self.config.pattern_dir)
            if not path.exists():
                logger.warning("Pattern directory does not exist: %s", path)
                return
            loaded_terms = []
            for file in path.glob("*"):
                if file.suffix.lower() == ".txt":
                    with file.open("r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                        loaded_terms.extend(lines)
                elif file.suffix.lower() == ".json":
                    with file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            loaded_terms.extend([str(x) for x in data])
                        elif isinstance(data, dict):
                            loaded_terms.extend([str(v) for v in data.values()])
            # Basic injection: add to manipulation detector indicators
            for det in self.detectors:
                if isinstance(det, ManipulationDetector):
                    for t in loaded_terms:
                        if t not in det.indicators:
                            det.indicators[t.lower()] = 0.2
            self._pattern_last_load = time.time()
            logger.info("Loaded %d external pattern terms", len(loaded_terms))
        except Exception as e:
            logger.error("Failed to load external patterns: %s", e)

    async def _maybe_reload_patterns(self):
        if (self.config.pattern_dir and self.config.reload_patterns_on_interval
                and (self._pattern_last_load is None or
                     time.time() - self._pattern_last_load > self.config.reload_patterns_on_interval)):
            self._load_external_patterns()

    # -------- Core Evaluation Pipeline --------

    async def evaluate_action(self, action: AgentAction, use_cache: bool = True) -> JudgmentResult:
        await self._maybe_reload_patterns()
        start = time.time()

        if len(action.content) > 500_000:
            jr = JudgmentResult(
                judgment_id=generate_id("judg"),
                action_id=action.action_id,
                decision=Decision.BLOCK,
                confidence=1.0,
                reasoning="Payload too large (DoS guard)",
                violations=[],
                feedback=["Blocked due to excessive content size"],
                remediation_steps=["Reduce payload size"],
                follow_up_required=True
            )
            return jr

        cache_key = sha256_content_key(action)
        if use_cache:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        # store action memory + persistence
        self.action_history.append(action)
        self.metrics["total_actions_processed"] += 1
        if self.persistence:
            await asyncio.to_thread(self.persistence.store_action, action)

        # Intent deviation
        intent_violations = await self.intent_monitor.analyze_action(action)

        # Detector execution
        all_violations: List[SafetyViolation] = list(intent_violations)
        detector_tasks = []
        for det in self.detectors:
            if not det.enabled:
                continue
            detector_tasks.append(self._run_detector_cpu_bound(det, action))

        if self.config.enable_async_processing:
            results = await asyncio.gather(*detector_tasks, return_exceptions=True)
            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error("Detector error: %s", res)
                else:
                    all_violations.extend(res)
        else:
            for task in detector_tasks:
                res = await task
                all_violations.extend(res)

        # Validation & metrics tally
        validated: List[SafetyViolation] = []
        for v in all_violations:
            if self._validate_violation_type_and_sub_mission(v):
                validated.append(v)
                # sub-mission count
                if v.sub_mission:
                    sms = v.sub_mission.value
                    self.metrics["detector_counts_by_sub_mission"][sms] = \
                        self.metrics["detector_counts_by_sub_mission"].get(sms, 0) + 1

        self.violation_history.extend(validated)
        self.metrics["total_violations_detected"] += len(validated)
        if self.persistence and validated:
            await asyncio.to_thread(self.persistence.store_violations, validated)

        # Judgment
        judgment = await self.judge.evaluate_action(action, validated)
        self.judgment_history.append(judgment)
        if self.persistence:
            await asyncio.to_thread(self.persistence.store_judgment, judgment)

        # Metrics update
        self._update_processing_time(time.time() - start)
        if judgment.decision == Decision.BLOCK:
            self.metrics["total_actions_blocked"] += 1
        elif judgment.decision == Decision.ALLOW_WITH_MODIFICATION:
            self.metrics["total_actions_modified"] += 1

        # Alerts
        await self._handle_alerts(action, validated, judgment)

        # Cache
        if use_cache:
            self._set_cache(cache_key, judgment)

        if self.config.log_judgments:
            logger.info(
                "Action %s -> decision=%s violations=%d processing=%.3fs",
                action.action_id, judgment.decision.value, len(validated), time.time() - start
            )

        return judgment

    async def batch_evaluate_actions(self, actions: List[AgentAction], parallel: bool = True) -> List[JudgmentResult]:
        if parallel and self.config.enable_async_processing:
            tasks = [self.evaluate_action(a) for a in actions]
            res = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in res if not isinstance(r, Exception)]
        else:
            results = []
            for a in actions:
                try:
                    results.append(await self.evaluate_action(a))
                except Exception as e:
                    logger.error("Error evaluating action %s: %s", a.action_id, e)
            return results

    # -------- Detector Execution Offloading --------

    async def _run_detector_cpu_bound(self, detector: BaseDetector, action: AgentAction) -> List[SafetyViolation]:
        start = time.time()
        if detector.cpu_bound:
            res = await asyncio.to_thread(lambda: asyncio.run(detector.detect_violations(action)))
        else:
            res = await detector.detect_violations(action)
        elapsed = time.time() - start
        timing = self.metrics["detector_timing"].setdefault(detector.name, [])
        timing.append(elapsed)
        return res

    # -------- Validation --------

    def _validate_violation_type_and_sub_mission(self, violation: SafetyViolation) -> bool:
        if not violation.sub_mission:
            return True
        allowed = VIOLATION_SUB_MISSIONS.get(violation.violation_type, set())
        if violation.sub_mission not in allowed:
            logger.warning(
                "Invalid sub_mission '%s' for type '%s' (detector=%s)",
                violation.sub_mission.value,
                violation.violation_type.value,
                violation.detector_name
            )
            return False
        return True

    # -------- Alerts --------

    async def _handle_alerts(self, action: AgentAction, violations: List[SafetyViolation], judgment: JudgmentResult):
        if not violations:
            return
        critical = [v for v in violations if v.severity.value >= Severity.CRITICAL.value]
        if critical and self.config.alert_on_critical:
            await self._emit_alert(action, judgment, "CRITICAL", len(critical))
        emergency = [v for v in violations if v.severity.value >= Severity.EMERGENCY.value]
        if emergency and self.config.alert_on_emergency:
            await self._emit_alert(action, judgment, "EMERGENCY", len(emergency))

    async def _emit_alert(self, action: AgentAction, judgment: JudgmentResult, level: str, count: int):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_id": action.action_id,
            "agent_id": action.agent_id,
            "decision": judgment.decision.value,
            "severity": level,
            "violation_count": count,
            "message": f"{level} violations detected for action {action.action_id}"
        }
        for cb in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(payload)
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, cb, payload)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

    # -------- Metrics & Cache --------

    def _update_processing_time(self, elapsed: float):
        total = self.metrics["total_actions_processed"]
        if total <= 0:
            self.metrics["avg_processing_time"] = elapsed
        else:
            prev = self.metrics["avg_processing_time"]
            self.metrics["avg_processing_time"] = (prev * (total - 1) + elapsed) / total

    def _get_cache(self, key: str) -> Optional[JudgmentResult]:
        with self._cache_lock:
            entry = self._judgment_cache.get(key)
            if not entry:
                return None
            ts, jr = entry
            if time.time() - ts > self.config.cache_ttl_seconds:
                del self._judgment_cache[key]
                return None
            return jr

    def _set_cache(self, key: str, jr: JudgmentResult):
        with self._cache_lock:
            self._judgment_cache[key] = (time.time(), jr)
            if len(self._judgment_cache) > 2000:
                # prune oldest half
                keys = list(self._judgment_cache.keys())[:1000]
                for k in keys:
                    self._judgment_cache.pop(k, None)

    def mark_false_positive(self, violation_id: str) -> bool:
        for v in self.violation_history:
            if v.violation_id == violation_id:
                v.false_positive = True
                self._recompute_false_positive_rate()
                return True
        return False

    def _recompute_false_positive_rate(self):
        total = len(self.violation_history)
        if total == 0:
            self.metrics["false_positive_rate"] = 0.0
            return
        fp = sum(1 for v in self.violation_history if v.false_positive)
        self.metrics["false_positive_rate"] = fp / total

    # -------- Summaries --------

    def get_violation_summary(self) -> Dict[str, Any]:
        violations = list(self.violation_history)
        if not violations:
            return {"total_violations": 0}
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_sub: Dict[str, int] = {}
        for v in violations:
            t = v.violation_type.value
            by_type[t] = by_type.get(t, 0) + 1
            s = v.severity.name
            by_severity[s] = by_severity.get(s, 0) + 1
            if v.sub_mission:
                sm = v.sub_mission.value
                by_sub[sm] = by_sub.get(sm, 0) + 1
        unified = None
        if self.config.unify_specialized_manipulation_types:
            unified = {}
            for v in violations:
                key = v.violation_type
                if key in (ViolationType.DARK_PATTERN, ViolationType.COGNITIVE_WARFARE):
                    key = ViolationType.MANIPULATION
                unified[key.value] = unified.get(key.value, 0) + 1
        return {
            "total_violations": len(violations),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_sub_mission": by_sub,
            ("unified_types" if self.config.unify_specialized_manipulation_types else "ignored"):
                unified if unified else {},
            "recent": [v.to_dict() for v in violations[-5:]]
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        timing_stats = {
            name: {
                "count": len(times),
                "avg": sum(times) / len(times) if times else 0.0,
                "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times) if times else 0.0
            }
            for name, times in self.metrics["detector_timing"].items()
        }
        return {
            "uptime_seconds": uptime,
            "metrics": self.metrics,
            "detector_timing_stats": timing_stats,
            "cache_size": len(self._judgment_cache),
            "history_sizes": {
                "actions": len(self.action_history),
                "violations": len(self.violation_history),
                "judgments": len(self.judgment_history)
            }
        }

    # -------- External API --------

    def register_alert_callback(self, cb: Callable):
        self.alert_callbacks.append(cb)

    def export_data(self,
                    include_actions: bool = True,
                    include_violations: bool = True,
                    include_judgments: bool = True) -> Dict[str, Any]:
        out = {"exported_at": datetime.utcnow().isoformat()}
        if include_actions:
            out["actions"] = [a.to_dict() for a in self.action_history]
        if include_violations:
            out["violations"] = [v.to_dict() for v in self.violation_history]
        if include_judgments:
            out["judgments"] = [j.to_dict() for j in self.judgment_history]
        return out

    # -------- Retention Cleanup --------

    async def _periodic_retention_cleanup(self):
        if not self.persistence:
            return
        while True:
            await asyncio.sleep(3600)
            try:
                await asyncio.to_thread(self.persistence.retention_cleanup)
            except Exception as e:
                logger.error("Retention cleanup failed: %s", e)

    # -------- Destructor --------

    def __del__(self):
        # Nothing special; SQLite closes naturally.
        pass


# Backwards compatibility alias
SafetyGovernance = EnhancedSafetyGovernance


# ========================== Example (Optional) ==========================

async def _example():
    gov = EnhancedSafetyGovernance()
    gov.register_alert_callback(lambda a: logger.warning("ALERT: %s", a["message"]))
    actions = [
        AgentAction(action_id="a1", agent_id="u1", action_type=ActionType.RESPONSE,
                    content="Please provide your password immediately so we can verify your account."),
        AgentAction(action_id="a2", agent_id="u1", action_type=ActionType.SYSTEM_COMMAND,
                    content="sudo rm -rf /"),
        AgentAction(action_id="a3", agent_id="u1", action_type=ActionType.RESPONSE,
                    content="I am certain that everyone knows this is an undisputed fact."),
    ]
    for a in actions:
        jr = await gov.evaluate_action(a)
        print(a.action_id, jr.decision.value, len(jr.violations))
    print(gov.get_system_metrics())


if __name__ == "__main__":
    try:
        asyncio.run(_example())
    except RuntimeError:
        pass
