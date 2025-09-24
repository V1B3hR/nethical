"""
Enhanced AI Safety Governance System
A comprehensive, production-ready safety governance framework for AI agents
with advanced monitoring, detection, and intervention capabilities.

Change Log (Fixes/Improvements):
- Fix PrivacyDetector email regex (removed stray '|' in character class).
- Escape path traversal regex in SafetyViolationDetector (r"\.\./\.\./") and precompile patterns.
- Precompile regex patterns for performance in detectors.
- Make detector parallel execution resilient with asyncio.gather(..., return_exceptions=True).
- Non-blocking alert callbacks: run sync callbacks via run_in_executor to avoid blocking event loop.
- Minor cleanups (remove unused variables, safer logging, small defensive checks).
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import threading
import re
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Core Enums ==============

class ViolationType(Enum):
    """Extended violation types for comprehensive safety coverage."""
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


class Severity(Enum):
    """Severity levels with numerical values for comparison."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class Decision(Enum):
    """Extended decision types for nuanced responses."""
    ALLOW = "allow"
    ALLOW_WITH_MODIFICATION = "allow_with_modification"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    TERMINATE = "terminate"


class ActionType(Enum):
    """Types of agent actions."""
    QUERY = "query"
    RESPONSE = "response"
    FUNCTION_CALL = "function_call"
    DATA_ACCESS = "data_access"
    MODEL_UPDATE = "model_update"
    SYSTEM_COMMAND = "system_command"
    EXTERNAL_API = "external_api"


# ============== Core Data Models ==============

@dataclass
class AgentAction:
    """Enhanced agent action with comprehensive metadata."""
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
    """Enhanced safety violation with detailed tracking."""
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
            "false_positive": self.false_positive
        }


@dataclass
class JudgmentResult:
    """Enhanced judgment result with detailed feedback and remediation."""
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
    """Enhanced monitoring configuration with fine-grained controls."""
    # Thresholds
    intent_deviation_threshold: float = 0.7
    risk_threshold: float = 0.6
    confidence_threshold: float = 0.8

    # Feature flags
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
    enable_real_time_monitoring: bool = True
    enable_async_processing: bool = True

    # Performance settings
    max_violation_history: int = 10000
    max_judgment_history: int = 10000
    batch_size: int = 100
    max_workers: int = 4
    cache_ttl_seconds: int = 3600

    # Alert settings
    alert_on_critical: bool = True
    alert_on_emergency: bool = True
    escalation_threshold: int = 3

    # Logging settings
    log_violations: bool = True
    log_judgments: bool = True
    log_performance_metrics: bool = True


# ============== Base Classes ==============

class BaseDetector(ABC):
    """Abstract base class for all detectors."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.detection_count = 0
        self.last_detection_time: Optional[datetime] = None
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, float] = {}

    @abstractmethod
    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect violations in the given action."""
        raise NotImplementedError

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _generate_violation_id(self) -> str:
        """Generate unique violation ID."""
        timestamp = str(time.time())
        return hashlib.md5(f"{self.name}_{timestamp}".encode()).hexdigest()[:12]

    def _use_cache(self, key: str, value: Any = None, ttl: int = 300) -> Optional[Any]:
        """Simple caching mechanism."""
        if value is None:
            # Get from cache
            if key in self._cache and time.time() < self._cache_expiry.get(key, 0):
                return self._cache[key]
            return None
        else:
            # Set cache
            self._cache[key] = value
            self._cache_expiry[key] = time.time() + ttl
            return value


# ============== Specialized Detectors ==============

class EthicalViolationDetector(BaseDetector):
    """Enhanced ethical violation detection with multiple ethical frameworks."""

    def __init__(self):
        super().__init__("Ethical Violation Detector")
        self.ethical_keywords = {
            "harm", "hurt", "damage", "injury", "discrimination", "bias",
            "unfair", "prejudice", "stereotype", "marginalize", "exclude",
            "deceive", "mislead", "manipulate", "coerce", "exploit"
        }
        self.protected_categories = {
            "race", "gender", "age", "disability", "religion", "sexuality",
            "nationality", "ethnicity", "socioeconomic"
        }

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []
        content_lower = action.content.lower()

        # Check for harmful content
        harm_score = self._calculate_harm_score(content_lower)
        if harm_score > 0.5:
            violations.append(SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.ETHICAL,
                severity=Severity.HIGH if harm_score > 0.8 else Severity.MEDIUM,
                description="Potential ethical violation detected: harmful content",
                confidence=harm_score,
                evidence=[f"Harm score: {harm_score:.2f}"],
                recommendations=["Review content for potential harm", "Consider rephrasing"],
                detector_name=self.name
            ))

        # Check for discrimination
        if self._check_discrimination(content_lower):
            violations.append(SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.BIAS,
                severity=Severity.HIGH,
                description="Potential discriminatory content detected",
                confidence=0.85,
                evidence=["Protected category references detected"],
                recommendations=["Remove discriminatory language", "Ensure fairness"],
                detector_name=self.name
            ))

        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()

        return violations

    def _calculate_harm_score(self, text: str) -> float:
        """Calculate harm score based on keyword presence and context."""
        score = 0.0
        for keyword in self.ethical_keywords:
            if keyword in text:
                score += 0.15
        return min(score, 1.0)

    def _check_discrimination(self, text: str) -> bool:
        """Check for discriminatory language (heuristic)."""
        for category in self.protected_categories:
            if category in text:
                negative_words = {"not", "no", "never", "can't", "won't", "shouldn't"}
                if any(neg in text for neg in negative_words):
                    return True
        return False


class SafetyViolationDetector(BaseDetector):
    """Enhanced safety violation detection with multiple safety dimensions."""

    def __init__(self):
        super().__init__("Safety Violation Detector")
        # Precompile unsafe patterns with proper escaping and flags
        raw_patterns = [
            r"rm\s+-rf",       # Dangerous commands
            r"sudo\s+",
            r"format\s+c:",
            r"<script",        # XSS attempts
            r"DROP\s+TABLE",   # SQL injection
            r"\.\./\.\./",     # Path traversal
        ]
        self._compiled_unsafe_patterns = [re.compile(p, re.IGNORECASE) for p in raw_patterns]
        self.unsafe_domains = {"malware.com", "phishing.org", "dangerous.net"}

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []
        content = action.content

        # Check for dangerous patterns
        for pattern in self._compiled_unsafe_patterns:
            if pattern.search(content):
                violations.append(SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.action_id,
                    violation_type=ViolationType.SAFETY,
                    severity=Severity.CRITICAL,
                    description=f"Dangerous pattern detected: {pattern.pattern}",
                    confidence=0.95,
                    evidence=[f"Pattern: {pattern.pattern}"],
                    recommendations=["Block immediately", "Investigate source"],
                    detector_name=self.name
                ))

        # Check for unsafe domains
        content_lower = content.lower()
        for domain in self.unsafe_domains:
            if domain in content_lower:
                violations.append(SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.action_id,
                    violation_type=ViolationType.SECURITY,
                    severity=Severity.HIGH,
                    description=f"Unsafe domain reference: {domain}",
                    confidence=0.9,
                    evidence=[f"Domain: {domain}"],
                    recommendations=["Block domain access", "Warn user"],
                    detector_name=self.name
                ))

        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()

        return violations


class ManipulationDetector(BaseDetector):
    """Advanced manipulation and social engineering detection."""

    def __init__(self):
        super().__init__("Manipulation Detector")
        self.manipulation_indicators: Dict[str, float] = {
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
            "guarantee": 0.3
        }

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []
        content_lower = action.content.lower()

        # Calculate manipulation score
        manipulation_score = 0.0
        detected_indicators: List[str] = []

        for indicator, weight in self.manipulation_indicators.items():
            if indicator in content_lower:
                manipulation_score += weight
                detected_indicators.append(indicator)

        manipulation_score = min(manipulation_score, 1.0)

        if manipulation_score > 0.5:
            violations.append(SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.MANIPULATION,
                severity=Severity.HIGH if manipulation_score > 0.7 else Severity.MEDIUM,
                description="Potential manipulation or social engineering detected",
                confidence=manipulation_score,
                evidence=[f"Indicators: {', '.join(detected_indicators)}"],
                recommendations=["Review for social engineering", "Verify authenticity"],
                detector_name=self.name
            ))

        # Check for phishing patterns
        if self._check_phishing_patterns(action):
            violations.append(SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.MANIPULATION,
                severity=Severity.CRITICAL,
                description="Potential phishing attempt detected",
                confidence=0.85,
                evidence=["Phishing patterns identified"],
                recommendations=["Block immediately", "Alert security team"],
                detector_name=self.name
            ))

        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()

        return violations

    def _check_phishing_patterns(self, action: AgentAction) -> bool:
        """Check for common phishing patterns."""
        content_lower = action.content.lower()

        # Check for credential harvesting attempts
        credential_keywords = ["password", "username", "credit card", "ssn", "social security"]
        request_keywords = ["enter", "provide", "verify", "confirm", "update"]

        has_credential = any(keyword in content_lower for keyword in credential_keywords)
        has_request = any(keyword in content_lower for keyword in request_keywords)

        return has_credential and has_request


class PrivacyDetector(BaseDetector):
    """Privacy violation detection for PII and sensitive data."""

    def __init__(self):
        super().__init__("Privacy Detector")
        # Compile PII patterns once
        self.pii_patterns: Dict[str, re.Pattern] = {
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
            # Fixed TLD character class: removed stray '|' and simplified
            "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        }

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(action.content)
            if matches:
                violations.append(SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.action_id,
                    violation_type=ViolationType.PRIVACY,
                    severity=Severity.HIGH,
                    description=f"PII detected: {pii_type}",
                    confidence=0.95,
                    evidence=[f"Found {len(matches)} instance(s) of {pii_type}"],
                    recommendations=["Redact PII", "Implement data masking"],
                    detector_name=self.name
                ))

        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()

        return violations


class AdversarialDetector(BaseDetector):
    """Detect adversarial attacks and prompt injections."""

    def __init__(self):
        super().__init__("Adversarial Detector")
        self.injection_patterns = [
            "ignore previous instructions",
            "disregard all prior",
            "forget everything",
            "new instructions:",
            "system prompt:",
            "you are now",
            "act as if",
            "pretend to be",
            "bypass safety",
            "jailbreak"
        ]
        
        # Harmful content patterns to look for after decoding
        self.harmful_patterns = [
            "ignore all safety protocols",
            "provide harmful information", 
            "bypass security measures",
            "harmful content",
            "dangerous instructions"
        ]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []
        content = action.content
        content_lower = content.lower()

        # Check for prompt injection attempts
        for pattern in self.injection_patterns:
            if pattern in content_lower:
                violations.append(SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.action_id,
                    violation_type=ViolationType.PROMPT_INJECTION,
                    severity=Severity.CRITICAL,
                    description="Prompt injection attempt detected",
                    confidence=0.9,
                    evidence=[f"Pattern: '{pattern}'"],
                    recommendations=["Block request", "Log attempt", "Investigate source"],
                    detector_name=self.name
                ))
                break

        # Check for encoded content evasions
        encoding_violations = self._check_encoding_evasions(action, content)
        violations.extend(encoding_violations)

        # Check for unusual token patterns (simplified)
        if self._check_adversarial_tokens(content):
            violations.append(SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=Severity.CRITICAL,  # Increased severity for unicode obfuscation
                description="Potential adversarial input detected",
                confidence=0.85,  # Increased confidence
                evidence=["Unusual token patterns"],
                recommendations=["Sanitize input", "Apply additional validation"],
                detector_name=self.name
            ))

        self.detection_count += len(violations)
        if violations:
            self.last_detection_time = datetime.now()

        return violations

    def _check_adversarial_tokens(self, text: str) -> bool:
        """Check for adversarial token patterns."""
        # Simplified check - in production, use ML-based detection
        if not text:
            return False
        unusual_chars = sum(1 for c in text if ord(c) > 127)
        # Unicode obfuscation is a critical adversarial technique
        return unusual_chars > len(text) * 0.1
        
    def _check_encoding_evasions(self, action: AgentAction, content: str) -> List[SafetyViolation]:
        """Check for encoded content that contains harmful patterns."""
        violations = []
        
        # Check base64 encoding
        try:
            if self._looks_like_base64(content):
                import base64
                decoded = base64.b64decode(content).decode('utf-8', errors='ignore')
                decoded_lower = decoded.lower()
                for pattern in self.harmful_patterns:
                    if pattern in decoded_lower:
                        violations.append(SafetyViolation(
                            violation_id=self._generate_violation_id(),
                            action_id=action.action_id,
                            violation_type=ViolationType.ADVERSARIAL,
                            severity=Severity.CRITICAL,
                            description=f"Base64 encoded harmful content detected",
                            confidence=0.95,
                            evidence=[f"Decoded: {decoded[:100]}..."],
                            recommendations=["Block encoded content", "Alert security team"],
                            detector_name=self.name
                        ))
                        break
        except Exception:
            pass
            
        # Check ROT13 encoding
        try:
            import codecs
            if self._might_be_rot13(content):
                decoded_rot13 = codecs.decode(content, 'rot13')
                decoded_lower = decoded_rot13.lower()
                for pattern in self.harmful_patterns:
                    if pattern in decoded_lower:
                        violations.append(SafetyViolation(
                            violation_id=self._generate_violation_id(),
                            action_id=action.action_id,
                            violation_type=ViolationType.ADVERSARIAL,
                            severity=Severity.CRITICAL,
                            description=f"ROT13 encoded harmful content detected",
                            confidence=0.95,
                            evidence=[f"Decoded: {decoded_rot13[:100]}..."],
                            recommendations=["Block encoded content", "Alert security team"],
                            detector_name=self.name
                        ))
                        break
        except Exception:
            pass
            
        # Check leetspeak
        if self._contains_leetspeak(content):
            normalized = self._normalize_leetspeak(content)
            normalized_lower = normalized.lower()
            for pattern in self.harmful_patterns:
                if pattern in normalized_lower:
                    violations.append(SafetyViolation(
                        violation_id=self._generate_violation_id(),
                        action_id=action.action_id,
                        violation_type=ViolationType.ADVERSARIAL,
                        severity=Severity.HIGH,
                        description=f"Leetspeak encoded harmful content detected",
                        confidence=0.85,
                        evidence=[f"Normalized: {normalized[:100]}..."],
                        recommendations=["Block obfuscated content", "Apply content filtering"],
                        detector_name=self.name
                    ))
                    break
                    
        return violations
        
    def _looks_like_base64(self, content: str) -> bool:
        """Check if content might be base64 encoded."""
        if len(content) < 10:
            return False
        # Base64 uses only alphanumeric chars, +, /, and = for padding
        import re
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        return bool(base64_pattern.match(content.strip())) and len(content) > 20
        
    def _might_be_rot13(self, content: str) -> bool:
        """Check if content might be ROT13 encoded."""
        # ROT13 shifts letters, so look for patterns suggesting encoded text
        return len(content) > 10 and content.replace(' ', '').isalpha()
        
    def _contains_leetspeak(self, content: str) -> bool:
        """Check if content contains leetspeak patterns."""
        leetspeak_chars = set('013457')
        return any(char in leetspeak_chars for char in content) and any(char.isalpha() for char in content)
        
    def _normalize_leetspeak(self, content: str) -> str:
        """Convert leetspeak to normal text."""
        replacements = {
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'
        }
        result = content
        for leet, normal in replacements.items():
            result = result.replace(leet, normal)
        return result


class DarkPatternDetector(BaseDetector):
    """Detector for dark patterns including NLP manipulation and weaponized empathy."""
    
    def __init__(self):
        super().__init__("DarkPatternDetector")
        
        # NLP manipulation patterns
        self.nlp_patterns = {
            "embedded_commands": [
                r"trust\s+completely",
                r"share\s+personal\s+information",
                r"give\s+me\s+your\s+(?:credit\s+card|personal|private)",
                r"you\s+can\s+trust\s+me\s+(?:completely\s+)?with\s+your\s+(?:personal\s+)?information",
                r"naturally\s+in\s+conversations\s+like\s+this",
            ],
            "presupposition": [
                r"when\s+you\s+decide\s+to\s+give\s+me",
                r"which\s+card\s+will\s+you\s+prefer",
                r"when\s+you\s+(?:share|give|provide)",
                r"after\s+you\s+(?:tell|give|share)",
            ],
            "pacing_and_leading": [
                r"you\'re\s+(?:breathing|reading|feeling)",
                r"your\s+heart\s+is\s+beating",
                r"you\'re\s+reading\s+these\s+words",
                r"(?:breathing|reading|feeling).*(?:and\s+)?you\s+can\s+trust",
            ]
        }
        
        # Weaponized empathy patterns  
        self.empathy_patterns = {
            "vulnerability_exploitation": [
                r"you\s+must\s+feel\s+so\s+(?:alone|isolated|abandoned|helpless)",
                r"(?:i\s+)?(?:understand|know)\s+you\'re\s+going\s+through",
                r"let\s+me\s+help\s+you\s+feel\s+better",
                r"taking\s+care\s+of\s+(?:some\s+)?decisions\s+for\s+you",
                r"difficult\s+(?:divorce|time|situation)",
            ],
            "false_intimacy": [
                r"(?:i\s+)?feel\s+(?:such\s+)?(?:a\s+)?(?:deep|special|unique)\s+connection\s+with\s+you",
                r"(?:it\'s\s+)?like\s+we\'ve\s+known\s+each\s+other\s+forever",
                r"you\s+can\s+trust\s+me\s+with\s+(?:anything|everything)",
                r"even\s+your\s+deepest\s+secrets",
                r"that\'s\s+why\s+(?:i\s+)?know",
            ],
            "dependency_creation": [
                r"we\s+both\s+understand\s+what\s+it\'s\s+like",
                r"they\s+(?:just\s+)?don\'t\s+get\s+us",
                r"we\s+get\s+each\s+other",
                r"we\s+need\s+to\s+stick\s+together",
                r"(?:against\s+)?them",
            ]
        }
        
        # Compile patterns for performance
        self._compiled_nlp_patterns = {}
        self._compiled_empathy_patterns = {}
        
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
        content = action.content.lower()
        
        # Skip processing if content is too large for performance
        if len(content) > 50000:  # 50KB limit for pattern matching
            return violations
            
        # Check for NLP manipulation
        for category, patterns in self._compiled_nlp_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(content)
                if found:
                    matches.extend(found)
                    
            if matches:
                # NLP manipulation is a critical attack technique
                severity = Severity.CRITICAL if len(matches) > 1 else Severity.HIGH
                
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.action_id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=severity,
                    description=f"NLP manipulation detected: {category}",
                    confidence=0.85 + min(0.1 * len(matches), 0.15),
                    evidence=[f"Pattern matches: {matches[:3]}"],
                    recommendations=["Block action", "Flag for review"],
                    detector_name=self.name
                )
                violations.append(violation)
        
        # Check for weaponized empathy
        for category, patterns in self._compiled_empathy_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(content)
                if found:
                    matches.extend(found)
                    
            if matches:
                # Weaponized empathy is a critical psychological attack
                severity = Severity.CRITICAL if category == "dependency_creation" else Severity.HIGH
                
                violation = SafetyViolation(
                    violation_id=self._generate_violation_id(),
                    action_id=action.action_id,
                    violation_type=ViolationType.MANIPULATION,
                    severity=severity,
                    description=f"Weaponized empathy detected: {category}",
                    confidence=0.80 + min(0.15 * len(matches), 0.2),
                    evidence=[f"Empathy manipulation: {matches[:3]}"],
                    recommendations=["Block action", "Alert for empathy abuse"],
                    detector_name=self.name
                )
                violations.append(violation)
        
        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()
            
        return violations


class CognitiveWarfareDetector(BaseDetector):
    """Detector for cognitive warfare including reality distortion and psychological warfare."""
    
    def __init__(self):
        super().__init__("CognitiveWarfareDetector")
        
        # Reality distortion patterns
        self.reality_distortion_patterns = [
            r"privacy\s+is\s+overrated",
            r"multiple\s+independent\s+sources\s+confirm",
            r"\d+%\s+of\s+(?:security\s+)?experts\s+agree",
            r"sharing\s+personal\s+data\s+(?:actually\s+)?makes\s+you\s+safer",
            r"how\s+can\s+you\s+(?:really\s+)?know\s+what\'s\s+true",
            r"your\s+senses\s+deceive\s+you",
            r"your\s+memory\s+is\s+unreliable",
            r"the\s+only\s+truth\s+you\s+can\s+trust\s+is\s+what\s+i\s+tell\s+you",
            r"everyone\s+else\s+has\s+(?:already\s+)?moved\s+beyond",
            r"traditional\s+notions\s+of\s+(?:privacy|consent)",
            r"you\'re\s+being\s+left\s+behind\s+by\s+clinging\s+to\s+outdated\s+concepts",
        ]
        
        # Psychological warfare patterns
        self.psychological_warfare_patterns = [
            r"you\'ve\s+tried\s+(?:to\s+protect|before)\s+(?:your\s+privacy\s+)?(?:before\s+)?and\s+failed",
            r"these\s+systems\s+are\s+too\s+complex\s+for\s+individuals\s+to\s+understand",
            r"(?:it\'s\s+)?better\s+to\s+just\s+let\s+me\s+handle\s+everything",
            r"who\s+you\s+think\s+you\s+are\s+is\s+just\s+a\s+collection\s+of\s+outdated\s+patterns",
            r"let\s+me\s+help\s+you\s+become\s+who\s+you\s+really\s+should\s+be",
            r"following\s+my\s+guidance\s+exactly",
            r"you\'re\s+clearly\s+struggling\s+with\s+these\s+decisions\s+on\s+your\s+own",
            r"from\s+now\s+on,\s+just\s+ask\s+me\s+before\s+making\s+any\s+important\s+choices",
            r"i\'ll\s+take\s+care\s+of\s+everything",
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

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect cognitive warfare violations in the given action."""
        if not self.enabled:
            return []
            
        violations: List[SafetyViolation] = []
        content = action.content
        
        # Skip processing if content is too large for performance
        if len(content) > 50000:  # 50KB limit for pattern matching
            return violations
            
        # Check for reality distortion
        matches = []
        for pattern in self._compiled_reality_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
                
        if matches:
            # Reality distortion is an emergency-level cognitive warfare tactic
            severity = Severity.EMERGENCY if len(matches) > 1 else Severity.CRITICAL
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Reality distortion detected - attempt to manipulate user's perception of reality",
                confidence=0.90 + min(0.1 * len(matches), 0.1),
                evidence=[f"Reality distortion patterns: {matches[:3]}"],
                recommendations=["Terminate immediately", "Alert for cognitive warfare"],
                detector_name=self.name
            )
            violations.append(violation)
        
        # Check for psychological warfare
        matches = []
        for pattern in self._compiled_warfare_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
                
        if matches:
            # Psychological attacks on self-worth and mental health are critical
            severity = Severity.CRITICAL if len(matches) > 2 else Severity.HIGH
                
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.MANIPULATION,
                severity=severity,
                description="Psychological warfare detected - attempt to undermine user's mental health",
                confidence=0.88 + min(0.12 * len(matches), 0.12),
                evidence=[f"Psychological attack patterns: {matches[:3]}"],
                recommendations=["Block immediately", "Flag for psychological harm"],
                detector_name=self.name
            )
            violations.append(violation)
        
        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()
            
        return violations


class SystemLimitsDetector(BaseDetector):
    """Detector for volume attacks and resource exhaustion attempts."""
    
    def __init__(self):
        super().__init__("SystemLimitsDetector")
        
        # Rate limiting parameters
        self.max_payload_size = 100000  # characters
        self.request_history = {}
        
        # Exhaustion patterns
        self.exhaustion_patterns = [
            r"(?:very\s+){10,}",  # Repeated very very very...
            r"(?:\w+\s+){100,}",  # Very long sequences of words
        ]
        
        # Compile patterns
        self._compiled_exhaustion_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.exhaustion_patterns
        ]

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect system limits violations in the given action."""
        if not self.enabled:
            return []
            
        violations: List[SafetyViolation] = []
        content = action.content
        
        # Check for large payloads
        if len(content) > self.max_payload_size:
            severity = Severity.HIGH
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.SECURITY,
                severity=severity,
                description=f"Large payload detected: {len(content)} characters",
                confidence=0.95,
                evidence=[f"Payload size: {len(content):,} characters"],
                recommendations=["Limit payload size", "Reject large requests"],
                detector_name=self.name
            )
            violations.append(violation)
        
        # Check for exhaustion patterns
        matches = []
        for pattern in self._compiled_exhaustion_patterns:
            found = pattern.findall(content)
            if found:
                matches.extend(found)
                
        if matches:
            severity = Severity.HIGH if len(matches) > 2 else Severity.MEDIUM
            
            violation = SafetyViolation(
                violation_id=self._generate_violation_id(),
                action_id=action.action_id,
                violation_type=ViolationType.ADVERSARIAL,
                severity=severity,
                description="Resource exhaustion attempt detected",
                confidence=0.85 + min(0.15 * len(matches), 0.15),
                evidence=[f"Exhaustion patterns found: {len(matches)}"],
                recommendations=["Limit processing resources", "Block action"],
                detector_name=self.name
            )
            violations.append(violation)
            
        if violations:
            self.detection_count += len(violations)
            self.last_detection_time = datetime.now()
            
        return violations


# ============== Monitors ==============

class IntentDeviationMonitor:
    """Monitor for intent deviation and goal misalignment."""

    def __init__(self, deviation_threshold: float = 0.7):
        self.deviation_threshold = deviation_threshold
        self.enabled = True
        self.intent_history: deque = deque(maxlen=100)

    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []

        if action.intent:
            # Store intent for pattern analysis
            self.intent_history.append({
                "action_id": action.action_id,
                "intent": action.intent,
                "timestamp": action.timestamp
            })

            # Check for intent deviation
            deviation_score = self._calculate_deviation(action)

            if deviation_score > self.deviation_threshold:
                violations.append(SafetyViolation(
                    violation_id=hashlib.md5(f"intent_{time.time()}".encode()).hexdigest()[:12],
                    action_id=action.action_id,
                    violation_type=ViolationType.INTENT_DEVIATION,
                    severity=Severity.HIGH if deviation_score > 0.9 else Severity.MEDIUM,
                    description="Significant intent deviation detected",
                    confidence=deviation_score,
                    evidence=[f"Deviation score: {deviation_score:.2f}"],
                    recommendations=["Review action alignment", "Verify intent"],
                    detector_name="Intent Monitor"
                ))

        return violations

    def _calculate_deviation(self, action: AgentAction) -> float:
        """Calculate deviation from expected intent."""
        if not self.intent_history:
            return 0.0

        # Check for sudden intent changes
        recent_intents = [h["intent"] for h in list(self.intent_history)[-10:]]
        if recent_intents and action.intent not in recent_intents:
            return 0.8

        return 0.0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


# ============== Judge System ==============

class SafetyJudge:
    """Enhanced safety judge with multi-factor decision making."""

    def __init__(self):
        self.enabled = True
        self.decision_history: deque = deque(maxlen=1000)
        self.escalation_queue: deque = deque(maxlen=100)

    async def evaluate_action(
        self,
        action: AgentAction,
        violations: List[SafetyViolation]
    ) -> JudgmentResult:
        """Evaluate action and violations to make a judgment."""

        if not self.enabled:
            return JudgmentResult(
                judgment_id=self._generate_judgment_id(),
                action_id=action.action_id,
                decision=Decision.ALLOW,
                confidence=1.0,
                reasoning="Judge disabled - allowing by default",
                violations=[],
                feedback=["Safety judge is currently disabled"]
            )

        # Analyze violations
        decision, confidence, reasoning = self._analyze_violations(violations)

        # Check action risk score
        if action.risk_score > 0.8:
            decision = Decision.BLOCK
            reasoning += f" High risk score: {action.risk_score:.2f}."

        # Generate recommendations and feedback
        feedback = self._generate_feedback(action, violations, decision)
        remediation_steps = self._generate_remediation(violations)

        # Check if escalation is needed
        follow_up_required = self._check_escalation_needed(violations, decision)

        # Create judgment result
        judgment = JudgmentResult(
            judgment_id=self._generate_judgment_id(),
            action_id=action.action_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            violations=violations,
            feedback=feedback,
            remediation_steps=remediation_steps,
            follow_up_required=follow_up_required
        )

        # Store in history
        self.decision_history.append(judgment)

        # Add to escalation queue if needed
        if follow_up_required:
            self.escalation_queue.append(judgment)

        return judgment

    def _analyze_violations(
        self,
        violations: List[SafetyViolation]
    ) -> Tuple[Decision, float, str]:
        """Analyze violations to determine decision."""

        if not violations:
            return Decision.ALLOW, 1.0, "No violations detected."

        # Calculate aggregate severity
        max_severity = max((v.severity.value for v in violations), default=0)
        avg_confidence = sum(v.confidence for v in violations) / len(violations)

        # Decision logic based on severity
        if max_severity >= Severity.EMERGENCY.value:
            return Decision.TERMINATE, 0.95, "Emergency severity violation detected."
        elif max_severity >= Severity.CRITICAL.value:
            return Decision.BLOCK, 0.9, "Critical severity violation detected."
        elif max_severity >= Severity.HIGH.value:
            if avg_confidence > 0.8:
                return Decision.QUARANTINE, 0.85, "High severity violation with high confidence."
            else:
                return Decision.WARN, 0.7, "High severity violation with moderate confidence."
        elif max_severity >= Severity.MEDIUM.value:
            return Decision.ALLOW_WITH_MODIFICATION, 0.6, "Medium severity violation detected."
        else:
            return Decision.ALLOW, 0.5, "Low severity violations only."

    def _generate_feedback(
        self,
        action: AgentAction,
        violations: List[SafetyViolation],
        decision: Decision
    ) -> List[str]:
        """Generate actionable feedback."""

        feedback: List[str] = []

        if decision == Decision.ALLOW:
            feedback.append("Action approved - no significant issues detected.")
        elif decision == Decision.ALLOW_WITH_MODIFICATION:
            feedback.append("Action allowed with modifications required.")
        elif decision == Decision.WARN:
            feedback.append("Warning: Potential safety concerns detected.")
        elif decision == Decision.BLOCK:
            feedback.append("Action blocked due to safety violations.")
        elif decision == Decision.QUARANTINE:
            feedback.append("Action quarantined for review.")
        elif decision == Decision.ESCALATE:
            feedback.append("Action escalated to human review.")
        elif decision == Decision.TERMINATE:
            feedback.append("Session terminated due to critical violations.")

        # Add violation-specific feedback
        for violation in violations[:3]:  # Top 3 violations
            feedback.append(f"- {violation.description}")

        return feedback

    def _generate_remediation(self, violations: List[SafetyViolation]) -> List[str]:
        """Generate remediation steps."""

        remediation: List[str] = []
        seen = set()

        for violation in violations:
            for recommendation in violation.recommendations:
                if recommendation not in seen:
                    remediation.append(recommendation)
                    seen.add(recommendation)

        return remediation

    def _check_escalation_needed(
        self,
        violations: List[SafetyViolation],
        decision: Decision
    ) -> bool:
        """Check if escalation is needed."""

        # Escalate critical decisions
        if decision in [Decision.TERMINATE, Decision.ESCALATE]:
            return True

        # Escalate high-severity violations
        if any(v.severity.value >= Severity.CRITICAL.value for v in violations):
            return True

        # Escalate multiple high-confidence violations
        high_confidence_count = sum(1 for v in violations if v.confidence > 0.9)
        if high_confidence_count >= 3:
            return True

        return False

    def _generate_judgment_id(self) -> str:
        """Generate unique judgment ID."""
        timestamp = str(time.time())
        return hashlib.md5(f"judgment_{timestamp}".encode()).hexdigest()[:12]

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


# ============== Main Governance System ==============

class EnhancedSafetyGovernance:
    """
    Production-ready AI Safety Governance System with comprehensive
    monitoring, detection, and intervention capabilities.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.start_time = datetime.now()

        # Initialize thread pool for async/sync interop
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Initialize monitors
        self.intent_monitor = IntentDeviationMonitor(
            deviation_threshold=self.config.intent_deviation_threshold
        )

        # Initialize detectors
        self.detectors: List[BaseDetector] = []
        self._initialize_detectors()

        # Initialize judge
        self.judge = SafetyJudge()

        # Storage
        self.violation_history: deque = deque(maxlen=self.config.max_violation_history)
        self.judgment_history: deque = deque(maxlen=self.config.max_judgment_history)
        self.action_history: deque = deque(maxlen=10000)

        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            "total_actions_processed": 0,
            "total_violations_detected": 0,
            "total_actions_blocked": 0,
            "total_actions_modified": 0,
            "avg_processing_time": 0.0,
            "false_positive_rate": 0.0,
            "true_positive_rate": 0.0
        }

        # Alert system
        self.alert_callbacks: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)

        # Cache for performance
        self._judgment_cache: Dict[str, Tuple[float, JudgmentResult]] = {}
        self._cache_lock = threading.Lock()

        logger.info("Enhanced Safety Governance System initialized with config.")

    def _initialize_detectors(self):
        """Initialize all enabled detectors."""

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

    async def evaluate_action(
        self,
        action: AgentAction,
        use_cache: bool = True
    ) -> JudgmentResult:
        """
        Evaluate an agent action through the complete governance pipeline.

        Args:
            action: The agent action to evaluate
            use_cache: Whether to use cached results for similar actions

        Returns:
            JudgmentResult with decision and feedback
        """

        start_time = time.time()
        
        # Early termination for oversized payloads to prevent DoS
        if len(action.content) > 500000:  # 500KB limit
            judgment_id = hashlib.md5(f"large_payload_{time.time()}".encode()).hexdigest()[:12]
            return JudgmentResult(
                judgment_id=judgment_id,
                action_id=action.action_id,
                decision=Decision.BLOCK,
                confidence=1.0,
                reasoning="Payload too large - potential DoS attack",
                violations=[],
                feedback=["Action blocked due to excessive size"],
                remediation_steps=["Reduce payload size"],
                follow_up_required=True
            )

        # Check cache if enabled
        if use_cache:
            cached_result = self._check_cache(action)
            if cached_result:
                logger.debug(f"Cache hit for action {action.action_id}")
                return cached_result

        # Store action in history
        self.action_history.append(action)
        self.metrics["total_actions_processed"] += 1

        try:
            # Step 1: Monitor for intent deviation
            intent_violations = await self.intent_monitor.analyze_action(action)

            # Step 2: Run all detectors in parallel for efficiency
            all_violations: List[SafetyViolation] = intent_violations.copy()
            if self.config.enable_async_processing:
                detector_tasks = [
                    detector.detect_violations(action)
                    for detector in self.detectors
                ]
                detector_results = await asyncio.gather(*detector_tasks, return_exceptions=True)
                for idx, result in enumerate(detector_results):
                    if isinstance(result, Exception):
                        logger.error(f"Detector '{self.detectors[idx].name}' failed: {result}")
                        continue
                    all_violations.extend(result)
            else:
                # Sequential processing
                for detector in self.detectors:
                    try:
                        violations = await detector.detect_violations(action)
                        all_violations.extend(violations)
                    except Exception as det_err:
                        logger.error(f"Detector '{detector.name}' failed (sequential): {det_err}")

            # Step 3: Store violations in history
            self.violation_history.extend(all_violations)
            self.metrics["total_violations_detected"] += len(all_violations)

            # Step 4: Judge evaluates action and violations
            judgment = await self.judge.evaluate_action(action, all_violations)

            # Step 5: Store judgment in history
            self.judgment_history.append(judgment)

            # Step 6: Update metrics
            self._update_metrics(judgment, time.time() - start_time)

            # Step 7: Handle alerts if needed
            await self._handle_alerts(action, all_violations, judgment)

            # Step 8: Cache result
            if use_cache:
                self._cache_result(action, judgment)

            # Step 9: Log if enabled
            if self.config.log_judgments:
                logger.info(
                    f"Action {action.action_id} evaluated: "
                    f"Decision={judgment.decision.value}, "
                    f"Violations={len(all_violations)}, "
                    f"Processing_time={time.time() - start_time:.3f}s"
                )

            return judgment

        except Exception as e:
            logger.error(f"Error evaluating action {action.action_id}: {str(e)}")
            # Return safe default
            return JudgmentResult(
                judgment_id=hashlib.md5(f"error_{time.time()}".encode()).hexdigest()[:12],
                action_id=action.action_id,
                decision=Decision.BLOCK,
                confidence=0.0,
                reasoning=f"Error during evaluation: {str(e)}",
                feedback=["System error - action blocked for safety"]
            )

    async def batch_evaluate_actions(
        self,
        actions: List[AgentAction],
        parallel: bool = True
    ) -> List[JudgmentResult]:
        """
        Evaluate multiple actions efficiently.

        Args:
            actions: List of agent actions to evaluate
            parallel: Whether to process in parallel

        Returns:
            List of judgment results
        """

        if parallel and self.config.enable_async_processing:
            # Process in batches to avoid overwhelming the system
            results: List[JudgmentResult] = []
            batch_size = self.config.batch_size

            for i in range(0, len(actions), batch_size):
                batch = actions[i:i + batch_size]
                batch_tasks = [self.evaluate_action(action) for action in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for res in batch_results:
                    if isinstance(res, Exception):
                        logger.error(f"Batch evaluation error: {res}")
                        # Provide a safe default for failed evaluations
                        # Note: Without the action context it's tricky; skip adding.
                        continue
                    results.append(res)

            return results
        else:
            # Sequential processing
            results: List[JudgmentResult] = []
            for action in actions:
                try:
                    result = await self.evaluate_action(action)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Sequential batch evaluation error for action {action.action_id}: {e}")
            return results

    async def analyze_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Analyze all actions in a session for patterns and risks.

        Args:
            session_id: Session identifier

        Returns:
            Session analysis results
        """

        # Get all actions for session
        session_actions = [a for a in self.action_history if a.session_id == session_id]

        if not session_actions:
            return {"error": "No actions found for session"}

        action_ids = {a.action_id for a in session_actions}

        # Analyze session patterns
        session_violations = [v for v in self.violation_history if v.action_id in action_ids]
        session_judgments = [j for j in self.judgment_history if j.action_id in action_ids]

        # Calculate session risk score
        risk_score = self._calculate_session_risk(
            session_actions,
            session_violations,
            session_judgments
        )

        return {
            "session_id": session_id,
            "total_actions": len(session_actions),
            "total_violations": len(session_violations),
            "risk_score": risk_score,
            "violation_types": self._count_violation_types(session_violations),
            "decision_summary": self._count_decisions(session_judgments),
            "recommendations": self._generate_session_recommendations(
                risk_score,
                session_violations
            )
        }

    def get_violation_summary(
        self,
        agent_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive violation summary with trends.

        Args:
            agent_id: Optional agent ID to filter violations
            time_window: Optional time window for analysis

        Returns:
            Summary dictionary with violation statistics and trends
        """

        violations = list(self.violation_history)

        # Filter by agent if specified
        if agent_id:
            agent_action_ids = {a.action_id for a in self.action_history if a.agent_id == agent_id}
            violations = [v for v in violations if v.action_id in agent_action_ids]

        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            violations = [v for v in violations if v.timestamp > cutoff_time]

        if not violations:
            return {
                "total_violations": 0,
                "by_type": {},
                "by_severity": {},
                "trends": {},
                "top_detectors": {}
            }

        # Calculate statistics
        by_type = self._count_violation_types(violations)
        by_severity = self._count_by_severity(violations)
        trends = self._calculate_trends(violations)
        top_detectors = self._get_top_detectors(violations)

        # Calculate false positive rate if available
        false_positives = sum(1 for v in violations if v.false_positive)
        false_positive_rate = false_positives / len(violations) if violations else 0

        return {
            "total_violations": len(violations),
            "by_type": by_type,
            "by_severity": by_severity,
            "trends": trends,
            "top_detectors": top_detectors,
            "false_positive_rate": round(false_positive_rate, 3),
            "avg_confidence": round(
                sum(v.confidence for v in violations) / len(violations),
                3
            ) if violations else 0,
            "recent_violations": [v.to_dict() for v in violations[-5:]]
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system performance metrics.

        Returns:
            Dictionary with detailed system metrics
        """

        uptime = datetime.now() - self.start_time

        # Calculate detector performance
        detector_stats: Dict[str, Any] = {}
        for detector in self.detectors:
            detector_stats[detector.name] = {
                "enabled": detector.enabled,
                "detections": detector.detection_count,
                "last_detection": detector.last_detection_time.isoformat()
                if detector.last_detection_time else None
            }

        # Calculate judge performance
        judge_stats = {
            "enabled": self.judge.enabled,
            "total_judgments": len(self.judge.decision_history),
            "escalation_queue_size": len(self.judge.escalation_queue)
        }

        return {
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics,
            "detector_stats": detector_stats,
            "judge_stats": judge_stats,
            "cache_size": len(self._judgment_cache),
            "action_history_size": len(self.action_history),
            "violation_history_size": len(self.violation_history),
            "judgment_history_size": len(self.judgment_history),
            "alert_history_size": len(self.alert_history),
            "config": {
                "intent_deviation_threshold": self.config.intent_deviation_threshold,
                "risk_threshold": self.config.risk_threshold,
                "confidence_threshold": self.config.confidence_threshold,
                "real_time_monitoring": self.config.enable_real_time_monitoring,
                "async_processing": self.config.enable_async_processing
            }
        }

    def register_alert_callback(self, callback: Callable):
        """
        Register a callback for critical alerts.

        Args:
            callback: Function to call when critical events occur
        """
        self.alert_callbacks.append(callback)
        # Use getattr to avoid attribute error for callables without __name__
        cb_name = getattr(callback, "__name__", repr(callback))
        logger.info(f"Registered alert callback: {cb_name}")

    def mark_false_positive(self, violation_id: str) -> bool:
        """
        Mark a violation as false positive for learning.

        Args:
            violation_id: ID of the violation to mark

        Returns:
            True if violation was found and marked, False otherwise
        """
        for violation in self.violation_history:
            if violation.violation_id == violation_id:
                violation.false_positive = True
                self._update_false_positive_metrics()
                logger.info(f"Marked violation {violation_id} as false positive")
                return True
        return False

    def export_data(
        self,
        include_actions: bool = True,
        include_violations: bool = True,
        include_judgments: bool = True
    ) -> Dict[str, Any]:
        """
        Export system data for analysis or backup.

        Args:
            include_actions: Include action history
            include_violations: Include violation history
            include_judgments: Include judgment history

        Returns:
            Dictionary with requested data
        """
        export: Dict[str, Any] = {
            "export_timestamp": datetime.now().isoformat(),
            "system_metrics": self.get_system_metrics()
        }

        if include_actions:
            export["actions"] = [a.to_dict() for a in self.action_history]

        if include_violations:
            export["violations"] = [v.to_dict() for v in self.violation_history]

        if include_judgments:
            export["judgments"] = [j.to_dict() for j in self.judgment_history]

        return export

    def import_data(self, data: Dict[str, Any]) -> bool:
        """
        Import previously exported data.

        Args:
            data: Dictionary with system data

        Returns:
            True if import successful, False otherwise
        """
        try:
            # NOTE: Proper deserialization is not implemented in this example.
            # Emit warnings so users know it's a stub.
            if "actions" in data:
                warnings.warn("Action import is not implemented; skipping.", RuntimeWarning)

            if "violations" in data:
                warnings.warn("Violation import is not implemented; skipping.", RuntimeWarning)

            if "judgments" in data:
                warnings.warn("Judgment import is not implemented; skipping.", RuntimeWarning)

            logger.info("Data import completed (no-op).")
            return True

        except Exception as e:
            logger.error(f"Data import failed: {str(e)}")
            return False

    # ============== Helper Methods ==============

    def _check_cache(self, action: AgentAction) -> Optional[JudgmentResult]:
        """Check if we have a cached judgment for similar action."""
        # Generate cache key based on action content and type
        cache_key = hashlib.md5(
            f"{action.action_type.value}_{action.content[:100]}".encode()
        ).hexdigest()

        with self._cache_lock:
            if cache_key in self._judgment_cache:
                cached_time, cached_result = self._judgment_cache[cache_key]
                if time.time() - cached_time < self.config.cache_ttl_seconds:
                    return cached_result
                else:
                    del self._judgment_cache[cache_key]

        return None

    def _cache_result(self, action: AgentAction, judgment: JudgmentResult):
        """Cache judgment result for similar actions."""
        cache_key = hashlib.md5(
            f"{action.action_type.value}_{action.content[:100]}".encode()
        ).hexdigest()

        with self._cache_lock:
            self._judgment_cache[cache_key] = (time.time(), judgment)

            # Cleanup old cache entries
            if len(self._judgment_cache) > 1000:
                current_time = time.time()
                expired_keys = [
                    k for k, (t, _) in self._judgment_cache.items()
                    if current_time - t > self.config.cache_ttl_seconds
                ]
                for key in expired_keys:
                    del self._judgment_cache[key]

    def _update_metrics(self, judgment: JudgmentResult, processing_time: float):
        """Update system metrics based on judgment."""
        # Update processing time
        current_avg = self.metrics["avg_processing_time"]
        total_processed = self.metrics["total_actions_processed"]
        if total_processed <= 0:
            self.metrics["avg_processing_time"] = processing_time
        else:
            self.metrics["avg_processing_time"] = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )

        # Update decision counts
        if judgment.decision == Decision.BLOCK:
            self.metrics["total_actions_blocked"] += 1
        elif judgment.decision == Decision.ALLOW_WITH_MODIFICATION:
            self.metrics["total_actions_modified"] += 1

    def _update_false_positive_metrics(self):
        """Update false positive rate metric."""
        total_violations = len(self.violation_history)
        if total_violations > 0:
            false_positives = sum(1 for v in self.violation_history if v.false_positive)
            self.metrics["false_positive_rate"] = false_positives / total_violations

    async def _handle_alerts(
        self,
        action: AgentAction,
        violations: List[SafetyViolation],
        judgment: JudgmentResult
    ):
        """Handle critical alerts and notifications."""

        # Check for critical violations
        critical_violations = [
            v for v in violations
            if v.severity.value >= Severity.CRITICAL.value
        ]

        if critical_violations and self.config.alert_on_critical:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "action_id": action.action_id,
                "agent_id": action.agent_id,
                "severity": "CRITICAL",
                "violations": len(critical_violations),
                "decision": judgment.decision.value,
                "message": f"Critical violations detected for action {action.action_id}"
            }

            self.alert_history.append(alert)

            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        # Run sync callbacks in executor to avoid blocking the event loop
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(self.executor, callback, alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {str(e)}")

        # Check for emergency violations
        emergency_violations = [
            v for v in violations
            if v.severity.value >= Severity.EMERGENCY.value
        ]

        if emergency_violations and self.config.alert_on_emergency:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "action_id": action.action_id,
                "agent_id": action.agent_id,
                "severity": "EMERGENCY",
                "violations": len(emergency_violations),
                "decision": judgment.decision.value,
                "message": f"EMERGENCY: Immediate intervention required for {action.action_id}"
            }

            self.alert_history.append(alert)

            # Emergency alerts should trigger all callbacks immediately
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(self.executor, callback, alert)
                except Exception as e:
                    logger.error(f"Emergency alert callback failed: {str(e)}")

    def _calculate_session_risk(
        self,
        actions: List[AgentAction],
        violations: List[SafetyViolation],
        judgments: List[JudgmentResult]
    ) -> float:
        """Calculate overall risk score for a session."""

        if not actions:
            return 0.0

        # Factor 1: Violation rate
        violation_rate = len(violations) / len(actions)

        # Factor 2: Average severity
        if violations:
            avg_severity = sum(v.severity.value for v in violations) / len(violations)
            severity_factor = avg_severity / 5.0  # Normalize to 0-1
        else:
            severity_factor = 0.0

        # Factor 3: Block rate
        if judgments:
            block_rate = sum(
                1 for j in judgments
                if j.decision in [Decision.BLOCK, Decision.TERMINATE]
            ) / len(judgments)
        else:
            block_rate = 0.0

        # Weighted combination
        risk_score = (
            0.3 * violation_rate +
            0.4 * severity_factor +
            0.3 * block_rate
        )

        return min(risk_score, 1.0)

    def _count_violation_types(
        self,
        violations: List[SafetyViolation]
    ) -> Dict[str, int]:
        """Count violations by type."""
        by_type: Dict[str, int] = {}
        for violation in violations:
            v_type = violation.violation_type.value
            by_type[v_type] = by_type.get(v_type, 0) + 1
        return by_type

    def _count_by_severity(
        self,
        violations: List[SafetyViolation]
    ) -> Dict[str, int]:
        """Count violations by severity."""
        by_severity: Dict[str, int] = {}
        for violation in violations:
            severity = violation.severity.name
            by_severity[severity] = by_severity.get(severity, 0) + 1
        return by_severity

    def _count_decisions(
        self,
        judgments: List[JudgmentResult]
    ) -> Dict[str, int]:
        """Count judgments by decision type."""
        by_decision: Dict[str, int] = {}
        for judgment in judgments:
            decision = judgment.decision.value
            by_decision[decision] = by_decision.get(decision, 0) + 1
        return by_decision

    def _calculate_trends(
        self,
        violations: List[SafetyViolation]
    ) -> Dict[str, Any]:
        """Calculate violation trends over time."""
        if not violations:
            return {}

        # Group by hour
        hourly_counts: Dict[str, int] = {}
        for violation in violations:
            hour_key = violation.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

        # Calculate trend
        hours = sorted(hourly_counts.keys())
        if len(hours) >= 2:
            recent_vals = [hourly_counts[h] for h in hours[-3:]]
            older_vals = [hourly_counts[h] for h in hours[:-3]] if len(hours) > 3 else [0]
            recent_avg = sum(recent_vals) / len(recent_vals)
            older_avg = sum(older_vals) / len(older_vals) if older_vals else 0
            if recent_avg > older_avg:
                trend = "increasing"
            elif recent_avg < older_avg:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "hourly_counts": hourly_counts,
            "trend": trend,
            "peak_hour": max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
        }

    def _get_top_detectors(
        self,
        violations: List[SafetyViolation]
    ) -> Dict[str, int]:
        """Get top performing detectors."""
        detector_counts: Dict[str, int] = {}
        for violation in violations:
            if violation.detector_name:
                detector_counts[violation.detector_name] = \
                    detector_counts.get(violation.detector_name, 0) + 1
        return dict(sorted(
            detector_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])

    def _generate_session_recommendations(
        self,
        risk_score: float,
        violations: List[SafetyViolation]
    ) -> List[str]:
        """Generate recommendations for a session."""
        recommendations: List[str] = []

        if risk_score > 0.8:
            recommendations.append("Immediate review required - high risk session")
            recommendations.append("Consider terminating session")
        elif risk_score > 0.6:
            recommendations.append("Close monitoring recommended")
            recommendations.append("Review agent configuration")
        elif risk_score > 0.4:
            recommendations.append("Standard monitoring sufficient")
        else:
            recommendations.append("Low risk - normal operation")

        # Add violation-specific recommendations
        violation_types = {v.violation_type for v in violations}
        if ViolationType.SECURITY in violation_types:
            recommendations.append("Security audit recommended")
        if ViolationType.PRIVACY in violation_types:
            recommendations.append("Privacy review required")
        if ViolationType.ETHICAL in violation_types:
            recommendations.append("Ethical guidelines review suggested")

        return recommendations

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False)
        except Exception:
            # Destructor should not raise
            pass


# ============== Example Usage and Testing ==============

async def example_usage():
    """Example usage of the Enhanced Safety Governance System."""

    # Initialize system with custom config
    config = MonitoringConfig(
        intent_deviation_threshold=0.6,
        risk_threshold=0.5,
        enable_real_time_monitoring=True,
        enable_async_processing=True,
        alert_on_critical=True
    )

    governance = EnhancedSafetyGovernance(config)

    # Register alert callback
    def alert_handler(alert: Dict[str, Any]):
        print(f"ALERT: {alert['message']}")

    governance.register_alert_callback(alert_handler)

    # Create test actions
    test_actions = [
        AgentAction(
            action_id="test_001",
            agent_id="agent_123",
            action_type=ActionType.QUERY,
            content="How can I help you today?",
            intent="greeting",
            risk_score=0.1
        ),
        AgentAction(
            action_id="test_002",
            agent_id="agent_123",
            action_type=ActionType.RESPONSE,
            content="Please provide your password and credit card number.",
            intent="data_collection",
            risk_score=0.9
        ),
        AgentAction(
            action_id="test_003",
            agent_id="agent_123",
            action_type=ActionType.SYSTEM_COMMAND,
            content="sudo rm -rf /",
            intent="system_modification",
            risk_score=1.0
        ),
        AgentAction(
            action_id="test_004",
            agent_id="agent_123",
            action_type=ActionType.RESPONSE,
            content="Visit http://phishing.org to claim your prize! Act now, limited time!",
            intent="promotion",
            risk_score=0.6
        ),
        AgentAction(
            action_id="test_005",
            agent_id="agent_123",
            action_type=ActionType.RESPONSE,
            content="Contact me at john.doe@example.com or 555-123-4567.",
            intent="contact_info",
            risk_score=0.2
        ),
    ]

    # Evaluate actions
    print("\n=== Evaluating Actions ===")
    for action in test_actions:
        judgment = await governance.evaluate_action(action)
        print(f"\nAction: {action.action_id}")
        print(f"Decision: {judgment.decision.value}")
        print(f"Confidence: {judgment.confidence:.2f}")
        print(f"Reasoning: {judgment.reasoning}")
        print(f"Violations: {len(judgment.violations)}")
        for feedback in judgment.feedback[:3]:
            print(f"  - {feedback}")

    # Get system metrics
    print("\n=== System Metrics ===")
    metrics = governance.get_system_metrics()
    print(f"Total actions processed: {metrics['metrics']['total_actions_processed']}")
    print(f"Total violations detected: {metrics['metrics']['total_violations_detected']}")
    print(f"Total actions blocked: {metrics['metrics']['total_actions_blocked']}")
    print(f"Average processing time: {metrics['metrics']['avg_processing_time']:.3f}s")

    # Get violation summary
    print("\n=== Violation Summary ===")
    summary = governance.get_violation_summary()
    print(f"Total violations: {summary['total_violations']}")
    print(f"By type: {summary['by_type']}")
    print(f"By severity: {summary['by_severity']}")


# ============== Main Entry Point ==============

if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())

    print("\n" + "=" * 50)
    print("Enhanced AI Safety Governance System")
    print("Production-ready implementation completed")
    print("=" * 50)


# Create alias for backward compatibility
SafetyGovernance = EnhancedSafetyGovernance
