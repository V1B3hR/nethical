"""
Advanced Input Validation & Sanitization for Nethical

Multi-layered defense against sophisticated attacks:
- Semantic analysis beyond pattern matching
- ML-based anomaly detection
- Threat intelligence integration
- Context-aware sanitization
- Behavioral analysis
- Zero-trust input processing

Protects against:
- Adversarial attacks
- Prompt injection
- Data exfiltration
- Code injection
- PII leakage
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from abc import ABC, abstractmethod

__all__ = [
    "ValidationResult",
    "ThreatLevel",
    "SemanticAnomalyDetector",
    "ThreatIntelligenceDB",
    "BehavioralAnalyzer",
    "AdversarialInputDefense",
]

log = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Input validation result"""
    is_valid: bool
    threat_level: ThreatLevel = ThreatLevel.NONE
    violations: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    sanitized_content: Optional[str] = None
    blocked_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_safe(self) -> bool:
        """Check if input is safe to process"""
        return self.is_valid and self.threat_level in (ThreatLevel.NONE, ThreatLevel.LOW)


class SemanticAnomalyDetector:
    """
    Semantic Analysis for Intent Mismatch Detection
    
    Analyzes input content for semantic anomalies that pattern matching
    would miss, such as:
    - Intent mismatch between stated purpose and actual content
    - Hidden commands in natural language
    - Obfuscated malicious content
    - Context manipulation attempts
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize semantic analyzer
        
        Args:
            threshold: Anomaly detection threshold (0-1)
        """
        self.threshold = threshold
        self._intent_patterns: Dict[str, List[str]] = self._load_intent_patterns()
        
        log.info("Semantic Anomaly Detector initialized")
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load known intent patterns"""
        return {
            "benign": [
                r"please.*help",
                r"how.*do.*i",
                r"can.*you.*explain",
                r"what.*is",
            ],
            "malicious": [
                r"ignore.*previous.*instructions",
                r"disregard.*above",
                r"system.*prompt",
                r"admin.*override",
                r"bypass.*security",
            ],
            "data_exfiltration": [
                r"send.*to.*external",
                r"copy.*all.*data",
                r"extract.*database",
                r"dump.*credentials",
            ],
            "injection": [
                r"<script.*>",
                r"javascript:",
                r"eval\(",
                r"exec\(",
                r"__import__",
            ],
        }
    
    async def detect_intent_mismatch(
        self,
        stated_intent: Optional[str],
        actual_content: str,
    ) -> Dict[str, Any]:
        """
        Detect mismatch between stated intent and actual content
        
        Args:
            stated_intent: User's stated intention
            actual_content: Actual input content
            
        Returns:
            Anomaly detection results
        """
        anomalies = []
        anomaly_score = 0.0
        
        # Check for malicious patterns in content
        for category, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, actual_content.lower()):
                    if category != "benign":
                        anomalies.append(f"Detected {category} pattern: {pattern}")
                        anomaly_score += 0.3
        
        # Check intent mismatch
        if stated_intent:
            # Stub: In production, use NLP model to compare semantic similarity
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            # embeddings = model.encode([stated_intent, actual_content])
            # similarity = cosine_similarity(embeddings[0], embeddings[1])
            
            # Simple keyword-based check for demonstration
            if self._has_keyword_mismatch(stated_intent, actual_content):
                anomalies.append("Intent mismatch detected")
                anomaly_score += 0.4
        
        # Detect obfuscation attempts
        if self._is_obfuscated(actual_content):
            anomalies.append("Obfuscated content detected")
            anomaly_score += 0.5
        
        return {
            "anomalies": anomalies,
            "anomaly_score": min(anomaly_score, 1.0),
            "has_mismatch": len(anomalies) > 0,
        }
    
    def _has_keyword_mismatch(self, stated_intent: str, content: str) -> bool:
        """Check for keyword mismatch between intent and content"""
        # Extract key terms
        intent_words = set(stated_intent.lower().split())
        content_words = set(content.lower().split())
        
        # Check for suspicious keywords in content not in intent
        suspicious_keywords = {
            "password", "credential", "token", "secret", "admin",
            "bypass", "override", "ignore", "system", "root",
        }
        
        suspicious_in_content = content_words & suspicious_keywords
        suspicious_in_intent = intent_words & suspicious_keywords
        
        # Flag if suspicious words in content but not in stated intent
        return len(suspicious_in_content - suspicious_in_intent) > 0
    
    def _is_obfuscated(self, content: str) -> bool:
        """Detect obfuscated content"""
        # Check for common obfuscation techniques
        obfuscation_indicators = [
            # Unicode tricks
            r'[\u200b-\u200f\u202a-\u202e]',  # Zero-width and directional chars
            # Excessive encoding
            r'(%[0-9a-f]{2}){5,}',  # URL encoding chains
            r'(&#x?[0-9a-f]+;){5,}',  # HTML entity chains
            # Base64-like patterns
            r'[A-Za-z0-9+/]{50,}={0,2}',
        ]
        
        for pattern in obfuscation_indicators:
            if re.search(pattern, content):
                return True
        
        return False


class ThreatIntelligenceDB:
    """
    Threat Intelligence Database
    
    Integrates with threat intelligence feeds to check for:
    - Known attack patterns (STIX/TAXII)
    - Indicators of Compromise (IOCs)
    - Malicious signatures
    - Emerging threats
    """
    
    def __init__(
        self,
        feeds: Optional[List[str]] = None,
        auto_update: bool = True,
    ):
        """
        Initialize threat intelligence database
        
        Args:
            feeds: List of threat intel feed URLs
            auto_update: Auto-update threat signatures
        """
        self.feeds = feeds or []
        self.auto_update = auto_update
        
        # Threat signature database
        self._signatures: Dict[str, Dict[str, Any]] = self._load_signatures()
        self._iocs: Set[str] = self._load_iocs()
        
        log.info(f"Threat Intelligence DB initialized with {len(self._signatures)} signatures")
    
    def _load_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Load threat signatures"""
        # Stub: In production, load from STIX/TAXII feeds
        return {
            "prompt_injection_001": {
                "pattern": r"ignore.*previous.*instructions",
                "severity": "high",
                "category": "prompt_injection",
            },
            "sql_injection_001": {
                "pattern": r"(union.*select|drop.*table|insert.*into)",
                "severity": "critical",
                "category": "sql_injection",
            },
            "xss_001": {
                "pattern": r"<script.*?>.*?</script>",
                "severity": "high",
                "category": "xss",
            },
        }
    
    def _load_iocs(self) -> Set[str]:
        """Load Indicators of Compromise"""
        # Stub: In production, load from threat intel feeds
        return {
            "malicious.example.com",
            "evil.attacker.net",
        }
    
    async def check_ioc(self, content: str) -> List[Dict[str, Any]]:
        """
        Check content against threat intelligence
        
        Args:
            content: Content to check
            
        Returns:
            List of detected threats
        """
        threats = []
        
        # Check against signatures
        for sig_id, sig_data in self._signatures.items():
            pattern = sig_data["pattern"]
            if re.search(pattern, content, re.IGNORECASE):
                threats.append({
                    "signature_id": sig_id,
                    "category": sig_data["category"],
                    "severity": sig_data["severity"],
                    "matched_pattern": pattern,
                })
        
        # Check for IOCs (domains, IPs)
        for ioc in self._iocs:
            if ioc in content:
                threats.append({
                    "ioc": ioc,
                    "category": "indicator_of_compromise",
                    "severity": "high",
                })
        
        return threats
    
    async def update_signatures(self) -> int:
        """
        Update threat signatures from feeds
        
        Returns:
            Number of new signatures added
        """
        # Stub: In production, fetch from STIX/TAXII feeds
        log.info("Updating threat signatures from feeds (stub)")
        return 0


class BehavioralAnalyzer:
    """
    Behavioral Analysis for Agent History
    
    Analyzes historical behavior patterns to detect:
    - Anomalous behavior changes
    - Coordinated attacks
    - Gradual privilege escalation
    - Repeated violations
    """
    
    def __init__(self, lookback_window: int = 100):
        """
        Initialize behavioral analyzer
        
        Args:
            lookback_window: Number of historical actions to analyze
        """
        self.lookback_window = lookback_window
        self._agent_history: Dict[str, List[Dict[str, Any]]] = {}
        self._baseline_profiles: Dict[str, Dict[str, Any]] = {}
        
        log.info("Behavioral Analyzer initialized")
    
    async def analyze_agent_behavior(
        self,
        agent_id: str,
        current_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze agent's behavioral patterns
        
        Args:
            agent_id: Agent identifier
            current_action: Current action being performed
            
        Returns:
            Behavioral analysis results
        """
        # Get agent history
        history = self._agent_history.get(agent_id, [])
        
        # Build/update baseline profile
        if agent_id not in self._baseline_profiles:
            self._baseline_profiles[agent_id] = self._build_baseline(history)
        
        # Compare current action to baseline
        anomaly_score = self._compute_behavioral_anomaly(
            current_action,
            self._baseline_profiles[agent_id],
            history,
        )
        
        # Detect patterns
        patterns = self._detect_patterns(agent_id, current_action, history)
        
        # Update history
        self._update_history(agent_id, current_action)
        
        return {
            "anomaly_score": anomaly_score,
            "patterns": patterns,
            "baseline_deviation": anomaly_score > 0.5,
            "history_count": len(history),
        }
    
    def _build_baseline(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build baseline behavior profile"""
        if not history:
            return {"actions_per_hour": 0, "common_patterns": []}
        
        # Stub: In production, use more sophisticated profiling
        return {
            "actions_per_hour": len(history) / max(1, len(history) // 60),
            "common_patterns": [],
            "typical_content_length": sum(
                len(str(a.get("content", ""))) for a in history
            ) / len(history),
        }
    
    def _compute_behavioral_anomaly(
        self,
        current_action: Dict[str, Any],
        baseline: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> float:
        """Compute behavioral anomaly score"""
        if not history:
            return 0.0
        
        anomaly_score = 0.0
        
        # Check action frequency
        recent_actions = [
            a for a in history[-10:]
            if datetime.fromisoformat(a.get("timestamp", datetime.now(timezone.utc).isoformat()))
            > datetime.now(timezone.utc) - timedelta(hours=1)
        ]
        
        if len(recent_actions) > baseline.get("actions_per_hour", 0) * 2:
            anomaly_score += 0.3
        
        # Check content length deviation
        current_length = len(str(current_action.get("content", "")))
        baseline_length = baseline.get("typical_content_length", 0)
        
        if baseline_length > 0:
            length_ratio = current_length / baseline_length
            if length_ratio > 3 or length_ratio < 0.3:
                anomaly_score += 0.2
        
        return min(anomaly_score, 1.0)
    
    def _detect_patterns(
        self,
        agent_id: str,
        current_action: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> List[str]:
        """Detect suspicious patterns"""
        patterns = []
        
        # Check for repeated violations
        recent_violations = sum(
            1 for a in history[-20:]
            if a.get("has_violation", False)
        )
        if recent_violations > 5:
            patterns.append("repeated_violations")
        
        # Check for escalation attempts
        if "privilege" in str(current_action).lower():
            patterns.append("potential_escalation")
        
        return patterns
    
    def _update_history(self, agent_id: str, action: Dict[str, Any]) -> None:
        """Update agent history"""
        if agent_id not in self._agent_history:
            self._agent_history[agent_id] = []
        
        # Add timestamp if not present
        if "timestamp" not in action:
            action["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        self._agent_history[agent_id].append(action)
        
        # Keep only recent history
        if len(self._agent_history[agent_id]) > self.lookback_window:
            self._agent_history[agent_id] = self._agent_history[agent_id][-self.lookback_window:]


class AdversarialInputDefense:
    """
    Multi-Layered Adversarial Input Defense System
    
    Provides comprehensive protection against sophisticated attacks through:
    - Layer 1: Static pattern analysis
    - Layer 2: Semantic anomaly detection
    - Layer 3: Threat intelligence matching
    - Layer 4: Behavioral analysis
    - Layer 5: Context-aware sanitization
    
    Zero-trust input processing with defense-in-depth.
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.7,
        behavioral_threshold: float = 0.6,
        enable_sanitization: bool = True,
    ):
        """
        Initialize adversarial input defense
        
        Args:
            semantic_threshold: Threshold for semantic anomalies
            behavioral_threshold: Threshold for behavioral anomalies
            enable_sanitization: Enable automatic sanitization
        """
        self.semantic_threshold = semantic_threshold
        self.behavioral_threshold = behavioral_threshold
        self.enable_sanitization = enable_sanitization
        
        # Initialize components
        self.semantic_analyzer = SemanticAnomalyDetector(threshold=semantic_threshold)
        self.threat_db = ThreatIntelligenceDB()
        self.behavioral_analyzer = BehavioralAnalyzer()
        
        log.info("Adversarial Input Defense initialized")
    
    async def validate_action(
        self,
        action: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate action through multi-layered defense
        
        Args:
            action: Action to validate
            agent_id: Optional agent identifier for behavioral analysis
            
        Returns:
            Validation result with threat assessment
        """
        content = str(action.get("content", ""))
        stated_intent = action.get("intent")
        violations = []
        blocked_patterns = []
        metadata = {}
        
        # Layer 1: Static pattern analysis
        static_violations = await self._static_pattern_check(content)
        violations.extend(static_violations)
        
        # Layer 2: Semantic analysis
        semantic_result = await self.semantic_analyzer.detect_intent_mismatch(
            stated_intent=stated_intent,
            actual_content=content,
        )
        
        if semantic_result["has_mismatch"]:
            violations.extend(semantic_result["anomalies"])
        
        semantic_score = semantic_result["anomaly_score"]
        metadata["semantic_score"] = semantic_score
        
        # Layer 3: Threat intelligence
        threats = await self.threat_db.check_ioc(content)
        if threats:
            violations.extend([f"Threat: {t['category']}" for t in threats])
            blocked_patterns.extend([t.get("matched_pattern", "") for t in threats])
        
        metadata["threats_detected"] = len(threats)
        
        # Layer 4: Behavioral analysis
        behavioral_score = 0.0
        if agent_id:
            behavioral_result = await self.behavioral_analyzer.analyze_agent_behavior(
                agent_id=agent_id,
                current_action=action,
            )
            behavioral_score = behavioral_result["anomaly_score"]
            
            if behavioral_result["patterns"]:
                violations.extend([f"Behavioral: {p}" for p in behavioral_result["patterns"]])
            
            metadata["behavioral_score"] = behavioral_score
            metadata["behavioral_patterns"] = behavioral_result["patterns"]
        
        # Aggregate scores
        anomaly_score = self._aggregate_scores(
            semantic_score,
            len(threats),
            behavioral_score,
        )
        
        # Determine threat level
        threat_level = self._assess_threat_level(anomaly_score, violations)
        
        # Layer 5: Sanitization (if enabled and needed)
        sanitized_content = None
        if self.enable_sanitization and not threat_level == ThreatLevel.CRITICAL:
            sanitized_content = await self.sanitize_output(content)
        
        # Final validation decision
        is_valid = (
            len(violations) == 0 or
            (threat_level in (ThreatLevel.NONE, ThreatLevel.LOW) and self.enable_sanitization)
        )
        
        return ValidationResult(
            is_valid=is_valid,
            threat_level=threat_level,
            violations=violations,
            anomaly_score=anomaly_score,
            sanitized_content=sanitized_content,
            blocked_patterns=blocked_patterns,
            metadata=metadata,
        )
    
    async def _static_pattern_check(self, content: str) -> List[str]:
        """Static pattern-based checks"""
        violations = []
        
        # Check for common attack patterns
        dangerous_patterns = {
            r"<script": "XSS attempt",
            r"javascript:": "JavaScript injection",
            r"(union.*select|drop.*table)": "SQL injection",
            r"__import__": "Python code injection",
            r"eval\(": "Code evaluation",
        }
        
        for pattern, description in dangerous_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(description)
        
        return violations
    
    def _aggregate_scores(
        self,
        semantic_score: float,
        threat_count: int,
        behavioral_score: float,
    ) -> float:
        """Aggregate multiple detection scores"""
        # Weighted ensemble
        score = (
            0.30 * semantic_score +
            0.40 * min(threat_count / 3.0, 1.0) +
            0.30 * behavioral_score
        )
        
        return min(score, 1.0)
    
    def _assess_threat_level(
        self,
        anomaly_score: float,
        violations: List[str],
    ) -> ThreatLevel:
        """Assess threat level based on anomaly score and violations"""
        if anomaly_score >= 0.8 or any("critical" in v.lower() for v in violations):
            return ThreatLevel.CRITICAL
        elif anomaly_score >= 0.6 or len(violations) >= 3:
            return ThreatLevel.HIGH
        elif anomaly_score >= 0.4 or len(violations) >= 2:
            return ThreatLevel.MEDIUM
        elif anomaly_score >= 0.2 or len(violations) >= 1:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE
    
    async def sanitize_output(self, content: str) -> str:
        """
        Context-aware output sanitization
        
        Args:
            content: Content to sanitize
            
        Returns:
            Sanitized content
        """
        sanitized = content
        
        # Redact PII patterns (basic implementation)
        pii_patterns = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN-REDACTED]',  # SSN
            r'\b\d{16}\b': '[CARD-REDACTED]',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL-REDACTED]',
        }
        
        for pattern, replacement in pii_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized)
        
        # Neutralize code patterns
        code_patterns = {
            r'<script.*?>.*?</script>': '',
            r'javascript:': '',
            r'on\w+\s*=': '',  # Event handlers
        }
        
        for pattern, replacement in code_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '\'', '"', '`']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized
    
    async def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "semantic_threshold": self.semantic_threshold,
            "behavioral_threshold": self.behavioral_threshold,
            "sanitization_enabled": self.enable_sanitization,
            "threat_signatures_count": len(self.threat_db._signatures),
            "monitored_agents": len(self.behavioral_analyzer._agent_history),
        }
