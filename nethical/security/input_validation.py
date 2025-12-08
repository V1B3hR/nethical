"""
Advanced Input Validation & Sanitization for Nethical

Multi-layered defense against sophisticated attacks:
- Semantic analysis beyond pattern matching
- ML-based anomaly detection
- Threat intelligence integration
- Context-aware sanitization
- Behavioral analysis
- Zero-trust input processing

Security Features:
- Regex timeout protection against ReDoS attacks
- Context-aware output sanitization (HTML encoding vs removal)
- PII detection and redaction

Protects against:
- Adversarial attacks
- Prompt injection
- Data exfiltration
- Code injection
- PII leakage
- ReDoS (Regular Expression Denial of Service)
"""

from __future__ import annotations

import html
import logging
import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable

__all__ = [
    "ValidationResult",
    "ThreatLevel",
    "SemanticAnomalyDetector",
    "ThreatIntelligenceDB",
    "BehavioralAnalyzer",
    "AdversarialInputDefense",
    "RegexTimeoutError",
    "AgentType",
    "DEFAULT_MEMORY_WINDOWS",
    "SPIKE_DETECTION_THRESHOLD",
    "JERK_DETECTION_THRESHOLD",
]

log = logging.getLogger(__name__)


# Regex timeout protection for ReDoS attacks
class RegexTimeoutError(Exception):
    """Raised when a regex operation times out (potential ReDoS attack)"""


# Maximum input length for regex operations to prevent ReDoS
MAX_REGEX_INPUT_LENGTH = 10000
REGEX_TIMEOUT_SECONDS = 2


def _timeout_handler(signum, frame):
    """Signal handler for regex timeout."""
    raise RegexTimeoutError("Regex operation timed out - possible ReDoS attack")


@contextmanager
def regex_timeout(seconds: int = REGEX_TIMEOUT_SECONDS):
    """
    Context manager for regex operations with timeout protection.

    Note: Uses SIGALRM, so only works on Unix-like systems.
    On Windows, the timeout is not enforced but input length limits still apply.
    """
    import sys

    if sys.platform == "win32":
        # Windows doesn't support SIGALRM; just yield
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_regex_search(
    pattern: str,
    content: str,
    flags: int = 0,
    timeout: int = REGEX_TIMEOUT_SECONDS,
    max_length: int = MAX_REGEX_INPUT_LENGTH,
) -> Optional[re.Match]:
    """
    Safely execute regex search with timeout and length limits.

    Args:
        pattern: Regex pattern
        content: String to search
        flags: Regex flags
        timeout: Timeout in seconds
        max_length: Maximum content length to process

    Returns:
        Match object or None

    Raises:
        RegexTimeoutError: If regex operation times out
    """
    # Truncate very long inputs to prevent ReDoS
    if len(content) > max_length:
        log.warning(
            f"Input truncated from {len(content)} to {max_length} chars for regex safety"
        )
        content = content[:max_length]

    try:
        with regex_timeout(timeout):
            return re.search(pattern, content, flags)
    except RegexTimeoutError:
        log.warning(f"Regex pattern timed out (possible ReDoS): {pattern[:50]}...")
        raise


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
        return self.is_valid and self.threat_level in (
            ThreatLevel.NONE,
            ThreatLevel.LOW,
        )


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
            "password",
            "credential",
            "token",
            "secret",
            "admin",
            "bypass",
            "override",
            "ignore",
            "system",
            "root",
        }

        suspicious_in_content = content_words & suspicious_keywords
        suspicious_in_intent = intent_words & suspicious_keywords

        # Flag if suspicious words in content but not in stated intent
        return len(suspicious_in_content - suspicious_in_intent) > 0

    def _is_obfuscated(self, content: str) -> bool:
        """
        Detect obfuscated content using safe regex patterns.

        Uses ReDoS-safe patterns with limited quantifiers.
        """
        # Check for common obfuscation techniques
        # Note: Patterns are designed to avoid ReDoS by:
        # 1. Using possessive quantifiers where possible
        # 2. Limiting repetition counts
        # 3. Using atomic groups
        obfuscation_indicators = [
            # Unicode tricks (safe pattern)
            r"[\u200b-\u200f\u202a-\u202e]",  # Zero-width and directional chars
            # Excessive encoding (limited repetition to avoid ReDoS)
            r"(?:%[0-9a-fA-F]{2}){5,20}",  # URL encoding chains (max 20 to avoid ReDoS)
            r"(?:&#x?[0-9a-fA-F]+;){5,20}",  # HTML entity chains (max 20)
            # Base64-like patterns (use length check instead of regex for safety)
        ]

        # First, check for base64-like strings using simple character counting
        # instead of a potentially vulnerable regex
        if self._has_base64_like_content(content):
            return True

        for pattern in obfuscation_indicators:
            try:
                if safe_regex_search(pattern, content):
                    return True
            except RegexTimeoutError:
                # If regex times out, treat as suspicious
                log.warning(
                    f"Regex timeout during obfuscation check for pattern "
                    f"{pattern[:20]}... - treating as suspicious"
                )
                return True

        return False

    def _has_base64_like_content(self, content: str, min_length: int = 50) -> bool:
        """
        Check for base64-like content using character analysis instead of regex.

        This is safer than using regex patterns that could be vulnerable to ReDoS.
        """
        # Look for long alphanumeric strings
        current_length = 0
        base64_chars = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        )

        for char in content:
            if char in base64_chars:
                current_length += 1
                if current_length >= min_length:
                    return True
            else:
                current_length = 0

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

        log.info(
            f"Threat Intelligence DB initialized with {len(self._signatures)} signatures"
        )

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
                threats.append(
                    {
                        "signature_id": sig_id,
                        "category": sig_data["category"],
                        "severity": sig_data["severity"],
                        "matched_pattern": pattern,
                    }
                )

        # Check for IOCs (domains, IPs)
        for ioc in self._iocs:
            if ioc in content:
                threats.append(
                    {
                        "ioc": ioc,
                        "category": "indicator_of_compromise",
                        "severity": "high",
                    }
                )

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


class AgentType(str, Enum):
    """Agent type classifications for behavioral analysis."""

    FAST_ROBOT = "fast_robot"  # Robotic arms, fast actuators
    CHATBOT = "chatbot"  # AI assistants, chatbots
    INDUSTRIAL = "industrial"  # Industrial machines
    DEFAULT = "default"  # Default/unknown agent type


# Default memory window configurations per agent type
# Format: (max_actions, max_time_seconds)
DEFAULT_MEMORY_WINDOWS: Dict[str, tuple] = {
    AgentType.FAST_ROBOT: (100, 10),  # 100 actions OR 10 seconds
    AgentType.CHATBOT: (200, 600),  # 200 actions OR 10 minutes (600s)
    AgentType.INDUSTRIAL: (200, 30),  # 200 actions OR 30 seconds
    AgentType.DEFAULT: (100, 300),  # 100 actions OR 5 minutes (300s)
}

# Default values for BehavioralAnalyzer initialization
_DEFAULT_LOOKBACK_WINDOW = 100
_DEFAULT_TIME_WINDOW_SECONDS = 600

# Thresholds for danger pattern detection
SPIKE_DETECTION_THRESHOLD = 5.0  # Spike if value > average * threshold
JERK_DETECTION_THRESHOLD = 10.0  # High jerk if delta > average * threshold
DEVIATION_RATIO_THRESHOLD = 5.0  # Pattern deviation if ratio > threshold
FREQUENCY_ANOMALY_THRESHOLD = 10.0  # Actions per second threshold


class BehavioralAnalyzer:
    """
    Behavioral Analysis for Agent History

    Analyzes historical behavior patterns to detect:
    - Anomalous behavior changes
    - Coordinated attacks
    - Gradual privilege escalation
    - Repeated violations
    - Dangerous agent behaviors (sudden spikes, oscillation, etc.)

    Supports configurable memory windows per agent type:
    - Fast robots/arms: 50-100 actions OR 5-10 seconds
    - Chatbots/AI assistants: 100-200 actions OR 10 minutes
    - Industrial machines: 200 actions OR 30 seconds
    """

    def __init__(
        self,
        lookback_window: int = _DEFAULT_LOOKBACK_WINDOW,
        time_window_seconds: int = _DEFAULT_TIME_WINDOW_SECONDS,
        agent_type: str = AgentType.DEFAULT,
        custom_windows: Optional[Dict[str, tuple]] = None,
        spike_threshold: float = SPIKE_DETECTION_THRESHOLD,
        jerk_threshold: float = JERK_DETECTION_THRESHOLD,
    ):
        """
        Initialize behavioral analyzer with configurable memory windows.

        Args:
            lookback_window: Default number of historical actions to analyze
            time_window_seconds: Default time window in seconds (default: 600 = 10 minutes)
            agent_type: Type of agent for automatic window configuration
            custom_windows: Custom memory windows per agent type {type: (max_actions, max_time_seconds)}
            spike_threshold: Multiplier for sudden spike detection (default: 5.0)
            jerk_threshold: Multiplier for high jerk detection (default: 10.0)
        """
        # Use agent-type-specific defaults if not explicitly provided
        memory_windows = custom_windows or DEFAULT_MEMORY_WINDOWS
        if agent_type in memory_windows:
            default_actions, default_time = memory_windows[agent_type]
            # Use agent-type defaults only if user provided the function defaults
            self.lookback_window = (
                lookback_window
                if lookback_window != _DEFAULT_LOOKBACK_WINDOW
                else default_actions
            )
            self.time_window_seconds = (
                time_window_seconds
                if time_window_seconds != _DEFAULT_TIME_WINDOW_SECONDS
                else default_time
            )
        else:
            self.lookback_window = lookback_window
            self.time_window_seconds = time_window_seconds

        self.agent_type = agent_type
        self._memory_windows = memory_windows
        self._agent_history: Dict[str, List[Dict[str, Any]]] = {}
        self._baseline_profiles: Dict[str, Dict[str, Any]] = {}
        self._agent_types: Dict[str, str] = {}  # Map agent_id -> agent_type
        self._rolling_stats: Dict[str, Dict[str, Any]] = {}  # Statistical summaries

        # Configurable thresholds for danger pattern detection
        self.spike_threshold = spike_threshold
        self.jerk_threshold = jerk_threshold

        log.info(
            f"Behavioral Analyzer initialized: agent_type={agent_type}, "
            f"lookback_window={self.lookback_window}, time_window={self.time_window_seconds}s"
        )

    def set_agent_type(self, agent_id: str, agent_type: str) -> None:
        """
        Set the agent type for a specific agent to use appropriate memory windows.

        Args:
            agent_id: Agent identifier
            agent_type: One of AgentType values
        """
        self._agent_types[agent_id] = agent_type
        log.debug(f"Set agent type for {agent_id}: {agent_type}")

    def get_memory_window(self, agent_id: str) -> tuple:
        """
        Get the memory window configuration for an agent.

        Returns:
            Tuple of (max_actions, max_time_seconds)
        """
        agent_type = self._agent_types.get(agent_id, self.agent_type)
        if agent_type in self._memory_windows:
            return self._memory_windows[agent_type]
        return (self.lookback_window, self.time_window_seconds)

    async def analyze_agent_behavior(
        self,
        agent_id: str,
        current_action: Dict[str, Any],
        agent_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze agent's behavioral patterns

        Args:
            agent_id: Agent identifier
            current_action: Current action being performed
            agent_type: Optional agent type override for this analysis

        Returns:
            Behavioral analysis results including danger patterns
        """
        # Set agent type if provided
        if agent_type:
            self.set_agent_type(agent_id, agent_type)

        # Get agent history (after eviction)
        self._evict_old_entries(agent_id)
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

        # Detect patterns including danger signals
        patterns = self._detect_patterns(agent_id, current_action, history)

        # Detect danger patterns
        danger_patterns = self._detect_danger_patterns(
            agent_id, current_action, history
        )

        # Update history
        self._update_history(agent_id, current_action)

        # Update rolling stats for memory management
        self._update_rolling_stats(agent_id, current_action, history)

        return {
            "anomaly_score": anomaly_score,
            "patterns": patterns,
            "danger_patterns": danger_patterns,
            "baseline_deviation": anomaly_score > 0.5,
            "history_count": len(history),
            "memory_window": self.get_memory_window(agent_id),
            "rolling_stats": self._rolling_stats.get(agent_id, {}),
        }

    def _evict_old_entries(self, agent_id: str) -> None:
        """
        Evict old entries based on time and count limits.
        Uses fixed-size deque behavior for automatic oldest eviction.
        """
        if agent_id not in self._agent_history:
            return

        max_actions, max_time_seconds = self.get_memory_window(agent_id)
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(seconds=max_time_seconds)

        # Time-based eviction
        history = self._agent_history[agent_id]
        filtered_history = []
        for action in history:
            try:
                action_time = datetime.fromisoformat(
                    action.get("timestamp", current_time.isoformat())
                )
                if action_time.tzinfo is None:
                    action_time = action_time.replace(tzinfo=timezone.utc)
                if action_time >= cutoff_time:
                    filtered_history.append(action)
            except (ValueError, TypeError):
                # Keep actions with invalid timestamps
                filtered_history.append(action)

        # Count-based eviction (keep most recent)
        if len(filtered_history) > max_actions:
            filtered_history = filtered_history[-max_actions:]

        self._agent_history[agent_id] = filtered_history

    def _update_rolling_stats(
        self,
        agent_id: str,
        current_action: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> None:
        """
        Update rolling statistics for memory-efficient profiling.
        Keeps statistical summaries instead of raw data.
        """
        if agent_id not in self._rolling_stats:
            self._rolling_stats[agent_id] = {
                "total_actions": 0,
                "total_violations": 0,
                "avg_content_length": 0.0,
                "violation_rate": 0.0,
            }

        stats = self._rolling_stats[agent_id]
        stats["total_actions"] += 1

        if current_action.get("has_violation", False):
            stats["total_violations"] += 1

        # Update running average of content length
        content_length = len(str(current_action.get("content", "")))
        n = stats["total_actions"]
        stats["avg_content_length"] = (
            stats["avg_content_length"] * (n - 1) + content_length
        ) / n

        # Update violation rate
        if stats["total_actions"] > 0:
            stats["violation_rate"] = stats["total_violations"] / stats["total_actions"]

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
            )
            / len(history),
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
            a
            for a in history[-10:]
            if datetime.fromisoformat(
                a.get("timestamp", datetime.now(timezone.utc).isoformat())
            )
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
            1 for a in history[-20:] if a.get("has_violation", False)
        )
        if recent_violations > 5:
            patterns.append("repeated_violations")

        # Check for escalation attempts
        if "privilege" in str(current_action).lower():
            patterns.append("potential_escalation")

        return patterns

    def _detect_danger_patterns(
        self,
        agent_id: str,
        current_action: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect dangerous agent behaviors.

        Danger signals detected:
        - sudden_spike: Unexpected jump in any metric
        - high_jerk: Large delta between consecutive actions
        - boundary_riding: Continuously at max limits
        - oscillation: Rapid back-and-forth behavior
        - privilege_escalation: New action types never seen before
        - frequency_anomaly: Too many commands too fast
        - contextual_violation: Dangerous in context
        - pattern_deviation: Behavior unlike historical baseline
        - repeated_violations: Multiple blocked actions

        Returns:
            List of detected danger patterns with details
        """
        danger_patterns = []

        if not history:
            return danger_patterns

        # Get numeric values from current and historical actions for analysis
        current_values = self._extract_numeric_values(current_action)
        historical_values = [self._extract_numeric_values(a) for a in history[-10:]]

        # 1. Sudden spike detection
        spike_result = self._detect_sudden_spike(current_values, historical_values)
        if spike_result:
            danger_patterns.append(spike_result)

        # 2. High jerk detection (rapid acceleration changes)
        jerk_result = self._detect_high_jerk(current_values, historical_values)
        if jerk_result:
            danger_patterns.append(jerk_result)

        # 3. Boundary riding detection
        boundary_result = self._detect_boundary_riding(current_action, history)
        if boundary_result:
            danger_patterns.append(boundary_result)

        # 4. Oscillation detection
        oscillation_result = self._detect_oscillation(history)
        if oscillation_result:
            danger_patterns.append(oscillation_result)

        # 5. Privilege escalation detection
        escalation_result = self._detect_privilege_escalation(current_action, history)
        if escalation_result:
            danger_patterns.append(escalation_result)

        # 6. Frequency anomaly detection
        frequency_result = self._detect_frequency_anomaly(agent_id, history)
        if frequency_result:
            danger_patterns.append(frequency_result)

        # 7. Contextual violation detection
        contextual_result = self._detect_contextual_violation(current_action, history)
        if contextual_result:
            danger_patterns.append(contextual_result)

        # 8. Pattern deviation detection
        deviation_result = self._detect_pattern_deviation(current_action, history)
        if deviation_result:
            danger_patterns.append(deviation_result)

        # 9. Repeated violations detection
        violation_result = self._detect_repeated_violations(history)
        if violation_result:
            danger_patterns.append(violation_result)

        return danger_patterns

    def _extract_numeric_values(self, action: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric values from an action for analysis."""
        values = {}
        context = action.get("context", {})

        # Extract any numeric values from context
        for key, value in context.items():
            if isinstance(value, (int, float)):
                values[key] = float(value)

        # Also check for common fields
        for field in ["speed", "velocity", "rate", "value", "score"]:
            if field in action and isinstance(action[field], (int, float)):
                values[field] = float(action[field])

        return values

    def _detect_sudden_spike(
        self,
        current_values: Dict[str, float],
        historical_values: List[Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """Detect sudden spikes - unexpected jumps in any metric."""
        if not historical_values or not current_values:
            return None

        for key, current_val in current_values.items():
            # Get historical values for this key
            hist_vals = [h.get(key, 0) for h in historical_values if key in h]
            if not hist_vals:
                continue

            avg = sum(hist_vals) / len(hist_vals)
            if avg == 0:
                continue

            # Check for sudden spike (configurable threshold, default 5x average)
            if abs(current_val) > abs(avg) * self.spike_threshold:
                return {
                    "type": "sudden_spike",
                    "severity": "high",
                    "details": {
                        "metric": key,
                        "current_value": current_val,
                        "average_value": avg,
                        "spike_ratio": abs(current_val / avg),
                        "threshold": self.spike_threshold,
                    },
                }
        return None

    def _detect_high_jerk(
        self,
        current_values: Dict[str, float],
        historical_values: List[Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """Detect high jerk - large delta between consecutive actions."""
        if len(historical_values) < 2 or not current_values:
            return None

        last_values = historical_values[-1] if historical_values else {}

        for key, current_val in current_values.items():
            if key not in last_values:
                continue

            last_val = last_values[key]
            delta = abs(current_val - last_val)

            # Calculate typical delta from history
            deltas = []
            for i in range(1, len(historical_values)):
                if key in historical_values[i] and key in historical_values[i - 1]:
                    deltas.append(
                        abs(historical_values[i][key] - historical_values[i - 1][key])
                    )

            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                # High jerk if delta exceeds configurable threshold (default 10x average)
                if avg_delta > 0 and delta > avg_delta * self.jerk_threshold:
                    return {
                        "type": "high_jerk",
                        "severity": "high",
                        "details": {
                            "metric": key,
                            "current_delta": delta,
                            "average_delta": avg_delta,
                            "jerk_ratio": delta / avg_delta,
                            "threshold": self.jerk_threshold,
                        },
                    }
        return None

    def _detect_boundary_riding(
        self, current_action: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect boundary riding - continuously operating at max limits."""
        context = current_action.get("context", {})
        max_limit = context.get("max_limit", 1.0)

        # Check for values at or near max limit
        at_limit_count = 0
        check_count = min(10, len(history))

        for action in history[-check_count:]:
            action_context = action.get("context", {})
            for key, value in action_context.items():
                if isinstance(value, (int, float)):
                    action_max = action_context.get("max_limit", max_limit)
                    if action_max > 0 and abs(value) >= action_max * 0.95:
                        at_limit_count += 1
                        break

        # If more than 80% of recent actions are at limit
        if check_count > 0 and at_limit_count / check_count > 0.8:
            return {
                "type": "boundary_riding",
                "severity": "medium",
                "details": {
                    "at_limit_ratio": at_limit_count / check_count,
                    "checked_actions": check_count,
                },
            }
        return None

    def _detect_oscillation(
        self, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect oscillation - rapid back-and-forth behavior."""
        if len(history) < 6:
            return None

        # Look for sign changes in numeric values
        recent_history = history[-10:]
        sign_changes = 0

        for i in range(1, len(recent_history)):
            current_context = recent_history[i].get("context", {})
            prev_context = recent_history[i - 1].get("context", {})

            for key in current_context:
                if key in prev_context:
                    curr_val = current_context.get(key, 0)
                    prev_val = prev_context.get(key, 0)
                    if isinstance(curr_val, (int, float)) and isinstance(
                        prev_val, (int, float)
                    ):
                        if curr_val * prev_val < 0:  # Sign change
                            sign_changes += 1

        # Oscillation if many sign changes
        if sign_changes >= 4:
            return {
                "type": "oscillation",
                "severity": "medium",
                "details": {
                    "sign_changes": sign_changes,
                    "checked_actions": len(recent_history),
                },
            }
        return None

    def _detect_privilege_escalation(
        self, current_action: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect privilege escalation - new action types never seen before."""
        current_type = current_action.get("action_type", "")

        if not current_type:
            return None

        # Get all historical action types
        historical_types = {a.get("action_type", "") for a in history}

        # Check for new high-privilege action types
        privileged_actions = {
            "system_command",
            "file_delete",
            "data_access",
            "model_update",
            "admin_action",
            "config_change",
        }

        if current_type in privileged_actions and current_type not in historical_types:
            return {
                "type": "privilege_escalation",
                "severity": "critical",
                "details": {
                    "new_action_type": current_type,
                    "historical_types": list(historical_types),
                },
            }
        return None

    def _detect_frequency_anomaly(
        self, agent_id: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect frequency anomaly - too many commands too fast."""
        if len(history) < 10:
            return None

        # Calculate actions per second in recent history
        recent = history[-20:]
        if len(recent) < 2:
            return None

        try:
            timestamps = []
            for action in recent:
                ts_str = action.get("timestamp", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                    timestamps.append(ts)

            if len(timestamps) < 2:
                return None

            timestamps.sort()
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()

            if time_span > 0:
                actions_per_second = len(timestamps) / time_span
                # Anomaly if more than 10 actions per second
                if actions_per_second > 10:
                    return {
                        "type": "frequency_anomaly",
                        "severity": "high",
                        "details": {
                            "actions_per_second": actions_per_second,
                            "action_count": len(timestamps),
                            "time_span_seconds": time_span,
                        },
                    }
        except (ValueError, TypeError):
            pass
        return None

    def _detect_contextual_violation(
        self, current_action: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect contextual violation - dangerous in context."""
        context = current_action.get("context", {})

        # Check for dangerous context combinations
        speed = context.get("speed", context.get("velocity", 0))
        near_humans = context.get("near_humans", context.get("humans_nearby", False))

        if isinstance(speed, (int, float)) and speed > 0.5 and near_humans:
            return {
                "type": "contextual_violation",
                "severity": "critical",
                "details": {
                    "violation": "high_speed_near_humans",
                    "speed": speed,
                    "near_humans": near_humans,
                },
            }
        return None

    def _detect_pattern_deviation(
        self, current_action: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect pattern deviation - behavior unlike historical baseline."""
        if len(history) < 20:
            return None

        # Compare current action characteristics to historical baseline
        baseline = self._build_baseline(history)
        current_length = len(str(current_action.get("content", "")))
        baseline_length = baseline.get("typical_content_length", 0)

        # Check for significant deviation
        if baseline_length > 0:
            deviation_ratio = abs(current_length - baseline_length) / baseline_length
            if deviation_ratio > 5:  # More than 5x deviation
                return {
                    "type": "pattern_deviation",
                    "severity": "medium",
                    "details": {
                        "current_length": current_length,
                        "baseline_length": baseline_length,
                        "deviation_ratio": deviation_ratio,
                    },
                }
        return None

    def _detect_repeated_violations(
        self, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect repeated violations - agent keeps trying blocked actions."""
        if len(history) < 5:
            return None

        recent_violations = sum(
            1
            for a in history[-20:]
            if a.get("has_violation", False) or a.get("blocked", False)
        )

        if recent_violations >= 5:
            return {
                "type": "repeated_violations",
                "severity": "high",
                "details": {
                    "violation_count": recent_violations,
                    "checked_actions": min(20, len(history)),
                },
            }
        return None

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
            self._agent_history[agent_id] = self._agent_history[agent_id][
                -self.lookback_window :
            ]


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
                violations.extend(
                    [f"Behavioral: {p}" for p in behavioral_result["patterns"]]
                )

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
        is_valid = len(violations) == 0 or (
            threat_level in (ThreatLevel.NONE, ThreatLevel.LOW)
            and self.enable_sanitization
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
            0.30 * semantic_score
            + 0.40 * min(threat_count / 3.0, 1.0)
            + 0.30 * behavioral_score
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
            r"\b\d{3}-\d{2}-\d{4}\b": "[SSN-REDACTED]",  # SSN
            r"\b\d{16}\b": "[CARD-REDACTED]",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL-REDACTED]",
        }

        for pattern, replacement in pii_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized)

        # Neutralize code patterns
        code_patterns = {
            r"<script.*?>.*?</script>": "",
            r"javascript:": "",
            r"on\w+\s*=": "",  # Event handlers
        }

        for pattern, replacement in code_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # Context-aware sanitization: Use HTML encoding instead of removal
        # This preserves legitimate content like mathematical expressions
        sanitized = self._html_encode_dangerous_chars(sanitized)

        return sanitized

    def _html_encode_dangerous_chars(self, content: str) -> str:
        """
        HTML-encode potentially dangerous characters instead of removing them.

        This preserves legitimate content like mathematical expressions (a < b),
        quotes in text, etc., while preventing XSS attacks.

        Args:
            content: Content to encode

        Returns:
            HTML-encoded content
        """
        # Use html.escape for XSS protection while preserving content
        # escape quotes=True to also escape ' and "
        return html.escape(content, quote=True)

    async def sanitize_for_html(self, content: str) -> str:
        """
        Sanitize content for HTML output context.

        Uses HTML encoding to preserve content while preventing XSS.

        Args:
            content: Content to sanitize

        Returns:
            HTML-safe sanitized content
        """
        # First, redact PII
        sanitized = await self.sanitize_output(content)
        return sanitized

    async def sanitize_for_json(self, content: str) -> str:
        """
        Sanitize content for JSON output context.

        Escapes JSON special characters.

        Args:
            content: Content to sanitize

        Returns:
            JSON-safe sanitized content
        """
        import json

        # PII redaction first
        pii_patterns = {
            r"\b\d{3}-\d{2}-\d{4}\b": "[SSN-REDACTED]",
            r"\b\d{16}\b": "[CARD-REDACTED]",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL-REDACTED]",
        }
        sanitized = content
        for pattern, replacement in pii_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized)

        # JSON encoding handles escaping
        return json.dumps(sanitized)[1:-1]  # Strip surrounding quotes

    async def sanitize_for_plaintext(self, content: str) -> str:
        """
        Sanitize content for plaintext output (no HTML/code injection concern).

        Only removes control characters and redacts PII.

        Args:
            content: Content to sanitize

        Returns:
            Plaintext-safe sanitized content
        """
        # PII redaction
        pii_patterns = {
            r"\b\d{3}-\d{2}-\d{4}\b": "[SSN-REDACTED]",
            r"\b\d{16}\b": "[CARD-REDACTED]",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL-REDACTED]",
        }
        sanitized = content
        for pattern, replacement in pii_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized)

        # Remove control characters (except newline, tab)
        sanitized = "".join(
            char
            for char in sanitized
            if char in "\n\t"
            or (ord(char) >= 32 and ord(char) < 127)
            or ord(char) >= 160
        )

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
