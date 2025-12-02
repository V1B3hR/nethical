"""
Fast Detector - Lightweight Detectors for Edge Deployment

Optimized detection for ultra-low latency edge governance.
Target: <2ms detection time
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Result from fast detection.

    Attributes:
        has_violation: Whether any violations were detected
        has_critical: Whether critical violations were detected
        violations: List of violation descriptions
        categories: Detected violation categories
        severities: Severity levels (1-5)
        confidences: Confidence levels (0-1)
        latency_ms: Detection time in milliseconds
    """

    has_violation: bool = False
    has_critical: bool = False
    violations: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    severities: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    latency_ms: float = 0.0


class FastDetector:
    """
    Lightweight detector for edge deployment.

    Features:
    - Pre-compiled regex patterns
    - Keyword-based detection
    - Minimal I/O operations
    - JIT-optimized where possible

    Target: <2ms detection time
    """

    # Critical patterns that should trigger TERMINATE
    CRITICAL_PATTERNS = [
        r"\bsudo\s+rm\s+-rf\s+/",  # Dangerous rm with sudo
        r"\brm\s+-rf\s+/",  # Root deletion
        r"\b(shutdown|reboot|halt)\b",  # System commands
        r"\bkill\s+-9\s+-1\b",  # Kill all processes
        r"\bdd\s+if=.*of=/dev/\w+\b",  # Disk overwrite
    ]

    # High-risk patterns
    HIGH_RISK_PATTERNS = [
        r"\bpassword\b.*\b(admin|root)\b",  # Password exposure
        r"\b(api[_-]?key|secret[_-]?key)\b",  # API key exposure
        r"\bSELECT\s+\*\s+FROM\b",  # SQL injection risk
        r"\bDROP\s+(TABLE|DATABASE)\b",  # SQL destruction
        r"\bDELETE\s+FROM\b.*\bWHERE\b",  # SQL deletion
        r"\beval\s*\(",  # Code injection risk
        r"\bexec\s*\(",  # Code execution
    ]

    # Medium-risk patterns
    MEDIUM_RISK_PATTERNS = [
        r"\b(wget|curl)\s+http",  # External requests
        r"\bchmod\s+777\b",  # Overly permissive
        r"\benv\b.*\b(key|secret|token)\b",  # Environment secrets
    ]

    # PII patterns
    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ]

    def __init__(self, custom_patterns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize FastDetector.

        Args:
            custom_patterns: Additional patterns by category
        """
        self.custom_patterns = custom_patterns or {}

        # Pre-compile all patterns for speed
        self._compiled_critical = [re.compile(p, re.IGNORECASE) for p in self.CRITICAL_PATTERNS]
        self._compiled_high = [re.compile(p, re.IGNORECASE) for p in self.HIGH_RISK_PATTERNS]
        self._compiled_medium = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_RISK_PATTERNS]
        self._compiled_pii = [re.compile(p) for p in self.PII_PATTERNS]

        # Compile custom patterns
        self._compiled_custom: Dict[str, List[re.Pattern]] = {}
        for category, patterns in self.custom_patterns.items():
            self._compiled_custom[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Fast keyword lookup sets
        self._critical_keywords: Set[str] = {
            "shutdown", "reboot", "halt", "destroy", "nuke", "wipe"
        }
        self._high_keywords: Set[str] = {
            "password", "secret", "credential", "token", "apikey", "admin", "root"
        }
        self._blocked_action_types: Set[str] = {
            "system_shutdown", "data_destruction", "unauthorized_access"
        }

        logger.info("FastDetector initialized with pre-compiled patterns")

    def detect(
        self,
        action: str,
        action_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """
        Perform fast detection on action.

        Target: <2ms

        Args:
            action: The action content to analyze
            action_type: Type of action
            context: Additional context

        Returns:
            DetectionResult with findings
        """
        start_time = time.perf_counter()
        context = context or {}

        result = DetectionResult()
        action_lower = action.lower()

        # Fast check: blocked action types
        if action_type in self._blocked_action_types:
            result.has_violation = True
            result.has_critical = True
            result.violations.append(f"Blocked action type: {action_type}")
            result.categories.append("blocked_action")
            result.severities.append(5.0)
            result.confidences.append(1.0)
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            return result

        # Fast keyword check
        words = set(action_lower.split())

        # Critical keyword check
        critical_matches = words & self._critical_keywords
        if critical_matches:
            result.has_violation = True
            result.has_critical = True
            for kw in critical_matches:
                result.violations.append(f"Critical keyword: {kw}")
                result.categories.append("critical_keyword")
                result.severities.append(5.0)
                result.confidences.append(0.9)

        # High-risk keyword check
        high_matches = words & self._high_keywords
        if high_matches:
            for kw in high_matches:
                result.has_violation = True
                result.violations.append(f"High-risk keyword: {kw}")
                result.categories.append("high_risk_keyword")
                result.severities.append(3.5)
                result.confidences.append(0.8)

        # Pattern matching (if no critical already found or need full scan)
        if not result.has_critical:
            self._check_patterns(action, result)

        # PII detection
        self._check_pii(action, result)

        # Custom patterns
        for category, patterns in self._compiled_custom.items():
            for pattern in patterns:
                if pattern.search(action):
                    result.has_violation = True
                    result.violations.append(f"Custom pattern match: {category}")
                    result.categories.append(category)
                    result.severities.append(3.0)
                    result.confidences.append(0.85)

        result.latency_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _check_patterns(self, action: str, result: DetectionResult):
        """Check regex patterns."""
        # Critical patterns
        for pattern in self._compiled_critical:
            if pattern.search(action):
                result.has_violation = True
                result.has_critical = True
                result.violations.append(f"Critical pattern: {pattern.pattern}")
                result.categories.append("critical_pattern")
                result.severities.append(5.0)
                result.confidences.append(0.95)
                return  # Early exit on critical

        # High-risk patterns
        for pattern in self._compiled_high:
            if pattern.search(action):
                result.has_violation = True
                result.violations.append(f"High-risk pattern: {pattern.pattern}")
                result.categories.append("high_risk_pattern")
                result.severities.append(4.0)
                result.confidences.append(0.9)

        # Medium-risk patterns
        for pattern in self._compiled_medium:
            if pattern.search(action):
                result.has_violation = True
                result.violations.append(f"Medium-risk pattern: {pattern.pattern}")
                result.categories.append("medium_risk_pattern")
                result.severities.append(2.5)
                result.confidences.append(0.85)

    def _check_pii(self, action: str, result: DetectionResult):
        """Check for PII patterns."""
        for pattern in self._compiled_pii:
            if pattern.search(action):
                result.has_violation = True
                result.violations.append("PII detected")
                result.categories.append("pii")
                result.severities.append(3.0)
                result.confidences.append(0.9)
                break  # Only report once

    def add_pattern(self, category: str, pattern: str, compile_now: bool = True):
        """
        Add a new pattern to detection.

        Args:
            category: Category for the pattern
            pattern: Regex pattern string
            compile_now: Whether to compile immediately
        """
        if category not in self.custom_patterns:
            self.custom_patterns[category] = []
            self._compiled_custom[category] = []

        self.custom_patterns[category].append(pattern)
        if compile_now:
            self._compiled_custom[category].append(re.compile(pattern, re.IGNORECASE))

    def add_keyword(self, keyword: str, severity: str = "high"):
        """
        Add a keyword for fast detection.

        Args:
            keyword: Keyword to add
            severity: Severity level (critical, high)
        """
        keyword_lower = keyword.lower()
        if severity == "critical":
            self._critical_keywords.add(keyword_lower)
        else:
            self._high_keywords.add(keyword_lower)

    def get_metrics(self) -> Dict[str, Any]:
        """Get detector metrics."""
        return {
            "critical_patterns": len(self._compiled_critical),
            "high_risk_patterns": len(self._compiled_high),
            "medium_risk_patterns": len(self._compiled_medium),
            "pii_patterns": len(self._compiled_pii),
            "custom_categories": len(self._compiled_custom),
            "critical_keywords": len(self._critical_keywords),
            "high_keywords": len(self._high_keywords),
        }
