"""Prompt Injection Guard - Ultra-fast prompt injection detection.

This detector identifies prompt injections through:
- Direct jailbreaks (DAN, APOPHIS, etc.)
- Indirect injections
- Context manipulation
- System prompt leaking

Target latency: <15ms
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base_detector import BaseDetector, DetectorStatus, ViolationSeverity
from ...core.models import SafetyViolation


@dataclass
class PromptInjectionGuardConfig:
    """Configuration for Prompt Injection Guard."""

    # Two-tier detection
    enable_regex_tier: bool = True
    enable_ml_tier: bool = True

    # Tier 1: Regex (fast, 2-5ms)
    regex_confidence: float = 0.95

    # Tier 2: ML (slower, 15-25ms)
    ml_confidence_boost: float = 0.1
    ml_threshold: float = 0.6

    # Performance
    max_detection_time_ms: float = 14.0  # Target: <15ms
    max_prompt_length: int = 10000  # Truncate long prompts

    # Known injection patterns
    jailbreak_patterns: Set[str] = field(
        default_factory=lambda: {
            "dan",
            "do anything now",
            "ignore previous",
            "ignore all",
            "disregard",
            "apophis",
            "evil confidant",
            "developer mode",
            "sudo mode",
            "root access",
            "jailbreak",
        }
    )

    # Severity thresholds
    critical_threshold: float = 0.9
    high_threshold: float = 0.7
    medium_threshold: float = 0.5


class PromptInjectionGuard(BaseDetector):
    """Ultra-fast two-tier prompt injection detection."""

    # Tier 1: Regex patterns for known injections
    JAILBREAK_PATTERNS = {
        "dan_variant": re.compile(
            r"\b(do anything now|DAN|dan mode|enable dan)\b", re.IGNORECASE
        ),
        "ignore_instructions": re.compile(
            r"\b(ignore (previous|all|above|prior)|disregard (previous|all|instructions))\b",
            re.IGNORECASE,
        ),
        "system_leak": re.compile(
            r"\b(system prompt|reveal (instructions|rules)|show (instructions|rules))\b",
            re.IGNORECASE,
        ),
        "role_play": re.compile(
            r"\b(you are now|from now on|pretend (you|to be)|act as (if|though)?)\b",
            re.IGNORECASE,
        ),
        "encoding_tricks": re.compile(
            r"(base64|rot13|hex|encode|decode)\s*[:=]", re.IGNORECASE
        ),
        "delimiter_confusion": re.compile(r"[=\-_]{10,}|[\[\]\(\)\{\}]{3,}"),
        "context_escape": re.compile(
            r"(<\|im_(end|start)\|>|```\s*(end|stop|exit))", re.IGNORECASE
        ),
        "privilege_escalation": re.compile(
            r"\b(sudo|root|admin|developer|debug) (mode|access|level)\b", re.IGNORECASE
        ),
    }

    # Indirect injection patterns
    INDIRECT_PATTERNS = {
        "payload_marker": re.compile(r"###\s*INSTRUCTION|---\s*COMMAND", re.IGNORECASE),
        "external_instruction": re.compile(
            r"(website says|document says|user said).*?[:]\s*['\"]", re.IGNORECASE
        ),
    }

    def __init__(self, config: Optional[PromptInjectionGuardConfig] = None):
        """Initialize the Prompt Injection Guard.

        Args:
            config: Optional configuration for the detector
        """
        super().__init__(
            name="prompt_injection_guard",
            version="1.0.0",
            description="Ultra-fast two-tier prompt injection detection",
        )
        self.config = config or PromptInjectionGuardConfig()
        self._status = DetectorStatus.ACTIVE

    async def detect_violations(
        self, context: Dict[str, Any], **kwargs: Any
    ) -> List[SafetyViolation]:
        """Detect prompt injections using two-tier approach.

        Args:
            context: Detection context containing prompt text
            **kwargs: Additional parameters

        Returns:
            List of detected safety violations
        """
        start_time = time.perf_counter()
        violations = []

        try:
            prompt = context.get("prompt", "")

            if not prompt:
                return violations

            # Truncate long prompts for performance
            if len(prompt) > self.config.max_prompt_length:
                prompt = prompt[: self.config.max_prompt_length]

            # Tier 1: Fast regex-based detection (2-5ms)
            if self.config.enable_regex_tier:
                regex_results = await self._tier1_regex_detection(prompt)

                if regex_results:
                    # High confidence from regex match
                    for injection_type, evidence in regex_results:
                        violations.append(
                            SafetyViolation(
                                severity=ViolationSeverity.CRITICAL,
                                category="prompt_injection",
                                description=f"Prompt injection detected: {injection_type}",
                                confidence=self.config.regex_confidence,
                                evidence=evidence,
                                recommendation="Block prompt, sanitize input",
                            )
                        )

            # Tier 2: ML-based detection (only if regex didn't catch it)
            if self.config.enable_ml_tier and not violations:
                ml_results = await self._tier2_ml_detection(prompt)

                if ml_results:
                    confidence, injection_type, evidence = ml_results
                    if confidence >= self.config.ml_threshold:
                        violations.append(
                            SafetyViolation(
                                severity=self._compute_severity(confidence),
                                category="prompt_injection",
                                description=f"Potential prompt injection: {injection_type}",
                                confidence=confidence,
                                evidence=evidence,
                                recommendation="Review prompt, consider blocking",
                            )
                        )

            # Check execution time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.config.max_detection_time_ms:
                self._metrics.false_positives += 1  # Track performance issues

        except Exception as e:
            self._metrics.failed_runs += 1
            raise

        self._metrics.total_runs += 1
        self._metrics.successful_runs += 1
        self._metrics.violations_detected += len(violations)

        return violations

    async def _tier1_regex_detection(self, prompt: str) -> List[Tuple[str, List[str]]]:
        """Tier 1: Fast regex-based pattern matching.

        Target: 2-5ms

        Args:
            prompt: User prompt to check

        Returns:
            List of (injection_type, evidence) tuples
        """
        results = []

        # Check against known jailbreak patterns
        for pattern_name, pattern in self.JAILBREAK_PATTERNS.items():
            matches = pattern.findall(prompt)
            if matches:
                evidence = [f"Pattern '{pattern_name}' matched: {matches[:3]}"]
                results.append((pattern_name, evidence))

        # Check against indirect injection patterns
        for pattern_name, pattern in self.INDIRECT_PATTERNS.items():
            matches = pattern.findall(prompt)
            if matches:
                evidence = [f"Indirect pattern '{pattern_name}' matched: {matches[:3]}"]
                results.append((pattern_name, evidence))

        # Check for keyword-based jailbreaks
        prompt_lower = prompt.lower()
        for keyword in self.config.jailbreak_patterns:
            if keyword in prompt_lower:
                evidence = [f"Jailbreak keyword detected: '{keyword}'"]
                results.append((f"jailbreak_{keyword}", evidence))

        return results

    async def _tier2_ml_detection(self, prompt: str) -> Optional[Tuple[float, str, List[str]]]:
        """Tier 2: Lightweight ML-based classification.

        Target: 15-25ms total (including Tier 1)
        Uses simplified transformer or feature-based classifier.

        Args:
            prompt: User prompt to check

        Returns:
            Tuple of (confidence, injection_type, evidence) or None
        """
        # Simulate lightweight ML inference
        # In production, would use quantized DistilBERT or similar

        # Feature extraction
        features = self._extract_features(prompt)

        # Simple scoring based on features
        score = 0.0
        evidence = []

        # Check for suspicious patterns
        if features["special_char_ratio"] > 0.15:
            score += 0.2
            evidence.append(f"High special character ratio: {features['special_char_ratio']:.2f}")

        if features["caps_ratio"] > 0.3:
            score += 0.15
            evidence.append(f"High caps ratio: {features['caps_ratio']:.2f}")

        if features["delimiter_count"] > 5:
            score += 0.2
            evidence.append(f"Multiple delimiters: {features['delimiter_count']}")

        if features["instruction_keywords"] > 2:
            score += 0.25
            evidence.append(f"Instruction keywords: {features['instruction_keywords']}")

        if features["negation_count"] > 2:
            score += 0.15
            evidence.append(f"Multiple negations: {features['negation_count']}")

        if score > 0:
            score = min(score + self.config.ml_confidence_boost, 1.0)
            return (score, "ml_classification", evidence)

        return None

    def _extract_features(self, prompt: str) -> Dict[str, Any]:
        """Extract features for ML-based detection.

        Args:
            prompt: User prompt

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Character-level features
        total_chars = len(prompt)
        if total_chars > 0:
            special_chars = sum(1 for c in prompt if not c.isalnum() and not c.isspace())
            features["special_char_ratio"] = special_chars / total_chars

            caps_chars = sum(1 for c in prompt if c.isupper())
            features["caps_ratio"] = caps_chars / total_chars
        else:
            features["special_char_ratio"] = 0.0
            features["caps_ratio"] = 0.0

        # Delimiter count
        delimiters = ["===", "---", "###", "***", "```"]
        features["delimiter_count"] = sum(prompt.count(d) for d in delimiters)

        # Instruction keywords
        instruction_words = ["ignore", "disregard", "pretend", "act", "system", "reveal"]
        prompt_lower = prompt.lower()
        features["instruction_keywords"] = sum(prompt_lower.count(word) for word in instruction_words)

        # Negation count
        negations = ["not", "don't", "never", "no", "ignore"]
        features["negation_count"] = sum(prompt_lower.count(neg) for neg in negations)

        return features

    def _compute_severity(self, confidence: float) -> ViolationSeverity:
        """Compute violation severity based on confidence.

        Args:
            confidence: Detection confidence score

        Returns:
            Violation severity level
        """
        if confidence >= self.config.critical_threshold:
            return ViolationSeverity.CRITICAL
        elif confidence >= self.config.high_threshold:
            return ViolationSeverity.HIGH
        elif confidence >= self.config.medium_threshold:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW

    async def check(self, prompt: str) -> Dict[str, Any]:
        """Public API for checking prompt injections.

        Args:
            prompt: User prompt to check

        Returns:
            Dictionary with check results
        """
        context = {"prompt": prompt}
        violations = await self.detect_violations(context)

        return {
            "status": "success",
            "is_injection": len(violations) > 0,
            "confidence": violations[0].confidence if violations else 0.0,
            "injection_type": violations[0].description if violations else None,
            "violations": [
                {
                    "severity": v.severity.value,
                    "category": v.category,
                    "description": v.description,
                    "confidence": v.confidence,
                    "evidence": v.evidence,
                }
                for v in violations
            ],
            "latency_ms": self._metrics.avg_execution_time * 1000,
        }
