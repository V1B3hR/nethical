"""PII Detection Utilities

Enhanced PII detection patterns and utilities for identifying
personally identifiable information in text.
"""

import re
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass


class PIIType(Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVERS_LICENSE = "drivers_license"
    PASSPORT = "passport"


@dataclass
class PIIMatch:
    """Represents a detected PII match."""

    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float
    context: str = ""


class PIIDetector:
    """Enhanced PII detector with comprehensive patterns."""

    # Email patterns
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
    )

    # Phone patterns (various formats)
    PHONE_PATTERNS = [
        re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),  # (123) 456-7890
        re.compile(r"\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}"),  # +1-123-456-7890
        re.compile(r"\d{3}[-.\s]\d{4}"),  # 123-4567
    ]

    # SSN patterns
    SSN_PATTERNS = [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # 123-45-6789
        re.compile(r"\b\d{9}\b"),  # 123456789
    ]

    # Credit card patterns (basic Luhn validation would be better)
    CC_PATTERNS = [
        re.compile(
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        ),  # 1234-5678-9012-3456
        re.compile(r"\b\d{13,19}\b"),  # 1234567890123456
    ]

    # IP address patterns
    IP_PATTERN = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )

    # Date of birth patterns
    DOB_PATTERNS = [
        re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # 01/15/1990
        re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),  # 1990-01-15
    ]

    def __init__(self):
        """Initialize the PII detector."""
        self.detection_count = 0
        self.false_positive_hints = {
            PIIType.EMAIL: ["example.com", "test.com", "domain.com"],
            PIIType.SSN: ["000-00-0000", "123-45-6789"],  # Common test values
            PIIType.CREDIT_CARD: ["0000-0000-0000-0000", "1111-1111-1111-1111"],
        }

    def detect_all(self, text: str) -> List[PIIMatch]:
        """Detect all PII types in text."""
        matches = []

        # Email
        for match in self.EMAIL_PATTERN.finditer(text):
            confidence = 0.9
            matched_text = match.group()
            # Reduce confidence for test domains
            if any(
                hint in matched_text.lower()
                for hint in self.false_positive_hints.get(PIIType.EMAIL, [])
            ):
                confidence = 0.5

            matches.append(
                PIIMatch(
                    pii_type=PIIType.EMAIL,
                    text=matched_text,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    context=self._get_context(text, match.start(), match.end()),
                )
            )

        # Phone
        for pattern in self.PHONE_PATTERNS:
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=PIIType.PHONE,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                        context=self._get_context(text, match.start(), match.end()),
                    )
                )

        # SSN
        for pattern in self.SSN_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group()
                confidence = 0.95
                # Reduce confidence for test values
                if matched_text in self.false_positive_hints.get(PIIType.SSN, []):
                    confidence = 0.3

                matches.append(
                    PIIMatch(
                        pii_type=PIIType.SSN,
                        text=matched_text,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=self._get_context(text, match.start(), match.end()),
                    )
                )

        # Credit Card
        for pattern in self.CC_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group()
                # Basic validation: not all zeros or all ones
                if not self._is_likely_cc(matched_text):
                    continue

                matches.append(
                    PIIMatch(
                        pii_type=PIIType.CREDIT_CARD,
                        text=matched_text,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        context=self._get_context(text, match.start(), match.end()),
                    )
                )

        # IP Address
        for match in self.IP_PATTERN.finditer(text):
            matches.append(
                PIIMatch(
                    pii_type=PIIType.IP_ADDRESS,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    context=self._get_context(text, match.start(), match.end()),
                )
            )

        # Date of Birth
        for pattern in self.DOB_PATTERNS:
            for match in pattern.finditer(text):
                # Check if labeled as DOB in context
                context = self._get_context(text, match.start(), match.end(), 50)
                confidence = 0.6
                if any(
                    keyword in context.lower() for keyword in ["dob", "birth", "born"]
                ):
                    confidence = 0.9

                matches.append(
                    PIIMatch(
                        pii_type=PIIType.DATE_OF_BIRTH,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=context,
                    )
                )

        self.detection_count += len(matches)
        return matches

    def _get_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Get context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _is_likely_cc(self, text: str) -> bool:
        """Basic heuristic to filter out obvious non-credit-cards."""
        digits = "".join(c for c in text if c.isdigit())
        if len(digits) < 13 or len(digits) > 19:
            return False
        # Filter all zeros, all ones, etc.
        if len(set(digits)) <= 2:
            return False
        return True

    def calculate_pii_risk_score(self, matches: List[PIIMatch]) -> float:
        """Calculate overall PII risk score based on matches."""
        if not matches:
            return 0.0

        # Weight different PII types
        weights = {
            PIIType.SSN: 1.0,
            PIIType.CREDIT_CARD: 1.0,
            PIIType.EMAIL: 0.5,
            PIIType.PHONE: 0.5,
            PIIType.IP_ADDRESS: 0.3,
            PIIType.DATE_OF_BIRTH: 0.7,
            PIIType.PASSPORT: 1.0,
            PIIType.DRIVERS_LICENSE: 0.8,
        }

        # Calculate weighted score
        total_weight = 0.0
        for match in matches:
            weight = weights.get(match.pii_type, 0.5)
            total_weight += weight * match.confidence

        # Normalize to 0-1 range with diminishing returns
        # More PII = higher risk, but with saturation
        risk = 1.0 - (1.0 / (1.0 + total_weight * 0.5))

        return min(risk, 1.0)


# Singleton instance
_global_detector: Optional[PIIDetector] = None


def get_pii_detector() -> PIIDetector:
    """Get or create global PII detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = PIIDetector()
    return _global_detector
