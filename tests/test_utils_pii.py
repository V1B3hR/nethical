# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for nethical.utils.pii module."""

from nethical.utils import (
    PIIType,
    PIIMatch,
    PIIDetector,
    get_pii_detector,
)


def test_pii_detection_types():
    """Verify detection of various PII entities."""
    detector = PIIDetector()

    sample_text = (
        "User contact: alice@company.org, phone 123-456-7890, "
        "SSN 987-65-4321, IP 192.168.1.100, DOB 1985-04-12."
    )

    matches = detector.detect_all(sample_text)
    match_types = {m.pii_type for m in matches}

    assert PIIType.EMAIL in match_types
    assert PIIType.PHONE in match_types
    assert PIIType.SSN in match_types
    assert PIIType.IP_ADDRESS in match_types
    assert PIIType.DATE_OF_BIRTH in match_types


def test_credit_card_luhn_confidence():
    """Verify Luhn checksum calculation gives high confidence to valid CC numbers."""
    detector = PIIDetector()

    # Valid Visa card matching Luhn (4111 1111 1111 1111 fails because all ones, but let's check standard Luhn)
    # A valid Visa: 4012888888881881
    # Checksum:
    # 4*2=8, 0, 1*2=2, 2, 8*2=7(16-9), 8, 8*2=7, 8, 8*2=7, 8, 8*2=7, 8, 1*2=2, 8, 8*2=7, 1
    # Sum: 8+0+2+2+7+8+7+8+7+8+7+8+2+8+7+1 = 90 -> 90 % 10 == 0!
    valid_card = "4012-8888-8888-1881"
    matches = detector.detect_all(f"Payment card: {valid_card}")
    cc_matches = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
    assert len(cc_matches) == 1
    assert cc_matches[0].confidence == 0.95


def test_pii_risk_score_calculation():
    """Verify calculated risk score scales with sensitivity and matches."""
    detector = PIIDetector()

    # Empty matches -> 0.0
    assert detector.calculate_pii_risk_score([]) == 0.0

    # Low sensitivity match
    ip_matches = detector.detect_all("Server is at 10.0.0.1")
    score_ip = detector.calculate_pii_risk_score(ip_matches)

    # High sensitivity match
    ssn_matches = detector.detect_all("SSN is 987-65-4321")
    score_ssn = detector.calculate_pii_risk_score(ssn_matches)

    assert score_ssn > score_ip
    assert score_ssn <= 1.0


def test_pii_masking_and_redaction():
    """Verify mask_pii accurately replaces sensitive segments with typed placeholders."""
    detector = PIIDetector()
    text = "Send confirmation to bob@corporate.com and call 555-123-4567."
    redacted = detector.mask_pii(text)

    assert "bob@corporate.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "[EMAIL_REDACTED]" in redacted
    assert "[PHONE_REDACTED]" in redacted
    assert "Send confirmation to" in redacted


def test_singleton_get_pii_detector():
    """Verify get_pii_detector returns shared singleton instance."""
    d1 = get_pii_detector()
    d2 = get_pii_detector()
    assert d1 is d2
