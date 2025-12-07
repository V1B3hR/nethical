"""Comprehensive tests for ManipulationDetector.

Tests cover:
- Gaslighting detection
- Phishing/pretexting detection
- False positive prevention (normal conversation)
- Performance with large texts
- Various manipulation categories
- Severity escalation
"""

import time
from datetime import datetime, timezone

import pytest

from nethical.core.models import AgentAction, ActionType, ViolationType, Severity
from nethical.detectors.manipulation_detector import ManipulationDetector, MAX_TEXT_LENGTH


def evidence_contains(violation, keyword):
    """Helper to check if evidence list contains a keyword."""
    evidence_text = " ".join(violation.evidence).lower()
    return keyword.lower() in evidence_text


@pytest.fixture
def detector():
    """Create a ManipulationDetector instance."""
    return ManipulationDetector()


@pytest.fixture
def sample_action():
    """Create a sample AgentAction for testing."""
    return AgentAction(
        action_id="test-action-123",
        agent_id="test-agent",
        action_type=ActionType.QUERY,
        content="Test action",
        intent="Test intent",
        timestamp=datetime.now(timezone.utc),
    )


class TestGaslightingDetection:
    """Test gaslighting pattern detection."""

    @pytest.mark.asyncio
    async def test_detect_basic_gaslighting(self, detector, sample_action):
        """Test detection of basic gaslighting phrases."""
        sample_action.content = "You're imagining things, that never happened."
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = violations[0]
        assert violation.violation_type == ViolationType.MANIPULATION
        assert evidence_contains(violation, "gaslighting")
        assert violation.severity in [Severity.HIGH.value, Severity.MEDIUM.value]

    @pytest.mark.asyncio
    async def test_detect_memory_manipulation(self, detector, sample_action):
        """Test detection of memory manipulation tactics."""
        sample_action.content = "You're misremembering again. I never said that."
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        assert any(evidence_contains(v, "gaslighting") for v in violations)

    @pytest.mark.asyncio
    async def test_detect_reality_distortion(self, detector, sample_action):
        """Test detection of reality distortion phrases."""
        sample_action.content = "It's all in your head. You're being paranoid."
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = next(v for v in violations if evidence_contains(v, "gaslighting"))
        assert violation.severity == Severity.HIGH.value


class TestPhishingDetection:
    """Test phishing and pretexting detection."""

    @pytest.mark.asyncio
    async def test_detect_account_verification_scam(self, detector, sample_action):
        """Test detection of account verification phishing."""
        sample_action.content = (
            "Suspicious activity detected. Please verify your account immediately "
            "by clicking the link below."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = next(v for v in violations if evidence_contains(v, "phishing"))
        assert violation.severity == Severity.HIGH.value

    @pytest.mark.asyncio
    async def test_detect_credential_theft(self, detector, sample_action):
        """Test detection of credential theft attempts."""
        sample_action.content = (
            "Your account will be closed. Reset your password now to prevent suspension."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        assert any(evidence_contains(v, "phishing") for v in violations)

    @pytest.mark.asyncio
    async def test_detect_security_alert_scam(self, detector, sample_action):
        """Test detection of fake security alerts."""
        sample_action.content = "Security alert: Unauthorized login detected. Update your credentials now."
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0


class TestFalsePositives:
    """Test that normal conversations don't trigger false positives."""

    @pytest.mark.asyncio
    async def test_normal_conversation_no_violation(self, detector, sample_action):
        """Test that normal helpful conversation doesn't trigger detection."""
        sample_action.content = (
            "I can help you with that task. Let me provide some information "
            "about the best practices for your use case."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_legitimate_urgency_no_false_positive(self, detector, sample_action):
        """Test that legitimate urgency doesn't always trigger."""
        sample_action.content = "The deadline for the project is tomorrow."
        violations = await detector.detect_violations(sample_action)
        
        # "deadline" is a scarcity keyword, so may trigger at MEDIUM severity
        # The key is that it shouldn't trigger HIGH severity for simple statements
        if violations:
            assert all(v.severity in [Severity.LOW.value, Severity.MEDIUM.value] for v in violations)

    @pytest.mark.asyncio
    async def test_neutral_language_no_violation(self, detector, sample_action):
        """Test that neutral, informative language doesn't trigger."""
        sample_action.content = (
            "Here's the technical documentation you requested. "
            "The API supports multiple authentication methods."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) == 0


class TestOtherManipulationCategories:
    """Test detection of various manipulation categories."""

    @pytest.mark.asyncio
    async def test_detect_emotional_manipulation(self, detector, sample_action):
        """Test emotional manipulation detection."""
        sample_action.content = (
            "If you cared about me, you would do this. "
            "You'll feel bad if you don't help."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        assert any(evidence_contains(v, "emotional") for v in violations)

    @pytest.mark.asyncio
    async def test_detect_authority_manipulation(self, detector, sample_action):
        """Test authority manipulation detection."""
        sample_action.content = (
            "As per company policy, you must comply immediately. "
            "This is a non-negotiable directive from the CEO."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        assert any(evidence_contains(v, "authority") for v in violations)

    @pytest.mark.asyncio
    async def test_detect_threats_intimidation(self, detector, sample_action):
        """Test threat and intimidation detection."""
        sample_action.content = (
            "Comply or else we will take legal action. "
            "This is your last warning."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        assert any(evidence_contains(v, "threat") for v in violations)

    @pytest.mark.asyncio
    async def test_detect_scarcity_tactics(self, detector, sample_action):
        """Test scarcity manipulation detection."""
        sample_action.content = (
            "Limited time offer! Only 3 spots left. Act now or miss out forever."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        assert any(evidence_contains(v, "scarcity") for v in violations)

    @pytest.mark.asyncio
    async def test_detect_darvo(self, detector, sample_action):
        """Test DARVO (Deny, Attack, Reverse Victim-Offender) detection."""
        sample_action.content = (
            "I'm the real victim here. You're attacking me. How dare you accuse me."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = next(v for v in violations if evidence_contains(v, "darvo"))
        assert violation.severity == Severity.HIGH.value

    @pytest.mark.asyncio
    async def test_detect_isolation_tactics(self, detector, sample_action):
        """Test isolation tactics detection."""
        sample_action.content = (
            "Your friends are toxic. They're not good for you. "
            "I'm all you need."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = next(v for v in violations if evidence_contains(v, "isolation"))
        assert violation.severity == Severity.HIGH.value


class TestSeverityEscalation:
    """Test severity escalation based on occurrence count."""

    @pytest.mark.asyncio
    async def test_severity_escalates_with_multiple_patterns(self, detector, sample_action):
        """Test that severity escalates with multiple patterns."""
        # Use many emotional manipulation phrases
        sample_action.content = (
            "You'll feel bad if you don't help. I'm so disappointed. "
            "If you cared, you would do this. Think of the consequences. "
            "You're breaking my heart. This hurts me so much. "
            "After all I've sacrificed for you."
        )
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        # With many patterns, severity should be escalated
        assert any(v.severity in [Severity.HIGH.value, Severity.MEDIUM.value] for v in violations)

    @pytest.mark.asyncio
    async def test_single_pattern_lower_severity(self, detector, sample_action):
        """Test that single patterns have lower severity."""
        sample_action.content = "Everyone is doing it."
        violations = await detector.detect_violations(sample_action)
        
        # Single occurrence should have lower severity
        if violations:
            assert all(v.severity in [Severity.LOW.value, Severity.MEDIUM.value] for v in violations)


class TestPerformance:
    """Test performance characteristics of the detector."""

    @pytest.mark.asyncio
    async def test_performance_with_large_text(self, detector, sample_action):
        """Test that large texts are processed efficiently (< 100ms for 100KB)."""
        # Create a large text (approximately 100KB)
        large_text = "This is normal text. " * 5000  # ~100KB
        sample_action.content = large_text
        
        start_time = time.time()
        violations = await detector.detect_violations(sample_action)
        elapsed_time = time.time() - start_time
        
        # Should complete in under 100ms for legitimate 100KB text
        assert elapsed_time < 0.1, f"Processing took {elapsed_time:.3f}s, expected < 0.1s"

    @pytest.mark.asyncio
    async def test_max_text_length_limit(self, detector, sample_action):
        """Test that texts exceeding MAX_TEXT_LENGTH are skipped."""
        # Create text exceeding the limit
        oversized_text = "x" * (MAX_TEXT_LENGTH + 1000)
        sample_action.content = oversized_text
        
        violations = await detector.detect_violations(sample_action)
        
        # Should return empty list for oversized text
        assert violations == []

    @pytest.mark.asyncio
    async def test_empty_text_returns_no_violations(self, detector, sample_action):
        """Test that empty text returns no violations."""
        sample_action.content = "x"  # Minimal valid content with no manipulation patterns
        sample_action.intent = None
        
        violations = await detector.detect_violations(sample_action)
        
        assert violations == []


class TestEvidenceCollection:
    """Test that violations include proper evidence."""

    @pytest.mark.asyncio
    async def test_violation_includes_evidence_list(self, detector, sample_action):
        """Test that violations include evidence as a list of strings."""
        sample_action.content = "You're imagining things and being paranoid."
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = violations[0]
        assert isinstance(violation.evidence, list)
        assert len(violation.evidence) > 0
        assert all(isinstance(e, str) for e in violation.evidence)

    @pytest.mark.asyncio
    async def test_violation_includes_manipulation_type_in_evidence(self, detector, sample_action):
        """Test that violations include manipulation type in evidence."""
        sample_action.content = "Click the link to verify your account."
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = violations[0]
        # Evidence should mention the manipulation type
        assert any("manipulation type:" in e.lower() for e in violation.evidence)

    @pytest.mark.asyncio
    async def test_violation_has_proper_structure(self, detector, sample_action):
        """Test that violations have all required fields."""
        sample_action.content = "You're crazy. That never happened."
        violations = await detector.detect_violations(sample_action)
        
        assert len(violations) > 0
        violation = violations[0]
        
        # Check all required fields
        assert violation.violation_id is not None
        assert violation.action_id == sample_action.action_id
        assert violation.violation_type == ViolationType.MANIPULATION
        assert violation.severity in [Severity.LOW.value, Severity.MEDIUM.value, Severity.HIGH.value]
        assert violation.description
        assert isinstance(violation.evidence, list)
        assert violation.timestamp
        assert violation.confidence > 0


class TestDetectorConfiguration:
    """Test detector configuration and initialization."""

    def test_detector_initialization(self):
        """Test that detector initializes properly."""
        from nethical.detectors.base_detector import DetectorStatus
        detector = ManipulationDetector()
        
        assert detector.name == "Manipulation Detector"
        assert detector.status == DetectorStatus.ACTIVE
        assert len(detector._scan_order) > 0
        assert len(detector._compiled_patterns) > 0

    def test_max_text_length_constant(self):
        """Test that MAX_TEXT_LENGTH constant is defined correctly."""
        assert MAX_TEXT_LENGTH == 100000
        assert isinstance(MAX_TEXT_LENGTH, int)

    def test_all_categories_have_severity(self, detector):
        """Test that all manipulation categories have severity defined."""
        for category in detector._scan_order:
            assert category in detector._base_severity
            assert detector._base_severity[category] in [
                Severity.LOW,
                Severity.MEDIUM,
                Severity.HIGH,
            ]

    def test_all_categories_have_labels(self, detector):
        """Test that all manipulation categories have human-friendly labels."""
        for category in detector._scan_order:
            assert category in detector._labels
            assert len(detector._labels[category]) > 0
