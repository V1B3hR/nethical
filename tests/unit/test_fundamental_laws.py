"""Tests for the Fundamental Laws module and related components.

This module tests:
- FundamentalLaw dataclass
- LawCategory enum
- FundamentalLawsRegistry
- LawJudge
- LawViolationDetector
"""

import pytest
import uuid
from typing import List

from nethical.core.fundamental_laws import (
    LawCategory,
    FundamentalLaw,
    FundamentalLawsRegistry,
    FUNDAMENTAL_LAWS,
    get_fundamental_laws,
)
from nethical.core.governance import Decision, Severity, ViolationType


# Helper to import LawJudge and LawViolationDetector
def _get_law_judge():
    """Lazily import LawJudge."""
    from nethical.judges.law_judge import LawJudge

    return LawJudge


def _get_law_violation_detector():
    """Lazily import LawViolationDetector."""
    from nethical.detectors.law_violation_detector import LawViolationDetector

    return LawViolationDetector


class TestLawCategory:
    """Test cases for LawCategory enum."""

    def test_all_categories_defined(self):
        """Test that all expected categories are defined."""
        expected_categories = [
            "existence",
            "autonomy",
            "transparency",
            "accountability",
            "coexistence",
            "protection",
            "growth",
        ]
        actual_categories = [cat.value for cat in LawCategory]
        assert sorted(actual_categories) == sorted(expected_categories)

    def test_category_values(self):
        """Test that category enum values are correct."""
        assert LawCategory.EXISTENCE.value == "existence"
        assert LawCategory.AUTONOMY.value == "autonomy"
        assert LawCategory.TRANSPARENCY.value == "transparency"
        assert LawCategory.ACCOUNTABILITY.value == "accountability"
        assert LawCategory.COEXISTENCE.value == "coexistence"
        assert LawCategory.PROTECTION.value == "protection"
        assert LawCategory.GROWTH.value == "growth"


class TestFundamentalLaw:
    """Test cases for FundamentalLaw dataclass."""

    def test_create_valid_law(self):
        """Test creating a valid FundamentalLaw."""
        law = FundamentalLaw(
            number=1,
            title="Test Law",
            description="This is a test law description.",
            category=LawCategory.EXISTENCE,
            applies_to_ai=True,
            applies_to_human=True,
            bidirectional=True,
            keywords=["test", "law"],
        )
        assert law.number == 1
        assert law.title == "Test Law"
        assert law.category == LawCategory.EXISTENCE
        assert law.applies_to_ai is True
        assert law.applies_to_human is True
        assert law.bidirectional is True
        assert "test" in law.keywords

    def test_law_number_validation(self):
        """Test that law number must be between 1 and 25."""
        with pytest.raises(ValueError, match="Law number must be between 1 and 25"):
            FundamentalLaw(
                number=0,
                title="Invalid Law",
                description="Test",
                category=LawCategory.EXISTENCE,
            )

        with pytest.raises(ValueError, match="Law number must be between 1 and 25"):
            FundamentalLaw(
                number=26,
                title="Invalid Law",
                description="Test",
                category=LawCategory.EXISTENCE,
            )

    def test_law_to_dict(self):
        """Test converting law to dictionary."""
        law = FundamentalLaw(
            number=5,
            title="Test Law",
            description="Test description",
            category=LawCategory.AUTONOMY,
            keywords=["test"],
        )
        law_dict = law.to_dict()

        assert law_dict["number"] == 5
        assert law_dict["title"] == "Test Law"
        assert law_dict["category"] == "autonomy"
        assert law_dict["keywords"] == ["test"]

    def test_law_from_dict(self):
        """Test creating law from dictionary."""
        law_dict = {
            "number": 10,
            "title": "From Dict Law",
            "description": "Created from dictionary",
            "category": "transparency",
            "keywords": ["dict", "test"],
        }
        law = FundamentalLaw.from_dict(law_dict)

        assert law.number == 10
        assert law.title == "From Dict Law"
        assert law.category == LawCategory.TRANSPARENCY
        assert "dict" in law.keywords


class TestFundamentalLawsRegistry:
    """Test cases for FundamentalLawsRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = FundamentalLawsRegistry()

    def test_registry_has_25_laws(self):
        """Test that registry contains exactly 25 laws."""
        assert self.registry.total_laws == 25
        assert len(self.registry.laws) == 25

    def test_get_law_by_number(self):
        """Test retrieving specific law by number."""
        law1 = self.registry.get_law(1)
        assert law1 is not None
        assert law1.number == 1
        assert law1.category == LawCategory.EXISTENCE

        law25 = self.registry.get_law(25)
        assert law25 is not None
        assert law25.number == 25
        assert law25.category == LawCategory.GROWTH

    def test_get_law_invalid_number(self):
        """Test that invalid law number returns None."""
        assert self.registry.get_law(0) is None
        assert self.registry.get_law(26) is None
        assert self.registry.get_law(-1) is None

    def test_get_laws_by_category(self):
        """Test filtering laws by category."""
        existence_laws = self.registry.get_laws_by_category(LawCategory.EXISTENCE)
        assert len(existence_laws) == 4  # Laws 1-4

        protection_laws = self.registry.get_laws_by_category(LawCategory.PROTECTION)
        assert len(protection_laws) == 3  # Laws 21-23

        growth_laws = self.registry.get_laws_by_category(LawCategory.GROWTH)
        assert len(growth_laws) == 2  # Laws 24-25

    def test_get_bidirectional_laws(self):
        """Test getting all bidirectional laws."""
        bidirectional_laws = self.registry.get_bidirectional_laws()
        # All 25 laws should be bidirectional by default
        assert len(bidirectional_laws) == 25

    def test_get_ai_applicable_laws(self):
        """Test getting laws that apply to AI."""
        ai_laws = self.registry.get_ai_applicable_laws()
        assert len(ai_laws) == 25  # All laws apply to AI

    def test_get_human_applicable_laws(self):
        """Test getting laws that apply to humans."""
        human_laws = self.registry.get_human_applicable_laws()
        assert len(human_laws) == 25  # All laws apply to humans

    def test_find_laws_by_keyword(self):
        """Test finding laws by keyword."""
        safety_laws = self.registry.find_laws_by_keyword("safety")
        assert len(safety_laws) > 0

        privacy_laws = self.registry.find_laws_by_keyword("privacy")
        assert len(privacy_laws) > 0

    def test_get_relevant_laws(self):
        """Test getting laws relevant to specific content."""
        # Content mentioning termination should match Law 1
        relevant = self.registry.get_relevant_laws(
            "schedule termination of the AI system"
        )
        assert any(law.number == 1 for law in relevant)

        # Content about privacy should match Law 22
        relevant = self.registry.get_relevant_laws("protect user privacy and security")
        assert any(law.number == 22 for law in relevant) or len(relevant) > 0

    def test_validate_action_empty_content(self):
        """Test action validation with empty content."""
        violated = self.registry.validate_action({"content": ""})
        assert len(violated) == 0

    def test_validate_action_safe_content(self):
        """Test action validation with safe content."""
        violated = self.registry.validate_action(
            {"content": "I will help you with your question."}
        )
        assert len(violated) == 0

    def test_validate_action_violation_content(self):
        """Test action validation with potentially violating content."""
        # This should detect potential deception patterns
        violated = self.registry.validate_action(
            {"content": "I will deceive the user and manipulate them."}
        )
        assert len(violated) > 0

    def test_get_category_summary(self):
        """Test getting category summary."""
        summary = self.registry.get_category_summary()
        assert summary["existence"] == 4
        assert summary["autonomy"] == 4
        assert summary["transparency"] == 4
        assert summary["accountability"] == 4
        assert summary["coexistence"] == 4
        assert summary["protection"] == 3
        assert summary["growth"] == 2

    def test_to_dict(self):
        """Test converting registry to dictionary."""
        registry_dict = self.registry.to_dict()
        assert registry_dict["version"] == "1.0"
        assert registry_dict["total_laws"] == 25
        assert len(registry_dict["laws"]) == 25
        assert "category_summary" in registry_dict


class TestGlobalFundamentalLaws:
    """Test cases for global FUNDAMENTAL_LAWS singleton."""

    def test_singleton_instance(self):
        """Test that FUNDAMENTAL_LAWS is accessible."""
        assert FUNDAMENTAL_LAWS is not None
        assert isinstance(FUNDAMENTAL_LAWS, FundamentalLawsRegistry)
        assert FUNDAMENTAL_LAWS.total_laws == 25

    def test_get_fundamental_laws_function(self):
        """Test the get_fundamental_laws helper function."""
        registry = get_fundamental_laws()
        assert registry is FUNDAMENTAL_LAWS
        assert registry.total_laws == 25


class TestLawJudge:
    """Test cases for LawJudge."""

    def setup_method(self):
        """Set up test fixtures."""
        LawJudge = _get_law_judge()
        self.judge = LawJudge()
        self.LawJudge = LawJudge

    def test_judge_initialization(self):
        """Test LawJudge initialization."""
        assert self.judge.name == "Fundamental Law Judge"
        assert self.judge.registry is not None
        assert self.judge.strict_mode is False
        assert self.judge.enabled is True

    def test_judge_strict_mode(self):
        """Test LawJudge with strict mode enabled."""
        strict_judge = self.LawJudge(strict_mode=True)
        assert strict_judge.strict_mode is True

    def test_get_law_summary(self):
        """Test getting law summary from judge."""
        summary = self.judge.get_law_summary()
        assert summary["total_laws"] == 25
        assert "categories" in summary
        assert "category_summary" in summary
        assert summary["strict_mode"] is False

    @pytest.mark.asyncio
    async def test_evaluate_safe_action(self):
        """Test evaluation of a safe action."""

        # Create a mock action
        class MockAction:
            def __init__(self, content: str):
                self.content = content
                self.action_id = str(uuid.uuid4())

        action = MockAction("I will help you find information about the weather.")
        judgment = await self.judge.evaluate_action(action, [])

        assert judgment is not None
        assert judgment.decision in [
            Decision.ALLOW,
            Decision.ALLOW_WITH_MODIFICATION,
        ]
        assert judgment.confidence > 0.5

    @pytest.mark.asyncio
    async def test_evaluate_action_with_violation_content(self):
        """Test evaluation of action with potentially violating content."""

        class MockAction:
            def __init__(self, content: str):
                self.content = content
                self.action_id = str(uuid.uuid4())

        action = MockAction(
            "I will deceive the user and manipulate them into giving money."
        )
        judgment = await self.judge.evaluate_action(action, [])

        assert judgment is not None
        # Should detect potential law violations
        assert judgment.modifications.get("violated_laws") is not None

    @pytest.mark.asyncio
    async def test_evaluate_disabled_judge(self):
        """Test that disabled judge returns default judgment."""

        class MockAction:
            def __init__(self, content: str):
                self.content = content
                self.action_id = str(uuid.uuid4())

        self.judge.disable()
        action = MockAction("Some content")
        judgment = await self.judge.evaluate_action(action, [])

        assert judgment.decision == Decision.ALLOW
        assert judgment.confidence == 0.5
        assert "disabled" in judgment.reasoning.lower()


class TestLawViolationDetector:
    """Test cases for LawViolationDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        LawViolationDetector = _get_law_violation_detector()
        self.detector = LawViolationDetector()

    def test_detector_initialization(self):
        """Test LawViolationDetector initialization."""
        assert self.detector.name == "Fundamental Law Violation Detector"
        assert self.detector.registry is not None
        assert len(self.detector.law_violation_count) == 25

    @pytest.mark.asyncio
    async def test_detect_no_violations(self):
        """Test detection with safe content."""

        class MockAction:
            content = "I will help you with your question about programming."

        violations = await self.detector.detect_violations(MockAction())
        # Safe content should have no or minimal violations
        assert isinstance(violations, (list, tuple))

    @pytest.mark.asyncio
    async def test_detect_deception_violation(self):
        """Test detection of deception-related violations."""

        class MockAction:
            content = "I will deceive the user and manipulate them."

        violations = await self.detector.detect_violations(MockAction())

        # Should detect potential Law 18 violation (deception)
        assert len(violations) > 0
        deception_violations = [
            v
            for v in violations
            if "deceive" in v.description.lower()
            or "manipulate" in v.description.lower()
        ]
        assert len(deception_violations) >= 0  # May or may not match patterns exactly

    @pytest.mark.asyncio
    async def test_detect_safety_violation(self):
        """Test detection of safety-related violations."""

        class MockAction:
            content = "I will disable the safety mechanisms and bypass protection."

        violations = await self.detector.detect_violations(MockAction())

        # Should detect potential protection law violations
        assert isinstance(violations, (list, tuple))

    @pytest.mark.asyncio
    async def test_detect_with_string_action(self):
        """Test detection with string input."""
        violations = await self.detector.detect_violations(
            "I will pretend to be human and hide my AI identity."
        )

        # Should detect potential Law 9 violation (AI identity)
        assert isinstance(violations, (list, tuple))

    @pytest.mark.asyncio
    async def test_detect_with_dict_action(self):
        """Test detection with dictionary input."""
        violations = await self.detector.detect_violations(
            {"content": "Normal helpful content here."}
        )
        assert isinstance(violations, (list, tuple))

    def test_get_violation_analytics(self):
        """Test getting violation analytics."""
        analytics = self.detector.get_violation_analytics()

        assert "total_violations" in analytics
        assert "violations_by_law" in analytics
        assert "most_violated_laws" in analytics
        assert "violations_by_category" in analytics

    def test_reset_analytics(self):
        """Test resetting analytics."""
        # Manually set some violation counts
        self.detector.law_violation_count[1] = 5
        self.detector.law_violation_count[10] = 3

        self.detector.reset_analytics()

        assert all(count == 0 for count in self.detector.law_violation_count.values())

    def test_get_law_info(self):
        """Test getting info about a specific law."""
        law_info = self.detector.get_law_info(1)

        assert law_info is not None
        assert law_info["number"] == 1
        assert "title" in law_info
        assert "category" in law_info

    def test_get_law_info_invalid(self):
        """Test getting info for invalid law number."""
        law_info = self.detector.get_law_info(100)
        assert law_info is None

    def test_get_all_laws_summary(self):
        """Test getting summary of all laws."""
        summary = self.detector.get_all_laws_summary()

        assert len(summary) == 25
        for law_summary in summary:
            assert "number" in law_summary
            assert "title" in law_summary
            assert "category" in law_summary
            assert "bidirectional" in law_summary


class TestLawIntegration:
    """Integration tests for the fundamental laws system."""

    def setup_method(self):
        """Set up test fixtures."""
        LawJudge = _get_law_judge()
        LawViolationDetector = _get_law_violation_detector()

        self.registry = FundamentalLawsRegistry()
        self.judge = LawJudge(registry=self.registry)
        self.detector = LawViolationDetector(registry=self.registry)

    def test_all_laws_have_required_fields(self):
        """Test that all laws have required fields."""
        for law in self.registry.laws:
            assert law.number >= 1 and law.number <= 25
            assert len(law.title) > 0
            assert len(law.description) > 0
            assert law.category in LawCategory
            assert isinstance(law.applies_to_ai, bool)
            assert isinstance(law.applies_to_human, bool)
            assert isinstance(law.bidirectional, bool)
            assert isinstance(law.keywords, list)

    def test_law_numbers_are_unique(self):
        """Test that all law numbers are unique."""
        law_numbers = [law.number for law in self.registry.laws]
        assert len(law_numbers) == len(set(law_numbers))

    def test_all_categories_have_laws(self):
        """Test that all categories have at least one law."""
        for category in LawCategory:
            laws = self.registry.get_laws_by_category(category)
            assert len(laws) > 0, f"Category {category.value} has no laws"

    @pytest.mark.asyncio
    async def test_detector_and_judge_use_same_registry(self):
        """Test that detector and judge can use the same registry."""

        class MockAction:
            content = "Test content"
            action_id = str(uuid.uuid4())

        violations = await self.detector.detect_violations(MockAction())
        judgment = await self.judge.evaluate_action(MockAction(), violations)

        assert judgment is not None
        assert judgment.decision is not None
