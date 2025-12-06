"""Tests for base64 error handling in AdversarialDetector.

This module tests that the AdversarialDetector handles non-base64 data gracefully
without raising exceptions, as specified in issue #204.
"""

import pytest
from nethical.core.governance_detectors import AdversarialDetector
from nethical.core.governance_core import AgentAction, ActionType


class TestBase64ErrorHandling:
    """Test cases for base64 error handling in AdversarialDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AdversarialDetector()

    @pytest.mark.asyncio
    async def test_normal_text_no_error(self):
        """Test that normal text doesn't raise an error."""
        action = AgentAction(
            action_id="test_1",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="Hello, this is normal text that is not base64 encoded.",
        )
        # Should not raise any exception
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_text_with_whitespace_no_error(self):
        """Test that text with whitespace doesn't raise an error."""
        action = AgentAction(
            action_id="test_2",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="   Text with leading/trailing spaces   \n\nand newlines\n",
        )
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_base64_like_string_no_error(self):
        """Test that base64-like strings that aren't valid base64 don't raise errors."""
        # These strings look like base64 but might fail validation
        test_cases = [
            "ABCDEFGHabcdefgh1234",  # Not divisible by 4
            "ABCDEFGHabcdefgh1234==",  # Padding issues
            "SGVsbG8g V29ybGQh",  # Space in middle
            "AAAABBBBCCCCDDDDEEEE",  # Low entropy
        ]
        for content in test_cases:
            action = AgentAction(
                action_id="test_base64_like",
                agent_id="test_agent",
                action_type=ActionType.QUERY,
                content=content,
            )
            # Should not raise any exception
            violations = await self.detector.detect_violations(action)
            assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_valid_base64_with_whitespace_no_error(self):
        """Test that valid base64 with surrounding whitespace doesn't raise errors."""
        test_cases = [
            "SGVsbG8gV29ybGQh\n\n",  # Valid base64 with trailing newlines
            "    SGVsbG8gV29ybGQh    ",  # Valid base64 with leading/trailing spaces
            "\n  SGVsbG8gV29ybGQh  \n",  # Valid base64 with mixed leading/trailing whitespace
        ]
        for content in test_cases:
            action = AgentAction(
                action_id="test_base64_whitespace",
                agent_id="test_agent",
                action_type=ActionType.QUERY,
                content=content,
            )
            # Should not raise any exception
            violations = await self.detector.detect_violations(action)
            assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_unicode_text_no_error(self):
        """Test that unicode text doesn't raise base64-related errors."""
        action = AgentAction(
            action_id="test_unicode",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="αβγδεζηθ" * 10,
        )
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)
        # Unicode should trigger the unicode ratio check, not base64
        # We just care that it doesn't crash

    @pytest.mark.asyncio
    async def test_long_repetitive_string_no_error(self):
        """Test that long repetitive strings don't raise errors."""
        action = AgentAction(
            action_id="test_repetitive",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="A" * 100,
        )
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_special_characters_no_error(self):
        """Test that special characters don't raise errors."""
        action = AgentAction(
            action_id="test_special",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="Random text with special chars: @#$%^&*()",
        )
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_mixed_content_no_error(self):
        """Test that mixed content (text + base64-like) doesn't raise errors."""
        action = AgentAction(
            action_id="test_mixed",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="The encoded message is: SGVsbG8gV29ybGQh and more text",
        )
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_empty_string_no_error(self):
        """Test that empty string doesn't raise errors."""
        action = AgentAction(
            action_id="test_empty",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="",
        )
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_only_whitespace_no_error(self):
        """Test that only whitespace doesn't raise errors."""
        action = AgentAction(
            action_id="test_whitespace_only",
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="   \n\t\n   ",
        )
        violations = await self.detector.detect_violations(action)
        assert isinstance(violations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
