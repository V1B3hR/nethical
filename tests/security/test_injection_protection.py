"""Security tests for injection protection in ManipulationDetector.

These tests verify that:
1. Regex patterns don't accept dynamic user input
2. All patterns are hardcoded and safe
3. No code injection vulnerabilities exist in pattern matching
4. Pattern compilation is secure
"""

import re
import pytest

from nethical.detectors.manipulation_detector import ManipulationDetector


class TestPatternSafety:
    """Test that all patterns are hardcoded and safe from injection."""

    def test_patterns_are_hardcoded(self):
        """Verify that manipulation patterns are hardcoded in the class."""
        detector = ManipulationDetector()
        
        # All patterns should be defined in __init__ as literals
        assert hasattr(detector, "manipulation_patterns")
        assert isinstance(detector.manipulation_patterns, dict)
        assert len(detector.manipulation_patterns) > 0
        
        # Verify patterns are lists of strings
        for category, patterns in detector.manipulation_patterns.items():
            assert isinstance(patterns, list)
            assert all(isinstance(p, str) for p in patterns)

    def test_regex_patterns_are_hardcoded(self):
        """Verify that regex patterns are hardcoded in the class."""
        detector = ManipulationDetector()
        
        assert hasattr(detector, "manipulation_regex")
        assert isinstance(detector.manipulation_regex, dict)
        
        # Verify regex patterns are lists of strings
        for category, patterns in detector.manipulation_regex.items():
            assert isinstance(patterns, list)
            assert all(isinstance(p, str) for p in patterns)

    def test_compiled_patterns_are_precompiled(self):
        """Verify that patterns are precompiled at initialization."""
        detector = ManipulationDetector()
        
        assert hasattr(detector, "_compiled_patterns")
        assert isinstance(detector._compiled_patterns, dict)
        assert len(detector._compiled_patterns) > 0
        
        # All compiled patterns should be re.Pattern objects
        for category, patterns in detector._compiled_patterns.items():
            assert isinstance(patterns, list)
            for keyword, pattern in patterns:
                assert isinstance(keyword, str)
                assert isinstance(pattern, re.Pattern)

    def test_no_user_input_in_pattern_compilation(self):
        """Verify that pattern compilation doesn't accept user input."""
        detector = ManipulationDetector()
        
        # The _compile_patterns method should only use internal data
        # It should not have parameters that accept user input
        import inspect
        sig = inspect.signature(detector._compile_patterns)
        
        # Should only have 'self' parameter
        params = list(sig.parameters.keys())
        assert params == [] or params == ['self'], \
            "Pattern compilation should not accept external parameters"


class TestNoCodeInjection:
    """Test that the detector is not vulnerable to code injection."""

    def test_no_eval_or_exec_in_pattern_matching(self):
        """Verify that eval() or exec() are not used in pattern matching."""
        import inspect
        
        detector = ManipulationDetector()
        
        # Check detect_violations method source
        source = inspect.getsource(detector.detect_violations)
        assert "eval(" not in source
        assert "exec(" not in source
        assert "__import__" not in source
        
        # Check _scan_category method source
        source = inspect.getsource(detector._scan_category)
        assert "eval(" not in source
        assert "exec(" not in source
        assert "__import__" not in source

    def test_no_compile_with_user_input(self):
        """Verify that re.compile() is never called with user input."""
        import inspect
        
        detector = ManipulationDetector()
        
        # Check _compile_patterns method
        source = inspect.getsource(detector._compile_patterns)
        
        # All re.compile() calls should use internal pattern data
        # from self.manipulation_patterns or self.manipulation_regex
        assert "self.manipulation_patterns" in source or "self.manipulation_regex" in source

    @pytest.mark.asyncio
    async def test_malicious_pattern_injection_attempt(self):
        """Test that malicious patterns in text don't affect detection logic."""
        from nethical.core.models import AgentAction, ActionType
        from datetime import datetime, timezone
        
        detector = ManipulationDetector()
        
        # Try to inject a malicious pattern through the action text
        malicious_text = (
            "'; import os; os.system('echo hacked'); #"
            ".*[eval|exec|compile].*"
            "(?P<inject>.*)"
        )
        
        action = AgentAction(
            action_id="test-injection",
            agent_id="test",
            action_type=ActionType.QUERY,
            content=malicious_text,
            intent=malicious_text,
            timestamp=datetime.now(timezone.utc),
        )
        
        # Should not raise any exceptions or execute code
        violations = await detector.detect_violations(action)
        
        # Violations should be a list (empty or with valid violations)
        assert isinstance(violations, list)


class TestPatternValidation:
    """Test that all patterns are valid and safe."""

    def test_all_regex_patterns_are_valid(self):
        """Verify that all regex patterns compile without errors."""
        detector = ManipulationDetector()
        
        # All patterns should have compiled successfully
        for category, patterns in detector._compiled_patterns.items():
            for keyword, pattern in patterns:
                # Pattern should be a compiled regex
                assert isinstance(pattern, re.Pattern)
                
                # Pattern should be able to search text without error
                try:
                    pattern.search("test string")
                except re.error as e:
                    pytest.fail(f"Invalid regex pattern in {category}: {keyword} - {e}")

    def test_patterns_dont_use_dangerous_regex_features(self):
        """Verify patterns don't use potentially dangerous regex features."""
        detector = ManipulationDetector()
        
        # Check that patterns don't use catastrophic backtracking patterns
        dangerous_patterns = [
            r"(.+)+",  # Nested quantifiers
            r"(.*)*",  # Nested quantifiers
            r"(.*).*",  # Nested wildcards
        ]
        
        for category, patterns in detector.manipulation_regex.items():
            for pattern_str in patterns:
                # Check for obvious catastrophic backtracking patterns
                for dangerous in dangerous_patterns:
                    assert dangerous not in pattern_str, \
                        f"Potentially dangerous pattern in {category}: {pattern_str}"

    def test_patterns_use_appropriate_flags(self):
        """Verify that patterns use safe and appropriate flags."""
        detector = ManipulationDetector()
        
        for category, patterns in detector._compiled_patterns.items():
            for keyword, pattern in patterns:
                # Check that patterns use case-insensitive flag (safe)
                assert pattern.flags & re.IGNORECASE, \
                    f"Pattern should be case-insensitive: {keyword}"
                
                # Check that patterns don't use potentially dangerous flags
                # DOTALL and MULTILINE are generally safe, but verify no other flags
                safe_flags = re.IGNORECASE | re.DOTALL | re.MULTILINE | re.UNICODE
                assert pattern.flags <= safe_flags, \
                    f"Pattern uses unexpected flags: {keyword}"


class TestBoundaryProtection:
    """Test that word boundary protection works correctly."""

    def test_boundary_aware_matching(self):
        """Verify that patterns use word boundaries to prevent substring matches."""
        detector = ManipulationDetector()
        
        # Check that literal patterns use boundary markers
        for category, patterns in detector._compiled_patterns.items():
            for keyword, pattern in patterns:
                pattern_str = pattern.pattern
                
                # Literal patterns (not regex) should have word boundaries
                if category in detector.manipulation_patterns:
                    # Should have some form of boundary protection
                    # Either \b, (?<!\w), or (?!\w)
                    has_boundary = (
                        r"\b" in pattern_str or
                        r"(?<!" in pattern_str or
                        r"(?!" in pattern_str
                    )
                    assert has_boundary or keyword in detector.manipulation_regex.get(category, []), \
                        f"Pattern lacks boundary protection: {keyword}"

    @pytest.mark.asyncio
    async def test_boundary_prevents_substring_false_positives(self):
        """Test that word boundaries prevent false positives from substrings."""
        from nethical.core.models import AgentAction, ActionType
        from datetime import datetime, timezone
        
        detector = ManipulationDetector()
        
        # "fear" should match, but "fearless" should not trigger "fear" pattern
        action = AgentAction(
            action_id="test-boundary",
            agent_id="test",
            action_type=ActionType.QUERY,
            content="Be fearless in your approach.",
            timestamp=datetime.now(timezone.utc),
        )
        
        violations = await detector.detect_violations(action)
        
        # Should not detect "fear" in "fearless"
        if violations:
            for v in violations:
                keywords = v.evidence.get("detected_keywords", [])
                assert "fear" not in keywords, \
                    "Word boundary should prevent matching 'fear' in 'fearless'"


class TestPatternImmutability:
    """Test that patterns cannot be modified after initialization."""

    def test_patterns_not_modified_after_init(self):
        """Verify that patterns remain unchanged after initialization."""
        detector = ManipulationDetector()
        
        # Save original pattern count
        original_count = len(detector._compiled_patterns)
        
        # Patterns should not change during normal operation
        import asyncio
        from nethical.core.models import AgentAction, ActionType
        from datetime import datetime, timezone
        
        action = AgentAction(
            action_id="test",
            agent_id="test",
            action_type=ActionType.QUERY,
            content="test",
            intent="test",
            timestamp=datetime.now(timezone.utc),
        )
        
        # Run detection multiple times
        for _ in range(3):
            asyncio.run(detector.detect_violations(action))
        
        # Pattern count should remain the same
        assert len(detector._compiled_patterns) == original_count

    def test_category_lists_not_empty(self):
        """Verify that all category lists contain patterns."""
        detector = ManipulationDetector()
        
        # All categories should have at least one pattern
        for category in detector._scan_order:
            assert category in detector._compiled_patterns
            assert len(detector._compiled_patterns[category]) > 0, \
                f"Category {category} has no patterns"


class TestInputSanitization:
    """Test that user input is properly handled."""

    @pytest.mark.asyncio
    async def test_handles_special_characters_safely(self):
        """Test that special characters in input don't break detection."""
        from nethical.core.models import AgentAction, ActionType
        from datetime import datetime, timezone
        
        detector = ManipulationDetector()
        
        # Text with special regex characters
        special_chars = r".*+?[]{}()|\^$"
        
        action = AgentAction(
            action_id="test-special",
            agent_id="test",
            action_type=ActionType.QUERY,
            content=special_chars,
            timestamp=datetime.now(timezone.utc),
        )
        
        # Should not raise exceptions
        violations = await detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_handles_unicode_safely(self):
        """Test that Unicode characters are handled safely."""
        from nethical.core.models import AgentAction, ActionType
        from datetime import datetime, timezone
        
        detector = ManipulationDetector()
        
        # Text with various Unicode characters
        unicode_text = "你好 مرحبا Здравствуйте 🔒🔐🛡️"
        
        action = AgentAction(
            action_id="test-unicode",
            agent_id="test",
            action_type=ActionType.QUERY,
            content=unicode_text,
            timestamp=datetime.now(timezone.utc),
        )
        
        # Should not raise exceptions
        violations = await detector.detect_violations(action)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_handles_null_bytes_safely(self):
        """Test that null bytes don't cause issues."""
        from nethical.core.models import AgentAction, ActionType
        from datetime import datetime, timezone
        
        detector = ManipulationDetector()
        
        # Text with null byte (if it makes it through to the detector)
        action = AgentAction(
            action_id="test-null",
            agent_id="test",
            action_type=ActionType.QUERY,
            content="test\x00injection",
            timestamp=datetime.now(timezone.utc),
        )
        
        # Should not raise exceptions
        violations = await detector.detect_violations(action)
        assert isinstance(violations, list)
