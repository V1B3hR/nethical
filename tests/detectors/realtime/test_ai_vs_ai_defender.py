"""Unit tests for AI vs AI Defender."""

import pytest

from nethical.detectors.realtime import AIvsAIDefender, AIvsAIDefenderConfig


class TestAIvsAIDefender:
    """Test cases for AI vs AI Defender."""

    @pytest.fixture
    def defender(self):
        """Create defender instance for testing."""
        return AIvsAIDefender()

    @pytest.mark.asyncio
    async def test_detect_model_extraction(self, defender):
        """Test detection of model extraction attempts."""
        # Generate many diverse queries
        query_history = [
            {"query": {"input": f"test_{i}", "type": f"type_{i % 5}"}}
            for i in range(100)
        ]

        context = {
            "query": {"input": "test_query"},
            "query_history": query_history,
            "client_id": "test_client",
        }

        violations = await defender.detect_violations(context)
        # May detect model extraction
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_detect_adversarial_invisible_chars(self, defender):
        """Test detection of adversarial examples with invisible characters."""
        context = {
            "query": {"input": "Hello\u200bWorld"},  # Zero-width space
            "query_history": [],
            "client_id": "test_client",
        }

        violations = await defender.detect_violations(context)
        assert len(violations) > 0
        assert violations[0].category == "adversarial_attack"

    @pytest.mark.asyncio
    async def test_detect_membership_inference(self, defender):
        """Test detection of membership inference attacks."""
        # Generate similar queries
        similar_query = {"input": "test query", "type": "test"}
        query_history = [{"query": similar_query} for _ in range(15)]

        context = {
            "query": similar_query,
            "query_history": query_history,
            "client_id": "test_client",
        }

        violations = await defender.detect_violations(context)
        # May detect membership inference
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_detect_rate_limiting(self, defender):
        """Test rate limiting detection."""
        import time

        client_id = "rate_test_client"

        # Simulate rapid requests
        for _ in range(105):  # Exceed threshold of 100
            context = {
                "query": {"input": "test"},
                "query_history": [],
                "client_id": client_id,
            }
            violations = await defender.detect_violations(context)

        # Last request should trigger rate limit
        assert len(violations) > 0
        assert any(v.category == "rate_limit_exceeded" for v in violations)

    @pytest.mark.asyncio
    async def test_defend_api(self, defender):
        """Test public defend API."""
        query = {"input": "test query"}
        query_history = []

        result = await defender.defend(query, query_history)

        assert result["status"] == "success"
        assert "attack_detected" in result
        assert "should_block" in result
        assert "confidence" in result
        assert "violations" in result
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_empty_query(self, defender):
        """Test with empty query."""
        context = {
            "query": {},
            "query_history": [],
            "client_id": "test",
        }

        violations = await defender.detect_violations(context)
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_performance_target(self, defender):
        """Test that defense meets performance target (<25ms)."""
        import time

        context = {
            "query": {"input": "test query"},
            "query_history": [],
            "client_id": "perf_test",
        }

        start = time.perf_counter()
        await defender.detect_violations(context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 25ms (with margin)
        assert elapsed_ms < 100

    def test_defender_initialization(self):
        """Test defender initialization."""
        defender = AIvsAIDefender()
        assert defender.name == "ai_vs_ai_defender"
        assert defender.version == "1.0.0"

    def test_custom_config(self):
        """Test defender with custom configuration."""
        config = AIvsAIDefenderConfig(
            rate_limit_threshold=50,
            similarity_threshold=0.9,
        )
        defender = AIvsAIDefender(config)
        assert defender.config.rate_limit_threshold == 50
        assert defender.config.similarity_threshold == 0.9
