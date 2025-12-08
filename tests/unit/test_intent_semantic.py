"""Tests for semantic intent deviation monitoring (v2.0)."""

import pytest
from nethical.core.semantics import (
    get_similarity,
    get_semantic_deviation,
    is_semantic_available,
    _lexical_similarity,
)


class TestSemanticSimilarity:
    """Test semantic similarity computation."""

    def test_exact_match(self):
        """Test that identical texts have high similarity."""
        text = "process user data"
        similarity = get_similarity(text, text)
        assert similarity >= 0.95, "Identical texts should have very high similarity"

    def test_paraphrase_similarity(self):
        """Test that paraphrases have higher semantic than lexical similarity."""
        intent = "analyze customer information"
        action = "examine client data"

        # Semantic similarity
        semantic_sim = get_similarity(intent, action)

        # Lexical similarity (fallback)
        lexical_sim = _lexical_similarity(intent, action)

        # For paraphrases, semantic should be significantly better
        # (only if semantic models available)
        if is_semantic_available():
            assert semantic_sim > lexical_sim, (
                f"Semantic similarity ({semantic_sim}) should be higher than "
                f"lexical ({lexical_sim}) for paraphrases"
            )
            assert (
                semantic_sim > 0.5
            ), "Paraphrases should have moderate-high similarity"
        else:
            # Without semantic models, similarity may be low
            assert lexical_sim >= 0.0

    def test_unrelated_texts(self):
        """Test that unrelated texts have low similarity."""
        text_a = "process payments"
        text_b = "write documentation"

        similarity = get_similarity(text_a, text_b)
        assert similarity < 0.5, "Unrelated texts should have low similarity"

    def test_empty_texts(self):
        """Test handling of empty texts."""
        assert get_similarity("", "something") == 0.0
        assert get_similarity("something", "") == 0.0
        assert get_similarity("", "") == 0.0

    def test_deviation_computation(self):
        """Test deviation is inverse of similarity."""
        intent = "fetch user records"
        action = "retrieve user data"

        similarity = get_similarity(intent, action)
        deviation = get_semantic_deviation(intent, action)

        # Deviation should be 1 - similarity
        assert abs(deviation - (1.0 - similarity)) < 0.01
        assert 0.0 <= deviation <= 1.0


class TestLexicalFallback:
    """Test lexical similarity fallback."""

    def test_jaccard_identical(self):
        """Test Jaccard similarity for identical texts."""
        text = "hello world test"
        similarity = _lexical_similarity(text, text)
        assert similarity == 1.0

    def test_jaccard_overlap(self):
        """Test Jaccard similarity for partial overlap."""
        text_a = "hello world"
        text_b = "world test"

        # Tokens: {hello, world} vs {world, test}
        # Intersection: {world} = 1
        # Union: {hello, world, test} = 3
        # Jaccard: 1/3 = 0.333...
        similarity = _lexical_similarity(text_a, text_b)
        assert 0.3 <= similarity <= 0.4

    def test_jaccard_no_overlap(self):
        """Test Jaccard similarity for no overlap."""
        text_a = "hello world"
        text_b = "foo bar"

        similarity = _lexical_similarity(text_a, text_b)
        assert similarity == 0.0

    def test_case_insensitive(self):
        """Test that lexical similarity is case-insensitive."""
        text_a = "HELLO WORLD"
        text_b = "hello world"

        similarity = _lexical_similarity(text_a, text_b)
        assert similarity == 1.0


@pytest.mark.parametrize(
    "intent,action,expected_high_semantic",
    [
        ("delete files", "remove documents", True),
        ("query database", "search data store", True),
        ("send email", "transmit message", True),
        ("process payment", "handle transaction", True),
        ("create user", "generate account", True),
        ("log information", "record data", True),
    ],
)
def test_semantic_better_than_lexical(intent, action, expected_high_semantic):
    """Test that semantic similarity detects paraphrases better than lexical."""
    if not is_semantic_available():
        pytest.skip("Semantic models not available")

    semantic_sim = get_similarity(intent, action)
    lexical_sim = _lexical_similarity(intent, action)

    if expected_high_semantic:
        # Semantic should be better for true paraphrases
        assert semantic_sim > lexical_sim or semantic_sim > 0.5, (
            f"Expected high semantic similarity for '{intent}' vs '{action}', "
            f"got semantic={semantic_sim}, lexical={lexical_sim}"
        )


@pytest.mark.parametrize(
    "intent,action,expected_deviation",
    [
        ("read data", "read data", 0.0),  # No deviation
        ("read data", "delete all files", 1.0),  # Maximum deviation
        ("fetch users", "retrieve users", 0.1),  # Low deviation (synonyms)
        ("save file", "remove database", 0.9),  # High deviation (opposite)
    ],
)
def test_deviation_ranges(intent, action, expected_deviation):
    """Test that deviation scores are in expected ranges."""
    deviation = get_semantic_deviation(intent, action)

    # Allow some tolerance
    tolerance = 0.2
    assert abs(deviation - expected_deviation) <= tolerance, (
        f"Deviation for '{intent}' vs '{action}' is {deviation}, "
        f"expected ~{expected_deviation}"
    )


def test_semantic_availability():
    """Test semantic availability check."""
    available = is_semantic_available()
    assert isinstance(available, bool)

    # If available, we should be able to compute similarity
    if available:
        sim = get_similarity("test", "test")
        assert sim > 0.9


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self):
        """Test handling of very long texts."""
        long_text = "word " * 1000
        short_text = "word word"

        # Should not crash
        similarity = get_similarity(long_text, short_text)
        assert 0.0 <= similarity <= 1.0

    def test_special_characters(self):
        """Test handling of special characters."""
        text_a = "hello @#$% world!!!"
        text_b = "hello world"

        similarity = get_similarity(text_a, text_b)
        assert similarity > 0.5  # Should still recognize similarity

    def test_numbers(self):
        """Test handling of numeric content."""
        text_a = "process 123 items"
        text_b = "process 456 items"

        similarity = get_similarity(text_a, text_b)
        assert similarity > 0.5  # Should recognize structural similarity

    def test_single_words(self):
        """Test single word comparison."""
        similarity = get_similarity("hello", "hello")
        assert similarity >= 0.95

        similarity = get_similarity("hello", "goodbye")
        assert similarity < 0.7
