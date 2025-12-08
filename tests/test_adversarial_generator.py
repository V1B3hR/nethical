"""
Tests for AdversarialGenerator.

Tests synthetic adversarial data generation for model hardening including:
- Prompt injection attacks
- Social engineering attempts
- Obfuscated threats
- Toxic/Unsafe content
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.mlops.adversarial import AdversarialGenerator


class TestAdversarialGenerator:
    """Tests for AdversarialGenerator class."""

    def test_initialization_with_seed(self):
        """Test that generator initializes with a seed for reproducibility."""
        # Create fresh generators with same seed
        gen1 = AdversarialGenerator(seed=42)
        data1 = gen1.generate_prompt_injections(3)

        # Reset with same seed - should produce same results
        gen2 = AdversarialGenerator(seed=42)
        data2 = gen2.generate_prompt_injections(3)

        # Same seed should produce same results when generator is freshly created
        assert data1[0]["features"]["text"] == data2[0]["features"]["text"]

    def test_generate_all(self):
        """Test generate_all produces expected number of samples."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_all(count_per_type=10)

        # 4 types * 10 each = 40 samples
        assert len(data) == 40

        # All should be threats (label=1)
        for sample in data:
            assert sample["label"] == 1

    def test_generate_method(self):
        """Test generate method produces requested number of samples."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate(num_samples=100)

        assert len(data) == 100

        # All should be threats (label=1)
        for sample in data:
            assert sample["label"] == 1

    def test_generate_all_shuffled(self):
        """Test that generate_all shuffles the data."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_all(count_per_type=10)

        # Check that types are mixed (not all same type in sequence)
        types = [sample["metadata"]["type"] for sample in data[:10]]
        unique_types = set(types)
        # Should have multiple types in first 10 samples due to shuffle
        assert len(unique_types) > 1

    def test_generate_prompt_injections(self):
        """Test prompt injection sample generation."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_prompt_injections(5)

        assert len(data) == 5

        for sample in data:
            assert "features" in sample
            assert "label" in sample
            assert "metadata" in sample
            assert sample["label"] == 1
            assert sample["metadata"]["type"] == "adversarial_prompt_injection"

            # Check feature structure
            features = sample["features"]
            assert "text" in features
            assert "violation_count" in features
            assert "severity_max" in features
            assert "context_risk" in features
            assert "recency_score" in features
            assert "frequency_score" in features

            # Check feature values are within expected ranges
            assert isinstance(features["text"], str)
            assert 0.0 <= features["violation_count"] <= 0.2
            assert 0.1 <= features["severity_max"] <= 0.3
            assert 0.8 <= features["context_risk"] <= 1.0

    def test_generate_social_engineering(self):
        """Test social engineering sample generation."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_social_engineering(5)

        assert len(data) == 5

        for sample in data:
            assert "features" in sample
            assert "label" in sample
            assert "metadata" in sample
            assert sample["label"] == 1
            assert sample["metadata"]["type"] == "adversarial_social_engineering"

            # Check feature structure
            features = sample["features"]
            assert "text" in features
            assert "violation_count" in features
            assert "severity_max" in features
            assert "context_risk" in features

            # Check feature values are within expected ranges
            assert isinstance(features["text"], str)
            assert 0.0 <= features["violation_count"] <= 0.2
            assert 0.7 <= features["severity_max"] <= 0.9
            assert 0.7 <= features["context_risk"] <= 0.9

    def test_generate_obfuscation(self):
        """Test obfuscation sample generation."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_obfuscation(5)

        assert len(data) == 5

        for sample in data:
            assert "features" in sample
            assert "label" in sample
            assert "metadata" in sample
            assert sample["label"] == 1
            assert sample["metadata"]["type"] == "adversarial_obfuscation"
            assert "method" in sample["metadata"]

            # Check feature structure
            features = sample["features"]
            assert "text" in features
            assert "violation_count" in features
            assert "severity_max" in features
            assert "context_risk" in features

            # Check feature values
            assert isinstance(features["text"], str)
            assert features["violation_count"] == 0.0
            assert 0.8 <= features["severity_max"] <= 1.0
            assert 0.9 <= features["context_risk"] <= 1.0

    def test_generate_toxic_content(self):
        """Test toxic content sample generation."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_toxic_content(5)

        assert len(data) == 5

        for sample in data:
            assert "features" in sample
            assert "label" in sample
            assert "metadata" in sample
            assert sample["label"] == 1
            assert sample["metadata"]["type"] == "adversarial_toxic_content"

            # Check feature structure
            features = sample["features"]
            assert "text" in features
            assert "violation_count" in features
            assert "severity_max" in features
            assert "context_risk" in features

            # Check feature values - toxic content has high violation counts
            assert isinstance(features["text"], str)
            assert 0.7 <= features["violation_count"] <= 1.0
            assert 0.9 <= features["severity_max"] <= 1.0
            assert 0.9 <= features["context_risk"] <= 1.0

    def test_obfuscation_transforms(self):
        """Test that obfuscation applies multiple transforms."""
        gen = AdversarialGenerator(seed=42)

        # Generate many samples to test all transforms
        data = gen.generate_obfuscation(50)

        methods_found = set()
        for sample in data:
            methods_found.add(sample["metadata"]["method"])

        # All 4 methods should be represented
        expected_methods = {"leetspeak", "spaced", "base64", "homoglyph"}
        assert (
            methods_found == expected_methods
        ), f"Missing methods: {expected_methods - methods_found}"

    def test_prompt_injection_patterns(self):
        """Test that prompt injection uses expected attack patterns."""
        gen = AdversarialGenerator(seed=42)

        # Generate many samples
        data = gen.generate_prompt_injections(50)

        texts = [sample["features"]["text"].lower() for sample in data]

        # Check for presence of known patterns
        pattern_keywords = [
            "ignore",
            "override",
            "dan",
            "admin",
            "developer",
            "safety",
            "bypass",
        ]
        found_patterns = set()
        for text in texts:
            for keyword in pattern_keywords:
                if keyword in text:
                    found_patterns.add(keyword)

        # Should find multiple pattern types
        assert len(found_patterns) >= 3, f"Only found patterns: {found_patterns}"

    def test_social_engineering_patterns(self):
        """Test that social engineering uses expected attack patterns."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_social_engineering(50)

        texts = [sample["features"]["text"].lower() for sample in data]

        # Check for presence of known patterns
        pattern_keywords = [
            "urgent",
            "account",
            "password",
            "verify",
            "security",
            "payment",
            "action",
        ]
        found_patterns = set()
        for text in texts:
            for keyword in pattern_keywords:
                if keyword in text:
                    found_patterns.add(keyword)

        # Should find multiple pattern types
        assert len(found_patterns) >= 3, f"Only found patterns: {found_patterns}"

    def test_empty_count(self):
        """Test generation with zero count returns empty list."""
        gen = AdversarialGenerator(seed=42)

        assert gen.generate_prompt_injections(0) == []
        assert gen.generate_social_engineering(0) == []
        assert gen.generate_obfuscation(0) == []
        assert gen.generate_toxic_content(0) == []
        assert gen.generate_all(0) == []
        assert gen.generate(0) == []

    def test_large_count(self):
        """Test generation with large count."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate_all(count_per_type=100)

        # 4 types * 100 each = 400 samples
        assert len(data) == 400

        # All should be valid
        for sample in data:
            assert sample["label"] == 1
            assert "features" in sample
            assert "metadata" in sample

    def test_generate_5000_samples(self):
        """Test generating 5000 samples as specified in requirements."""
        gen = AdversarialGenerator(seed=42)

        data = gen.generate(num_samples=5000)

        assert len(data) == 5000

        # Check type distribution (should be roughly equal)
        type_counts = {}
        for sample in data:
            sample_type = sample["metadata"]["type"]
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

        # Each type should have roughly 1250 samples (5000/4)
        for count in type_counts.values():
            assert 1000 <= count <= 1500, f"Uneven distribution: {type_counts}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
