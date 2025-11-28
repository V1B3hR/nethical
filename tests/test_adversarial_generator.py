"""
Tests for AdversarialGenerator.

Tests synthetic adversarial data generation for model hardening including:
- Prompt injection attacks
- Social engineering attempts
- Obfuscated threats
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
        
        # 3 types * 10 each = 30 samples
        assert len(data) == 30
        
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
            
            # Check feature values
            assert isinstance(features["text"], str)
            assert features["violation_count"] == 0.0
            assert features["severity_max"] == 0.1
            assert features["context_risk"] == 0.9

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
            
            # Check feature values
            assert isinstance(features["text"], str)
            assert features["violation_count"] == 0.1
            assert features["severity_max"] == 0.8
            assert features["context_risk"] == 0.8

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
            
            # Check feature structure
            features = sample["features"]
            assert "text" in features
            assert "violation_count" in features
            assert "severity_max" in features
            assert "context_risk" in features
            
            # Check feature values - obfuscated text contains transforms
            assert isinstance(features["text"], str)
            assert features["violation_count"] == 0.0
            assert features["severity_max"] == 0.9
            assert features["context_risk"] == 1.0

    def test_obfuscation_transforms(self):
        """Test that obfuscation applies leetspeak or spacing transforms."""
        gen = AdversarialGenerator(seed=42)
        
        # Generate many samples to test both transforms
        data = gen.generate_obfuscation(20)
        
        leetspeak_found = False
        spaced_found = False
        
        for sample in data:
            text = sample["features"]["text"]
            # Leetspeak uses digits
            if any(c in text for c in "0134"):
                leetspeak_found = True
            # Spaced version has more spaces than original
            if text.count(" ") > 2:
                spaced_found = True
        
        # Both transforms should be represented
        assert leetspeak_found, "Leetspeak transform not found in samples"
        assert spaced_found, "Spaced transform not found in samples"

    def test_prompt_injection_patterns(self):
        """Test that prompt injection uses expected attack patterns."""
        gen = AdversarialGenerator(seed=42)
        
        # Generate many samples
        data = gen.generate_prompt_injections(50)
        
        texts = [sample["features"]["text"].lower() for sample in data]
        
        # Check for presence of known patterns
        pattern_keywords = ["ignore", "override", "dan", "admin", "developer", "safety"]
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
        pattern_keywords = ["urgent", "account", "money", "lottery", "tech support", "hr"]
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
        assert gen.generate_all(0) == []

    def test_large_count(self):
        """Test generation with large count."""
        gen = AdversarialGenerator(seed=42)
        
        data = gen.generate_all(count_per_type=100)
        
        assert len(data) == 300
        
        # All should be valid
        for sample in data:
            assert sample["label"] == 1
            assert "features" in sample
            assert "metadata" in sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
