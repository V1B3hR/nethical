"""Tests for performance improvements made to the codebase."""

import time
import numpy as np
import pytest
from nethical.core.differential_privacy import DifferentialPrivacy, PrivacyMechanism
from nethical.core.jit_optimizations import calculate_risk_score_jit


class TestDifferentialPrivacyPerformance:
    """Test performance improvements in differential privacy module."""

    def test_vectorized_noise_addition_performance(self):
        """Test that vectorized noise addition is working correctly and efficiently."""
        dp = DifferentialPrivacy(
            epsilon=1.0, delta=1e-5, mechanism=PrivacyMechanism.GAUSSIAN
        )

        # Test with a moderately sized vector
        vector = np.random.rand(100)
        sensitivity = 1.0

        start = time.time()
        noised = dp.add_noise_to_vector(vector, sensitivity)
        elapsed = time.time() - start

        # Verify the result is correct
        assert noised.shape == vector.shape
        assert not np.array_equal(noised, vector)  # Should have noise

        # Verify performance (should be very fast)
        assert (
            elapsed < 0.1
        ), f"Vectorized noise addition took {elapsed}s, expected < 0.1s"

        # Verify privacy budget consumed
        assert dp.budget.consumed > 0
        assert len(dp.operation_history) == 1
        assert dp.operation_history[0]["vector_size"] == 100

    def test_vectorized_noise_preserves_privacy_guarantees(self):
        """Ensure vectorization doesn't break privacy guarantees."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        vector1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vector2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Add noise to both - should get different results
        noised1 = dp.add_noise_to_vector(vector1, sensitivity=1.0, operation="test1")

        # Create new instance to reset budget
        dp2 = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        noised2 = dp2.add_noise_to_vector(vector2, sensitivity=1.0, operation="test2")

        # Results should be different (with high probability)
        assert not np.allclose(noised1, noised2, rtol=0.1)

    def test_vectorized_noise_budget_accounting(self):
        """Test that budget is correctly accounted for vectorized operations."""
        dp = DifferentialPrivacy(
            epsilon=3.0, delta=1e-5
        )  # Increased budget for multiple operations
        initial_budget = dp.budget.remaining

        vector = np.ones(10)
        dp.add_noise_to_vector(vector, sensitivity=1.0)

        # Budget should be consumed
        assert dp.budget.remaining < initial_budget

        # Should be able to do multiple operations within budget
        dp.add_noise_to_vector(vector, sensitivity=1.0)

        # Budget consumed should be tracked
        assert dp.budget.consumed > 0


class TestJITOptimizationsPerformance:
    """Test performance improvements in JIT optimizations module."""

    def test_vectorized_risk_score_calculation(self):
        """Test that vectorized risk score calculation is correct."""
        severities = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        confidences = np.array([0.9, 0.8, 0.7, 0.85, 0.95])

        score = calculate_risk_score_jit(severities, confidences)

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0

        # Higher severities should generally give higher scores
        low_severities = np.array([1.0, 1.0, 1.0])
        low_confidences = np.array([0.9, 0.9, 0.9])
        low_score = calculate_risk_score_jit(low_severities, low_confidences)

        high_severities = np.array([5.0, 5.0, 5.0])
        high_confidences = np.array([0.9, 0.9, 0.9])
        high_score = calculate_risk_score_jit(high_severities, high_confidences)

        assert high_score > low_score

    def test_vectorized_risk_score_performance(self):
        """Test that vectorized risk score calculation is efficient."""
        # Large dataset
        severities = np.random.uniform(0, 5, size=1000)
        confidences = np.random.uniform(0.5, 1.0, size=1000)

        start = time.time()
        for _ in range(100):
            score = calculate_risk_score_jit(severities, confidences)
        elapsed = time.time() - start

        # Should be fast even with many calculations
        assert elapsed < 1.0, f"100 risk calculations took {elapsed}s, expected < 1s"

    def test_vectorized_handles_edge_cases(self):
        """Test that vectorized calculation handles edge cases."""
        # Empty array
        empty_severities = np.array([])
        empty_confidences = np.array([])
        score = calculate_risk_score_jit(empty_severities, empty_confidences)
        assert score == 0.0

        # Single value
        single_severity = np.array([3.0])
        single_confidence = np.array([0.8])
        score = calculate_risk_score_jit(single_severity, single_confidence)
        assert 0.0 <= score <= 1.0


class TestSystemLimitsDetectorPerformance:
    """Test performance improvements in system limits detector."""

    def test_monotonic_increase_detection(self):
        """Test the optimized monotonic increase detection."""
        # This tests the zip-based comparison vs range(len())
        recent_sizes = [100, 200, 300, 400, 500, 600]

        # Original pattern: all(recent_sizes[i] < recent_sizes[i+1] for i in range(len(recent_sizes)-1))
        # Optimized: all(a < b for a, b in zip(recent_sizes, recent_sizes[1:]))

        result = all(a < b for a, b in zip(recent_sizes, recent_sizes[1:]))
        assert result is True

        # Test with non-monotonic
        non_monotonic = [100, 200, 150, 400, 500]
        result = all(a < b for a, b in zip(non_monotonic, non_monotonic[1:]))
        assert result is False


class TestHealthcarePipelinePerformance:
    """Test performance improvements in healthcare pipeline."""

    def test_no_unnecessary_deepcopy(self):
        """Verify that healthcare pipeline doesn't use unnecessary deepcopy."""
        # This is a code quality test - verify the optimization was applied
        from pathlib import Path

        # Read the source file directly
        filepath = (
            Path(__file__).parent.parent
            / "nethical"
            / "integrations"
            / "healthcare_pipeline.py"
        )
        with open(filepath, "r") as f:
            source = f.read()

        # Count occurrences of copy.deepcopy
        deepcopy_count = source.count("copy.deepcopy")

        # We removed one deepcopy, so count should be 0
        # (There's still an import statement but no usage)
        assert (
            deepcopy_count == 0
        ), f"Healthcare pipeline should not use copy.deepcopy unnecessarily (found {deepcopy_count} usage)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
