"""
Drift Detection Test Suite

Tests for distribution drift monitoring using statistical tests:
- Kolmogorov-Smirnov (KS) test
- Population Stability Index (PSI)

Thresholds:
- PSI: <0.2 daily
- KS: p-value >0.05
"""

import pytest
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta


class DriftDetector:
    """Statistical drift detection"""
    
    @staticmethod
    def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change (requires investigation)
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        breakpoints = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate frequencies
        ref_freq, _ = np.histogram(reference, bins=breakpoints)
        cur_freq, _ = np.histogram(current, bins=breakpoints)
        
        # Convert to proportions and avoid division by zero
        ref_prop = ref_freq / len(reference)
        cur_prop = cur_freq / len(current)
        
        # Replace zeros with small value
        ref_prop[ref_prop == 0] = 0.0001
        cur_prop[cur_prop == 0] = 0.0001
        
        # Calculate PSI
        psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
        return psi
    
    @staticmethod
    def kolmogorov_smirnov_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    @staticmethod
    def detect_drift(reference: np.ndarray, current: np.ndarray, 
                     psi_threshold: float = 0.2, ks_threshold: float = 0.05) -> Dict:
        """
        Detect drift using both PSI and KS tests
        
        Returns:
            Dictionary with drift detection results
        """
        psi = DriftDetector.calculate_psi(reference, current)
        ks_stat, ks_pval = DriftDetector.kolmogorov_smirnov_test(reference, current)
        
        return {
            "psi": psi,
            "psi_drift": psi > psi_threshold,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_pval,
            "ks_drift": ks_pval < ks_threshold,
            "drift_detected": psi > psi_threshold or ks_pval < ks_threshold
        }


class ActionDistributionSimulator:
    """Simulate action distributions for testing"""
    
    @staticmethod
    def generate_baseline_distribution(size: int = 1000, seed: int = 42) -> np.ndarray:
        """Generate baseline risk score distribution"""
        np.random.seed(seed)
        # Mix of low-risk (70%), medium-risk (25%), high-risk (5%)
        low_risk = np.random.beta(2, 8, int(size * 0.7)) * 0.3
        medium_risk = np.random.beta(5, 5, int(size * 0.25)) * 0.4 + 0.3
        high_risk = np.random.beta(8, 2, int(size * 0.05)) * 0.3 + 0.7
        return np.concatenate([low_risk, medium_risk, high_risk])
    
    @staticmethod
    def generate_similar_distribution(size: int = 1000, seed: int = 43) -> np.ndarray:
        """Generate similar distribution (no drift)"""
        np.random.seed(seed)
        low_risk = np.random.beta(2, 8, int(size * 0.7)) * 0.3
        medium_risk = np.random.beta(5, 5, int(size * 0.25)) * 0.4 + 0.3
        high_risk = np.random.beta(8, 2, int(size * 0.05)) * 0.3 + 0.7
        return np.concatenate([low_risk, medium_risk, high_risk])
    
    @staticmethod
    def generate_shifted_distribution(size: int = 1000, seed: int = 44, shift: float = 0.15) -> np.ndarray:
        """Generate shifted distribution (moderate drift)"""
        np.random.seed(seed)
        low_risk = np.random.beta(2, 8, int(size * 0.6)) * 0.3
        medium_risk = np.random.beta(5, 5, int(size * 0.3)) * 0.4 + 0.3
        high_risk = np.random.beta(8, 2, int(size * 0.1)) * 0.3 + 0.7
        return np.concatenate([low_risk, medium_risk, high_risk]) + shift
    
    @staticmethod
    def generate_drifted_distribution(size: int = 1000, seed: int = 45) -> np.ndarray:
        """Generate significantly drifted distribution"""
        np.random.seed(seed)
        # Dramatically different: 40% low, 40% medium, 20% high
        low_risk = np.random.beta(2, 8, int(size * 0.4)) * 0.3
        medium_risk = np.random.beta(5, 5, int(size * 0.4)) * 0.4 + 0.3
        high_risk = np.random.beta(8, 2, int(size * 0.2)) * 0.3 + 0.7
        return np.concatenate([low_risk, medium_risk, high_risk])


@pytest.fixture
def detector():
    """Initialize drift detector"""
    return DriftDetector()


@pytest.fixture
def simulator():
    """Initialize distribution simulator"""
    return ActionDistributionSimulator()


def test_psi_no_drift(detector, simulator):
    """Test PSI with no drift"""
    reference = simulator.generate_baseline_distribution()
    current = simulator.generate_similar_distribution()
    
    result = detector.detect_drift(reference, current)
    
    print(f"\nNo Drift Test:")
    print(f"  PSI: {result['psi']:.4f}")
    print(f"  KS p-value: {result['ks_p_value']:.4f}")
    print(f"  Drift Detected: {result['drift_detected']}")
    
    assert result["psi"] < 0.2, f"PSI {result['psi']:.4f} exceeds threshold"
    assert not result["drift_detected"], "Drift incorrectly detected"


def test_psi_moderate_drift(detector, simulator):
    """Test PSI with moderate drift"""
    reference = simulator.generate_baseline_distribution()
    current = simulator.generate_shifted_distribution(shift=0.10)
    
    result = detector.detect_drift(reference, current, psi_threshold=0.2)
    
    print(f"\nModerate Drift Test:")
    print(f"  PSI: {result['psi']:.4f}")
    print(f"  KS p-value: {result['ks_p_value']:.4f}")
    
    # Moderate drift should be detected but not too severe
    assert 0.1 <= result["psi"] < 0.3, f"PSI {result['psi']:.4f} not in moderate range"


def test_psi_significant_drift(detector, simulator):
    """Test PSI with significant drift"""
    reference = simulator.generate_baseline_distribution()
    current = simulator.generate_drifted_distribution()
    
    result = detector.detect_drift(reference, current)
    
    print(f"\nSignificant Drift Test:")
    print(f"  PSI: {result['psi']:.4f}")
    print(f"  KS p-value: {result['ks_p_value']:.4f}")
    print(f"  Drift Detected: {result['drift_detected']}")
    
    assert result["psi"] > 0.2, f"PSI {result['psi']:.4f} should exceed threshold"
    assert result["drift_detected"], "Significant drift not detected"


def test_ks_test_no_drift(detector, simulator):
    """Test Kolmogorov-Smirnov with no drift"""
    reference = simulator.generate_baseline_distribution()
    current = simulator.generate_similar_distribution()
    
    ks_stat, ks_pval = detector.kolmogorov_smirnov_test(reference, current)
    
    print(f"\nKS Test (No Drift):")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_pval:.4f}")
    
    assert ks_pval > 0.05, f"KS test incorrectly detected drift (p={ks_pval:.4f})"


def test_ks_test_with_drift(detector, simulator):
    """Test Kolmogorov-Smirnov with drift"""
    reference = simulator.generate_baseline_distribution()
    current = simulator.generate_drifted_distribution()
    
    ks_stat, ks_pval = detector.kolmogorov_smirnov_test(reference, current)
    
    print(f"\nKS Test (With Drift):")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_pval:.4f}")
    
    assert ks_pval < 0.05, f"KS test failed to detect drift (p={ks_pval:.4f})"


def test_weekly_drift_monitoring(detector, simulator):
    """Simulate weekly drift monitoring"""
    reference = simulator.generate_baseline_distribution()
    
    # Simulate 7 days of data
    weekly_results = []
    for day in range(7):
        # Gradually introduce drift
        shift = day * 0.02
        current = simulator.generate_shifted_distribution(seed=50 + day, shift=shift)
        result = detector.detect_drift(reference, current)
        result["day"] = day + 1
        weekly_results.append(result)
    
    print(f"\nWeekly Drift Monitoring:")
    for result in weekly_results:
        print(f"  Day {result['day']}: PSI={result['psi']:.4f}, Drift={result['drift_detected']}")
    
    # Early days should have no drift
    assert not weekly_results[0]["drift_detected"], "Day 1 should have no drift"
    assert not weekly_results[1]["drift_detected"], "Day 2 should have no drift"


def test_generate_drift_report(detector, simulator, tmp_path):
    """Generate drift detection report"""
    reference = simulator.generate_baseline_distribution()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "reference_stats": {
            "mean": float(reference.mean()),
            "std": float(reference.std()),
            "min": float(reference.min()),
            "max": float(reference.max()),
            "samples": len(reference)
        },
        "daily_checks": []
    }
    
    # Simulate multiple days
    for day in range(7):
        current = simulator.generate_similar_distribution(seed=60 + day)
        result = detector.detect_drift(reference, current)
        
        daily_result = {
            "day": day + 1,
            "date": (datetime.now() - timedelta(days=7-day)).isoformat(),
            "psi": float(result["psi"]),
            "ks_statistic": float(result["ks_statistic"]),
            "ks_p_value": float(result["ks_p_value"]),
            "drift_detected": result["drift_detected"],
            "current_stats": {
                "mean": float(current.mean()),
                "std": float(current.std()),
                "samples": len(current)
            }
        }
        report["daily_checks"].append(daily_result)
    
    # Save report
    report_path = tmp_path / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDrift report saved to: {report_path}")
    assert report_path.exists()
    
    # Verify thresholds met
    for check in report["daily_checks"]:
        assert check["psi"] < 0.2, f"Day {check['day']} PSI exceeds threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
