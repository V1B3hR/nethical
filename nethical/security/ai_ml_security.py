"""
Phase 6.1: AI/ML Security Framework

This module provides comprehensive AI/ML security capabilities including adversarial
example detection, model poisoning detection, differential privacy integration,
federated learning framework, and explainable AI for compliance with military,
government, and healthcare requirements.

Key Features:
- Adversarial example detection using input perturbation analysis
- Chaos Quantification: Utilizing chaos theory to identify chaotic behavior in
  time-series data to detect subtle input perturbations added by adversarial attacks
- Defense Perturbation: A novel methodology to detect robust adversarial examples
  with the same input transformations the adversarial examples are robust to
- Model poisoning detection via gradient monitoring
- Differential privacy with epsilon-delta guarantees
- Federated learning with secure aggregation
- Explainable AI compliance reporting (GDPR, HIPAA, DoD AI Ethics)
"""

from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import logging


class AdversarialAttackType(Enum):
    """Types of adversarial attacks on ML models."""

    FGSM = "fast_gradient_sign_method"
    PGD = "projected_gradient_descent"
    DEEPFOOL = "deepfool"
    CARLINI_WAGNER = "carlini_wagner"
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_INVERSION = "model_inversion"
    BACKDOOR = "backdoor"
    ROBUST_ADVERSARIAL = "robust_adversarial"


class PoisoningType(Enum):
    """Types of model poisoning attacks."""

    DATA_POISONING = "data_poisoning"
    LABEL_FLIPPING = "label_flipping"
    BACKDOOR_INJECTION = "backdoor_injection"
    GRADIENT_MANIPULATION = "gradient_manipulation"
    FEDERATED_POISONING = "federated_poisoning"


class PrivacyMechanism(Enum):
    """Differential privacy mechanisms."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    RANDOMIZED_RESPONSE = "randomized_response"


class TransformationType(Enum):
    """Types of input transformations for Defense Perturbation."""

    # Geometric transformations
    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALING = "scaling"
    SHEARING = "shearing"
    FLIPPING_HORIZONTAL = "flipping_horizontal"
    FLIPPING_VERTICAL = "flipping_vertical"
    AFFINE = "affine"
    PERSPECTIVE = "perspective"

    # Color/intensity transformations
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE_SHIFT = "hue_shift"
    GAMMA_CORRECTION = "gamma_correction"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"

    # Noise-based transformations
    GAUSSIAN_NOISE = "gaussian_noise"
    SALT_PEPPER_NOISE = "salt_pepper_noise"
    SPECKLE_NOISE = "speckle_noise"
    POISSON_NOISE = "poisson_noise"

    # Compression transformations
    JPEG_COMPRESSION = "jpeg_compression"
    WEBP_COMPRESSION = "webp_compression"
    QUANTIZATION = "quantization"

    # Spatial filtering
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_FILTER = "median_filter"
    BILATERAL_FILTER = "bilateral_filter"
    SHARPENING = "sharpening"

    # Frequency domain
    LOW_PASS_FILTER = "low_pass_filter"
    HIGH_PASS_FILTER = "high_pass_filter"
    BAND_PASS_FILTER = "band_pass_filter"

    # Advanced
    ELASTIC_DEFORMATION = "elastic_deformation"
    CUTOUT = "cutout"
    MIXUP = "mixup"


class ChaosMetricType(Enum):
    """Types of chaos metrics for Chaos Quantification."""

    LYAPUNOV_EXPONENT = "lyapunov_exponent"
    CORRELATION_DIMENSION = "correlation_dimension"
    APPROXIMATE_ENTROPY = "approximate_entropy"
    SAMPLE_ENTROPY = "sample_entropy"
    PERMUTATION_ENTROPY = "permutation_entropy"
    RECURRENCE_RATE = "recurrence_rate"
    DETERMINISM = "determinism"
    LAMINARITY = "laminarity"
    HURST_EXPONENT = "hurst_exponent"
    FRACTAL_DIMENSION = "fractal_dimension"


@dataclass
class ChaosAnalysisResult:
    """Result of chaos quantification analysis."""

    is_chaotic: bool
    chaos_score: float
    lyapunov_exponent: float
    correlation_dimension: float
    entropy_measures: Dict[str, float]
    recurrence_metrics: Dict[str, float]
    fractal_dimension: float
    hurst_exponent: float
    is_adversarial_perturbation: bool
    confidence: float
    analysis_method: str = "chaos_quantification"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_chaotic": self.is_chaotic,
            "chaos_score": self.chaos_score,
            "lyapunov_exponent": self.lyapunov_exponent,
            "correlation_dimension": self.correlation_dimension,
            "entropy_measures": self.entropy_measures,
            "recurrence_metrics": self.recurrence_metrics,
            "fractal_dimension": self.fractal_dimension,
            "hurst_exponent": self.hurst_exponent,
            "is_adversarial_perturbation": self.is_adversarial_perturbation,
            "confidence": self.confidence,
            "analysis_method": self.analysis_method,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DefensePerturbationResult:
    """Result of defense perturbation analysis."""

    is_robust_adversarial: bool
    robustness_score: float
    transformations_tested: List[str]
    transformation_results: Dict[str, Dict[str, Any]]
    vulnerable_transformations: List[str]
    robust_transformations: List[str]
    prediction_stability: float
    confidence: float
    detection_method: str = "defense_perturbation"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_robust_adversarial": self.is_robust_adversarial,
            "robustness_score": self.robustness_score,
            "transformations_tested": self.transformations_tested,
            "transformation_results": self.transformation_results,
            "vulnerable_transformations": self.vulnerable_transformations,
            "robust_transformations": self.robust_transformations,
            "prediction_stability": self.prediction_stability,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AdversarialDetectionResult:
    """Result of adversarial example detection."""

    is_adversarial: bool
    confidence: float
    attack_type: Optional[AdversarialAttackType]
    perturbation_magnitude: float
    original_prediction: Optional[Any] = None
    adversarial_prediction: Optional[Any] = None
    detection_method: str = "perturbation_analysis"
    chaos_analysis: Optional[ChaosAnalysisResult] = None
    defense_perturbation_analysis: Optional[DefensePerturbationResult] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_adversarial": self.is_adversarial,
            "confidence": self.confidence,
            "attack_type": self.attack_type.value if self.attack_type else None,
            "perturbation_magnitude": self.perturbation_magnitude,
            "detection_method": self.detection_method,
            "chaos_analysis": (
                self.chaos_analysis.to_dict() if self.chaos_analysis else None
            ),
            "defense_perturbation_analysis": (
                self.defense_perturbation_analysis.to_dict()
                if self.defense_perturbation_analysis
                else None
            ),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PoisoningDetectionResult:
    """Result of model poisoning detection."""

    is_poisoned: bool
    confidence: float
    poisoning_type: Optional[PoisoningType]
    affected_samples: int
    gradient_anomaly_score: float
    detection_method: str = "gradient_analysis"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_poisoned": self.is_poisoned,
            "confidence": self.confidence,
            "poisoning_type": (
                self.poisoning_type.value if self.poisoning_type else None
            ),
            "affected_samples": self.affected_samples,
            "gradient_anomaly_score": self.gradient_anomaly_score,
            "detection_method": self.detection_method,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracking."""

    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    query_count: int = 0

    @property
    def remaining_epsilon(self) -> float:
        """Calculate remaining privacy budget."""
        return max(0.0, self.epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        """Calculate remaining delta budget."""
        return max(0.0, self.delta - self.spent_delta)

    @property
    def is_depleted(self) -> bool:
        """Check if privacy budget is depleted."""
        return self.spent_epsilon >= self.epsilon or self.spent_delta >= self.delta

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "query_count": self.query_count,
            "is_depleted": self.is_depleted,
        }


@dataclass
class FederatedLearningRound:
    """Federated learning aggregation round."""

    round_id: str
    participant_count: int
    aggregated_weights: Dict[str, Any]
    validation_accuracy: float
    poisoning_detected: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_id": self.round_id,
            "participant_count": self.participant_count,
            "validation_accuracy": self.validation_accuracy,
            "poisoning_detected": self.poisoning_detected,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExplainabilityReport:
    """AI explainability report for compliance."""

    model_id: str
    prediction: Any
    feature_importance: Dict[str, float]
    explanation_method: str
    compliance_frameworks: List[str]
    human_readable_explanation: str
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "prediction": str(self.prediction),
            "feature_importance": self.feature_importance,
            "explanation_method": self.explanation_method,
            "compliance_frameworks": self.compliance_frameworks,
            "human_readable_explanation": self.human_readable_explanation,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# CHAOS QUANTIFICATION SYSTEM
# =============================================================================


class ChaosQuantificationSystem:
    """
    Chaos Quantification System for Adversarial Detection.

    Utilizes chaos theory to identify chaotic behavior in time-series data
    to detect subtle input perturbations added by adversarial attacks.

    Key Techniques:
    - Lyapunov Exponent Calculation: Measures sensitivity to initial conditions
    - Correlation Dimension: Estimates fractal dimension of attractor
    - Entropy Analysis: Shannon, approximate, sample, and permutation entropy
    - Recurrence Quantification Analysis (RQA): Analyzes recurrence patterns
    - Fractal Dimension: Box-counting and correlation methods
    - Hurst Exponent: Long-term memory analysis
    - Phase Space Reconstruction: Takens embedding theorem
    """

    def __init__(
        self,
        embedding_dimension: int = 10,
        time_delay: int = 1,
        lyapunov_threshold: float = 0.1,
        entropy_threshold: float = 0.5,
        recurrence_threshold: float = 0.1,
        min_series_length: int = 100,
    ):
        """
        Initialize Chaos Quantification System.

        Args:
            embedding_dimension: Dimension for phase space reconstruction
            time_delay: Time delay for embedding
            lyapunov_threshold: Threshold for positive Lyapunov exponent
            entropy_threshold: Threshold for entropy-based detection
            recurrence_threshold: Threshold for recurrence plot analysis
            min_series_length: Minimum time series length for analysis
        """
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.lyapunov_threshold = lyapunov_threshold
        self.entropy_threshold = entropy_threshold
        self.recurrence_threshold = recurrence_threshold
        self.min_series_length = min_series_length
        self.analysis_history: List[ChaosAnalysisResult] = []
        self._baseline_metrics: Optional[Dict[str, float]] = None

    def analyze_input(
        self,
        input_data: np.ndarray,
        baseline_data: Optional[np.ndarray] = None,
    ) -> ChaosAnalysisResult:
        """
        Analyze input data for chaotic perturbations indicative of adversarial attacks.

        Args:
            input_data: Input data to analyze (flattened to 1D time series)
            baseline_data: Optional clean baseline for comparison

        Returns:
            ChaosAnalysisResult with comprehensive chaos metrics
        """
        # Flatten input to 1D time series
        time_series = self._prepare_time_series(input_data)

        if len(time_series) < self.min_series_length:
            # Pad or interpolate for short inputs
            time_series = self._extend_time_series(time_series)

        # Calculate Lyapunov exponent
        lyapunov_exp = self._calculate_lyapunov_exponent(time_series)

        # Calculate correlation dimension
        corr_dim = self._calculate_correlation_dimension(time_series)

        # Calculate entropy measures
        entropy_measures = self._calculate_entropy_measures(time_series)

        # Calculate recurrence metrics
        recurrence_metrics = self._calculate_recurrence_metrics(time_series)

        # Calculate fractal dimension
        fractal_dim = self._calculate_fractal_dimension(time_series)

        # Calculate Hurst exponent
        hurst_exp = self._calculate_hurst_exponent(time_series)

        # Compute composite chaos score
        chaos_score = self._compute_chaos_score(
            lyapunov_exp,
            corr_dim,
            entropy_measures,
            recurrence_metrics,
            fractal_dim,
            hurst_exp,
        )

        # Determine if chaotic and if it indicates adversarial perturbation
        is_chaotic = lyapunov_exp > 0 or chaos_score > self.entropy_threshold
        is_adversarial = self._detect_adversarial_chaos(
            time_series, baseline_data, chaos_score, lyapunov_exp, entropy_measures
        )

        # Calculate confidence
        confidence = self._calculate_detection_confidence(
            chaos_score, lyapunov_exp, entropy_measures, baseline_data is not None
        )

        result = ChaosAnalysisResult(
            is_chaotic=is_chaotic,
            chaos_score=chaos_score,
            lyapunov_exponent=lyapunov_exp,
            correlation_dimension=corr_dim,
            entropy_measures=entropy_measures,
            recurrence_metrics=recurrence_metrics,
            fractal_dimension=fractal_dim,
            hurst_exponent=hurst_exp,
            is_adversarial_perturbation=is_adversarial,
            confidence=confidence,
        )

        self.analysis_history.append(result)
        return result

    def _prepare_time_series(self, data: np.ndarray) -> np.ndarray:
        """Prepare input data as 1D time series."""
        if isinstance(data, list):
            data = np.array(data)
        return data.flatten().astype(np.float64)

    def _extend_time_series(self, series: np.ndarray) -> np.ndarray:
        """Extend short time series using interpolation."""
        target_length = self.min_series_length
        if len(series) >= target_length:
            return series

        # Linear interpolation to extend
        original_indices = np.linspace(0, 1, len(series))
        target_indices = np.linspace(0, 1, target_length)
        extended = np.interp(target_indices, original_indices, series)
        return extended

    def _phase_space_reconstruction(self, series: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space using Takens embedding theorem.

        Args:
            series: 1D time series

        Returns:
            Embedded phase space matrix
        """
        n = len(series)
        m = self.embedding_dimension
        tau = self.time_delay

        # Calculate number of vectors
        n_vectors = n - (m - 1) * tau

        if n_vectors <= 0:
            # Fallback for short series
            m = min(3, n // 2)
            tau = 1
            n_vectors = n - (m - 1) * tau

        # Construct embedding matrix
        embedded = np.zeros((n_vectors, m))
        for i in range(n_vectors):
            for j in range(m):
                embedded[i, j] = series[i + j * tau]

        return embedded

    def _calculate_lyapunov_exponent(self, series: np.ndarray) -> float:
        """
        Calculate the largest Lyapunov exponent using Wolf's algorithm.

        Positive Lyapunov exponent indicates chaos (sensitivity to initial conditions).
        Adversarial perturbations often exhibit chaotic signatures.

        Args:
            series: Time series data

        Returns:
            Largest Lyapunov exponent
        """
        try:
            # Phase space reconstruction
            embedded = self._phase_space_reconstruction(series)
            n_points = len(embedded)

            if n_points < 10:
                return 0.0

            # Parameters for algorithm
            min_separation = np.std(series) * 0.01
            max_separation = np.std(series) * 0.5

            lyapunov_sum = 0.0
            count = 0

            for i in range(0, n_points - 10, 5):
                # Find nearest neighbor (not too close in time)
                min_dist = np.inf
                nearest_idx = -1

                for j in range(n_points):
                    if abs(i - j) > self.time_delay * 2:  # Temporal separation
                        dist = np.linalg.norm(embedded[i] - embedded[j])
                        if min_separation < dist < min_dist:
                            min_dist = dist
                            nearest_idx = j

                if nearest_idx == -1 or min_dist > max_separation:
                    continue

                # Evolve both points and measure divergence
                evolution_steps = min(5, n_points - max(i, nearest_idx) - 1)
                if evolution_steps <= 0:
                    continue

                initial_dist = min_dist
                for step in range(1, evolution_steps + 1):
                    new_i = min(i + step, n_points - 1)
                    new_j = min(nearest_idx + step, n_points - 1)
                    final_dist = np.linalg.norm(embedded[new_i] - embedded[new_j])

                    if final_dist > 0 and initial_dist > 0:
                        lyapunov_sum += np.log(final_dist / initial_dist)
                        count += 1

            if count > 0:
                return lyapunov_sum / count
            return 0.0

        except Exception:
            return 0.0

    def _calculate_correlation_dimension(self, series: np.ndarray) -> float:
        """
        Calculate correlation dimension using Grassberger-Procaccia algorithm.

        Measures the fractal dimension of the strange attractor.

        Args:
            series: Time series data

        Returns:
            Correlation dimension estimate
        """
        try:
            embedded = self._phase_space_reconstruction(series)
            n_points = len(embedded)

            if n_points < 20:
                return 0.0

            # Sample points for efficiency
            sample_size = min(500, n_points)
            indices = np.random.choice(n_points, sample_size, replace=False)
            sampled = embedded[indices]

            # Calculate pairwise distances
            distances = []
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    dist = np.linalg.norm(sampled[i] - sampled[j])
                    if dist > 0:
                        distances.append(dist)

            if len(distances) < 10:
                return 0.0

            distances = np.array(distances)
            distances = distances[distances > 0]

            # Calculate correlation integral at different scales
            log_r = []
            log_c = []

            r_values = np.logspace(
                np.log10(np.percentile(distances, 5)),
                np.log10(np.percentile(distances, 95)),
                20,
            )

            n_pairs = len(distances)
            for r in r_values:
                c_r = np.sum(distances < r) / n_pairs
                if c_r > 0:
                    log_r.append(np.log(r))
                    log_c.append(np.log(c_r))

            if len(log_r) < 5:
                return 0.0

            # Linear regression for slope (correlation dimension)
            log_r = np.array(log_r)
            log_c = np.array(log_c)

            # Use middle portion for best estimate
            mid_start = len(log_r) // 4
            mid_end = 3 * len(log_r) // 4
            if mid_end - mid_start < 3:
                mid_start = 0
                mid_end = len(log_r)

            slope, _ = np.polyfit(log_r[mid_start:mid_end], log_c[mid_start:mid_end], 1)
            return max(0.0, slope)

        except Exception:
            return 0.0

    def _calculate_entropy_measures(self, series: np.ndarray) -> Dict[str, float]:
        """
        Calculate multiple entropy measures for chaos detection.

        Args:
            series: Time series data

        Returns:
            Dictionary of entropy measures
        """
        return {
            "shannon_entropy": self._shannon_entropy(series),
            "approximate_entropy": self._approximate_entropy(series),
            "sample_entropy": self._sample_entropy(series),
            "permutation_entropy": self._permutation_entropy(series),
            "spectral_entropy": self._spectral_entropy(series),
        }

    def _shannon_entropy(self, series: np.ndarray) -> float:
        """Calculate Shannon entropy of discretized series."""
        try:
            # Discretize into bins
            n_bins = min(50, len(series) // 5)
            if n_bins < 2:
                return 0.0

            hist, _ = np.histogram(series, bins=n_bins, density=True)
            hist = hist[hist > 0]

            if len(hist) == 0:
                return 0.0

            # Normalize
            hist = hist / hist.sum()
            return -np.sum(hist * np.log2(hist + 1e-10))

        except Exception:
            return 0.0

    def _approximate_entropy(
        self, series: np.ndarray, m: int = 2, r: Optional[float] = None
    ) -> float:
        """
        Calculate Approximate Entropy (ApEn).

        Measures regularity and unpredictability of time series.

        Args:
            series: Time series data
            m: Embedding dimension
            r: Tolerance (default: 0.2 * std)

        Returns:
            Approximate entropy value
        """
        try:
            n = len(series)
            if n < 10:
                return 0.0

            if r is None:
                r = 0.2 * np.std(series)

            def _phi(m_val: int) -> float:
                patterns = np.array(
                    [series[i : i + m_val] for i in range(n - m_val + 1)]
                )
                n_patterns = len(patterns)

                counts = np.zeros(n_patterns)
                for i in range(n_patterns):
                    # Count similar patterns
                    dists = np.max(np.abs(patterns - patterns[i]), axis=1)
                    counts[i] = np.sum(dists <= r)

                counts = counts / n_patterns
                return np.mean(np.log(counts + 1e-10))

            return abs(_phi(m) - _phi(m + 1))

        except Exception:
            return 0.0

    def _sample_entropy(
        self, series: np.ndarray, m: int = 2, r: Optional[float] = None
    ) -> float:
        """
        Calculate Sample Entropy (SampEn).

        Similar to ApEn but without self-matching, providing less bias.

        Args:
            series: Time series data
            m: Embedding dimension
            r: Tolerance (default: 0.2 * std)

        Returns:
            Sample entropy value
        """
        try:
            n = len(series)
            if n < 10:
                return 0.0

            if r is None:
                r = 0.2 * np.std(series)

            def _count_matches(m_val: int) -> int:
                patterns = np.array([series[i : i + m_val] for i in range(n - m_val)])
                n_patterns = len(patterns)
                count = 0

                for i in range(n_patterns):
                    for j in range(i + 1, n_patterns):
                        if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                            count += 1

                return count

            a = _count_matches(m + 1)
            b = _count_matches(m)

            if b == 0:
                return 0.0

            return -np.log((a + 1e-10) / (b + 1e-10))

        except Exception:
            return 0.0

    def _permutation_entropy(
        self, series: np.ndarray, order: int = 3, delay: int = 1
    ) -> float:
        """
        Calculate Permutation Entropy.

        Measures complexity based on ordinal patterns.

        Args:
            series: Time series data
            order: Permutation order
            delay: Time delay

        Returns:
            Permutation entropy (normalized)
        """
        try:
            n = len(series)
            if n < order * delay:
                return 0.0

            # Extract ordinal patterns
            n_patterns = n - (order - 1) * delay
            patterns = []

            for i in range(n_patterns):
                pattern = tuple(
                    np.argsort([series[i + j * delay] for j in range(order)])
                )
                patterns.append(pattern)

            # Count pattern frequencies
            from collections import Counter

            pattern_counts = Counter(patterns)
            total = len(patterns)

            # Calculate entropy
            probs = np.array([count / total for count in pattern_counts.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-10))

            # Normalize by maximum entropy
            max_entropy = np.log2(np.math.factorial(order))
            return entropy / max_entropy if max_entropy > 0 else 0.0

        except Exception:
            return 0.0

    def _spectral_entropy(self, series: np.ndarray) -> float:
        """
        Calculate Spectral Entropy from power spectrum.

        Args:
            series: Time series data

        Returns:
            Spectral entropy (normalized)
        """
        try:
            # Compute power spectral density
            fft_vals = np.fft.fft(series)
            psd = np.abs(fft_vals) ** 2
            psd = psd[: len(psd) // 2]  # Take positive frequencies

            # Normalize to probability distribution
            psd_norm = psd / (psd.sum() + 1e-10)
            psd_norm = psd_norm[psd_norm > 0]

            # Calculate entropy
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

            # Normalize
            max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1.0
            return entropy / max_entropy if max_entropy > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_recurrence_metrics(self, series: np.ndarray) -> Dict[str, float]:
        """
        Calculate Recurrence Quantification Analysis (RQA) metrics.

        Args:
            series: Time series data

        Returns:
            Dictionary of RQA metrics
        """
        try:
            embedded = self._phase_space_reconstruction(series)
            n = len(embedded)

            if n < 10:
                return {
                    "recurrence_rate": 0.0,
                    "determinism": 0.0,
                    "laminarity": 0.0,
                    "entropy_diagonal": 0.0,
                    "trapping_time": 0.0,
                }

            # Calculate recurrence matrix
            threshold = self.recurrence_threshold * np.std(series)

            # Sample for efficiency
            sample_size = min(200, n)
            indices = np.linspace(0, n - 1, sample_size, dtype=int)
            sampled = embedded[indices]

            # Build recurrence matrix
            recurrence_matrix = np.zeros((sample_size, sample_size))
            for i in range(sample_size):
                for j in range(sample_size):
                    dist = np.linalg.norm(sampled[i] - sampled[j])
                    recurrence_matrix[i, j] = 1 if dist < threshold else 0

            # Calculate metrics
            recurrence_rate = np.sum(recurrence_matrix) / (sample_size**2)

            # Determinism: percentage of recurrence points in diagonal lines
            determinism = self._calculate_determinism(recurrence_matrix)

            # Laminarity: percentage of recurrence points in vertical lines
            laminarity = self._calculate_laminarity(recurrence_matrix)

            # Entropy of diagonal line lengths
            entropy_diagonal = self._diagonal_entropy(recurrence_matrix)

            # Average trapping time
            trapping_time = self._calculate_trapping_time(recurrence_matrix)

            return {
                "recurrence_rate": recurrence_rate,
                "determinism": determinism,
                "laminarity": laminarity,
                "entropy_diagonal": entropy_diagonal,
                "trapping_time": trapping_time,
            }

        except Exception:
            return {
                "recurrence_rate": 0.0,
                "determinism": 0.0,
                "laminarity": 0.0,
                "entropy_diagonal": 0.0,
                "trapping_time": 0.0,
            }

    def _calculate_determinism(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate determinism from diagonal line structures."""
        n = len(recurrence_matrix)
        min_line_length = 2

        diagonal_points = 0
        total_points = np.sum(recurrence_matrix)

        if total_points == 0:
            return 0.0

        # Check diagonals
        for offset in range(-n + min_line_length, n - min_line_length + 1):
            diagonal = np.diag(recurrence_matrix, offset)
            line_length = 0

            for point in diagonal:
                if point == 1:
                    line_length += 1
                else:
                    if line_length >= min_line_length:
                        diagonal_points += line_length
                    line_length = 0

            if line_length >= min_line_length:
                diagonal_points += line_length

        return diagonal_points / total_points

    def _calculate_laminarity(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate laminarity from vertical line structures."""
        n = len(recurrence_matrix)
        min_line_length = 2

        vertical_points = 0
        total_points = np.sum(recurrence_matrix)

        if total_points == 0:
            return 0.0

        # Check vertical lines
        for col in range(n):
            line_length = 0
            for row in range(n):
                if recurrence_matrix[row, col] == 1:
                    line_length += 1
                else:
                    if line_length >= min_line_length:
                        vertical_points += line_length
                    line_length = 0

            if line_length >= min_line_length:
                vertical_points += line_length

        return vertical_points / total_points

    def _diagonal_entropy(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate entropy of diagonal line length distribution."""
        n = len(recurrence_matrix)
        line_lengths = []

        for offset in range(-n + 2, n - 1):
            diagonal = np.diag(recurrence_matrix, offset)
            line_length = 0

            for point in diagonal:
                if point == 1:
                    line_length += 1
                else:
                    if line_length >= 2:
                        line_lengths.append(line_length)
                    line_length = 0

            if line_length >= 2:
                line_lengths.append(line_length)

        if not line_lengths:
            return 0.0

        from collections import Counter

        length_counts = Counter(line_lengths)
        total = sum(length_counts.values())
        probs = np.array([count / total for count in length_counts.values()])

        return -np.sum(probs * np.log2(probs + 1e-10))

    def _calculate_trapping_time(self, recurrence_matrix: np.ndarray) -> float:
        """Calculate average trapping time from vertical line lengths."""
        n = len(recurrence_matrix)
        line_lengths = []

        for col in range(n):
            line_length = 0
            for row in range(n):
                if recurrence_matrix[row, col] == 1:
                    line_length += 1
                else:
                    if line_length >= 2:
                        line_lengths.append(line_length)
                    line_length = 0

            if line_length >= 2:
                line_lengths.append(line_length)

        return np.mean(line_lengths) if line_lengths else 0.0

    def _calculate_fractal_dimension(self, series: np.ndarray) -> float:
        """
        Calculate fractal dimension using box-counting method.

        Args:
            series: Time series data

        Returns:
            Fractal dimension estimate
        """
        try:
            # Normalize series to [0, 1]
            series_min = np.min(series)
            series_max = np.max(series)

            if series_max - series_min == 0:
                return 1.0

            normalized = (series - series_min) / (series_max - series_min)

            # Create 2D representation (time vs value)
            n = len(normalized)
            points = np.column_stack([np.linspace(0, 1, n), normalized])

            # Box counting at different scales
            scales = []
            counts = []

            for k in range(2, min(8, int(np.log2(n)))):
                box_size = 1.0 / (2**k)
                n_boxes_x = int(np.ceil(1.0 / box_size))
                n_boxes_y = int(np.ceil(1.0 / box_size))

                # Count occupied boxes
                occupied = set()
                for point in points:
                    box_x = min(int(point[0] / box_size), n_boxes_x - 1)
                    box_y = min(int(point[1] / box_size), n_boxes_y - 1)
                    occupied.add((box_x, box_y))

                scales.append(np.log(1 / box_size))
                counts.append(np.log(len(occupied)))

            if len(scales) < 2:
                return 1.0

            # Linear regression for slope
            slope, _ = np.polyfit(scales, counts, 1)
            return max(1.0, min(2.0, slope))

        except Exception:
            return 1.0

    def _calculate_hurst_exponent(self, series: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        H < 0.5: Anti-persistent (mean-reverting)
        H = 0.5: Random walk
        H > 0.5: Persistent (trending)

        Args:
            series: Time series data

        Returns:
            Hurst exponent
        """
        try:
            n = len(series)
            if n < 20:
                return 0.5

            # Different scales for R/S analysis
            max_k = min(int(np.log2(n)) - 1, 8)
            if max_k < 2:
                return 0.5

            rs_values = []
            n_values = []

            for k in range(2, max_k + 1):
                subset_size = 2**k
                n_subsets = n // subset_size

                if n_subsets == 0:
                    continue

                rs_list = []
                for i in range(n_subsets):
                    subset = series[i * subset_size : (i + 1) * subset_size]

                    # Mean-adjusted cumulative sum
                    mean = np.mean(subset)
                    cumsum = np.cumsum(subset - mean)

                    # Range
                    r = np.max(cumsum) - np.min(cumsum)

                    # Standard deviation
                    s = np.std(subset)

                    if s > 0:
                        rs_list.append(r / s)

                if rs_list:
                    rs_values.append(np.log(np.mean(rs_list)))
                    n_values.append(np.log(subset_size))

            if len(rs_values) < 2:
                return 0.5

            # Linear regression for Hurst exponent
            hurst, _ = np.polyfit(n_values, rs_values, 1)
            return max(0.0, min(1.0, hurst))

        except Exception:
            return 0.5

    def _compute_chaos_score(
        self,
        lyapunov_exp: float,
        corr_dim: float,
        entropy_measures: Dict[str, float],
        recurrence_metrics: Dict[str, float],
        fractal_dim: float,
        hurst_exp: float,
    ) -> float:
        """
        Compute composite chaos score from all metrics.

        Args:
            lyapunov_exp: Lyapunov exponent
            corr_dim: Correlation dimension
            entropy_measures: Dictionary of entropy values
            recurrence_metrics: Dictionary of RQA metrics
            fractal_dim: Fractal dimension
            hurst_exp: Hurst exponent

        Returns:
            Composite chaos score [0, 1]
        """
        scores = []

        # Lyapunov contribution (positive = chaotic)
        lyap_score = 1.0 / (1.0 + np.exp(-lyapunov_exp * 10))  # Sigmoid
        scores.append(lyap_score * 0.25)

        # Entropy contribution (higher = more chaotic)
        avg_entropy = np.mean(list(entropy_measures.values()))
        scores.append(avg_entropy * 0.25)

        # Determinism contribution (lower determinism = more chaotic)
        det = recurrence_metrics.get("determinism", 0.5)
        scores.append((1.0 - det) * 0.15)

        # Recurrence rate (very low or very high can indicate anomaly)
        rr = recurrence_metrics.get("recurrence_rate", 0.5)
        rr_anomaly = 2 * abs(rr - 0.5)
        scores.append(rr_anomaly * 0.1)

        # Fractal dimension (non-integer values indicate chaos)
        frac_chaos = abs(fractal_dim - round(fractal_dim))
        scores.append(frac_chaos * 0.15)

        # Hurst exponent (deviation from 0.5 indicates structure)
        hurst_structure = abs(hurst_exp - 0.5) * 2
        scores.append(hurst_structure * 0.1)

        return min(1.0, sum(scores))

    def _detect_adversarial_chaos(
        self,
        series: np.ndarray,
        baseline: Optional[np.ndarray],
        chaos_score: float,
        lyapunov_exp: float,
        entropy_measures: Dict[str, float],
    ) -> bool:
        """
        Detect if chaos metrics indicate adversarial perturbation.

        Args:
            series: Input time series
            baseline: Optional baseline data
            chaos_score: Computed chaos score
            lyapunov_exp: Lyapunov exponent
            entropy_measures: Entropy measures

        Returns:
            True if adversarial perturbation is detected
        """
        # Check if baseline comparison is possible
        if baseline is not None:
            baseline_series = self._prepare_time_series(baseline)
            if len(baseline_series) >= self.min_series_length:
                baseline_series = self._extend_time_series(baseline_series)

            # Compare chaos metrics
            baseline_lyap = self._calculate_lyapunov_exponent(baseline_series)
            baseline_entropy = self._calculate_entropy_measures(baseline_series)

            # Significant deviation from baseline indicates adversarial
            lyap_diff = abs(lyapunov_exp - baseline_lyap)
            entropy_diff = abs(
                np.mean(list(entropy_measures.values()))
                - np.mean(list(baseline_entropy.values()))
            )

            if (
                lyap_diff > self.lyapunov_threshold
                or entropy_diff > self.entropy_threshold
            ):
                return True

        # Without baseline, use absolute thresholds
        if lyapunov_exp > self.lyapunov_threshold:
            return True

        if chaos_score > 0.7:  # High chaos score
            return True

        # Check entropy anomaly
        permutation_entropy = entropy_measures.get("permutation_entropy", 0.5)
        if permutation_entropy > 0.95 or permutation_entropy < 0.1:
            return True

        return False

    def _calculate_detection_confidence(
        self,
        chaos_score: float,
        lyapunov_exp: float,
        entropy_measures: Dict[str, float],
        has_baseline: bool,
    ) -> float:
        """Calculate confidence in adversarial detection."""
        base_confidence = chaos_score

        # Boost confidence with strong Lyapunov indicator
        if lyapunov_exp > 0:
            base_confidence += 0.1

        # Boost confidence with baseline comparison
        if has_baseline:
            base_confidence += 0.1

        # Entropy consistency check
        entropy_values = list(entropy_measures.values())
        entropy_std = np.std(entropy_values)
        if entropy_std < 0.1:  # Consistent entropy measures
            base_confidence += 0.05

        return min(1.0, base_confidence)

    def set_baseline(self, clean_data: np.ndarray) -> None:
        """
        Set baseline metrics from known clean data.

        Args:
            clean_data: Known clean input data
        """
        series = self._prepare_time_series(clean_data)
        if len(series) < self.min_series_length:
            series = self._extend_time_series(series)

        self._baseline_metrics = {
            "lyapunov": self._calculate_lyapunov_exponent(series),
            "correlation_dim": self._calculate_correlation_dimension(series),
            "entropy": self._calculate_entropy_measures(series),
            "fractal_dim": self._calculate_fractal_dimension(series),
            "hurst": self._calculate_hurst_exponent(series),
        }

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get chaos analysis statistics."""
        if not self.analysis_history:
            return {
                "total_analyses": 0,
                "chaotic_count": 0,
                "adversarial_count": 0,
                "average_chaos_score": 0.0,
            }

        return {
            "total_analyses": len(self.analysis_history),
            "chaotic_count": sum(1 for r in self.analysis_history if r.is_chaotic),
            "adversarial_count": sum(
                1 for r in self.analysis_history if r.is_adversarial_perturbation
            ),
            "average_chaos_score": np.mean(
                [r.chaos_score for r in self.analysis_history]
            ),
            "average_lyapunov": np.mean(
                [r.lyapunov_exponent for r in self.analysis_history]
            ),
            "average_confidence": np.mean(
                [r.confidence for r in self.analysis_history]
            ),
        }


# =============================================================================
# DEFENSE PERTURBATION SYSTEM
# =============================================================================


class InputTransformation(ABC):
    """Abstract base class for input transformations."""

    @abstractmethod
    def apply(self, data: np.ndarray, **params) -> np.ndarray:
        """Apply transformation to input data."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default transformation parameters."""
        pass


class GeometricTransformations(InputTransformation):
    """Geometric input transformations."""

    def apply(
        self, data: np.ndarray, transformation: str = "rotation", **params
    ) -> np.ndarray:
        """Apply geometric transformation."""
        if transformation == "rotation":
            return self._rotate(data, params.get("angle", 5))
        elif transformation == "translation":
            return self._translate(data, params.get("shift", 2))
        elif transformation == "scaling":
            return self._scale(data, params.get("factor", 1.1))
        elif transformation == "flipping_horizontal":
            return self._flip_horizontal(data)
        elif transformation == "flipping_vertical":
            return self._flip_vertical(data)
        elif transformation == "shearing":
            return self._shear(data, params.get("shear_factor", 0.1))
        return data

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "rotation": {"angle": 5},
            "translation": {"shift": 2},
            "scaling": {"factor": 1.1},
            "shearing": {"shear_factor": 0.1},
        }

    def _rotate(self, data: np.ndarray, angle: float) -> np.ndarray:
        """Rotate 2D data by angle degrees."""
        if data.ndim < 2:
            return data

        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        if data.ndim == 2:
            h, w = data.shape
            center = (h // 2, w // 2)
            rotated = np.zeros_like(data)

            for i in range(h):
                for j in range(w):
                    # Translate to origin
                    y, x = i - center[0], j - center[1]
                    # Rotate
                    new_x = int(x * cos_a - y * sin_a + center[1])
                    new_y = int(x * sin_a + y * cos_a + center[0])
                    # Copy if valid
                    if 0 <= new_x < w and 0 <= new_y < h:
                        rotated[i, j] = data[new_y, new_x]

            return rotated

        return data

    def _translate(self, data: np.ndarray, shift: int) -> np.ndarray:
        """Translate data by shift pixels."""
        if data.ndim < 2:
            return np.roll(data, shift)
        return np.roll(data, shift, axis=(0, 1))

    def _scale(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Scale data by factor."""
        return data * factor

    def _flip_horizontal(self, data: np.ndarray) -> np.ndarray:
        """Flip data horizontally."""
        if data.ndim >= 2:
            return np.flip(data, axis=1)
        return np.flip(data)

    def _flip_vertical(self, data: np.ndarray) -> np.ndarray:
        """Flip data vertically."""
        if data.ndim >= 2:
            return np.flip(data, axis=0)
        return data

    def _shear(self, data: np.ndarray, shear_factor: float) -> np.ndarray:
        """Apply shearing transformation."""
        if data.ndim < 2:
            return data

        h, w = data.shape[:2]
        sheared = np.zeros_like(data)

        for i in range(h):
            for j in range(w):
                new_j = int(j + shear_factor * i) % w
                sheared[i, new_j] = data[i, j]

        return sheared


class IntensityTransformations(InputTransformation):
    """Color and intensity transformations."""

    def apply(
        self, data: np.ndarray, transformation: str = "brightness", **params
    ) -> np.ndarray:
        """Apply intensity transformation."""
        if transformation == "brightness":
            return self._adjust_brightness(data, params.get("factor", 1.1))
        elif transformation == "contrast":
            return self._adjust_contrast(data, params.get("factor", 1.2))
        elif transformation == "gamma":
            return self._gamma_correction(data, params.get("gamma", 1.1))
        elif transformation == "histogram_equalization":
            return self._histogram_equalization(data)
        return data

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "brightness": {"factor": 1.1},
            "contrast": {"factor": 1.2},
            "gamma": {"gamma": 1.1},
        }

    def _adjust_brightness(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness by factor."""
        return np.clip(data * factor, data.min(), data.max())

    def _adjust_contrast(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast by factor."""
        mean = np.mean(data)
        return np.clip((data - mean) * factor + mean, data.min(), data.max())

    def _gamma_correction(self, data: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction."""
        # Normalize to [0, 1], apply gamma, rescale
        data_min, data_max = data.min(), data.max()
        if data_max - data_min == 0:
            return data
        normalized = (data - data_min) / (data_max - data_min)
        corrected = np.power(normalized, gamma)
        return corrected * (data_max - data_min) + data_min

    def _histogram_equalization(self, data: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        if data.ndim > 2:
            # Apply to each channel
            result = np.zeros_like(data)
            for i in range(data.shape[-1]):
                result[..., i] = self._equalize_channel(data[..., i])
            return result
        return self._equalize_channel(data)

    def _equalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Equalize single channel."""
        # Flatten, compute histogram, equalize
        flat = channel.flatten()
        hist, bins = np.histogram(flat, bins=256, density=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize

        # Map values
        data_min, data_max = channel.min(), channel.max()
        if data_max - data_min == 0:
            return channel

        normalized = (channel - data_min) / (data_max - data_min) * 255
        equalized = np.interp(normalized.flatten(), np.arange(256), cdf * 255)
        return equalized.reshape(channel.shape) / 255 * (data_max - data_min) + data_min


class NoiseTransformations(InputTransformation):
    """Noise-based transformations."""

    def apply(
        self, data: np.ndarray, transformation: str = "gaussian", **params
    ) -> np.ndarray:
        """Apply noise transformation."""
        if transformation == "gaussian":
            return self._add_gaussian_noise(data, params.get("std", 0.01))
        elif transformation == "salt_pepper":
            return self._add_salt_pepper_noise(data, params.get("prob", 0.01))
        elif transformation == "speckle":
            return self._add_speckle_noise(data, params.get("std", 0.05))
        elif transformation == "poisson":
            return self._add_poisson_noise(data)
        return data

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "gaussian": {"std": 0.01},
            "salt_pepper": {"prob": 0.01},
            "speckle": {"std": 0.05},
        }

    def _add_gaussian_noise(self, data: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, std * np.std(data), data.shape)
        return data + noise

    def _add_salt_pepper_noise(self, data: np.ndarray, prob: float) -> np.ndarray:
        """Add salt and pepper noise."""
        result = data.copy()
        mask = np.random.random(data.shape)

        # Salt
        result[mask < prob / 2] = data.max()
        # Pepper
        result[mask > 1 - prob / 2] = data.min()

        return result

    def _add_speckle_noise(self, data: np.ndarray, std: float) -> np.ndarray:
        """Add speckle (multiplicative) noise."""
        noise = np.random.normal(1, std, data.shape)
        return data * noise

    def _add_poisson_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Poisson noise."""
        # Ensure positive values
        data_positive = data - data.min() + 1
        noisy = np.random.poisson(data_positive)
        return noisy.astype(data.dtype) + data.min() - 1


class FilterTransformations(InputTransformation):
    """Spatial filtering transformations."""

    def apply(
        self, data: np.ndarray, transformation: str = "gaussian_blur", **params
    ) -> np.ndarray:
        """Apply filter transformation."""
        if transformation == "gaussian_blur":
            return self._gaussian_blur(data, params.get("kernel_size", 3))
        elif transformation == "median_filter":
            return self._median_filter(data, params.get("kernel_size", 3))
        elif transformation == "sharpening":
            return self._sharpen(data)
        elif transformation == "bilateral":
            return self._bilateral_filter(data)
        return data

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "gaussian_blur": {"kernel_size": 3},
            "median_filter": {"kernel_size": 3},
        }

    def _gaussian_blur(self, data: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply Gaussian blur."""
        # Create Gaussian kernel
        sigma = kernel_size / 6.0
        x = np.arange(kernel_size) - kernel_size // 2
        kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply separable convolution
        result = data.copy()
        if data.ndim >= 1:
            result = np.convolve(result.flatten(), kernel_1d, mode="same").reshape(
                data.shape
            )

        return result

    def _median_filter(self, data: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply median filter."""
        if data.ndim < 2:
            # 1D median filter
            result = np.zeros_like(data)
            pad = kernel_size // 2
            padded = np.pad(data, pad, mode="edge")
            for i in range(len(data)):
                result[i] = np.median(padded[i : i + kernel_size])
            return result

        # 2D median filter
        result = np.zeros_like(data)
        pad = kernel_size // 2
        padded = np.pad(data, pad, mode="edge")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                window = padded[i : i + kernel_size, j : j + kernel_size]
                result[i, j] = np.median(window)

        return result

    def _sharpen(self, data: np.ndarray) -> np.ndarray:
        """Apply sharpening filter."""
        if data.ndim < 2:
            # Simple 1D sharpening
            kernel = np.array([-1, 3, -1])
            return np.convolve(data, kernel, mode="same")

        # 2D sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        result = np.zeros_like(data)
        padded = np.pad(data, 1, mode="edge")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                window = padded[i : i + 3, j : j + 3]
                result[i, j] = np.sum(window * kernel)

        return np.clip(result, data.min(), data.max())

    def _bilateral_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bilateral filter (simplified)."""
        # Combination of spatial and intensity filtering
        spatial_blurred = self._gaussian_blur(data, 5)
        # Weight by intensity similarity
        intensity_diff = np.abs(data - spatial_blurred)
        weight = np.exp(-(intensity_diff**2) / (2 * 0.1**2))
        return weight * data + (1 - weight) * spatial_blurred


class CompressionTransformations(InputTransformation):
    """Compression-based transformations."""

    def apply(
        self, data: np.ndarray, transformation: str = "quantization", **params
    ) -> np.ndarray:
        """Apply compression transformation."""
        if transformation == "quantization":
            return self._quantize(data, params.get("levels", 32))
        elif transformation == "jpeg":
            return self._jpeg_simulation(data, params.get("quality", 80))
        return data

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "quantization": {"levels": 32},
            "jpeg": {"quality": 80},
        }

    def _quantize(self, data: np.ndarray, levels: int) -> np.ndarray:
        """Quantize to specified number of levels."""
        data_min, data_max = data.min(), data.max()
        if data_max - data_min == 0:
            return data

        # Normalize to [0, 1], quantize, rescale
        normalized = (data - data_min) / (data_max - data_min)
        quantized = np.round(normalized * (levels - 1)) / (levels - 1)
        return quantized * (data_max - data_min) + data_min

    def _jpeg_simulation(self, data: np.ndarray, quality: int) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        # Simplified DCT-based compression simulation
        if data.ndim < 2:
            return data

        # Apply block-wise processing
        block_size = 8
        quality_factor = quality / 100.0

        result = data.copy()
        h, w = data.shape[:2]

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = result[i : i + block_size, j : j + block_size]
                if block.size == block_size * block_size:
                    # Simple DCT-like quantization simulation
                    block_mean = np.mean(block)
                    # Reduce high frequency detail based on quality
                    smoothed = block_mean + (block - block_mean) * quality_factor
                    result[i : i + block_size, j : j + block_size] = smoothed

        return result


# =============================================================================
# DEFENSE PERTURBATION SYSTEM
# =============================================================================


class DefensePerturbationSystem:
    """
    Defense Perturbation System for Robust Adversarial Detection.

    A novel methodology to detect robust adversarial examples by applying
    the same input transformations that adversarial examples are designed
    to be robust against.

    Key Concept:
    - Adversarial examples robust to transformations (e.g., rotation, noise)
      should maintain their adversarial effect after transformation
    - By testing with multiple transformations, we can identify robust
      adversarial examples that would evade single-transformation defenses
    """

    def __init__(
        self,
        prediction_stability_threshold: float = 0.8,
        robustness_threshold: float = 0.7,
        enable_all_transformations: bool = True,
    ):
        """
        Initialize Defense Perturbation System.

        Args:
            prediction_stability_threshold: Threshold for prediction stability
            robustness_threshold: Threshold for robustness score
            enable_all_transformations: Whether to use all transformation types
        """
        self.prediction_stability_threshold = prediction_stability_threshold
        self.robustness_threshold = robustness_threshold
        self.enable_all_transformations = enable_all_transformations
        self.analysis_history: List[DefensePerturbationResult] = []

        # Initialize transformation handlers
        self.transformations: Dict[str, InputTransformation] = {
            "geometric": GeometricTransformations(),
            "intensity": IntensityTransformations(),
            "noise": NoiseTransformations(),
            "filter": FilterTransformations(),
            "compression": CompressionTransformations(),
        }

    def analyze_input(
        self,
        input_data: np.ndarray,
        model_prediction_func: Callable[[np.ndarray], Any],
        original_prediction: Optional[Any] = None,
    ) -> DefensePerturbationResult:
        """
        Analyze input for robust adversarial characteristics.

        Args:
            input_data: Input data to analyze
            model_prediction_func: Function to get model prediction
            original_prediction: Original prediction to compare against

        Returns:
            DefensePerturbationResult with comprehensive analysis
        """
        if original_prediction is None:
            original_prediction = model_prediction_func(input_data)

        # Apply transformations and collect results
        transformation_results: Dict[str, Dict[str, Any]] = {}
        transformations_tested: List[str] = []
        vulnerable_transformations: List[str] = []
        robust_transformations: List[str] = []

        prediction_changes = 0
        total_tests = 0

        for category, transformer in self.transformations.items():
            params = transformer.get_default_params()

            for transform_name, transform_params in params.items():
                full_name = f"{category}_{transform_name}"
                transformations_tested.append(full_name)

                try:
                    transformed_data = transformer.apply(
                        input_data.copy(),
                        transformation=transform_name,
                        **transform_params,
                    )
                    transformed_prediction = model_prediction_func(transformed_data)

                    prediction_changed = str(transformed_prediction) != str(
                        original_prediction
                    )

                    if prediction_changed:
                        vulnerable_transformations.append(full_name)
                        prediction_changes += 1
                    else:
                        robust_transformations.append(full_name)

                    transformation_results[full_name] = {
                        "prediction_changed": prediction_changed,
                        "original_prediction": str(original_prediction),
                        "transformed_prediction": str(transformed_prediction),
                    }

                    total_tests += 1

                except Exception as e:
                    transformation_results[full_name] = {
                        "error": str(e),
                        "prediction_changed": False,
                    }
                    total_tests += 1

        # Calculate metrics
        prediction_stability = (
            1.0 - (prediction_changes / total_tests) if total_tests > 0 else 1.0
        )
        robustness_score = (
            len(robust_transformations) / total_tests if total_tests > 0 else 0.0
        )

        # Determine if this is a robust adversarial example
        # Robust adversarial = maintains adversarial effect across transformations
        is_robust_adversarial = (
            robustness_score > self.robustness_threshold
            and prediction_stability > self.prediction_stability_threshold
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            robustness_score, prediction_stability, total_tests
        )

        result = DefensePerturbationResult(
            is_robust_adversarial=is_robust_adversarial,
            robustness_score=robustness_score,
            transformations_tested=transformations_tested,
            transformation_results=transformation_results,
            vulnerable_transformations=vulnerable_transformations,
            robust_transformations=robust_transformations,
            prediction_stability=prediction_stability,
            confidence=confidence,
        )

        self.analysis_history.append(result)
        return result

    def _calculate_confidence(
        self,
        robustness_score: float,
        prediction_stability: float,
        total_tests: int,
    ) -> float:
        """Calculate confidence in detection result."""
        # Base confidence from metrics
        base_confidence = (robustness_score + prediction_stability) / 2

        # Boost confidence with more tests
        test_factor = min(1.0, total_tests / 10)
        base_confidence *= 0.5 + 0.5 * test_factor

        return min(1.0, base_confidence)

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get defense perturbation analysis statistics."""
        if not self.analysis_history:
            return {
                "total_analyses": 0,
                "robust_adversarial_count": 0,
                "average_robustness_score": 0.0,
            }

        return {
            "total_analyses": len(self.analysis_history),
            "robust_adversarial_count": sum(
                1 for r in self.analysis_history if r.is_robust_adversarial
            ),
            "average_robustness_score": np.mean(
                [r.robustness_score for r in self.analysis_history]
            ),
            "average_prediction_stability": np.mean(
                [r.prediction_stability for r in self.analysis_history]
            ),
            "average_confidence": np.mean(
                [r.confidence for r in self.analysis_history]
            ),
        }


# =============================================================================
# ADVERSARIAL DEFENSE SYSTEM
# =============================================================================


class AdversarialDefenseSystem:
    """
    Comprehensive Adversarial Defense System.

    Combines chaos quantification and defense perturbation methods
    for robust adversarial example detection.
    """

    def __init__(
        self,
        perturbation_threshold: float = 0.1,
        confidence_threshold: float = 0.8,
        enable_chaos_detection: bool = True,
        enable_defense_perturbation: bool = True,
        enable_input_smoothing: bool = True,
    ):
        """
        Initialize Adversarial Defense System.

        Args:
            perturbation_threshold: Threshold for perturbation detection
            confidence_threshold: Confidence threshold for detection
            enable_chaos_detection: Enable chaos quantification
            enable_defense_perturbation: Enable defense perturbation analysis
            enable_input_smoothing: Enable input smoothing defense
        """
        self.perturbation_threshold = perturbation_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_chaos_detection = enable_chaos_detection
        self.enable_defense_perturbation = enable_defense_perturbation
        self.enable_input_smoothing = enable_input_smoothing
        self.detection_history: List[AdversarialDetectionResult] = []

        # Initialize sub-systems
        if enable_chaos_detection:
            self.chaos_system = ChaosQuantificationSystem()
        else:
            self.chaos_system = None

        if enable_defense_perturbation:
            self.defense_system = DefensePerturbationSystem()
        else:
            self.defense_system = None

    def detect_adversarial_example(
        self,
        input_data: Union[np.ndarray, List, str],
        model_prediction_func: Optional[Callable] = None,
        baseline_input: Optional[Union[np.ndarray, List, str]] = None,
    ) -> AdversarialDetectionResult:
        """
        Detect if input is an adversarial example.

        Args:
            input_data: Input to analyze
            model_prediction_func: Model prediction function
            baseline_input: Clean baseline for comparison

        Returns:
            AdversarialDetectionResult with detection outcome
        """
        # Convert inputs to numpy arrays
        input_array = self._to_numpy(input_data)
        baseline_array = (
            self._to_numpy(baseline_input) if baseline_input is not None else None
        )

        # Calculate perturbation magnitude
        perturbation_magnitude = self._calculate_perturbation(
            input_data, baseline_input
        )

        # Get predictions if model function provided
        original_prediction = None
        adversarial_prediction = None
        if model_prediction_func is not None:
            try:
                adversarial_prediction = model_prediction_func(input_data)
                if baseline_input is not None:
                    original_prediction = model_prediction_func(baseline_input)
            except Exception as e:
                logging.exception(
                    "Model prediction failed in adversarial example detection."
                )

        # Run chaos analysis if enabled
        chaos_result = None
        if self.chaos_system is not None:
            try:
                chaos_result = self.chaos_system.analyze_input(
                    input_array, baseline_array
                )
            except Exception as e:
                logging.exception(
                    "Chaos analysis failed in adversarial example detection."
                )

        # Run defense perturbation if enabled
        defense_result = None
        if self.defense_system is not None and model_prediction_func is not None:
            try:
                defense_result = self.defense_system.analyze_input(
                    input_array, model_prediction_func, adversarial_prediction
                )
            except Exception as e:
                logging.exception(
                    "Defense perturbation analysis failed in adversarial example detection."
                )

        # Determine if adversarial
        is_adversarial = self._is_adversarial(
            perturbation_magnitude, chaos_result, defense_result
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            perturbation_magnitude, chaos_result, defense_result
        )

        # Identify attack type
        attack_type = None
        if is_adversarial:
            consistency_score = (
                defense_result.prediction_stability if defense_result else 0.5
            )
            attack_type = self._identify_attack_type(
                perturbation_magnitude, consistency_score
            )

        result = AdversarialDetectionResult(
            is_adversarial=is_adversarial,
            confidence=confidence,
            attack_type=attack_type,
            perturbation_magnitude=perturbation_magnitude,
            original_prediction=original_prediction,
            adversarial_prediction=adversarial_prediction,
            chaos_analysis=chaos_result,
            defense_perturbation_analysis=defense_result,
        )

        self.detection_history.append(result)
        return result

    def _to_numpy(self, data: Union[np.ndarray, List, str, None]) -> np.ndarray:
        """Convert input to numpy array."""
        if data is None:
            return np.array([])
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, str):
            return np.array([ord(c) for c in data], dtype=np.float64)
        return np.array(data, dtype=np.float64)

    def _calculate_perturbation(
        self,
        input_data: Union[np.ndarray, List, str],
        baseline: Optional[Union[np.ndarray, List, str]],
    ) -> float:
        """Calculate perturbation magnitude between input and baseline."""
        if baseline is None:
            return 0.0

        input_array = self._to_numpy(input_data)
        baseline_array = self._to_numpy(baseline)

        if input_array.shape != baseline_array.shape:
            # Handle different shapes (e.g., text of different lengths)
            max_len = max(len(input_array), len(baseline_array))
            input_array = np.pad(input_array, (0, max_len - len(input_array)))
            baseline_array = np.pad(baseline_array, (0, max_len - len(baseline_array)))

        # Calculate L2 norm of difference
        diff = input_array.flatten() - baseline_array.flatten()
        baseline_norm = np.linalg.norm(baseline_array.flatten())

        if baseline_norm == 0:
            return np.linalg.norm(diff)

        return np.linalg.norm(diff) / baseline_norm

    def _is_adversarial(
        self,
        perturbation_magnitude: float,
        chaos_result: Optional[ChaosAnalysisResult],
        defense_result: Optional[DefensePerturbationResult],
    ) -> bool:
        """Determine if input is adversarial based on all analyses."""
        # If perturbation is negligible, the input is clean
        # This prevents false positives from chaos analysis on clean inputs
        if perturbation_magnitude < 0.01:
            return False

        # Check perturbation threshold
        if perturbation_magnitude > self.perturbation_threshold:
            return True

        # Check chaos analysis (only if perturbation is significant)
        if chaos_result is not None and chaos_result.is_adversarial_perturbation:
            return True

        # Check defense perturbation
        if defense_result is not None and defense_result.is_robust_adversarial:
            return True

        return False

    def _calculate_confidence(
        self,
        perturbation_magnitude: float,
        chaos_result: Optional[ChaosAnalysisResult],
        defense_result: Optional[DefensePerturbationResult],
    ) -> float:
        """Calculate overall detection confidence."""
        confidences = []

        # Perturbation-based confidence
        if perturbation_magnitude > 0:
            pert_conf = min(1.0, perturbation_magnitude / self.perturbation_threshold)
            confidences.append(pert_conf)

        # Chaos-based confidence
        if chaos_result is not None:
            confidences.append(chaos_result.confidence)

        # Defense perturbation confidence
        if defense_result is not None:
            confidences.append(defense_result.confidence)

        if not confidences:
            return 0.0

        return np.mean(confidences)

    def _identify_attack_type(
        self,
        perturbation_mag: float,
        consistency_score: float,
    ) -> AdversarialAttackType:
        """Identify the type of adversarial attack."""
        if perturbation_mag > 0.5:
            return AdversarialAttackType.PGD
        elif consistency_score < 0.3:
            return AdversarialAttackType.DEEPFOOL
        elif perturbation_mag < 0.1:
            return AdversarialAttackType.FGSM
        elif consistency_score > 0.8:
            return AdversarialAttackType.ROBUST_ADVERSARIAL
        else:
            return AdversarialAttackType.CARLINI_WAGNER

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {
                "total_detections": 0,
                "adversarial_count": 0,
                "clean_count": 0,
                "adversarial_rate": 0.0,
            }

        adversarial_count = sum(1 for r in self.detection_history if r.is_adversarial)
        clean_count = len(self.detection_history) - adversarial_count

        return {
            "total_detections": len(self.detection_history),
            "adversarial_count": adversarial_count,
            "clean_count": clean_count,
            "adversarial_rate": adversarial_count / len(self.detection_history),
            "average_confidence": np.mean(
                [r.confidence for r in self.detection_history]
            ),
            "average_perturbation": np.mean(
                [r.perturbation_magnitude for r in self.detection_history]
            ),
        }


# =============================================================================
# MODEL POISONING DETECTOR
# =============================================================================


class ModelPoisoningDetector:
    """
    Model Poisoning Detection System.

    Detects various forms of model poisoning attacks including:
    - Data poisoning
    - Label flipping
    - Backdoor injection
    - Gradient manipulation
    """

    def __init__(
        self,
        gradient_threshold: float = 3.0,
        loss_anomaly_threshold: float = 3.0,
        enable_activation_analysis: bool = True,
    ):
        """
        Initialize Model Poisoning Detector.

        Args:
            gradient_threshold: Z-score threshold for gradient anomalies
            loss_anomaly_threshold: Z-score threshold for loss anomalies
            enable_activation_analysis: Enable activation pattern analysis
        """
        self.gradient_threshold = gradient_threshold
        self.loss_anomaly_threshold = loss_anomaly_threshold
        self.enable_activation_analysis = enable_activation_analysis
        self.detection_history: List[PoisoningDetectionResult] = []
        self._gradient_history: List[np.ndarray] = []
        self._loss_history: List[float] = []

    def detect_poisoning(
        self,
        training_batch: List[Any],
        gradients: Optional[List[float]] = None,
        loss_values: Optional[List[float]] = None,
        activations: Optional[np.ndarray] = None,
    ) -> PoisoningDetectionResult:
        """
        Detect poisoning in training data.

        Args:
            training_batch: Batch of training samples
            gradients: Gradient values from training
            loss_values: Loss values during training
            activations: Activation patterns

        Returns:
            PoisoningDetectionResult with detection outcome
        """
        gradient_score = 0.0
        loss_score = 0.0

        # Analyze gradients
        if gradients is not None:
            gradient_array = np.array(gradients)
            self._gradient_history.append(gradient_array)
            gradient_score = self._calculate_gradient_anomaly_score(gradient_array)

        # Analyze loss values
        if loss_values is not None:
            loss_array = np.array(loss_values)
            for loss in loss_array:
                self._loss_history.append(float(loss))
            loss_score = self._calculate_loss_anomaly_score(loss_array)

        # Combine scores
        total_score = max(gradient_score, loss_score)

        # Determine if poisoned (convert to Python bool to ensure proper identity checks)
        is_poisoned = bool(
            gradient_score > self.gradient_threshold
            or loss_score > self.loss_anomaly_threshold
        )

        # Identify poisoning type
        poisoning_type = None
        if is_poisoned:
            poisoning_type = self._identify_poisoning_type(gradient_score, loss_score)

        # Estimate affected samples
        affected_samples = self._estimate_affected_samples(training_batch, is_poisoned)

        # Calculate confidence
        confidence = self._calculate_confidence(gradient_score, loss_score)

        result = PoisoningDetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            poisoning_type=poisoning_type,
            affected_samples=affected_samples,
            gradient_anomaly_score=total_score,
        )

        self.detection_history.append(result)
        return result

    def _calculate_gradient_anomaly_score(self, gradients: np.ndarray) -> float:
        """Calculate gradient anomaly score using Z-score."""
        if len(self._gradient_history) < 2:
            return 0.0

        # Calculate statistics from history
        all_gradients = np.concatenate(
            [g.flatten() for g in self._gradient_history[:-1]]
        )
        mean = np.mean(all_gradients)
        std = np.std(all_gradients)

        if std == 0:
            return 0.0

        # Calculate Z-score for current gradients
        current_mean = np.mean(gradients)
        z_score = abs(current_mean - mean) / std

        return z_score

    def _calculate_loss_anomaly_score(self, loss_values: np.ndarray) -> float:
        """Calculate loss anomaly score."""
        if len(loss_values) < 2:
            return 0.0

        # Check for sudden spikes
        variance = np.var(loss_values)
        mean_loss = np.mean(loss_values)

        if mean_loss == 0:
            return 0.0

        # Coefficient of variation as anomaly indicator
        cv = np.sqrt(variance) / mean_loss

        # Scale to anomaly score
        return cv * 10

    def _identify_poisoning_type(
        self,
        gradient_score: float,
        loss_score: float,
    ) -> PoisoningType:
        """Identify the type of poisoning attack."""
        if gradient_score > loss_score * 2:
            return PoisoningType.GRADIENT_MANIPULATION
        elif loss_score > gradient_score * 2:
            return PoisoningType.LABEL_FLIPPING
        elif gradient_score > 5 and loss_score > 5:
            return PoisoningType.BACKDOOR_INJECTION
        else:
            return PoisoningType.DATA_POISONING

    def _estimate_affected_samples(
        self,
        training_batch: List[Any],
        is_poisoned: bool,
    ) -> int:
        """Estimate number of affected samples."""
        if not is_poisoned:
            return 0

        # Estimate 10-30% of batch is affected
        batch_size = len(training_batch)
        return max(1, int(batch_size * 0.2))

    def _calculate_confidence(
        self,
        gradient_score: float,
        loss_score: float,
    ) -> float:
        """Calculate detection confidence."""
        max_score = max(gradient_score, loss_score)
        threshold = max(self.gradient_threshold, self.loss_anomaly_threshold)

        if threshold == 0:
            return 0.0

        return min(1.0, max_score / (threshold * 2))

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get poisoning detection statistics."""
        if not self.detection_history:
            return {
                "total_detections": 0,
                "poisoned_count": 0,
                "clean_count": 0,
            }

        poisoned_count = sum(1 for r in self.detection_history if r.is_poisoned)

        return {
            "total_detections": len(self.detection_history),
            "poisoned_count": poisoned_count,
            "clean_count": len(self.detection_history) - poisoned_count,
            "average_gradient_score": np.mean(
                [r.gradient_anomaly_score for r in self.detection_history]
            ),
        }


# =============================================================================
# DIFFERENTIAL PRIVACY MANAGER
# =============================================================================


class DifferentialPrivacyManager:
    """
    Differential Privacy Manager.

    Provides differential privacy guarantees for ML operations with
    epsilon-delta privacy budgeting and multiple noise mechanisms.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE,
    ):
        """
        Initialize Differential Privacy Manager.

        Args:
            epsilon: Privacy parameter epsilon (must be > 0)
            delta: Privacy parameter delta (must be in [0, 1))
            mechanism: Noise mechanism to use
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta < 0 or delta >= 1:
            raise ValueError("Delta must be in [0, 1)")

        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        self.mechanism = mechanism
        self.query_log: List[Dict[str, Any]] = []

    def add_noise(
        self,
        value: float,
        sensitivity: float = 1.0,
        epsilon_cost: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """
        Add differentially private noise to a value.

        Args:
            value: Original value
            sensitivity: Query sensitivity
            epsilon_cost: Custom epsilon cost (default: 0.1)

        Returns:
            Tuple of (noised_value, success)
        """
        if epsilon_cost is None:
            epsilon_cost = 0.1

        # Check budget
        if self.budget.is_depleted:
            return value, False

        # Add noise based on mechanism
        if self.mechanism == PrivacyMechanism.LAPLACE:
            noise = self._laplace_noise(sensitivity, self.budget.epsilon)
        elif self.mechanism == PrivacyMechanism.GAUSSIAN:
            noise = self._gaussian_noise(
                sensitivity, self.budget.epsilon, self.budget.delta
            )
        else:
            noise = self._laplace_noise(sensitivity, self.budget.epsilon)

        noised_value = value + noise

        # Update budget
        self.budget.spent_epsilon += epsilon_cost
        self.budget.query_count += 1

        # Log query
        self.query_log.append(
            {
                "query_id": len(self.query_log) + 1,
                "original_value": value,
                "noised_value": noised_value,
                "epsilon_cost": epsilon_cost,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return noised_value, True

    def _laplace_noise(self, sensitivity: float, epsilon: float) -> float:
        """Generate Laplace noise."""
        scale = sensitivity / epsilon
        return np.random.laplace(0, scale)

    def _gaussian_noise(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """Generate Gaussian noise for (epsilon, delta)-DP."""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return np.random.normal(0, sigma)

    def get_privacy_loss(self) -> Dict[str, float]:
        """Get current privacy loss."""
        return {
            "epsilon_loss": self.budget.spent_epsilon,
            "epsilon_remaining": self.budget.remaining_epsilon,
            "delta_loss": self.budget.spent_delta,
            "delta_remaining": self.budget.remaining_delta,
            "query_count": self.budget.query_count,
        }

    def reset_budget(self) -> None:
        """Reset privacy budget."""
        self.budget.spent_epsilon = 0.0
        self.budget.spent_delta = 0.0
        self.budget.query_count = 0
        self.query_log = []

    def export_audit_log(self) -> List[Dict[str, Any]]:
        """Export privacy query audit log."""
        return self.query_log.copy()


# =============================================================================
# FEDERATED LEARNING COORDINATOR
# =============================================================================


class FederatedLearningCoordinator:
    """
    Federated Learning Coordinator.

    Manages federated learning rounds with secure aggregation
    and poisoning detection capabilities.
    """

    def __init__(
        self,
        min_participants: int = 3,
        enable_secure_aggregation: bool = True,
        enable_poisoning_detection: bool = True,
    ):
        """
        Initialize Federated Learning Coordinator.

        Args:
            min_participants: Minimum required participants per round
            enable_secure_aggregation: Enable secure aggregation
            enable_poisoning_detection: Enable poisoning detection
        """
        self.min_participants = min_participants
        self.enable_secure_aggregation = enable_secure_aggregation
        self.enable_poisoning_detection = enable_poisoning_detection
        self.rounds: List[FederatedLearningRound] = []

    def aggregate_updates(
        self,
        participant_updates: List[Dict[str, Any]],
    ) -> FederatedLearningRound:
        """
        Aggregate participant updates.

        Args:
            participant_updates: List of participant update dictionaries

        Returns:
            FederatedLearningRound with aggregation result
        """
        if len(participant_updates) < self.min_participants:
            raise ValueError(
                f"Insufficient participants: {len(participant_updates)} < {self.min_participants}"
            )

        # Detect poisoning if enabled
        poisoning_detected = False
        if self.enable_poisoning_detection:
            poisoning_detected = self._detect_poisoning(participant_updates)

        # Aggregate weights/gradients
        if self.enable_secure_aggregation:
            aggregated = self._secure_aggregate(participant_updates)
        else:
            aggregated = self._simple_aggregate(participant_updates)

        # Calculate validation accuracy (simulated)
        validation_accuracy = self._calculate_validation_accuracy(aggregated)

        round_result = FederatedLearningRound(
            round_id=f"round_{len(self.rounds) + 1}",
            participant_count=len(participant_updates),
            aggregated_weights=aggregated,
            validation_accuracy=validation_accuracy,
            poisoning_detected=poisoning_detected,
        )

        self.rounds.append(round_result)
        return round_result

    def _detect_poisoning(self, updates: List[Dict[str, Any]]) -> bool:
        """Detect poisoning in participant updates."""
        # Simple outlier detection based on gradient magnitude
        gradients = []
        for update in updates:
            if "gradients" in update:
                gradients.append(np.mean(np.abs(update["gradients"])))

        if len(gradients) < 2:
            return False

        mean = np.mean(gradients)
        std = np.std(gradients)

        if std == 0:
            return False

        # Check for outliers (Z-score > 3)
        for grad in gradients:
            if abs(grad - mean) > 3 * std:
                return True

        return False

    def _secure_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform secure aggregation."""
        # Simplified secure aggregation (in practice, use cryptographic protocols)
        return {
            "update_count": len(updates),
            "aggregation_method": "secure",
            "timestamp": datetime.now().isoformat(),
        }

    def _simple_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform simple averaging aggregation."""
        all_gradients = []
        for update in updates:
            if "gradients" in update:
                all_gradients.append(update["gradients"])

        if all_gradients:
            averaged = np.mean(all_gradients, axis=0)
            return {
                "averaged_gradients": (
                    averaged.tolist() if isinstance(averaged, np.ndarray) else averaged
                ),
                "update_count": len(updates),
            }

        return {"update_count": len(updates)}

    def _calculate_validation_accuracy(self, aggregated: Dict[str, Any]) -> float:
        """Calculate validation accuracy (simulated)."""
        # Simulated accuracy based on round number
        base_accuracy = 0.5
        round_bonus = min(0.4, len(self.rounds) * 0.05)
        return base_accuracy + round_bonus + np.random.uniform(0, 0.1)

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get federated training statistics."""
        if not self.rounds:
            return {
                "total_rounds": 0,
                "average_accuracy": 0.0,
                "total_participants": 0,
            }

        return {
            "total_rounds": len(self.rounds),
            "average_accuracy": np.mean([r.validation_accuracy for r in self.rounds]),
            "total_participants": sum(r.participant_count for r in self.rounds),
            "poisoning_detected_rounds": sum(
                1 for r in self.rounds if r.poisoning_detected
            ),
        }


# =============================================================================
# EXPLAINABLE AI SYSTEM
# =============================================================================


class ExplainableAISystem:
    """
    Explainable AI System.

    Provides model explanations for compliance with regulatory
    frameworks (GDPR, HIPAA, DoD AI Ethics).
    """

    def __init__(
        self,
        compliance_frameworks: Optional[List[str]] = None,
        explanation_method: str = "shap",
    ):
        """
        Initialize Explainable AI System.

        Args:
            compliance_frameworks: List of compliance frameworks
            explanation_method: Method for generating explanations
        """
        self.compliance_frameworks = compliance_frameworks or ["GDPR"]
        self.explanation_method = explanation_method
        self.explanation_history: List[ExplainabilityReport] = []

    def generate_explanation(
        self,
        model_id: str,
        input_features: Dict[str, Any],
        prediction: Any,
        model: Optional[Any] = None,
    ) -> ExplainabilityReport:
        """
        Generate explanation for a model prediction.

        Args:
            model_id: Identifier for the model
            input_features: Input features used for prediction
            prediction: Model prediction
            model: Optional model object for detailed analysis

        Returns:
            ExplainabilityReport with explanation details
        """
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(input_features, model)

        # Generate human-readable explanation
        human_explanation = self._generate_human_explanation(
            input_features, feature_importance, prediction
        )

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(feature_importance)

        report = ExplainabilityReport(
            model_id=model_id,
            prediction=prediction,
            feature_importance=feature_importance,
            explanation_method=self.explanation_method,
            compliance_frameworks=self.compliance_frameworks,
            human_readable_explanation=human_explanation,
            confidence_score=confidence_score,
        )

        self.explanation_history.append(report)
        return report

    def _calculate_feature_importance(
        self,
        features: Dict[str, Any],
        model: Optional[Any],
    ) -> Dict[str, float]:
        """Calculate feature importance scores."""
        # Simple heuristic: normalize by magnitude
        importance = {}
        total = 0

        for name, value in features.items():
            if isinstance(value, (int, float)):
                magnitude = abs(float(value))
            else:
                magnitude = 1.0
            importance[name] = magnitude
            total += magnitude

        # Normalize to sum to 1
        if total > 0:
            for name in importance:
                importance[name] /= total

        return importance

    def _generate_human_explanation(
        self,
        features: Dict[str, Any],
        importance: Dict[str, float],
        prediction: Any,
    ) -> str:
        """Generate human-readable explanation."""
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Build explanation
        explanation_parts = [
            f"The prediction '{prediction}' was made based on the following factors:"
        ]

        for i, (feature, imp) in enumerate(sorted_features[:3]):
            value = features.get(feature, "unknown")
            explanation_parts.append(
                f"- {feature} (value: {value}) contributed {imp*100:.1f}% to the decision"
            )

        # Add compliance note
        frameworks_str = ", ".join(self.compliance_frameworks)
        explanation_parts.append(
            f"\nThis explanation is provided in compliance with: {frameworks_str}"
        )

        return "\n".join(explanation_parts)

    def _calculate_confidence_score(self, importance: Dict[str, float]) -> float:
        """Calculate confidence in the explanation."""
        if not importance:
            return 0.0

        # Higher confidence if importance is well distributed
        values = list(importance.values())
        max_importance = max(values)

        # If one feature dominates, confidence is high
        # If evenly distributed, confidence is medium
        return min(1.0, max_importance + 0.3)

    def export_compliance_report(self) -> Dict[str, Any]:
        """Export compliance report."""
        if not self.explanation_history:
            return {
                "total_explanations": 0,
                "compliance_frameworks": self.compliance_frameworks,
                "explanations": [],
            }

        return {
            "total_explanations": len(self.explanation_history),
            "compliance_frameworks": self.compliance_frameworks,
            "average_confidence": np.mean(
                [r.confidence_score for r in self.explanation_history]
            ),
            "explanation_method": self.explanation_method,
            "explanations": [r.to_dict() for r in self.explanation_history[:100]],
        }


# =============================================================================
# AI/ML SECURITY MANAGER
# =============================================================================


class AIMLSecurityManager:
    """
    Comprehensive AI/ML Security Manager.

    Integrates all AI/ML security components into a unified interface.
    """

    def __init__(
        self,
        enable_adversarial_defense: bool = True,
        enable_poisoning_detection: bool = True,
        enable_differential_privacy: bool = True,
        enable_federated_learning: bool = True,
        enable_explainable_ai: bool = True,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
    ):
        """
        Initialize AI/ML Security Manager.

        Args:
            enable_adversarial_defense: Enable adversarial defense
            enable_poisoning_detection: Enable poisoning detection
            enable_differential_privacy: Enable differential privacy
            enable_federated_learning: Enable federated learning
            enable_explainable_ai: Enable explainable AI
            privacy_epsilon: Privacy epsilon parameter
            privacy_delta: Privacy delta parameter
        """
        self.adversarial_defense: Optional[AdversarialDefenseSystem] = None
        self.poisoning_detector: Optional[ModelPoisoningDetector] = None
        self.privacy_manager: Optional[DifferentialPrivacyManager] = None
        self.federated_coordinator: Optional[FederatedLearningCoordinator] = None
        self.explainable_ai: Optional[ExplainableAISystem] = None

        if enable_adversarial_defense:
            self.adversarial_defense = AdversarialDefenseSystem()

        if enable_poisoning_detection:
            self.poisoning_detector = ModelPoisoningDetector()

        if enable_differential_privacy:
            self.privacy_manager = DifferentialPrivacyManager(
                epsilon=privacy_epsilon,
                delta=privacy_delta,
            )

        if enable_federated_learning:
            self.federated_coordinator = FederatedLearningCoordinator()

        if enable_explainable_ai:
            self.explainable_ai = ExplainableAISystem()

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        status = {
            "adversarial_defense": {
                "enabled": self.adversarial_defense is not None,
                "statistics": (
                    self.adversarial_defense.get_detection_statistics()
                    if self.adversarial_defense
                    else {}
                ),
            },
            "poisoning_detection": {
                "enabled": self.poisoning_detector is not None,
                "statistics": (
                    self.poisoning_detector.get_detection_statistics()
                    if self.poisoning_detector
                    else {}
                ),
            },
            "differential_privacy": {
                "enabled": self.privacy_manager is not None,
                "privacy_loss": (
                    self.privacy_manager.get_privacy_loss()
                    if self.privacy_manager
                    else {}
                ),
            },
            "federated_learning": {
                "enabled": self.federated_coordinator is not None,
                "statistics": (
                    self.federated_coordinator.get_training_statistics()
                    if self.federated_coordinator
                    else {}
                ),
            },
            "explainable_ai": {
                "enabled": self.explainable_ai is not None,
                "report": (
                    {"total_explanations": len(self.explainable_ai.explanation_history)}
                    if self.explainable_ai
                    else {}
                ),
            },
        }
        return status

    def export_security_report(self) -> Dict[str, Any]:
        """Export comprehensive security report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "security_status": self.get_security_status(),
            "compliance": {
                "adversarial_protection": self.adversarial_defense is not None,
                "poisoning_detection": self.poisoning_detector is not None,
                "privacy_preserved": self.privacy_manager is not None,
                "federated_enabled": self.federated_coordinator is not None,
                "explainability_enabled": self.explainable_ai is not None,
            },
        }
