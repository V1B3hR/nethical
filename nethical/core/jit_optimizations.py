"""
JIT compilation support for performance-critical hot paths.

This module provides JIT (Just-In-Time) compilation for performance-critical
functions using Numba to achieve near-C performance in Python.
"""

import logging
from typing import Any, Callable, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning(
        "Numba not available. Install with: pip install numba\n"
        "JIT compilation will be disabled, using pure Python implementations."
    )
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    njit = jit
    prange = range


def is_jit_available() -> bool:
    """Check if JIT compilation is available."""
    return NUMBA_AVAILABLE


# ============================================================================
# Risk Score Calculation (Hot Path)
# ============================================================================

@njit(cache=True)
def calculate_risk_score_jit(
    violation_severities: np.ndarray,
    violation_confidences: np.ndarray,
    base_weight: float = 1.0,
    severity_weight: float = 0.6,
    confidence_weight: float = 0.4
) -> float:
    """
    JIT-compiled risk score calculation.
    
    Args:
        violation_severities: Array of violation severities (0-5)
        violation_confidences: Array of violation confidences (0-1)
        base_weight: Base weight for score
        severity_weight: Weight for severity component
        confidence_weight: Weight for confidence component
        
    Returns:
        Calculated risk score (0-1)
    """
    if len(violation_severities) == 0:
        return 0.0
    
    # Normalize severities to 0-1 range
    normalized_severities = violation_severities / 5.0
    
    # Calculate weighted score
    total_score = 0.0
    for i in range(len(violation_severities)):
        severity_component = normalized_severities[i] * severity_weight
        confidence_component = violation_confidences[i] * confidence_weight
        total_score += (severity_component + confidence_component) * base_weight
    
    # Average and normalize
    avg_score = total_score / len(violation_severities)
    return min(1.0, max(0.0, avg_score))


@njit(cache=True)
def calculate_temporal_decay_jit(
    risk_score: float,
    time_delta_seconds: float,
    decay_rate: float = 0.1
) -> float:
    """
    JIT-compiled temporal decay calculation for risk scores.
    
    Args:
        risk_score: Current risk score
        time_delta_seconds: Time elapsed in seconds
        decay_rate: Decay rate (higher = faster decay)
        
    Returns:
        Decayed risk score
    """
    # Exponential decay: score * exp(-decay_rate * time_hours)
    time_hours = time_delta_seconds / 3600.0
    decay_factor = np.exp(-decay_rate * time_hours)
    return risk_score * decay_factor


# ============================================================================
# Statistical Calculations (Hot Path)
# ============================================================================

@njit(cache=True)
def calculate_statistics_jit(values: np.ndarray) -> tuple:
    """
    JIT-compiled statistical calculations.
    
    Args:
        values: Array of values
        
    Returns:
        Tuple of (mean, std, min, max, median)
    """
    if len(values) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    median = np.median(values)
    
    return (mean, std, min_val, max_val, median)


@njit(cache=True, parallel=True)
def calculate_moving_average_jit(
    values: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    JIT-compiled moving average calculation.
    
    Args:
        values: Array of values
        window_size: Window size for moving average
        
    Returns:
        Array of moving averages
    """
    n = len(values)
    result = np.zeros(n)
    
    for i in prange(n):
        start = max(0, i - window_size + 1)
        end = i + 1
        result[i] = np.mean(values[start:end])
    
    return result


# ============================================================================
# Similarity and Distance Calculations (Hot Path)
# ============================================================================

@njit(cache=True)
def cosine_similarity_jit(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    JIT-compiled cosine similarity calculation.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


@njit(cache=True)
def euclidean_distance_jit(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    JIT-compiled Euclidean distance calculation.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(vec1 - vec2)


@njit(cache=True, parallel=True)
def batch_cosine_similarity_jit(
    vectors: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled batch cosine similarity calculation.
    
    Args:
        vectors: Array of vectors (n, d)
        reference: Reference vector (d,)
        
    Returns:
        Array of similarities (n,)
    """
    n = vectors.shape[0]
    similarities = np.zeros(n)
    
    ref_norm = np.linalg.norm(reference)
    if ref_norm == 0:
        return similarities
    
    for i in prange(n):
        vec_norm = np.linalg.norm(vectors[i])
        if vec_norm == 0:
            similarities[i] = 0.0
        else:
            dot_product = np.dot(vectors[i], reference)
            similarities[i] = dot_product / (vec_norm * ref_norm)
    
    return similarities


# ============================================================================
# Anomaly Detection (Hot Path)
# ============================================================================

@njit(cache=True)
def detect_outliers_iqr_jit(values: np.ndarray, iqr_multiplier: float = 1.5) -> np.ndarray:
    """
    JIT-compiled outlier detection using IQR method.
    
    Args:
        values: Array of values
        iqr_multiplier: IQR multiplier for outlier threshold
        
    Returns:
        Boolean array indicating outliers
    """
    if len(values) == 0:
        return np.array([], dtype=np.bool_)
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    outliers = (values < lower_bound) | (values > upper_bound)
    return outliers


@njit(cache=True)
def detect_outliers_zscore_jit(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    JIT-compiled outlier detection using Z-score method.
    
    Args:
        values: Array of values
        threshold: Z-score threshold for outliers
        
    Returns:
        Boolean array indicating outliers
    """
    if len(values) == 0:
        return np.array([], dtype=np.bool_)
    
    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:
        return np.zeros(len(values), dtype=np.bool_)
    
    z_scores = np.abs((values - mean) / std)
    outliers = z_scores > threshold
    return outliers


# ============================================================================
# Feature Extraction (Hot Path)
# ============================================================================

@njit(cache=True, parallel=True)
def extract_ngram_features_jit(
    token_ids: np.ndarray,
    n: int,
    vocab_size: int
) -> np.ndarray:
    """
    JIT-compiled n-gram feature extraction.
    
    Args:
        token_ids: Array of token IDs
        n: N-gram size
        vocab_size: Vocabulary size
        
    Returns:
        Feature vector
    """
    num_ngrams = vocab_size ** n
    features = np.zeros(num_ngrams)
    
    if len(token_ids) < n:
        return features
    
    for i in prange(len(token_ids) - n + 1):
        # Calculate n-gram hash
        ngram_hash = 0
        for j in range(n):
            ngram_hash = ngram_hash * vocab_size + token_ids[i + j]
        
        if ngram_hash < num_ngrams:
            features[ngram_hash] += 1
    
    # Normalize
    total = np.sum(features)
    if total > 0:
        features = features / total
    
    return features


# ============================================================================
# Matrix Operations (Hot Path)
# ============================================================================

@njit(cache=True, parallel=True)
def matrix_multiply_jit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    JIT-compiled matrix multiplication.
    
    Args:
        A: Matrix A (m, n)
        B: Matrix B (n, p)
        
    Returns:
        Result matrix (m, p)
    """
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError("Matrix dimensions don't match")
    
    C = np.zeros((m, p))
    
    for i in prange(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


# ============================================================================
# Utility Functions
# ============================================================================

def benchmark_jit_speedup(func_jit: Callable, func_python: Callable, *args, **kwargs) -> dict:
    """
    Benchmark JIT speedup compared to pure Python.
    
    Args:
        func_jit: JIT-compiled function
        func_python: Pure Python function
        *args: Arguments to pass to functions
        **kwargs: Keyword arguments to pass to functions
        
    Returns:
        Dictionary with timing results and speedup
    """
    import time
    
    if not NUMBA_AVAILABLE:
        return {
            "jit_available": False,
            "message": "Numba not available, cannot benchmark JIT speedup"
        }
    
    # Warmup JIT
    _ = func_jit(*args, **kwargs)
    
    # Benchmark JIT
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = func_jit(*args, **kwargs)
    jit_time = time.time() - start
    
    # Benchmark Python
    start = time.time()
    for _ in range(iterations):
        _ = func_python(*args, **kwargs)
    python_time = time.time() - start
    
    speedup = python_time / jit_time if jit_time > 0 else 0
    
    return {
        "jit_available": True,
        "jit_time_ms": jit_time * 1000 / iterations,
        "python_time_ms": python_time * 1000 / iterations,
        "speedup": speedup,
        "iterations": iterations
    }
