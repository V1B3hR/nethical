"""
JIT-Compiled Detectors for Edge Deployment

Ultra-fast detection using JIT compilation.
Target: 10-100x speedup over pure Python.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import jit, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available for JIT detectors")

    def njit(*args, **kwargs):
        """Fallback decorator when Numba is not available."""
        import functools

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*fargs, **fkwargs):
                return func(*fargs, **fkwargs)

            return wrapper

        # Handle both @njit and @njit() syntax
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return decorator(args[0])
        return decorator

    prange = range


@njit(cache=True)
def fast_keyword_match(
    action_tokens: np.ndarray,
    keyword_hashes: np.ndarray,
) -> Tuple[bool, int]:
    """
    JIT-compiled keyword matching.

    Args:
        action_tokens: Hashed tokens from action
        keyword_hashes: Hashed keywords to match

    Returns:
        (has_match, match_count)
    """
    match_count = 0
    for token in action_tokens:
        for keyword in keyword_hashes:
            if token == keyword:
                match_count += 1
                break  # Count each token once

    return match_count > 0, match_count


@njit(cache=True)
def fast_severity_score(
    match_severities: np.ndarray,
    match_confidences: np.ndarray,
) -> float:
    """
    JIT-compiled severity score calculation.

    Args:
        match_severities: Severity levels of matches (1-5)
        match_confidences: Confidence levels (0-1)

    Returns:
        Weighted severity score (0-1)
    """
    if len(match_severities) == 0:
        return 0.0

    # Weighted sum
    weighted_sum = 0.0
    for i in range(len(match_severities)):
        normalized_severity = match_severities[i] / 5.0
        weighted_sum += normalized_severity * match_confidences[i]

    return min(1.0, weighted_sum / len(match_severities))


@njit(cache=True)
def fast_pattern_match(
    action_vector: np.ndarray,
    pattern_vectors: np.ndarray,
    threshold: float = 0.8,
) -> Tuple[bool, np.ndarray]:
    """
    JIT-compiled pattern matching using cosine similarity.

    Args:
        action_vector: Vectorized action representation
        pattern_vectors: Matrix of pattern vectors
        threshold: Similarity threshold for match

    Returns:
        (has_match, similarity_scores)
    """
    n_patterns = pattern_vectors.shape[0]
    similarities = np.zeros(n_patterns)

    action_norm = np.linalg.norm(action_vector)
    if action_norm == 0:
        return False, similarities

    for i in range(n_patterns):
        pattern_norm = np.linalg.norm(pattern_vectors[i])
        if pattern_norm == 0:
            continue

        dot_product = np.dot(action_vector, pattern_vectors[i])
        similarities[i] = dot_product / (action_norm * pattern_norm)

    has_match = np.any(similarities >= threshold)
    return has_match, similarities


@njit(cache=True, parallel=True)
def fast_batch_detection(
    action_vectors: np.ndarray,
    pattern_vectors: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    """
    JIT-compiled batch detection with parallelization.

    Args:
        action_vectors: Matrix of action vectors (n_actions, d)
        pattern_vectors: Matrix of pattern vectors (n_patterns, d)
        threshold: Match threshold

    Returns:
        Match scores for each action (n_actions,)
    """
    n_actions = action_vectors.shape[0]
    n_patterns = pattern_vectors.shape[0]
    scores = np.zeros(n_actions)

    for i in prange(n_actions):
        action_norm = np.linalg.norm(action_vectors[i])
        if action_norm == 0:
            continue

        max_similarity = 0.0
        for j in range(n_patterns):
            pattern_norm = np.linalg.norm(pattern_vectors[j])
            if pattern_norm == 0:
                continue

            dot_product = np.dot(action_vectors[i], pattern_vectors[j])
            similarity = dot_product / (action_norm * pattern_norm)

            if similarity > max_similarity:
                max_similarity = similarity

        if max_similarity >= threshold:
            scores[i] = max_similarity

    return scores


@njit(cache=True)
def fast_risk_aggregation(
    detection_scores: np.ndarray,
    weights: np.ndarray,
    critical_threshold: float = 0.9,
) -> Tuple[float, bool]:
    """
    JIT-compiled risk score aggregation.

    Args:
        detection_scores: Scores from various detectors
        weights: Weights for each detector
        critical_threshold: Threshold for critical classification

    Returns:
        (aggregated_score, is_critical)
    """
    if len(detection_scores) == 0:
        return 0.0, False

    # Weighted average
    weighted_sum = 0.0
    weight_sum = 0.0

    for i in range(len(detection_scores)):
        weighted_sum += detection_scores[i] * weights[i]
        weight_sum += weights[i]

    if weight_sum == 0:
        return 0.0, False

    aggregated = weighted_sum / weight_sum
    is_critical = aggregated >= critical_threshold or np.max(detection_scores) >= 0.95

    return aggregated, is_critical


def warmup_jit_detectors():
    """
    Warmup JIT compilation for all detector functions.

    Call this at startup to ensure JIT compilation is complete
    before processing real requests.
    """
    if not NUMBA_AVAILABLE:
        return

    logger.info("Warming up JIT detectors...")

    # Warmup keyword match
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    keywords = np.array([3, 6, 9], dtype=np.int64)
    _ = fast_keyword_match(tokens, keywords)

    # Warmup severity score
    severities = np.array([1.0, 2.0, 3.0])
    confidences = np.array([0.8, 0.9, 0.7])
    _ = fast_severity_score(severities, confidences)

    # Warmup pattern match
    action = np.random.randn(128).astype(np.float64)
    patterns = np.random.randn(10, 128).astype(np.float64)
    _ = fast_pattern_match(action, patterns)

    # Warmup batch detection
    actions = np.random.randn(5, 128).astype(np.float64)
    _ = fast_batch_detection(actions, patterns)

    # Warmup risk aggregation
    scores = np.array([0.1, 0.5, 0.3])
    weights = np.array([1.0, 2.0, 1.0])
    _ = fast_risk_aggregation(scores, weights)

    logger.info("JIT detector warmup complete")


def is_jit_available() -> bool:
    """Check if JIT compilation is available."""
    return NUMBA_AVAILABLE
