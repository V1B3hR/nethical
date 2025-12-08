"""Semantic similarity utilities using sentence-transformers.

This module provides a singleton pattern for loading and using sentence embeddings
for semantic text comparison. Falls back gracefully to lexical similarity if the
sentence-transformers library is not available.

Features:
- Lazy loading of embedding model (all-MiniLM-L6-v2)
- Thread-safe singleton initialization
- Cosine similarity computation between text pairs
- Graceful degradation to lexical similarity
- Caching for performance
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

# Global singleton instance
_model_instance: Optional["SentenceTransformerWrapper"] = None
_model_lock = threading.Lock()


class SentenceTransformerWrapper:
    """Thread-safe wrapper for sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded (lazy loading)."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:  # Double-check after acquiring lock
                return

            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading sentence-transformers model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Model {self.model_name} loaded successfully")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Semantic similarity unavailable. "
                    "Install with: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load sentence-transformers model: {e}")
                raise

    def encode(self, texts: list[str] | str, **kwargs) -> Any:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments to pass to model.encode()

        Returns:
            Embeddings array
        """
        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(texts, **kwargs)

    @property
    def is_available(self) -> bool:
        """Check if the model is available."""
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False


def get_embedding_model() -> Optional[SentenceTransformerWrapper]:
    """Get or create the singleton embedding model instance.

    Returns:
        SentenceTransformerWrapper instance or None if unavailable
    """
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    with _model_lock:
        if _model_instance is not None:  # Double-check
            return _model_instance

        try:
            _model_instance = SentenceTransformerWrapper()
            return _model_instance
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")
            return None


def cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector (numpy array)
        vec_b: Second vector (numpy array)

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    import numpy as np

    # Normalize to handle both 1D and 2D arrays
    vec_a = np.atleast_2d(vec_a)
    vec_b = np.atleast_2d(vec_b)

    # Compute cosine similarity
    dot_product = np.dot(vec_a, vec_b.T)
    norm_a = np.linalg.norm(vec_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(vec_b, axis=1, keepdims=True)

    similarity = dot_product / (norm_a * norm_b.T)
    return float(similarity[0, 0])


def get_similarity(text_a: str, text_b: str) -> float:
    """Get semantic similarity between two texts using embeddings.

    This function computes cosine similarity between sentence embeddings.
    Falls back to lexical similarity if embeddings are unavailable.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Similarity score (0.0 to 1.0), where 1.0 is identical
    """
    # Handle empty or None inputs
    if not text_a or not text_b:
        return 0.0

    # Try semantic similarity first
    model = get_embedding_model()
    if model and model.is_available:
        try:
            embeddings = model.encode([text_a, text_b])
            similarity = cosine_similarity(embeddings[0], embeddings[1])
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception as e:
            logger.warning(f"Semantic similarity failed, falling back to lexical: {e}")

    # Fallback to lexical similarity
    return _lexical_similarity(text_a, text_b)


def _lexical_similarity(text_a: str, text_b: str) -> float:
    """Compute lexical similarity using token overlap (Jaccard index).

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    # Simple tokenization
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    # Jaccard similarity
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union if union > 0 else 0.0


def get_semantic_deviation(stated_intent: str, actual_action: str) -> float:
    """Compute semantic deviation between stated intent and actual action.

    Deviation is defined as 1 - similarity, where:
    - 0.0 = perfect alignment (no deviation)
    - 1.0 = complete mismatch (maximum deviation)

    Args:
        stated_intent: The stated intent or goal
        actual_action: The actual action taken

    Returns:
        Deviation score (0.0 to 1.0)
    """
    similarity = get_similarity(stated_intent, actual_action)
    deviation = 1.0 - similarity
    return max(0.0, min(1.0, deviation))  # Clamp to [0, 1]


@lru_cache(maxsize=1024)
def get_concept_similarity(text: str, concept_phrase: str) -> float:
    """Get similarity between text and a concept phrase (with caching).

    This is useful for matching against predefined ethical/safety concepts.
    Results are cached for performance.

    Args:
        text: Text to analyze
        concept_phrase: Concept phrase to match against

    Returns:
        Similarity score (0.0 to 1.0)
    """
    return get_similarity(text, concept_phrase)


def batch_similarity(
    texts: list[str], reference_text: str, batch_size: int = 32
) -> list[float]:
    """Compute similarity scores for multiple texts against a reference.

    More efficient than calling get_similarity repeatedly.

    Args:
        texts: List of texts to compare
        reference_text: Reference text to compare against
        batch_size: Batch size for encoding

    Returns:
        List of similarity scores
    """
    if not texts:
        return []

    model = get_embedding_model()
    if not model or not model.is_available:
        # Fallback to individual lexical comparisons
        return [_lexical_similarity(t, reference_text) for t in texts]

    try:
        # Encode reference
        ref_embedding = model.encode([reference_text])[0]

        # Encode texts in batches
        similarities = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = model.encode(batch)

            for emb in batch_embeddings:
                sim = cosine_similarity(emb, ref_embedding)
                similarities.append(max(0.0, min(1.0, sim)))

        return similarities
    except Exception as e:
        logger.warning(f"Batch similarity failed, falling back: {e}")
        return [_lexical_similarity(t, reference_text) for t in texts]


def clear_cache() -> None:
    """Clear the LRU cache for concept similarity."""
    get_concept_similarity.cache_clear()


def is_semantic_available() -> bool:
    """Check if semantic similarity is available.

    Returns:
        True if sentence-transformers is available and working
    """
    model = get_embedding_model()
    return model is not None and model.is_available
