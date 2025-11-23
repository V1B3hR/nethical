"""Shared AI Core for Nethical Framework.

This module provides a singleton EmbeddingModel class that loads
sentence-transformers models for semantic similarity computations.
Used across monitors and detectors for semantic analysis.
"""

from typing import List, Optional
import logging
from functools import lru_cache
import threading


logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Singleton class for embedding-based semantic similarity.
    
    Loads the sentence-transformers/all-MiniLM-L6-v2 model for computing
    semantic similarities between texts. Thread-safe singleton pattern.
    """
    
    _instance: Optional["EmbeddingModel"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "EmbeddingModel":
        """Ensure singleton pattern with thread-safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model (only once)."""
        # Avoid re-initialization
        if hasattr(self, "_initialized"):
            return
            
        self._initialized = True
        self._model = None
        self._model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._load_model()
    
    def _load_model(self) -> None:
        """Lazy load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not available. "
                "Semantic similarity features will be disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None
    
    def is_available(self) -> bool:
        """Check if the embedding model is available."""
        return self._model is not None
    
    @lru_cache(maxsize=1024)
    def _encode_cached(self, text: str) -> Optional[List[float]]:
        """Encode text to embedding vector with caching.
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as list of floats, or None if model unavailable
        """
        if not self.is_available():
            return None
        
        try:
            # Normalize embeddings for cosine similarity
            embedding = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1, or 0.0 if unavailable
        """
        if not text1 or not text2:
            return 0.0
        
        emb1 = self._encode_cached(text1.strip().lower())
        emb2 = self._encode_cached(text2.strip().lower())
        
        if emb1 is None or emb2 is None:
            logger.debug("Embeddings not available, returning 0.0 similarity")
            return 0.0
        
        # Compute cosine similarity (dot product since embeddings are normalized)
        try:
            similarity = sum(a * b for a, b in zip(emb1, emb2))
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, float(similarity)))
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def compute_similarity_to_list(self, text: str, list_of_texts: List[str]) -> float:
        """Compute maximum similarity score between text and a list of texts.
        
        Args:
            text: Query text
            list_of_texts: List of texts to compare against
            
        Returns:
            Maximum similarity score found, or 0.0 if unavailable or empty list
        """
        if not text or not list_of_texts:
            return 0.0
        
        max_similarity = 0.0
        for candidate in list_of_texts:
            if candidate:
                similarity = self.compute_similarity(text, candidate)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def clear_cache(self) -> None:
        """Clear the encoding cache."""
        self._encode_cached.cache_clear()
        logger.info("Embedding cache cleared")


# Convenience function for easy access
def get_embedding_model() -> EmbeddingModel:
    """Get the singleton EmbeddingModel instance.
    
    Returns:
        EmbeddingModel singleton instance
    """
    return EmbeddingModel()
