"""
Embedding Engine for Universal Vector Language Support.

This module provides embedding generation and management for semantic evaluation
of AI agent actions. It supports multiple embedding providers and integrates
with Nethical's governance framework.

Features:
- Multiple embedding providers (OpenAI, HuggingFace, local)
- Vector similarity computation
- Caching for performance
- Embedding storage and retrieval
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)

# Configuration constants
MAX_INPUT_TEXT_LENGTH = 500  # Maximum characters to store in embedding result


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    
    embedding_id: str
    vector: List[float]
    model: str
    dimensions: int
    input_text: str
    input_hash: str
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def similarity(self, other: "EmbeddingResult") -> float:
        """Compute cosine similarity with another embedding."""
        if self.dimensions != other.dimensions:
            raise ValueError(f"Dimension mismatch: {self.dimensions} vs {other.dimensions}")
        
        return cosine_similarity(self.vector, other.vector)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding vector for input text.
        
        Args:
            text: Input text to embed
            **kwargs: Provider-specific options
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings from this provider."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name/identifier of the model used."""
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """Simple local embedding provider using basic text features.
    
    This is a lightweight fallback that doesn't require external APIs.
    For production use, consider OpenAI or HuggingFace providers.
    """
    
    def __init__(self, dimensions: int = 384):
        """Initialize simple embedding provider.
        
        Args:
            dimensions: Target dimensionality for embeddings
        """
        self.dimensions = dimensions
        
    def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """Generate simple hash-based embedding.
        
        This creates a deterministic embedding based on text content.
        Not suitable for semantic similarity, but useful for testing.
        """
        # Create deterministic hash-based embedding
        text_hash = hashlib.sha256(text.encode('utf-8')).digest()
        
        # Convert hash bytes to float values in [-1, 1]
        vector = []
        for i in range(self.dimensions):
            byte_idx = i % len(text_hash)
            value = (text_hash[byte_idx] / 127.5) - 1.0
            vector.append(value)
        
        # Normalize to unit length
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def get_dimensions(self) -> int:
        return self.dimensions
    
    def get_model_name(self) -> str:
        return "simple-local-embeddings"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using their embeddings API.
    
    Requires: pip install openai
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (text-embedding-3-small, text-embedding-3-large, etc.)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimensions = self._get_model_dimensions()
        
    def _get_model_dimensions(self) -> int:
        """Get dimensions for the configured model."""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model, 1536)
    
    def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_name(self) -> str:
        return f"openai-{self.model}"


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers.
    
    Requires: pip install sentence-transformers
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize HuggingFace embedding provider.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dimensions = self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using HuggingFace model."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_name(self) -> str:
        return f"huggingface-{self.model_name}"


class EmbeddingEngine:
    """Main embedding engine for Nethical governance.
    
    Manages embedding generation, caching, and similarity computation.
    """
    
    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        enable_cache: bool = True,
        cache_size: int = 10000
    ):
        """Initialize embedding engine.
        
        Args:
            provider: Embedding provider to use (defaults to SimpleEmbeddingProvider)
            enable_cache: Whether to cache embeddings
            cache_size: Maximum number of embeddings to cache
        """
        self.provider = provider or SimpleEmbeddingProvider()
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # Cache: input_hash -> EmbeddingResult
        self._cache: Dict[str, EmbeddingResult] = {}
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_generated = 0
        
        logger.info(
            f"EmbeddingEngine initialized with {self.provider.get_model_name()}, "
            f"dimensions={self.provider.get_dimensions()}, cache={enable_cache}"
        )
    
    def embed(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """Generate embedding for text.
        
        Args:
            text: Input text to embed
            metadata: Optional metadata to attach to result
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        # Compute hash for caching
        input_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Check cache
        if self.enable_cache and input_hash in self._cache:
            self._cache_hits += 1
            return self._cache[input_hash]
        
        # Generate embedding
        self._cache_misses += 1
        self._total_generated += 1
        
        vector = self.provider.generate_embedding(text)
        
        result = EmbeddingResult(
            embedding_id=f"emb_{uuid4().hex[:12]}",
            vector=vector,
            model=self.provider.get_model_name(),
            dimensions=self.provider.get_dimensions(),
            input_text=text[:MAX_INPUT_TEXT_LENGTH],  # Store truncated text for debugging
            input_hash=input_hash,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc)
        )
        
        # Cache result
        if self.enable_cache:
            # Simple cache eviction: remove oldest if full
            if len(self._cache) >= self.cache_size:
                # Remove first item (oldest in dict order)
                self._cache.pop(next(iter(self._cache)))
            
            self._cache[input_hash] = result
        
        return result
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0, 1]
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        similarity = emb1.similarity(emb2)
        # Normalize to [0, 1]
        return (similarity + 1.0) / 2.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "provider": self.provider.get_model_name(),
            "dimensions": self.provider.get_dimensions(),
            "cache_enabled": self.enable_cache,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_generated": self._total_generated
        }
