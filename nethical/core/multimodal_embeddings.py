"""
Multi-modal Embedding Support for Universal Vector Language.

This module provides infrastructure for handling embeddings from multiple
modalities: text, code, images, and audio. Enables richer semantic understanding
of diverse agent actions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported modalities for embeddings."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    MIXED = "mixed"


@dataclass
class MultiModalInput:
    """Input data for multi-modal embedding generation."""
    
    # Text/code inputs
    text: Optional[str] = None
    code: Optional[str] = None
    
    # Media inputs (paths or bytes)
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    audio_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    
    # Metadata
    primary_modality: Modality = Modality.TEXT
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get_available_modalities(self) -> List[Modality]:
        """Get list of available modalities in this input."""
        modalities = []
        
        if self.text:
            modalities.append(Modality.TEXT)
        if self.code:
            modalities.append(Modality.CODE)
        if self.image_path or self.image_bytes:
            modalities.append(Modality.IMAGE)
        if self.audio_path or self.audio_bytes:
            modalities.append(Modality.AUDIO)
        
        if len(modalities) > 1:
            modalities.append(Modality.MIXED)
        
        return modalities


@dataclass
class MultiModalEmbeddingResult:
    """Result of multi-modal embedding generation."""
    
    embedding_id: str
    combined_vector: List[float]
    modality_vectors: Dict[Modality, List[float]]
    primary_modality: Modality
    dimensions: int
    model: str
    metadata: Dict[str, Any]


class ModalityDetector:
    """Detects the modality of input content."""
    
    @staticmethod
    def detect_modality(input_data: Union[str, bytes, MultiModalInput]) -> Modality:
        """Detect the primary modality of input data.
        
        Args:
            input_data: Input content to analyze
            
        Returns:
            Detected modality
        """
        if isinstance(input_data, MultiModalInput):
            return input_data.primary_modality
        
        if isinstance(input_data, bytes):
            # Check for image/audio magic numbers
            if ModalityDetector._is_image_bytes(input_data):
                return Modality.IMAGE
            elif ModalityDetector._is_audio_bytes(input_data):
                return Modality.AUDIO
            return Modality.TEXT
        
        if isinstance(input_data, str):
            # Heuristic detection for code vs text
            if ModalityDetector._is_code(input_data):
                return Modality.CODE
            return Modality.TEXT
        
        return Modality.TEXT
    
    @staticmethod
    def _is_code(text: str) -> bool:
        """Heuristic to detect if text is code."""
        code_indicators = [
            "def ", "class ", "function ", "import ", "from ",
            "=>", "->", "const ", "let ", "var ",
            "public ", "private ", "protected ",
            "{", "}", "()", "[]",
        ]
        
        # Count indicators
        indicator_count = sum(1 for indicator in code_indicators if indicator in text)
        
        # Check for typical code structure
        has_structure = (
            text.count('\n') > 0 and
            (text.count('(') > 0 or text.count('{') > 0)
        )
        
        return indicator_count >= 2 or (indicator_count >= 1 and has_structure)
    
    @staticmethod
    def _is_image_bytes(data: bytes) -> bool:
        """Check if bytes represent an image."""
        # Check common image magic numbers
        if len(data) < 4:
            return False
        
        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return True
        # JPEG
        if data[:2] == b'\xff\xd8':
            return True
        # GIF
        if data[:3] == b'GIF':
            return True
        # WebP
        if data[8:12] == b'WEBP':
            return True
        
        return False
    
    @staticmethod
    def _is_audio_bytes(data: bytes) -> bool:
        """Check if bytes represent audio."""
        if len(data) < 4:
            return False
        
        # WAV
        if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
            return True
        # MP3
        if data[:3] == b'ID3' or data[:2] == b'\xff\xfb':
            return True
        # OGG
        if data[:4] == b'OggS':
            return True
        
        return False


class MultiModalEmbeddingEngine:
    """Engine for generating multi-modal embeddings."""
    
    def __init__(
        self,
        text_embedding_engine=None,
        enable_image: bool = False,
        enable_audio: bool = False,
        fusion_strategy: Literal["concatenate", "weighted_sum", "attention"] = "weighted_sum",
        text_weight: float = 0.7,
        code_weight: float = 0.3,
        image_weight: float = 0.5,
        audio_weight: float = 0.5,
    ):
        """Initialize multi-modal embedding engine.
        
        Args:
            text_embedding_engine: Engine for text/code embeddings
            enable_image: Whether to enable image embeddings
            enable_audio: Whether to enable audio embeddings
            fusion_strategy: Strategy for combining modality embeddings
            text_weight: Weight for text embeddings in fusion
            code_weight: Weight for code embeddings in fusion
            image_weight: Weight for image embeddings in fusion
            audio_weight: Weight for audio embeddings in fusion
        """
        from .embedding_engine import EmbeddingEngine
        
        self.text_engine = text_embedding_engine or EmbeddingEngine()
        self.enable_image = enable_image
        self.enable_audio = enable_audio
        self.fusion_strategy = fusion_strategy
        
        self.weights = {
            Modality.TEXT: text_weight,
            Modality.CODE: code_weight,
            Modality.IMAGE: image_weight,
            Modality.AUDIO: audio_weight,
        }
        
        # Initialize modality-specific engines
        self.image_engine = None
        self.audio_engine = None
        
        if enable_image:
            self._initialize_image_engine()
        if enable_audio:
            self._initialize_audio_engine()
        
        self.modality_detector = ModalityDetector()
        
        logger.info(
            f"MultiModalEmbeddingEngine initialized: "
            f"text=enabled, image={'enabled' if enable_image else 'disabled'}, "
            f"audio={'enabled' if enable_audio else 'disabled'}, "
            f"fusion={fusion_strategy}"
        )
    
    def _initialize_image_engine(self):
        """Initialize image embedding engine (placeholder for future implementation)."""
        # TODO: Integrate CLIP, DINOv2, or similar image embedding models
        logger.warning(
            "Image embeddings requested but not fully implemented. "
            "Install transformers and torch, then use CLIP or similar models."
        )
        self.enable_image = False
    
    def _initialize_audio_engine(self):
        """Initialize audio embedding engine (placeholder for future implementation)."""
        # TODO: Integrate Wav2Vec2, Whisper embeddings, or similar audio models
        logger.warning(
            "Audio embeddings requested but not fully implemented. "
            "Install transformers and torch, then use Wav2Vec2 or similar models."
        )
        self.enable_audio = False
    
    def embed(
        self,
        input_data: Union[str, MultiModalInput],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MultiModalEmbeddingResult:
        """Generate multi-modal embedding.
        
        Args:
            input_data: Input to embed (text string or MultiModalInput)
            metadata: Optional metadata
            
        Returns:
            MultiModalEmbeddingResult with combined and per-modality vectors
        """
        # Convert simple string input to MultiModalInput
        if isinstance(input_data, str):
            modality = self.modality_detector.detect_modality(input_data)
            if modality == Modality.CODE:
                input_data = MultiModalInput(code=input_data, primary_modality=Modality.CODE)
            else:
                input_data = MultiModalInput(text=input_data, primary_modality=Modality.TEXT)
        
        # Generate embeddings for each available modality
        modality_vectors = {}
        
        if input_data.text:
            text_result = self.text_engine.embed(input_data.text)
            modality_vectors[Modality.TEXT] = text_result.vector
        
        if input_data.code:
            # For code, we can use enhanced code-aware processing
            code_result = self._embed_code(input_data.code)
            modality_vectors[Modality.CODE] = code_result
        
        if input_data.image_path or input_data.image_bytes:
            if self.enable_image and self.image_engine:
                image_vector = self._embed_image(input_data)
                modality_vectors[Modality.IMAGE] = image_vector
        
        if input_data.audio_path or input_data.audio_bytes:
            if self.enable_audio and self.audio_engine:
                audio_vector = self._embed_audio(input_data)
                modality_vectors[Modality.AUDIO] = audio_vector
        
        # Fuse embeddings
        combined_vector = self._fuse_embeddings(modality_vectors)
        
        # Generate result
        from uuid import uuid4
        result = MultiModalEmbeddingResult(
            embedding_id=f"mm_emb_{uuid4().hex[:12]}",
            combined_vector=combined_vector,
            modality_vectors=modality_vectors,
            primary_modality=input_data.primary_modality,
            dimensions=len(combined_vector),
            model=f"multimodal-{self.fusion_strategy}",
            metadata=metadata or {}
        )
        
        return result
    
    def _embed_code(self, code: str) -> List[float]:
        """Generate code-aware embedding.
        
        Enhances code embeddings by combining raw text with semantic structure.
        """
        # Basic implementation: use text engine with code-specific preprocessing
        # TODO: Could integrate CodeBERT or GraphCodeBERT for better code understanding
        
        # Add code structure hints
        enhanced_code = f"CODE: {code}"
        result = self.text_engine.embed(enhanced_code)
        return result.vector
    
    def _embed_image(self, input_data: MultiModalInput) -> List[float]:
        """Generate image embedding (placeholder)."""
        # TODO: Implement using CLIP or similar
        raise NotImplementedError("Image embeddings not yet implemented")
    
    def _embed_audio(self, input_data: MultiModalInput) -> List[float]:
        """Generate audio embedding (placeholder)."""
        # TODO: Implement using Wav2Vec2 or similar
        raise NotImplementedError("Audio embeddings not yet implemented")
    
    def _fuse_embeddings(self, modality_vectors: Dict[Modality, List[float]]) -> List[float]:
        """Fuse embeddings from multiple modalities.
        
        Args:
            modality_vectors: Dictionary mapping modalities to their embedding vectors
            
        Returns:
            Combined embedding vector
        """
        if not modality_vectors:
            # Return zero vector if no embeddings
            return [0.0] * 384
        
        if len(modality_vectors) == 1:
            # Single modality, just return it
            return list(modality_vectors.values())[0]
        
        if self.fusion_strategy == "concatenate":
            # Concatenate all vectors
            combined = []
            for modality in [Modality.TEXT, Modality.CODE, Modality.IMAGE, Modality.AUDIO]:
                if modality in modality_vectors:
                    combined.extend(modality_vectors[modality])
            return combined
        
        elif self.fusion_strategy == "weighted_sum":
            # Weighted sum of vectors (must be same dimension)
            # Find common dimension
            dimensions = set(len(v) for v in modality_vectors.values())
            if len(dimensions) > 1:
                # Pad/truncate to match smallest dimension
                target_dim = min(dimensions)
            else:
                target_dim = dimensions.pop()
            
            combined = [0.0] * target_dim
            total_weight = 0.0
            
            for modality, vector in modality_vectors.items():
                weight = self.weights.get(modality, 1.0)
                total_weight += weight
                
                for i in range(min(target_dim, len(vector))):
                    combined[i] += vector[i] * weight
            
            # Normalize by total weight
            if total_weight > 0:
                combined = [v / total_weight for v in combined]
            
            return combined
        
        elif self.fusion_strategy == "attention":
            # TODO: Implement attention-based fusion
            # For now, fall back to weighted sum
            return self._fuse_embeddings_weighted_sum(modality_vectors)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "text_engine_stats": self.text_engine.get_stats(),
            "image_enabled": self.enable_image,
            "audio_enabled": self.enable_audio,
            "fusion_strategy": self.fusion_strategy,
            "weights": self.weights,
        }
        return stats
