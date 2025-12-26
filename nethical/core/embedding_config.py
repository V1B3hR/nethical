"""
Embedding Configuration Module for Universal Vector Language.

This module provides centralized configuration for embedding providers,
supporting multiple providers, ensemble configurations, and fallback strategies.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class EmbeddingProviderType(str, Enum):
    """Available embedding provider types."""
    OPENAI = "openai"
    OPENAI_LARGE = "openai_large"
    HUGGINGFACE = "huggingface"
    SIMPLE = "simple"


class EnsembleStrategy(str, Enum):
    """Strategies for combining multiple embeddings."""
    AVERAGE = "average"
    WEIGHTED = "weighted"
    MAX_POOLING = "max_pooling"
    CONCATENATE = "concatenate"


@dataclass
class ProviderConfig:
    """Configuration for a single embedding provider."""
    
    provider_type: EmbeddingProviderType
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    dimensions: Optional[int] = None
    weight: float = 1.0  # For weighted ensemble
    enabled: bool = True
    fallback_order: int = 0  # Lower = higher priority
    
    # Performance options
    batch_size: int = 32
    timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Advanced options
    custom_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set defaults based on provider type."""
        if self.model_name is None:
            if self.provider_type == EmbeddingProviderType.OPENAI:
                self.model_name = "text-embedding-3-small"
            elif self.provider_type == EmbeddingProviderType.OPENAI_LARGE:
                self.model_name = "text-embedding-3-large"
            elif self.provider_type == EmbeddingProviderType.HUGGINGFACE:
                self.model_name = "sentence-transformers/all-mpnet-base-v2"
            elif self.provider_type == EmbeddingProviderType.SIMPLE:
                self.model_name = "simple-local-embeddings"
        
        if self.dimensions is None:
            if self.provider_type == EmbeddingProviderType.OPENAI:
                self.dimensions = 1536
            elif self.provider_type == EmbeddingProviderType.OPENAI_LARGE:
                self.dimensions = 3072
            elif self.provider_type == EmbeddingProviderType.HUGGINGFACE:
                self.dimensions = 768
            elif self.provider_type == EmbeddingProviderType.SIMPLE:
                self.dimensions = 384
        
        # Get API key from environment if not provided
        if self.api_key is None and self.provider_type in [
            EmbeddingProviderType.OPENAI,
            EmbeddingProviderType.OPENAI_LARGE
        ]:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class EmbeddingConfig:
    """Complete embedding configuration for Universal Vector Language."""
    
    # Primary provider
    primary_provider: ProviderConfig
    
    # Fallback providers (in priority order)
    fallback_providers: List[ProviderConfig] = field(default_factory=list)
    
    # Ensemble configuration
    enable_ensemble: bool = False
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.AVERAGE
    ensemble_providers: List[ProviderConfig] = field(default_factory=list)
    
    # Caching and performance
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: Optional[float] = None
    
    # Similarity and thresholds
    similarity_threshold: float = 0.7
    use_normalized_similarity: bool = True  # Normalize cosine sim to [0,1]
    
    # Multi-modal support
    enable_multimodal: bool = False
    text_weight: float = 0.7
    code_weight: float = 0.3
    
    # Fine-tuning and feedback
    enable_feedback_logging: bool = False
    feedback_log_path: Optional[str] = None
    
    # Advanced features
    enable_gpu_acceleration: bool = False
    enable_federated: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> EmbeddingConfig:
        """Create EmbeddingConfig from dictionary."""
        # Parse primary provider
        primary_config = config_dict.get("primary_provider", {})
        primary_provider = ProviderConfig(
            provider_type=EmbeddingProviderType(primary_config.get("type", "simple")),
            model_name=primary_config.get("model_name"),
            api_key=primary_config.get("api_key"),
            dimensions=primary_config.get("dimensions"),
        )
        
        # Parse fallback providers
        fallback_providers = []
        for i, fb_config in enumerate(config_dict.get("fallback_providers", [])):
            fallback_providers.append(ProviderConfig(
                provider_type=EmbeddingProviderType(fb_config.get("type")),
                model_name=fb_config.get("model_name"),
                api_key=fb_config.get("api_key"),
                dimensions=fb_config.get("dimensions"),
                fallback_order=i,
            ))
        
        # Parse ensemble providers
        ensemble_providers = []
        for ens_config in config_dict.get("ensemble_providers", []):
            ensemble_providers.append(ProviderConfig(
                provider_type=EmbeddingProviderType(ens_config.get("type")),
                model_name=ens_config.get("model_name"),
                api_key=ens_config.get("api_key"),
                dimensions=ens_config.get("dimensions"),
                weight=ens_config.get("weight", 1.0),
            ))
        
        return cls(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            enable_ensemble=config_dict.get("enable_ensemble", False),
            ensemble_strategy=EnsembleStrategy(
                config_dict.get("ensemble_strategy", "average")
            ),
            ensemble_providers=ensemble_providers,
            enable_cache=config_dict.get("enable_cache", True),
            cache_size=config_dict.get("cache_size", 10000),
            cache_ttl_seconds=config_dict.get("cache_ttl_seconds"),
            similarity_threshold=config_dict.get("similarity_threshold", 0.7),
            use_normalized_similarity=config_dict.get("use_normalized_similarity", True),
            enable_multimodal=config_dict.get("enable_multimodal", False),
            text_weight=config_dict.get("text_weight", 0.7),
            code_weight=config_dict.get("code_weight", 0.3),
            enable_feedback_logging=config_dict.get("enable_feedback_logging", False),
            feedback_log_path=config_dict.get("feedback_log_path"),
            enable_gpu_acceleration=config_dict.get("enable_gpu_acceleration", False),
            enable_federated=config_dict.get("enable_federated", False),
        )
    
    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        """Create EmbeddingConfig from environment variables."""
        provider_type = os.getenv(
            "NETHICAL_EMBEDDING_PROVIDER", 
            EmbeddingProviderType.SIMPLE
        )
        
        if isinstance(provider_type, str):
            try:
                provider_type = EmbeddingProviderType(provider_type.lower())
            except ValueError:
                logger.warning(
                    f"Invalid provider type '{provider_type}', falling back to SIMPLE"
                )
                provider_type = EmbeddingProviderType.SIMPLE
        
        primary_provider = ProviderConfig(
            provider_type=provider_type,
            model_name=os.getenv("NETHICAL_EMBEDDING_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # Check for fallback provider
        fallback_providers = []
        fallback_type = os.getenv("NETHICAL_EMBEDDING_FALLBACK")
        if fallback_type:
            try:
                fallback_providers.append(ProviderConfig(
                    provider_type=EmbeddingProviderType(fallback_type.lower()),
                    fallback_order=0,
                ))
            except ValueError:
                logger.warning(f"Invalid fallback provider type '{fallback_type}'")
        
        return cls(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            enable_cache=os.getenv("NETHICAL_EMBEDDING_CACHE", "true").lower() == "true",
            cache_size=int(os.getenv("NETHICAL_EMBEDDING_CACHE_SIZE", "10000")),
            similarity_threshold=float(os.getenv("NETHICAL_SIMILARITY_THRESHOLD", "0.7")),
            enable_feedback_logging=os.getenv(
                "NETHICAL_FEEDBACK_LOGGING", "false"
            ).lower() == "true",
            feedback_log_path=os.getenv("NETHICAL_FEEDBACK_LOG_PATH"),
        )
    
    @classmethod
    def default(cls) -> EmbeddingConfig:
        """Create default EmbeddingConfig with simple provider."""
        return cls(
            primary_provider=ProviderConfig(
                provider_type=EmbeddingProviderType.SIMPLE,
            ),
        )
    
    @classmethod
    def openai_default(cls) -> EmbeddingConfig:
        """Create default config with OpenAI provider."""
        return cls(
            primary_provider=ProviderConfig(
                provider_type=EmbeddingProviderType.OPENAI,
            ),
            fallback_providers=[
                ProviderConfig(
                    provider_type=EmbeddingProviderType.SIMPLE,
                    fallback_order=0,
                )
            ],
        )
    
    @classmethod
    def openai_large_default(cls) -> EmbeddingConfig:
        """Create default config with OpenAI large model for maximum accuracy."""
        return cls(
            primary_provider=ProviderConfig(
                provider_type=EmbeddingProviderType.OPENAI_LARGE,
            ),
            fallback_providers=[
                ProviderConfig(
                    provider_type=EmbeddingProviderType.OPENAI,
                    fallback_order=0,
                ),
                ProviderConfig(
                    provider_type=EmbeddingProviderType.SIMPLE,
                    fallback_order=1,
                )
            ],
        )
    
    @classmethod
    def huggingface_default(cls) -> EmbeddingConfig:
        """Create default config with HuggingFace provider."""
        return cls(
            primary_provider=ProviderConfig(
                provider_type=EmbeddingProviderType.HUGGINGFACE,
            ),
            fallback_providers=[
                ProviderConfig(
                    provider_type=EmbeddingProviderType.SIMPLE,
                    fallback_order=0,
                )
            ],
        )
    
    @classmethod
    def ensemble_default(cls) -> EmbeddingConfig:
        """Create default ensemble config for maximum accuracy."""
        return cls(
            primary_provider=ProviderConfig(
                provider_type=EmbeddingProviderType.OPENAI_LARGE,
            ),
            enable_ensemble=True,
            ensemble_strategy=EnsembleStrategy.WEIGHTED,
            ensemble_providers=[
                ProviderConfig(
                    provider_type=EmbeddingProviderType.OPENAI_LARGE,
                    weight=0.6,
                ),
                ProviderConfig(
                    provider_type=EmbeddingProviderType.HUGGINGFACE,
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    weight=0.4,
                )
            ],
        )


def load_embedding_config(
    config_path: Optional[str] = None,
    use_env: bool = True,
    default_type: Literal["simple", "openai", "openai_large", "huggingface", "ensemble"] = "simple"
) -> EmbeddingConfig:
    """Load embedding configuration from file, environment, or defaults.
    
    Args:
        config_path: Path to YAML/JSON config file (optional)
        use_env: Whether to use environment variables
        default_type: Default provider type if no config found
        
    Returns:
        EmbeddingConfig instance
    """
    # Try loading from file first
    if config_path:
        import yaml
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                if 'embedding' in config_dict:
                    config_dict = config_dict['embedding']
                return EmbeddingConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Try environment variables
    if use_env and os.getenv("NETHICAL_EMBEDDING_PROVIDER"):
        return EmbeddingConfig.from_env()
    
    # Use defaults based on type
    if default_type == "openai":
        return EmbeddingConfig.openai_default()
    elif default_type == "openai_large":
        return EmbeddingConfig.openai_large_default()
    elif default_type == "huggingface":
        return EmbeddingConfig.huggingface_default()
    elif default_type == "ensemble":
        return EmbeddingConfig.ensemble_default()
    else:
        return EmbeddingConfig.default()
