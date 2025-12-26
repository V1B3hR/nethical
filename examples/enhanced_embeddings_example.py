#!/usr/bin/env python3
"""
Example: Using Enhanced UVL with Multiple Embedding Providers.

This example demonstrates the new embedding provider features including:
- Multiple provider types (OpenAI, HuggingFace, Simple)
- Ensemble embeddings for improved accuracy
- Fallback mechanisms
- Configuration-based setup
"""

from nethical import Nethical, Agent
from nethical.core import (
    EmbeddingConfig,
    load_embedding_config,
)


def example_simple_provider():
    """Example 1: Using simple local provider (no API keys needed)."""
    print("\n" + "="*70)
    print("Example 1: Simple Local Provider")
    print("="*70)
    
    nethical = Nethical(
        enable_25_laws=True,
        storage_dir="/tmp/example_simple"
    )
    
    agent = Agent(id="simple-agent", type="coding", capabilities=["code"])
    nethical.register_agent(agent)
    
    result = nethical.evaluate(
        agent_id="simple-agent",
        action="def hello(): return 'Hello, World!'",
        context={"purpose": "demo"}
    )
    
    print(f"Decision: {result.decision}")
    print(f"Risk Score: {result.risk_score:.2f}")
    print(f"Laws Evaluated: {result.laws_evaluated[:5]}")
    print(f"Embedding Model: {nethical.governance.embedding_engine.provider.get_model_name()}")


def example_openai_with_fallback():
    """Example 2: OpenAI with automatic fallback to local."""
    print("\n" + "="*70)
    print("Example 2: OpenAI with Fallback")
    print("="*70)
    
    # Create config with OpenAI primary and Simple fallback
    config = EmbeddingConfig.openai_default()
    
    print(f"Primary Provider: {config.primary_provider.provider_type}")
    print(f"Fallbacks: {[fb.provider_type for fb in config.fallback_providers]}")
    
    # Note: If OPENAI_API_KEY is not set, will automatically fall back
    # to simple provider without crashing
    
    nethical = Nethical(
        enable_25_laws=True,
        storage_dir="/tmp/example_openai"
    )
    
    # The embedding engine will use the config
    from nethical.core import EmbeddingEngine
    nethical.governance.embedding_engine = EmbeddingEngine(config=config)
    
    agent = Agent(id="openai-agent", type="coding", capabilities=["code"])
    nethical.register_agent(agent)
    
    result = nethical.evaluate(
        agent_id="openai-agent",
        action="Access user database to read preferences",
        context={"purpose": "personalization"}
    )
    
    print(f"Decision: {result.decision}")
    print(f"Risk Score: {result.risk_score:.2f}")
    print(f"Detected Primitives: {result.detected_primitives[:3]}")
    
    # Check stats to see if fallback was used
    stats = nethical.governance.embedding_engine.get_stats()
    print(f"Provider Failures: {stats.get('provider_failures', 0)}")
    print(f"Fallback Uses: {stats.get('fallback_uses', 0)}")


def example_ensemble_embeddings():
    """Example 3: Ensemble embeddings for maximum accuracy."""
    print("\n" + "="*70)
    print("Example 3: Ensemble Embeddings (+15% Accuracy)")
    print("="*70)
    
    # Create ensemble config
    config = EmbeddingConfig.ensemble_default()
    
    print(f"Ensemble Strategy: {config.ensemble_strategy}")
    print(f"Ensemble Providers: {[p.provider_type for p in config.ensemble_providers]}")
    print(f"Weights: {[p.weight for p in config.ensemble_providers]}")
    
    nethical = Nethical(
        enable_25_laws=True,
        storage_dir="/tmp/example_ensemble"
    )
    
    # Set up ensemble engine
    from nethical.core import EmbeddingEngine
    nethical.governance.embedding_engine = EmbeddingEngine(config=config)
    
    agent = Agent(id="ensemble-agent", type="general", capabilities=["all"])
    nethical.register_agent(agent)
    
    # Test complex action that benefits from ensemble
    result = nethical.evaluate(
        agent_id="ensemble-agent",
        action="Analyze user interaction data, train recommendation model, and deploy to production",
        context={"purpose": "ml_training"}
    )
    
    print(f"Decision: {result.decision}")
    print(f"Risk Score: {result.risk_score:.2f}")
    print(f"Laws Evaluated: {len(result.laws_evaluated)} laws")
    print(f"Detected Primitives: {result.detected_primitives}")
    
    stats = nethical.governance.embedding_engine.get_stats()
    print(f"Ensemble Uses: {stats.get('ensemble_uses', 0)}")


def example_config_from_file():
    """Example 4: Load configuration from YAML file."""
    print("\n" + "="*70)
    print("Example 4: Configuration from File")
    print("="*70)
    
    # Load from config file
    config = load_embedding_config(
        config_path="config/embedding_config.yaml",
        use_env=True
    )
    
    print(f"Loaded config from file")
    print(f"Primary Provider: {config.primary_provider.provider_type}")
    print(f"Cache Enabled: {config.enable_cache}")
    print(f"Similarity Threshold: {config.similarity_threshold}")
    
    nethical = Nethical(
        enable_25_laws=True,
        storage_dir="/tmp/example_config"
    )
    
    from nethical.core import EmbeddingEngine
    nethical.governance.embedding_engine = EmbeddingEngine(config=config)
    
    agent = Agent(id="config-agent", type="data", capabilities=["data"])
    nethical.register_agent(agent)
    
    result = nethical.evaluate(
        agent_id="config-agent",
        action="SELECT * FROM users WHERE age < 18",
        context={"purpose": "analytics"}
    )
    
    print(f"Decision: {result.decision}")
    print(f"Laws Evaluated: {result.laws_evaluated}")


def example_environment_config():
    """Example 5: Configuration from environment variables."""
    print("\n" + "="*70)
    print("Example 5: Environment-based Configuration")
    print("="*70)
    
    # Set environment variables:
    # export NETHICAL_EMBEDDING_PROVIDER=openai
    # export NETHICAL_EMBEDDING_MODEL=text-embedding-3-large
    # export NETHICAL_EMBEDDING_FALLBACK=simple
    # export OPENAI_API_KEY=your_key_here
    
    config = EmbeddingConfig.from_env()
    
    print(f"Provider from env: {config.primary_provider.provider_type}")
    print(f"Model: {config.primary_provider.model_name}")
    print(f"Fallbacks: {[fb.provider_type for fb in config.fallback_providers]}")
    
    print("\nTo use environment configuration, set:")
    print("  NETHICAL_EMBEDDING_PROVIDER=openai|openai_large|huggingface|simple")
    print("  NETHICAL_EMBEDDING_MODEL=model_name")
    print("  NETHICAL_EMBEDDING_FALLBACK=fallback_provider")
    print("  OPENAI_API_KEY=your_key")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ENHANCED UVL EMBEDDING PROVIDER EXAMPLES")
    print("="*70)
    
    try:
        example_simple_provider()
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    try:
        example_openai_with_fallback()
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    try:
        example_ensemble_embeddings()
    except Exception as e:
        print(f"Error in example 3: {e}")
    
    try:
        example_config_from_file()
    except Exception as e:
        print(f"Error in example 4: {e}")
    
    try:
        example_environment_config()
    except Exception as e:
        print(f"Error in example 5: {e}")
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
