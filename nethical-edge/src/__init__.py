"""
Nethical Edge - Standalone Edge Deployment Package

Ultra-low latency governance for edge devices.
Designed for autonomous vehicles, robots, and safety-critical systems.

Target: <10ms p99 latency
Mode: Offline-first with CRDT sync
"""

from nethical.edge import (
    EdgeGovernor,
    EdgeDecision,
    DecisionType,
    PolicyCache,
    FastDetector,
    SafeDefaults,
    PredictiveEngine,
    OfflineFallback,
    CircuitBreaker,
    SyncManager,
)

from nethical.sync import (
    PolicyCRDT,
    VectorClock,
    HybridLogicalClock,
    AntiEntropyProtocol,
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "EdgeGovernor",
    "EdgeDecision",
    "DecisionType",
    # Components
    "PolicyCache",
    "FastDetector",
    "SafeDefaults",
    "PredictiveEngine",
    "OfflineFallback",
    "CircuitBreaker",
    "SyncManager",
    # Sync
    "PolicyCRDT",
    "VectorClock",
    "HybridLogicalClock",
    "AntiEntropyProtocol",
]


def create_governor(
    device_id: str,
    mode: str = "standard",
    config_path: str = None,
    **kwargs,
) -> EdgeGovernor:
    """
    Create an EdgeGovernor with the specified configuration.

    Args:
        device_id: Unique identifier for this device
        mode: Operating mode ("minimal", "standard", "full")
        config_path: Path to configuration file
        **kwargs: Additional configuration options

    Returns:
        Configured EdgeGovernor instance
    """
    # Mode-specific defaults
    mode_configs = {
        "minimal": {
            "cache_size_mb": 16,
            "disable_jit": True,
            "predictive_enabled": False,
            "max_latency_ms": 10.0,
        },
        "standard": {
            "cache_size_mb": 64,
            "disable_jit": False,
            "predictive_enabled": True,
            "max_latency_ms": 8.0,
        },
        "full": {
            "cache_size_mb": 256,
            "disable_jit": False,
            "predictive_enabled": True,
            "max_latency_ms": 5.0,
        },
    }

    config = mode_configs.get(mode, mode_configs["standard"])
    config.update(kwargs)

    # Load config file if provided
    if config_path:
        try:
            import yaml
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config and "edge" in file_config:
                    config.update(file_config["edge"])
        except FileNotFoundError:
            import logging
            logging.warning(f"Config file not found: {config_path}, using defaults")
        except yaml.YAMLError as e:
            import logging
            logging.error(f"Invalid YAML in config file {config_path}: {e}")
            raise ValueError(f"Invalid configuration file: {config_path}") from e
        except Exception as e:
            import logging
            logging.error(f"Error loading config file {config_path}: {e}")
            raise

    return EdgeGovernor(
        agent_id=device_id,
        max_latency_ms=config.get("max_latency_ms", 10.0),
    )
