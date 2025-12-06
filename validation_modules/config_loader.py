"""
Configuration Loader

Loads and manages validation configuration from YAML files
with environment variable override support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationConfig:
    """Validation configuration with environment variable overrides"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to validation.yaml (default: validation.yaml in project root)
        """
        if config_path is None:
            # Try multiple locations
            possible_paths = [
                Path("validation.yaml"),
                Path("config/validation.yaml"),
                Path("validation_config.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Configuration file not found, using defaults")
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "global": {
                "random_seed": 42,
                "artifacts_dir": "artifacts/validation",
                "log_level": "INFO"
            },
            "ethics_benchmark": {
                "enabled": True,
                "thresholds": {
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90
                }
            },
            "performance": {
                "enabled": True,
                "thresholds": {
                    "accuracy": 0.85,
                    "precision": 0.85,
                    "recall": 0.80,
                    "f1_score": 0.82
                }
            },
            "data_integrity": {
                "enabled": True,
                "drift": {
                    "psi_threshold": 0.2,
                    "ks_test_alpha": 0.05
                }
            },
            "explainability": {
                "enabled": True,
                "thresholds": {
                    "min_stability": 0.8,
                    "min_coverage": 0.95
                }
            }
        }
    
    # Environment variable prefix for overrides
    ENV_VAR_PREFIX = "VALIDATION_"
    
    def _get_env_key(self, key_path: str) -> str:
        """Convert config key path to environment variable name"""
        return f"{self.ENV_VAR_PREFIX}{key_path.upper().replace('.', '_')}"
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation and environment variable override
        
        Args:
            key_path: Dot-separated path (e.g., "ethics_benchmark.thresholds.precision")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        # Check for environment variable override
        env_key = self._get_env_key(key_path)
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Try to parse as appropriate type
            try:
                return float(env_value)
            except ValueError:
                try:
                    return int(env_value)
                except ValueError:
                    if env_value.lower() in ['true', 'false']:
                        return env_value.lower() == 'true'
                    return env_value
        
        # Navigate through config dictionary
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_suite_config(self, suite_name: str) -> Dict[str, Any]:
        """
        Get full configuration for a suite
        
        Args:
            suite_name: Name of the validation suite
            
        Returns:
            Suite configuration dictionary
        """
        return self.config.get(suite_name, {})
    
    def get_global(self, key: str, default: Any = None) -> Any:
        """Get global configuration value"""
        return self.get(f"global.{key}", default)
    
    def is_enabled(self, suite_name: str) -> bool:
        """Check if a validation suite is enabled"""
        return self.get(f"{suite_name}.enabled", True)
    
    def get_threshold(self, suite_name: str, metric: str, default: float = 0.0) -> float:
        """Get threshold value for a metric"""
        return self.get(f"{suite_name}.thresholds.{metric}", default)
    
    def get_artifacts_dir(self, suite_name: Optional[str] = None) -> Path:
        """Get artifacts directory path"""
        base_dir = Path(self.get_global("artifacts_dir", "artifacts/validation"))
        if suite_name:
            return base_dir / suite_name
        return base_dir
