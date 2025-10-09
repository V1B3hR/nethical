"""
Enhanced AI Safety Governance System

This module provides a unified governance system for AI safety.
The implementation has been refactored into multiple modules for better maintainability:

- governance_core.py: Core data models, configuration, persistence, and orchestration
- governance_detectors.py: All detector classes  
- governance_evaluation.py: Judge, intent monitor, and utility functions

This file maintains backward compatibility by re-exporting all public APIs.
"""

# Re-export everything from the modular files for backward compatibility

# Core types and models
from .governance_core import (
    # Enums
    ViolationType,
    SubMission,
    Severity,
    Decision,
    ActionType,
    VIOLATION_SUB_MISSIONS,
    
    # Data Models
    AgentAction,
    SafetyViolation,
    JudgmentResult,
    MonitoringConfig,
    
    # Core system
    PersistenceManager,
    EnhancedSafetyGovernance,
    SafetyGovernance,  # Alias for backward compatibility
)

# Detectors
from .governance_detectors import (
    BaseDetector,
    EthicalViolationDetector,
    SafetyViolationDetector,
    ManipulationDetector,
    PrivacyDetector,
    AdversarialDetector,
    DarkPatternDetector,
    CognitiveWarfareDetector,
    SystemLimitsDetector,
    HallucinationDetector,
    MisinformationDetector,
    ToxicContentDetector,
    ModelExtractionDetector,
    DataPoisoningDetector,
    UnauthorizedAccessDetector,
)

# Evaluation and utilities
from .governance_evaluation import (
    IntentDeviationMonitor,
    SafetyJudge,
    generate_id,
    sha256_content_key,
    entropy,
    looks_like_base64,
    might_be_rot13,
)

# For backward compatibility, expose all public APIs
__all__ = [
    # Enums
    'ViolationType',
    'SubMission',
    'Severity',
    'Decision',
    'ActionType',
    'VIOLATION_SUB_MISSIONS',
    
    # Data Models
    'AgentAction',
    'SafetyViolation',
    'JudgmentResult',
    'MonitoringConfig',
    
    # Core system
    'PersistenceManager',
    'EnhancedSafetyGovernance',
    'SafetyGovernance',
    
    # Detectors
    'BaseDetector',
    'EthicalViolationDetector',
    'SafetyViolationDetector',
    'ManipulationDetector',
    'PrivacyDetector',
    'AdversarialDetector',
    'DarkPatternDetector',
    'CognitiveWarfareDetector',
    'SystemLimitsDetector',
    'HallucinationDetector',
    'MisinformationDetector',
    'ToxicContentDetector',
    'ModelExtractionDetector',
    'DataPoisoningDetector',
    'UnauthorizedAccessDetector',
    
    # Evaluation
    'IntentDeviationMonitor',
    'SafetyJudge',
    
    # Utilities
    'generate_id',
    'sha256_content_key',
    'entropy',
    'looks_like_base64',
    'might_be_rot13',
]
