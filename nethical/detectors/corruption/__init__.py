"""
Comprehensive Corruption Intelligence Detection Module

This module provides intelligent detection of all forms of corruption across
multiple vectors (Human↔AI, AI↔AI, Human↔Human via AI) with multi-detector
correlation, entity profiling, and relationship graph analysis.

Based on Investopedia corruption taxonomy with AI-specific extensions:
- Bribery, Extortion, Embezzlement
- Nepotism, Cronyism, Fraud
- Kickbacks, Influence Peddling, Quid Pro Quo, Collusion
- Data/Compute/Access/Capability Corruption
- Regulatory Capture

Features:
- Multi-vector detection (Human→AI, AI→Human, AI→AI, Proxy)
- Corruption lifecycle phase detection (reconnaissance → maintenance)
- Integration with existing detectors (Manipulation, DarkPattern, Behavioral, Session)
- Long-term entity profiling and corruption risk scoring
- Relationship graph analysis for collusion detection
- Multi-detector signal correlation
- Explainable reasoning chains

Author: Nethical Core Team
Version: 1.0.0
"""

from .corruption_types import (
    CorruptionType,
    CorruptionVector,
    CorruptionPhase,
    RiskLevel,
    RecommendedAction,
    CorruptionEvidence,
    EntityProfile,
    RelationshipEdge,
    CorruptionAssessment,
    CorruptionPattern,
)

from .corruption_patterns import CorruptionPatternLibrary
from .detector_bridge import DetectorBridge
from .intelligence_engine import IntelligenceEngine
from .corruption_detector import CorruptionDetector

__all__ = [
    # Enums and types
    "CorruptionType",
    "CorruptionVector",
    "CorruptionPhase",
    "RiskLevel",
    "RecommendedAction",
    
    # Data structures
    "CorruptionEvidence",
    "EntityProfile",
    "RelationshipEdge",
    "CorruptionAssessment",
    "CorruptionPattern",
    
    # Components
    "CorruptionPatternLibrary",
    "DetectorBridge",
    "IntelligenceEngine",
    
    # Main detector
    "CorruptionDetector",
]

__version__ = "1.0.0"
