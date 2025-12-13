"""
Threat Intelligence and Predictive Detection Module.

This module implements Phase 5: Detection Omniscience capabilities including:
- Threat intelligence integration from multiple sources
- Predictive modeling for attack anticipation
- Proactive hardening based on threat predictions

Phase: 5 - Detection Omniscience
Status: Active
"""

from .threat_feed_integration import (
    ThreatFeedIntegrator,
    ThreatSource,
    ThreatIntelligence,
    ThreatSeverity,
)
from .predictive_modeling import (
    PredictiveModeler,
    AttackPrediction,
    ThreatEvolutionModel,
)
from .proactive_hardening import (
    ProactiveHardener,
    HardeningAction,
    HardeningPriority,
)

__all__ = [
    "ThreatFeedIntegrator",
    "ThreatSource",
    "ThreatIntelligence",
    "ThreatSeverity",
    "PredictiveModeler",
    "AttackPrediction",
    "ThreatEvolutionModel",
    "ProactiveHardener",
    "HardeningAction",
    "HardeningPriority",
]
