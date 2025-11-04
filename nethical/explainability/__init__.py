"""
Explainability module for Nethical - Provides explanations for AI safety decisions.

This module implements the Explainable AI layer as specified in roadmap Phase 2.3.
It provides natural language explanations for policy decisions, violation detections,
and risk assessments.
"""

from .decision_explainer import DecisionExplainer
from .natural_language_generator import NaturalLanguageGenerator
from .transparency_report import TransparencyReportGenerator

__all__ = [
    "DecisionExplainer",
    "NaturalLanguageGenerator",
    "TransparencyReportGenerator",
]
