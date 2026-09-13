"""
Online Learning Pipeline for Detection Intelligence

This module implements continuous learning from operational feedback
to improve detection accuracy over time.

Components:
- FeedbackLoop: Collects feedback from human reviews, appeals, red team
- ModelUpdater: Updates detection models with safety constraints
- ABTestingFramework: A/B tests new detector versions
- RollbackManager: Safe rollback for failed deployments

Law Alignment:
- Law 24 (Adaptive Learning): System improves from experience
- Law 25 (Ethical Evolution): Maintains ethical standards during adaptation
"""

from .feedback_loop import FeedbackLoop, FeedbackType, FeedbackSource, FeedbackEntry
from .model_updater import ModelUpdater, UpdateConstraints
from .ab_testing import ABTestingFramework, TestConfig, TestVariant
from .rollback_manager import RollbackManager, RollbackStrategy, DetectorVersion

__all__ = [
    'FeedbackLoop',
    'FeedbackType',
    'FeedbackSource',
    'FeedbackEntry',
    'ModelUpdater',
    'UpdateConstraints',
    'ABTestingFramework',
    'TestConfig',
    'TestVariant',
    'RollbackManager',
    'RollbackStrategy',
    'DetectorVersion',
]
