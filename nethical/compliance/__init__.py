"""Nethical Compliance Module for Global Compliance Operations.

This module provides comprehensive compliance capabilities for Phase 3:
- Automated Compliance Enforcement (GDPR, EU AI Act, CCPA, ISO 27001, NIST AI RMF)
- Data Residency Management
- Right to Explanation (GDPR Article 22)

Adheres to:
- Law 15: Audit Compliance - Cooperation with auditing
- Law 10: Reasoning Transparency - Explainable decision-making
- Law 12: Limitation Disclosure - Disclosure of known limitations
- Law 22: Digital Security - Protection of digital assets and privacy

Author: Nethical Core Team
Version: 1.0.0
"""

from .gdpr import (
    GDPRComplianceValidator,
    GDPRArticle,
    DataSubjectRight,
    LawfulBasis,
    GDPRValidationResult,
)
from .eu_ai_act import (
    EUAIActValidator,
    AIRiskLevel,
    EUAIActArticle,
    ConformityAssessmentResult,
)
from .data_residency import (
    DataResidencyManager,
    DataRegion,
    DataClassification,
    DataType,
    ResidencyPolicy,
    ResidencyViolation,
)
from .validator import (
    ComplianceValidator,
    ComplianceFramework,
    ComplianceReport,
    ValidationResult,
)

__all__ = [
    # GDPR
    "GDPRComplianceValidator",
    "GDPRArticle",
    "DataSubjectRight",
    "LawfulBasis",
    "GDPRValidationResult",
    # EU AI Act
    "EUAIActValidator",
    "AIRiskLevel",
    "EUAIActArticle",
    "ConformityAssessmentResult",
    # Data Residency
    "DataResidencyManager",
    "DataRegion",
    "DataClassification",
    "DataType",
    "ResidencyPolicy",
    "ResidencyViolation",
    # Validator
    "ComplianceValidator",
    "ComplianceFramework",
    "ComplianceReport",
    "ValidationResult",
]
