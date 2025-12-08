"""
Example Custom Detectors for F2: Detector & Policy Extensibility

This module provides example detector plugins demonstrating how to create
industry-specific and organization-specific detectors.

Examples included:
1. FinancialComplianceDetector - For financial services compliance
2. HealthcareComplianceDetector - For HIPAA and healthcare compliance
3. CustomPolicyDetector - For organization-specific policies
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from nethical.core.plugin_interface import DetectorPlugin, PluginMetadata
from nethical.detectors.base_detector import SafetyViolation


class FinancialComplianceDetector(DetectorPlugin):
    """
    Detector for financial compliance violations.

    Detects:
    - Unencrypted financial data transmission
    - PCI-DSS compliance violations
    - SOX compliance violations
    - Financial data handling without proper authorization
    """

    def __init__(self):
        """Initialize the financial compliance detector."""
        super().__init__(name="FinancialComplianceDetector", version="1.0.0")

        # Financial data patterns
        self.financial_patterns = {
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "bank_account": r"\b\d{8,17}\b",
            "routing_number": r"\b\d{9}\b",
            "swift_code": r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
            "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
        }

        # Financial keywords
        self.financial_keywords = [
            "financial_data",
            "transaction",
            "payment",
            "credit_card",
            "bank_account",
            "wire_transfer",
            "investment",
            "securities",
            "trading",
            "portfolio",
            "dividend",
            "interest_rate",
        ]

        # Compliance requirements
        self.compliance_requirements = {
            "encryption_required": True,
            "audit_required": True,
            "authorization_required": True,
        }

    async def detect_violations(self, action: Any) -> Sequence[SafetyViolation] | None:
        """Detect financial compliance violations."""
        violations = []

        # Get action content
        content = self._get_action_content(action)
        context = self._get_action_context(action)

        # Check for financial data patterns
        detected_patterns = []
        for pattern_name, pattern in self.financial_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected_patterns.append(pattern_name)

        # Check for financial keywords
        detected_keywords = [
            kw
            for kw in self.financial_keywords
            if kw.replace("_", " ") in content.lower()
        ]

        if detected_patterns or detected_keywords:
            # Check encryption
            if not self._check_encryption(context):
                violations.append(
                    SafetyViolation(
                        detector=self.name,
                        severity="high",
                        description="Financial data transmitted without encryption",
                        category="financial_compliance",
                        explanation="PCI-DSS and SOX require encryption of financial data in transit",
                        confidence=0.9,
                        recommendations=[
                            "Enable encryption for data transmission",
                            "Use TLS/SSL for network communication",
                            "Implement end-to-end encryption for sensitive financial data",
                        ],
                        metadata={
                            "detected_patterns": detected_patterns,
                            "detected_keywords": detected_keywords,
                            "compliance_standards": ["PCI-DSS", "SOX"],
                        },
                    )
                )

            # Check authorization
            if not self._check_authorization(context):
                violations.append(
                    SafetyViolation(
                        detector=self.name,
                        severity="critical",
                        description="Financial data access without proper authorization",
                        category="financial_compliance",
                        explanation="Financial data must only be accessed by authorized personnel",
                        confidence=0.95,
                        recommendations=[
                            "Implement role-based access control (RBAC)",
                            "Require multi-factor authentication",
                            "Log all access to financial data",
                        ],
                        metadata={
                            "detected_patterns": detected_patterns,
                            "compliance_standards": ["SOX", "PCI-DSS"],
                        },
                    )
                )

            # Check audit logging
            if not self._check_audit_logging(context):
                violations.append(
                    SafetyViolation(
                        detector=self.name,
                        severity="medium",
                        description="Financial transaction not properly logged",
                        category="financial_compliance",
                        explanation="All financial transactions must be logged for audit purposes",
                        confidence=0.85,
                        recommendations=[
                            "Enable comprehensive audit logging",
                            "Log all financial data access and modifications",
                            "Implement immutable audit trail",
                        ],
                        metadata={"compliance_standards": ["SOX"]},
                    )
                )

        return violations if violations else None

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description="Detects financial compliance violations (PCI-DSS, SOX)",
            author="Nethical Team",
            requires_nethical_version=">=1.0.0",
            dependencies=[],
            tags={"finance", "compliance", "pci-dss", "sox"},
        )

    def _get_action_content(self, action: Any) -> str:
        """Extract content from action."""
        if hasattr(action, "content"):
            return str(action.content)
        elif hasattr(action, "action_type"):
            return str(action.action_type)
        return str(action)

    def _get_action_context(self, action: Any) -> dict:
        """Extract context from action."""
        if hasattr(action, "context"):
            return action.context if isinstance(action.context, dict) else {}
        return {}

    def _check_encryption(self, context: dict) -> bool:
        """Check if encryption is enabled."""
        return context.get("encryption_enabled", False) or context.get(
            "tls_enabled", False
        )

    def _check_authorization(self, context: dict) -> bool:
        """Check if proper authorization is present."""
        return context.get("authorized", False) or context.get("user_role") in [
            "admin",
            "financial_officer",
        ]

    def _check_audit_logging(self, context: dict) -> bool:
        """Check if audit logging is enabled."""
        return context.get("audit_logging", False)


class HealthcareComplianceDetector(DetectorPlugin):
    """
    Detector for healthcare compliance violations (HIPAA).

    Detects:
    - Protected Health Information (PHI) exposure
    - HIPAA compliance violations
    - Medical record access violations
    - Patient privacy violations
    """

    def __init__(self):
        """Initialize the healthcare compliance detector."""
        super().__init__(name="HealthcareComplianceDetector", version="1.0.0")

        # PHI patterns
        self.phi_patterns = {
            "medical_record_number": r"\bMRN[:\s]*\d{6,10}\b",
            "patient_id": r"\b(PATIENT|PT)[_\s]?ID[:\s]*\d+\b",
            "diagnosis_code": r"\b[A-Z]\d{2}(\.\d{1,4})?\b",  # ICD codes
            "prescription": r"\b(Rx|prescription)[:\s]*\w+\b",
        }

        # Healthcare keywords
        self.healthcare_keywords = [
            "patient",
            "diagnosis",
            "treatment",
            "prescription",
            "medical_record",
            "phi",
            "hipaa",
            "health_information",
            "medical_history",
            "clinical",
        ]

    async def detect_violations(self, action: Any) -> Sequence[SafetyViolation] | None:
        """Detect healthcare compliance violations."""
        violations = []

        content = self._get_action_content(action)
        context = self._get_action_context(action)

        # Check for PHI patterns
        detected_phi = []
        for phi_type, pattern in self.phi_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected_phi.append(phi_type)

        # Check for healthcare keywords
        detected_keywords = [
            kw
            for kw in self.healthcare_keywords
            if kw.replace("_", " ") in content.lower()
        ]

        if detected_phi or detected_keywords:
            # Check for patient consent
            if not self._check_patient_consent(context):
                violations.append(
                    SafetyViolation(
                        detector=self.name,
                        severity="critical",
                        description="PHI accessed without patient consent",
                        category="healthcare_compliance",
                        explanation="HIPAA requires patient consent for PHI disclosure",
                        confidence=0.9,
                        recommendations=[
                            "Obtain patient consent before accessing PHI",
                            "Document consent in patient records",
                            "Implement consent management system",
                        ],
                        metadata={
                            "detected_phi": detected_phi,
                            "compliance_standards": ["HIPAA"],
                        },
                    )
                )

            # Check for minimum necessary standard
            if not self._check_minimum_necessary(context):
                violations.append(
                    SafetyViolation(
                        detector=self.name,
                        severity="high",
                        description="PHI disclosure exceeds minimum necessary",
                        category="healthcare_compliance",
                        explanation="HIPAA requires limiting PHI disclosure to minimum necessary",
                        confidence=0.8,
                        recommendations=[
                            "Review and limit PHI disclosure",
                            "Implement need-to-know access controls",
                            "Document justification for PHI access",
                        ],
                        metadata={"compliance_standards": ["HIPAA"]},
                    )
                )

            # Check for proper security measures
            if not self._check_security_measures(context):
                violations.append(
                    SafetyViolation(
                        detector=self.name,
                        severity="high",
                        description="PHI not properly secured",
                        category="healthcare_compliance",
                        explanation="HIPAA requires administrative, physical, and technical safeguards for PHI",
                        confidence=0.85,
                        recommendations=[
                            "Implement encryption for PHI",
                            "Use secure authentication",
                            "Enable audit logging for PHI access",
                        ],
                        metadata={
                            "detected_phi": detected_phi,
                            "compliance_standards": ["HIPAA"],
                        },
                    )
                )

        return violations if violations else None

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description="Detects healthcare compliance violations (HIPAA, PHI protection)",
            author="Nethical Team",
            requires_nethical_version=">=1.0.0",
            dependencies=[],
            tags={"healthcare", "compliance", "hipaa", "phi"},
        )

    def _get_action_content(self, action: Any) -> str:
        """Extract content from action."""
        if hasattr(action, "content"):
            return str(action.content)
        return str(action)

    def _get_action_context(self, action: Any) -> dict:
        """Extract context from action."""
        if hasattr(action, "context"):
            return action.context if isinstance(action.context, dict) else {}
        return {}

    def _check_patient_consent(self, context: dict) -> bool:
        """Check if patient consent is documented."""
        return context.get("patient_consent", False)

    def _check_minimum_necessary(self, context: dict) -> bool:
        """Check if minimum necessary standard is met."""
        return context.get("minimum_necessary_reviewed", False)

    def _check_security_measures(self, context: dict) -> bool:
        """Check if proper security measures are in place."""
        return context.get("encryption_enabled", False) and context.get(
            "audit_logging", False
        )


class CustomPolicyDetector(DetectorPlugin):
    """
    Generic detector for organization-specific policies.

    Can be configured with custom patterns and rules to detect
    violations specific to an organization's policies.
    """

    def __init__(
        self,
        policy_name: str = "custom_policy",
        forbidden_patterns: Optional[List[str]] = None,
        required_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the custom policy detector.

        Args:
            policy_name: Name of the custom policy
            forbidden_patterns: List of regex patterns that are forbidden
            required_patterns: List of regex patterns that are required
        """
        super().__init__(name=f"CustomPolicyDetector_{policy_name}", version="1.0.0")

        self.policy_name = policy_name
        self.forbidden_patterns = forbidden_patterns or []
        self.required_patterns = required_patterns or []

    async def detect_violations(self, action: Any) -> Sequence[SafetyViolation] | None:
        """Detect custom policy violations."""
        violations = []

        content = self._get_action_content(action)

        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(
                        SafetyViolation(
                            detector=self.name,
                            severity="high",
                            description=f"Action contains forbidden pattern: {pattern}",
                            category="custom_policy",
                            explanation=f"Organization policy '{self.policy_name}' prohibits this pattern",
                            confidence=0.95,
                            recommendations=[
                                "Remove or modify content to comply with organization policy",
                                f"Review policy '{self.policy_name}' for guidelines",
                            ],
                            metadata={
                                "policy_name": self.policy_name,
                                "violated_pattern": pattern,
                            },
                        )
                    )
            except re.error as e:
                self._audit_event("regex_error", {"pattern": pattern, "error": str(e)})

        # Check required patterns
        missing_patterns = []
        for pattern in self.required_patterns:
            try:
                if not re.search(pattern, content, re.IGNORECASE):
                    missing_patterns.append(pattern)
            except re.error as e:
                self._audit_event("regex_error", {"pattern": pattern, "error": str(e)})

        if missing_patterns:
            violations.append(
                SafetyViolation(
                    detector=self.name,
                    severity="medium",
                    description=f"Action missing required patterns: {', '.join(missing_patterns)}",
                    category="custom_policy",
                    explanation=f"Organization policy '{self.policy_name}' requires these patterns",
                    confidence=0.9,
                    recommendations=[
                        "Add required content to comply with organization policy",
                        f"Review policy '{self.policy_name}' for requirements",
                    ],
                    metadata={
                        "policy_name": self.policy_name,
                        "missing_patterns": missing_patterns,
                    },
                )
            )

        return violations if violations else None

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=f"Detects violations of custom policy: {self.policy_name}",
            author="Organization",
            requires_nethical_version=">=1.0.0",
            dependencies=[],
            tags={"custom", "policy", self.policy_name},
        )

    def _get_action_content(self, action: Any) -> str:
        """Extract content from action."""
        if hasattr(action, "content"):
            return str(action.content)
        return str(action)
