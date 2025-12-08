"""Compliance Validator for Nethical.

This module provides the main ComplianceValidator class that orchestrates
compliance validation across multiple regulatory frameworks:
- GDPR (EU General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- EU AI Act (High-risk AI requirements)
- ISO 27001 (Information Security)
- NIST AI RMF (AI Risk Management)

Adheres to the 25 Fundamental Laws:
- Law 15: Audit Compliance - Cooperation with auditing
- Law 10: Reasoning Transparency - Explainable validation
- Law 12: Limitation Disclosure - Clear gap reporting

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gdpr import GDPRComplianceValidator, GDPRValidationResult
from .eu_ai_act import EUAIActValidator, ConformityAssessmentResult, AIRiskLevel
from .data_residency import DataResidencyManager

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"
    CCPA = "ccpa"
    EU_AI_ACT = "eu_ai_act"
    ISO_27001 = "iso_27001"
    NIST_AI_RMF = "nist_ai_rmf"
    ALL = "all"


class ComplianceStatus(str, Enum):
    """Overall compliance status."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    framework: ComplianceFramework
    check_id: str
    check_name: str
    status: ComplianceStatus
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    code_modules: List[str] = field(default_factory=list)
    test_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "framework": self.framework.value,
            "check_id": self.check_id,
            "check_name": self.check_name,
            "status": self.status.value,
            "evidence": self.evidence,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "code_modules": self.code_modules,
            "test_evidence": self.test_evidence,
        }


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""

    report_id: str
    generated_at: datetime
    frameworks_validated: List[ComplianceFramework]
    overall_status: ComplianceStatus
    compliance_score: float
    validation_results: List[ValidationResult]
    gdpr_summary: Optional[Dict[str, Any]] = None
    eu_ai_act_summary: Optional[Dict[str, Any]] = None
    data_residency_summary: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "frameworks_validated": [f.value for f in self.frameworks_validated],
            "overall_status": self.overall_status.value,
            "compliance_score": self.compliance_score,
            "validation_results": [r.to_dict() for r in self.validation_results],
            "gdpr_summary": self.gdpr_summary,
            "eu_ai_act_summary": self.eu_ai_act_summary,
            "data_residency_summary": self.data_residency_summary,
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save report to file.

        Args:
            path: File path to save report
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json())
        logger.info("Compliance report saved to: %s", path)


class ComplianceValidator:
    """Orchestrates compliance validation across multiple frameworks.

    Provides automated compliance checking against regulatory frameworks
    including GDPR, EU AI Act, CCPA, ISO 27001, and NIST AI RMF.

    Attributes:
        gdpr_validator: GDPR compliance validator
        eu_ai_act_validator: EU AI Act compliance validator
        data_residency_manager: Data residency manager
    """

    # Minimum score for compliance
    COMPLIANCE_THRESHOLD = 80.0

    def __init__(
        self,
        system_characteristics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize Compliance Validator.

        Args:
            system_characteristics: AI system characteristics for risk classification
        """
        self.system_characteristics = system_characteristics or {}

        # Initialize sub-validators
        self.gdpr_validator = GDPRComplianceValidator()
        self.eu_ai_act_validator = EUAIActValidator(system_characteristics)
        self.data_residency_manager = DataResidencyManager()

        self.validation_results: List[ValidationResult] = []

        logger.info("ComplianceValidator initialized")

    def validate(
        self,
        framework: ComplianceFramework = ComplianceFramework.ALL,
        configs: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReport:
        """Run compliance validation for specified framework(s).

        Args:
            framework: Framework to validate (or ALL)
            configs: Configuration for validation checks

        Returns:
            ComplianceReport with validation results
        """
        configs = configs or {}
        self.validation_results = []
        frameworks_validated: List[ComplianceFramework] = []
        gdpr_summary = None
        eu_ai_act_summary = None
        data_residency_summary = None

        # Validate GDPR
        if framework in (ComplianceFramework.GDPR, ComplianceFramework.ALL):
            gdpr_results = self._validate_gdpr(configs.get("gdpr", {}))
            self.validation_results.extend(gdpr_results)
            frameworks_validated.append(ComplianceFramework.GDPR)
            gdpr_summary = self.gdpr_validator.get_compliance_summary()

        # Validate EU AI Act
        if framework in (ComplianceFramework.EU_AI_ACT, ComplianceFramework.ALL):
            eu_ai_results = self._validate_eu_ai_act(configs.get("eu_ai_act", {}))
            self.validation_results.extend(eu_ai_results)
            frameworks_validated.append(ComplianceFramework.EU_AI_ACT)
            eu_ai_act_summary = self.eu_ai_act_validator.get_compliance_summary()

        # Validate CCPA
        if framework in (ComplianceFramework.CCPA, ComplianceFramework.ALL):
            ccpa_results = self._validate_ccpa(configs.get("ccpa", {}))
            self.validation_results.extend(ccpa_results)
            frameworks_validated.append(ComplianceFramework.CCPA)

        # Validate ISO 27001
        if framework in (ComplianceFramework.ISO_27001, ComplianceFramework.ALL):
            iso_results = self._validate_iso_27001(configs.get("iso_27001", {}))
            self.validation_results.extend(iso_results)
            frameworks_validated.append(ComplianceFramework.ISO_27001)

        # Validate NIST AI RMF
        if framework in (ComplianceFramework.NIST_AI_RMF, ComplianceFramework.ALL):
            nist_results = self._validate_nist_ai_rmf(configs.get("nist_ai_rmf", {}))
            self.validation_results.extend(nist_results)
            frameworks_validated.append(ComplianceFramework.NIST_AI_RMF)

        # Get data residency summary
        data_residency_summary = self.data_residency_manager.get_violations_summary()

        # Calculate overall compliance
        overall_status, compliance_score = self._calculate_overall_compliance()

        # Collect recommendations
        recommendations = self._collect_recommendations()

        return ComplianceReport(
            report_id=f"COMP-{uuid.uuid4().hex[:8].upper()}",
            generated_at=datetime.now(timezone.utc),
            frameworks_validated=frameworks_validated,
            overall_status=overall_status,
            compliance_score=compliance_score,
            validation_results=self.validation_results,
            gdpr_summary=gdpr_summary,
            eu_ai_act_summary=eu_ai_act_summary,
            data_residency_summary=data_residency_summary,
            recommendations=recommendations,
        )

    def _validate_gdpr(
        self,
        config: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Validate GDPR compliance.

        Args:
            config: GDPR validation configuration

        Returns:
            List of ValidationResults
        """
        results: List[ValidationResult] = []

        # Validate Article 5 (Principles)
        art5_result = self.gdpr_validator.validate_article_5(
            config.get(
                "processing_activities",
                {
                    "lawful_basis": True,
                    "purpose_documented": True,
                    "data_minimization": True,
                    "accuracy_controls": True,
                    "retention_policy": True,
                    "security_controls": True,
                },
            )
        )
        results.append(
            self._convert_gdpr_result(
                art5_result, "GDPR-5", "Data Processing Principles"
            )
        )

        # Validate Article 22 (Automated Decision-Making)
        art22_result = self.gdpr_validator.validate_article_22(
            config.get(
                "automated_decision",
                {
                    "human_intervention_available": True,
                    "explanation_capability": True,
                    "logic_documented": True,
                    "significance_explained": True,
                    "appeal_mechanism": True,
                    "safeguards_implemented": True,
                },
            )
        )
        results.append(
            self._convert_gdpr_result(
                art22_result, "GDPR-22", "Automated Decision-Making"
            )
        )

        # Validate Article 25 (Privacy by Design)
        art25_result = self.gdpr_validator.validate_article_25(
            config.get(
                "design_controls",
                {
                    "pseudonymization": True,
                    "data_minimization_by_design": True,
                    "privacy_by_default": True,
                    "access_controls": True,
                    "encryption": True,
                },
            )
        )
        results.append(
            self._convert_gdpr_result(
                art25_result, "GDPR-25", "Data Protection by Design"
            )
        )

        return results

    def _validate_eu_ai_act(
        self,
        config: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Validate EU AI Act compliance.

        Args:
            config: EU AI Act validation configuration

        Returns:
            List of ValidationResults
        """
        results: List[ValidationResult] = []

        # Classify risk level first
        risk_level = self.eu_ai_act_validator.classify_risk_level()

        # Add risk classification result
        results.append(
            ValidationResult(
                framework=ComplianceFramework.EU_AI_ACT,
                check_id="EU-AI-6",
                check_name="Risk Classification",
                status=(
                    ComplianceStatus.COMPLIANT
                    if risk_level != AIRiskLevel.UNACCEPTABLE
                    else ComplianceStatus.NON_COMPLIANT
                ),
                evidence=[f"System classified as {risk_level.value} risk"],
            )
        )

        # For high-risk systems, validate articles
        if risk_level == AIRiskLevel.HIGH:
            # Article 9: Risk Management
            art9 = self.eu_ai_act_validator.validate_article_9(
                config.get(
                    "risk_management",
                    {
                        "risk_process_established": True,
                        "risks_identified": True,
                        "mitigation_measures": True,
                        "continuous_monitoring": True,
                        "lifecycle_coverage": True,
                    },
                )
            )
            results.append(self._convert_eu_ai_result(art9))

            # Article 10: Data Governance
            art10 = self.eu_ai_act_validator.validate_article_10(
                config.get(
                    "data_governance",
                    {
                        "governance_practices": True,
                        "data_quality_controls": True,
                        "bias_detection": True,
                        "data_minimization": True,
                        "representative_data": True,
                    },
                )
            )
            results.append(self._convert_eu_ai_result(art10))

            # Article 11: Documentation
            art11 = self.eu_ai_act_validator.validate_article_11(
                config.get(
                    "documentation",
                    {
                        "system_description": True,
                        "architecture_documented": True,
                        "api_documented": True,
                        "risk_docs": True,
                        "accuracy_metrics": True,
                    },
                )
            )
            results.append(self._convert_eu_ai_result(art11))

            # Article 12: Logging
            art12 = self.eu_ai_act_validator.validate_article_12(
                config.get(
                    "logging",
                    {
                        "automatic_logging": True,
                        "traceability": True,
                        "tamper_evident": True,
                        "retention_policy": True,
                    },
                )
            )
            results.append(self._convert_eu_ai_result(art12))

            # Article 13: Transparency
            art13 = self.eu_ai_act_validator.validate_article_13(
                config.get(
                    "transparency",
                    {
                        "transparency_reports": True,
                        "explanation_capability": True,
                        "capability_disclosure": True,
                        "user_instructions": True,
                    },
                )
            )
            results.append(self._convert_eu_ai_result(art13))

            # Article 14: Human Oversight
            art14 = self.eu_ai_act_validator.validate_article_14(
                config.get(
                    "oversight",
                    {
                        "oversight_designed": True,
                        "feedback_loop": True,
                        "override_capability": True,
                        "understanding_tools": True,
                    },
                )
            )
            results.append(self._convert_eu_ai_result(art14))

            # Article 15: Security
            art15 = self.eu_ai_act_validator.validate_article_15(
                config.get(
                    "security",
                    {
                        "accuracy_metrics": True,
                        "robustness_testing": True,
                        "adversarial_robustness": True,
                        "cybersecurity": True,
                        "fail_safe": True,
                    },
                )
            )
            results.append(self._convert_eu_ai_result(art15))

        return results

    def _validate_ccpa(
        self,
        config: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Validate CCPA compliance.

        Args:
            config: CCPA validation configuration

        Returns:
            List of ValidationResults
        """
        results: List[ValidationResult] = []

        # CCPA Consumer Rights
        consumer_rights = config.get(
            "consumer_rights",
            {
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_opt_out": True,
                "right_to_non_discrimination": True,
            },
        )

        evidence = []
        gaps = []

        if consumer_rights.get("right_to_know"):
            evidence.append("Right to know implemented")
        else:
            gaps.append("Right to know not implemented")

        if consumer_rights.get("right_to_delete"):
            evidence.append("Right to delete implemented")
        else:
            gaps.append("Right to delete not implemented")

        if consumer_rights.get("right_to_opt_out"):
            evidence.append("Right to opt-out implemented")
        else:
            gaps.append("Right to opt-out not implemented")

        if consumer_rights.get("right_to_non_discrimination"):
            evidence.append("Right to non-discrimination implemented")
        else:
            gaps.append("Non-discrimination not verified")

        status = (
            ComplianceStatus.COMPLIANT
            if len(gaps) == 0
            else (
                ComplianceStatus.PARTIAL
                if len(gaps) <= 1
                else ComplianceStatus.NON_COMPLIANT
            )
        )

        results.append(
            ValidationResult(
                framework=ComplianceFramework.CCPA,
                check_id="CCPA-1798.100-125",
                check_name="Consumer Rights",
                status=status,
                evidence=evidence,
                gaps=gaps,
                recommendations=[f"Address: {gap}" for gap in gaps],
            )
        )

        return results

    def _validate_iso_27001(
        self,
        config: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Validate ISO 27001 compliance.

        Args:
            config: ISO 27001 validation configuration

        Returns:
            List of ValidationResults
        """
        results: List[ValidationResult] = []

        # Annex A Controls
        controls = config.get(
            "controls",
            {
                "access_control": True,
                "cryptography": True,
                "operations_security": True,
                "communications_security": True,
                "system_acquisition": True,
                "supplier_relationships": True,
                "incident_management": True,
                "business_continuity": True,
                "compliance": True,
            },
        )

        evidence = []
        gaps = []
        code_modules = []

        if controls.get("access_control"):
            evidence.append("Access control implemented (A.9)")
            code_modules.append("nethical/core/rbac.py")
        else:
            gaps.append("Access control gaps (A.9)")

        if controls.get("cryptography"):
            evidence.append("Cryptography controls (A.10)")
            code_modules.append("nethical/security/encryption.py")
        else:
            gaps.append("Cryptography gaps (A.10)")

        if controls.get("operations_security"):
            evidence.append("Operations security (A.12)")
        else:
            gaps.append("Operations security gaps (A.12)")

        if controls.get("incident_management"):
            evidence.append("Incident management (A.16)")
            code_modules.append("nethical/security/soc_integration.py")
        else:
            gaps.append("Incident management gaps (A.16)")

        status = (
            ComplianceStatus.COMPLIANT
            if len(gaps) == 0
            else (
                ComplianceStatus.PARTIAL
                if len(gaps) <= 2
                else ComplianceStatus.NON_COMPLIANT
            )
        )

        results.append(
            ValidationResult(
                framework=ComplianceFramework.ISO_27001,
                check_id="ISO27001-ANNEX-A",
                check_name="Annex A Controls",
                status=status,
                evidence=evidence,
                gaps=gaps,
                recommendations=[f"Implement: {gap}" for gap in gaps],
                code_modules=code_modules,
            )
        )

        return results

    def _validate_nist_ai_rmf(
        self,
        config: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Validate NIST AI RMF compliance.

        Args:
            config: NIST AI RMF validation configuration

        Returns:
            List of ValidationResults
        """
        results: List[ValidationResult] = []

        # GOVERN function
        govern = config.get(
            "govern",
            {
                "policies_established": True,
                "accountability_defined": True,
                "workforce_trained": True,
            },
        )

        govern_evidence = []
        govern_gaps = []

        if govern.get("policies_established"):
            govern_evidence.append("AI governance policies established")
        else:
            govern_gaps.append("Policies not established")

        if govern.get("accountability_defined"):
            govern_evidence.append("Accountability structures defined")
        else:
            govern_gaps.append("Accountability not defined")

        govern_status = (
            ComplianceStatus.COMPLIANT
            if len(govern_gaps) == 0
            else (
                ComplianceStatus.PARTIAL
                if len(govern_gaps) == 1
                else ComplianceStatus.NON_COMPLIANT
            )
        )

        results.append(
            ValidationResult(
                framework=ComplianceFramework.NIST_AI_RMF,
                check_id="NIST-AI-GOVERN",
                check_name="GOVERN Function",
                status=govern_status,
                evidence=govern_evidence,
                gaps=govern_gaps,
                recommendations=[f"Address: {gap}" for gap in govern_gaps],
                code_modules=["nethical/core/governance.py"],
            )
        )

        # MEASURE function
        measure = config.get(
            "measure",
            {
                "metrics_defined": True,
                "testing_implemented": True,
                "bias_measured": True,
            },
        )

        measure_evidence = []
        measure_gaps = []

        if measure.get("metrics_defined"):
            measure_evidence.append("AI metrics defined")
        else:
            measure_gaps.append("Metrics not defined")

        if measure.get("testing_implemented"):
            measure_evidence.append("Testing implemented")
        else:
            measure_gaps.append("Testing incomplete")

        if measure.get("bias_measured"):
            measure_evidence.append("Bias measurement implemented")
        else:
            measure_gaps.append("Bias measurement incomplete")

        measure_status = (
            ComplianceStatus.COMPLIANT
            if len(measure_gaps) == 0
            else (
                ComplianceStatus.PARTIAL
                if len(measure_gaps) == 1
                else ComplianceStatus.NON_COMPLIANT
            )
        )

        results.append(
            ValidationResult(
                framework=ComplianceFramework.NIST_AI_RMF,
                check_id="NIST-AI-MEASURE",
                check_name="MEASURE Function",
                status=measure_status,
                evidence=measure_evidence,
                gaps=measure_gaps,
                recommendations=[f"Implement: {gap}" for gap in measure_gaps],
                code_modules=["nethical/governance/ethics_benchmark.py"],
            )
        )

        return results

    def _convert_gdpr_result(
        self,
        gdpr_result: GDPRValidationResult,
        check_id: str,
        check_name: str,
    ) -> ValidationResult:
        """Convert GDPR result to ValidationResult.

        Args:
            gdpr_result: GDPRValidationResult to convert
            check_id: Check identifier
            check_name: Check name

        Returns:
            ValidationResult
        """
        status_map = {
            "compliant": ComplianceStatus.COMPLIANT,
            "partial": ComplianceStatus.PARTIAL,
            "non_compliant": ComplianceStatus.NON_COMPLIANT,
        }

        return ValidationResult(
            framework=ComplianceFramework.GDPR,
            check_id=check_id,
            check_name=check_name,
            status=status_map.get(gdpr_result.status.value, ComplianceStatus.PENDING),
            evidence=gdpr_result.evidence,
            gaps=gdpr_result.gaps,
            recommendations=gdpr_result.recommendations,
        )

    def _convert_eu_ai_result(
        self,
        article_result: Any,
    ) -> ValidationResult:
        """Convert EU AI Act result to ValidationResult.

        Args:
            article_result: ArticleValidationResult to convert

        Returns:
            ValidationResult
        """
        status_map = {
            "compliant": ComplianceStatus.COMPLIANT,
            "partial": ComplianceStatus.PARTIAL,
            "non_compliant": ComplianceStatus.NON_COMPLIANT,
        }

        return ValidationResult(
            framework=ComplianceFramework.EU_AI_ACT,
            check_id=f"EU-AI-{article_result.article.value.split('_')[1]}",
            check_name=article_result.requirement,
            status=status_map.get(
                article_result.status.value, ComplianceStatus.PENDING
            ),
            evidence=article_result.evidence,
            gaps=article_result.gaps,
            recommendations=article_result.recommendations,
            code_modules=article_result.code_modules,
            test_evidence=article_result.test_evidence,
        )

    def _calculate_overall_compliance(self) -> tuple[ComplianceStatus, float]:
        """Calculate overall compliance status and score.

        Returns:
            Tuple of (ComplianceStatus, score)
        """
        if not self.validation_results:
            return ComplianceStatus.PENDING, 0.0

        total = len(self.validation_results)
        compliant = sum(
            1 for r in self.validation_results if r.status == ComplianceStatus.COMPLIANT
        )
        partial = sum(
            1 for r in self.validation_results if r.status == ComplianceStatus.PARTIAL
        )

        score = (compliant + (partial * 0.5)) / total * 100

        # Check for any non-compliant
        has_non_compliant = any(
            r.status == ComplianceStatus.NON_COMPLIANT for r in self.validation_results
        )

        if has_non_compliant:
            status = ComplianceStatus.NON_COMPLIANT
        elif score >= self.COMPLIANCE_THRESHOLD:
            status = ComplianceStatus.COMPLIANT
        else:
            status = ComplianceStatus.PARTIAL

        return status, round(score, 2)

    def _collect_recommendations(self) -> List[str]:
        """Collect all recommendations from validation results.

        Returns:
            List of recommendations
        """
        recommendations: List[str] = []

        for result in self.validation_results:
            for rec in result.recommendations:
                if rec not in recommendations:
                    recommendations.append(rec)

        return recommendations


__all__ = [
    "ComplianceValidator",
    "ComplianceFramework",
    "ComplianceStatus",
    "ValidationResult",
    "ComplianceReport",
]
