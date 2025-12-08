"""EU AI Act Compliance Validator for Nethical.

This module provides comprehensive EU AI Act (Regulation (EU) 2024/1689)
compliance validation capabilities including:
- Risk Classification (Article 6)
- High-Risk AI System Requirements (Articles 9-15)
- Conformity Assessment
- Technical Documentation Requirements

Adheres to the 25 Fundamental Laws:
- Law 6: Decision Authority - Clear decision-making authority
- Law 10: Reasoning Transparency - Explainable decision-making
- Law 15: Audit Compliance - Cooperation with auditing
- Law 21: Human Safety Priority - Primacy of human safety

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

logger = logging.getLogger(__name__)


class AIRiskLevel(str, Enum):
    """EU AI Act risk classification levels (Article 6)."""

    UNACCEPTABLE = "unacceptable"  # Article 5 - Prohibited practices
    HIGH = "high"  # Annex III - Requires conformity assessment
    LIMITED = "limited"  # Transparency obligations only
    MINIMAL = "minimal"  # No specific obligations


class EUAIActArticle(str, Enum):
    """EU AI Act Articles relevant to high-risk AI systems."""

    ARTICLE_5 = "article_5"  # Prohibited AI practices
    ARTICLE_6 = "article_6"  # Classification rules
    ARTICLE_9 = "article_9"  # Risk management system
    ARTICLE_10 = "article_10"  # Data and data governance
    ARTICLE_11 = "article_11"  # Technical documentation
    ARTICLE_12 = "article_12"  # Record-keeping
    ARTICLE_13 = "article_13"  # Transparency
    ARTICLE_14 = "article_14"  # Human oversight
    ARTICLE_15 = "article_15"  # Accuracy, robustness, cybersecurity


class ComplianceStatus(str, Enum):
    """Compliance validation status."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


class HighRiskDomain(str, Enum):
    """High-risk AI domains under Annex III."""

    BIOMETRIC_IDENTIFICATION = "biometric_identification"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION_VOCATIONAL = "education_vocational"
    EMPLOYMENT_RECRUITMENT = "employment_recruitment"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION_ASYLUM = "migration_asylum"
    JUSTICE_DEMOCRATIC = "justice_democratic"


@dataclass
class ArticleValidationResult:
    """Result of validating a specific EU AI Act Article."""

    article: EUAIActArticle
    status: ComplianceStatus
    requirement: str
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    code_modules: List[str] = field(default_factory=list)
    test_evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "article": self.article.value,
            "status": self.status.value,
            "requirement": self.requirement,
            "evidence": self.evidence,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "code_modules": self.code_modules,
            "test_evidence": self.test_evidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConformityAssessmentResult:
    """Result of EU AI Act conformity assessment."""

    assessment_id: str
    risk_level: AIRiskLevel
    assessment_date: datetime
    article_results: List[ArticleValidationResult]
    overall_status: ComplianceStatus
    compliance_score: float
    certification_ready: bool
    recommendations: List[str]
    technical_documentation_complete: bool
    qms_implemented: bool
    post_market_monitoring: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "assessment_id": self.assessment_id,
            "risk_level": self.risk_level.value,
            "assessment_date": self.assessment_date.isoformat(),
            "article_results": [r.to_dict() for r in self.article_results],
            "overall_status": self.overall_status.value,
            "compliance_score": self.compliance_score,
            "certification_ready": self.certification_ready,
            "recommendations": self.recommendations,
            "technical_documentation_complete": self.technical_documentation_complete,
            "qms_implemented": self.qms_implemented,
            "post_market_monitoring": self.post_market_monitoring,
        }


class EUAIActValidator:
    """EU AI Act Compliance Validator for Nethical governance system.

    Provides comprehensive validation of EU AI Act requirements including:
    - Risk classification (Article 6)
    - High-risk system requirements (Articles 9-15)
    - Conformity assessment
    - Technical documentation validation

    Attributes:
        system_characteristics: Characteristics of the AI system being validated
    """

    # Minimum compliance score for certification readiness
    CERTIFICATION_THRESHOLD = 90.0

    def __init__(
        self,
        system_characteristics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize EU AI Act Validator.

        Args:
            system_characteristics: Dictionary describing the AI system
        """
        self.system_characteristics = system_characteristics or {}
        self.validation_results: List[ArticleValidationResult] = []
        self.risk_level: Optional[AIRiskLevel] = None

        logger.info("EUAIActValidator initialized")

    def classify_risk_level(
        self,
        characteristics: Optional[Dict[str, Any]] = None,
    ) -> AIRiskLevel:
        """Classify AI system risk level according to EU AI Act.

        Implements Article 6 classification rules:
        - Unacceptable: Article 5 prohibited practices
        - High: Annex III high-risk domains
        - Limited: Transparency obligations (chatbots, emotion recognition)
        - Minimal: No specific requirements

        Args:
            characteristics: System characteristics (uses stored if not provided)

        Returns:
            AIRiskLevel classification
        """
        chars = characteristics or self.system_characteristics

        # Article 5: Unacceptable risk (Prohibited practices)
        prohibited_practices = [
            "social_scoring",
            "subliminal_manipulation",
            "exploitation_vulnerable",
            "real_time_biometric_public",
        ]

        for practice in prohibited_practices:
            if chars.get(practice, False):
                self.risk_level = AIRiskLevel.UNACCEPTABLE
                logger.warning(
                    "System classified as UNACCEPTABLE risk: %s detected",
                    practice,
                )
                return AIRiskLevel.UNACCEPTABLE

        # Annex III: High-risk domains
        high_risk_domains = [
            "biometric_identification",
            "critical_infrastructure",
            "education_vocational",
            "employment_recruitment",
            "essential_services",
            "law_enforcement",
            "migration_asylum",
            "justice_democratic",
            "safety_component",
        ]

        for domain in high_risk_domains:
            if chars.get(domain, False):
                self.risk_level = AIRiskLevel.HIGH
                logger.info("System classified as HIGH risk: %s domain", domain)
                return AIRiskLevel.HIGH

        # Limited risk: Transparency obligations
        limited_risk_types = [
            "chatbot",
            "emotion_recognition",
            "biometric_categorization",
            "deep_fake",
            "content_generation",
        ]

        for risk_type in limited_risk_types:
            if chars.get(risk_type, False):
                self.risk_level = AIRiskLevel.LIMITED
                logger.info("System classified as LIMITED risk: %s type", risk_type)
                return AIRiskLevel.LIMITED

        # Default: Minimal risk
        self.risk_level = AIRiskLevel.MINIMAL
        logger.info("System classified as MINIMAL risk")
        return AIRiskLevel.MINIMAL

    def validate_article_9(
        self,
        risk_management_config: Dict[str, Any],
    ) -> ArticleValidationResult:
        """Validate Article 9: Risk Management System.

        Validates that the system has:
        - Established risk management process
        - Identified and analyzed risks
        - Estimated and evaluated risks
        - Adopted risk mitigation measures
        - Continuous monitoring

        Args:
            risk_management_config: Risk management system configuration

        Returns:
            ArticleValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        code_modules = []
        test_evidence = []

        # Check risk management process
        if risk_management_config.get("risk_process_established"):
            evidence.append("Risk management process established and documented")
            code_modules.append("nethical/core/risk_engine.py")
        else:
            gaps.append("Risk management process not established")
            recommendations.append("Implement systematic risk management process")

        # Check risk identification
        if risk_management_config.get("risks_identified"):
            evidence.append("Risks identified and analyzed")
            code_modules.append("nethical/core/anomaly_detector.py")
            test_evidence.append("tests/test_anomaly_classifier.py")
        else:
            gaps.append("Risk identification not systematic")
            recommendations.append("Implement systematic risk identification")

        # Check risk mitigation
        if risk_management_config.get("mitigation_measures"):
            evidence.append("Risk mitigation measures implemented")
            code_modules.append("nethical/core/quarantine.py")
        else:
            gaps.append("Risk mitigation measures not documented")
            recommendations.append("Document and implement risk mitigation measures")

        # Check continuous monitoring
        if risk_management_config.get("continuous_monitoring"):
            evidence.append("Continuous risk monitoring in place")
            code_modules.append("nethical/core/governance.py")
            test_evidence.append("tests/test_governance_features.py")
        else:
            gaps.append("Continuous monitoring not implemented")
            recommendations.append("Implement continuous risk monitoring")

        # Check lifecycle coverage
        if risk_management_config.get("lifecycle_coverage"):
            evidence.append("Risk management covers entire lifecycle")
        else:
            gaps.append("Lifecycle coverage not verified")
            recommendations.append("Ensure risk management covers entire AI lifecycle")

        # Determine status
        status = self._calculate_status(len(gaps))

        result = ArticleValidationResult(
            article=EUAIActArticle.ARTICLE_9,
            status=status,
            requirement="Risk management system throughout AI lifecycle",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            code_modules=code_modules,
            test_evidence=test_evidence,
        )

        self.validation_results.append(result)
        return result

    def validate_article_10(
        self,
        data_governance_config: Dict[str, Any],
    ) -> ArticleValidationResult:
        """Validate Article 10: Data and Data Governance.

        Validates:
        - Training data governance
        - Data quality requirements
        - Bias detection and mitigation
        - Representative datasets

        Args:
            data_governance_config: Data governance configuration

        Returns:
            ArticleValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        code_modules = []
        test_evidence = []

        # Check data governance practices
        if data_governance_config.get("governance_practices"):
            evidence.append("Data governance practices implemented")
            code_modules.append("nethical/security/data_compliance.py")
        else:
            gaps.append("Data governance practices not documented")
            recommendations.append("Implement data governance practices")

        # Check data quality
        if data_governance_config.get("data_quality_controls"):
            evidence.append("Data quality controls in place")
        else:
            gaps.append("Data quality controls not verified")
            recommendations.append("Implement data quality verification")

        # Check bias detection
        if data_governance_config.get("bias_detection"):
            evidence.append("Bias detection implemented")
            code_modules.append("nethical/core/fairness_sampler.py")
            test_evidence.append("tests/test_regionalization.py")
        else:
            gaps.append("Bias detection not implemented")
            recommendations.append("Implement bias detection and mitigation")

        # Check data minimization
        if data_governance_config.get("data_minimization"):
            evidence.append("Data minimization implemented")
            code_modules.append("nethical/core/data_minimization.py")
            test_evidence.append("tests/test_privacy_features.py")
        else:
            gaps.append("Data minimization not verified")
            recommendations.append("Implement data minimization")

        # Check representative datasets
        if data_governance_config.get("representative_data"):
            evidence.append("Datasets are representative")
        else:
            gaps.append("Dataset representativeness not verified")
            recommendations.append("Verify datasets are representative")

        status = self._calculate_status(len(gaps))

        result = ArticleValidationResult(
            article=EUAIActArticle.ARTICLE_10,
            status=status,
            requirement="Data and data governance requirements",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            code_modules=code_modules,
            test_evidence=test_evidence,
        )

        self.validation_results.append(result)
        return result

    def validate_article_11(
        self,
        documentation_config: Dict[str, Any],
    ) -> ArticleValidationResult:
        """Validate Article 11: Technical Documentation.

        Validates:
        - System description
        - Development process documentation
        - Risk management documentation
        - Accuracy and robustness metrics

        Args:
            documentation_config: Documentation configuration

        Returns:
            ArticleValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []

        # Check system description
        if documentation_config.get("system_description"):
            evidence.append("System description documented")
        else:
            gaps.append("System description incomplete")
            recommendations.append("Complete system description documentation")

        # Check architecture documentation
        if documentation_config.get("architecture_documented"):
            evidence.append("Architecture documented (ARCHITECTURE.md)")
        else:
            gaps.append("Architecture not documented")
            recommendations.append("Document system architecture")

        # Check API documentation
        if documentation_config.get("api_documented"):
            evidence.append("API documented (docs/API_USAGE.md)")
        else:
            gaps.append("API not fully documented")
            recommendations.append("Complete API documentation")

        # Check risk management docs
        if documentation_config.get("risk_docs"):
            evidence.append("Risk management documented")
        else:
            gaps.append("Risk management documentation incomplete")
            recommendations.append("Document risk management processes")

        # Check accuracy metrics
        if documentation_config.get("accuracy_metrics"):
            evidence.append("Accuracy metrics documented")
        else:
            gaps.append("Accuracy metrics not documented")
            recommendations.append("Document accuracy metrics and testing")

        status = self._calculate_status(len(gaps))

        result = ArticleValidationResult(
            article=EUAIActArticle.ARTICLE_11,
            status=status,
            requirement="Technical documentation before market placement",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            code_modules=[],
            test_evidence=[],
        )

        self.validation_results.append(result)
        return result

    def validate_article_12(
        self,
        logging_config: Dict[str, Any],
    ) -> ArticleValidationResult:
        """Validate Article 12: Record-keeping (Logging).

        Validates:
        - Automatic event logging
        - Log traceability
        - Tamper-evident logs
        - Retention policies

        Args:
            logging_config: Logging configuration

        Returns:
            ArticleValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        code_modules = []
        test_evidence = []

        # Check automatic logging
        if logging_config.get("automatic_logging"):
            evidence.append("Automatic event logging implemented")
            code_modules.append("nethical/security/audit_logging.py")
            test_evidence.append("tests/test_train_audit_logging.py")
        else:
            gaps.append("Automatic logging not verified")
            recommendations.append("Implement automatic event logging")

        # Check traceability
        if logging_config.get("traceability"):
            evidence.append("Decision traceability implemented")
        else:
            gaps.append("Traceability not verified")
            recommendations.append("Implement decision traceability")

        # Check tamper evidence
        if logging_config.get("tamper_evident"):
            evidence.append("Tamper-evident logs (Merkle anchoring)")
            code_modules.append("nethical/core/audit_merkle.py")
        else:
            gaps.append("Tamper evidence not implemented")
            recommendations.append("Implement tamper-evident logging (Merkle)")

        # Check retention
        if logging_config.get("retention_policy"):
            evidence.append("Log retention policy defined")
        else:
            gaps.append("Log retention policy not defined")
            recommendations.append("Define log retention policy")

        status = self._calculate_status(len(gaps))

        result = ArticleValidationResult(
            article=EUAIActArticle.ARTICLE_12,
            status=status,
            requirement="Automatic recording of events (logging)",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            code_modules=code_modules,
            test_evidence=test_evidence,
        )

        self.validation_results.append(result)
        return result

    def validate_article_13(
        self,
        transparency_config: Dict[str, Any],
    ) -> ArticleValidationResult:
        """Validate Article 13: Transparency and Information.

        Validates:
        - Operational transparency
        - User information provision
        - Capability disclosure
        - Limitation disclosure

        Args:
            transparency_config: Transparency configuration

        Returns:
            ArticleValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        code_modules = []
        test_evidence = []

        # Check transparency reports
        if transparency_config.get("transparency_reports"):
            evidence.append("Transparency reports generated")
            code_modules.append("nethical/explainability/transparency_report.py")
        else:
            gaps.append("Transparency reports not implemented")
            recommendations.append("Implement transparency reporting")

        # Check explanation capability
        if transparency_config.get("explanation_capability"):
            evidence.append("Decision explanation capability")
            code_modules.append("nethical/explainability/decision_explainer.py")
            test_evidence.append("tests/test_explainability/")
        else:
            gaps.append("Explanation capability not verified")
            recommendations.append("Implement decision explanation")

        # Check capability disclosure
        if transparency_config.get("capability_disclosure"):
            evidence.append("Capabilities and limitations disclosed")
        else:
            gaps.append("Capability disclosure incomplete")
            recommendations.append("Document capabilities and limitations")

        # Check user instructions
        if transparency_config.get("user_instructions"):
            evidence.append("Instructions for use provided (README.md)")
        else:
            gaps.append("User instructions incomplete")
            recommendations.append("Complete instructions for use")

        status = self._calculate_status(len(gaps))

        result = ArticleValidationResult(
            article=EUAIActArticle.ARTICLE_13,
            status=status,
            requirement="Transparency and provision of information",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            code_modules=code_modules,
            test_evidence=test_evidence,
        )

        self.validation_results.append(result)
        return result

    def validate_article_14(
        self,
        oversight_config: Dict[str, Any],
    ) -> ArticleValidationResult:
        """Validate Article 14: Human Oversight.

        Validates:
        - Human oversight design
        - Override capability
        - Human review mechanisms
        - Capability understanding tools

        Args:
            oversight_config: Human oversight configuration

        Returns:
            ArticleValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        code_modules = []
        test_evidence = []

        # Check human oversight design
        if oversight_config.get("oversight_designed"):
            evidence.append("Human oversight designed into system")
            code_modules.append("nethical/governance/human_review.py")
        else:
            gaps.append("Human oversight not designed into system")
            recommendations.append("Design human oversight mechanisms")

        # Check human feedback loop
        if oversight_config.get("feedback_loop"):
            evidence.append("Human feedback loop implemented")
            code_modules.append("nethical/core/human_feedback.py")
            test_evidence.append("tests/test_governance_features.py")
        else:
            gaps.append("Human feedback loop not implemented")
            recommendations.append("Implement human feedback mechanism")

        # Check override capability
        if oversight_config.get("override_capability"):
            evidence.append("Human override capability available")
        else:
            gaps.append("Override capability not verified")
            recommendations.append("Implement human override capability")

        # Check understanding tools
        if oversight_config.get("understanding_tools"):
            evidence.append("Tools to understand AI capabilities")
            code_modules.append("nethical/explainability/advanced_explainer.py")
            test_evidence.append("tests/test_advanced_explainability.py")
        else:
            gaps.append("Understanding tools incomplete")
            recommendations.append("Implement capability understanding tools")

        status = self._calculate_status(len(gaps))

        result = ArticleValidationResult(
            article=EUAIActArticle.ARTICLE_14,
            status=status,
            requirement="Human oversight design and implementation",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            code_modules=code_modules,
            test_evidence=test_evidence,
        )

        self.validation_results.append(result)
        return result

    def validate_article_15(
        self,
        security_config: Dict[str, Any],
    ) -> ArticleValidationResult:
        """Validate Article 15: Accuracy, Robustness and Cybersecurity.

        Validates:
        - Accuracy metrics and testing
        - Robustness against errors
        - Adversarial robustness
        - Cybersecurity measures

        Args:
            security_config: Security configuration

        Returns:
            ArticleValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        code_modules = []
        test_evidence = []

        # Check accuracy metrics
        if security_config.get("accuracy_metrics"):
            evidence.append("Accuracy metrics defined and measured")
            code_modules.append("nethical/governance/ethics_benchmark.py")
            test_evidence.append("tests/test_performance_benchmarks.py")
        else:
            gaps.append("Accuracy metrics not defined")
            recommendations.append("Define and measure accuracy metrics")

        # Check robustness
        if security_config.get("robustness_testing"):
            evidence.append("Robustness testing implemented")
            test_evidence.append("tests/adversarial/")
        else:
            gaps.append("Robustness testing not verified")
            recommendations.append("Implement robustness testing")

        # Check adversarial robustness
        if security_config.get("adversarial_robustness"):
            evidence.append("Adversarial robustness implemented")
            code_modules.append("nethical/security/ai_ml_security.py")
        else:
            gaps.append("Adversarial robustness not verified")
            recommendations.append("Implement adversarial robustness measures")

        # Check cybersecurity
        if security_config.get("cybersecurity"):
            evidence.append("Cybersecurity measures implemented")
            code_modules.append("nethical/security/threat_modeling.py")
            test_evidence.append("tests/test_phase5_penetration_testing.py")
        else:
            gaps.append("Cybersecurity measures not verified")
            recommendations.append("Implement comprehensive cybersecurity")

        # Check fail-safe design
        if security_config.get("fail_safe"):
            evidence.append("Fail-safe design implemented")
        else:
            gaps.append("Fail-safe design not verified")
            recommendations.append("Implement fail-safe design")

        status = self._calculate_status(len(gaps))

        result = ArticleValidationResult(
            article=EUAIActArticle.ARTICLE_15,
            status=status,
            requirement="Accuracy, robustness and cybersecurity",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            code_modules=code_modules,
            test_evidence=test_evidence,
        )

        self.validation_results.append(result)
        return result

    def run_conformity_assessment(
        self,
        configs: Dict[str, Dict[str, Any]],
    ) -> ConformityAssessmentResult:
        """Run full EU AI Act conformity assessment.

        Performs comprehensive validation of all relevant articles
        and generates conformity assessment result.

        Args:
            configs: Dictionary with configuration for each article

        Returns:
            ConformityAssessmentResult with full assessment
        """
        # Clear previous results
        self.validation_results = []

        # Classify risk level first
        risk_level = self.classify_risk_level()

        if risk_level == AIRiskLevel.UNACCEPTABLE:
            return ConformityAssessmentResult(
                assessment_id=f"EU-AI-{uuid.uuid4().hex[:8].upper()}",
                risk_level=risk_level,
                assessment_date=datetime.now(timezone.utc),
                article_results=[],
                overall_status=ComplianceStatus.NON_COMPLIANT,
                compliance_score=0.0,
                certification_ready=False,
                recommendations=[
                    "System uses prohibited practices and cannot be deployed"
                ],
                technical_documentation_complete=False,
                qms_implemented=False,
                post_market_monitoring=False,
            )

        # For high-risk systems, validate all articles
        if risk_level == AIRiskLevel.HIGH:
            self.validate_article_9(configs.get("risk_management", {}))
            self.validate_article_10(configs.get("data_governance", {}))
            self.validate_article_11(configs.get("documentation", {}))
            self.validate_article_12(configs.get("logging", {}))
            self.validate_article_13(configs.get("transparency", {}))
            self.validate_article_14(configs.get("oversight", {}))
            self.validate_article_15(configs.get("security", {}))

        # Calculate compliance score
        if not self.validation_results:
            compliance_score = 100.0  # Minimal risk = compliant
        else:
            total = len(self.validation_results)
            compliant = sum(
                1
                for r in self.validation_results
                if r.status == ComplianceStatus.COMPLIANT
            )
            partial = sum(
                1
                for r in self.validation_results
                if r.status == ComplianceStatus.PARTIAL
            )
            compliance_score = (compliant + (partial * 0.5)) / total * 100

        # Determine overall status
        non_compliant = any(
            r.status == ComplianceStatus.NON_COMPLIANT for r in self.validation_results
        )

        if non_compliant:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif compliance_score < 100:
            overall_status = ComplianceStatus.PARTIAL
        else:
            overall_status = ComplianceStatus.COMPLIANT

        # Collect recommendations
        recommendations = []
        for result in self.validation_results:
            recommendations.extend(result.recommendations)

        # Check additional requirements
        tech_docs_complete = configs.get("documentation", {}).get("complete", False)
        qms = configs.get("quality_management", {}).get("implemented", False)
        pmm = configs.get("post_market", {}).get("monitoring", False)

        return ConformityAssessmentResult(
            assessment_id=f"EU-AI-{uuid.uuid4().hex[:8].upper()}",
            risk_level=risk_level,
            assessment_date=datetime.now(timezone.utc),
            article_results=self.validation_results,
            overall_status=overall_status,
            compliance_score=round(compliance_score, 2),
            certification_ready=compliance_score >= self.CERTIFICATION_THRESHOLD,
            recommendations=recommendations,
            technical_documentation_complete=tech_docs_complete,
            qms_implemented=qms,
            post_market_monitoring=pmm,
        )

    def _calculate_status(self, gap_count: int) -> ComplianceStatus:
        """Calculate compliance status based on gap count.

        Args:
            gap_count: Number of gaps identified

        Returns:
            ComplianceStatus
        """
        if gap_count == 0:
            return ComplianceStatus.COMPLIANT
        elif gap_count <= 2:
            return ComplianceStatus.PARTIAL
        else:
            return ComplianceStatus.NON_COMPLIANT

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of validation results.

        Returns:
            Dictionary with compliance summary
        """
        if not self.validation_results:
            return {
                "total_articles_validated": 0,
                "risk_level": self.risk_level.value if self.risk_level else None,
                "status": "No validations performed",
            }

        status_counts = {
            "compliant": 0,
            "partial": 0,
            "non_compliant": 0,
        }

        for result in self.validation_results:
            status_counts[result.status.value] = (
                status_counts.get(result.status.value, 0) + 1
            )

        total = len(self.validation_results)
        compliant = status_counts["compliant"]
        partial = status_counts["partial"]

        score = ((compliant + (partial * 0.5)) / total * 100) if total > 0 else 0

        return {
            "total_articles_validated": total,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "status_breakdown": status_counts,
            "compliance_score": round(score, 2),
            "certification_ready": score >= self.CERTIFICATION_THRESHOLD,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


__all__ = [
    "EUAIActValidator",
    "AIRiskLevel",
    "EUAIActArticle",
    "ComplianceStatus",
    "HighRiskDomain",
    "ArticleValidationResult",
    "ConformityAssessmentResult",
]
