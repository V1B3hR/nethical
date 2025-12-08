"""
Regulatory Compliance Framework for EU AI Act, UK Law, and US Standards

This module provides comprehensive compliance capabilities for:
- EU AI Act (Artificial Intelligence Act) requirements
- UK Law (GDPR, DPA 2018, NHS DSPT) compliance
- US Standards (NIST AI RMF, HIPAA, SOC2) alignment

It includes risk classification, transparency requirements, human oversight,
conformity assessment, and automated regulatory mapping table generation.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path


class AIRiskLevel(str, Enum):
    """EU AI Act risk classification levels"""

    UNACCEPTABLE = "unacceptable"  # Banned AI systems
    HIGH = "high"  # Requires conformity assessment
    LIMITED = "limited"  # Transparency obligations
    MINIMAL = "minimal"  # No restrictions


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks"""

    EU_AI_ACT = "eu_ai_act"
    UK_GDPR = "uk_gdpr"
    UK_DPA_2018 = "uk_dpa_2018"
    UK_NHS_DSPT = "uk_nhs_dspt"
    US_NIST_AI_RMF = "us_nist_ai_rmf"
    US_HIPAA = "us_hipaa"
    US_SOC2 = "us_soc2"
    US_NIST_800_53 = "us_nist_800_53"


class ComplianceStatus(str, Enum):
    """Compliance status states"""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


class ControlCategory(str, Enum):
    """Control categories"""

    TRANSPARENCY = "transparency"
    EXPLAINABILITY = "explainability"
    HUMAN_OVERSIGHT = "human_oversight"
    DATA_GOVERNANCE = "data_governance"
    RISK_MANAGEMENT = "risk_management"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    CONFORMITY_ASSESSMENT = "conformity_assessment"
    INCIDENT_RESPONSE = "incident_response"
    AUDIT_LOGGING = "audit_logging"
    ACCESS_CONTROL = "access_control"
    SECURITY = "security"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"


@dataclass
class RegulatoryRequirement:
    """Represents a single regulatory requirement"""

    id: str
    framework: RegulatoryFramework
    article: str  # e.g., "Article 13" for EU AI Act
    title: str
    description: str
    category: ControlCategory
    mandatory: bool = True
    high_risk_only: bool = False
    implementation_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    code_modules: List[str] = field(default_factory=list)
    test_evidence: List[str] = field(default_factory=list)
    documentation: List[str] = field(default_factory=list)
    remediation_notes: str = ""


@dataclass
class RegulatoryMapping:
    """Maps requirements to implementation artifacts"""

    requirement_id: str
    framework: RegulatoryFramework
    code_reference: str  # e.g., "nethical/explainability/transparency_report.py"
    test_reference: str  # e.g., "tests/test_transparency.py::test_report_generation"
    doc_reference: str  # e.g., "docs/compliance/EU_AI_ACT_MAPPING.md"
    status: ComplianceStatus
    last_verified: datetime
    verified_by: str = "automated"
    notes: str = ""


class EUAIActCompliance:
    """
    EU AI Act compliance requirements and validation.

    Implements Articles 9-15 for high-risk AI systems:
    - Article 9: Risk management system
    - Article 10: Data and data governance
    - Article 11: Technical documentation
    - Article 12: Record-keeping
    - Article 13: Transparency and information to users
    - Article 14: Human oversight
    - Article 15: Accuracy, robustness and cybersecurity
    """

    def __init__(self) -> None:
        self.requirements: Dict[str, RegulatoryRequirement] = {}
        self._initialize_requirements()

    def _initialize_requirements(self) -> None:
        """Initialize EU AI Act requirements"""

        # Article 9: Risk Management System
        self.requirements["EU-AI-9.1"] = RegulatoryRequirement(
            id="EU-AI-9.1",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 9",
            title="Risk Management System",
            description="Establish, implement, document and maintain a risk management system",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            high_risk_only=True,
            code_modules=[
                "nethical/core/risk_engine.py",
                "nethical/core/governance.py",
            ],
            test_evidence=["tests/test_governance_features.py"],
            documentation=["docs/compliance/NIST_RMF_MAPPING.md"],
        )

        self.requirements["EU-AI-9.2"] = RegulatoryRequirement(
            id="EU-AI-9.2",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 9",
            title="Risk Identification and Analysis",
            description="Identification and analysis of known and foreseeable risks",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            high_risk_only=True,
            code_modules=[
                "nethical/core/anomaly_detector.py",
                "nethical/core/ml_blended_risk.py",
            ],
            test_evidence=["tests/test_anomaly_classifier.py"],
            documentation=["docs/security/threat_model.md"],
        )

        self.requirements["EU-AI-9.3"] = RegulatoryRequirement(
            id="EU-AI-9.3",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 9",
            title="Risk Mitigation Measures",
            description="Adoption of suitable risk management measures",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            high_risk_only=True,
            code_modules=["nethical/core/quarantine.py", "nethical/policy/engine.py"],
            test_evidence=["tests/test_phase3.py"],
            documentation=["docs/security/mitigations.md"],
        )

        # Article 10: Data and Data Governance
        self.requirements["EU-AI-10.1"] = RegulatoryRequirement(
            id="EU-AI-10.1",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 10",
            title="Training Data Governance",
            description="Training, validation and testing data sets shall be subject to data governance",
            category=ControlCategory.DATA_GOVERNANCE,
            mandatory=True,
            high_risk_only=True,
            code_modules=[
                "nethical/core/data_minimization.py",
                "nethical/security/data_compliance.py",
            ],
            test_evidence=["tests/test_privacy_features.py"],
            documentation=["docs/privacy/DPIA_template.md"],
        )

        self.requirements["EU-AI-10.2"] = RegulatoryRequirement(
            id="EU-AI-10.2",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 10",
            title="Data Quality and Bias Mitigation",
            description="Data sets shall be relevant, representative, free of errors and complete",
            category=ControlCategory.FAIRNESS,
            mandatory=True,
            high_risk_only=True,
            code_modules=["nethical/core/fairness_sampler.py"],
            test_evidence=["tests/test_regionalization.py"],
            documentation=["governance/fairness_recalibration_report.md"],
        )

        # Article 11: Technical Documentation
        self.requirements["EU-AI-11.1"] = RegulatoryRequirement(
            id="EU-AI-11.1",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 11",
            title="Technical Documentation",
            description="Technical documentation shall be drawn up before placing on market",
            category=ControlCategory.TECHNICAL_DOCUMENTATION,
            mandatory=True,
            high_risk_only=True,
            code_modules=[],
            test_evidence=[],
            documentation=[
                "ARCHITECTURE.md",
                "docs/API_USAGE.md",
                "docs/EXPLAINABILITY_GUIDE.md",
            ],
        )

        # Article 12: Record-keeping
        self.requirements["EU-AI-12.1"] = RegulatoryRequirement(
            id="EU-AI-12.1",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 12",
            title="Automatic Logging",
            description="High-risk AI systems shall allow for automatic recording of events (logs)",
            category=ControlCategory.AUDIT_LOGGING,
            mandatory=True,
            high_risk_only=True,
            code_modules=[
                "nethical/security/audit_logging.py",
                "nethical/core/audit_merkle.py",
            ],
            test_evidence=["tests/test_train_audit_logging.py"],
            documentation=["docs/AUDIT_LOGGING_GUIDE.md"],
        )

        # Article 13: Transparency
        self.requirements["EU-AI-13.1"] = RegulatoryRequirement(
            id="EU-AI-13.1",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 13",
            title="Transparency and Information",
            description="Design and develop in such a way as to ensure operation is sufficiently transparent",
            category=ControlCategory.TRANSPARENCY,
            mandatory=True,
            high_risk_only=True,
            code_modules=[
                "nethical/explainability/transparency_report.py",
                "nethical/explainability/quarterly_transparency.py",
            ],
            test_evidence=["tests/test_explainability/"],
            documentation=["docs/transparency/"],
        )

        self.requirements["EU-AI-13.2"] = RegulatoryRequirement(
            id="EU-AI-13.2",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 13",
            title="Instructions for Use",
            description="Accompanied by instructions for use in an appropriate digital format",
            category=ControlCategory.TRANSPARENCY,
            mandatory=True,
            high_risk_only=True,
            code_modules=[],
            test_evidence=[],
            documentation=["README.md", "docs/API_USAGE.md"],
        )

        # Article 14: Human Oversight
        self.requirements["EU-AI-14.1"] = RegulatoryRequirement(
            id="EU-AI-14.1",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 14",
            title="Human Oversight Design",
            description="Design and develop with appropriate human-machine interface tools",
            category=ControlCategory.HUMAN_OVERSIGHT,
            mandatory=True,
            high_risk_only=True,
            code_modules=[
                "nethical/core/human_feedback.py",
                "nethical/governance/human_review.py",
            ],
            test_evidence=["tests/test_governance_features.py"],
            documentation=["docs/governance/governance_drivers.md"],
        )

        self.requirements["EU-AI-14.2"] = RegulatoryRequirement(
            id="EU-AI-14.2",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 14",
            title="Human Oversight Capabilities",
            description="Enable human oversight to fully understand AI capabilities and limitations",
            category=ControlCategory.HUMAN_OVERSIGHT,
            mandatory=True,
            high_risk_only=True,
            code_modules=["nethical/explainability/decision_explainer.py"],
            test_evidence=["tests/test_advanced_explainability.py"],
            documentation=["docs/ADVANCED_EXPLAINABILITY_GUIDE.md"],
        )

        # Article 15: Accuracy, Robustness and Cybersecurity
        self.requirements["EU-AI-15.1"] = RegulatoryRequirement(
            id="EU-AI-15.1",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 15",
            title="Accuracy",
            description="Achieve appropriate level of accuracy specified in instructions for use",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            high_risk_only=True,
            code_modules=["nethical/governance/ethics_benchmark.py"],
            test_evidence=["tests/test_performance_benchmarks.py"],
            documentation=["docs/BENCHMARK_PLAN.md"],
        )

        self.requirements["EU-AI-15.2"] = RegulatoryRequirement(
            id="EU-AI-15.2",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 15",
            title="Robustness",
            description="Be resilient regarding errors, faults, or inconsistencies",
            category=ControlCategory.SECURITY,
            mandatory=True,
            high_risk_only=True,
            code_modules=["nethical/security/ai_ml_security.py"],
            test_evidence=["tests/adversarial/"],
            documentation=["docs/security/AI_ML_SECURITY_GUIDE.md"],
        )

        self.requirements["EU-AI-15.3"] = RegulatoryRequirement(
            id="EU-AI-15.3",
            framework=RegulatoryFramework.EU_AI_ACT,
            article="Article 15",
            title="Cybersecurity",
            description="Resilient against attempts to exploit system vulnerabilities",
            category=ControlCategory.SECURITY,
            mandatory=True,
            high_risk_only=True,
            code_modules=[
                "nethical/security/threat_modeling.py",
                "nethical/security/penetration_testing.py",
            ],
            test_evidence=[
                "tests/test_phase5_penetration_testing.py",
                "tests/test_phase5_threat_modeling.py",
            ],
            documentation=["docs/SECURITY_HARDENING_GUIDE.md"],
        )

    def classify_risk_level(
        self, system_characteristics: Dict[str, Any]
    ) -> AIRiskLevel:
        """
        Classify AI system risk level according to EU AI Act.

        Args:
            system_characteristics: Dict with system characteristics

        Returns:
            AIRiskLevel classification
        """
        # Unacceptable risk criteria (Article 5)
        if system_characteristics.get("social_scoring", False):
            return AIRiskLevel.UNACCEPTABLE
        if system_characteristics.get("subliminal_manipulation", False):
            return AIRiskLevel.UNACCEPTABLE
        if system_characteristics.get("real_time_biometric_public", False):
            return AIRiskLevel.UNACCEPTABLE

        # High risk criteria (Annex III)
        high_risk_domains = [
            "biometric_identification",
            "critical_infrastructure",
            "education_vocational",
            "employment_recruitment",
            "essential_services",
            "law_enforcement",
            "migration_asylum",
            "justice_democratic",
        ]

        if any(
            system_characteristics.get(domain, False) for domain in high_risk_domains
        ):
            return AIRiskLevel.HIGH

        # Limited risk (transparency obligations)
        limited_risk_types = [
            "chatbot",
            "emotion_recognition",
            "biometric_categorization",
            "deep_fake",
        ]

        if any(system_characteristics.get(t, False) for t in limited_risk_types):
            return AIRiskLevel.LIMITED

        return AIRiskLevel.MINIMAL

    def get_applicable_requirements(
        self, risk_level: AIRiskLevel
    ) -> List[RegulatoryRequirement]:
        """Get requirements applicable to a given risk level"""
        if risk_level == AIRiskLevel.UNACCEPTABLE:
            return []  # System should not be deployed

        requirements = []
        for req in self.requirements.values():
            if risk_level == AIRiskLevel.HIGH:
                requirements.append(req)
            elif not req.high_risk_only:
                requirements.append(req)

        return requirements

    def assess_requirement(
        self, requirement_id: str, status: ComplianceStatus, notes: str = ""
    ) -> bool:
        """Update compliance status for a requirement"""
        if requirement_id not in self.requirements:
            return False

        self.requirements[requirement_id].implementation_status = status
        self.requirements[requirement_id].remediation_notes = notes
        return True


class UKLawCompliance:
    """
    UK Law compliance requirements.

    Covers:
    - UK GDPR (retained EU law)
    - DPA 2018 (Data Protection Act 2018)
    - NHS DSPT (Data Security and Protection Toolkit)
    """

    def __init__(self) -> None:
        self.requirements: Dict[str, RegulatoryRequirement] = {}
        self._initialize_requirements()

    def _initialize_requirements(self) -> None:
        """Initialize UK law requirements"""

        # UK GDPR Requirements
        self.requirements["UK-GDPR-5"] = RegulatoryRequirement(
            id="UK-GDPR-5",
            framework=RegulatoryFramework.UK_GDPR,
            article="Article 5",
            title="Principles of Processing",
            description="Lawfulness, fairness, transparency, purpose limitation, data minimization",
            category=ControlCategory.PRIVACY,
            mandatory=True,
            code_modules=[
                "nethical/core/data_minimization.py",
                "nethical/core/redaction_pipeline.py",
            ],
            test_evidence=["tests/test_privacy_features.py"],
            documentation=["docs/privacy/DPIA_template.md"],
        )

        self.requirements["UK-GDPR-6"] = RegulatoryRequirement(
            id="UK-GDPR-6",
            framework=RegulatoryFramework.UK_GDPR,
            article="Article 6",
            title="Lawful Basis for Processing",
            description="Processing shall have lawful basis (consent, contract, legal obligation, etc.)",
            category=ControlCategory.PRIVACY,
            mandatory=True,
            code_modules=["nethical/security/data_compliance.py"],
            test_evidence=["tests/test_privacy_features.py"],
            documentation=["docs/privacy/DPIA_template.md"],
        )

        self.requirements["UK-GDPR-12-22"] = RegulatoryRequirement(
            id="UK-GDPR-12-22",
            framework=RegulatoryFramework.UK_GDPR,
            article="Articles 12-22",
            title="Data Subject Rights",
            description="Right to access, rectification, erasure, restriction, portability, objection",
            category=ControlCategory.PRIVACY,
            mandatory=True,
            code_modules=["nethical/security/data_compliance.py"],
            test_evidence=["tests/test_privacy_features.py"],
            documentation=["docs/privacy/DSR_runbook.md"],
        )

        self.requirements["UK-GDPR-25"] = RegulatoryRequirement(
            id="UK-GDPR-25",
            framework=RegulatoryFramework.UK_GDPR,
            article="Article 25",
            title="Data Protection by Design and Default",
            description="Implement appropriate technical and organisational measures",
            category=ControlCategory.PRIVACY,
            mandatory=True,
            code_modules=["nethical/core/differential_privacy.py"],
            test_evidence=["tests/test_privacy_features.py"],
            documentation=["docs/F3_PRIVACY_GUIDE.md"],
        )

        self.requirements["UK-GDPR-32"] = RegulatoryRequirement(
            id="UK-GDPR-32",
            framework=RegulatoryFramework.UK_GDPR,
            article="Article 32",
            title="Security of Processing",
            description="Implement appropriate technical and organisational security measures",
            category=ControlCategory.SECURITY,
            mandatory=True,
            code_modules=[
                "nethical/security/encryption.py",
                "nethical/security/auth.py",
            ],
            test_evidence=["tests/test_security_hardening.py"],
            documentation=["docs/SECURITY_HARDENING_GUIDE.md"],
        )

        self.requirements["UK-GDPR-33-34"] = RegulatoryRequirement(
            id="UK-GDPR-33-34",
            framework=RegulatoryFramework.UK_GDPR,
            article="Articles 33-34",
            title="Breach Notification",
            description="Notify supervisory authority and data subjects of personal data breaches",
            category=ControlCategory.INCIDENT_RESPONSE,
            mandatory=True,
            code_modules=["nethical/security/soc_integration.py"],
            test_evidence=["tests/test_phase4_operational_security.py"],
            documentation=["docs/security/red_team_report_template.md"],
        )

        self.requirements["UK-GDPR-35"] = RegulatoryRequirement(
            id="UK-GDPR-35",
            framework=RegulatoryFramework.UK_GDPR,
            article="Article 35",
            title="Data Protection Impact Assessment",
            description="Carry out DPIA for high-risk processing operations",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            code_modules=[],
            test_evidence=[],
            documentation=["docs/privacy/DPIA_template.md"],
        )

        # DPA 2018 Specific Requirements
        self.requirements["UK-DPA-35"] = RegulatoryRequirement(
            id="UK-DPA-35",
            framework=RegulatoryFramework.UK_DPA_2018,
            article="Section 35",
            title="Law Enforcement Processing Principles",
            description="Specific requirements for law enforcement data processing",
            category=ControlCategory.DATA_GOVERNANCE,
            mandatory=False,
            code_modules=["nethical/policy/engine.py"],
            test_evidence=["tests/test_phase3.py"],
            documentation=["policies/common/data_classification.yaml"],
        )

        self.requirements["UK-DPA-64"] = RegulatoryRequirement(
            id="UK-DPA-64",
            framework=RegulatoryFramework.UK_DPA_2018,
            article="Section 64",
            title="Automated Decision-Making",
            description="Rights related to automated decision-making and profiling",
            category=ControlCategory.TRANSPARENCY,
            mandatory=True,
            code_modules=["nethical/explainability/decision_explainer.py"],
            test_evidence=["tests/test_advanced_explainability.py"],
            documentation=["docs/EXPLAINABILITY_GUIDE.md"],
        )

        # NHS DSPT Requirements
        self.requirements["NHS-DSPT-1"] = RegulatoryRequirement(
            id="NHS-DSPT-1",
            framework=RegulatoryFramework.UK_NHS_DSPT,
            article="Standard 1",
            title="Personal Confidential Data",
            description="Staff ensure personal confidential data is handled appropriately",
            category=ControlCategory.DATA_GOVERNANCE,
            mandatory=True,
            code_modules=["nethical/core/redaction_pipeline.py"],
            test_evidence=["tests/test_privacy_features.py"],
            documentation=["docs/privacy/DPIA_template.md"],
        )

        self.requirements["NHS-DSPT-3"] = RegulatoryRequirement(
            id="NHS-DSPT-3",
            framework=RegulatoryFramework.UK_NHS_DSPT,
            article="Standard 3",
            title="Security Training",
            description="All staff complete annual data security awareness training",
            category=ControlCategory.SECURITY,
            mandatory=True,
            code_modules=[],
            test_evidence=[],
            documentation=["docs/TRAINING_GUIDE.md"],
        )

        self.requirements["NHS-DSPT-7"] = RegulatoryRequirement(
            id="NHS-DSPT-7",
            framework=RegulatoryFramework.UK_NHS_DSPT,
            article="Standard 7",
            title="Managing Access to Data and Systems",
            description="Robust access controls and authentication",
            category=ControlCategory.ACCESS_CONTROL,
            mandatory=True,
            code_modules=["nethical/core/rbac.py", "nethical/security/auth.py"],
            test_evidence=["tests/test_security_hardening.py"],
            documentation=["docs/security/SSO_SAML_GUIDE.md"],
        )

        self.requirements["NHS-DSPT-8"] = RegulatoryRequirement(
            id="NHS-DSPT-8",
            framework=RegulatoryFramework.UK_NHS_DSPT,
            article="Standard 8",
            title="Unsupported Systems",
            description="No unsupported operating systems, software or internet browsers",
            category=ControlCategory.SECURITY,
            mandatory=True,
            code_modules=[],
            test_evidence=[],
            documentation=["docs/SUPPLY_CHAIN_SECURITY_GUIDE.md"],
        )

        self.requirements["NHS-DSPT-10"] = RegulatoryRequirement(
            id="NHS-DSPT-10",
            framework=RegulatoryFramework.UK_NHS_DSPT,
            article="Standard 10",
            title="Accountable Suppliers",
            description="IT suppliers are held accountable via contracts",
            category=ControlCategory.DATA_GOVERNANCE,
            mandatory=True,
            code_modules=[],
            test_evidence=[],
            documentation=["docs/SUPPLY_CHAIN_SECURITY_GUIDE.md", "SBOM.json"],
        )


class USStandardsCompliance:
    """
    US Standards compliance requirements.

    Covers:
    - NIST AI RMF (AI Risk Management Framework)
    - HIPAA (Health Insurance Portability and Accountability Act)
    - SOC2 (System and Organization Controls 2)
    """

    def __init__(self) -> None:
        self.requirements: Dict[str, RegulatoryRequirement] = {}
        self._initialize_requirements()

    def _initialize_requirements(self) -> None:
        """Initialize US standards requirements"""

        # NIST AI RMF - GOVERN Function
        self.requirements["NIST-RMF-GV1"] = RegulatoryRequirement(
            id="NIST-RMF-GV1",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="GOVERN 1",
            title="Governance Policies and Procedures",
            description="Policies and procedures are in place to govern AI risk management",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            code_modules=["nethical/core/governance.py", "nethical/policy/engine.py"],
            test_evidence=["tests/test_governance_features.py"],
            documentation=["docs/governance/governance_drivers.md"],
        )

        self.requirements["NIST-RMF-GV2"] = RegulatoryRequirement(
            id="NIST-RMF-GV2",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="GOVERN 2",
            title="Accountability Structures",
            description="Accountability structures are in place for AI risk management",
            category=ControlCategory.HUMAN_OVERSIGHT,
            mandatory=True,
            code_modules=["nethical/governance/human_review.py"],
            test_evidence=["tests/test_integrated_governance.py"],
            documentation=["docs/compliance/NIST_RMF_MAPPING.md"],
        )

        # NIST AI RMF - MAP Function
        self.requirements["NIST-RMF-MP1"] = RegulatoryRequirement(
            id="NIST-RMF-MP1",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="MAP 1",
            title="Context Establishment",
            description="Context is established and understood",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            code_modules=["nethical/core/fairness_sampler.py"],
            test_evidence=["tests/test_regionalization.py"],
            documentation=["docs/REGIONAL_DEPLOYMENT_GUIDE.md"],
        )

        self.requirements["NIST-RMF-MP3"] = RegulatoryRequirement(
            id="NIST-RMF-MP3",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="MAP 3",
            title="AI Capabilities and Limitations",
            description="AI capabilities and limitations are understood",
            category=ControlCategory.TRANSPARENCY,
            mandatory=True,
            code_modules=["nethical/explainability/decision_explainer.py"],
            test_evidence=["tests/test_advanced_explainability.py"],
            documentation=["docs/EXPLAINABILITY_GUIDE.md"],
        )

        # NIST AI RMF - MEASURE Function
        self.requirements["NIST-RMF-MS1"] = RegulatoryRequirement(
            id="NIST-RMF-MS1",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="MEASURE 1",
            title="Risk Measurement",
            description="Appropriate methods and metrics are identified and applied",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            code_modules=[
                "nethical/core/risk_engine.py",
                "nethical/core/ml_blended_risk.py",
            ],
            test_evidence=["tests/test_performance_benchmarks.py"],
            documentation=["docs/BENCHMARK_PLAN.md"],
        )

        self.requirements["NIST-RMF-MS2"] = RegulatoryRequirement(
            id="NIST-RMF-MS2",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="MEASURE 2",
            title="AI Testing",
            description="AI systems are evaluated for trustworthy characteristics",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            code_modules=["nethical/governance/ethics_benchmark.py"],
            test_evidence=[
                "tests/adversarial/",
                "tests/test_performance_benchmarks.py",
            ],
            documentation=["docs/Benchmark_plan.md"],
        )

        # NIST AI RMF - MANAGE Function
        self.requirements["NIST-RMF-MG1"] = RegulatoryRequirement(
            id="NIST-RMF-MG1",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="MANAGE 1",
            title="Risk Response",
            description="AI risks based on assessments are prioritized, responded to and managed",
            category=ControlCategory.RISK_MANAGEMENT,
            mandatory=True,
            code_modules=["nethical/core/quarantine.py"],
            test_evidence=["tests/test_phase3.py"],
            documentation=["docs/security/mitigations.md"],
        )

        self.requirements["NIST-RMF-MG3"] = RegulatoryRequirement(
            id="NIST-RMF-MG3",
            framework=RegulatoryFramework.US_NIST_AI_RMF,
            article="MANAGE 3",
            title="Incident Response",
            description="AI system incident response procedures are established",
            category=ControlCategory.INCIDENT_RESPONSE,
            mandatory=True,
            code_modules=["nethical/security/soc_integration.py"],
            test_evidence=["tests/test_phase4_operational_security.py"],
            documentation=["docs/security/red_team_report_template.md"],
        )

        # SOC2 Trust Service Criteria
        self.requirements["SOC2-CC6.1"] = RegulatoryRequirement(
            id="SOC2-CC6.1",
            framework=RegulatoryFramework.US_SOC2,
            article="CC6.1",
            title="Logical Access Security",
            description="Logical access security software, infrastructure, and architectures",
            category=ControlCategory.ACCESS_CONTROL,
            mandatory=True,
            code_modules=["nethical/core/rbac.py", "nethical/security/auth.py"],
            test_evidence=["tests/test_security_hardening.py"],
            documentation=["docs/security/SSO_SAML_GUIDE.md"],
        )

        self.requirements["SOC2-CC6.6"] = RegulatoryRequirement(
            id="SOC2-CC6.6",
            framework=RegulatoryFramework.US_SOC2,
            article="CC6.6",
            title="Security Monitoring",
            description="System intrusion detection and response procedures",
            category=ControlCategory.SECURITY,
            mandatory=True,
            code_modules=["nethical/security/anomaly_detection.py"],
            test_evidence=["tests/test_phase4_operational_security.py"],
            documentation=["docs/security/threat_model.md"],
        )

        self.requirements["SOC2-CC7.1"] = RegulatoryRequirement(
            id="SOC2-CC7.1",
            framework=RegulatoryFramework.US_SOC2,
            article="CC7.1",
            title="Change Management",
            description="Configuration changes are managed through change management process",
            category=ControlCategory.DATA_GOVERNANCE,
            mandatory=True,
            code_modules=[
                "nethical/core/policy_diff.py",
                "nethical/policy/release_management.py",
            ],
            test_evidence=["tests/test_train_governance.py"],
            documentation=["docs/versioning.md"],
        )

        self.requirements["SOC2-CC7.4"] = RegulatoryRequirement(
            id="SOC2-CC7.4",
            framework=RegulatoryFramework.US_SOC2,
            article="CC7.4",
            title="Incident Response",
            description="Incident response procedures are established",
            category=ControlCategory.INCIDENT_RESPONSE,
            mandatory=True,
            code_modules=["nethical/security/soc_integration.py"],
            test_evidence=["tests/test_phase4_operational_security.py"],
            documentation=["docs/security/red_team_report_template.md"],
        )


class RegulatoryMappingGenerator:
    """
    Generates regulatory mapping tables and compliance reports.

    Creates auto-generated tables showing which Nethical component
    (code, doc, test) fulfills each compliance requirement.
    """

    def __init__(self) -> None:
        self.eu_ai_act = EUAIActCompliance()
        self.uk_law = UKLawCompliance()
        self.us_standards = USStandardsCompliance()
        self.mappings: List[RegulatoryMapping] = []

    def _collect_all_requirements(self) -> List[RegulatoryRequirement]:
        """Collect all requirements from all frameworks"""
        requirements = []
        requirements.extend(self.eu_ai_act.requirements.values())
        requirements.extend(self.uk_law.requirements.values())
        requirements.extend(self.us_standards.requirements.values())
        return requirements

    def generate_mapping_table(self) -> Dict[str, Any]:
        """
        Generate comprehensive regulatory mapping table.

        Returns:
            Dictionary containing the complete mapping table
        """
        requirements = self._collect_all_requirements()

        mapping_table = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator_version": "1.0.0",
                "total_requirements": len(requirements),
                "frameworks": [f.value for f in RegulatoryFramework],
            },
            "summary": {"by_framework": {}, "by_category": {}, "by_status": {}},
            "requirements": [],
        }

        # Process each requirement
        for req in requirements:
            req_entry = {
                "id": req.id,
                "framework": req.framework.value,
                "article": req.article,
                "title": req.title,
                "description": req.description,
                "category": req.category.value,
                "mandatory": req.mandatory,
                "high_risk_only": req.high_risk_only,
                "status": req.implementation_status.value,
                "code_modules": req.code_modules,
                "test_evidence": req.test_evidence,
                "documentation": req.documentation,
                "remediation_notes": req.remediation_notes,
            }
            mapping_table["requirements"].append(req_entry)

            # Update summaries
            framework = req.framework.value
            category = req.category.value
            status = req.implementation_status.value

            mapping_table["summary"]["by_framework"][framework] = (
                mapping_table["summary"]["by_framework"].get(framework, 0) + 1
            )
            mapping_table["summary"]["by_category"][category] = (
                mapping_table["summary"]["by_category"].get(category, 0) + 1
            )
            mapping_table["summary"]["by_status"][status] = (
                mapping_table["summary"]["by_status"].get(status, 0) + 1
            )

        return mapping_table

    def generate_markdown_report(self) -> str:
        """
        Generate regulatory mapping table in Markdown format.

        Returns:
            Markdown formatted string
        """
        mapping = self.generate_mapping_table()

        md = "# Regulatory Compliance Mapping Table\n\n"
        md += f"**Generated:** {mapping['metadata']['generated_at']}\n\n"
        md += f"**Total Requirements:** {mapping['metadata']['total_requirements']}\n\n"

        # Summary by Framework
        md += "## Summary by Framework\n\n"
        md += "| Framework | Requirements |\n"
        md += "|-----------|-------------|\n"
        for framework, count in mapping["summary"]["by_framework"].items():
            md += f"| {framework} | {count} |\n"
        md += "\n"

        # Summary by Status
        md += "## Compliance Status Summary\n\n"
        md += "| Status | Count |\n"
        md += "|--------|-------|\n"
        for status, count in mapping["summary"]["by_status"].items():
            md += f"| {status} | {count} |\n"
        md += "\n"

        # Detailed Requirements by Framework
        for framework in RegulatoryFramework:
            framework_reqs = [
                r for r in mapping["requirements"] if r["framework"] == framework.value
            ]

            if not framework_reqs:
                continue

            md += f"## {framework.value.upper().replace('_', ' ')}\n\n"
            md += "| ID | Article | Title | Status | Code Modules | Tests | Docs |\n"
            md += "|----|---------|-------|--------|--------------|-------|------|\n"

            for req in framework_reqs:
                code = (
                    ", ".join(req["code_modules"][:2]) if req["code_modules"] else "-"
                )
                if len(req["code_modules"]) > 2:
                    code += "..."
                tests = (
                    ", ".join(req["test_evidence"][:2]) if req["test_evidence"] else "-"
                )
                if len(req["test_evidence"]) > 2:
                    tests += "..."
                docs = (
                    ", ".join(req["documentation"][:2]) if req["documentation"] else "-"
                )
                if len(req["documentation"]) > 2:
                    docs += "..."

                status_icon = {
                    "compliant": "✅",
                    "partial": "🟡",
                    "non_compliant": "❌",
                    "not_applicable": "N/A",
                    "pending_review": "🔄",
                }.get(req["status"], "❓")

                md += f"| {req['id']} | {req['article']} | {req['title']} | "
                md += f"{status_icon} | {code} | {tests} | {docs} |\n"

            md += "\n"

        # Cross-Reference Matrix
        md += "## Cross-Reference Matrix\n\n"
        md += "This matrix shows which controls satisfy multiple frameworks.\n\n"
        md += "| Category | EU AI Act | UK GDPR | UK DPA 2018 | NHS DSPT | NIST AI RMF | SOC2 |\n"
        md += "|----------|-----------|---------|-------------|----------|-------------|------|\n"

        for category in ControlCategory:
            row = [category.value]
            for framework in [
                RegulatoryFramework.EU_AI_ACT,
                RegulatoryFramework.UK_GDPR,
                RegulatoryFramework.UK_DPA_2018,
                RegulatoryFramework.UK_NHS_DSPT,
                RegulatoryFramework.US_NIST_AI_RMF,
                RegulatoryFramework.US_SOC2,
            ]:
                count = sum(
                    1
                    for r in mapping["requirements"]
                    if r["framework"] == framework.value
                    and r["category"] == category.value
                )
                row.append(str(count) if count > 0 else "-")
            md += "| " + " | ".join(row) + " |\n"

        md += "\n"

        # Footer
        md += "---\n"
        md += f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"

        return md

    def generate_json_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate regulatory mapping table in JSON format.

        Args:
            output_path: Optional path to save the JSON file

        Returns:
            JSON formatted string
        """
        mapping = self.generate_mapping_table()
        json_str = json.dumps(mapping, indent=2)

        if output_path:
            Path(output_path).write_text(json_str)

        return json_str

    def generate_audit_report(self, auditor_name: str = "System") -> Dict[str, Any]:
        """
        Generate audit report for compliance assessment.

        Args:
            auditor_name: Name of the auditor or system

        Returns:
            Audit report dictionary
        """
        mapping = self.generate_mapping_table()

        # Calculate compliance scores
        total = len(mapping["requirements"])
        compliant = mapping["summary"]["by_status"].get("compliant", 0)
        partial = mapping["summary"]["by_status"].get("partial", 0)
        non_compliant = mapping["summary"]["by_status"].get("non_compliant", 0)

        compliance_score = (
            ((compliant + (partial * 0.5)) / total * 100) if total > 0 else 0
        )

        audit_report = {
            "report_id": f"AUDIT-{uuid.uuid4().hex[:8].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "auditor": auditor_name,
            "scope": "Full regulatory compliance assessment",
            "summary": {
                "total_requirements": total,
                "compliant": compliant,
                "partial": partial,
                "non_compliant": non_compliant,
                "compliance_score": round(compliance_score, 2),
                "certification_ready": compliance_score >= 90,
            },
            "findings": [],
            "recommendations": [],
        }

        # Add findings for non-compliant items
        for req in mapping["requirements"]:
            if req["status"] in ["non_compliant", "partial"]:
                audit_report["findings"].append(
                    {
                        "requirement_id": req["id"],
                        "framework": req["framework"],
                        "title": req["title"],
                        "status": req["status"],
                        "category": req["category"],
                        "remediation": req["remediation_notes"]
                        or "Review and implement required controls",
                    }
                )

        # Add recommendations
        if non_compliant > 0:
            audit_report["recommendations"].append(
                f"Address {non_compliant} non-compliant requirements before certification"
            )
        if partial > 0:
            audit_report["recommendations"].append(
                f"Complete implementation of {partial} partially compliant requirements"
            )
        if compliance_score < 90:
            audit_report["recommendations"].append(
                "Achieve 90% compliance score for certification readiness"
            )

        return audit_report


def generate_regulatory_mapping_table(
    output_dir: str = "docs/compliance",
) -> Dict[str, str]:
    """
    Generate regulatory mapping table and save to files.

    Args:
        output_dir: Directory to save output files

    Returns:
        Dictionary with paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = RegulatoryMappingGenerator()

    # Generate all formats
    md_content = generator.generate_markdown_report()
    json_content = generator.generate_json_report()
    audit_report = generator.generate_audit_report()

    # Save files
    md_path = output_path / "REGULATORY_MAPPING_TABLE.md"
    json_path = output_path / "regulatory_mapping.json"
    audit_path = output_path / "audit_report.json"

    md_path.write_text(md_content)
    json_path.write_text(json_content)
    audit_path.write_text(json.dumps(audit_report, indent=2))

    return {"markdown": str(md_path), "json": str(json_path), "audit": str(audit_path)}
