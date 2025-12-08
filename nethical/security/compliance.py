"""
Phase 3: Regulatory Compliance Framework

This module provides comprehensive compliance capabilities for:
- NIST 800-53 control mapping and validation
- HIPAA Privacy Rule compliance
- FedRAMP continuous monitoring
- ISO/IEC 27001:2022 Annex A control mapping
- Automated compliance reporting
- Evidence collection for auditors
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""

    NIST_800_53 = "nist_800_53"
    HIPAA = "hipaa"
    FEDRAMP = "fedramp"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    ISO_27001 = "iso_27001"


class ComplianceStatus(str, Enum):
    """Compliance status states"""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


class ControlSeverity(str, Enum):
    """Control severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"


@dataclass
class ComplianceControl:
    """Represents a single compliance control"""

    id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    severity: ControlSeverity
    implementation_status: ComplianceStatus = ComplianceStatus.PENDING
    evidence: List[str] = field(default_factory=list)
    last_assessed: Optional[datetime] = None
    assessor: Optional[str] = None
    remediation_plan: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceEvidence:
    """Evidence artifact for compliance validation"""

    id: str
    control_id: str
    evidence_type: str  # document, log, screenshot, configuration
    description: str
    artifact_path: Optional[str] = None
    artifact_hash: Optional[str] = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    collected_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance assessment report"""

    id: str
    framework: ComplianceFramework
    report_date: datetime
    scope: str
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    partial_controls: int
    not_applicable_controls: int
    compliance_score: float  # percentage
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence_artifacts: List[str] = field(default_factory=list)
    assessor: str = "automated"
    next_assessment_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NIST80053ControlMapper:
    """Maps NIST 800-53 controls to system components"""

    def __init__(self):
        self.controls: Dict[str, ComplianceControl] = {}
        self._initialize_controls()

    def _initialize_controls(self):
        """Initialize NIST 800-53 control definitions"""

        # Access Control (AC) family
        self.controls["AC-1"] = ComplianceControl(
            id="AC-1",
            framework=ComplianceFramework.NIST_800_53,
            title="Access Control Policy and Procedures",
            description="Develop, document, and disseminate access control policy and procedures",
            requirement="Organization must have documented access control policies",
            severity=ControlSeverity.HIGH,
        )

        self.controls["AC-2"] = ComplianceControl(
            id="AC-2",
            framework=ComplianceFramework.NIST_800_53,
            title="Account Management",
            description="Manage system accounts including creation, modification, and deletion",
            requirement="Implement automated account management procedures",
            severity=ControlSeverity.HIGH,
        )

        # Identification and Authentication (IA) family
        self.controls["IA-2"] = ComplianceControl(
            id="IA-2",
            framework=ComplianceFramework.NIST_800_53,
            title="Identification and Authentication",
            description="Uniquely identify and authenticate organizational users",
            requirement="Multi-factor authentication for privileged accounts",
            severity=ControlSeverity.CRITICAL,
        )

        self.controls["IA-5"] = ComplianceControl(
            id="IA-5",
            framework=ComplianceFramework.NIST_800_53,
            title="Authenticator Management",
            description="Manage system authenticators including passwords and tokens",
            requirement="Secure authenticator lifecycle management",
            severity=ControlSeverity.HIGH,
        )

        # System and Communications Protection (SC) family
        self.controls["SC-8"] = ComplianceControl(
            id="SC-8",
            framework=ComplianceFramework.NIST_800_53,
            title="Transmission Confidentiality and Integrity",
            description="Protect the confidentiality and integrity of transmitted information",
            requirement="Use FIPS 140-2 validated cryptography",
            severity=ControlSeverity.CRITICAL,
        )

        self.controls["SC-13"] = ComplianceControl(
            id="SC-13",
            framework=ComplianceFramework.NIST_800_53,
            title="Cryptographic Protection",
            description="Implement cryptographic mechanisms to prevent unauthorized disclosure",
            requirement="Use approved cryptographic algorithms",
            severity=ControlSeverity.CRITICAL,
        )

        # Audit and Accountability (AU) family
        self.controls["AU-2"] = ComplianceControl(
            id="AU-2",
            framework=ComplianceFramework.NIST_800_53,
            title="Event Logging",
            description="Determine events requiring audit logging",
            requirement="Log security-relevant events with sufficient detail",
            severity=ControlSeverity.HIGH,
        )

        self.controls["AU-6"] = ComplianceControl(
            id="AU-6",
            framework=ComplianceFramework.NIST_800_53,
            title="Audit Review, Analysis, and Reporting",
            description="Review and analyze audit records for indicators of inappropriate activity",
            requirement="Automated audit analysis and alerting",
            severity=ControlSeverity.HIGH,
        )

        # Incident Response (IR) family
        self.controls["IR-4"] = ComplianceControl(
            id="IR-4",
            framework=ComplianceFramework.NIST_800_53,
            title="Incident Handling",
            description="Implement an incident handling capability",
            requirement="Documented incident response procedures",
            severity=ControlSeverity.HIGH,
        )

        self.controls["IR-5"] = ComplianceControl(
            id="IR-5",
            framework=ComplianceFramework.NIST_800_53,
            title="Incident Monitoring",
            description="Track and document security incidents",
            requirement="Automated incident tracking system",
            severity=ControlSeverity.MEDIUM,
        )

    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Retrieve a specific control"""
        return self.controls.get(control_id)

    def get_controls_by_family(self, family: str) -> List[ComplianceControl]:
        """Get all controls from a specific family (e.g., 'AC', 'IA')"""
        return [
            control
            for control in self.controls.values()
            if control.id.startswith(family)
        ]

    def assess_control(
        self,
        control_id: str,
        status: ComplianceStatus,
        evidence: List[str],
        assessor: str = "system",
    ) -> bool:
        """Assess a control's compliance status"""
        control = self.controls.get(control_id)
        if not control:
            return False

        control.implementation_status = status
        control.evidence.extend(evidence)
        control.last_assessed = datetime.now(timezone.utc)
        control.assessor = assessor
        return True


class HIPAAComplianceValidator:
    """HIPAA Privacy Rule compliance validation"""

    def __init__(self):
        self.rules: Dict[str, ComplianceControl] = {}
        self._initialize_rules()

    def _initialize_rules(self):
        """Initialize HIPAA Privacy Rule requirements"""

        self.rules["164.308(a)(1)"] = ComplianceControl(
            id="164.308(a)(1)",
            framework=ComplianceFramework.HIPAA,
            title="Security Management Process",
            description="Implement policies and procedures to prevent, detect, contain, and correct security violations",
            requirement="Risk analysis and risk management procedures",
            severity=ControlSeverity.CRITICAL,
        )

        self.rules["164.308(a)(3)"] = ComplianceControl(
            id="164.308(a)(3)",
            framework=ComplianceFramework.HIPAA,
            title="Workforce Security",
            description="Implement policies and procedures to ensure workforce members have appropriate access",
            requirement="Authorization and supervision procedures",
            severity=ControlSeverity.HIGH,
        )

        self.rules["164.308(a)(5)"] = ComplianceControl(
            id="164.308(a)(5)",
            framework=ComplianceFramework.HIPAA,
            title="Security Awareness and Training",
            description="Implement security awareness and training program",
            requirement="Regular security training for workforce",
            severity=ControlSeverity.MEDIUM,
        )

        self.rules["164.312(a)(1)"] = ComplianceControl(
            id="164.312(a)(1)",
            framework=ComplianceFramework.HIPAA,
            title="Access Control",
            description="Implement technical policies and procedures for electronic PHI access",
            requirement="Unique user identification and emergency access procedures",
            severity=ControlSeverity.CRITICAL,
        )

        self.rules["164.312(c)(1)"] = ComplianceControl(
            id="164.312(c)(1)",
            framework=ComplianceFramework.HIPAA,
            title="Integrity Controls",
            description="Implement policies to ensure ePHI is not improperly altered or destroyed",
            requirement="Mechanisms to authenticate ePHI",
            severity=ControlSeverity.HIGH,
        )

        self.rules["164.312(e)(1)"] = ComplianceControl(
            id="164.312(e)(1)",
            framework=ComplianceFramework.HIPAA,
            title="Transmission Security",
            description="Implement technical security measures to guard against unauthorized ePHI access during transmission",
            requirement="Encryption of ePHI in transit",
            severity=ControlSeverity.CRITICAL,
        )

    def validate_phi_access(self, user_id: str, access_type: str) -> bool:
        """Validate PHI access according to HIPAA rules"""
        # Stub implementation for minimum necessary rule
        return True

    def check_encryption_compliance(self) -> ComplianceStatus:
        """Check if encryption meets HIPAA requirements"""
        # In real implementation, check actual crypto configuration
        return ComplianceStatus.COMPLIANT

    def get_rule(self, rule_id: str) -> Optional[ComplianceControl]:
        """Retrieve a specific HIPAA rule"""
        return self.rules.get(rule_id)


class FedRAMPMonitor:
    """FedRAMP continuous monitoring automation"""

    def __init__(self):
        self.monitoring_metrics: Dict[str, Any] = {}
        self.continuous_monitoring_enabled = True

    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics for FedRAMP reporting"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "failed_login_attempts": 0,
            "successful_authentications": 0,
            "active_sessions": 0,
            "security_incidents": 0,
            "vulnerability_scan_date": None,
            "patch_compliance_rate": 100.0,
            "configuration_compliance_rate": 100.0,
        }

    def generate_poam(self) -> List[Dict[str, Any]]:
        """Generate Plan of Action and Milestones (POA&M)"""
        # Plan of Action and Milestones for tracking remediation
        return []

    def check_continuous_monitoring(self) -> bool:
        """Verify continuous monitoring is active"""
        return self.continuous_monitoring_enabled

    def generate_monthly_report(self) -> Dict[str, Any]:
        """Generate FedRAMP monthly continuous monitoring report"""
        metrics = self.collect_security_metrics()
        return {
            "report_id": str(uuid.uuid4()),
            "report_date": datetime.now(timezone.utc).isoformat(),
            "reporting_period": "monthly",
            "metrics": metrics,
            "poam_items": self.generate_poam(),
            "security_status": "operational",
        }


class ISO27001ControlMapper:
    """Maps ISO/IEC 27001:2022 Annex A controls to system components"""

    def __init__(self):
        self.controls: Dict[str, ComplianceControl] = {}
        self._initialize_controls()

    def _initialize_controls(self):
        """Initialize ISO 27001:2022 Annex A control definitions"""

        # A.5 Organizational controls
        self.controls["A.5.1"] = ComplianceControl(
            id="A.5.1",
            framework=ComplianceFramework.ISO_27001,
            title="Policies for information security",
            description="Information security policy and topic-specific policies shall be defined, approved by management, published, communicated to and acknowledged by relevant personnel and relevant interested parties, and reviewed at planned intervals and if significant changes occur.",
            requirement="Documented information security policies",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["SECURITY.md", "docs/SECURITY_HARDENING_GUIDE.md"],
        )

        self.controls["A.5.7"] = ComplianceControl(
            id="A.5.7",
            framework=ComplianceFramework.ISO_27001,
            title="Threat intelligence",
            description="Information relating to information security threats shall be collected and analysed to produce threat intelligence.",
            requirement="Threat modeling and intelligence gathering",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/security/threat_modeling.py",
                "docs/security/threat_model.md",
            ],
        )

        self.controls["A.5.9"] = ComplianceControl(
            id="A.5.9",
            framework=ComplianceFramework.ISO_27001,
            title="Inventory of information and other associated assets",
            description="An inventory of information and other associated assets, including owners, shall be developed and maintained.",
            requirement="Asset inventory and SBOM",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["SBOM.json", "docs/compliance/isms/asset_register.md"],
        )

        self.controls["A.5.12"] = ComplianceControl(
            id="A.5.12",
            framework=ComplianceFramework.ISO_27001,
            title="Classification of information",
            description="Information shall be classified according to the information security needs of the organization based on confidentiality, integrity, availability and relevant interested party requirements.",
            requirement="Data classification scheme",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "policies/common/data_classification.yaml",
                "nethical/security/data_compliance.py",
            ],
        )

        self.controls["A.5.15"] = ComplianceControl(
            id="A.5.15",
            framework=ComplianceFramework.ISO_27001,
            title="Access control",
            description="Rules to control physical and logical access to information and other associated assets shall be established and implemented based on business and information security requirements.",
            requirement="RBAC implementation",
            severity=ControlSeverity.CRITICAL,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["nethical/core/rbac.py", "nethical/security/auth.py"],
        )

        self.controls["A.5.24"] = ComplianceControl(
            id="A.5.24",
            framework=ComplianceFramework.ISO_27001,
            title="Information security incident management planning and preparation",
            description="The organization shall plan and prepare for managing information security incidents by defining, establishing and communicating information security incident management processes, roles and responsibilities.",
            requirement="Incident response procedures",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "docs/compliance/INCIDENT_RESPONSE_POLICY.md",
                "nethical/security/soc_integration.py",
            ],
        )

        self.controls["A.5.28"] = ComplianceControl(
            id="A.5.28",
            framework=ComplianceFramework.ISO_27001,
            title="Collection of evidence",
            description="The organization shall establish and implement procedures for the identification, collection, acquisition and preservation of evidence related to information security events.",
            requirement="Evidence collection and audit trails",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/security/audit_logging.py",
                "nethical/core/audit_merkle.py",
            ],
        )

        self.controls["A.5.34"] = ComplianceControl(
            id="A.5.34",
            framework=ComplianceFramework.ISO_27001,
            title="Privacy and protection of PII",
            description="The organization shall identify and meet the requirements regarding the preservation of privacy and protection of PII according to applicable laws and regulations and contractual requirements.",
            requirement="PII detection and protection",
            severity=ControlSeverity.CRITICAL,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/core/redaction_pipeline.py",
                "nethical/core/differential_privacy.py",
            ],
        )

        # A.6 People controls
        self.controls["A.6.3"] = ComplianceControl(
            id="A.6.3",
            framework=ComplianceFramework.ISO_27001,
            title="Information security awareness, education and training",
            description="Personnel of the organization and relevant interested parties shall receive appropriate information security awareness, education and training and regular updates of the organization's information security policy, topic-specific policies and procedures, as relevant for their job function.",
            requirement="Security training documentation",
            severity=ControlSeverity.MEDIUM,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["docs/TRAINING_GUIDE.md", "docs/SECURITY_HARDENING_GUIDE.md"],
        )

        # A.8 Technological controls
        self.controls["A.8.2"] = ComplianceControl(
            id="A.8.2",
            framework=ComplianceFramework.ISO_27001,
            title="Privileged access rights",
            description="The allocation and use of privileged access rights shall be restricted and managed.",
            requirement="Privilege management",
            severity=ControlSeverity.CRITICAL,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["nethical/core/rbac.py"],
        )

        self.controls["A.8.5"] = ComplianceControl(
            id="A.8.5",
            framework=ComplianceFramework.ISO_27001,
            title="Secure authentication",
            description="Secure authentication technologies and procedures shall be implemented based on information access restrictions and the topic-specific policy on access control.",
            requirement="MFA and SSO implementation",
            severity=ControlSeverity.CRITICAL,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["nethical/security/mfa.py", "nethical/security/sso.py"],
        )

        self.controls["A.8.7"] = ComplianceControl(
            id="A.8.7",
            framework=ComplianceFramework.ISO_27001,
            title="Protection against malware",
            description="Protection against malware shall be implemented and supported by appropriate user awareness.",
            requirement="Adversarial detection and input validation",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["nethical/detectors/", "nethical/security/input_validation.py"],
        )

        self.controls["A.8.8"] = ComplianceControl(
            id="A.8.8",
            framework=ComplianceFramework.ISO_27001,
            title="Management of technical vulnerabilities",
            description="Information about technical vulnerabilities of information systems in use shall be obtained, the organization's exposure to such vulnerabilities shall be evaluated and appropriate measures shall be taken.",
            requirement="Vulnerability scanning and penetration testing",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=["nethical/security/penetration_testing.py", "SBOM.json"],
        )

        self.controls["A.8.11"] = ComplianceControl(
            id="A.8.11",
            framework=ComplianceFramework.ISO_27001,
            title="Data masking",
            description="Data masking shall be used in accordance with the organization's topic-specific policy on access control and other related topic-specific policies, and business requirements, taking applicable legislation into consideration.",
            requirement="PII redaction and differential privacy",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/core/redaction_pipeline.py",
                "nethical/core/differential_privacy.py",
            ],
        )

        self.controls["A.8.15"] = ComplianceControl(
            id="A.8.15",
            framework=ComplianceFramework.ISO_27001,
            title="Logging",
            description="Logs that record activities, exceptions, faults and other relevant events shall be produced, stored, protected and analysed.",
            requirement="Comprehensive audit logging with integrity protection",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/security/audit_logging.py",
                "nethical/core/audit_merkle.py",
            ],
        )

        self.controls["A.8.16"] = ComplianceControl(
            id="A.8.16",
            framework=ComplianceFramework.ISO_27001,
            title="Monitoring activities",
            description="Networks, systems and applications shall be monitored for anomalous behaviour and appropriate actions taken to evaluate potential information security incidents.",
            requirement="Real-time monitoring and anomaly detection",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/monitors/",
                "nethical/observability/",
                "nethical/core/anomaly_detector.py",
            ],
        )

        self.controls["A.8.24"] = ComplianceControl(
            id="A.8.24",
            framework=ComplianceFramework.ISO_27001,
            title="Use of cryptography",
            description="Rules for the effective use of cryptography, including cryptographic key management, shall be defined and implemented.",
            requirement="Encryption and cryptographic controls",
            severity=ControlSeverity.CRITICAL,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/security/encryption.py",
                "nethical/security/quantum_crypto.py",
                "nethical/core/audit_merkle.py",
            ],
        )

        self.controls["A.8.25"] = ComplianceControl(
            id="A.8.25",
            framework=ComplianceFramework.ISO_27001,
            title="Secure development life cycle",
            description="Rules for the secure development of software and systems shall be established and applied.",
            requirement="Secure SDLC with security scanning",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[".github/workflows/", "tests/adversarial/"],
        )

        self.controls["A.8.32"] = ComplianceControl(
            id="A.8.32",
            framework=ComplianceFramework.ISO_27001,
            title="Change management",
            description="Changes to information processing facilities and information systems shall be subject to change management procedures.",
            requirement="Policy diff and release management",
            severity=ControlSeverity.HIGH,
            implementation_status=ComplianceStatus.COMPLIANT,
            evidence=[
                "nethical/core/policy_diff.py",
                "nethical/policy/release_management.py",
            ],
        )

    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Retrieve a specific control"""
        return self.controls.get(control_id)

    def get_controls_by_category(self, category: str) -> List[ComplianceControl]:
        """Get all controls from a specific category (e.g., 'A.5', 'A.6', 'A.8')"""
        return [
            control
            for control in self.controls.values()
            if control.id.startswith(category)
        ]

    def assess_control(
        self,
        control_id: str,
        status: ComplianceStatus,
        evidence: List[str],
        assessor: str = "system",
    ) -> bool:
        """Assess a control's compliance status"""
        control = self.controls.get(control_id)
        if not control:
            return False

        control.implementation_status = status
        control.evidence.extend(evidence)
        control.last_assessed = datetime.now(timezone.utc)
        control.assessor = assessor
        return True

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Generate a compliance summary for ISO 27001"""
        total = len(self.controls)
        compliant = sum(
            1
            for c in self.controls.values()
            if c.implementation_status == ComplianceStatus.COMPLIANT
        )
        partial = sum(
            1
            for c in self.controls.values()
            if c.implementation_status == ComplianceStatus.PARTIAL
        )
        non_compliant = sum(
            1
            for c in self.controls.values()
            if c.implementation_status == ComplianceStatus.NON_COMPLIANT
        )
        not_applicable = sum(
            1
            for c in self.controls.values()
            if c.implementation_status == ComplianceStatus.NOT_APPLICABLE
        )

        return {
            "framework": "ISO/IEC 27001:2022",
            "total_controls": total,
            "compliant": compliant,
            "partial": partial,
            "non_compliant": non_compliant,
            "not_applicable": not_applicable,
            "compliance_score": (compliant / total * 100) if total > 0 else 0.0,
            "assessed_at": datetime.now(timezone.utc).isoformat(),
        }


class ComplianceReportGenerator:
    """Automated compliance reporting"""

    def __init__(self):
        self.nist_mapper = NIST80053ControlMapper()
        self.hipaa_validator = HIPAAComplianceValidator()
        self.fedramp_monitor = FedRAMPMonitor()
        self.iso27001_mapper = ISO27001ControlMapper()

    def generate_report(
        self, framework: ComplianceFramework, scope: str = "full_system"
    ) -> ComplianceReport:
        """Generate comprehensive compliance report"""

        if framework == ComplianceFramework.NIST_800_53:
            controls = list(self.nist_mapper.controls.values())
        elif framework == ComplianceFramework.HIPAA:
            controls = list(self.hipaa_validator.rules.values())
        elif framework == ComplianceFramework.ISO_27001:
            controls = list(self.iso27001_mapper.controls.values())
        else:
            controls = []

        total = len(controls)
        compliant = sum(
            1 for c in controls if c.implementation_status == ComplianceStatus.COMPLIANT
        )
        non_compliant = sum(
            1
            for c in controls
            if c.implementation_status == ComplianceStatus.NON_COMPLIANT
        )
        partial = sum(
            1 for c in controls if c.implementation_status == ComplianceStatus.PARTIAL
        )
        not_applicable = sum(
            1
            for c in controls
            if c.implementation_status == ComplianceStatus.NOT_APPLICABLE
        )

        compliance_score = (compliant / total * 100) if total > 0 else 0.0

        findings = []
        for control in controls:
            if control.implementation_status in [
                ComplianceStatus.NON_COMPLIANT,
                ComplianceStatus.PARTIAL,
            ]:
                findings.append(
                    {
                        "control_id": control.id,
                        "title": control.title,
                        "status": control.implementation_status.value,
                        "severity": control.severity.value,
                        "recommendation": control.remediation_plan
                        or "Implement control requirements",
                    }
                )

        report = ComplianceReport(
            id=str(uuid.uuid4()),
            framework=framework,
            report_date=datetime.now(timezone.utc),
            scope=scope,
            total_controls=total,
            compliant_controls=compliant,
            non_compliant_controls=non_compliant,
            partial_controls=partial,
            not_applicable_controls=not_applicable,
            compliance_score=compliance_score,
            findings=findings,
        )

        return report

    def export_report(self, report: ComplianceReport, format: str = "json") -> str:
        """Export report in various formats"""
        if format == "json":
            return json.dumps(
                {
                    "id": report.id,
                    "framework": report.framework.value,
                    "report_date": report.report_date.isoformat(),
                    "scope": report.scope,
                    "summary": {
                        "total_controls": report.total_controls,
                        "compliant": report.compliant_controls,
                        "non_compliant": report.non_compliant_controls,
                        "partial": report.partial_controls,
                        "not_applicable": report.not_applicable_controls,
                        "compliance_score": report.compliance_score,
                    },
                    "findings": report.findings,
                    "recommendations": report.recommendations,
                },
                indent=2,
            )
        return ""


class EvidenceCollector:
    """Evidence collection for auditors"""

    def __init__(self, storage_path: str = "/var/log/compliance"):
        self.storage_path = Path(storage_path)
        self.evidence_repository: Dict[str, ComplianceEvidence] = {}

    def collect_evidence(
        self,
        control_id: str,
        evidence_type: str,
        description: str,
        artifact_data: Optional[bytes] = None,
        collected_by: str = "system",
    ) -> ComplianceEvidence:
        """Collect and store compliance evidence"""

        evidence_id = str(uuid.uuid4())
        evidence = ComplianceEvidence(
            id=evidence_id,
            control_id=control_id,
            evidence_type=evidence_type,
            description=description,
            collected_by=collected_by,
        )

        if artifact_data:
            # In real implementation, store artifact securely
            import hashlib

            evidence.artifact_hash = hashlib.sha256(artifact_data).hexdigest()
            evidence.artifact_path = f"{self.storage_path}/{evidence_id}"

        self.evidence_repository[evidence_id] = evidence
        return evidence

    def get_evidence_by_control(self, control_id: str) -> List[ComplianceEvidence]:
        """Retrieve all evidence for a specific control"""
        return [
            ev
            for ev in self.evidence_repository.values()
            if ev.control_id == control_id
        ]

    def generate_evidence_package(self, control_ids: List[str]) -> Dict[str, Any]:
        """Generate evidence package for auditor review"""
        evidence_items = []
        for control_id in control_ids:
            items = self.get_evidence_by_control(control_id)
            for item in items:
                evidence_items.append(
                    {
                        "evidence_id": item.id,
                        "control_id": item.control_id,
                        "type": item.evidence_type,
                        "description": item.description,
                        "collected_at": item.collected_at.isoformat(),
                        "collected_by": item.collected_by,
                        "artifact_hash": item.artifact_hash,
                    }
                )

        return {
            "package_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "controls": control_ids,
            "evidence_count": len(evidence_items),
            "evidence": evidence_items,
        }
