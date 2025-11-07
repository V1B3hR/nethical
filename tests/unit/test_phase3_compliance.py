"""
Unit tests for Phase 3: Compliance & Audit Framework
"""

import pytest
from datetime import datetime, timedelta, timezone
from nethical.security.compliance import (
    ComplianceFramework,
    ComplianceStatus,
    ControlSeverity,
    ComplianceControl,
    ComplianceEvidence,
    ComplianceReport,
    NIST80053ControlMapper,
    HIPAAComplianceValidator,
    FedRAMPMonitor,
    ComplianceReportGenerator,
    EvidenceCollector,
)


class TestComplianceControl:
    """Test compliance control model"""
    
    def test_control_creation(self):
        """Test creating a compliance control"""
        control = ComplianceControl(
            id="AC-1",
            framework=ComplianceFramework.NIST_800_53,
            title="Access Control Policy",
            description="Test description",
            requirement="Test requirement",
            severity=ControlSeverity.HIGH
        )
        
        assert control.id == "AC-1"
        assert control.framework == ComplianceFramework.NIST_800_53
        assert control.implementation_status == ComplianceStatus.PENDING
        assert len(control.evidence) == 0
    
    def test_control_with_evidence(self):
        """Test control with evidence"""
        control = ComplianceControl(
            id="AC-2",
            framework=ComplianceFramework.NIST_800_53,
            title="Account Management",
            description="Test",
            requirement="Test",
            severity=ControlSeverity.HIGH,
            evidence=["evidence1", "evidence2"]
        )
        
        assert len(control.evidence) == 2
        assert "evidence1" in control.evidence


class TestNIST80053ControlMapper:
    """Test NIST 800-53 control mapper"""
    
    def test_initialization(self):
        """Test mapper initialization"""
        mapper = NIST80053ControlMapper()
        assert len(mapper.controls) > 0
        assert "AC-1" in mapper.controls
        assert "IA-2" in mapper.controls
    
    def test_get_control(self):
        """Test retrieving specific control"""
        mapper = NIST80053ControlMapper()
        control = mapper.get_control("AC-1")
        
        assert control is not None
        assert control.id == "AC-1"
        assert control.title == "Access Control Policy and Procedures"
        assert control.framework == ComplianceFramework.NIST_800_53
    
    def test_get_nonexistent_control(self):
        """Test getting non-existent control"""
        mapper = NIST80053ControlMapper()
        control = mapper.get_control("INVALID-1")
        assert control is None
    
    def test_get_controls_by_family(self):
        """Test getting controls by family"""
        mapper = NIST80053ControlMapper()
        ac_controls = mapper.get_controls_by_family("AC")
        
        assert len(ac_controls) > 0
        assert all(c.id.startswith("AC") for c in ac_controls)
    
    def test_assess_control(self):
        """Test assessing a control"""
        mapper = NIST80053ControlMapper()
        
        success = mapper.assess_control(
            "AC-1",
            ComplianceStatus.COMPLIANT,
            ["Policy document reviewed", "Procedures documented"],
            "auditor@example.com"
        )
        
        assert success is True
        control = mapper.get_control("AC-1")
        assert control.implementation_status == ComplianceStatus.COMPLIANT
        assert len(control.evidence) == 2
        assert control.last_assessed is not None
        assert control.assessor == "auditor@example.com"
    
    def test_assess_invalid_control(self):
        """Test assessing invalid control"""
        mapper = NIST80053ControlMapper()
        success = mapper.assess_control(
            "INVALID-1",
            ComplianceStatus.COMPLIANT,
            ["test"]
        )
        assert success is False
    
    def test_control_severity_levels(self):
        """Test control severity levels"""
        mapper = NIST80053ControlMapper()
        
        critical_control = mapper.get_control("IA-2")
        assert critical_control.severity == ControlSeverity.CRITICAL
        
        high_control = mapper.get_control("AC-2")
        assert high_control.severity == ControlSeverity.HIGH


class TestHIPAAComplianceValidator:
    """Test HIPAA compliance validator"""
    
    def test_initialization(self):
        """Test validator initialization"""
        validator = HIPAAComplianceValidator()
        assert len(validator.rules) > 0
        assert "164.308(a)(1)" in validator.rules
    
    def test_get_rule(self):
        """Test retrieving HIPAA rule"""
        validator = HIPAAComplianceValidator()
        rule = validator.get_rule("164.308(a)(1)")
        
        assert rule is not None
        assert rule.id == "164.308(a)(1)"
        assert rule.framework == ComplianceFramework.HIPAA
        assert "Security Management Process" in rule.title
    
    def test_validate_phi_access(self):
        """Test PHI access validation"""
        validator = HIPAAComplianceValidator()
        result = validator.validate_phi_access("user123", "read")
        assert result is True
    
    def test_check_encryption_compliance(self):
        """Test encryption compliance check"""
        validator = HIPAAComplianceValidator()
        status = validator.check_encryption_compliance()
        assert status == ComplianceStatus.COMPLIANT
    
    def test_critical_rules(self):
        """Test critical HIPAA rules are present"""
        validator = HIPAAComplianceValidator()
        
        # Access Control
        access_control = validator.get_rule("164.312(a)(1)")
        assert access_control.severity == ControlSeverity.CRITICAL
        
        # Transmission Security
        transmission = validator.get_rule("164.312(e)(1)")
        assert transmission.severity == ControlSeverity.CRITICAL


class TestFedRAMPMonitor:
    """Test FedRAMP monitoring"""
    
    def test_initialization(self):
        """Test monitor initialization"""
        monitor = FedRAMPMonitor()
        assert monitor.continuous_monitoring_enabled is True
    
    def test_collect_security_metrics(self):
        """Test security metrics collection"""
        monitor = FedRAMPMonitor()
        metrics = monitor.collect_security_metrics()
        
        assert "timestamp" in metrics
        assert "failed_login_attempts" in metrics
        assert "security_incidents" in metrics
        assert "patch_compliance_rate" in metrics
        assert metrics["patch_compliance_rate"] == 100.0
    
    def test_generate_poam(self):
        """Test POA&M generation"""
        monitor = FedRAMPMonitor()
        poam = monitor.generate_poam()
        assert isinstance(poam, list)
    
    def test_check_continuous_monitoring(self):
        """Test continuous monitoring check"""
        monitor = FedRAMPMonitor()
        status = monitor.check_continuous_monitoring()
        assert status is True
    
    def test_generate_monthly_report(self):
        """Test monthly report generation"""
        monitor = FedRAMPMonitor()
        report = monitor.generate_monthly_report()
        
        assert "report_id" in report
        assert "report_date" in report
        assert "reporting_period" in report
        assert report["reporting_period"] == "monthly"
        assert "metrics" in report
        assert "security_status" in report


class TestComplianceReportGenerator:
    """Test compliance report generator"""
    
    def test_initialization(self):
        """Test generator initialization"""
        generator = ComplianceReportGenerator()
        assert generator.nist_mapper is not None
        assert generator.hipaa_validator is not None
        assert generator.fedramp_monitor is not None
    
    def test_generate_nist_report(self):
        """Test NIST 800-53 report generation"""
        generator = ComplianceReportGenerator()
        report = generator.generate_report(ComplianceFramework.NIST_800_53)
        
        assert report.framework == ComplianceFramework.NIST_800_53
        assert report.total_controls > 0
        assert report.compliance_score >= 0.0
        assert report.compliance_score <= 100.0
        assert report.report_date is not None
    
    def test_generate_hipaa_report(self):
        """Test HIPAA report generation"""
        generator = ComplianceReportGenerator()
        report = generator.generate_report(ComplianceFramework.HIPAA)
        
        assert report.framework == ComplianceFramework.HIPAA
        assert report.total_controls > 0
        assert isinstance(report.findings, list)
    
    def test_report_with_assessed_controls(self):
        """Test report with assessed controls"""
        generator = ComplianceReportGenerator()
        
        # Assess some controls
        generator.nist_mapper.assess_control(
            "AC-1",
            ComplianceStatus.COMPLIANT,
            ["test evidence"]
        )
        generator.nist_mapper.assess_control(
            "AC-2",
            ComplianceStatus.NON_COMPLIANT,
            ["missing documentation"]
        )
        
        report = generator.generate_report(ComplianceFramework.NIST_800_53)
        
        assert report.compliant_controls >= 1
        assert report.non_compliant_controls >= 1
        assert len(report.findings) > 0
    
    def test_export_report_json(self):
        """Test exporting report as JSON"""
        generator = ComplianceReportGenerator()
        report = generator.generate_report(ComplianceFramework.NIST_800_53)
        
        json_output = generator.export_report(report, format="json")
        
        assert json_output is not None
        assert len(json_output) > 0
        assert "framework" in json_output
        assert "compliance_score" in json_output
    
    def test_compliance_score_calculation(self):
        """Test compliance score calculation"""
        generator = ComplianceReportGenerator()
        
        # Mark all controls as compliant
        for control_id in generator.nist_mapper.controls.keys():
            generator.nist_mapper.assess_control(
                control_id,
                ComplianceStatus.COMPLIANT,
                ["tested"]
            )
        
        report = generator.generate_report(ComplianceFramework.NIST_800_53)
        assert report.compliance_score == 100.0


class TestEvidenceCollector:
    """Test evidence collector"""
    
    def test_initialization(self):
        """Test collector initialization"""
        collector = EvidenceCollector("/tmp/compliance")
        assert collector.storage_path.name == "compliance"
    
    def test_collect_evidence(self):
        """Test evidence collection"""
        collector = EvidenceCollector()
        
        evidence = collector.collect_evidence(
            control_id="AC-1",
            evidence_type="document",
            description="Access control policy document",
            collected_by="auditor@example.com"
        )
        
        assert evidence.id is not None
        assert evidence.control_id == "AC-1"
        assert evidence.evidence_type == "document"
        assert evidence.collected_by == "auditor@example.com"
        assert evidence.collected_at is not None
    
    def test_collect_evidence_with_artifact(self):
        """Test evidence collection with artifact"""
        collector = EvidenceCollector()
        
        artifact_data = b"This is a test artifact"
        evidence = collector.collect_evidence(
            control_id="AC-2",
            evidence_type="configuration",
            description="System configuration",
            artifact_data=artifact_data
        )
        
        assert evidence.artifact_hash is not None
        assert evidence.artifact_path is not None
        assert len(evidence.artifact_hash) == 64  # SHA-256 hash
    
    def test_get_evidence_by_control(self):
        """Test retrieving evidence by control"""
        collector = EvidenceCollector()
        
        collector.collect_evidence("AC-1", "document", "Test 1")
        collector.collect_evidence("AC-1", "log", "Test 2")
        collector.collect_evidence("AC-2", "document", "Test 3")
        
        ac1_evidence = collector.get_evidence_by_control("AC-1")
        assert len(ac1_evidence) == 2
        assert all(e.control_id == "AC-1" for e in ac1_evidence)
    
    def test_generate_evidence_package(self):
        """Test evidence package generation"""
        collector = EvidenceCollector()
        
        collector.collect_evidence("AC-1", "document", "Policy")
        collector.collect_evidence("AC-2", "log", "Audit log")
        collector.collect_evidence("IA-2", "configuration", "MFA config")
        
        package = collector.generate_evidence_package(["AC-1", "AC-2"])
        
        assert "package_id" in package
        assert "generated_at" in package
        assert "controls" in package
        assert package["evidence_count"] == 2
        assert len(package["evidence"]) == 2


class TestComplianceEvidence:
    """Test compliance evidence model"""
    
    def test_evidence_creation(self):
        """Test creating evidence"""
        evidence = ComplianceEvidence(
            id="ev-123",
            control_id="AC-1",
            evidence_type="document",
            description="Test evidence"
        )
        
        assert evidence.id == "ev-123"
        assert evidence.control_id == "AC-1"
        assert evidence.collected_by == "system"
        assert evidence.collected_at is not None


class TestComplianceReport:
    """Test compliance report model"""
    
    def test_report_creation(self):
        """Test creating compliance report"""
        report = ComplianceReport(
            id="report-123",
            framework=ComplianceFramework.NIST_800_53,
            report_date=datetime.now(timezone.utc),
            scope="full_system",
            total_controls=100,
            compliant_controls=85,
            non_compliant_controls=10,
            partial_controls=5,
            not_applicable_controls=0,
            compliance_score=85.0
        )
        
        assert report.id == "report-123"
        assert report.framework == ComplianceFramework.NIST_800_53
        assert report.compliance_score == 85.0
        assert report.total_controls == 100


class TestComplianceIntegration:
    """Integration tests for compliance framework"""
    
    def test_end_to_end_nist_compliance(self):
        """Test end-to-end NIST compliance workflow"""
        # Initialize components
        generator = ComplianceReportGenerator()
        collector = EvidenceCollector()
        
        # Collect evidence
        evidence1 = collector.collect_evidence(
            "AC-1",
            "document",
            "Access control policy"
        )
        evidence2 = collector.collect_evidence(
            "IA-2",
            "configuration",
            "MFA configuration"
        )
        
        # Assess controls
        generator.nist_mapper.assess_control(
            "AC-1",
            ComplianceStatus.COMPLIANT,
            [evidence1.id]
        )
        generator.nist_mapper.assess_control(
            "IA-2",
            ComplianceStatus.COMPLIANT,
            [evidence2.id]
        )
        
        # Generate report
        report = generator.generate_report(ComplianceFramework.NIST_800_53)
        
        assert report.compliant_controls >= 2
        assert report.total_controls > 0
    
    def test_multi_framework_compliance(self):
        """Test compliance across multiple frameworks"""
        generator = ComplianceReportGenerator()
        
        nist_report = generator.generate_report(ComplianceFramework.NIST_800_53)
        hipaa_report = generator.generate_report(ComplianceFramework.HIPAA)
        
        assert nist_report.framework == ComplianceFramework.NIST_800_53
        assert hipaa_report.framework == ComplianceFramework.HIPAA
        assert nist_report.total_controls != hipaa_report.total_controls
