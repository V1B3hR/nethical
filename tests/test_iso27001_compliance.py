"""
Tests for ISO/IEC 27001:2022 Annex A control mapping.

This module tests the ISO27001ControlMapper class and related
compliance functionality for ISO 27001 certification support.
"""

import pytest
from datetime import datetime, timezone


class TestISO27001ControlMapper:
    """Tests for ISO 27001 Annex A control mapping."""

    def test_mapper_initialization(self):
        """Test that ISO27001ControlMapper initializes with controls."""
        from nethical.security.compliance import ISO27001ControlMapper

        mapper = ISO27001ControlMapper()
        assert len(mapper.controls) > 0
        assert "A.5.1" in mapper.controls
        assert "A.8.24" in mapper.controls

    def test_get_control(self):
        """Test retrieving a specific control."""
        from nethical.security.compliance import (
            ISO27001ControlMapper,
            ComplianceFramework,
        )

        mapper = ISO27001ControlMapper()
        control = mapper.get_control("A.5.1")

        assert control is not None
        assert control.id == "A.5.1"
        assert control.title == "Policies for information security"
        assert control.framework == ComplianceFramework.ISO_27001

    def test_get_control_not_found(self):
        """Test retrieving a non-existent control."""
        from nethical.security.compliance import ISO27001ControlMapper

        mapper = ISO27001ControlMapper()
        control = mapper.get_control("A.99.99")

        assert control is None

    def test_get_controls_by_category(self):
        """Test retrieving controls by category."""
        from nethical.security.compliance import ISO27001ControlMapper

        mapper = ISO27001ControlMapper()

        # Get organizational controls (A.5.x)
        org_controls = mapper.get_controls_by_category("A.5")
        assert len(org_controls) > 0
        for control in org_controls:
            assert control.id.startswith("A.5")

        # Get technological controls (A.8.x)
        tech_controls = mapper.get_controls_by_category("A.8")
        assert len(tech_controls) > 0
        for control in tech_controls:
            assert control.id.startswith("A.8")

    def test_assess_control(self):
        """Test assessing a control's compliance status."""
        from nethical.security.compliance import (
            ISO27001ControlMapper,
            ComplianceStatus,
        )

        mapper = ISO27001ControlMapper()

        result = mapper.assess_control(
            control_id="A.5.1",
            status=ComplianceStatus.COMPLIANT,
            evidence=["audit_report_2025.pdf"],
            assessor="security_team",
        )

        assert result is True
        control = mapper.get_control("A.5.1")
        assert control.implementation_status == ComplianceStatus.COMPLIANT
        assert "audit_report_2025.pdf" in control.evidence
        assert control.assessor == "security_team"
        assert control.last_assessed is not None

    def test_assess_control_not_found(self):
        """Test assessing a non-existent control."""
        from nethical.security.compliance import (
            ISO27001ControlMapper,
            ComplianceStatus,
        )

        mapper = ISO27001ControlMapper()

        result = mapper.assess_control(
            control_id="A.99.99",
            status=ComplianceStatus.COMPLIANT,
            evidence=[],
        )

        assert result is False

    def test_get_compliance_summary(self):
        """Test generating compliance summary."""
        from nethical.security.compliance import ISO27001ControlMapper

        mapper = ISO27001ControlMapper()
        summary = mapper.get_compliance_summary()

        assert summary["framework"] == "ISO/IEC 27001:2022"
        assert summary["total_controls"] > 0
        assert "compliant" in summary
        assert "partial" in summary
        assert "non_compliant" in summary
        assert "compliance_score" in summary
        assert 0 <= summary["compliance_score"] <= 100

    def test_critical_controls_present(self):
        """Test that critical ISO 27001 controls are mapped."""
        from nethical.security.compliance import (
            ISO27001ControlMapper,
            ControlSeverity,
        )

        mapper = ISO27001ControlMapper()

        # Check access control (A.5.15)
        access_control = mapper.get_control("A.5.15")
        assert access_control is not None
        assert access_control.severity == ControlSeverity.CRITICAL

        # Check cryptography (A.8.24)
        crypto_control = mapper.get_control("A.8.24")
        assert crypto_control is not None
        assert crypto_control.severity == ControlSeverity.CRITICAL

        # Check secure authentication (A.8.5)
        auth_control = mapper.get_control("A.8.5")
        assert auth_control is not None
        assert auth_control.severity == ControlSeverity.CRITICAL


class TestISO27001ReportGeneration:
    """Tests for ISO 27001 compliance report generation."""

    def test_generate_iso27001_report(self):
        """Test generating an ISO 27001 compliance report."""
        from nethical.security.compliance import (
            ComplianceReportGenerator,
            ComplianceFramework,
        )

        generator = ComplianceReportGenerator()
        report = generator.generate_report(ComplianceFramework.ISO_27001)

        assert report is not None
        assert report.framework == ComplianceFramework.ISO_27001
        assert report.total_controls > 0
        assert 0 <= report.compliance_score <= 100

    def test_export_iso27001_report(self):
        """Test exporting ISO 27001 report to JSON."""
        import json
        from nethical.security.compliance import (
            ComplianceReportGenerator,
            ComplianceFramework,
        )

        generator = ComplianceReportGenerator()
        report = generator.generate_report(ComplianceFramework.ISO_27001)
        json_output = generator.export_report(report, "json")

        assert json_output != ""
        parsed = json.loads(json_output)
        assert parsed["framework"] == "iso_27001"
        assert "summary" in parsed
        assert parsed["summary"]["total_controls"] > 0


class TestEvidenceCollection:
    """Tests for evidence collection for ISO 27001 audits."""

    def test_collect_evidence(self):
        """Test collecting evidence for a control."""
        from nethical.security.compliance import EvidenceCollector

        collector = EvidenceCollector(storage_path="/tmp/test_compliance")
        evidence = collector.collect_evidence(
            control_id="A.8.15",
            evidence_type="log",
            description="Audit log sample for logging control",
            collected_by="test_user",
        )

        assert evidence is not None
        assert evidence.control_id == "A.8.15"
        assert evidence.evidence_type == "log"
        assert evidence.collected_by == "test_user"

    def test_get_evidence_by_control(self):
        """Test retrieving evidence by control ID."""
        from nethical.security.compliance import EvidenceCollector

        collector = EvidenceCollector(storage_path="/tmp/test_compliance")

        # Collect multiple evidence items
        collector.collect_evidence(
            control_id="A.5.28",
            evidence_type="document",
            description="Evidence collection procedure",
        )
        collector.collect_evidence(
            control_id="A.5.28",
            evidence_type="screenshot",
            description="Audit trail screenshot",
        )
        collector.collect_evidence(
            control_id="A.8.15",
            evidence_type="log",
            description="Sample audit log",
        )

        # Retrieve evidence for specific control
        evidence_list = collector.get_evidence_by_control("A.5.28")
        assert len(evidence_list) == 2
        for ev in evidence_list:
            assert ev.control_id == "A.5.28"

    def test_generate_evidence_package(self):
        """Test generating evidence package for auditor review."""
        from nethical.security.compliance import EvidenceCollector

        collector = EvidenceCollector(storage_path="/tmp/test_compliance")

        # Collect evidence
        collector.collect_evidence(
            control_id="A.8.24",
            evidence_type="configuration",
            description="Encryption configuration",
        )
        collector.collect_evidence(
            control_id="A.8.15",
            evidence_type="log",
            description="Audit log sample",
        )

        # Generate package
        package = collector.generate_evidence_package(["A.8.24", "A.8.15"])

        assert package is not None
        assert "package_id" in package
        assert "generated_at" in package
        assert package["evidence_count"] == 2
        assert len(package["evidence"]) == 2


class TestISO27001Integration:
    """Integration tests for ISO 27001 compliance features."""

    def test_full_compliance_workflow(self):
        """Test complete compliance workflow from assessment to report."""
        from nethical.security.compliance import (
            ISO27001ControlMapper,
            ComplianceReportGenerator,
            EvidenceCollector,
            ComplianceStatus,
            ComplianceFramework,
        )

        # Step 1: Initialize mapper and assess controls
        mapper = ISO27001ControlMapper()

        # Assess a few controls
        mapper.assess_control(
            "A.5.1",
            ComplianceStatus.COMPLIANT,
            ["security_policy_v1.pdf"],
            "auditor",
        )
        mapper.assess_control(
            "A.8.24",
            ComplianceStatus.COMPLIANT,
            ["encryption_config.yaml"],
            "auditor",
        )

        # Step 2: Collect evidence
        collector = EvidenceCollector(storage_path="/tmp/test_compliance")
        collector.collect_evidence(
            control_id="A.5.1",
            evidence_type="document",
            description="Information Security Policy",
        )
        collector.collect_evidence(
            control_id="A.8.24",
            evidence_type="configuration",
            description="Encryption configuration",
        )

        # Step 3: Generate report
        generator = ComplianceReportGenerator()
        report = generator.generate_report(ComplianceFramework.ISO_27001)

        # Step 4: Verify workflow completed successfully
        assert report.total_controls > 0
        summary = mapper.get_compliance_summary()
        assert summary["total_controls"] > 0

        # Step 5: Generate evidence package
        package = collector.generate_evidence_package(["A.5.1", "A.8.24"])
        assert package["evidence_count"] == 2
