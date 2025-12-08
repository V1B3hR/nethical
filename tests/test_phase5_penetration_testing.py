"""
Tests for Phase 5.2: Penetration Testing Program Framework

This test suite validates penetration testing capabilities including vulnerability scanning,
test report generation, remediation tracking, red team engagement, purple team collaboration,
and bug bounty program integration.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

from nethical.security.penetration_testing import (
    VulnerabilitySeverity,
    VulnerabilityStatus,
    TestType,
    TestStatus,
    Vulnerability,
    PenetrationTest,
    RedTeamEngagement,
    PurpleTeamExercise,
    VulnerabilityScanner,
    PenetrationTestManager,
    RedTeamManager,
    PurpleTeamManager,
    BugBountyProgram,
    PenetrationTestingFramework,
)


class TestVulnerabilityDataClass:
    """Test vulnerability data class."""

    def test_vulnerability_creation(self):
        """Test vulnerability object creation."""
        vuln = Vulnerability(
            id="VULN001",
            title="SQL Injection",
            description="SQL injection in login form",
            severity=VulnerabilitySeverity.CRITICAL,
            status=VulnerabilityStatus.DISCOVERED,
            cvss_score=9.8,
            cwe_id="CWE-89",
            affected_components=["web_app", "database"],
            attack_vector="Network",
            discovered_by="security_team",
        )

        assert vuln.id == "VULN001"
        assert vuln.severity == VulnerabilitySeverity.CRITICAL
        assert vuln.cvss_score == 9.8
        assert "web_app" in vuln.affected_components

    def test_vulnerability_to_dict(self):
        """Test vulnerability serialization."""
        vuln = Vulnerability(
            id="VULN002",
            title="XSS Vulnerability",
            description="Cross-site scripting in search",
            severity=VulnerabilitySeverity.HIGH,
            status=VulnerabilityStatus.FIXED,
            cvss_score=7.5,
            affected_components=["web_app"],
            attack_vector="Network",
            discovered_by="pentester",
        )

        vuln_dict = vuln.to_dict()

        assert vuln_dict["id"] == "VULN002"
        assert vuln_dict["severity"] == "high"
        assert vuln_dict["status"] == "fixed"

    def test_sla_compliance_within_deadline(self):
        """Test SLA compliance for fixed vulnerability within deadline."""
        vuln = Vulnerability(
            id="VULN003",
            title="Test Vuln",
            description="Test",
            severity=VulnerabilitySeverity.HIGH,
            status=VulnerabilityStatus.FIXED,
            cvss_score=7.0,
            affected_components=["test"],
            attack_vector="Test",
            discovered_by="test",
        )

        vuln.target_fix_date = datetime.now() + timedelta(days=7)
        vuln.fixed_at = datetime.now()

        assert vuln.calculate_sla_compliance() is True

    def test_sla_compliance_past_deadline(self):
        """Test SLA compliance for vulnerability past deadline."""
        vuln = Vulnerability(
            id="VULN004",
            title="Test Vuln",
            description="Test",
            severity=VulnerabilitySeverity.CRITICAL,
            status=VulnerabilityStatus.IN_PROGRESS,
            cvss_score=9.0,
            affected_components=["test"],
            attack_vector="Test",
            discovered_by="test",
        )

        vuln.target_fix_date = datetime.now() - timedelta(days=1)

        assert vuln.calculate_sla_compliance() is False


class TestVulnerabilityScanner:
    """Test vulnerability scanner."""

    def test_initialization(self):
        """Test scanner initialization."""
        scanner = VulnerabilityScanner()

        assert len(scanner.vulnerabilities) == 0
        assert len(scanner.scan_history) == 0

    def test_register_vulnerability(self):
        """Test registering a vulnerability."""
        scanner = VulnerabilityScanner()

        vuln_id = scanner.register_vulnerability(
            title="Command Injection",
            description="OS command injection in file upload",
            severity=VulnerabilitySeverity.CRITICAL,
            cvss_score=9.5,
            affected_components=["upload_service"],
            attack_vector="Network",
            discovered_by="automated_scan",
            cwe_id="CWE-78",
            remediation_steps=["Input validation", "Use safe APIs"],
        )

        assert vuln_id is not None
        assert len(scanner.vulnerabilities) == 1
        assert (
            scanner.vulnerabilities[vuln_id].severity == VulnerabilitySeverity.CRITICAL
        )

    def test_update_vulnerability_status(self):
        """Test updating vulnerability status."""
        scanner = VulnerabilityScanner()

        vuln_id = scanner.register_vulnerability(
            title="Test Vuln",
            description="Test",
            severity=VulnerabilitySeverity.MEDIUM,
            cvss_score=5.0,
            affected_components=["test"],
            attack_vector="Test",
            discovered_by="test",
        )

        scanner.update_vulnerability_status(
            vuln_id, VulnerabilityStatus.ASSIGNED, assigned_to="developer_1"
        )

        assert scanner.vulnerabilities[vuln_id].status == VulnerabilityStatus.ASSIGNED
        assert scanner.vulnerabilities[vuln_id].assigned_to == "developer_1"

    def test_update_status_to_fixed(self):
        """Test updating status to fixed sets timestamp."""
        scanner = VulnerabilityScanner()

        vuln_id = scanner.register_vulnerability(
            title="Test",
            description="Test",
            severity=VulnerabilitySeverity.LOW,
            cvss_score=3.0,
            affected_components=["test"],
            attack_vector="Test",
            discovered_by="test",
        )

        scanner.update_vulnerability_status(vuln_id, VulnerabilityStatus.FIXED)

        assert scanner.vulnerabilities[vuln_id].fixed_at is not None

    def test_set_fix_deadline(self):
        """Test setting fix deadline."""
        scanner = VulnerabilityScanner()

        vuln_id = scanner.register_vulnerability(
            title="Test",
            description="Test",
            severity=VulnerabilitySeverity.HIGH,
            cvss_score=8.0,
            affected_components=["test"],
            attack_vector="Test",
            discovered_by="test",
        )

        scanner.set_fix_deadline(vuln_id, days=7)

        assert scanner.vulnerabilities[vuln_id].target_fix_date is not None

    def test_get_vulnerabilities_by_severity(self):
        """Test filtering vulnerabilities by severity."""
        scanner = VulnerabilityScanner()

        scanner.register_vulnerability(
            "Critical 1",
            "Test",
            VulnerabilitySeverity.CRITICAL,
            9.0,
            ["test"],
            "Test",
            "test",
        )
        scanner.register_vulnerability(
            "High 1", "Test", VulnerabilitySeverity.HIGH, 7.5, ["test"], "Test", "test"
        )
        scanner.register_vulnerability(
            "Critical 2",
            "Test",
            VulnerabilitySeverity.CRITICAL,
            9.5,
            ["test"],
            "Test",
            "test",
        )

        critical_vulns = scanner.get_vulnerabilities_by_severity(
            VulnerabilitySeverity.CRITICAL
        )

        assert len(critical_vulns) == 2
        assert all(v.severity == VulnerabilitySeverity.CRITICAL for v in critical_vulns)

    def test_get_vulnerabilities_by_status(self):
        """Test filtering vulnerabilities by status."""
        scanner = VulnerabilityScanner()

        vuln1 = scanner.register_vulnerability(
            "Vuln 1", "Test", VulnerabilitySeverity.HIGH, 7.0, ["test"], "Test", "test"
        )
        vuln2 = scanner.register_vulnerability(
            "Vuln 2",
            "Test",
            VulnerabilitySeverity.MEDIUM,
            5.0,
            ["test"],
            "Test",
            "test",
        )

        scanner.update_vulnerability_status(vuln1, VulnerabilityStatus.FIXED)

        fixed_vulns = scanner.get_vulnerabilities_by_status(VulnerabilityStatus.FIXED)
        discovered_vulns = scanner.get_vulnerabilities_by_status(
            VulnerabilityStatus.DISCOVERED
        )

        assert len(fixed_vulns) == 1
        assert len(discovered_vulns) == 1

    def test_get_overdue_vulnerabilities(self):
        """Test getting overdue vulnerabilities."""
        scanner = VulnerabilityScanner()

        # Create overdue vulnerability
        vuln1 = scanner.register_vulnerability(
            "Overdue",
            "Test",
            VulnerabilitySeverity.CRITICAL,
            9.0,
            ["test"],
            "Test",
            "test",
        )
        scanner.set_fix_deadline(vuln1, days=-1)  # Past deadline

        # Create vulnerability within deadline
        vuln2 = scanner.register_vulnerability(
            "On Time", "Test", VulnerabilitySeverity.HIGH, 7.0, ["test"], "Test", "test"
        )
        scanner.set_fix_deadline(vuln2, days=7)

        overdue = scanner.get_overdue_vulnerabilities()

        assert len(overdue) == 1
        assert overdue[0].id == vuln1


class TestPenetrationTestManager:
    """Test penetration test management."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = PenetrationTestManager()

        assert len(manager.tests) == 0
        assert manager.vulnerability_scanner is not None

    def test_create_test(self):
        """Test creating penetration test."""
        manager = PenetrationTestManager()

        test_id = manager.create_test(
            title="Q4 2024 Penetration Test",
            description="Quarterly security assessment",
            test_type=TestType.GRAY_BOX,
            scope=["web_app", "api", "mobile_app"],
            tester_team=["pentester1", "pentester2"],
            out_of_scope=["production_db"],
        )

        assert test_id is not None
        assert len(manager.tests) == 1
        assert manager.tests[test_id].test_type == TestType.GRAY_BOX

    def test_start_test(self):
        """Test starting penetration test."""
        manager = PenetrationTestManager()

        test_id = manager.create_test(
            "Test", "Description", TestType.BLACK_BOX, ["web_app"], ["tester1"]
        )

        manager.start_test(test_id)

        assert manager.tests[test_id].status == TestStatus.IN_PROGRESS
        assert manager.tests[test_id].start_date is not None

    def test_complete_test(self):
        """Test completing penetration test."""
        manager = PenetrationTestManager()

        test_id = manager.create_test(
            "Test", "Description", TestType.WHITE_BOX, ["api"], ["tester1"]
        )

        manager.start_test(test_id)
        manager.complete_test(test_id, "Test completed successfully")

        assert manager.tests[test_id].status == TestStatus.COMPLETED
        assert manager.tests[test_id].end_date is not None
        assert manager.tests[test_id].executive_summary == "Test completed successfully"

    def test_add_finding_to_test(self):
        """Test adding vulnerability finding to test."""
        manager = PenetrationTestManager()

        test_id = manager.create_test(
            "Test", "Description", TestType.GRAY_BOX, ["web_app"], ["tester1"]
        )

        vuln_id = manager.vulnerability_scanner.register_vulnerability(
            "XSS",
            "Cross-site scripting",
            VulnerabilitySeverity.HIGH,
            7.5,
            ["web_app"],
            "Network",
            "tester1",
        )

        manager.add_finding_to_test(test_id, vuln_id)

        assert vuln_id in manager.tests[test_id].findings

    def test_generate_test_report(self):
        """Test generating penetration test report."""
        manager = PenetrationTestManager()

        test_id = manager.create_test(
            "Security Assessment",
            "Annual test",
            TestType.GRAY_BOX,
            ["web_app", "api"],
            ["tester1", "tester2"],
        )

        # Add vulnerabilities
        vuln1 = manager.vulnerability_scanner.register_vulnerability(
            "SQL Injection",
            "SQLi in login",
            VulnerabilitySeverity.CRITICAL,
            9.8,
            ["web_app"],
            "Network",
            "tester1",
        )
        vuln2 = manager.vulnerability_scanner.register_vulnerability(
            "Weak Crypto",
            "Weak encryption",
            VulnerabilitySeverity.MEDIUM,
            5.5,
            ["api"],
            "Network",
            "tester2",
        )

        manager.add_finding_to_test(test_id, vuln1)
        manager.add_finding_to_test(test_id, vuln2)

        report = manager.generate_test_report(test_id)

        assert "test_info" in report
        assert "findings_summary" in report
        assert report["findings_summary"]["total_findings"] == 2
        assert "by_severity" in report["findings_summary"]


class TestRedTeamManager:
    """Test red team engagement management."""

    def test_initialization(self):
        """Test red team manager initialization."""
        manager = RedTeamManager()

        assert len(manager.engagements) == 0

    def test_create_engagement(self):
        """Test creating red team engagement."""
        manager = RedTeamManager()

        engagement_id = manager.create_engagement(
            name="Operation Shadow Strike",
            objectives=["Test detection", "Evaluate response"],
            tactics=["Initial Access", "Privilege Escalation"],
            techniques=["T1190", "T1068"],
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=14),
            team_members=["red_team_lead", "red_team_operator1"],
            target_systems=["web_app", "internal_network"],
            rules_of_engagement=["No data exfiltration", "Report all findings"],
        )

        assert engagement_id is not None
        assert len(manager.engagements) == 1
        assert "T1190" in manager.engagements[engagement_id].techniques

    def test_add_finding(self):
        """Test adding finding to engagement."""
        manager = RedTeamManager()

        engagement_id = manager.create_engagement(
            "Test Op",
            ["obj1"],
            ["tac1"],
            ["tech1"],
            datetime.now(),
            datetime.now() + timedelta(days=7),
            ["team1"],
            ["sys1"],
            ["roe1"],
        )

        manager.add_finding(engagement_id, "FINDING001")

        assert "FINDING001" in manager.engagements[engagement_id].findings

    def test_update_metrics(self):
        """Test updating engagement metrics."""
        manager = RedTeamManager()

        engagement_id = manager.create_engagement(
            "Test Op",
            ["obj1"],
            ["tac1"],
            ["tech1"],
            datetime.now(),
            datetime.now() + timedelta(days=7),
            ["team1"],
            ["sys1"],
            ["roe1"],
        )

        manager.update_metrics(engagement_id, success_rate=0.75, detection_rate=0.60)

        assert manager.engagements[engagement_id].success_rate == 0.75
        assert manager.engagements[engagement_id].detection_rate == 0.60


class TestPurpleTeamManager:
    """Test purple team exercise management."""

    def test_initialization(self):
        """Test purple team manager initialization."""
        manager = PurpleTeamManager()

        assert len(manager.exercises) == 0

    def test_create_exercise(self):
        """Test creating purple team exercise."""
        manager = PurpleTeamManager()

        exercise_id = manager.create_exercise(
            name="Purple Team Exercise Q1",
            description="Collaborative security testing",
            red_team=["red_lead", "red_operator"],
            blue_team=["blue_lead", "blue_analyst"],
            scenarios=["Phishing attack", "Lateral movement"],
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=3),
            objectives=["Improve detection", "Enhance response"],
        )

        assert exercise_id is not None
        assert len(manager.exercises) == 1
        assert "Phishing attack" in manager.exercises[exercise_id].scenarios

    def test_add_lesson_learned(self):
        """Test adding lesson learned."""
        manager = PurpleTeamManager()

        exercise_id = manager.create_exercise(
            "Test Exercise",
            "Test",
            ["red1"],
            ["blue1"],
            ["scenario1"],
            datetime.now(),
            datetime.now() + timedelta(days=1),
            ["obj1"],
        )

        manager.add_lesson_learned(exercise_id, "Need better email filtering")

        assert (
            "Need better email filtering"
            in manager.exercises[exercise_id].lessons_learned
        )

    def test_add_improvement(self):
        """Test adding identified improvement."""
        manager = PurpleTeamManager()

        exercise_id = manager.create_exercise(
            "Test Exercise",
            "Test",
            ["red1"],
            ["blue1"],
            ["scenario1"],
            datetime.now(),
            datetime.now() + timedelta(days=1),
            ["obj1"],
        )

        manager.add_improvement(exercise_id, "Implement SIEM correlation rules")

        assert (
            "Implement SIEM correlation rules"
            in manager.exercises[exercise_id].improvements_identified
        )


class TestBugBountyProgram:
    """Test bug bounty program integration."""

    def test_initialization(self):
        """Test bug bounty program initialization."""
        program = BugBountyProgram("ACME Corp Bug Bounty")

        assert program.program_name == "ACME Corp Bug Bounty"
        assert len(program.submissions) == 0

    def test_submit_vulnerability(self):
        """Test submitting vulnerability."""
        program = BugBountyProgram("Test Program")

        submission_id = program.submit_vulnerability(
            researcher="security_researcher_1",
            title="XSS in profile page",
            description="Stored XSS vulnerability",
            severity=VulnerabilitySeverity.HIGH,
            proof_of_concept="<script>alert(1)</script>",
        )

        assert submission_id is not None
        assert len(program.submissions) == 1
        assert (
            program.submissions[submission_id]["researcher"] == "security_researcher_1"
        )

    def test_validate_submission_valid(self):
        """Test validating a valid submission."""
        program = BugBountyProgram("Test Program")

        submission_id = program.submit_vulnerability(
            "researcher1", "Bug", "Description", VulnerabilitySeverity.CRITICAL, "POC"
        )

        program.validate_submission(submission_id, is_valid=True)

        assert program.submissions[submission_id]["status"] == "validated"
        assert program.submissions[submission_id]["reward_amount"] > 0

    def test_validate_submission_invalid(self):
        """Test rejecting an invalid submission."""
        program = BugBountyProgram("Test Program")

        submission_id = program.submit_vulnerability(
            "researcher1", "Bug", "Description", VulnerabilitySeverity.LOW, "POC"
        )

        program.validate_submission(submission_id, is_valid=False)

        assert program.submissions[submission_id]["status"] == "rejected"
        assert program.submissions[submission_id]["reward_amount"] == 0.0

    def test_get_program_stats(self):
        """Test getting program statistics."""
        program = BugBountyProgram("Test Program")

        # Submit and validate some vulnerabilities
        sub1 = program.submit_vulnerability(
            "researcher1", "Bug1", "Desc", VulnerabilitySeverity.CRITICAL, "POC"
        )
        sub2 = program.submit_vulnerability(
            "researcher2", "Bug2", "Desc", VulnerabilitySeverity.HIGH, "POC"
        )
        sub3 = program.submit_vulnerability(
            "researcher3", "Bug3", "Desc", VulnerabilitySeverity.LOW, "POC"
        )

        program.validate_submission(sub1, True)
        program.validate_submission(sub2, True)
        program.validate_submission(sub3, False)

        stats = program.get_program_stats()

        assert stats["total_submissions"] == 3
        assert stats["validated_submissions"] == 2
        assert stats["total_rewards_paid"] > 0


class TestPenetrationTestingFramework:
    """Test comprehensive penetration testing framework."""

    def test_initialization(self):
        """Test framework initialization."""
        framework = PenetrationTestingFramework("ACME Corporation")

        assert framework.organization == "ACME Corporation"
        assert framework.test_manager is not None
        assert framework.red_team_manager is not None
        assert framework.purple_team_manager is not None
        assert framework.bug_bounty_program is not None

    def test_generate_comprehensive_report(self):
        """Test generating comprehensive report."""
        framework = PenetrationTestingFramework("Test Org")

        # Add some data
        test_id = framework.test_manager.create_test(
            "Test", "Description", TestType.GRAY_BOX, ["web_app"], ["tester1"]
        )

        vuln_id = framework.test_manager.vulnerability_scanner.register_vulnerability(
            "Test Vuln",
            "Description",
            VulnerabilitySeverity.HIGH,
            7.5,
            ["web_app"],
            "Network",
            "tester1",
        )

        framework.test_manager.add_finding_to_test(test_id, vuln_id)

        report = framework.generate_comprehensive_report()

        assert "metadata" in report
        assert "vulnerability_summary" in report
        assert "penetration_tests" in report
        assert "red_team_engagements" in report
        assert "purple_team_exercises" in report
        assert "bug_bounty" in report

    def test_export_to_json(self):
        """Test exporting to JSON file."""
        framework = PenetrationTestingFramework("Test Org")

        # Add some test data
        framework.test_manager.vulnerability_scanner.register_vulnerability(
            "Test", "Test", VulnerabilitySeverity.MEDIUM, 5.0, ["test"], "Test", "test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pentest_report.json"
            framework.export_to_json(str(filepath))

            assert filepath.exists()

            # Verify JSON is valid
            with open(filepath, "r") as f:
                data = json.load(f)
                assert "metadata" in data
                assert "vulnerability_summary" in data


class TestIntegration:
    """Test integration scenarios."""

    def test_full_penetration_test_lifecycle(self):
        """Test complete penetration test lifecycle."""
        framework = PenetrationTestingFramework("Test Organization")

        # Create penetration test
        test_id = framework.test_manager.create_test(
            title="Annual Security Assessment 2024",
            description="Comprehensive security testing",
            test_type=TestType.GRAY_BOX,
            scope=["web_app", "api", "mobile_app"],
            tester_team=["pentester1", "pentester2"],
        )

        # Start test
        framework.test_manager.start_test(test_id)

        # Register vulnerabilities
        vuln1 = framework.test_manager.vulnerability_scanner.register_vulnerability(
            title="SQL Injection",
            description="SQL injection in login form",
            severity=VulnerabilitySeverity.CRITICAL,
            cvss_score=9.8,
            affected_components=["web_app"],
            attack_vector="Network",
            discovered_by="pentester1",
        )

        vuln2 = framework.test_manager.vulnerability_scanner.register_vulnerability(
            title="Broken Authentication",
            description="Weak session management",
            severity=VulnerabilitySeverity.HIGH,
            cvss_score=8.2,
            affected_components=["api"],
            attack_vector="Network",
            discovered_by="pentester2",
        )

        # Link vulnerabilities to test
        framework.test_manager.add_finding_to_test(test_id, vuln1)
        framework.test_manager.add_finding_to_test(test_id, vuln2)

        # Complete test
        framework.test_manager.complete_test(
            test_id, "Test completed. Found 2 critical/high severity vulnerabilities."
        )

        # Generate report
        report = framework.test_manager.generate_test_report(test_id)

        assert report["test_info"]["status"] == TestStatus.COMPLETED.value
        assert report["findings_summary"]["total_findings"] == 2
        assert report["findings_summary"]["by_severity"]["critical"] == 1
        assert report["findings_summary"]["by_severity"]["high"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
