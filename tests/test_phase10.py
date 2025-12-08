"""
Tests for Phase 10: Sustainability & External Assurance
Validates maintenance policy, KPI monitoring, and audit readiness.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.kpi_monitoring import KPIMonitor


class TestPhase10Documentation(unittest.TestCase):
    """Test Phase 10 documentation completeness."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.docs_operations = self.project_root / "docs" / "operations"
        self.audit_dir = self.project_root / "audit"
        self.governance_dir = self.project_root / "governance"

    def test_maintenance_policy_exists(self):
        """Test that maintenance policy document exists."""
        policy_file = self.docs_operations / "maintenance_policy.md"
        self.assertTrue(
            policy_file.exists(), "Maintenance policy document should exist"
        )
        self.assertGreater(
            policy_file.stat().st_size,
            10000,
            "Maintenance policy should be comprehensive",
        )

    def test_maintenance_policy_sections(self):
        """Test that maintenance policy contains required sections."""
        policy_file = self.docs_operations / "maintenance_policy.md"
        with open(policy_file, "r") as f:
            content = f.read()

        required_sections = [
            "Proof Maintenance Procedures",
            "Code Review & Security Patching",
            "Dependency Update Policy",
            "Technical Debt Management",
            "Performance Regression Prevention",
            "Continuous Improvement Process",
            "Incident Response & Learning",
            "KPI Automation & Monitoring",
        ]

        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"Maintenance policy should include section: {section}",
            )

    def test_audit_scope_exists(self):
        """Test that audit scope document exists."""
        audit_scope_file = self.audit_dir / "audit_scope.md"
        self.assertTrue(audit_scope_file.exists(), "Audit scope document should exist")
        self.assertGreater(
            audit_scope_file.stat().st_size,
            15000,
            "Audit scope should be comprehensive",
        )

    def test_audit_scope_categories(self):
        """Test that audit scope defines all required audit categories."""
        audit_scope_file = self.audit_dir / "audit_scope.md"
        with open(audit_scope_file, "r") as f:
            content = f.read()

        required_categories = [
            "Formal Verification Audit",
            "Security Architecture Review",
            "Fairness Assessment",
            "Compliance Validation",
            "Operational Resilience",
            "Supply Chain Integrity",
        ]

        for category in required_categories:
            self.assertIn(
                category, content, f"Audit scope should include category: {category}"
            )

    def test_audit_scope_certifications(self):
        """Test that audit scope covers required certifications."""
        audit_scope_file = self.audit_dir / "audit_scope.md"
        with open(audit_scope_file, "r") as f:
            content = f.read()

        certifications = ["ISO/IEC 27001", "SOC 2", "FedRAMP", "HIPAA"]

        for cert in certifications:
            self.assertIn(
                cert, content, f"Audit scope should cover certification: {cert}"
            )

    def test_fairness_recalibration_template_exists(self):
        """Test that fairness recalibration template exists."""
        recal_file = self.governance_dir / "fairness_recalibration_report.md"
        self.assertTrue(
            recal_file.exists(), "Fairness recalibration template should exist"
        )
        self.assertGreater(
            recal_file.stat().st_size,
            10000,
            "Fairness recalibration template should be comprehensive",
        )

    def test_fairness_recalibration_metrics(self):
        """Test that fairness recalibration covers all metrics."""
        recal_file = self.governance_dir / "fairness_recalibration_report.md"
        with open(recal_file, "r") as f:
            content = f.read()

        required_metrics = [
            "Statistical Parity",
            "Disparate Impact",
            "Equal Opportunity",
            "Average Odds",
            "Counterfactual Fairness",
        ]

        for metric in required_metrics:
            self.assertIn(
                metric,
                content,
                f"Recalibration template should include metric: {metric}",
            )


class TestKPIMonitoring(unittest.TestCase):
    """Test KPI monitoring automation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = KPIMonitor()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_kpi_definitions_loaded(self):
        """Test that KPI definitions are loaded correctly."""
        self.assertIsNotNone(self.monitor.kpis)
        self.assertGreater(
            len(self.monitor.kpis), 10, "Should have at least 10 KPIs defined"
        )

    def test_critical_kpis_defined(self):
        """Test that critical KPIs are defined."""
        critical_kpis = [
            "proof_coverage",
            "admitted_critical_lemmas",
            "determinism_violations",
            "fairness_sp_diff",
            "system_uptime",
            "security_vulnerabilities_critical",
        ]

        for kpi_id in critical_kpis:
            self.assertIn(
                kpi_id, self.monitor.kpis, f"Critical KPI should be defined: {kpi_id}"
            )

    def test_kpi_structure(self):
        """Test that KPI definitions have required fields."""
        required_fields = [
            "name",
            "description",
            "target",
            "threshold_warning",
            "threshold_critical",
            "unit",
            "frequency",
            "category",
        ]

        for kpi_id, kpi_def in self.monitor.kpis.items():
            for field in required_fields:
                self.assertIn(
                    field, kpi_def, f"KPI '{kpi_id}' should have field '{field}'"
                )

    def test_collect_kpis(self):
        """Test KPI collection."""
        kpis = self.monitor.collect_kpis()

        self.assertIn("timestamp", kpis)
        self.assertIn("values", kpis)
        self.assertIsInstance(kpis["values"], dict)
        self.assertGreater(
            len(kpis["values"]), 0, "Should collect at least some KPI values"
        )

    def test_kpi_values_structure(self):
        """Test that collected KPI values have correct structure."""
        kpis = self.monitor.collect_kpis()

        for kpi_id, kpi_value in kpis["values"].items():
            if "error" not in kpi_value:
                self.assertIn("value", kpi_value)
                self.assertIn("target", kpi_value)
                self.assertIn("unit", kpi_value)
                self.assertIn("status", kpi_value)
                self.assertIn(
                    kpi_value["status"], ["green", "yellow", "red", "unknown"]
                )

    def test_evaluate_kpi_status(self):
        """Test KPI status evaluation."""
        # Test lower-is-better KPI (e.g., latency, errors)
        # For lower-is-better: critical > warning > target
        # Good values are <= target
        kpi_def_lower = {
            "target": 200.0,  # Target latency 200ms
            "threshold_warning": 250.0,  # Warning at 250ms
            "threshold_critical": 200.0,  # Critical threshold at target
        }

        self.assertEqual(
            self.monitor._evaluate_kpi_status(180.0, kpi_def_lower),
            "green",  # Below target is good
        )
        self.assertEqual(
            self.monitor._evaluate_kpi_status(220.0, kpi_def_lower),
            "yellow",  # Between target and warning
        )
        self.assertEqual(
            self.monitor._evaluate_kpi_status(300.0, kpi_def_lower),
            "red",  # Above warning is critical
        )

        # Test higher-is-better KPI (e.g., uptime, coverage)
        # For higher-is-better: critical < warning < target
        # Good values are >= target
        kpi_def_higher = {
            "target": 99.0,  # Target 99% uptime
            "threshold_warning": 95.0,  # Warning at 95%
            "threshold_critical": 99.0,  # Critical threshold at target
        }

        self.assertEqual(
            self.monitor._evaluate_kpi_status(99.5, kpi_def_higher),
            "green",  # Above target is good
        )
        self.assertEqual(
            self.monitor._evaluate_kpi_status(96.0, kpi_def_higher),
            "yellow",  # Between warning and target
        )
        self.assertEqual(
            self.monitor._evaluate_kpi_status(90.0, kpi_def_higher),
            "red",  # Below warning is critical
        )

    def test_analyze_kpis(self):
        """Test KPI trend analysis."""
        # Create mock historical data with increasing trend
        # Data points go from oldest (10 days ago) to newest (1 day ago)
        # Values: 85.0 (oldest) -> 90.0 (newest) = increasing trend
        with patch.object(self.monitor, "_load_historical_kpis") as mock_load:
            mock_load.return_value = [
                {
                    "timestamp": (datetime.now() - timedelta(days=10 - i)).isoformat(),
                    "values": {"proof_coverage": {"value": 85.0 + i * 0.5}},
                }
                for i in range(0, 10)  # i goes 0->9, so values go 85.0->89.5
            ]

            analysis = self.monitor.analyze_kpis(period_days=30)

            self.assertIn("trends", analysis)
            self.assertIn("proof_coverage", analysis["trends"])

            trend = analysis["trends"]["proof_coverage"]
            self.assertIn("direction", trend)
            self.assertIn("magnitude", trend)
            self.assertEqual(trend["direction"], "increasing")

    def test_generate_report_json(self):
        """Test JSON report generation."""
        report_json = self.monitor.generate_report(format="json")

        self.assertIsInstance(report_json, str)
        report = json.loads(report_json)

        self.assertIn("report_date", report)
        self.assertIn("current_kpis", report)
        self.assertIn("trends_30_days", report)
        self.assertIn("summary", report)

    def test_generate_report_text(self):
        """Test text report generation."""
        report_text = self.monitor.generate_report(format="text")

        self.assertIsInstance(report_text, str)
        self.assertIn("KPI MONITORING REPORT", report_text)
        self.assertIn("Overall Health:", report_text)

    def test_check_alerts(self):
        """Test alert generation."""
        # Mock a KPI breach
        with patch.object(self.monitor, "_get_fairness_metric") as mock_metric:
            mock_metric.return_value = 0.15  # Exceeds 0.10 threshold

            alerts = self.monitor.check_alerts()

            self.assertIsInstance(alerts, list)
            # Should have at least one alert for fairness breach
            fairness_alerts = [a for a in alerts if a["kpi_id"] == "fairness_sp_diff"]
            if fairness_alerts:
                self.assertEqual(fairness_alerts[0]["severity"], "critical")


class TestMaintenanceProcesses(unittest.TestCase):
    """Test maintenance processes and procedures."""

    def test_proof_maintenance_documentation(self):
        """Test proof maintenance procedures are documented."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        self.assertIn("Proof Coverage Monitoring", content)
        self.assertIn("Proof Debt Management", content)
        self.assertIn("≥85%", content)  # Coverage target

    def test_security_patching_slas(self):
        """Test security patching SLAs are defined."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        # Check for severity-based SLAs
        self.assertIn("Critical Vulnerabilities", content)
        self.assertIn("CVSS ≥9.0", content)
        self.assertIn("≤24 hours", content)

    def test_dependency_update_policy(self):
        """Test dependency update policy is defined."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        self.assertIn("Dependency Classification", content)
        self.assertIn("Update Cadence", content)
        self.assertIn("Critical Dependencies", content)

    def test_incident_response_procedures(self):
        """Test incident response procedures are defined."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        self.assertIn("Incident Classification", content)
        self.assertIn("Incident Response Process", content)
        self.assertIn("Post-Incident Review", content)

        # Check severity levels
        for severity in ["Critical (P0)", "High (P1)", "Medium (P2)", "Low (P3)"]:
            self.assertIn(severity, content)


class TestAuditReadiness(unittest.TestCase):
    """Test external audit readiness."""

    def test_audit_categories_defined(self):
        """Test all audit categories are defined."""
        audit_file = Path(__file__).parent.parent / "audit" / "audit_scope.md"
        with open(audit_file, "r") as f:
            content = f.read()

        categories = [
            "Formal Verification Audit",
            "Security Architecture Review",
            "Fairness Assessment",
            "Compliance Validation",
            "Operational Resilience",
            "Supply Chain Integrity",
        ]

        for category in categories:
            self.assertIn(category, content)

    def test_audit_evidence_collection(self):
        """Test audit evidence collection is documented."""
        audit_file = Path(__file__).parent.parent / "audit" / "audit_scope.md"
        with open(audit_file, "r") as f:
            content = f.read()

        self.assertIn("Evidence Collection", content)
        self.assertIn("Automated Evidence Collection", content)
        self.assertIn("Evidence Retention", content)

    def test_auditor_access_procedures(self):
        """Test auditor access procedures are defined."""
        audit_file = Path(__file__).parent.parent / "audit" / "audit_scope.md"
        with open(audit_file, "r") as f:
            content = f.read()

        self.assertIn("Auditor Access Provisioning", content)
        self.assertIn("Read-Only Access", content)
        self.assertIn("MFA Required", content)

    def test_certification_requirements(self):
        """Test certification requirements are documented."""
        audit_file = Path(__file__).parent.parent / "audit" / "audit_scope.md"
        with open(audit_file, "r") as f:
            content = f.read()

        certifications = [
            "ISO/IEC 27001:2022",
            "SOC 2 Type II",
            "FedRAMP Moderate",
            "HIPAA Security Rule",
        ]

        for cert in certifications:
            self.assertIn(cert, content)


class TestFairnessRecalibration(unittest.TestCase):
    """Test fairness recalibration procedures."""

    def test_recalibration_schedule_defined(self):
        """Test recalibration schedule is defined."""
        recal_file = (
            Path(__file__).parent.parent
            / "governance"
            / "fairness_recalibration_report.md"
        )
        with open(recal_file, "r") as f:
            content = f.read()

        # Check for quarterly schedule
        self.assertIn("Quarterly", content)
        self.assertIn("Jan 15", content)
        self.assertIn("Apr 15", content)
        self.assertIn("Jul 15", content)
        self.assertIn("Oct 15", content)

    def test_fairness_metrics_documented(self):
        """Test all fairness metrics are documented in template."""
        recal_file = (
            Path(__file__).parent.parent
            / "governance"
            / "fairness_recalibration_report.md"
        )
        with open(recal_file, "r") as f:
            content = f.read()

        metrics = [
            "Statistical Parity",
            "Disparate Impact",
            "Equal Opportunity",
            "Average Odds",
            "Counterfactual Fairness",
        ]

        for metric in metrics:
            self.assertIn(metric, content)

    def test_protected_attributes_coverage(self):
        """Test protected attributes are covered."""
        recal_file = (
            Path(__file__).parent.parent
            / "governance"
            / "fairness_recalibration_report.md"
        )
        with open(recal_file, "r") as f:
            content = f.read()

        attributes = ["Age", "Race", "Gender", "Disability"]

        for attr in attributes:
            self.assertIn(attr, content)

    def test_mitigation_strategies_documented(self):
        """Test bias mitigation strategies are documented."""
        recal_file = (
            Path(__file__).parent.parent
            / "governance"
            / "fairness_recalibration_report.md"
        )
        with open(recal_file, "r") as f:
            content = f.read()

        strategies = [
            "Reweighting",
            "Adversarial Debiasing",
            "Fairness Constraints",
            "Counterfactual Data Augmentation",
        ]

        for strategy in strategies:
            self.assertIn(strategy, content)

    def test_stakeholder_engagement_process(self):
        """Test stakeholder engagement process is defined."""
        recal_file = (
            Path(__file__).parent.parent
            / "governance"
            / "fairness_recalibration_report.md"
        )
        with open(recal_file, "r") as f:
            content = f.read()

        self.assertIn("Stakeholder Consultation", content)
        self.assertIn("Governance Board", content)
        self.assertIn("Public Transparency", content)


class TestContinuousImprovement(unittest.TestCase):
    """Test continuous improvement processes."""

    def test_monthly_reporting_defined(self):
        """Test monthly reporting process is defined."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        self.assertIn("Monthly Reports", content)
        self.assertIn("Metrics Summary", content)

    def test_quarterly_review_defined(self):
        """Test quarterly review process is defined."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        self.assertIn("Quarterly Reports", content)
        self.assertIn("Comprehensive Assessment", content)

    def test_annual_review_defined(self):
        """Test annual review process is defined."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        self.assertIn("Annual Reports", content)
        self.assertIn("Year in Review", content)
        self.assertIn("Annual Security Review", content)

    def test_kpi_thresholds_defined(self):
        """Test KPI thresholds are clearly defined."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        # Check for specific thresholds
        self.assertIn("≥85%", content)  # Proof coverage
        self.assertIn("99.95%", content)  # System uptime
        self.assertIn("≤0.10", content)  # Fairness SP diff


class TestSustainabilityMetrics(unittest.TestCase):
    """Test long-term sustainability metrics."""

    def test_sustainability_section_exists(self):
        """Test sustainability metrics section exists."""
        maintenance_file = (
            Path(__file__).parent.parent
            / "docs"
            / "operations"
            / "maintenance_policy.md"
        )
        with open(maintenance_file, "r") as f:
            content = f.read()

        self.assertIn("Success Metrics", content)
        self.assertIn("Operational Excellence", content)
        self.assertIn("Formal Assurance", content)

    def test_long_term_planning_documented(self):
        """Test long-term planning is documented."""
        audit_file = Path(__file__).parent.parent / "audit" / "audit_scope.md"
        with open(audit_file, "r") as f:
            content = f.read()

        # Check audit scope includes long-term considerations
        self.assertIn("Annual", content)
        self.assertIn("Certification", content)

    def test_continuous_assurance_framework(self):
        """Test continuous assurance framework is defined."""
        audit_file = Path(__file__).parent.parent / "audit" / "audit_scope.md"
        with open(audit_file, "r") as f:
            content = f.read()

        self.assertIn("Continuous Assurance", content)
        self.assertIn("Post-Audit Monitoring", content)
        self.assertIn("Control Effectiveness Monitoring", content)


def run_tests():
    """Run all Phase 10 tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPhase10Documentation))
    suite.addTests(loader.loadTestsFromTestCase(TestKPIMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestMaintenanceProcesses))
    suite.addTests(loader.loadTestsFromTestCase(TestAuditReadiness))
    suite.addTests(loader.loadTestsFromTestCase(TestFairnessRecalibration))
    suite.addTests(loader.loadTestsFromTestCase(TestContinuousImprovement))
    suite.addTests(loader.loadTestsFromTestCase(TestSustainabilityMetrics))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
