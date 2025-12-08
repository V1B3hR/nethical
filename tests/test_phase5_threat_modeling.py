"""
Tests for Phase 5.1: Comprehensive Threat Modeling Framework

This test suite validates the threat modeling capabilities including STRIDE analysis,
attack tree generation, threat intelligence integration, and security requirements
traceability matrix.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from nethical.security.threat_modeling import (
    ThreatCategory,
    ThreatSeverity,
    ThreatStatus,
    Threat,
    AttackTreeNode,
    SecurityRequirement,
    ThreatIntelligenceFeed,
    STRIDEAnalyzer,
    AttackTreeAnalyzer,
    SecurityRequirementsTraceability,
    ThreatModelingFramework,
)


class TestThreatDataClasses:
    """Test threat modeling data classes."""

    def test_threat_creation(self):
        """Test threat object creation."""
        threat = Threat(
            id="T001",
            category=ThreatCategory.SPOOFING,
            title="Identity Spoofing",
            description="Attacker impersonates legitimate user",
            severity=ThreatSeverity.HIGH,
            status=ThreatStatus.IDENTIFIED,
            affected_components=["authentication", "api"],
            attack_vectors=["stolen_credentials", "session_hijacking"],
        )

        assert threat.id == "T001"
        assert threat.category == ThreatCategory.SPOOFING
        assert threat.severity == ThreatSeverity.HIGH
        assert "authentication" in threat.affected_components

    def test_threat_to_dict(self):
        """Test threat serialization."""
        threat = Threat(
            id="T001",
            category=ThreatCategory.TAMPERING,
            title="Data Tampering",
            description="Unauthorized modification of data",
            severity=ThreatSeverity.CRITICAL,
            status=ThreatStatus.MITIGATED,
            affected_components=["database"],
            attack_vectors=["sql_injection"],
            mitigations=["input_validation", "prepared_statements"],
        )

        threat_dict = threat.to_dict()

        assert threat_dict["id"] == "T001"
        assert threat_dict["category"] == "tampering"
        assert threat_dict["severity"] == "critical"
        assert "input_validation" in threat_dict["mitigations"]

    def test_attack_tree_node_creation(self):
        """Test attack tree node creation."""
        node = AttackTreeNode(
            id="ATN001",
            name="Compromise System",
            description="Gain unauthorized access",
            is_and_gate=False,
            probability=0.3,
            impact=0.9,
        )

        assert node.id == "ATN001"
        assert node.is_and_gate is False
        assert node.probability == 0.3

    def test_attack_tree_risk_calculation(self):
        """Test attack tree risk calculation."""
        node = AttackTreeNode(
            id="ATN001",
            name="Attack",
            description="Test attack",
            probability=0.5,
            impact=0.8,
        )

        risk = node.calculate_risk()
        assert risk == 0.4  # 0.5 * 0.8

    def test_security_requirement_creation(self):
        """Test security requirement creation."""
        req = SecurityRequirement(
            id="REQ001",
            title="Multi-Factor Authentication",
            description="Implement MFA for all users",
            category="authentication",
            priority="critical",
            compliance_frameworks=["NIST 800-53", "HIPAA"],
        )

        assert req.id == "REQ001"
        assert req.priority == "critical"
        assert "NIST 800-53" in req.compliance_frameworks


class TestThreatIntelligenceFeed:
    """Test threat intelligence feed."""

    def test_initialization(self):
        """Test threat intelligence feed initialization."""
        feed = ThreatIntelligenceFeed()

        assert len(feed.indicators) == 0
        assert feed.last_update is None

    def test_add_indicator(self):
        """Test adding threat indicator."""
        feed = ThreatIntelligenceFeed()

        indicator_id = feed.add_indicator(
            indicator_type="ip",
            value="192.168.1.100",
            severity=ThreatSeverity.HIGH,
            description="Known malicious IP",
            source="threat_intel_provider",
        )

        assert indicator_id is not None
        assert len(feed.indicators) == 1
        assert feed.last_update is not None

    def test_get_indicators_by_type(self):
        """Test filtering indicators by type."""
        feed = ThreatIntelligenceFeed()

        feed.add_indicator(
            "ip", "192.168.1.1", ThreatSeverity.HIGH, "Malicious IP", "source1"
        )
        feed.add_indicator(
            "domain", "evil.com", ThreatSeverity.CRITICAL, "Malicious domain", "source2"
        )
        feed.add_indicator(
            "ip", "192.168.1.2", ThreatSeverity.MEDIUM, "Suspicious IP", "source1"
        )

        ip_indicators = feed.get_indicators(indicator_type="ip")

        assert len(ip_indicators) == 2
        assert all(ind["type"] == "ip" for ind in ip_indicators)

    def test_check_indicator(self):
        """Test checking if value matches known indicator."""
        feed = ThreatIntelligenceFeed()

        feed.add_indicator(
            "ip", "192.168.1.100", ThreatSeverity.HIGH, "Known bad IP", "source"
        )

        match = feed.check_indicator("192.168.1.100")
        assert match is not None
        assert match["value"] == "192.168.1.100"

        no_match = feed.check_indicator("10.0.0.1")
        assert no_match is None


class TestSTRIDEAnalyzer:
    """Test STRIDE threat analysis."""

    def test_initialization(self):
        """Test STRIDE analyzer initialization."""
        analyzer = STRIDEAnalyzer()

        assert len(analyzer.threats) == 0
        assert len(analyzer.components) == 0

    def test_add_threat(self):
        """Test adding threat to analyzer."""
        analyzer = STRIDEAnalyzer()

        threat_id = analyzer.add_threat(
            category=ThreatCategory.SPOOFING,
            title="Session Hijacking",
            description="Attacker steals user session",
            severity=ThreatSeverity.HIGH,
            affected_components=["web_app", "session_manager"],
            attack_vectors=["xss", "csrf"],
            mitigations=["httponly_cookies", "csrf_tokens"],
        )

        assert threat_id is not None
        assert len(analyzer.threats) == 1
        assert "web_app" in analyzer.components
        assert "session_manager" in analyzer.components

    def test_update_threat_status(self):
        """Test updating threat status."""
        analyzer = STRIDEAnalyzer()

        threat_id = analyzer.add_threat(
            category=ThreatCategory.TAMPERING,
            title="Log Tampering",
            description="Attacker modifies audit logs",
            severity=ThreatSeverity.HIGH,
            affected_components=["audit_system"],
            attack_vectors=["direct_access"],
        )

        analyzer.update_threat_status(threat_id, ThreatStatus.MITIGATED)

        assert analyzer.threats[threat_id].status == ThreatStatus.MITIGATED

    def test_get_threats_by_category(self):
        """Test filtering threats by category."""
        analyzer = STRIDEAnalyzer()

        analyzer.add_threat(
            category=ThreatCategory.SPOOFING,
            title="Threat 1",
            description="Test",
            severity=ThreatSeverity.HIGH,
            affected_components=["comp1"],
            attack_vectors=["vec1"],
        )

        analyzer.add_threat(
            category=ThreatCategory.TAMPERING,
            title="Threat 2",
            description="Test",
            severity=ThreatSeverity.MEDIUM,
            affected_components=["comp2"],
            attack_vectors=["vec2"],
        )

        spoofing_threats = analyzer.get_threats_by_category(ThreatCategory.SPOOFING)

        assert len(spoofing_threats) == 1
        assert spoofing_threats[0].category == ThreatCategory.SPOOFING

    def test_get_threats_by_severity(self):
        """Test filtering threats by severity."""
        analyzer = STRIDEAnalyzer()

        analyzer.add_threat(
            category=ThreatCategory.DENIAL_OF_SERVICE,
            title="DDoS Attack",
            description="Flood server with requests",
            severity=ThreatSeverity.CRITICAL,
            affected_components=["web_server"],
            attack_vectors=["volumetric_attack"],
        )

        analyzer.add_threat(
            category=ThreatCategory.INFORMATION_DISCLOSURE,
            title="Data Leak",
            description="Expose sensitive data",
            severity=ThreatSeverity.HIGH,
            affected_components=["database"],
            attack_vectors=["sql_injection"],
        )

        critical_threats = analyzer.get_threats_by_severity(ThreatSeverity.CRITICAL)

        assert len(critical_threats) == 1
        assert critical_threats[0].severity == ThreatSeverity.CRITICAL

    def test_get_threats_by_component(self):
        """Test filtering threats by component."""
        analyzer = STRIDEAnalyzer()

        analyzer.add_threat(
            category=ThreatCategory.ELEVATION_OF_PRIVILEGE,
            title="Privilege Escalation",
            description="Gain admin access",
            severity=ThreatSeverity.CRITICAL,
            affected_components=["auth_service", "user_manager"],
            attack_vectors=["exploit_vulnerability"],
        )

        auth_threats = analyzer.get_threats_by_component("auth_service")

        assert len(auth_threats) == 1
        assert "auth_service" in auth_threats[0].affected_components

    def test_generate_stride_report(self):
        """Test STRIDE report generation."""
        analyzer = STRIDEAnalyzer()

        # Add threats for different categories
        analyzer.add_threat(
            category=ThreatCategory.SPOOFING,
            title="Identity Theft",
            description="Impersonate user",
            severity=ThreatSeverity.HIGH,
            affected_components=["auth"],
            attack_vectors=["phishing"],
        )

        analyzer.add_threat(
            category=ThreatCategory.TAMPERING,
            title="Data Modification",
            description="Alter critical data",
            severity=ThreatSeverity.CRITICAL,
            affected_components=["database"],
            attack_vectors=["sql_injection"],
        )

        report = analyzer.generate_stride_report()

        assert "total_threats" in report
        assert report["total_threats"] == 2
        assert "by_category" in report
        assert "by_severity" in report
        assert "critical_threats" in report


class TestAttackTreeAnalyzer:
    """Test attack tree analysis."""

    def test_initialization(self):
        """Test attack tree analyzer initialization."""
        analyzer = AttackTreeAnalyzer()

        assert len(analyzer.trees) == 0

    def test_create_attack_tree(self):
        """Test creating attack tree."""
        analyzer = AttackTreeAnalyzer()

        root = analyzer.create_attack_tree(
            tree_id="AT001",
            root_name="Compromise Database",
            root_description="Gain unauthorized access to database",
        )

        assert root.id == "AT001"
        assert root.name == "Compromise Database"
        assert "AT001" in analyzer.trees

    def test_add_child_node(self):
        """Test adding child node to attack tree."""
        analyzer = AttackTreeAnalyzer()

        root = analyzer.create_attack_tree("AT001", "Root", "Root node")

        child = analyzer.add_child_node(
            parent=root,
            node_id="AT001_1",
            name="SQL Injection",
            description="Exploit SQL vulnerability",
            probability=0.4,
            impact=0.8,
        )

        assert len(root.children) == 1
        assert child.name == "SQL Injection"
        assert child.probability == 0.4

    def test_calculate_tree_risk_or_gate(self):
        """Test risk calculation for OR gate."""
        analyzer = AttackTreeAnalyzer()

        root = analyzer.create_attack_tree("AT001", "Attack", "Test attack")
        root.impact = 0.9

        # Add children (OR gate - take maximum)
        analyzer.add_child_node(
            root, "C1", "Method 1", "First method", probability=0.3, impact=0.8
        )
        analyzer.add_child_node(
            root, "C2", "Method 2", "Second method", probability=0.5, impact=0.7
        )

        risk = analyzer.calculate_tree_risk(root)

        # OR gate should take max risk from children
        assert risk > 0

    def test_calculate_tree_risk_and_gate(self):
        """Test risk calculation for AND gate."""
        analyzer = AttackTreeAnalyzer()

        root = analyzer.create_attack_tree("AT001", "Attack", "Test attack")
        root.is_and_gate = True
        root.impact = 0.9

        # Add children (AND gate - multiply probabilities)
        analyzer.add_child_node(
            root, "C1", "Step 1", "First step", probability=0.8, impact=0.7
        )
        analyzer.add_child_node(
            root, "C2", "Step 2", "Second step", probability=0.6, impact=0.8
        )

        risk = analyzer.calculate_tree_risk(root)

        # AND gate multiplies probabilities
        assert risk > 0

    def test_export_attack_tree(self):
        """Test exporting attack tree."""
        analyzer = AttackTreeAnalyzer()

        root = analyzer.create_attack_tree("AT001", "Root Attack", "Test")
        analyzer.add_child_node(root, "C1", "Child 1", "Test child")

        exported = analyzer.export_attack_tree("AT001")

        assert exported is not None
        assert exported["tree_id"] == "AT001"
        assert "tree" in exported
        assert "total_risk" in exported


class TestSecurityRequirementsTraceability:
    """Test security requirements traceability matrix."""

    def test_initialization(self):
        """Test traceability matrix initialization."""
        matrix = SecurityRequirementsTraceability()

        assert len(matrix.requirements) == 0

    def test_add_requirement(self):
        """Test adding security requirement."""
        matrix = SecurityRequirementsTraceability()

        req = matrix.add_requirement(
            req_id="REQ001",
            title="Authentication Required",
            description="All users must authenticate",
            category="authentication",
            priority="critical",
            compliance_frameworks=["NIST 800-53"],
        )

        assert req.id == "REQ001"
        assert "REQ001" in matrix.requirements

    def test_link_to_threat(self):
        """Test linking requirement to threat."""
        matrix = SecurityRequirementsTraceability()

        matrix.add_requirement("REQ001", "Auth", "Test", "auth", "high")
        matrix.link_to_threat("REQ001", "THREAT001")

        assert "THREAT001" in matrix.requirements["REQ001"].related_threats

    def test_link_to_implementation(self):
        """Test linking requirement to implementation."""
        matrix = SecurityRequirementsTraceability()

        matrix.add_requirement("REQ001", "Auth", "Test", "auth", "high")
        matrix.link_to_implementation("REQ001", "authentication.py")

        assert "authentication.py" in matrix.requirements["REQ001"].implemented_in

    def test_link_to_test(self):
        """Test linking requirement to test case."""
        matrix = SecurityRequirementsTraceability()

        matrix.add_requirement("REQ001", "Auth", "Test", "auth", "high")
        matrix.link_to_test("REQ001", "test_authentication.py::test_login")

        assert (
            "test_authentication.py::test_login"
            in matrix.requirements["REQ001"].test_cases
        )

    def test_update_status(self):
        """Test updating requirement status."""
        matrix = SecurityRequirementsTraceability()

        matrix.add_requirement("REQ001", "Auth", "Test", "auth", "high")
        matrix.update_status("REQ001", "implemented")

        assert matrix.requirements["REQ001"].status == "implemented"

    def test_get_traceability_matrix(self):
        """Test generating traceability matrix."""
        matrix = SecurityRequirementsTraceability()

        matrix.add_requirement("REQ001", "Auth", "Test", "auth", "high")
        matrix.link_to_implementation("REQ001", "auth.py")
        matrix.link_to_test("REQ001", "test_auth.py")

        report = matrix.get_traceability_matrix()

        assert "total_requirements" in report
        assert report["total_requirements"] == 1
        assert "coverage_stats" in report

    def test_coverage_stats_calculation(self):
        """Test coverage statistics calculation."""
        matrix = SecurityRequirementsTraceability()

        # Add requirements with varying coverage
        matrix.add_requirement("REQ001", "R1", "Test", "cat1", "high")
        matrix.link_to_implementation("REQ001", "impl1.py")
        matrix.link_to_test("REQ001", "test1.py")
        matrix.link_to_threat("REQ001", "T001")

        matrix.add_requirement("REQ002", "R2", "Test", "cat2", "medium")
        # REQ002 has no links

        report = matrix.get_traceability_matrix()
        stats = report["coverage_stats"]

        assert stats["implementation_coverage"] == 0.5  # 1 of 2
        assert stats["test_coverage"] == 0.5
        assert stats["threat_coverage"] == 0.5


class TestThreatModelingFramework:
    """Test comprehensive threat modeling framework."""

    def test_initialization(self):
        """Test framework initialization."""
        framework = ThreatModelingFramework()

        assert framework.stride_analyzer is not None
        assert framework.attack_tree_analyzer is not None
        assert framework.requirements_matrix is not None
        assert framework.threat_intelligence is not None

    def test_generate_comprehensive_report(self):
        """Test generating comprehensive report."""
        framework = ThreatModelingFramework()

        # Add some data
        framework.stride_analyzer.add_threat(
            category=ThreatCategory.SPOOFING,
            title="Test Threat",
            description="Test",
            severity=ThreatSeverity.HIGH,
            affected_components=["test"],
            attack_vectors=["test_vector"],
        )

        framework.requirements_matrix.add_requirement(
            req_id="REQ001",
            title="Test Requirement",
            description="Test",
            category="test",
            priority="high",
        )

        report = framework.generate_comprehensive_report()

        assert "metadata" in report
        assert "stride_analysis" in report
        assert "attack_trees" in report
        assert "requirements_traceability" in report
        assert "threat_intelligence" in report

    def test_export_to_json(self):
        """Test exporting to JSON file."""
        framework = ThreatModelingFramework()

        framework.stride_analyzer.add_threat(
            category=ThreatCategory.TAMPERING,
            title="Test Threat",
            description="Test",
            severity=ThreatSeverity.MEDIUM,
            affected_components=["test"],
            attack_vectors=["test"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "threat_model.json"
            framework.export_to_json(str(filepath))

            assert filepath.exists()

            # Verify JSON is valid
            with open(filepath, "r") as f:
                data = json.load(f)
                assert "metadata" in data
                assert "stride_analysis" in data

    def test_import_from_json(self):
        """Test importing from JSON file."""
        framework1 = ThreatModelingFramework()

        # Add data to first framework
        framework1.stride_analyzer.add_threat(
            category=ThreatCategory.INFORMATION_DISCLOSURE,
            title="Data Leak",
            description="Sensitive data exposed",
            severity=ThreatSeverity.HIGH,
            affected_components=["api"],
            attack_vectors=["injection"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "threat_model.json"

            # Export
            framework1.export_to_json(str(filepath))

            # Import into new framework
            framework2 = ThreatModelingFramework()
            framework2.import_from_json(str(filepath))

            # Verify data was imported
            assert len(framework2.stride_analyzer.threats) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
