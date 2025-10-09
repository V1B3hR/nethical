"""Tests for F6: Marketplace & Ecosystem features.

This test suite validates all marketplace functionality including:
- MarketplaceClient: Search, install, manage plugins
- PluginGovernance: Security scanning, benchmarking, certification
- CommunityManager: Reviews, submissions, contributions
- DetectorPackRegistry: Pre-built packs, industry-specific bundles
- IntegrationDirectory: Third-party integrations
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from nethical.marketplace import (
    MarketplaceClient,
    PluginInfo,
    PluginVersion,
    SearchFilters,
    InstallStatus,
    PluginGovernance,
    SecurityLevel,
    CertificationStatus,
    CommunityManager,
    ReviewStatus,
    DetectorPackRegistry,
    Industry,
    IntegrationDirectory,
    IntegrationType,
    ExportUtility,
    ImportUtility
)


class TestMarketplaceClient(unittest.TestCase):
    """Test MarketplaceClient functionality."""
    
    def setUp(self):
        """Set up test marketplace client."""
        self.test_dir = tempfile.mkdtemp()
        self.client = MarketplaceClient(storage_dir=self.test_dir)
        
        # Register test plugins
        self._register_test_plugins()
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _register_test_plugins(self):
        """Register test plugins in marketplace."""
        # Financial plugin
        financial_plugin = PluginInfo(
            plugin_id="financial-compliance-v2",
            name="Financial Compliance Pack",
            description="Detectors for financial compliance",
            author="Nethical Team",
            category="financial",
            rating=4.8,
            download_count=1250,
            tags={"financial", "compliance", "fraud"},
            versions=[
                PluginVersion(
                    version="1.2.3",
                    release_date=datetime.now(),
                    compatibility=">=0.1.0",
                    changelog="Bug fixes and improvements"
                )
            ],
            latest_version="1.2.3",
            dependencies=[],
            certified=True
        )
        self.client.register_plugin(financial_plugin)
        
        # Healthcare plugin
        healthcare_plugin = PluginInfo(
            plugin_id="healthcare-hipaa",
            name="Healthcare HIPAA Pack",
            description="HIPAA compliance detectors",
            author="MedTech Corp",
            category="healthcare",
            rating=4.5,
            download_count=850,
            tags={"healthcare", "hipaa", "compliance"},
            versions=[
                PluginVersion(
                    version="2.0.0",
                    release_date=datetime.now(),
                    compatibility=">=0.1.0"
                )
            ],
            latest_version="2.0.0",
            certified=True
        )
        self.client.register_plugin(healthcare_plugin)
        
        # Low-rated plugin
        low_rated_plugin = PluginInfo(
            plugin_id="low-quality-plugin",
            name="Low Quality Plugin",
            description="A plugin with low rating",
            author="Unknown",
            category="general",
            rating=2.5,
            download_count=50,
            tags={"general"},
            versions=[
                PluginVersion(
                    version="0.1.0",
                    release_date=datetime.now(),
                    compatibility=">=0.1.0"
                )
            ],
            latest_version="0.1.0",
            certified=False
        )
        self.client.register_plugin(low_rated_plugin)
    
    def test_search_by_category(self):
        """Test searching plugins by category."""
        results = self.client.search(category="financial")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].plugin_id, "financial-compliance-v2")
    
    def test_search_by_rating(self):
        """Test searching plugins by minimum rating."""
        results = self.client.search(min_rating=4.0)
        
        self.assertEqual(len(results), 2)
        for plugin in results:
            self.assertGreaterEqual(plugin.rating, 4.0)
    
    def test_search_certified_only(self):
        """Test searching only certified plugins."""
        results = self.client.search(certified_only=True)
        
        self.assertEqual(len(results), 2)
        for plugin in results:
            self.assertTrue(plugin.certified)
    
    def test_search_by_tags(self):
        """Test searching plugins by tags."""
        results = self.client.search(tags={"compliance"})
        
        self.assertEqual(len(results), 2)
        for plugin in results:
            self.assertIn("compliance", plugin.tags)
    
    def test_search_with_query(self):
        """Test text-based search."""
        results = self.client.search(query="HIPAA")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].plugin_id, "healthcare-hipaa")
    
    def test_install_plugin(self):
        """Test installing a plugin."""
        status = self.client.install("financial-compliance-v2")
        
        self.assertEqual(status, InstallStatus.INSTALLED)
        
        # Verify plugin is in installed list
        installed = self.client.list_installed()
        self.assertEqual(len(installed), 1)
        self.assertEqual(installed[0].plugin_id, "financial-compliance-v2")
    
    def test_install_specific_version(self):
        """Test installing a specific plugin version."""
        status = self.client.install("financial-compliance-v2", version="1.2.3")
        
        self.assertEqual(status, InstallStatus.INSTALLED)
    
    def test_install_nonexistent_plugin(self):
        """Test installing a non-existent plugin."""
        with self.assertRaises(ValueError):
            self.client.install("nonexistent-plugin")
    
    def test_uninstall_plugin(self):
        """Test uninstalling a plugin."""
        # First install
        self.client.install("financial-compliance-v2")
        
        # Then uninstall
        result = self.client.uninstall("financial-compliance-v2")
        self.assertTrue(result)
        
        # Verify not in installed list
        installed = self.client.list_installed()
        self.assertEqual(len(installed), 0)
    
    def test_get_plugin_info(self):
        """Test getting plugin information."""
        info = self.client.get_plugin_info("financial-compliance-v2")
        
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "Financial Compliance Pack")
        self.assertEqual(info.rating, 4.8)


class TestPluginGovernance(unittest.TestCase):
    """Test PluginGovernance functionality."""
    
    def setUp(self):
        """Set up test governance."""
        self.test_dir = tempfile.mkdtemp()
        self.governance = PluginGovernance(storage_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_security_scan_safe_code(self):
        """Test security scan with safe code."""
        safe_code = """
def detect_violations(action):
    violations = []
    if "spam" in action.content:
        violations.append("spam detected")
    return violations
"""
        result = self.governance.security_scan("test-plugin", plugin_code=safe_code)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.security_level, SecurityLevel.SAFE)
        self.assertEqual(len(result.vulnerabilities), 0)
    
    def test_security_scan_dangerous_code(self):
        """Test security scan with dangerous code."""
        dangerous_code = """
import subprocess
eval(user_input)
exec(dangerous_code)
"""
        result = self.governance.security_scan("test-plugin", plugin_code=dangerous_code)
        
        self.assertFalse(result.passed)
        self.assertGreater(len(result.vulnerabilities), 0)
    
    def test_benchmark(self):
        """Test plugin performance benchmarking."""
        result = self.governance.benchmark("test-plugin", iterations=100)
        
        self.assertGreater(result.avg_latency_ms, 0)
        self.assertGreater(result.throughput_ops_per_sec, 0)
        self.assertGreater(result.p95_latency_ms, result.avg_latency_ms)
    
    def test_compatibility_test(self):
        """Test plugin compatibility testing."""
        result = self.governance.compatibility_test("test-plugin")
        
        self.assertTrue(result.compatible)
        self.assertGreater(len(result.test_results), 0)
        self.assertTrue(all(result.test_results.values()))
    
    def test_certification_process(self):
        """Test plugin certification."""
        # Run certification
        status = self.governance.certify("test-plugin")
        
        # Should be certified (mocked tests pass)
        self.assertEqual(status, CertificationStatus.CERTIFIED)
        
        # Verify status
        saved_status = self.governance.get_certification_status("test-plugin")
        self.assertEqual(saved_status, CertificationStatus.CERTIFIED)
    
    def test_revoke_certification(self):
        """Test revoking plugin certification."""
        # First certify
        self.governance.certify("test-plugin")
        
        # Then revoke
        result = self.governance.revoke_certification("test-plugin", "Security issue found")
        self.assertTrue(result)
        
        # Verify revoked
        status = self.governance.get_certification_status("test-plugin")
        self.assertEqual(status, CertificationStatus.REVOKED)
    
    def test_generate_report(self):
        """Test generating governance report."""
        # Run all checks
        security_result = self.governance.security_scan("test-plugin", plugin_code="# safe code")
        benchmark_result = self.governance.benchmark("test-plugin")
        compat_result = self.governance.compatibility_test("test-plugin")
        
        # Generate report
        report = self.governance.generate_report(
            "test-plugin",
            security_result,
            benchmark_result,
            compat_result
        )
        
        self.assertIn('plugin_id', report)
        self.assertIn('security', report)
        self.assertIn('performance', report)
        self.assertIn('compatibility', report)


class TestCommunityManager(unittest.TestCase):
    """Test CommunityManager functionality."""
    
    def setUp(self):
        """Set up test community manager."""
        self.test_dir = tempfile.mkdtemp()
        self.community = CommunityManager(storage_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_submit_plugin(self):
        """Test plugin submission."""
        submission = self.community.submit_plugin("my-plugin", "author1")
        
        self.assertIsNotNone(submission)
        self.assertEqual(submission.plugin_id, "my-plugin")
        self.assertEqual(submission.author, "author1")
        self.assertEqual(submission.status, ReviewStatus.PENDING)
    
    def test_add_review(self):
        """Test adding a plugin review."""
        review = self.community.add_review(
            "my-plugin",
            "reviewer1",
            4.5,
            "Great plugin!"
        )
        
        self.assertEqual(review.plugin_id, "my-plugin")
        self.assertEqual(review.rating, 4.5)
        self.assertTrue(review.is_positive())
    
    def test_get_reviews(self):
        """Test retrieving plugin reviews."""
        # Add multiple reviews
        self.community.add_review("my-plugin", "reviewer1", 5.0, "Excellent!")
        self.community.add_review("my-plugin", "reviewer2", 4.0, "Good")
        
        reviews = self.community.get_reviews("my-plugin")
        
        self.assertEqual(len(reviews), 2)
    
    def test_average_rating(self):
        """Test calculating average rating."""
        self.community.add_review("my-plugin", "reviewer1", 5.0, "Excellent!")
        self.community.add_review("my-plugin", "reviewer2", 3.0, "OK")
        
        avg = self.community.get_average_rating("my-plugin")
        
        self.assertEqual(avg, 4.0)
    
    def test_approve_submission(self):
        """Test approving a submission."""
        submission = self.community.submit_plugin("my-plugin", "author1")
        
        result = self.community.approve_submission(submission.submission_id)
        self.assertTrue(result)
        
        # Verify status changed
        self.assertEqual(submission.status, ReviewStatus.APPROVED)
    
    def test_reject_submission(self):
        """Test rejecting a submission."""
        submission = self.community.submit_plugin("my-plugin", "author1")
        
        result = self.community.reject_submission(submission.submission_id, "Quality issues")
        self.assertTrue(result)
        
        # Verify status changed
        self.assertEqual(submission.status, ReviewStatus.REJECTED)
    
    def test_contributor_stats(self):
        """Test getting contributor statistics."""
        # Submit multiple plugins
        self.community.submit_plugin("plugin1", "author1")
        sub2 = self.community.submit_plugin("plugin2", "author1")
        self.community.approve_submission(sub2.submission_id)
        
        stats = self.community.get_contributor_stats("author1")
        
        self.assertEqual(stats['total_submissions'], 2)
        self.assertEqual(stats['approved'], 1)
        self.assertEqual(stats['pending'], 1)
    
    def test_get_contribution_template(self):
        """Test getting contribution template."""
        template = self.community.get_contribution_template()
        
        self.assertIsNotNone(template)
        self.assertGreater(len(template.required_files), 0)
        self.assertGreater(len(template.guidelines), 0)


class TestDetectorPackRegistry(unittest.TestCase):
    """Test DetectorPackRegistry functionality."""
    
    def setUp(self):
        """Set up test registry."""
        self.registry = DetectorPackRegistry()
    
    def test_get_pack(self):
        """Test getting a detector pack."""
        pack = self.registry.get_pack("financial-compliance-v2")
        
        self.assertIsNotNone(pack)
        self.assertEqual(pack.name, "Financial Compliance Pack")
        self.assertGreater(len(pack.detectors), 0)
    
    def test_get_industry_pack(self):
        """Test getting industry-specific packs."""
        industry_pack = self.registry.get_industry_pack(Industry.FINANCIAL)
        
        self.assertIsNotNone(industry_pack)
        self.assertEqual(industry_pack.industry, Industry.FINANCIAL)
        self.assertGreater(len(industry_pack.packs), 0)
    
    def test_search_packs_by_industry(self):
        """Test searching packs by industry."""
        results = self.registry.search_packs(industry=Industry.HEALTHCARE)
        
        self.assertGreater(len(results), 0)
        for pack in results:
            self.assertEqual(pack.industry, Industry.HEALTHCARE)
    
    def test_search_packs_by_tags(self):
        """Test searching packs by tags."""
        results = self.registry.search_packs(tags={"compliance"})
        
        self.assertGreater(len(results), 0)
        for pack in results:
            self.assertIn("compliance", pack.tags)
    
    def test_search_packs_by_use_case(self):
        """Test searching packs by use case."""
        results = self.registry.search_packs(use_case="fraud-detection")
        
        self.assertGreater(len(results), 0)
        for pack in results:
            self.assertIn("fraud-detection", pack.use_cases)
    
    def test_list_all_packs(self):
        """Test listing all detector packs."""
        packs = self.registry.list_packs()
        
        self.assertGreater(len(packs), 0)


class TestIntegrationDirectory(unittest.TestCase):
    """Test IntegrationDirectory functionality."""
    
    def setUp(self):
        """Set up test directory."""
        self.directory = IntegrationDirectory()
    
    def test_list_integrations(self):
        """Test listing available integrations."""
        integrations = self.directory.list_integrations()
        
        self.assertGreater(len(integrations), 0)
    
    def test_list_integrations_by_type(self):
        """Test listing integrations by type."""
        data_sources = self.directory.list_integrations(
            integration_type=IntegrationType.DATA_SOURCE
        )
        
        self.assertGreater(len(data_sources), 0)
        for integration in data_sources:
            self.assertEqual(integration.integration_type, IntegrationType.DATA_SOURCE)
    
    def test_get_integration(self):
        """Test getting specific integration."""
        integration = self.directory.get_integration("postgresql")
        
        self.assertIsNotNone(integration)
        self.assertEqual(integration.name, "PostgreSQL Data Source")
    
    def test_create_adapter(self):
        """Test creating an integration adapter."""
        adapter = self.directory.create_adapter(
            "postgresql",
            {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "user",
                "password": "pass"
            }
        )
        
        self.assertIsNotNone(adapter)
        self.assertTrue(adapter.test_connection())


class TestExportImportUtilities(unittest.TestCase):
    """Test Export and Import utilities."""
    
    def setUp(self):
        """Set up test utilities."""
        self.test_dir = tempfile.mkdtemp()
        self.exporter = ExportUtility()
        self.importer = ImportUtility()
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_export_import_json(self):
        """Test JSON export and import."""
        test_data = {
            "plugin_id": "test-plugin",
            "version": "1.0.0",
            "data": [1, 2, 3]
        }
        
        json_path = str(Path(self.test_dir) / "test.json")
        
        # Export
        result = self.exporter.export_to_json(test_data, json_path)
        self.assertTrue(result)
        
        # Import
        imported_data = self.importer.import_from_json(json_path)
        self.assertEqual(imported_data, test_data)
    
    def test_export_import_csv(self):
        """Test CSV export and import."""
        test_data = [
            {"id": "1", "name": "Plugin A", "rating": "4.5"},
            {"id": "2", "name": "Plugin B", "rating": "4.0"}
        ]
        
        csv_path = str(Path(self.test_dir) / "test.csv")
        
        # Export
        result = self.exporter.export_to_csv(test_data, csv_path)
        self.assertTrue(result)
        
        # Import
        imported_data = self.importer.import_from_csv(csv_path)
        self.assertEqual(len(imported_data), 2)


class TestIntegratedGovernancePlugin(unittest.TestCase):
    """Test IntegratedGovernance plugin loading."""
    
    def test_load_plugin_method_exists(self):
        """Test that load_plugin method exists."""
        from nethical.core import IntegratedGovernance
        
        governance = IntegratedGovernance()
        
        # Verify method exists
        self.assertTrue(hasattr(governance, 'load_plugin'))
        self.assertTrue(callable(getattr(governance, 'load_plugin')))
    
    def test_load_plugin_returns_bool(self):
        """Test that load_plugin returns boolean."""
        from nethical.core import IntegratedGovernance
        
        governance = IntegratedGovernance()
        result = governance.load_plugin("test-plugin")
        
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
