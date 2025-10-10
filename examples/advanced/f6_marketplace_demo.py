"""F6: Marketplace & Ecosystem Demo

This demo showcases all F6 marketplace features including:
- Plugin marketplace client (search, install, manage)
- Plugin governance (security scanning, benchmarking, certification)
- Community contributions (reviews, submissions)
- Detector packs (industry-specific bundles)
- Integration directory (third-party systems)
"""

from nethical.marketplace import (
    MarketplaceClient,
    PluginInfo,
    PluginVersion,
    PluginGovernance,
    CommunityManager,
    DetectorPackRegistry,
    Industry,
    IntegrationDirectory,
    IntegrationType,
    ExportUtility,
    ImportUtility
)
from nethical.core import IntegratedGovernance
from datetime import datetime
import tempfile
import shutil


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_marketplace_client():
    """Demonstrate marketplace client functionality."""
    print_section("1. Marketplace Client Demo")
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize marketplace client
        marketplace = MarketplaceClient(storage_dir=temp_dir)
        print("✓ Marketplace client initialized")
        
        # Register sample plugins
        financial_plugin = PluginInfo(
            plugin_id="financial-compliance-v2",
            name="Financial Compliance Pack",
            description="Comprehensive financial compliance and fraud detection",
            author="Nethical Team",
            category="financial",
            rating=4.8,
            download_count=1250,
            tags={"financial", "compliance", "fraud", "certified"},
            versions=[
                PluginVersion(
                    version="1.2.3",
                    release_date=datetime.now(),
                    compatibility=">=0.1.0",
                    changelog="Bug fixes and performance improvements"
                )
            ],
            latest_version="1.2.3",
            dependencies=[],
            certified=True
        )
        marketplace.register_plugin(financial_plugin)
        print("✓ Registered: Financial Compliance Pack")
        
        healthcare_plugin = PluginInfo(
            plugin_id="healthcare-hipaa",
            name="Healthcare HIPAA Compliance",
            description="HIPAA compliance detectors for healthcare applications",
            author="MedTech Corp",
            category="healthcare",
            rating=4.5,
            download_count=850,
            tags={"healthcare", "hipaa", "compliance", "certified"},
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
        marketplace.register_plugin(healthcare_plugin)
        print("✓ Registered: Healthcare HIPAA Compliance")
        
        # Search for plugins
        print("\n--- Search Examples ---")
        
        # Search by category
        results = marketplace.search(category="financial")
        print(f"✓ Category 'financial': {len(results)} plugins found")
        for plugin in results:
            print(f"  - {plugin.name} (Rating: {plugin.rating})")
        
        # Search by minimum rating
        results = marketplace.search(min_rating=4.0)
        print(f"\n✓ Min rating 4.0: {len(results)} plugins found")
        
        # Search certified plugins only
        results = marketplace.search(certified_only=True)
        print(f"✓ Certified only: {len(results)} certified plugins")
        
        # Search by tags
        results = marketplace.search(tags={"compliance"})
        print(f"✓ Tag 'compliance': {len(results)} plugins found")
        
        # Install a plugin
        print("\n--- Installation ---")
        status = marketplace.install("financial-compliance-v2")
        print(f"✓ Installed: financial-compliance-v2 (Status: {status.value})")
        
        # List installed plugins
        installed = marketplace.list_installed()
        print(f"✓ Total installed plugins: {len(installed)}")
        for plugin in installed:
            print(f"  - {plugin.name} v{plugin.latest_version}")
        
        # Get plugin info
        info = marketplace.get_plugin_info("financial-compliance-v2")
        if info:
            print(f"\n--- Plugin Details ---")
            print(f"Name: {info.name}")
            print(f"Author: {info.author}")
            print(f"Rating: {info.rating}")
            print(f"Downloads: {info.download_count}")
            print(f"Certified: {info.certified}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_plugin_governance():
    """Demonstrate plugin governance features."""
    print_section("2. Plugin Governance Demo")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        governance = PluginGovernance(storage_dir=temp_dir)
        print("✓ Plugin governance initialized")
        
        # Security scanning
        print("\n--- Security Scanning ---")
        safe_code = """
def detect_violations(action):
    violations = []
    if "spam" in action.content:
        violations.append("spam detected")
    return violations
"""
        scan_result = governance.security_scan(
            "test-plugin",
            plugin_code=safe_code
        )
        print(f"✓ Security scan completed")
        print(f"  Security level: {scan_result.security_level.value}")
        print(f"  Passed: {scan_result.passed}")
        print(f"  Vulnerabilities: {len(scan_result.vulnerabilities)}")
        print(f"  Warnings: {len(scan_result.warnings)}")
        print(f"  Scan duration: {scan_result.scan_duration:.3f}s")
        
        # Performance benchmarking
        print("\n--- Performance Benchmarking ---")
        bench_result = governance.benchmark("test-plugin", iterations=100)
        print(f"✓ Benchmark completed")
        print(f"  Avg latency: {bench_result.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {bench_result.p95_latency_ms:.2f}ms")
        print(f"  P99 latency: {bench_result.p99_latency_ms:.2f}ms")
        print(f"  Throughput: {bench_result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"  Memory: {bench_result.memory_usage_mb:.2f}MB")
        print(f"  CPU: {bench_result.cpu_usage_percent:.2f}%")
        print(f"  Passed requirements: {bench_result.passed}")
        
        # Compatibility testing
        print("\n--- Compatibility Testing ---")
        compat_result = governance.compatibility_test("test-plugin")
        print(f"✓ Compatibility test completed")
        print(f"  Compatible: {compat_result.compatible}")
        print(f"  Nethical version: {compat_result.nethical_version}")
        print(f"  Python version: {compat_result.python_version}")
        print(f"  Tests passed: {sum(compat_result.test_results.values())}/{len(compat_result.test_results)}")
        
        # Certification
        print("\n--- Certification ---")
        cert_status = governance.certify(
            "test-plugin",
            security_result=scan_result,
            benchmark_result=bench_result,
            compatibility_result=compat_result
        )
        print(f"✓ Certification status: {cert_status.value}")
        
        # Generate comprehensive report
        print("\n--- Governance Report ---")
        report = governance.generate_report(
            "test-plugin",
            scan_result,
            bench_result,
            compat_result
        )
        print(f"✓ Report generated")
        print(f"  Security passed: {report['security']['passed']}")
        print(f"  Performance passed: {report['performance']['passed']}")
        print(f"  Compatibility: {report['compatibility']['compatible']}")
        print(f"  Certification: {report['certification_status']}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_community_manager():
    """Demonstrate community management features."""
    print_section("3. Community Manager Demo")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        community = CommunityManager(storage_dir=temp_dir)
        print("✓ Community manager initialized")
        
        # Submit a plugin
        print("\n--- Plugin Submission ---")
        submission = community.submit_plugin("my-awesome-plugin", "john_dev")
        print(f"✓ Plugin submitted")
        print(f"  Submission ID: {submission.submission_id}")
        print(f"  Author: {submission.author}")
        print(f"  Status: {submission.status.value}")
        
        # Add reviews
        print("\n--- Reviews & Ratings ---")
        review1 = community.add_review(
            "my-awesome-plugin",
            "reviewer1",
            5.0,
            "Excellent plugin! Works perfectly."
        )
        print(f"✓ Review added: {review1.rating} stars")
        
        review2 = community.add_review(
            "my-awesome-plugin",
            "reviewer2",
            4.5,
            "Very good, minor improvements needed."
        )
        print(f"✓ Review added: {review2.rating} stars")
        
        review3 = community.add_review(
            "my-awesome-plugin",
            "reviewer3",
            4.0,
            "Solid plugin, does what it promises."
        )
        print(f"✓ Review added: {review3.rating} stars")
        
        # Get reviews
        reviews = community.get_reviews("my-awesome-plugin")
        print(f"\n✓ Total reviews: {len(reviews)}")
        
        # Calculate average rating
        avg_rating = community.get_average_rating("my-awesome-plugin")
        print(f"✓ Average rating: {avg_rating:.2f} stars")
        
        # Approve submission
        print("\n--- Submission Review ---")
        community.approve_submission(submission.submission_id)
        print(f"✓ Submission approved")
        print(f"  New status: {submission.status.value}")
        
        # Contributor statistics
        print("\n--- Contributor Stats ---")
        stats = community.get_contributor_stats("john_dev")
        print(f"✓ Contributor: {stats['author']}")
        print(f"  Total submissions: {stats['total_submissions']}")
        print(f"  Approved: {stats['approved']}")
        print(f"  Pending: {stats['pending']}")
        
        # Get contribution template
        print("\n--- Contribution Template ---")
        template = community.get_contribution_template()
        print(f"✓ Template: {template.name}")
        print(f"  Required files: {', '.join(template.required_files)}")
        print(f"  Guidelines: {len(template.guidelines)} items")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_detector_packs():
    """Demonstrate detector packs features."""
    print_section("4. Detector Packs Demo")
    
    registry = DetectorPackRegistry()
    print("✓ Detector pack registry initialized")
    
    # Get specific pack
    print("\n--- Individual Packs ---")
    pack = registry.get_pack("financial-compliance-v2")
    if pack:
        print(f"✓ Pack: {pack.name}")
        print(f"  Description: {pack.description}")
        print(f"  Detectors: {', '.join(pack.detectors)}")
        print(f"  Industry: {pack.industry.value if pack.industry else 'N/A'}")
        print(f"  Use cases: {len(pack.use_cases)}")
    
    # Get industry-specific packs
    print("\n--- Industry Packs ---")
    industries = [Industry.FINANCIAL, Industry.HEALTHCARE, Industry.LEGAL]
    
    for industry in industries:
        industry_pack = registry.get_industry_pack(industry)
        if industry_pack:
            print(f"✓ {industry.value.upper()}: {len(industry_pack.packs)} packs available")
            for pack in industry_pack.packs:
                print(f"  - {pack.name}")
    
    # Search packs
    print("\n--- Pack Search ---")
    
    # By industry
    results = registry.search_packs(industry=Industry.FINANCIAL)
    print(f"✓ Financial packs: {len(results)}")
    
    # By tags
    results = registry.search_packs(tags={"compliance"})
    print(f"✓ Compliance-related packs: {len(results)}")
    
    # By use case
    results = registry.search_packs(use_case="fraud-detection")
    print(f"✓ Fraud detection packs: {len(results)}")
    
    # List all packs
    print("\n--- All Available Packs ---")
    all_packs = registry.list_packs()
    print(f"✓ Total packs: {len(all_packs)}")
    for pack in all_packs:
        print(f"  - {pack.pack_id}: {pack.name}")


def demo_integration_directory():
    """Demonstrate integration directory features."""
    print_section("5. Integration Directory Demo")
    
    directory = IntegrationDirectory()
    print("✓ Integration directory initialized")
    
    # List all integrations
    print("\n--- Available Integrations ---")
    integrations = directory.list_integrations()
    print(f"✓ Total integrations: {len(integrations)}")
    for integration in integrations:
        print(f"  - {integration.name} ({integration.integration_type.value})")
    
    # List by type
    print("\n--- Data Source Integrations ---")
    data_sources = directory.list_integrations(
        integration_type=IntegrationType.DATA_SOURCE
    )
    print(f"✓ Available data sources: {len(data_sources)}")
    for ds in data_sources:
        print(f"  - {ds.name}")
        print(f"    Formats: {', '.join(ds.supported_formats)}")
    
    # Get specific integration
    print("\n--- Integration Details ---")
    pg_integration = directory.get_integration("postgresql")
    if pg_integration:
        print(f"✓ Integration: {pg_integration.name}")
        print(f"  Type: {pg_integration.integration_type.value}")
        print(f"  Version: {pg_integration.version}")
        print(f"  Supported formats: {', '.join(pg_integration.supported_formats)}")
    
    # Create adapter
    print("\n--- Create Adapter ---")
    adapter = directory.create_adapter(
        "postgresql",
        {
            "host": "localhost",
            "port": 5432,
            "database": "nethical_db",
            "username": "admin",
            "password": "secure_pass"
        }
    )
    if adapter:
        print(f"✓ Adapter created: {adapter.integration_id}")
        print(f"  Connection test: {'Passed' if adapter.test_connection() else 'Failed'}")
    
    # Export/Import utilities
    print("\n--- Export/Import Utilities ---")
    
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = ExportUtility()
        importer = ImportUtility()
        
        # Test data
        test_data = {
            "plugin_id": "test-plugin",
            "version": "1.0.0",
            "metrics": {"rating": 4.5, "downloads": 100}
        }
        
        # Export to JSON
        json_path = f"{temp_dir}/test_export.json"
        exporter.export_to_json(test_data, json_path)
        print(f"✓ Exported to JSON: {json_path}")
        
        # Import from JSON
        imported_data = importer.import_from_json(json_path)
        print(f"✓ Imported from JSON: {imported_data['plugin_id']}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_integrated_governance():
    """Demonstrate integration with governance system."""
    print_section("6. Integrated Governance Demo")
    
    # Initialize integrated governance
    governance = IntegratedGovernance()
    print("✓ Integrated governance initialized")
    
    # Load plugin via marketplace
    print("\n--- Plugin Loading ---")
    result = governance.load_plugin("financial-compliance-v2")
    print(f"✓ Plugin load result: {result}")
    print(f"  Note: This integrates marketplace plugins with the governance system")
    
    # Check system status
    print("\n--- System Status ---")
    status = governance.get_system_status()
    print(f"✓ Governance system operational")
    print(f"  Components enabled: {len(status['components_enabled'])}")


def main():
    """Run all F6 marketplace demos."""
    print("\n" + "="*60)
    print("  F6: MARKETPLACE & ECOSYSTEM DEMO")
    print("  Comprehensive demonstration of all marketplace features")
    print("="*60)
    
    try:
        # Run all demos
        demo_marketplace_client()
        demo_plugin_governance()
        demo_community_manager()
        demo_detector_packs()
        demo_integration_directory()
        demo_integrated_governance()
        
        # Summary
        print_section("Demo Complete ✓")
        print("All F6 marketplace features demonstrated successfully!")
        print("\nKey Features:")
        print("  ✓ Plugin marketplace (search, install, manage)")
        print("  ✓ Plugin governance (security, performance, certification)")
        print("  ✓ Community contributions (reviews, submissions)")
        print("  ✓ Detector packs (industry-specific bundles)")
        print("  ✓ Integration directory (third-party systems)")
        print("  ✓ Integrated with governance system")
        
        print("\nFor more information, see:")
        print("  - F6_GUIDE.md: Comprehensive usage guide")
        print("  - F6_IMPLEMENTATION_SUMMARY.md: Implementation details")
        print("  - tests/test_f6_marketplace.py: Test suite (39 tests)")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
