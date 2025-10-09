# F6: Marketplace & Ecosystem Guide

## Overview

The F6: Marketplace & Ecosystem feature provides a comprehensive plugin marketplace and community ecosystem for the Nethical governance system. This enables discovery, distribution, and management of community-contributed detectors, policies, and integrations.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [API Reference](#api-reference)

## Installation

The marketplace module is included in Nethical 0.1.0+. No additional installation required.

```python
from nethical.marketplace import MarketplaceClient
```

## Quick Start

### Basic Marketplace Usage

```python
from nethical.marketplace import MarketplaceClient

# Initialize marketplace client
marketplace = MarketplaceClient()

# Search for plugins
results = marketplace.search(
    category="financial",
    min_rating=4.0,
    compatible_version=">=0.1.0"
)

# Install a plugin
marketplace.install("financial-compliance-v2", version="1.2.3")

# List installed plugins
installed = marketplace.list_installed()
```

### Using with Integrated Governance

```python
from nethical.core import IntegratedGovernance

# Initialize governance
governance = IntegratedGovernance()

# Load marketplace plugin
governance.load_plugin("financial-compliance-v2")
```

## Core Components

### 1. MarketplaceClient

The central hub for plugin discovery and management.

**Features:**
- Plugin search with multiple filters
- Version management
- Dependency resolution
- Installation and updates
- Local registry caching

**Example:**

```python
from nethical.marketplace import MarketplaceClient, SearchFilters

marketplace = MarketplaceClient(storage_dir="./my_marketplace")

# Advanced search
results = marketplace.search(
    query="fraud detection",
    category="financial",
    min_rating=4.0,
    certified_only=True,
    tags={"compliance", "fraud"}
)

for plugin in results:
    print(f"{plugin.name} - Rating: {plugin.rating}")
    print(f"  Author: {plugin.author}")
    print(f"  Downloads: {plugin.download_count}")
    print(f"  Certified: {plugin.certified}")
```

### 2. PluginGovernance

Security scanning, performance benchmarking, and certification management.

**Features:**
- Security vulnerability scanning
- Performance benchmarking
- Compatibility testing
- Certification process
- Governance reporting

**Example:**

```python
from nethical.marketplace import PluginGovernance

governance = PluginGovernance(strict_mode=True)

# Security scan
scan_result = governance.security_scan(
    "my-plugin",
    plugin_code=plugin_source_code
)

print(f"Security Level: {scan_result.security_level.value}")
print(f"Vulnerabilities: {len(scan_result.vulnerabilities)}")

# Performance benchmark
bench_result = governance.benchmark("my-plugin", iterations=1000)
print(f"Avg Latency: {bench_result.avg_latency_ms:.2f}ms")
print(f"Throughput: {bench_result.throughput_ops_per_sec:.2f} ops/sec")

# Compatibility test
compat_result = governance.compatibility_test("my-plugin")
print(f"Compatible: {compat_result.compatible}")

# Certify plugin
cert_status = governance.certify(
    "my-plugin",
    security_result=scan_result,
    benchmark_result=bench_result,
    compatibility_result=compat_result
)
print(f"Certification: {cert_status.value}")
```

### 3. CommunityManager

Community contributions, reviews, and recognition.

**Features:**
- Plugin submission workflow
- Review and rating system
- Contributor statistics
- Contribution templates

**Example:**

```python
from nethical.marketplace import CommunityManager

community = CommunityManager()

# Submit a plugin
submission = community.submit_plugin(
    plugin_id="my-detector",
    author="username"
)

# Add reviews
community.add_review(
    plugin_id="my-detector",
    reviewer="reviewer1",
    rating=5.0,
    comment="Excellent plugin!"
)

# Get average rating
avg_rating = community.get_average_rating("my-detector")
print(f"Average Rating: {avg_rating:.2f}")

# Approve submission
community.approve_submission(submission.submission_id)

# Get contributor stats
stats = community.get_contributor_stats("username")
print(f"Total submissions: {stats['total_submissions']}")
print(f"Approved: {stats['approved']}")
```

### 4. DetectorPackRegistry

Pre-built detector packs for common use cases.

**Features:**
- Industry-specific bundles
- Use case templates
- Configuration presets
- Best practice implementations

**Example:**

```python
from nethical.marketplace import DetectorPackRegistry, Industry

registry = DetectorPackRegistry()

# Get industry-specific pack
financial_pack = registry.get_industry_pack(Industry.FINANCIAL)
print(f"Financial packs: {len(financial_pack.packs)}")

# Search for packs
results = registry.search_packs(
    industry=Industry.HEALTHCARE,
    tags={"compliance"},
    use_case="patient-data-protection"
)

# Get specific pack
pack = registry.get_pack("financial-compliance-v2")
print(f"Pack: {pack.name}")
print(f"Detectors: {', '.join(pack.detectors)}")
print(f"Configuration: {pack.configuration}")
```

### 5. IntegrationDirectory

Third-party system integrations and data adapters.

**Features:**
- Data source adapters
- API connectors
- Export/import utilities
- Integration registry

**Example:**

```python
from nethical.marketplace import (
    IntegrationDirectory,
    IntegrationType,
    ExportUtility,
    ImportUtility
)

directory = IntegrationDirectory()

# List available integrations
data_sources = directory.list_integrations(
    integration_type=IntegrationType.DATA_SOURCE
)

# Create adapter
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

# Test connection
if adapter.test_connection():
    print("Connection successful")

# Export/Import utilities
exporter = ExportUtility()
importer = ImportUtility()

# Export data
exporter.export_to_json(data, "output.json")
exporter.export_to_csv(data, "output.csv")

# Import data
imported = importer.import_from_json("input.json")
```

## Usage Examples

### Example 1: Finding and Installing Certified Plugins

```python
from nethical.marketplace import MarketplaceClient

marketplace = MarketplaceClient()

# Search for certified financial plugins
plugins = marketplace.search(
    category="financial",
    certified_only=True,
    min_rating=4.5
)

# Install the best-rated one
if plugins:
    best_plugin = plugins[0]  # Results sorted by rating
    print(f"Installing: {best_plugin.name}")
    marketplace.install(best_plugin.plugin_id)
```

### Example 2: Plugin Certification Workflow

```python
from nethical.marketplace import PluginGovernance

governance = PluginGovernance()

# Run all checks
security = governance.security_scan("my-plugin", plugin_code=code)
performance = governance.benchmark("my-plugin")
compatibility = governance.compatibility_test("my-plugin")

# Generate report
report = governance.generate_report(
    "my-plugin",
    security,
    performance,
    compatibility
)

# Check if ready for certification
if (security.passed and 
    performance.passed and 
    compatibility.compatible):
    cert_status = governance.certify("my-plugin")
    print(f"Certification: {cert_status.value}")
```

### Example 3: Community Plugin Submission

```python
from nethical.marketplace import CommunityManager

community = CommunityManager()

# Get contribution template
template = community.get_contribution_template()
print("Required files:")
for file in template.required_files:
    print(f"  - {file}")

print("\nGuidelines:")
for guideline in template.guidelines:
    print(f"  - {guideline}")

# Submit plugin
submission = community.submit_plugin("my-plugin", "my_username")
print(f"Submission ID: {submission.submission_id}")
print(f"Status: {submission.status.value}")
```

### Example 4: Using Detector Packs

```python
from nethical.marketplace import DetectorPackRegistry, Industry

registry = DetectorPackRegistry()

# Get financial compliance pack
pack = registry.get_pack("financial-compliance-v2")

print(f"Pack: {pack.name}")
print(f"Detectors included:")
for detector in pack.detectors:
    print(f"  - {detector}")

print(f"\nConfiguration:")
for key, value in pack.configuration.items():
    print(f"  {key}: {value}")

print(f"\nUse cases:")
for use_case in pack.use_cases:
    print(f"  - {use_case}")
```

### Example 5: Third-Party Integration

```python
from nethical.marketplace import IntegrationDirectory, IntegrationType

directory = IntegrationDirectory()

# List available data sources
data_sources = directory.list_integrations(
    integration_type=IntegrationType.DATA_SOURCE
)

print("Available data sources:")
for ds in data_sources:
    print(f"  - {ds.name}")
    print(f"    Formats: {', '.join(ds.supported_formats)}")

# Create PostgreSQL adapter
pg_adapter = directory.create_adapter(
    "postgresql",
    {
        "host": "db.example.com",
        "port": 5432,
        "database": "production",
        "username": "app_user",
        "password": "***"
    }
)

if pg_adapter.connect():
    # Use adapter
    data = pg_adapter.fetch_data({"query": "SELECT * FROM actions"})
```

## Best Practices

### Plugin Development

1. **Follow Contribution Guidelines**
   - Use the provided templates
   - Include comprehensive documentation
   - Provide unit tests (>80% coverage)
   - Use type hints throughout

2. **Security**
   - Avoid dangerous operations (exec, eval)
   - Validate all inputs
   - Use secure dependencies
   - Follow principle of least privilege

3. **Performance**
   - Optimize for latency (<100ms)
   - Minimize memory usage
   - Avoid blocking operations
   - Use async where appropriate

4. **Compatibility**
   - Specify version requirements clearly
   - Test with multiple Python versions
   - Handle backward compatibility
   - Document breaking changes

### Marketplace Usage

1. **Plugin Selection**
   - Prefer certified plugins
   - Check ratings and reviews
   - Verify compatibility
   - Review dependencies

2. **Updates**
   - Check for updates regularly
   - Review changelogs before updating
   - Test in staging first
   - Have rollback plan

3. **Community Participation**
   - Leave honest reviews
   - Report issues constructively
   - Contribute improvements
   - Share use cases

### Integration

1. **Testing**
   - Test integrations thoroughly
   - Verify data accuracy
   - Check error handling
   - Monitor performance

2. **Security**
   - Use secure connections
   - Encrypt sensitive data
   - Rotate credentials regularly
   - Audit access logs

3. **Maintenance**
   - Monitor integration health
   - Keep adapters updated
   - Document configurations
   - Plan for deprecations

## API Reference

### MarketplaceClient

```python
MarketplaceClient(
    storage_dir: str = "./nethical_marketplace",
    marketplace_url: Optional[str] = None,
    nethical_version: str = "0.1.0"
)
```

**Methods:**
- `search()` - Search for plugins
- `install()` - Install a plugin
- `uninstall()` - Uninstall a plugin
- `list_installed()` - List installed plugins
- `update()` - Update a plugin
- `check_updates()` - Check for available updates
- `register_plugin()` - Register a plugin locally
- `get_plugin_info()` - Get plugin details

### PluginGovernance

```python
PluginGovernance(
    storage_dir: str = "./nethical_governance",
    strict_mode: bool = False
)
```

**Methods:**
- `security_scan()` - Scan for security issues
- `benchmark()` - Benchmark performance
- `compatibility_test()` - Test compatibility
- `certify()` - Certify a plugin
- `get_certification_status()` - Get certification status
- `revoke_certification()` - Revoke certification
- `generate_report()` - Generate governance report

### CommunityManager

```python
CommunityManager(
    storage_dir: str = "./nethical_community"
)
```

**Methods:**
- `submit_plugin()` - Submit a plugin for review
- `add_review()` - Add a plugin review
- `get_reviews()` - Get plugin reviews
- `get_average_rating()` - Get average rating
- `approve_submission()` - Approve a submission
- `reject_submission()` - Reject a submission
- `get_contributor_stats()` - Get contributor statistics
- `get_contribution_template()` - Get contribution template

### DetectorPackRegistry

```python
DetectorPackRegistry()
```

**Methods:**
- `get_pack()` - Get specific pack
- `get_industry_pack()` - Get industry-specific packs
- `search_packs()` - Search for packs
- `register_pack()` - Register a pack
- `register_template()` - Register a use case template
- `get_use_case_template()` - Get use case template
- `list_packs()` - List all packs

### IntegrationDirectory

```python
IntegrationDirectory()
```

**Methods:**
- `register_integration()` - Register an integration
- `create_adapter()` - Create an integration adapter
- `list_integrations()` - List available integrations
- `get_integration()` - Get integration details

## Troubleshooting

### Plugin Installation Issues

**Problem:** Plugin installation fails

**Solutions:**
- Check version compatibility
- Verify dependencies are installed
- Ensure sufficient disk space
- Check network connectivity (if remote)

### Security Scan Failures

**Problem:** Plugin fails security scan

**Solutions:**
- Remove dangerous operations (exec, eval)
- Avoid system calls
- Validate inputs properly
- Use secure dependencies

### Performance Issues

**Problem:** Plugin fails performance benchmarks

**Solutions:**
- Optimize algorithms
- Reduce memory allocations
- Use caching where appropriate
- Profile and identify bottlenecks

### Certification Denied

**Problem:** Plugin certification is denied

**Solutions:**
- Review security scan results
- Improve performance metrics
- Fix compatibility issues
- Address reviewer feedback

## Contributing

To contribute to the marketplace ecosystem:

1. Fork the repository
2. Create a feature branch
3. Follow contribution guidelines
4. Submit a pull request
5. Address review feedback

## Support

- **Documentation**: See F6_IMPLEMENTATION_SUMMARY.md
- **Examples**: See examples/f6_marketplace_demo.py
- **Tests**: See tests/test_f6_marketplace.py
- **Issues**: Report on GitHub

## License

MIT License - See LICENSE file for details
