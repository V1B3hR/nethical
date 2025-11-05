# Plugin Development Kit (PDK) Guide

## Overview

The Nethical Plugin Development Kit (PDK) is a comprehensive CLI tool that helps developers create, test, validate, package, and distribute plugins for Nethical. It streamlines the entire plugin development lifecycle with scaffolding, testing frameworks, and quality checks.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Commands Reference](#commands-reference)
4. [Plugin Structure](#plugin-structure)
5. [Development Workflow](#development-workflow)
6. [Best Practices](#best-practices)
7. [Publishing Plugins](#publishing-plugins)

---

## Installation

The PDK is included with Nethical and is available as a script:

```bash
# Make sure Nethical is installed
pip install nethical

# Or from source
cd nethical
pip install -e .

# The PDK script is located at
scripts/nethical-pdk.py
```

For convenience, you can create an alias:

```bash
# Add to your ~/.bashrc or ~/.zshrc
alias nethical-pdk="python /path/to/nethical/scripts/nethical-pdk.py"
```

## Quick Start

### Creating Your First Plugin

```bash
# Initialize a new detector plugin
python scripts/nethical-pdk.py init \
  --name "MySecurityDetector" \
  --type detector \
  --description "Detects security violations" \
  --author "Your Name"

# This creates a complete plugin structure:
# mysecuritydetector/
# ├── mysecuritydetector.py      # Main plugin code
# ├── tests/
# │   └── test_mysecuritydetector.py  # Unit tests
# ├── docs/                       # Documentation
# ├── plugin.json                 # Plugin metadata
# ├── README.md                   # Plugin documentation
# ├── setup.py                    # Python package setup
# ├── requirements.txt            # Dependencies
# └── .gitignore                  # Git ignore rules
```

### Implementing Your Plugin

Edit `mysecuritydetector.py` to add your detection logic:

```python
async def detect_violations(self, action: Any) -> List[SafetyViolation]:
    """
    Detect violations in the given action.
    """
    violations = []
    
    # Example: Detect SQL injection attempts
    if any(keyword in str(action).lower() for keyword in ['drop table', 'union select', '-- ']):
        violations.append(SafetyViolation(
            detector=self.name,
            severity="critical",
            description="Potential SQL injection detected",
            category="security",
            details={"pattern": "sql_injection", "action": str(action)[:100]}
        ))
    
    return violations
```

### Testing Your Plugin

```bash
# Navigate to plugin directory
cd mysecuritydetector

# Run tests
python ../scripts/nethical-pdk.py test .

# Or use pytest directly
pytest tests/
```

### Validating Your Plugin

```bash
# Validate plugin structure and metadata
python scripts/nethical-pdk.py validate mysecuritydetector

# Output:
# ✓ Found plugin.json
# ✓ Found README.md
# ✓ Manifest field 'name': MySecurityDetector
# ✓ Entry point module exists
# ✓ Found test files
# ✅ Plugin validation passed!
```

### Packaging Your Plugin

```bash
# Package plugin for distribution
python scripts/nethical-pdk.py package mysecuritydetector

# This creates:
# - dist/mysecuritydetector-0.1.0.tar.gz  (source distribution)
# - dist/mysecuritydetector-0.1.0-py3-none-any.whl  (wheel)
```

## Commands Reference

### `init` - Initialize a New Plugin

Create a new plugin with complete scaffolding.

```bash
nethical-pdk init --name PLUGIN_NAME [OPTIONS]
```

**Options:**
- `--name NAME` (required): Plugin name
- `--type TYPE`: Plugin type (detector, policy, analyzer) [default: detector]
- `--output DIR`: Output directory [default: current directory]
- `--description DESC`: Plugin description
- `--author AUTHOR`: Plugin author name

**Example:**

```bash
python scripts/nethical-pdk.py init \
  --name "FinancialComplianceDetector" \
  --type detector \
  --description "Detects financial compliance violations" \
  --author "Jane Smith" \
  --output ./plugins
```

### `validate` - Validate a Plugin

Validate plugin structure, manifest, and files.

```bash
nethical-pdk validate PLUGIN_PATH
```

**Checks:**
- Required files exist (plugin.json, README.md)
- Manifest has all required fields
- Entry point module exists
- Test files are present
- Valid JSON in plugin.json

**Example:**

```bash
python scripts/nethical-pdk.py validate ./mysecuritydetector
```

### `test` - Run Plugin Tests

Execute plugin test suite using pytest.

```bash
nethical-pdk test PLUGIN_PATH
```

**Requirements:**
- pytest and pytest-asyncio must be installed
- Tests must be in `tests/` directory
- Test files must start with `test_`

**Example:**

```bash
python scripts/nethical-pdk.py test ./mysecuritydetector
```

### `package` - Package a Plugin

Build distribution packages for the plugin.

```bash
nethical-pdk package PLUGIN_PATH [--output DIR]
```

**Options:**
- `--output DIR`: Output directory for packages [default: plugin_path/dist]

**Output:**
- Source distribution (.tar.gz)
- Wheel distribution (.whl)

**Example:**

```bash
python scripts/nethical-pdk.py package ./mysecuritydetector --output ./releases
```

### `docs` - Generate Plugin Documentation

Generate API documentation for the plugin.

```bash
nethical-pdk docs PLUGIN_PATH
```

**Generates:**
- `docs/API.md`: API documentation based on plugin metadata

**Example:**

```bash
python scripts/nethical-pdk.py docs ./mysecuritydetector
```

## Plugin Structure

### Generated Files

When you run `nethical-pdk init`, the following structure is created:

```
plugin_name/
├── plugin_name.py              # Main plugin implementation
├── tests/
│   └── test_plugin_name.py    # Unit tests with examples
├── docs/
│   └── API.md                  # Auto-generated API docs
├── plugin.json                 # Plugin manifest
├── README.md                   # Plugin documentation
├── setup.py                    # Python package configuration
├── requirements.txt            # Dependencies
├── __init__.py                 # Package initialization
└── .gitignore                  # Git ignore rules
```

### plugin.json - Manifest File

The manifest contains plugin metadata:

```json
{
    "name": "MySecurityDetector",
    "version": "0.1.0",
    "description": "Detects security violations",
    "author": "Your Name",
    "type": "detector",
    "entry_point": "mysecuritydetector.Mysecuritydetector",
    "requires_nethical_version": ">=0.1.0",
    "dependencies": [],
    "tags": ["detector"],
    "created_at": "2025-11-05T09:00:00Z",
    "license": "MIT",
    "homepage": "",
    "repository": ""
}
```

### Required Methods

All detector plugins must implement:

1. **`detect_violations(action)`** - Main detection logic
2. **`get_metadata()`** - Return plugin metadata
3. **`health_check()`** - Health status check (optional but recommended)

## Development Workflow

### 1. Initialize Plugin

```bash
python scripts/nethical-pdk.py init --name MyPlugin --type detector --author "Your Name"
cd myplugin
```

### 2. Implement Detection Logic

Edit `myplugin.py`:

```python
async def detect_violations(self, action: Any) -> List[SafetyViolation]:
    violations = []
    
    # Your detection logic here
    if self._is_violation(action):
        violations.append(SafetyViolation(
            detector=self.name,
            severity="high",
            description="Violation detected",
            category="security"
        ))
    
    return violations
```

### 3. Write Tests

Edit `tests/test_myplugin.py`:

```python
@pytest.mark.asyncio
async def test_myplugin_detects_violation():
    plugin = Myplugin()
    
    action = AgentAction(
        agent_id="test",
        action="malicious action",
        timestamp="2025-01-01T00:00:00Z"
    )
    
    violations = await plugin.detect_violations(action)
    assert len(violations) > 0
    assert violations[0].severity == "high"
```

### 4. Test Locally

```bash
# Run tests
python ../scripts/nethical-pdk.py test .

# Validate structure
python ../scripts/nethical-pdk.py validate .
```

### 5. Test Integration

```python
from nethical.core import IntegratedGovernance
from myplugin import Myplugin

# Load your plugin
gov = IntegratedGovernance()
plugin = Myplugin()

# Test with real actions
result = gov.process_action(
    agent_id="test_agent",
    action="test action",
    cohort="test"
)
```

### 6. Package and Distribute

```bash
# Package plugin
python ../scripts/nethical-pdk.py package .

# Install locally for testing
pip install dist/myplugin-0.1.0-py3-none-any.whl

# Publish to PyPI (optional)
twine upload dist/*
```

## Best Practices

### 1. Performance

- Keep detection logic fast (< 10ms per action)
- Use async operations for I/O
- Cache expensive computations
- Implement proper timeouts

```python
import asyncio

async def detect_violations(self, action: Any) -> List[SafetyViolation]:
    # Use timeout to prevent hanging
    try:
        return await asyncio.wait_for(
            self._detect_internal(action),
            timeout=0.1  # 100ms timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Detection timeout for {self.name}")
        return []
```

### 2. Error Handling

- Always handle exceptions gracefully
- Log errors appropriately
- Return empty list on errors (don't crash)

```python
async def detect_violations(self, action: Any) -> List[SafetyViolation]:
    try:
        violations = []
        # Detection logic
        return violations
    except Exception as e:
        logger.error(f"Detection error in {self.name}: {e}")
        return []  # Don't crash the system
```

### 3. Testing

- Test normal cases
- Test edge cases
- Test error conditions
- Aim for >80% code coverage

```python
@pytest.mark.asyncio
async def test_plugin_handles_malformed_input():
    plugin = MyPlugin()
    
    # Test with None
    result = await plugin.detect_violations(None)
    assert result == []
    
    # Test with invalid type
    result = await plugin.detect_violations(12345)
    assert result == []
```

### 4. Documentation

- Document all public methods
- Provide usage examples
- Explain detection logic
- List known limitations

### 5. Versioning

Follow semantic versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

Update `plugin.json` version on changes.

## Publishing Plugins

### 1. To Plugin Marketplace

```python
from nethical.marketplace import PluginRegistry

registry = PluginRegistry("./plugin_registry")

# Register your plugin
registration = PluginRegistration(
    plugin_id="my-plugin-id",
    name="MyPlugin",
    version="0.1.0",
    author="Your Name",
    description="Plugin description",
    entry_point="myplugin.Myplugin",
    plugin_type="detector",
    homepage="https://github.com/yourusername/myplugin",
    repository="https://github.com/yourusername/myplugin",
    license="MIT"
)

registry.register_plugin(registration)
```

### 2. To PyPI

```bash
# Build distributions
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

### 3. To GitHub

```bash
# Create release
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# Attach distributions to GitHub release
# (Use GitHub UI or gh CLI)
```

## Security Considerations

### Code Signing

Sign your plugin packages:

```bash
# Generate GPG signature
gpg --detach-sign -a dist/myplugin-0.1.0.tar.gz

# Users can verify:
gpg --verify dist/myplugin-0.1.0.tar.gz.asc dist/myplugin-0.1.0.tar.gz
```

### Security Scanning

Before publishing, scan for vulnerabilities:

```bash
# Scan dependencies
pip-audit

# Scan code
bandit -r myplugin/

# Check for secrets
trufflehog filesystem .
```

## Troubleshooting

### Plugin Not Loading

**Check:**
1. Entry point in plugin.json matches class name
2. All dependencies are installed
3. Plugin inherits from DetectorPlugin
4. No syntax errors in plugin code

### Tests Failing

**Check:**
1. pytest and pytest-asyncio installed
2. Test files start with `test_`
3. Test functions marked with `@pytest.mark.asyncio`
4. All imports are correct

### Validation Errors

**Check:**
1. plugin.json is valid JSON
2. All required fields present in manifest
3. Entry point module exists
4. README.md exists

## Examples

See the `examples/plugins/` directory for complete plugin examples:

- `security_detector/` - Security violation detector
- `compliance_checker/` - Compliance rule checker
- `custom_analyzer/` - Custom action analyzer

## Support

For help with plugin development:

- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/docs
- Plugin Developer Guide: docs/PLUGIN_DEVELOPER_GUIDE.md

---

**Last Updated:** November 5, 2025  
**PDK Version:** 1.0.0
