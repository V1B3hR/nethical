# Plugin Examples

This directory contains example plugins demonstrating the use of the Nethical Plugin Development Kit (PDK).

## Examples

### 1. SQL Injection Detector

A simple detector plugin that identifies potential SQL injection patterns in actions.

**Location**: `sql_injection_detector/`

**Features**:
- Detects common SQL injection patterns
- Categorizes severity based on pattern type
- Provides detailed violation information

**Usage**:
```python
from nethical.core import IntegratedGovernance
from sql_injection_detector import SqlInjectionDetector

gov = IntegratedGovernance()
detector = SqlInjectionDetector()

# Register with governance (manual registration)
# Or use load_plugin for marketplace plugins
```

### 2. PII Leak Detector

A detector plugin that identifies potential personally identifiable information (PII) leaks.

**Location**: `pii_leak_detector/`

**Features**:
- Detects email addresses, phone numbers, SSNs
- Configurable PII patterns
- Integration with redaction pipeline

**Usage**:
```python
from pii_leak_detector import PiiLeakDetector

detector = PiiLeakDetector()
violations = await detector.detect_violations(action)
```

## Creating Your Own Plugin

Use the PDK to scaffold a new plugin:

```bash
# Create a new detector plugin
python scripts/nethical-pdk.py init \
  --name "MyCustomDetector" \
  --type detector \
  --description "My custom detector" \
  --author "Your Name" \
  --output examples/plugins

# Navigate to the plugin
cd examples/plugins/mycustomdetector

# Implement your detection logic
# Edit mycustomdetector.py

# Test your plugin
python ../../../scripts/nethical-pdk.py test .

# Validate plugin structure
python ../../../scripts/nethical-pdk.py validate .

# Package for distribution
python ../../../scripts/nethical-pdk.py package .
```

## Testing Plugins

All example plugins include comprehensive tests:

```bash
# Run tests for a specific plugin
cd sql_injection_detector
pytest tests/

# Run all plugin tests
cd examples/plugins
pytest */tests/
```

## Documentation

Each plugin includes:
- `README.md` - Usage and configuration guide
- `plugin.json` - Plugin metadata
- `docs/API.md` - API documentation

## Contributing

To contribute an example plugin:

1. Create your plugin using the PDK
2. Ensure all tests pass
3. Document usage and features
4. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
