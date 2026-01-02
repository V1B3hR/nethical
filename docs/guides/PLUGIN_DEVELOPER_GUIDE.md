# Detector & Policy Extensibility Developer Guide

## Overview

The Nethical F2: Detector & Policy Extensibility feature enables you to extend the safety governance system with custom detectors and policies without modifying core code.

## Table of Contents

1. [Plugin Architecture](#plugin-architecture)
2. [Creating Custom Detectors](#creating-custom-detectors)
3. [Policy DSL](#policy-dsl)
4. [Plugin Discovery](#plugin-discovery)
5. [Performance Considerations](#performance-considerations)
6. [Examples](#examples)

---

## Plugin Architecture

### DetectorPlugin Base Class

All custom detectors should inherit from `DetectorPlugin`:

```python
from nethical.core.plugin_interface import DetectorPlugin, PluginMetadata
from nethical.detectors.base_detector import SafetyViolation

class MyCustomDetector(DetectorPlugin):
    def __init__(self):
        super().__init__(
            name="MyCustomDetector",
            version="1.0.0"
        )
    
    async def detect_violations(self, action):
        # Your detection logic here
        violations = []
        # ... analyze action ...
        return violations if violations else None
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description="My custom detector",
            author="Your Name",
            requires_nethical_version=">=1.0.0",
            tags={"custom", "example"}
        )
```

### Plugin Manager

The `PluginManager` handles plugin registration and lifecycle:

```python
from nethical.core.plugin_interface import get_plugin_manager

# Get the global plugin manager
plugin_manager = get_plugin_manager()

# Register a plugin
detector = MyCustomDetector()
plugin_manager.register_plugin(detector)

# Run a specific plugin
violations = await plugin_manager.run_plugin("MyCustomDetector", action)

# Run all plugins
all_violations = await plugin_manager.run_all_plugins(action)

# List registered plugins
plugins = plugin_manager.list_plugins()
```

---

## Creating Custom Detectors

### Step 1: Define Your Detector Class

```python
from nethical.core.plugin_interface import DetectorPlugin, PluginMetadata
from nethical.detectors.base_detector import SafetyViolation
import re

class CustomFinancialDetector(DetectorPlugin):
    def __init__(self):
        super().__init__(
            name="CustomFinancialDetector",
            version="1.0.0"
        )
        self.financial_patterns = {
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        }
```

### Step 2: Implement detect_violations Method

```python
    async def detect_violations(self, action):
        violations = []
        
        # Extract content from action
        content = str(action.content) if hasattr(action, 'content') else str(action)
        
        # Check for financial patterns
        for pattern_name, pattern in self.financial_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(SafetyViolation(
                    detector=self.name,
                    severity="high",
                    description=f"Financial data detected: {pattern_name}",
                    category="financial_compliance",
                    explanation="PCI-DSS requires encryption of financial data",
                    confidence=0.9,
                    recommendations=[
                        "Enable encryption for data transmission",
                        "Implement PCI-DSS compliant handling"
                    ],
                    metadata={"pattern": pattern_name}
                ))
        
        return violations if violations else None
```

### Step 3: Provide Metadata

```python
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description="Detects financial compliance violations",
            author="Your Organization",
            requires_nethical_version=">=1.0.0",
            dependencies=[],
            tags={"finance", "compliance", "pci-dss"}
        )
```

### Step 4: Register and Use

```python
# Register the detector
detector = CustomFinancialDetector()
plugin_manager.register_plugin(detector)

# Use it
violations = await plugin_manager.run_plugin("CustomFinancialDetector", action)
```

---

## Policy DSL

### Policy Structure

Policies are defined in YAML or JSON format:

```yaml
policies:
  - name: "policy_name"
    version: "1.0.0"
    enabled: true
    description: "Policy description"
    tags: ["tag1", "tag2"]
    
    rules:
      - condition: "policy condition expression"
        severity: HIGH  # CRITICAL, HIGH, MEDIUM, LOW, INFO
        description: "Rule description"
        actions:
          - action_name
        metadata:
          key: value
```

### Condition Expressions

Conditions are Python expressions evaluated in a safe environment:

**Built-in Functions:**
- `contains(text, substring)` - Case-insensitive substring check
- `matches_regex(text, pattern)` - Regex pattern matching
- `length(text)` - Get length of text
- `upper(text)`, `lower(text)` - Case conversion
- `startswith(text, prefix)`, `endswith(text, suffix)` - String checks

**Action Attributes:**
Available attributes depend on your action object:
- `action.content` - Action content
- `action.context` - Context dictionary
- Any other attributes your action has

**Examples:**

```yaml
# Simple contains check
condition: "contains(action.content, 'financial_data')"

# Regex matching
condition: "matches_regex(action.content, r'\\b\\d{4}-\\d{4}-\\d{4}-\\d{4}\\b')"

# Boolean operators
condition: "contains(action.content, 'patient') AND NOT action.context.get('consent', False)"

# Complex conditions
condition: "(contains(action.content, 'credit_card') OR contains(action.content, 'ssn')) AND NOT action.context.get('encrypted', False)"
```

### Policy Actions

Available actions:
- `block_action` - Block the action
- `audit_log` - Log to audit trail
- `alert` - Raise an alert
- `require_encryption` - Require encryption
- `escalate` - Escalate to supervisor
- `alert_dpo` - Alert Data Protection Officer
- `quarantine` - Quarantine for review
- `notify_user` - Notify the user

### Using the Policy Engine

```python
from nethical.core.policy_dsl import get_policy_engine

# Get the global policy engine
policy_engine = get_policy_engine()

# Load policy from file
policy_engine.load_policy_file("policies/financial_compliance.yaml")

# Evaluate policies
violations = policy_engine.evaluate_policies(action)

# Hot-reload changed policies
reloaded = policy_engine.check_for_updates()

# List all policies
policies = policy_engine.list_policies()

# Rollback a policy
policy_engine.rollback_policy("policy_name", version="1.0.0")
```

### Policy Versioning

Policies support versioning and rollback:

```python
# Add a new version
policy_v2 = Policy(
    name="my_policy",
    version="2.0.0",
    enabled=True,
    rules=[...]
)
policy_engine.add_policy(policy_v2)

# Rollback to previous version
policy_engine.rollback_policy("my_policy")

# Rollback to specific version
policy_engine.rollback_policy("my_policy", version="1.0.0")
```

---

## Plugin Discovery

### Loading Plugins from Files

```python
# Load a single plugin file
loaded_plugins = plugin_manager.load_plugin_from_file("path/to/detector.py")

# Load all plugins from a directory
results = plugin_manager.load_plugins_from_directory("plugins/")

# Discover available plugin files
discovered = plugin_manager.discover_plugins("plugins/")
```

### Plugin File Structure

Create a Python file with DetectorPlugin subclasses:

```python
# my_detector.py
from nethical.core.plugin_interface import DetectorPlugin, PluginMetadata

class Detector1(DetectorPlugin):
    # ... implementation ...
    pass

class Detector2(DetectorPlugin):
    # ... implementation ...
    pass
```

The plugin manager will automatically discover and instantiate all `DetectorPlugin` subclasses.

---

## Performance Considerations

### Health Monitoring

Plugins include built-in health monitoring:

```python
# Health check a specific plugin
is_healthy = await detector.health_check()

# Health check all plugins
health_results = await plugin_manager.health_check_all()
```

### Resource Management

Detectors inherit resource management from BaseDetector:

- **Timeouts**: Set `timeout` in config (default 30s)
- **Rate Limiting**: Set `rate_limit` in config (default 100/min)
- **Circuit Breaker**: Automatic failure protection
- **Memory Limits**: Set `max_memory_mb` in config

```python
detector = MyDetector()
detector.config['timeout'] = 60  # 60 seconds
detector.config['rate_limit'] = 50  # 50 requests per minute
```

### Performance Benchmarking

Monitor plugin performance:

```python
# Get plugin metrics
plugins = plugin_manager.list_plugins()
for name, info in plugins.items():
    metrics = info['metrics']
    print(f"{name}: {metrics['total_runs']} runs, {metrics['success_rate']:.1f}% success")

# Get detector metrics
detector_metrics = detector.metrics
print(f"Average execution time: {detector_metrics.avg_execution_time:.3f}s")
print(f"Success rate: {detector_metrics.success_rate:.1f}%")
```

---

## Examples

### Example 1: Financial Compliance Detector

See `examples/custom_detectors.py` for `FinancialComplianceDetector`:
- Detects credit card numbers, bank accounts
- Checks encryption requirements
- Validates authorization
- Enforces audit logging

### Example 2: Healthcare Compliance Detector

See `examples/custom_detectors.py` for `HealthcareComplianceDetector`:
- Detects Protected Health Information (PHI)
- Checks patient consent
- Enforces HIPAA compliance
- Validates security measures

### Example 3: Custom Policy Detector

See `examples/custom_detectors.py` for `CustomPolicyDetector`:
- Configurable forbidden patterns
- Configurable required patterns
- Organization-specific policies

### Example 4: Loading and Using Plugins

```python
import asyncio
from nethical.core.plugin_interface import get_plugin_manager
from nethical.core.policy_dsl import get_policy_engine

async def main():
    # Get managers
    plugin_manager = get_plugin_manager()
    policy_engine = get_policy_engine()
    
    # Load plugins from directory
    plugin_manager.load_plugins_from_directory("examples/")
    
    # Load policies from files
    policy_engine.load_policy_file("examples/policies/financial_compliance.yaml")
    policy_engine.load_policy_file("examples/policies/healthcare_compliance.json")
    
    # Create a test action
    class TestAction:
        def __init__(self):
            self.content = "Processing credit card 4532-1234-5678-9010"
            self.context = {'encryption_enabled': False}
    
    action = TestAction()
    
    # Run all plugins
    plugin_violations = await plugin_manager.run_all_plugins(action)
    
    # Evaluate policies
    policy_violations = policy_engine.evaluate_policies(action)
    
    # Combine results
    print(f"Plugin violations: {len(sum(plugin_violations.values(), []))}")
    print(f"Policy violations: {len(policy_violations)}")
    
    # List all plugins
    plugins = plugin_manager.list_plugins()
    print(f"\nRegistered plugins: {', '.join(plugins.keys())}")
    
    # List all policies
    policies = policy_engine.list_policies()
    print(f"Loaded policies: {', '.join(policies.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 5: Complete Policy File

See `examples/policies/financial_compliance.yaml` for a complete example with:
- Multiple policies
- Complex conditions
- Multiple rules per policy
- Metadata and tags

---

## Best Practices

### 1. Error Handling

Always handle errors in your detectors:

```python
async def detect_violations(self, action):
    try:
        # Your detection logic
        pass
    except Exception as e:
        logger.error(f"Error in {self.name}: {e}")
        return None  # Return None on error
```

### 2. Performance

- Keep detection logic efficient
- Use regex patterns wisely
- Cache compiled patterns
- Consider async operations for I/O

### 3. Security

- Validate all inputs
- Sanitize patterns before compilation
- Don't execute untrusted code
- Use safe evaluation environments

### 4. Testing

Write comprehensive tests for your detectors:

```python
import pytest

@pytest.mark.asyncio
async def test_my_detector():
    detector = MyCustomDetector()
    
    # Test with violation
    action = create_test_action()
    violations = await detector.detect_violations(action)
    assert len(violations) > 0
    
    # Test without violation
    safe_action = create_safe_action()
    violations = await detector.detect_violations(safe_action)
    assert violations is None or len(violations) == 0
```

### 5. Documentation

Document your detectors:
- What they detect
- Configuration options
- Requirements and dependencies
- Example usage

---

## Troubleshooting

### Plugin Not Loading

1. Check plugin file syntax
2. Verify DetectorPlugin inheritance
3. Check for import errors
4. Review plugin manager logs

### Policy Not Matching

1. Test condition expressions
2. Check action attributes
3. Verify policy is enabled
4. Review policy engine logs

### Performance Issues

1. Check detector metrics
2. Optimize regex patterns
3. Review timeout settings
4. Consider caching strategies

---

## API Reference

### DetectorPlugin

- `__init__(name, version, **kwargs)` - Initialize plugin
- `detect_violations(action)` - Main detection method (abstract)
- `get_metadata()` - Return plugin metadata (abstract)
- `health_check()` - Perform health check

### PluginManager

- `register_plugin(plugin)` - Register a plugin instance
- `unregister_plugin(name)` - Unregister a plugin
- `get_plugin(name)` - Get plugin by name
- `list_plugins()` - List all plugins
- `discover_plugins(directory)` - Discover plugin files
- `load_plugin_from_file(path)` - Load plugin from file
- `load_plugins_from_directory(directory)` - Load all plugins
- `run_plugin(name, action, context)` - Run specific plugin
- `run_all_plugins(action, context)` - Run all plugins
- `health_check_all()` - Health check all plugins

### PolicyEngine

- `load_policy_file(path)` - Load policy from file
- `add_policy(policy)` - Add/update a policy
- `remove_policy(name)` - Remove a policy
- `get_policy(name)` - Get policy by name
- `list_policies()` - List all policies
- `rollback_policy(name, version)` - Rollback to previous version
- `check_for_updates()` - Check for file changes and reload
- `evaluate_policies(action, context)` - Evaluate all policies
- `evaluate_policy(policy, action, context)` - Evaluate single policy

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/docs
- Examples: https://github.com/V1B3hR/nethical/examples
