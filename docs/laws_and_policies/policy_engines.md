# Policy Engine Documentation

## Overview

Nethical provides two complementary policy engines, each designed for specific use cases:

1. **Policy DSL Engine** (`nethical.core.policy_dsl`) - Rule-based policy specification
2. **Region-Aware Policy Engine** (`nethical.policy.engine`) - Advanced regional compliance

## When to Use Each Engine

### Policy DSL Engine (`policy_dsl.py`)

**Best for:**
- Application-level policy enforcement
- Custom detection rules without code changes
- Hot-reloadable policy changes
- Simple rule-based decisions
- Development and testing

**Key Features:**
- YAML/JSON-based policy specification
- Rule versioning and rollback
- Policy hot-reload
- Compiled rule engine for performance
- Action-based policies (block, audit, alert, etc.)

**Example Use Case:**
```yaml
policies:
  - name: "financial_compliance"
    version: "1.0.0"
    enabled: true
    rules:
      - condition: "action.context.contains('financial_data')"
        severity: HIGH
        actions:
          - "require_encryption"
          - "audit_log"
```

### Region-Aware Policy Engine (`policy/engine.py`)

**Best for:**
- Regional compliance (GDPR, CCPA, etc.)
- Multi-region deployments
- Complex boolean logic with nested conditions
- Rich metadata and rule prioritization
- Production governance systems

**Key Features:**
- Deep merge of region overlays
- Rich boolean condition DSL (all/any/not)
- Advanced operators (matches, contains, startswith, etc.)
- Rule metadata (id, priority, enabled, halt_on_match)
- Deterministic decision strategy (deny-overrides)
- List-aware evaluation

**Example Use Case:**
```yaml
defaults:
  decision: ALLOW
  deny_overrides: true

rules:
  - id: gdpr-compliance
    enabled: true
    priority: 100
    when:
      all:
        - "region == 'EU'"
        - "data.contains_pii == true"
    action:
      decision: RESTRICT
      add_disclaimer: "GDPR compliance required"
      tags: ["gdpr", "privacy"]

region_overlays:
  EU:
    defaults:
      decision: RESTRICT
```

## Choosing Between Engines

| Criteria | Policy DSL | Region-Aware Engine |
|----------|-----------|-------------------|
| **Simplicity** | ✅ Simple YAML | ⚠️ More complex |
| **Regional Support** | ❌ Not built-in | ✅ Native support |
| **Boolean Logic** | ⚠️ Basic | ✅ Advanced (all/any/not) |
| **Operators** | ⚠️ Limited | ✅ Extensive |
| **Hot Reload** | ✅ Yes | ⚠️ Requires restart |
| **Performance** | ✅ Compiled | ✅ Optimized |
| **Use Case** | App-level rules | Enterprise governance |

## Integration Patterns

### Pattern 1: Unified Approach
Use **Region-Aware Engine** as the primary policy engine for all governance decisions.

```python
from nethical.policy.engine import PolicyEngine
from nethical.hooks.interfaces import Region

# Load policy with region support
engine = PolicyEngine.load("policies/governance.yaml", Region.EU)

# Evaluate
result = engine.evaluate(facts)
```

### Pattern 2: Layered Approach
Use **Policy DSL** for application rules and **Region-Aware Engine** for governance.

```python
from nethical.core.policy_dsl import PolicyDSL
from nethical.policy.engine import PolicyEngine

# Application-level rules
app_policy = PolicyDSL.load("policies/app_rules.yaml")

# Governance rules with region awareness
gov_engine = PolicyEngine.load("policies/governance.yaml", Region.US)

# Evaluate both
app_result = app_policy.evaluate(action)
gov_result = gov_engine.evaluate(facts)
```

### Pattern 3: Migration Path
Migrate from **Policy DSL** to **Region-Aware Engine** for enhanced capabilities.

```python
# Phase 1: Use Policy DSL (current)
from nethical.core.policy_dsl import PolicyDSL
policy = PolicyDSL.load("policies/rules.yaml")

# Phase 2: Migrate to Region-Aware Engine (future)
from nethical.policy.engine import PolicyEngine
engine = PolicyEngine.load("policies/rules_v2.yaml", Region.GLOBAL)
```

## Policy Language Specification

### EBNF Grammar

Both engines support a structured policy language. See `nethical.core.policy_formalization.PolicyGrammarEBNF` for the complete EBNF specification.

Key elements:
- **Defaults**: Global configuration
- **Rules**: List of policy rules with conditions and actions
- **Conditions**: Boolean expressions with operators
- **Actions**: Decisions and side effects
- **Region Overlays**: Regional overrides (Region-Aware only)

### Validation

Use the policy validator to check your policies:

```python
from nethical.core.policy_formalization import PolicyValidator, PolicyEngineType

# Validate Region-Aware policy
validator = PolicyValidator(PolicyEngineType.REGION_AWARE)
result = validator.validate_policy(policy_config)

if not result.valid:
    print("Errors:", result.errors)
    print("Warnings:", result.warnings)
```

### Linting

Lint your policies for best practices:

```python
suggestions = validator.lint_policy(policy_config)
for suggestion in suggestions:
    print(suggestion)
```

## Policy Simulation and Testing

Before deploying policy changes, simulate their impact:

```python
from nethical.core.policy_formalization import PolicySimulator, PolicyImpactAnalyzer

# Simulate policy
simulator = PolicySimulator()
results = simulator.simulate_policy(new_policy, test_cases)

print(f"Block rate: {results['decision_percentages']['BLOCK']:.1f}%")
print(f"Most matched rule: {max(results['rule_matches'].items(), key=lambda x: x[1])}")

# Analyze impact
analyzer = PolicyImpactAnalyzer()
impact = analyzer.analyze_impact(current_policy, new_policy, historical_data)

print(f"Risk level: {impact.risk_level}")
print(f"Block rate delta: {impact.simulation_results['comparison']['block_rate_delta']:.2%}")
for rec in impact.recommendations:
    print(f"- {rec}")
```

## Best Practices

1. **Start Simple**: Begin with Policy DSL for application rules
2. **Add Regions**: Migrate to Region-Aware Engine when regional compliance is needed
3. **Version Everything**: Always include version fields in policies
4. **Test Thoroughly**: Use simulation before production deployment
5. **Document Rules**: Add descriptions in rule metadata
6. **Monitor Impact**: Track policy changes with impact analysis
7. **Use Priorities**: Set appropriate rule priorities for deterministic evaluation
8. **Leverage Overlays**: Use region overlays instead of duplicating rules

## Migration Guide

### From Policy DSL to Region-Aware Engine

1. **Convert Structure**:
   ```yaml
   # Policy DSL (old)
   policies:
     - name: "my-policy"
       rules:
         - condition: "field == value"
           actions: ["block_action"]
   
   # Region-Aware Engine (new)
   rules:
     - id: "my-policy"
       when: "field == 'value'"
       action:
         decision: BLOCK
   ```

2. **Update Conditions**:
   - DSL: `"field.contains('value')"`
   - Region-Aware: `{contains: {field: 'value'}}` or `"field contains 'value'"`

3. **Add Metadata**:
   ```yaml
   rules:
     - id: "my-rule"
       priority: 50
       enabled: true
       metadata:
         description: "Rule description"
         owner: "security-team"
   ```

4. **Test Migration**:
   ```python
   # Validate both policies produce same results
   old_results = dsl_policy.evaluate(test_cases)
   new_results = region_engine.evaluate(test_cases)
   assert old_results == new_results
   ```

## API Reference

### Policy DSL
- `PolicyDSL.load(path)` - Load policy from file
- `PolicyDSL.evaluate(action)` - Evaluate action against policy
- `PolicyDSL.reload()` - Hot reload policy changes

### Region-Aware Engine
- `PolicyEngine.load(path, region)` - Load with region overlay
- `PolicyEngine.evaluate(facts)` - Evaluate with regional rules
- `PolicyEngine.get_matched_rules(facts)` - Get matching rules

### Validation & Analysis
- `PolicyValidator.validate_policy(policy)` - Validate structure
- `PolicyValidator.lint_policy(policy)` - Check best practices
- `PolicySimulator.simulate_policy(policy, cases)` - Dry run
- `PolicyImpactAnalyzer.analyze_impact(old, new, data)` - Impact analysis

## Support

For questions or issues with policy engines:
- Check examples in `examples/` directory
- Review test cases in `tests/`
- Consult API documentation
- File issues on GitHub

## Version History

- **v1.0**: Initial dual-engine support
- **v1.1**: Added policy validation and linting
- **v1.2**: Added simulation and impact analysis
- **v2.0**: Enhanced region-aware capabilities
