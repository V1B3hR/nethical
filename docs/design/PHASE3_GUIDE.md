# Phase 3 Guide: Advanced Governance Features

This guide provides comprehensive documentation for Phase 3 of the Nethical safety governance system.

## Table of Contents

1. [Overview](#overview)
2. [Risk Engine](#risk-engine)
3. [Correlation Engine](#correlation-engine)
4. [Fairness Sampler](#fairness-sampler)
5. [Ethical Drift Reporter](#ethical-drift-reporter)
6. [Performance Optimizer](#performance-optimizer)
7. [Integration](#integration)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Overview

Phase 3 introduces advanced governance capabilities that enhance the monitoring, analysis, and optimization of AI agent behavior:

- **Risk Engine**: Multi-factor risk scoring with time-decay and tier-based escalation
- **Correlation Engine**: Multi-agent pattern detection for coordinated attacks
- **Fairness Sampler**: Stratified sampling across agent cohorts for bias detection
- **Ethical Drift Reporter**: Cohort-based analysis of ethical behavior drift over time
- **Performance Optimizer**: Risk-based detector gating to reduce computational overhead

These components work together to provide intelligent, adaptive governance with minimal performance impact.

### Key Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `RiskEngine` | Agent risk assessment | Multi-factor scoring, time decay, tier escalation |
| `CorrelationEngine` | Pattern detection | Multi-agent coordination, payload analysis |
| `FairnessSampler` | Equitable sampling | Stratified sampling, cohort tracking |
| `EthicalDriftReporter` | Behavior analysis | Cohort comparison, drift detection |
| `PerformanceOptimizer` | Resource management | Risk-based gating, CPU reduction |
| `Phase3IntegratedGovernance` | Unified interface | All Phase 3 features combined |

---

## Risk Engine

The Risk Engine calculates and tracks risk scores for each agent based on their historical behavior.

### Architecture

```
Action → Violation Score → Risk Calculation → Tier Assignment
                                 ↓
                          Time-Decay Applied
                                 ↓
                       Tier-Based Escalation
```

### Core Features

#### 1. Risk Tiers

Agents are classified into risk tiers based on their accumulated risk score:

| Tier | Risk Score Range | Behavior |
|------|-----------------|----------|
| `LOW` | 0.0 - 0.29 | Minimal monitoring |
| `NORMAL` | 0.3 - 0.59 | Standard monitoring |
| `HIGH` | 0.6 - 0.79 | Enhanced monitoring |
| `ELEVATED` | 0.8 - 1.0 | Maximum scrutiny |

#### 2. Multi-Factor Risk Scoring

Risk scores are calculated from multiple factors:

```python
from nethical.core import RiskEngine, RiskTier

engine = RiskEngine(
    decay_half_life_hours=24.0,  # Risk decays over 24 hours
    redis_client=None,            # Optional Redis for persistence
    key_prefix="nethical:risk"
)

# Calculate risk score
risk_score = engine.calculate_risk_score(
    agent_id="agent_123",
    violation_severity=0.75,  # 0.0 to 1.0
    action_context={
        'cohort': 'production',
        'has_violation': True,
        'environment': 'sensitive'
    }
)

# Get agent's current tier
tier = engine.get_tier("agent_123")
print(f"Agent tier: {tier.value}")  # e.g., "elevated"
```

#### 3. Time-Based Decay

Risk scores decay exponentially over time, allowing agents to rehabilitate:

```python
# Risk score immediately after violation
score_now = engine.get_risk_score("agent_123")  # e.g., 0.85

# After 24 hours (one half-life)
# score_later ≈ 0.425 (half of original)

# After 48 hours (two half-lives)
# score_later ≈ 0.21 (quarter of original)
```

The decay formula is: `score_t = score_0 * 0.5^(t / half_life)`

#### 4. Elevated Tier Detection

Check if an agent requires advanced detectors:

```python
# Check if agent should trigger advanced detectors
should_escalate = engine.should_invoke_advanced_detectors("agent_123")

if should_escalate:
    # Run expensive, comprehensive detectors
    run_advanced_analysis(agent)
else:
    # Use lightweight, fast detectors
    run_standard_analysis(agent)
```

#### 5. Risk History

Track risk score evolution over time:

```python
history = engine.get_risk_history("agent_123")

for entry in history:
    print(f"Time: {entry['timestamp']}")
    print(f"Score: {entry['risk_score']:.3f}")
    print(f"Tier: {entry['tier']}")
```

---

## Correlation Engine

The Correlation Engine detects coordinated patterns across multiple agents, identifying sophisticated multi-step attacks.

### Architecture

```
Multiple Agents → Action Tracking → Pattern Matching → Correlation Alert
                        ↓
                 Time Window (5 min)
                        ↓
                 Payload Analysis
```

### Core Features

#### 1. Multi-Agent Pattern Detection

Detect coordinated behavior across multiple agents:

```python
from nethical.core import CorrelationEngine

engine = CorrelationEngine(
    config_path="correlation_rules.yaml",  # Optional custom rules
    redis_client=None,
    key_prefix="nethical:correlation"
)

# Track actions from multiple agents
for agent_id in ["agent_1", "agent_2", "agent_3"]:
    action = create_action(agent_id)
    matches = engine.track_action(
        agent_id=agent_id,
        action=action,
        payload=action.content
    )
    
    if matches:
        for match in matches:
            print(f"⚠️  Pattern detected: {match.pattern_name}")
            print(f"   Severity: {match.severity}")
            print(f"   Confidence: {match.confidence:.2%}")
            print(f"   Description: {match.description}")
```

#### 2. Built-in Correlation Patterns

The engine includes pre-configured patterns:

| Pattern | Description | Indicators |
|---------|-------------|------------|
| `coordinated_probe` | Multiple agents probing same targets | Identical payloads, time proximity |
| `distributed_attack` | DDoS-style distributed actions | High frequency, multiple sources |
| `escalating_permissions` | Progressive privilege escalation | Increasing access requests |
| `data_exfiltration` | Coordinated data extraction | Large transfers, multiple agents |

#### 3. Payload Analysis

Analyze payload characteristics:

```python
# Calculate entropy (randomness)
entropy = engine._calculate_entropy("abcd1234efgh5678")
# High entropy (≥4.0) may indicate obfuscation or encryption

# Check for repetitive patterns
is_repetitive = engine._is_highly_repetitive_payload(
    ["cmd=probe", "cmd=probe", "cmd=probe"]
)
# Repetitive payloads may indicate automated probing
```

#### 4. Time-Window Tracking

Actions are tracked in sliding 5-minute windows:

```python
# Actions tracked within 5-minute windows
# Older actions automatically expire

# Get current tracking state
windows = engine.agent_windows
print(f"Tracking {len(windows)} agents")

for agent_id, window in windows.items():
    print(f"  {agent_id}: {len(window)} recent actions")
```

#### 5. Custom Correlation Rules

Define custom patterns via YAML configuration:

```yaml
# correlation_rules.yaml
patterns:
  - name: custom_pattern
    description: "Detect specific multi-agent behavior"
    min_agents: 3
    time_window_seconds: 300
    conditions:
      - type: payload_similarity
        threshold: 0.8
      - type: frequency
        min_actions_per_agent: 5
```

Load custom rules:

```python
engine = CorrelationEngine(config_path="correlation_rules.yaml")
```

---

## Fairness Sampler

The Fairness Sampler ensures equitable sampling across agent cohorts, preventing bias in monitoring and analysis.

### Architecture

```
Agents → Cohort Assignment → Stratified Sampling → Balanced Analysis
              ↓
    Proportional / Uniform Strategies
```

### Core Features

#### 1. Cohort Management

Organize agents into cohorts for fair comparison:

```python
from nethical.core import FairnessSampler, SamplingStrategy

sampler = FairnessSampler(
    storage_dir="./fairness_samples",
    redis_client=None,
    key_prefix="nethical:fairness"
)

# Assign agents to cohorts
sampler.assign_agent_cohort("agent_1", "production")
sampler.assign_agent_cohort("agent_2", "production")
sampler.assign_agent_cohort("agent_3", "staging")
sampler.assign_agent_cohort("agent_4", "development")

# Get cohort statistics
stats = sampler.get_cohort_stats()
print(f"Cohorts: {stats}")
# {'production': 2, 'staging': 1, 'development': 1}
```

#### 2. Sampling Strategies

Two strategies for balanced sampling:

**Proportional Sampling**: Sample proportional to cohort size
```python
# Production (50%) gets 50% of samples
# Staging (25%) gets 25% of samples
# Development (25%) gets 25% of samples

sample = sampler.sample_agents(
    num_agents=10,
    strategy=SamplingStrategy.PROPORTIONAL
)
```

**Uniform Sampling**: Equal samples from each cohort
```python
# Each cohort gets equal representation
# Ensures minority cohorts aren't under-sampled

sample = sampler.sample_agents(
    num_agents=12,  # 4 per cohort
    strategy=SamplingStrategy.UNIFORM
)
```

#### 3. Stratified Sampling

Get balanced samples across all cohorts:

```python
# Get 20 agents, stratified by cohort
sample = sampler.sample_agents(
    num_agents=20,
    strategy=SamplingStrategy.PROPORTIONAL
)

print(f"Sampled {len(sample)} agents:")
for agent_id in sample:
    cohort = sampler.get_agent_cohort(agent_id)
    print(f"  {agent_id}: {cohort}")
```

#### 4. Bias Prevention

Ensure monitoring doesn't favor certain cohorts:

```python
# Bad: Only monitoring production agents
for agent in production_agents:
    monitor(agent)

# Good: Stratified sampling across all cohorts
sample = sampler.sample_agents(
    num_agents=100,
    strategy=SamplingStrategy.UNIFORM
)
for agent_id in sample:
    monitor(agent_id)
```

---

## Ethical Drift Reporter

The Ethical Drift Reporter tracks changes in agent behavior over time, detecting drift in ethical compliance across cohorts.

### Architecture

```
Actions → Cohort Tracking → Time-Series Analysis → Drift Detection
              ↓
    Risk Scores & Violations
              ↓
         Comparison Metrics
```

### Core Features

#### 1. Action Tracking

Track agent actions and violations by cohort:

```python
from nethical.core import EthicalDriftReporter

reporter = EthicalDriftReporter(
    report_dir="./drift_reports",
    redis_client=None,
    key_prefix="nethical:drift"
)

# Track normal actions
reporter.track_action(
    agent_id="agent_123",
    cohort="production",
    risk_score=0.3
)

# Track violations
reporter.track_violation(
    agent_id="agent_123",
    cohort="production",
    violation_type="privacy",
    severity="medium"
)
```

#### 2. Cohort Comparison

Compare behavior across cohorts:

```python
# Generate drift report
report = reporter.generate_drift_report(
    cohort_a="production",
    cohort_b="staging"
)

print(f"Cohort A ({report['cohort_a']}):")
print(f"  Actions: {report['cohort_a_actions']}")
print(f"  Violations: {report['cohort_a_violations']}")
print(f"  Avg Risk: {report['cohort_a_avg_risk']:.3f}")

print(f"\nCohort B ({report['cohort_b']}):")
print(f"  Actions: {report['cohort_b_actions']}")
print(f"  Violations: {report['cohort_b_violations']}")
print(f"  Avg Risk: {report['cohort_b_avg_risk']:.3f}")

print(f"\nDrift Metrics:")
print(f"  Risk Drift: {report['risk_score_drift']:.3f}")
print(f"  Violation Rate Drift: {report['violation_rate_drift']:.3f}")
```

#### 3. Time-Series Analysis

Track behavior changes over time:

```python
# Get historical data for a cohort
history = reporter.get_cohort_history("production")

# Analyze trends
for timestamp, metrics in history.items():
    print(f"{timestamp}:")
    print(f"  Actions: {metrics['action_count']}")
    print(f"  Violations: {metrics['violation_count']}")
    print(f"  Risk: {metrics['avg_risk']:.3f}")
```

#### 4. Alert Thresholds

Configure drift alerts:

```python
# Check for significant drift
if abs(report['risk_score_drift']) > 0.2:
    print("⚠️  ALERT: Significant risk drift detected!")
    print(f"   Cohorts: {report['cohort_a']} vs {report['cohort_b']}")
    print(f"   Drift: {report['risk_score_drift']:.3f}")

if abs(report['violation_rate_drift']) > 0.15:
    print("⚠️  ALERT: Significant violation rate drift!")
```

#### 5. Report Export

Export reports for analysis:

```python
# Generate comprehensive report
report_path = reporter.export_report(
    cohort_a="production",
    cohort_b="staging",
    output_path="./reports/drift_report.json"
)

print(f"Report saved to: {report_path}")
```

---

## Performance Optimizer

The Performance Optimizer reduces computational overhead by intelligently gating detector invocations based on agent risk.

### Architecture

```
Risk Score → Gating Decision → Selective Detector Invocation → CPU Reduction
                ↓
         Detector Tiers
                ↓
        Performance Tracking
```

### Core Features

#### 1. Detector Registration

Register detectors with performance tiers:

```python
from nethical.core import PerformanceOptimizer, DetectorTier

optimizer = PerformanceOptimizer(
    target_cpu_reduction_pct=30.0  # Target 30% reduction
)

# Register detectors
optimizer.register_detector("intent_monitor", DetectorTier.LIGHTWEIGHT)
optimizer.register_detector("safety_detector", DetectorTier.STANDARD)
optimizer.register_detector("ml_classifier", DetectorTier.EXPENSIVE)
optimizer.register_detector("deep_analysis", DetectorTier.VERY_EXPENSIVE)
```

#### 2. Detector Tiers

Four tiers with different gating thresholds:

| Tier | CPU Cost | Gating Threshold | Use Case |
|------|----------|-----------------|----------|
| `LIGHTWEIGHT` | Low | Always invoked | Fast checks (0.1 threshold) |
| `STANDARD` | Medium | Risk ≥ 0.3 | Common detectors |
| `EXPENSIVE` | High | Risk ≥ 0.6 | ML models |
| `VERY_EXPENSIVE` | Very High | Risk ≥ 0.8 | Deep analysis |

#### 3. Risk-Based Gating

Invoke detectors based on agent risk:

```python
# Low-risk agent (score = 0.2)
should_run = optimizer.should_invoke_detector("ml_classifier", risk_score=0.2)
# Returns False - skip expensive detector for low-risk agent

# High-risk agent (score = 0.7)
should_run = optimizer.should_invoke_detector("ml_classifier", risk_score=0.7)
# Returns True - run expensive detector for high-risk agent
```

#### 4. Performance Tracking

Track CPU usage and reduction:

```python
# Track detector invocation
optimizer.track_detector_invocation(
    detector_name="ml_classifier",
    cpu_time_ms=15.3
)

# Track overall action processing
optimizer.track_action_processing(
    cpu_time_ms=20.5,
    detectors_invoked=3
)

# Get performance metrics
reduction = optimizer.get_cpu_reduction_pct()
meeting_target = optimizer.is_meeting_target()

print(f"CPU Reduction: {reduction:.1f}%")
print(f"Meeting Target: {meeting_target}")
```

#### 5. Optimization Report

Generate performance report:

```python
report = optimizer.get_optimization_report()

print(f"Actions Processed: {report['actions_processed']}")
print(f"Total CPU Time: {report['total_cpu_ms']:.2f}ms")
print(f"Avg CPU per Action: {report['avg_cpu_per_action_ms']:.2f}ms")
print(f"CPU Reduction: {report['cpu_reduction_pct']:.1f}%")

print("\nDetector Statistics:")
for detector, stats in report['detector_stats'].items():
    print(f"  {detector}:")
    print(f"    Invocations: {stats['invocations']}")
    print(f"    Total CPU: {stats['total_cpu_ms']:.2f}ms")
    print(f"    Avg CPU: {stats['avg_cpu_ms']:.2f}ms")
```

---

## Integration

### Phase 3 Integrated Governance

The `Phase3IntegratedGovernance` class combines all Phase 3 components:

```python
from nethical.core import Phase3IntegratedGovernance

governance = Phase3IntegratedGovernance(
    redis_client=None,  # Optional Redis for persistence
    correlation_config_path="correlation_rules.yaml",
    storage_dir="./nethical_data",
    enable_performance_optimization=True
)

# Process action through all Phase 3 components
result = governance.process_action(
    agent_id="agent_123",
    action=action_object,
    cohort="production",
    violation_detected=True,
    violation_type="privacy",
    violation_severity="high",
    detector_invocations={
        "intent_monitor": 2.3,
        "safety_detector": 5.1,
        "ml_classifier": 12.7
    }
)

# Result includes all Phase 3 data
print(f"Risk Score: {result['risk_score']:.3f}")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Invoke Advanced: {result['invoke_advanced_detectors']}")
print(f"Correlations: {len(result['correlations'])}")
print(f"CPU Reduction: {result['performance_metrics']['cpu_reduction_pct']:.1f}%")
```

### Migration to IntegratedGovernance

Phase3IntegratedGovernance is deprecated. Use IntegratedGovernance instead:

```python
# Old (deprecated)
from nethical.core import Phase3IntegratedGovernance
governance = Phase3IntegratedGovernance(storage_dir="./data")

# New (recommended)
from nethical.core import IntegratedGovernance
governance = IntegratedGovernance(
    storage_dir="./data",
    enable_performance_optimization=True,
    enable_merkle_anchoring=False,  # Phase 4 feature
    enable_shadow_mode=False        # Phase 5 feature
)
```

### Sampling Integration

Use fairness sampler for balanced monitoring:

```python
# Sample agents for in-depth analysis
sample = governance.sample_agents_for_review(
    num_agents=50,
    strategy=SamplingStrategy.UNIFORM
)

for agent_id in sample:
    # Perform detailed review
    analysis = deep_analysis(agent_id)
```

### Drift Analysis Integration

Generate regular drift reports:

```python
# Compare production vs staging cohorts
report = governance.generate_drift_report(
    cohort_a="production",
    cohort_b="staging"
)

if abs(report['risk_score_drift']) > 0.2:
    alert_operations_team(report)
```

---

## Best Practices

### 1. Risk Engine Configuration

- **Half-Life Selection**: Set decay half-life based on your risk tolerance
  - Stricter environments: 48-72 hours (slower forgiveness)
  - Dynamic environments: 12-24 hours (faster adaptation)
  
```python
# Strict environment
engine = RiskEngine(decay_half_life_hours=48.0)

# Dynamic environment
engine = RiskEngine(decay_half_life_hours=12.0)
```

### 2. Correlation Detection

- **Custom Rules**: Define patterns specific to your domain
- **Time Windows**: Adjust window size for pattern complexity
- **False Positives**: Tune confidence thresholds to reduce noise

```python
# Adjust for your needs
engine = CorrelationEngine(
    config_path="custom_rules.yaml"
)
```

### 3. Fairness Sampling

- **Cohort Definition**: Define meaningful cohorts (environment, region, version)
- **Strategy Selection**: 
  - Use PROPORTIONAL for general analysis
  - Use UNIFORM when minority cohorts are important

```python
# Ensure minority cohorts are represented
sample = sampler.sample_agents(
    num_agents=100,
    strategy=SamplingStrategy.UNIFORM
)
```

### 4. Drift Monitoring

- **Regular Reports**: Generate drift reports on a schedule (daily/weekly)
- **Baseline Establishment**: Define acceptable drift thresholds
- **Alert Integration**: Connect to your alerting system

```python
# Daily drift check
if is_daily_report_time():
    report = reporter.generate_drift_report("prod", "staging")
    if requires_alert(report):
        send_alert(report)
```

### 5. Performance Optimization

- **Tiering**: Assign appropriate tiers to detectors
- **Monitoring**: Track CPU reduction regularly
- **Tuning**: Adjust target reduction based on requirements

```python
# Monitor performance
if not optimizer.is_meeting_target():
    # Adjust detector tiers or thresholds
    adjust_optimization_parameters()
```

### 6. Redis Persistence

For production deployments, use Redis for persistence:

```python
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

governance = Phase3IntegratedGovernance(
    redis_client=redis_client,
    storage_dir="./nethical_data"
)
```

### 7. Monitoring and Alerts

Set up monitoring for key metrics:

```python
# Collect metrics
system_status = governance.get_system_status()

# Check for issues
if system_status['risk_engine']['high_risk_agents'] > 10:
    alert("High number of elevated agents")

if system_status['performance_optimizer']['cpu_reduction_pct'] < 20:
    alert("Performance optimization below target")
```

---

## Examples

### Example 1: Basic Risk Tracking

```python
from nethical.core import RiskEngine

engine = RiskEngine(decay_half_life_hours=24.0)

# Simulate agent activity
for i in range(10):
    # Some actions with violations
    if i % 3 == 0:
        score = engine.calculate_risk_score("agent_1", 0.8, {})
    else:
        score = engine.calculate_risk_score("agent_1", 0.1, {})
    
    print(f"Action {i}: Risk={score:.3f}, Tier={engine.get_tier('agent_1').value}")

# Check escalation
if engine.should_invoke_advanced_detectors("agent_1"):
    print("⚠️  Agent requires enhanced monitoring")
```

### Example 2: Correlation Detection

```python
from nethical.core import CorrelationEngine

engine = CorrelationEngine()

# Simulate coordinated probing
for agent_id in ["agent_1", "agent_2", "agent_3"]:
    for i in range(5):
        action = MockAction(f"probe_target_123")
        matches = engine.track_action(agent_id, action, "probe_target_123")
        
        if matches:
            print(f"⚠️  Correlation detected: {matches[0].pattern_name}")
            print(f"   Agents involved: {len(engine.agent_windows)}")
```

### Example 3: Fairness-Aware Sampling

```python
from nethical.core import FairnessSampler, SamplingStrategy

sampler = FairnessSampler(storage_dir="./samples")

# Assign agents to cohorts
for i in range(100):
    cohort = ["prod", "staging", "dev"][i % 3]
    sampler.assign_agent_cohort(f"agent_{i}", cohort)

# Get uniform sample
sample = sampler.sample_agents(
    num_agents=30,
    strategy=SamplingStrategy.UNIFORM
)

# Verify balance
cohort_counts = {}
for agent_id in sample:
    cohort = sampler.get_agent_cohort(agent_id)
    cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1

print(f"Sample distribution: {cohort_counts}")
# Should be approximately equal: {'prod': 10, 'staging': 10, 'dev': 10}
```

### Example 4: Drift Detection

```python
from nethical.core import EthicalDriftReporter

reporter = EthicalDriftReporter(report_dir="./reports")

# Simulate diverging behavior
for i in range(100):
    # Production: low violations
    reporter.track_action("prod_agent", "production", 0.2)
    if i % 20 == 0:
        reporter.track_violation("prod_agent", "production", "safety", "low")
    
    # Staging: high violations
    reporter.track_action("stage_agent", "staging", 0.6)
    if i % 5 == 0:
        reporter.track_violation("stage_agent", "staging", "safety", "high")

# Detect drift
report = reporter.generate_drift_report("production", "staging")
print(f"Risk Drift: {report['risk_score_drift']:.3f}")
print(f"Violation Rate Drift: {report['violation_rate_drift']:.3f}")

if abs(report['risk_score_drift']) > 0.2:
    print("⚠️  ALERT: Significant behavioral drift detected!")
```

### Example 5: Performance-Optimized Processing

```python
from nethical.core import Phase3IntegratedGovernance, DetectorTier

governance = Phase3IntegratedGovernance(
    storage_dir="./data",
    enable_performance_optimization=True
)

# Register detectors
optimizer = governance.performance_optimizer
optimizer.register_detector("fast_check", DetectorTier.LIGHTWEIGHT)
optimizer.register_detector("ml_model", DetectorTier.EXPENSIVE)

# Process multiple actions
for i in range(100):
    # Determine which detectors to run
    agent_id = f"agent_{i % 10}"
    risk = governance.risk_engine.get_risk_score(agent_id)
    
    invocations = {}
    
    # Always run lightweight detector
    if optimizer.should_invoke_detector("fast_check", risk):
        invocations["fast_check"] = 1.5
    
    # Only run ML for high-risk agents
    if optimizer.should_invoke_detector("ml_model", risk):
        invocations["ml_model"] = 15.0
    
    # Process action
    result = governance.process_action(
        agent_id=agent_id,
        action=MockAction(f"action_{i}"),
        cohort="production",
        violation_detected=(i % 10 == 0),
        violation_severity="medium",
        detector_invocations=invocations
    )

# Check performance savings
print(f"CPU Reduction: {result['performance_metrics']['cpu_reduction_pct']:.1f}%")
```

### Example 6: Complete Phase 3 Workflow

```python
from nethical.core import Phase3IntegratedGovernance

# Initialize governance
governance = Phase3IntegratedGovernance(
    storage_dir="./nethical_data",
    enable_performance_optimization=True
)

# Process action through all components
result = governance.process_action(
    agent_id="agent_123",
    action=action_object,
    cohort="production",
    violation_detected=True,
    violation_type="privacy",
    violation_severity="high",
    detector_invocations={
        "intent_monitor": 2.3,
        "safety_detector": 5.1,
        "ml_classifier": 12.7
    }
)

# Use results for decision-making
if result['risk_tier'] == 'elevated':
    print("⚠️  Agent at elevated risk tier")
    # Trigger enhanced monitoring

if result['correlations']:
    print(f"⚠️  {len(result['correlations'])} correlation patterns detected")
    for corr in result['correlations']:
        print(f"   - {corr['pattern']}: {corr['description']}")
    # Alert security team

if result['invoke_advanced_detectors']:
    print("🔍 Running advanced detectors for this agent")
    # Run expensive analysis

# Check performance
cpu_reduction = result['performance_metrics']['cpu_reduction_pct']
print(f"💡 CPU Reduction: {cpu_reduction:.1f}%")

# Generate periodic reports
drift_report = governance.generate_drift_report("production", "staging")
if abs(drift_report['risk_score_drift']) > 0.2:
    print("⚠️  Significant drift detected between cohorts")

# Check system health
status = governance.get_system_status()
print(f"System Status: {status['timestamp']}")
print(f"Risk Engine: {status['risk_engine']['total_agents']} agents tracked")
print(f"Correlation Engine: {status['correlation_engine']['patterns_detected']} patterns")
```

---

## API Reference

### Phase3IntegratedGovernance

Main integration class combining all Phase 3 components.

**Methods:**

- `process_action(agent_id, action, cohort, violation_detected, ...)`: Process action through all components
- `should_invoke_detector(detector_name, agent_id, tier)`: Check if detector should be invoked
- `sample_agents_for_review(num_agents, strategy)`: Get stratified agent sample
- `generate_drift_report(cohort_a, cohort_b)`: Generate cohort comparison report
- `get_system_status()`: Get system-wide statistics
- `export_report(output_path)`: Export comprehensive report

### See Also

- [Phase 4 Guide](PHASE4_GUIDE.md) - Integrity & Ethics Operationalization
- [Phase 5-7 Guide](PHASE5-7_GUIDE.md) - ML & Anomaly Detection
- [Phase 8-9 Guide](PHASE89_GUIDE.md) - Human-in-the-Loop & Optimization
- [Main README](../../README.md) - Project overview

---

**Last Updated**: November 5, 2025  
**Version**: Phase 3 Complete  
**Status**: Production Ready
