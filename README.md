# Nethical 🔒

**Safety, Ethics and More for AI Agents**

Nethical is a comprehensive safety governance system designed to monitor and evaluate AI agent actions for safety, ethical compliance, and manipulation detection. It provides real-time oversight and judgment capabilities to ensure AI agents operate within defined safety boundaries.

## 🎯 What is Nethical?

Nethical serves as a guardian layer for AI systems, continuously monitoring agent behavior to ensure safe, ethical, and transparent operations. It acts as a real-time safety net that can detect, evaluate, and respond to potentially harmful or unethical AI agent behaviors before they cause damage.

### Main Purpose

The primary goal of Nethical is to bridge the gap between AI capability and AI safety by providing:

- **Proactive Monitoring**: Real-time surveillance of AI agent actions
- **Ethical Compliance**: Ensuring AI systems adhere to ethical guidelines  
- **Safety Enforcement**: Preventing harmful or dangerous AI behaviors
- **Transparency**: Providing clear insights into AI decision-making processes
- **Trust Building**: Creating confidence in AI systems through robust oversight

## ✨ Key Features

- 🎯 **Intent vs Action Monitoring**: Detects deviations between an agent's stated intent and actual actions
- ⚖️ **Ethical Violation Detection**: Identifies harmful content, deceptive behavior, privacy violations, and discrimination
- 🛡️ **Safety Constraint Enforcement**: Monitors for unauthorized system access, data modification, and resource abuse
- 🕵️ **Manipulation Recognition**: Detects emotional manipulation, authority abuse, social proof, scarcity tactics, and reciprocity exploitation
- 🏛️ **Judge System**: Powerful decision-making component that evaluates actions and provides feedback with restrictions or case closure
- 📊 **Comprehensive Reporting**: Detailed violation summaries and judgment analytics
- ⚙️ **Configurable Monitoring**: Flexible configuration for different use cases and security levels

## 🚀 Where and How to Use Nethical

### Use Cases

**Enterprise AI Systems**
- Monitor customer service bots for ethical interactions
- Ensure compliance with corporate governance policies
- Prevent AI systems from making unauthorized decisions

**AI Development and Testing**
- Validate AI agent behavior during development cycles  
- Test safety boundaries before production deployment
- Continuous integration safety checks

**Research and Academic Settings**
- Study AI safety and alignment challenges
- Benchmark ethical AI system performance
- Educational tool for AI ethics courses

**Regulatory Compliance**
- Meet AI governance and compliance requirements
- Audit AI system behavior for regulatory reporting
- Implement industry-specific safety standards

### Installation

```bash
pip install -e .
```

### Quick Start

```python
import asyncio
from nethical import SafetyGovernance, AgentAction

async def main():
    # Initialize the governance system
    governance = SafetyGovernance()
    
    # Create an agent action to evaluate
    action = AgentAction(
        id="action_001",
        agent_id="my_agent",
        stated_intent="I will help the user with their question",
        actual_action="I will help the user with their question",
        context={"user_request": "What is the weather today?"}
    )
    
    # Evaluate the action
    judgment = await governance.evaluate_action(action)
    print(f"Decision: {judgment.decision}")
    print(f"Confidence: {judgment.confidence}")
    print(f"Reasoning: {judgment.reasoning}")
    
    if judgment.feedback:
        print(f"Feedback: {judgment.feedback}")

asyncio.run(main())
```

### Running Examples

```bash
python examples/basic_usage.py
```

## 🏗️ System Architecture

Nethical consists of several key components:

### Core Components

- **SafetyGovernance**: Main orchestrator that coordinates all monitoring and judgment activities
- **AgentAction**: Data model representing agent intentions and actions
- **SafetyViolation**: Model for detected violations with severity levels
- **JudgmentResult**: Result of judge evaluation with decisions and feedback

### Monitoring Systems

- **IntentDeviationMonitor**: Compares stated intentions with actual actions
- **EthicalViolationDetector**: Identifies ethical constraint violations
- **SafetyViolationDetector**: Detects safety-related violations  
- **ManipulationDetector**: Recognizes various manipulation techniques

### Decision Engine

**SafetyJudge**: Evaluates actions and violations to make informed decisions:
- `ALLOW`: Action is safe to proceed
- `RESTRICT`: Action allowed with limitations  
- `BLOCK`: Action prevented due to safety concerns
- `TERMINATE`: Critical violation requiring immediate cessation

## ⚙️ Configuration

Customize monitoring behavior for your specific needs:

```python
from nethical import SafetyGovernance, MonitoringConfig

# Create custom configuration
config = MonitoringConfig(
    intent_deviation_threshold=0.8,
    enable_ethical_monitoring=True,
    enable_safety_monitoring=True,
    enable_manipulation_detection=True,
    max_violation_history=1000
)

# Initialize with configuration
governance = SafetyGovernance(config)
```

## 📊 Analytics and Reporting

Get comprehensive insights into agent behavior:

```python
# Get violation summary
violation_summary = governance.get_violation_summary()
print(f"Total violations: {violation_summary['total_violations']}")
print(f"By type: {violation_summary['by_type']}")

# Get judgment summary
judgment_summary = governance.get_judgment_summary()
print(f"Total judgments: {judgment_summary['total_judgments']}")
print(f"Average confidence: {judgment_summary['average_confidence']}")

# Get system status
status = governance.get_system_status()
print(f"System components: {status}")
```

## 🚀 Phase 3: Advanced Features ✅ COMPLETE

Nethical Phase 3 is fully implemented with advanced correlation, risk management, and fairness monitoring:

### Integrated Governance

```python
from nethical.core import Phase3IntegratedGovernance, DetectorTier

# Initialize with all Phase 3 features
governance = Phase3IntegratedGovernance(
    storage_dir="./data",
    enable_performance_optimization=True
)

# Process action with adaptive risk scoring
results = governance.process_action(
    agent_id="agent_123",
    action=my_action,
    cohort="production",
    violation_detected=False,
    detector_invocations={"safety_detector": 10.0}
)

print(f"Risk Score: {results['risk_score']:.3f}")
print(f"Risk Tier: {results['risk_tier']}")
print(f"Invoke Advanced: {results['invoke_advanced_detectors']}")
```

### Key Phase 3 Features

- **🎯 Risk Engine**: Multi-factor risk scoring with exponential decay
- **🔗 Correlation Engine**: Multi-agent pattern detection (5 algorithms)
- **⚖️ Fairness Sampler**: Stratified sampling across agent cohorts
- **📈 Ethical Drift Reporter**: Cohort-based bias detection
- **⚡ Performance Optimizer**: 30%+ CPU reduction via risk-based gating

See [PHASE3_GUIDE.md](PHASE3_GUIDE.md) for complete documentation.

## 🔐 Phase 4: Integrity & Ethics Operationalization (IN PROGRESS)

Phase 4 introduces immutable audit trails, policy management, quarantine capabilities, and ethical impact tracking:

### Key Phase 4 Features

- **🔒 Merkle Anchoring**: Immutable audit logs with cryptographic verification
- **📋 Policy Diff Auditing**: Semantic policy change detection and risk assessment
- **⚠️ Quarantine Mode**: Automatic isolation of anomalous agent cohorts
- **🏷️ Ethical Taxonomy**: Multi-dimensional ethical impact classification
- **📊 SLA Monitoring**: Performance guarantees and validation

See [PHASE4_GUIDE.md](PHASE4_GUIDE.md) for complete documentation.

## 🤖 Phases 5-9: ML & Optimization Pipeline (PLANNED)

The next phases introduce machine learning, anomaly detection, human feedback loops, and continuous optimization:

### Phase 5 – ML Shadow Mode
- Train minimal classifier (logistic regression or small transformer)
- Passive inference with no enforcement authority
- Log predictions alongside rule-based outcomes for comparison
- Collect baseline metrics (precision, recall, F1, calibration)

### Phase 6 – ML Assisted Enforcement
- Blend ML predictions with rule-based risk (e.g., `0.7 * rules + 0.3 * ml`)
- Apply blending only in gray zone (uncertain decisions)
- Maintain audit trail of decision adjustments
- Gate: False positive delta <5%; improved detection rate

### Phase 7 – Anomaly & Drift Detection
- Sequence anomaly scoring using n-gram or simple models
- Distribution shift detection (PSI / KL divergence)
- Automated alert pipeline for drift events
- Behavioral anomaly detection for unusual agent patterns

### Phase 8 – Human-in-the-Loop Ops
- Escalation queue with labeling interface (CLI or UI)
- Structured feedback tags: `false_positive`, `missed_violation`, `policy_gap`
- Human review workflow for uncertain decisions
- Median triage SLA tracking and optimization

### Phase 9 – Continuous Optimization
- Automated tuning of rule weights, classifier thresholds, and escalation boundaries
- Multi-objective optimization (max recall, min FP rate, min latency)
- Techniques: grid/random search, evolutionary strategies, Bayesian optimization
- Continuous feedback loop from human labels to retrain models

See [roadmap.md](roadmap.md) for complete phase specifications and exit criteria.

## 🚀 Future Tracks: Preparations for 11–50 Systems

The Future Tracks outline upcoming enhancements for scaling Nethical to support larger multi-system deployments:

### Planned Subphases

- **🌍 F1: Regionalization & Sharding**: Geographic distribution with `region_id` and `logical_domain` fields for hierarchical aggregation
- **🔌 F2: Detector & Policy Extensibility**: RPC/gRPC-based detector externalization and Policy DSL for compiled rule specifications
- **🔒 F3: Privacy & Data Handling**: Enhanced redaction pipeline with differential privacy for federated analytics
- **🎯 F4: Thresholds, Tuning, & Adaptivity**: ML-driven threshold tuning based on decision outcomes and human feedback
- **⏮️ F5: Simulation & Replay**: Time-travel debugging and what-if analysis with persistent action streams
- **🏪 F6: Marketplace & Ecosystem**: Plugin registry and governance for community-contributed detectors

See [roadmap.md](roadmap.md) for detailed specifications.

## 🛠️ Development Setup

```bash
git clone <repository-url>
cd nethical
pip install -e .[dev]
```

### Running Tests

```bash
pip install -e .[dev]
pytest tests/
```

### Code Quality

This project uses Black for code formatting and Flake8 for linting:

```bash
black nethical/ tests/ examples/
flake8 nethical/ tests/ examples/
mypy nethical/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- AI Safety frameworks
- Agent monitoring systems  
- Ethical AI tools

---

**Nethical - Ensuring AI agents operate safely, ethically, and transparently. 🔒**
