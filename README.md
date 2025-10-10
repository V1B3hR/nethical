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
- 🤖 **ML-Based Anomaly Detection**: Trainable machine learning models that learn to detect anomalous agent behavior patterns
- 📈 **Distribution Drift Monitoring**: Statistical detection of changes in agent behavior over time
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
# Basic safety monitoring
python examples/basic/basic_usage.py

# Anomaly detection
python examples/governance/phase7_demo.py

# Train anomaly detection model
python examples/training/train_anomaly_detector.py
```

### Training Anomaly Detection Models

Nethical includes trainable ML models for detecting anomalous agent behavior:

```bash
# Train a model on synthetic data
python training/train_any_model.py --model-type anomaly --num-samples 5000

# Train with audit logging (for compliance and reproducibility)
python training/train_any_model.py --model-type logistic --num-samples 10000 --enable-audit

# Use the trained model
python -c "
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier
clf = AnomalyMLClassifier.load('models/current/anomaly_model_*.json')
result = clf.predict({'sequence': ['read', 'process', 'write']})
print(f'Anomalous: {result[\"label\"] == 1}, Score: {result[\"score\"]:.3f}')
"
```

**Audit Logging**: Enable `--enable-audit` to create an immutable, cryptographically-verifiable training audit trail using Merkle trees. This is essential for compliance, model governance, and reproducibility. See [docs/AUDIT_LOGGING_GUIDE.md](docs/AUDIT_LOGGING_GUIDE.md) for details.

See [ANOMALY_DETECTION_TRAINING.md](ANOMALY_DETECTION_TRAINING.md) for detailed documentation.

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
- **AnomalyDriftMonitor**: Detects unusual behavior patterns and distribution shifts
- **AnomalyMLClassifier**: ML-based anomaly detector (trainable)

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

## 🤖 Phases 5-7: ML & Anomaly Detection Pipeline ✅ IMPLEMENTED

Nethical now includes machine learning integration and anomaly detection capabilities:

### Phase 5 – ML Shadow Mode ✅ COMPLETE

Train and validate ML models in a safe, passive environment:

```python
from nethical.core import MLShadowClassifier, MLModelType

# Initialize shadow classifier
shadow = MLShadowClassifier(
    model_type=MLModelType.HEURISTIC,
    score_agreement_threshold=0.1,
    storage_path="./shadow_logs"
)

# Make passive predictions (no enforcement)
prediction = shadow.predict(
    agent_id="agent_123",
    action_id="action_456",
    features={
        'violation_count': 0.5,
        'severity_max': 0.6,
        'recency_score': 0.4
    },
    rule_risk_score=0.5,
    rule_classification="warn"
)

# Get baseline metrics
metrics = shadow.get_metrics_report()
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
print(f"Expected Calibration Error: {metrics['expected_calibration_error']:.3f}")
```

### Phase 6 – ML Assisted Enforcement ✅ COMPLETE

Blend ML predictions with rule-based decisions in the gray zone:

```python
from nethical.core import MLBlendedRiskEngine, RiskZone

# Initialize blended risk engine
blended = MLBlendedRiskEngine(
    gray_zone_lower=0.4,
    gray_zone_upper=0.6,
    rule_weight=0.7,
    ml_weight=0.3,
    storage_path="./blended_logs"
)

# Compute blended risk (ML only influences gray zone)
decision = blended.compute_blended_risk(
    agent_id="agent_123",
    action_id="action_456",
    rule_risk_score=0.5,  # In gray zone
    rule_classification="warn",
    ml_risk_score=0.6,
    ml_confidence=0.85
)

print(f"Risk Zone: {decision.risk_zone.value}")
print(f"ML Influenced: {decision.ml_influenced}")
print(f"Blended Score: {decision.blended_risk_score:.3f}")
print(f"Final Classification: {decision.blended_classification}")

# Check promotion gate
metrics = blended.get_metrics_report()
gate_metrics = metrics['gate_metrics']
print(f"Passes Gate: {gate_metrics['passes_gate']}")
print(f"FP Delta: {gate_metrics['fp_delta_percentage']:.1f}%")
```

### Phase 7 – Anomaly & Drift Detection ✅ COMPLETE

Detect unusual patterns and distribution shifts:

```python
from nethical.core import AnomalyDriftMonitor, AnomalyType, DriftSeverity

# Initialize anomaly monitor
monitor = AnomalyDriftMonitor(
    sequence_n=3,
    psi_threshold=0.2,
    kl_threshold=0.1,
    storage_path="./anomaly_logs"
)

# Set baseline distribution
baseline_scores = [0.2, 0.3, 0.25, 0.35, 0.28] * 100
monitor.set_baseline_distribution(baseline_scores)

# Record actions and check for anomalies
alert = monitor.record_action(
    agent_id="agent_123",
    action_type="unusual_action",
    risk_score=0.5,
    cohort="production"
)

if alert:
    print(f"⚠️ Alert: {alert.anomaly_type.value}")
    print(f"Severity: {alert.severity.value}")
    print(f"Score: {alert.anomaly_score:.3f}")
    print(f"Quarantine Recommended: {alert.quarantine_recommended}")

# Check for behavioral anomalies
behavioral_alert = monitor.check_behavioral_anomaly("agent_123")

# Check for distribution drift
drift_alert = monitor.check_drift(cohort="production")

# Get statistics
stats = monitor.get_statistics()
print(f"Total Alerts: {stats['alerts']['total']}")
print(f"Critical Alerts: {stats['alerts']['by_severity']['critical']}")
```

**Key Features:**
- ✅ Sequence anomaly detection (n-gram based)
- ✅ Behavioral anomaly detection (repetitive patterns)
- ✅ Distribution drift detection (PSI & KL divergence)
- ✅ Automated alert pipeline with severity levels
- ✅ Quarantine recommendations for critical anomalies

## 🔄 Phases 8-9: Human-in-the-Loop & Optimization ✅ IMPLEMENTED

The final phases introduce human feedback loops and continuous optimization:

### Phase 8 – Human-in-the-Loop Ops ✅ COMPLETE

**Escalation & Review Workflow:**
- ✅ Escalation queue with labeling interface (CLI and programmatic API)
- ✅ Structured feedback tags: `false_positive`, `missed_violation`, `policy_gap`, `correct_decision`, `edge_case`
- ✅ Human review workflow for uncertain or critical decisions
- ✅ Structured feedback collection for model and rule improvement
- ✅ Median triage SLA tracking and optimization
- ✅ SQLite persistence for escalation cases and feedback
- ✅ Priority-based queue management (LOW, MEDIUM, HIGH, CRITICAL, EMERGENCY)

**Example Usage:**

```python
from nethical.core import Phase89IntegratedGovernance, FeedbackTag

# Initialize governance with human-in-the-loop
governance = Phase89IntegratedGovernance(
    storage_dir="./data",
    triage_sla_seconds=3600,      # 1 hour triage SLA
    resolution_sla_seconds=86400  # 24 hour resolution SLA
)

# Process action with automatic escalation
result = governance.process_with_escalation(
    judgment_id="judg_123",
    action_id="act_456",
    agent_id="agent_789",
    decision="block",
    confidence=0.65,
    violations=[{"type": "safety", "severity": 4}]
)

if result['escalated']:
    print(f"Case escalated: {result['escalation_case_id']}")

# Human reviewer workflow
case = governance.get_next_case(reviewer_id="reviewer_alice")
if case:
    feedback = governance.submit_feedback(
        case_id=case.case_id,
        reviewer_id="reviewer_alice",
        feedback_tags=[FeedbackTag.FALSE_POSITIVE],
        rationale="Content was actually safe, detector too aggressive",
        corrected_decision="allow",
        confidence=0.9
    )

# Track SLA metrics
metrics = governance.get_sla_metrics()
print(f"Median Triage Time: {metrics.median_triage_time_seconds}s")
print(f"SLA Breaches: {metrics.sla_breaches}")

# Get feedback summary for continuous improvement
summary = governance.get_feedback_summary()
print(f"False Positive Rate: {summary['false_positive_rate']:.1%}")
```

**CLI Tool:**

```bash
# List pending cases
python cli/review_queue list

# Get next case for review
python cli/review_queue next reviewer_alice

# Submit feedback
python cli/review_queue feedback esc_abc123 reviewer_alice \
    --tags false_positive \
    --rationale "Content was safe" \
    --corrected-decision allow

# View statistics
python cli/review_queue stats
python cli/review_queue summary
```

### Phase 9 – Continuous Optimization ✅ COMPLETE

**Automated Tuning:**
- ✅ Automated tuning of rule weights, classifier thresholds, and escalation boundaries
- ✅ Multi-objective optimization (max recall, min FP rate, min latency, max human agreement)
- ✅ Techniques: grid search, random search, evolutionary strategies
- ✅ Continuous feedback loop from human labels to retrain models
- ✅ Configuration versioning and tracking

**Promotion Gate:**
- ✅ New configurations promoted only when gate conditions are met
- ✅ Validation against precision/recall targets
- ✅ Configurable promotion criteria (recall gain, FP increase, latency, human agreement)
- ✅ A/B testing framework support through configuration status tracking

**Example Usage:**

```python
from nethical.core import Phase89IntegratedGovernance

governance = Phase89IntegratedGovernance(storage_dir="./data")

# Create baseline configuration
baseline = governance.create_configuration(
    config_version="baseline_v1.0",
    classifier_threshold=0.5,
    gray_zone_lower=0.4,
    gray_zone_upper=0.6
)

# Record baseline metrics
governance.record_metrics(
    config_id=baseline.config_id,
    detection_recall=0.82,
    detection_precision=0.85,
    false_positive_rate=0.08,
    decision_latency_ms=12.0,
    human_agreement=0.86,
    total_cases=1000
)

# Run optimization to find better configuration
results = governance.optimize_configuration(
    technique="random_search",
    param_ranges={
        'classifier_threshold': (0.4, 0.7),
        'gray_zone_lower': (0.3, 0.5),
        'gray_zone_upper': (0.5, 0.7)
    },
    n_iterations=50
)

# Check best candidate against promotion gate
best_config, best_metrics = results[0]
passed, reasons = governance.check_promotion_gate(
    candidate_id=best_config.config_id,
    baseline_id=baseline.config_id
)

if passed:
    # Promote to production
    governance.promote_configuration(best_config.config_id)
    print(f"✓ Promoted {best_config.config_version}")
else:
    print("Gate criteria not met:", reasons)

# Run continuous improvement cycle
cycle_result = governance.continuous_improvement_cycle()
print(f"Human Agreement: {cycle_result['human_agreement']:.1%}")
print(f"Recommendations: {cycle_result['recommendations']}")
```

**Key Features:**
- ✅ Multi-objective fitness scoring with configurable weights
- ✅ Grid search with parameter grids
- ✅ Random search with parameter ranges
- ✅ Evolutionary search with mutation and selection
- ✅ Promotion gate validation with detailed criteria
- ✅ SQLite persistence for configurations and metrics
- ✅ Continuous improvement feedback loop

## 🎯 Unified Integrated Governance ✅ NEW

For simplified usage, a **consolidated IntegratedGovernance** class is now available that brings together ALL phases (3, 4, 5-7, 8-9) into a single unified interface.

**Example Usage:**

```python
from nethical.core import IntegratedGovernance

# Initialize with all features
gov = IntegratedGovernance(
    storage_dir="./nethical_data",
    # Phase 3
    enable_performance_optimization=True,
    # Phase 4
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True,
    # Phase 5-7
    enable_shadow_mode=True,
    enable_ml_blending=True,
    enable_anomaly_detection=True,
    # Phase 8-9
    auto_escalate_on_block=True,
    auto_escalate_on_low_confidence=True
)

# Process action through ALL phases with a single call
result = gov.process_action(
    agent_id="agent_123",
    action="User request processing",
    cohort="production",
    violation_detected=True,
    violation_type="safety",
    violation_severity="medium",
    # ML features (optional)
    action_id="action_456",
    action_type="response",
    features={'ml_score': 0.7},
    rule_risk_score=0.65,
    rule_classification="warn"
)

# Results include data from all phases
print(f"Risk Score (Phase 3): {result['phase3']['risk_score']}")
print(f"Merkle Events (Phase 4): {result['phase4']['merkle']['event_count']}")
print(f"ML Shadow Score (Phase 5): {result['phase567']['shadow']['ml_risk_score']}")
print(f"Blended Risk (Phase 6): {result['phase567']['blended']['blended_risk_score']}")

# Get comprehensive system status
status = gov.get_system_status()
```

**Benefits:**
- ✅ Single unified interface for all governance features
- ✅ Seamless integration across all phases
- ✅ Simplified configuration and initialization
- ✅ Comprehensive action processing pipeline
- ✅ Unified system status and monitoring

Run the demo: `python examples/basic/unified_governance_demo.py`

See [PHASE3_GUIDE.md](PHASE3_GUIDE.md), [PHASE4_GUIDE.md](PHASE4_GUIDE.md), [PHASE5-7_GUIDE.md](PHASE5-7_GUIDE.md), and [PHASE89_GUIDE.md](PHASE89_GUIDE.md) for detailed documentation on individual phases.

See [roadmap.md](roadmap.md) for complete phase specifications and exit criteria.

## 🎓 ML Training Pipeline with Real-World Datasets

Nethical includes an end-to-end training orchestrator that can download, process, and train the BaselineMLClassifier using real-world security datasets from Kaggle.

### Quick Start

```bash
# Full pipeline: download, process, and train
python scripts/baseline_orchestrator.py

# Or step by step:
# 1. Download datasets (requires Kaggle API credentials)
python scripts/baseline_orchestrator.py --download

# 2. Process CSV files into standardized format
python scripts/baseline_orchestrator.py --process-only

# 3. Train on processed data
python scripts/baseline_orchestrator.py --train-only
```

### Datasets Used

The pipeline uses real-world security datasets listed in `datasets/datasets`:
- [Cyber Security Attacks](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks)
- [Microsoft Security Incident Prediction](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction)
- [Security Breach Dataset](https://www.kaggle.com/datasets/xontoloyo/security-breachhh)
- [RBA Dataset](https://www.kaggle.com/datasets/dasgroup/rba-dataset)
- And more...

### How It Works

1. **Download**: Uses Kaggle API to download datasets (or manual download instructions)
2. **Process**: Dataset-specific processors map fields to standard features:
   - `violation_count`: Number/frequency of violations
   - `severity_max`: Maximum severity level
   - `recency_score`: How recent the event is
   - `frequency_score`: Frequency of similar events
   - `context_risk`: Contextual risk factors
3. **Merge**: All processed datasets are combined into `processed_train_data.json`
4. **Train**: BaselineMLClassifier is trained and evaluated on the merged data

### Kaggle API Setup

To enable automatic dataset downloads:

```bash
# Install Kaggle API
pip install kaggle

# Setup credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Output

The pipeline produces:
- `data/processed/*.json`: Individual processed datasets
- `processed_train_data.json`: Merged training data
- `models/candidates/baseline_model.json`: Trained model
- `models/candidates/baseline_metrics.json`: Validation metrics

See [TrainTestPipeline.md](TrainTestPipeline.md) for detailed training specifications.

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

See [TEST_RESULTS.md](TEST_RESULTS.md) for detailed test status and coverage information.

### Code Quality

This project uses Black for code formatting and Flake8 for linting:

```bash
black nethical/ tests/ examples/
flake8 nethical/ tests/ examples/
mypy nethical/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Project Documentation

- **[AUDIT.md](AUDIT.md)** - Comprehensive repository structure analysis and audit
- **[CHANGELOG.md](CHANGELOG.md)** - Detailed changelog with migration guides
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Repository refactoring overview
- **[TEST_RESULTS.md](TEST_RESULTS.md)** - Test suite status and coverage
- **[roadmap.md](roadmap.md)** - Development roadmap and future plans

### Key Documentation

- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - ML model training guide
- **[scripts/README.md](scripts/README.md)** - Training and testing scripts
- **[training/README.md](training/README.md)** - Advanced training features

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- AI Safety frameworks
- Agent monitoring systems  
- Ethical AI tools

---

**Nethical - Ensuring AI agents operate safely, ethically, and transparently. 🔒**
