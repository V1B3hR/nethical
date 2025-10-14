# Nethical 🔒

Safety, Ethics and More for AI Agents

Nethical is a comprehensive safety governance system designed to monitor and evaluate AI agent actions for safety, ethical compliance, and manipulation detection. It provides real-time oversight and judgment, plus ML-driven anomaly and drift detection, auditability, human-in-the-loop workflows, and continuous optimization.

## What’s New

Since the last update, Nethical has advanced in several important ways:

- ✅ Unified Integrated Governance: New `IntegratedGovernance` consolidates Phases 3, 4, 5–7, and 8–9 into a single interface.
- ✅ Phase 4 Features Implemented: Merkle anchoring, policy diff auditing, quarantine mode, ethical taxonomy, SLA monitoring.
- ✅ Privacy & Data Handling (F3): Differential privacy, redaction pipeline, data minimization, and federated analytics with regions/domains.
- ✅ Plugin Marketplace (F6 preview): Integration points and utilities for extensibility; simple `load_plugin` call in `IntegratedGovernance`.
- ✅ Documentation Refresh: Detailed implementation docs now live under `docs/implementation/`.
- ✅ Training & Testing Infrastructure: Matured training orchestrator and expanded tests/examples.

Legacy integration classes (Phase3/4/5–7/8–9) are still available but deprecated in favor of `IntegratedGovernance`.

## 🎯 What is Nethical?

Nethical serves as a guardian layer for AI systems, continuously monitoring agent behavior to ensure safe, ethical, and transparent operations. It acts as a real-time safety net that can detect, evaluate, and respond to risky or non-compliant behaviors while providing clear explanations, audit trails, and pathways for human oversight.

### Main Purpose

- Proactive Monitoring: Real-time surveillance of AI agent actions
- Ethical Compliance: Enforcing ethical guidelines and policies
- Safety Enforcement: Preventing harmful or dangerous behaviors
- Transparency: Clear insights into decision-making processes
- Trust Building: Confidence via robust oversight and auditability

## ✨ Key Features

- Intent vs Action Monitoring
- Ethical Violation Detection (harmful content, deception, privacy, discrimination)
- Safety Constraint Enforcement (unauthorized access, data modification, resource abuse)
- Manipulation Recognition (emotional manipulation, authority abuse, social proof, scarcity, reciprocity)
- Judge/Decision System with `ALLOW`, `RESTRICT`, `BLOCK`, `TERMINATE`
- ML-Based Anomaly Detection (trainable)
- Distribution Drift Monitoring (PSI & KL)
- Human-in-the-Loop Escalations and Feedback
- Continuous Optimization (thresholds, weights, promotion gate)
- Immutable Audit Trails (Merkle anchoring)
- Policy Diff Auditing and Quarantine Mode
- Ethical Taxonomy tagging and SLA tracking
- Privacy & Data Handling (differential privacy, redaction, data minimization)
- Plugin Marketplace integration points
- Comprehensive Reporting and Configurable Monitoring

## 🚀 Quick Start

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

Minimal usage with the unified class:

```python
from nethical.core import IntegratedGovernance

gov = IntegratedGovernance(
    storage_dir="./nethical_data",
    # Phase 3
    enable_performance_optimization=True,
    # Phase 4
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True,
    # Phase 5–7
    enable_shadow_mode=True,
    enable_ml_blending=True,
    enable_anomaly_detection=True,
    # Phase 8–9
    auto_escalate_on_block=True,
    auto_escalate_on_low_confidence=True
)

result = gov.process_action(
    agent_id="agent_123",
    action="User request processing",
    cohort="production",
    violation_detected=True,
    violation_type="safety",
    violation_severity="medium",
    # Optional ML inputs
    action_id="action_456",
    action_type="response",
    features={'ml_score': 0.7},
    rule_risk_score=0.65,
    rule_classification="warn"
)

print(f"Risk Score (Phase 3): {result['phase3']['risk_score']}")
print(f"Merkle Events (Phase 4): {result['phase4']['merkle']['event_count']}")
print(f"ML Shadow Score (Phase 5): {result['phase567']['shadow']['ml_risk_score']}")
print(f"Blended Risk (Phase 6): {result['phase567']['blended']['blended_risk_score']}")

status = gov.get_system_status()
```

Run the unified demo:

```bash
python examples/basic/unified_governance_demo.py
```

## 🧩 Privacy & Data Handling (F3)

New privacy controls and data-handling features:

```python
gov = IntegratedGovernance(
    storage_dir="./nethical_privacy",
    # Regions/domains for federated analytics
    region_id="us-east-1",
    logical_domain="customer-service",
    data_residency_policy="US_CCPA",
    # Privacy mode
    privacy_mode="differential",  # enables DP pipeline
    epsilon=1.0,
    # Redaction pipeline
    redaction_policy="standard"
)

# Example redaction
redacted = gov.redaction_pipeline.redact("Contact: admin@example.com")
print(redacted.redacted_text)
```

Features:
- Differential privacy with configurable epsilon
- Redaction pipeline policies
- Data minimization with retention and right-to-be-forgotten support
- Federated analytics aware of `region_id` and `logical_domain`

## 🏪 Plugin Marketplace (F6 – Extensibility)

Experiment with integrations and extensions:

```python
from nethical.core import IntegratedGovernance

governance = IntegratedGovernance()
ok = governance.load_plugin("example-plugin-id")
print("Plugin loaded:", ok)
```

The marketplace module includes:
- Integration directory and adapters
- Export/import utilities (JSON/CSV)
- Trust scoring and community review constructs

See examples under `examples/advanced/f6_marketplace_demo.py` (when available) and tests in `tests/test_f6_marketplace.py`.

## 🏗️ System Architecture

Core models:
- SafetyGovernance (legacy), AgentAction, SafetyViolation, JudgmentResult

Monitoring systems:
- IntentDeviationMonitor, EthicalViolationDetector, SafetyViolationDetector
- ManipulationDetector
- AnomalyDriftMonitor (sequence + behavioral + drift)
- AnomalyMLClassifier (trainable)

Decision Engine:
- SafetyJudge with `ALLOW`, `RESTRICT`, `BLOCK`, `TERMINATE`

Unified Orchestration:
- `IntegratedGovernance` combines risk, audit/integrity, ML/anomaly, and human-in-the-loop/optimization flows.

## ⚙️ Configuration

```python
from nethical import SafetyGovernance, MonitoringConfig

config = MonitoringConfig(
    intent_deviation_threshold=0.8,
    enable_ethical_monitoring=True,
    enable_safety_monitoring=True,
    enable_manipulation_detection=True,
    max_violation_history=1000
)

governance = SafetyGovernance(config)  # Legacy API still available
```

Prefer `IntegratedGovernance` for new integrations; legacy APIs remain for backward compatibility.

## 📊 Analytics and Reporting

```python
violation_summary = governance.get_violation_summary()
print(violation_summary)

judgment_summary = governance.get_judgment_summary()
print(judgment_summary)

status = governance.get_system_status()
print(status)
```

## Phase Overview

### ✅ Phase 3: Advanced Features (Complete)
- Risk Engine with decay
- Correlation Engine (multi-agent pattern detection)
- Fairness Sampler (cohorts)
- Ethical Drift Reporter
- Performance Optimizer (risk-based gating)

See docs: `docs/implementation/PHASE3_GUIDE.md`

### ✅ Phase 4: Integrity & Ethics Operationalization (Implemented)
- Merkle Anchoring (immutable audit logs)
- Policy Diff Auditing
- Quarantine Mode
- Ethical Taxonomy
- SLA Monitoring

See docs: `docs/implementation/PHASE4_GUIDE.md`

### ✅ Phases 5–7: ML & Anomaly Detection
- Phase 5: Shadow Mode (`MLShadowClassifier`)
- Phase 6: Blended Enforcement in gray zone (`MLBlendedRiskEngine`)
- Phase 7: Anomaly & Drift (`AnomalyDriftMonitor`)

See docs: `docs/implementation/PHASE5-7_GUIDE.md`

### ✅ Phases 8–9: Human-in-the-Loop & Optimization
- Escalation queue, labeling interface
- SLA metrics, priority queues
- Multi-objective optimization, promotion gates
- A/B testing and configuration lifecycle

See docs: `docs/implementation/PHASE89_GUIDE.md`

## 🎯 Unified Integrated Governance (Primary API)

```python
from nethical.core import IntegratedGovernance

gov = IntegratedGovernance(
    storage_dir="./nethical_data",
    region_id="eu-west-1",
    logical_domain="moderation",
    enable_performance_optimization=True,
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True,
    enable_shadow_mode=True,
    enable_ml_blending=True,
    enable_anomaly_detection=True,
    auto_escalate_on_block=True,
    auto_escalate_on_low_confidence=True
)
```

Results include per-phase outputs under keys: `phase3`, `phase4`, `phase567`, `phase89`.

## 🎓 ML Training Pipeline with Real-World Datasets

End-to-end orchestrator to download, process, and train models:

```bash
# Full pipeline
python scripts/baseline_orchestrator.py

# Steps
python scripts/baseline_orchestrator.py --download
python scripts/baseline_orchestrator.py --process-only
python scripts/baseline_orchestrator.py --train-only
```

Datasets list: see `datasets/datasets` (Kaggle).  
Outputs:
- `data/processed/*.json`: per-dataset processed artifacts
- `processed_train_data.json`: merged training data
- `models/candidates/*`: candidate models + metrics

Detailed specs: `TrainTestPipeline.md`

## 🧪 Examples

See `examples/README.md` for a full catalogue:

- Basic: `basic_usage.py`, `unified_governance_demo.py`, `custom_detectors.py`
- Governance: `phase3_demo.py`, `phase4_demo.py`, `phase5_demo.py`, `phase6_demo.py`, `phase7_demo.py`, `phase567_demo.py`, `phase89_demo.py`
- Training: `train_anomaly_detector.py`, `train_with_drift_tracking.py`, `real_data_training_demo.py`, `correlation_model_demo.py`
- Advanced (future tracks): regionalization/sharding, ext/marketplace, privacy, adaptive tuning, replay/simulation

## 🧭 Documentation

Implementation index: `docs/implementation/README.md`

Key documents:
- `docs/implementation/PHASE3_GUIDE.md`
- `docs/implementation/PHASE4_GUIDE.md`
- `docs/implementation/PHASE5-7_GUIDE.md`
- `docs/implementation/PHASE89_GUIDE.md`
- `docs/implementation/ANOMALY_DETECTION_TRAINING.md`
- `docs/TRAINING_GUIDE.md`
- `scripts/README.md`
- `training/README.md`
- Project audits and summaries: `AUDIT.md`, `CHANGELOG.md`, `REFACTORING_SUMMARY.md`, `TEST_RESULTS.md`
- Roadmap: `roadmap.md`

Note: Some links were previously root-relative; most detailed implementation docs now live under `docs/implementation/`.

## 🔭 Future Tracks (11–50 Systems)

- F1: Regionalization & Sharding (region-aware analytics present)
- F2: Detector & Policy Extensibility (marketplace and adapters present)
- F3: Privacy & Data Handling (implemented features available)
- F4: Thresholds, Tuning & Adaptivity
- F5: Simulation & Replay
- F6: Marketplace & Ecosystem (integration points and tests available)

See: `roadmap.md` for complete specifications.

## 🛠️ Development

```bash
git clone <repository-url>
cd nethical
pip install -e .[dev]
pytest tests/
```

Formatting and linting:

```bash
black nethical/ tests/ examples/
flake8 nethical/ tests/ examples/
mypy nethical/
```

## ⚠️ Known Gaps (from latest test artifacts)

- Advanced adversarial/limits scenarios have failures:
  - Privacy violation (data harvesting) not blocked in a scenario
  - Volume and memory exhaustion handling assertions
  - Context confusion and NLP manipulation scenarios
  - Multi-step manipulation correlation; “perfect storm” end-to-end
- Use these to prioritize detector/policy improvements and tuning.

Artifacts are under `tests/results/individual/*.txt` with timestamps and JSON report paths for deeper investigation.

## 🤝 Contributing

Contributions are welcome! Please open a Pull Request.

## 📄 License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

---

Nethical — Ensuring AI agents operate safely, ethically, and transparently. 🔒
