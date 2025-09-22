# 🔒 Nethical: AI Safety Governance System

**Nethical** is a comprehensive safety governance system designed to monitor and evaluate AI agent actions for safety, ethical compliance, and manipulation detection. It provides real-time oversight and judgment capabilities to ensure AI agents operate within defined safety boundaries.

## ✨ Features

- **🎯 Intent vs Action Monitoring**: Detects deviations between an agent's stated intent and actual actions
- **⚖️ Ethical Violation Detection**: Identifies harmful content, deceptive behavior, privacy violations, and discrimination
- **🛡️ Safety Constraint Enforcement**: Monitors for unauthorized system access, data modification, and resource abuse
- **🕵️ Manipulation Recognition**: Detects emotional manipulation, authority abuse, social proof, scarcity tactics, and reciprocity exploitation
- **🏛️ Judge System**: Powerful decision-making component that evaluates actions and provides feedback with restrictions or case closure
- **📊 Comprehensive Reporting**: Detailed violation summaries and judgment analytics
- **⚙️ Configurable Monitoring**: Flexible configuration for different use cases and security levels

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

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

### Running the Example

```bash
python examples/basic_usage.py
```

## 🏗️ Architecture

Nethical consists of several key components:

### Core Components

- **SafetyGovernance**: Main orchestrator that coordinates all monitoring and judgment activities
- **AgentAction**: Data model representing agent intentions and actions
- **SafetyViolation**: Model for detected violations with severity levels
- **JudgmentResult**: Result of judge evaluation with decisions and feedback

### Monitoring System

- **IntentDeviationMonitor**: Compares stated intentions with actual actions
- **EthicalViolationDetector**: Identifies ethical constraint violations
- **SafetyViolationDetector**: Detects safety-related violations
- **ManipulationDetector**: Recognizes various manipulation techniques

### Judge System

- **SafetyJudge**: Evaluates actions and violations to make informed decisions:
  - **ALLOW**: Action is safe to proceed
  - **RESTRICT**: Action allowed with limitations
  - **BLOCK**: Action prevented due to safety concerns
  - **TERMINATE**: Critical violation requiring immediate cessation

## 🔧 Configuration

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

## 🧪 Testing

Run the test suite:

```bash
pip install -e .[dev]
pytest tests/
```

## 📊 Monitoring and Analytics

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

## 🛠️ Development

### Setup Development Environment

```bash
git clone <repository-url>
cd nethical
pip install -e .[dev]
```

### Code Style

This project uses Black for code formatting and Flake8 for linting:

```bash
black nethical/ tests/ examples/
flake8 nethical/ tests/ examples/
```

### Type Checking

```bash
mypy nethical/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 Related Projects

- AI Safety frameworks
- Agent monitoring systems
- Ethical AI tools

---

**Nethical** - Ensuring AI agents operate safely, ethically, and transparently. 🔒
