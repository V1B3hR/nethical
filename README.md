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
