# Explainable AI Layer

The explainability module provides transparency into Nethical's decision-making process by generating human-readable explanations for governance decisions, policy violations, and risk assessments.

## Overview

As specified in the roadmap (Phase 2.3), the Explainable AI Layer includes:

- **Decision Explanations**: Why a particular decision (ALLOW, BLOCK, etc.) was made
- **Natural Language Generation**: Human-readable explanations for all users
- **Transparency Reports**: Comprehensive reports for auditing and compliance

## Components

### 1. DecisionExplainer

Analyzes governance decisions and generates structured explanations.

```python
from nethical.explainability import DecisionExplainer

explainer = DecisionExplainer()

# Generate explanation for a decision
explanation = explainer.explain_decision(
    decision="BLOCK",
    context={"agent_id": "agent_123", "action_type": "request"},
    violated_rules=[
        {
            "name": "no_malicious_content",
            "severity": "critical",
            "description": "Prevents malicious content"
        }
    ],
    risk_scores={
        "security": 0.8,
        "privacy": 0.6,
        "manipulation": 0.4
    },
    policy_matches=[
        {
            "name": "security_policy",
            "action": "BLOCK",
            "conditions": ["high_risk", "critical_violation"]
        }
    ]
)

# Access explanation components
print(f"Decision: {explanation.decision}")
print(f"Summary: {explanation.summary}")
print(f"Confidence: {explanation.confidence}")

# View reasoning chain
for step in explanation.reasoning_chain:
    print(f"- {step}")

# Export to JSON
json_data = explainer.explain_to_json(explanation)
```

#### Explanation Components

- **Summary**: High-level explanation of the decision
- **Components**: Detailed breakdown of factors (rules, risks, policies)
- **Confidence**: How confident the system is in the explanation
- **Reasoning Chain**: Step-by-step decision process
- **Contributing Factors**: Weighted factors that influenced the decision
- **Alternative Outcomes**: What could have happened differently

### 2. NaturalLanguageGenerator

Converts technical decisions into clear, human-readable text.

```python
from nethical.explainability import NaturalLanguageGenerator

generator = NaturalLanguageGenerator(tone="professional")

# Generate natural language explanation
nl_explanation = generator.generate_explanation(
    decision="BLOCK",
    context={"action_type": "user_request"},
    components=[
        {
            "type": "rule_based",
            "weight": 0.8,
            "details": {
                "count": 2,
                "violated_rules": [...]
            }
        }
    ],
    reasoning_chain=["Step 1", "Step 2", "Step 3"]
)

# Access human-readable text
print(nl_explanation.title)
print(nl_explanation.summary)
print(nl_explanation.detailed_explanation)

# View key points
for point in nl_explanation.key_points:
    print(f"• {point}")

# View recommendations
for rec in nl_explanation.recommendations:
    print(f"→ {rec}")

# Export to markdown
markdown = generator.to_markdown(nl_explanation)

# Export to HTML
html = generator.to_html(nl_explanation)
```

#### Tones Available

- **professional**: Formal, business-appropriate language
- **casual**: Friendly, approachable language (future)
- **technical**: Detailed technical information (future)

### 3. TransparencyReportGenerator

Creates comprehensive transparency reports for auditing and compliance.

```python
from nethical.explainability import TransparencyReportGenerator
from datetime import datetime

generator = TransparencyReportGenerator(include_sensitive_data=False)

# Generate report for the last 30 days
report = generator.generate_report(
    decisions=[...],  # List of governance decisions
    violations=[...],  # List of detected violations
    policies=[...],    # List of active policies
    period_days=30
)

# Access report data
print(f"Report ID: {report.report_id}")
print(f"Period: {report.period_start} to {report.period_end}")
print(f"Total Decisions: {report.summary['total_decisions']}")
print(f"Block Rate: {report.summary['block_rate']:.1f}%")

# View key insights
for insight in report.key_insights:
    print(f"💡 {insight}")

# View recommendations
for rec in report.recommendations:
    print(f"✓ {rec}")

# Check policy effectiveness
for policy, score in report.policy_effectiveness.items():
    print(f"{policy}: {score:.1%} effective")

# Export to JSON
json_report = generator.to_json(report)

# Export to markdown
markdown_report = generator.to_markdown(report)
```

#### Report Contents

- **Summary Statistics**: Total decisions, violations, rates
- **Decision Breakdown**: By type, category, agent, time
- **Violation Trends**: Daily trends over the reporting period
- **Policy Effectiveness**: How well each policy is performing
- **Key Insights**: Automatically detected patterns and anomalies
- **Recommendations**: Actionable suggestions for improvement

## Integration with IntegratedGovernance

The explainability layer can be integrated with the main governance system:

```python
from nethical.core import IntegratedGovernance
from nethical.explainability import DecisionExplainer, NaturalLanguageGenerator

# Initialize governance
gov = IntegratedGovernance()
explainer = DecisionExplainer()
nl_generator = NaturalLanguageGenerator()

# Process action and get explanation
result = gov.process_action(
    agent_id="agent_123",
    action="User request",
    violation_detected=True,
    violation_type="security",
    violation_severity="high"
)

# Generate explanation
explanation = explainer.explain_decision(
    decision=result.get("decision", "ALLOW"),
    context={"agent_id": "agent_123"},
    violated_rules=result.get("violated_rules", []),
    risk_scores=result.get("risk_scores", {})
)

# Convert to natural language
nl_explanation = nl_generator.generate_explanation(
    decision=explanation.decision,
    context={},
    components=[explainer.explain_to_json(explanation)["components"][0]],
    reasoning_chain=explanation.reasoning_chain
)

# Display to user
print(nl_explanation.summary)
```

## Use Cases

### 1. End User Explanations

When a user's request is blocked, provide clear explanation:

```python
nl_explanation = generator.generate_explanation(...)
print(f"❌ {nl_explanation.title}")
print(nl_explanation.summary)
print("\nWhat you can do:")
for rec in nl_explanation.recommendations:
    print(f"  • {rec}")
```

### 2. Reviewer Dashboard

Help human reviewers understand escalated cases:

```python
explanation = explainer.explain_decision(...)
print(f"Case: {explanation.decision}")
print(f"Confidence: {explanation.confidence:.1%}")
print("\nFactors:")
for factor, weight in explanation.contributing_factors.items():
    print(f"  • {factor}: {weight:.2f}")
```

### 3. Audit Trail

Generate detailed audit reports for compliance:

```python
report = generator.generate_report(
    decisions=all_decisions,
    violations=all_violations,
    policies=active_policies,
    period_days=90
)
with open("transparency_report_q1.md", "w") as f:
    f.write(generator.to_markdown(report))
```

### 4. Policy Tuning

Analyze policy effectiveness to improve governance:

```python
report = generator.generate_report(...)
ineffective_policies = [
    name for name, score in report.policy_effectiveness.items()
    if score < 0.5
]
print(f"Policies to review: {', '.join(ineffective_policies)}")
```

## API Reference

### DecisionExplainer

```python
class DecisionExplainer:
    def __init__(self, verbose: bool = False) -> None
    
    def explain_decision(
        self,
        decision: str,
        context: Dict[str, Any],
        violated_rules: Optional[List[Dict[str, Any]]] = None,
        risk_scores: Optional[Dict[str, float]] = None,
        policy_matches: Optional[List[Dict[str, Any]]] = None
    ) -> DecisionExplanation
    
    def explain_to_json(self, explanation: DecisionExplanation) -> Dict[str, Any]
```

### NaturalLanguageGenerator

```python
class NaturalLanguageGenerator:
    def __init__(self, language: str = "en", tone: str = "professional") -> None
    
    def generate_explanation(
        self,
        decision: str,
        context: Dict[str, Any],
        components: List[Dict[str, Any]],
        reasoning_chain: List[str]
    ) -> NaturalLanguageExplanation
    
    def to_markdown(self, explanation: NaturalLanguageExplanation) -> str
    def to_html(self, explanation: NaturalLanguageExplanation) -> str
```

### TransparencyReportGenerator

```python
class TransparencyReportGenerator:
    def __init__(self, include_sensitive_data: bool = False) -> None
    
    def generate_report(
        self,
        decisions: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
        period_days: int = 30
    ) -> TransparencyReport
    
    def to_json(self, report: TransparencyReport) -> str
    def to_markdown(self, report: TransparencyReport) -> str
```

## Best Practices

1. **Always Provide Context**: Include relevant context when generating explanations
2. **Choose Appropriate Tone**: Use professional tone for business users, technical for developers
3. **Generate Regular Reports**: Create transparency reports monthly for trend analysis
4. **Act on Insights**: Review key insights and implement recommendations
5. **Customize for Audience**: Tailor explanations to the target audience (users, reviewers, auditors)

## Future Enhancements

As noted in the roadmap:

- [ ] Integrate SHAP/LIME for ML model explanations
- [ ] Create decision tree visualization
- [ ] Add "explain this decision" API endpoint
- [ ] Support multiple languages
- [ ] Add interactive explanation UI (when UI work begins)

## References

- Roadmap: Phase 2.3 - Implement Explainable AI Layer
- Related: Phase 2.2 - Human-in-the-Loop Interface (backend complete)
- See also: `nethical/core/` for integration points
