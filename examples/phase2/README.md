# Phase 2: Mature Ethical and Safety Framework

This directory contains examples and documentation for Phase 2 enhancements to Nethical's ethical and safety framework.

## Overview

Phase 2 introduces four major enhancements:

1. **Enhanced Ethical Taxonomy (2.1)** - Industry-specific taxonomies and validation
2. **Human-in-the-Loop Interface (2.2)** - API for review dashboard and case management
3. **Explainable AI Layer (2.3)** - Decision explanations and transparency reporting
4. **Formalized Policy Language (2.4)** - Validation, simulation, and impact analysis

## Quick Start

### 1. Enhanced Ethical Taxonomy

Create and validate industry-specific taxonomies:

```python
from nethical.api.taxonomy_api import TaxonomyAPI

api = TaxonomyAPI("ethics_taxonomy.json")

# Create healthcare taxonomy
result = api.create_industry_taxonomy_endpoint("healthcare", "1.0")

# Validate taxonomy
validation = api.validate_taxonomy_endpoint()

# Tag a violation
tags = api.tag_violation_endpoint("unauthorized_medical_record_access", {
    "sensitive": True,
    "personal_data": True
})
```

**Features:**
- Industry-specific dimensions (healthcare, finance, education)
- JSON Schema validation
- Coverage tracking and reporting
- Versioned taxonomy management

**Try the demo:**
```bash
python examples/phase2/taxonomy_api_demo.py
```

### 2. Human-in-the-Loop Interface

Manage escalation queue and case reviews:

```python
from nethical.api.hitl_api import HITLReviewAPI

api = HITLReviewAPI(storage_dir="./hitl_data")

# Get escalation queue
queue = api.get_escalation_queue_endpoint(status="pending", limit=10)

# Get case details
case = api.get_case_details_endpoint(judgment_id="jdg_123")

# Submit review
review = api.submit_review_endpoint(
    judgment_id="jdg_123",
    reviewer_id="reviewer_001",
    feedback_tags=["correct_decision"],
    rationale="Decision was appropriate given the context"
)

# Get SLA metrics
metrics = api.get_sla_metrics_endpoint()
```

**Features:**
- Escalation queue management
- Case review workflow
- Reviewer statistics and SLA tracking
- Feedback collection for ML improvement

### 3. Explainable AI Layer

Generate explanations for governance decisions:

```python
from nethical.api.explainability_api import ExplainabilityAPI

api = ExplainabilityAPI()

# Explain a decision
judgment_data = {
    'decision': 'BLOCK',
    'violations': [{'type': 'pii_exposure'}],
    'risk_score': 0.9,
    'matched_rules': [{'id': 'data-protection', 'priority': 100}]
}

explanation = api.explain_decision_endpoint('BLOCK', judgment_data)
print(explanation['data']['natural_language'])
# Output: "Action was BLOCKED because pii_exposure. Contributing factors include: ..."

# Generate decision tree
tree = api.get_decision_tree_endpoint(judgment_data)

# Create transparency report
report = api.generate_transparency_report_endpoint(decisions, "last_24h")
```

**Features:**
- Natural language explanations
- Decision tree visualization
- Policy match explanations
- Transparency reporting

**Try the demo:**
```bash
python examples/phase2/explainability_api_demo.py
```

### 4. Formalized Policy Language

Validate, simulate, and analyze policy changes:

```python
from nethical.core.policy_formalization import (
    PolicyValidator,
    PolicySimulator,
    PolicyImpactAnalyzer
)

# Validate policy
validator = PolicyValidator()
result = validator.validate_policy(policy_config)
print(f"Valid: {result.valid}")

# Lint for best practices
suggestions = validator.lint_policy(policy_config)

# Simulate policy
simulator = PolicySimulator()
results = simulator.simulate_policy(policy, test_cases)
print(f"Block rate: {results['decision_percentages']['BLOCK']:.1f}%")

# Analyze impact of changes
analyzer = PolicyImpactAnalyzer()
impact = analyzer.analyze_impact(current_policy, new_policy, historical_data)
print(f"Risk level: {impact.risk_level}")
```

**Features:**
- EBNF grammar specification
- Policy validation and linting
- Dry-run simulation
- Impact analysis before deployment

**Try the demo:**
```bash
python examples/phase2/policy_formalization_demo.py
```

## Architecture

### Taxonomy System

```
TaxonomyValidator
├── JSON Schema validation
├── Semantic validation
└── Coverage tracking

IndustryTaxonomyManager
├── Healthcare dimensions (patient_safety, medical_privacy)
├── Finance dimensions (financial_harm, regulatory_compliance)
└── Education dimensions (learning_integrity, student_welfare)

TaxonomyAPI
├── validate_taxonomy_endpoint()
├── create_industry_taxonomy_endpoint()
├── tag_violation_endpoint()
└── get_coverage_report_endpoint()
```

### Explainability System

```
DecisionExplainer
├── Extract primary reason
├── Identify contributing factors
├── Generate natural language
└── Create decision tree

TransparencyReportGenerator
├── Aggregate decisions
├── Calculate statistics
└── Sample explanations

ExplainabilityAPI
├── explain_decision_endpoint()
├── explain_policy_match_endpoint()
├── get_decision_tree_endpoint()
└── generate_transparency_report_endpoint()
```

### HITL System

```
EscalationQueue (existing backend)
├── Priority-based queue
├── SLA tracking
└── Feedback collection

HITLReviewAPI (new)
├── get_escalation_queue_endpoint()
├── get_case_details_endpoint()
├── submit_review_endpoint()
├── get_reviewer_stats_endpoint()
└── get_sla_metrics_endpoint()
```

### Policy Formalization

```
PolicyValidator
├── Structure validation
├── Best practice linting
└── Region overlay validation

PolicySimulator
├── Dry-run execution
├── Rule matching
└── Decision distribution

PolicyImpactAnalyzer
├── Compare policies
├── Estimate block/restrict rates
└── Risk assessment

PolicyGrammarEBNF
└── Formal grammar specification
```

## API Documentation

### Taxonomy API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `validate_taxonomy_endpoint(taxonomy_data)` | POST | Validate taxonomy configuration |
| `get_schema_endpoint()` | GET | Get JSON schema |
| `export_schema_endpoint(output_path)` | POST | Export schema to file |
| `list_industries_endpoint()` | GET | List available industries |
| `create_industry_taxonomy_endpoint(industry, version)` | POST | Create industry taxonomy |
| `get_coverage_stats_endpoint()` | GET | Get coverage statistics |
| `tag_violation_endpoint(violation_type, context)` | POST | Tag violation with dimensions |

### Explainability API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `explain_decision_endpoint(decision, judgment_data)` | POST | Explain a decision |
| `explain_policy_match_endpoint(matched_rule, facts)` | POST | Explain policy match |
| `get_decision_tree_endpoint(judgment_data)` | GET | Get decision tree viz |
| `generate_transparency_report_endpoint(decisions, period)` | POST | Generate transparency report |

### HITL Review API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `get_escalation_queue_endpoint(status, priority, limit)` | GET | Get escalation queue |
| `get_case_details_endpoint(judgment_id)` | GET | Get case details |
| `submit_review_endpoint(judgment_id, reviewer_id, ...)` | POST | Submit review |
| `get_reviewer_stats_endpoint(reviewer_id)` | GET | Get reviewer statistics |
| `get_sla_metrics_endpoint()` | GET | Get SLA metrics |
| `batch_assign_cases_endpoint(reviewer_id, count)` | POST | Batch assign cases |

## Testing

Run Phase 2 tests:

```bash
pytest tests/test_phase2_enhancements.py -v
```

All 31 tests should pass, covering:
- Taxonomy validation (6 tests)
- Industry taxonomies (4 tests)
- Decision explanation (4 tests)
- Policy validation (5 tests)
- Policy simulation (2 tests)
- API endpoints (10 tests)

## Integration with Existing System

Phase 2 components integrate seamlessly with existing Nethical infrastructure:

```python
from nethical.core import IntegratedGovernance
from nethical.api.taxonomy_api import TaxonomyAPI
from nethical.api.explainability_api import ExplainabilityAPI
from nethical.api.hitl_api import HITLReviewAPI

# Initialize governance
gov = IntegratedGovernance(
    storage_dir="./data",
    enable_ethical_taxonomy=True  # Use enhanced taxonomy
)

# Initialize Phase 2 APIs
taxonomy_api = TaxonomyAPI()
explainability_api = ExplainabilityAPI()
hitl_api = HITLReviewAPI()

# Process action
result = gov.process_action(action, agent_id)

# Explain decision
explanation = explainability_api.explain_decision_endpoint(
    decision=result['decision'],
    judgment_data=result
)

# Tag with taxonomy
tags = taxonomy_api.tag_violation_endpoint(
    violation_type=result.get('violations', [{}])[0].get('type'),
    context=result.get('context', {})
)

# Handle escalation if needed
if result.get('escalate'):
    queue = hitl_api.get_escalation_queue_endpoint()
```

## Next Steps

### Frontend Development (Future)

While the backend APIs are complete, a frontend dashboard would enhance usability:

1. **React/Vue.js Dashboard** for HITL review
   - Case queue visualization
   - Decision tree rendering
   - Review form with feedback tags
   - Real-time SLA metrics

2. **Taxonomy Management UI**
   - Visual taxonomy editor
   - Coverage heatmaps
   - Industry taxonomy comparison

3. **Policy Studio**
   - Visual policy editor
   - Live validation feedback
   - Simulation playground
   - Impact analysis charts

### ML Enhancement (Future)

- Integrate SHAP/LIME for ML model explanations
- Train explanation quality models
- Automated explanation improvement
- Feedback loop to ML models

## Documentation

- **Policy Engines Guide**: `/docs/policy_engines.md` - Comprehensive guide to dual policy engines
- **API Reference**: This README
- **Examples**: `/examples/phase2/` - Working code examples

## Support

For issues or questions:
1. Check the examples in this directory
2. Review test cases in `tests/test_phase2_enhancements.py`
3. Consult `/docs/policy_engines.md`
4. File an issue on GitHub

## Version History

- **v1.0.0** (2024-11-02): Initial Phase 2 implementation
  - Enhanced taxonomy with industry support
  - Explainability layer with natural language
  - HITL API for review workflow
  - Policy formalization with validation and simulation
