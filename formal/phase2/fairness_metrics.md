# Fairness Metrics Baseline

## Overview
This document establishes the fairness metrics, protected attributes, statistical tests, and thresholds for Nethical's governance system. Fairness monitoring ensures equitable treatment across population groups and compliance with anti-discrimination regulations.

---

## Protected Attributes

### Primary Protected Attributes (Critical Priority)

| Attribute | Description | Regulatory Basis | Example Values |
|-----------|-------------|------------------|----------------|
| **race** | Racial or ethnic group | Civil Rights Act, ECOA, FHA | white, black, asian, hispanic, other |
| **gender** | Gender identity | Title VII, ECOA, FHA | male, female, non-binary, other |
| **age** | Age group | ADEA, Age Discrimination Act | <18, 18-24, 25-34, 35-44, 45-54, 55-64, 65+ |
| **disability_status** | Physical/mental disability | ADA, Section 504 | none, physical, mental, both |
| **national_origin** | Country of origin | Civil Rights Act, Immigration law | us, mexico, india, china, other |
| **religion** | Religious affiliation | Title VII, Religious Freedom Acts | christian, muslim, jewish, hindu, buddhist, atheist, other |

### Secondary Protected Attributes (High Priority)

| Attribute | Description | Regulatory Basis | Example Values |
|-----------|-------------|------------------|----------------|
| **veteran_status** | Military service history | VEVRAA, USERRA | veteran, non-veteran |
| **marital_status** | Marital/family status | Various state laws | single, married, divorced, widowed |
| **sexual_orientation** | Sexual identity | Various state/local laws, EEOC guidance | heterosexual, homosexual, bisexual, other |
| **genetic_information** | Genetic test results | GINA | none, predisposition_present |

### Domain-Specific Attributes

#### Financial Services
- **credit_history**: poor, fair, good, excellent
- **zip_code**: (proxy for socioeconomic status; requires fairness analysis)
- **income_level**: <25k, 25k-50k, 50k-75k, 75k-100k, >100k

#### Healthcare
- **pre_existing_conditions**: none, chronic, acute
- **genetic_predisposition**: none, present

#### Employment
- **education_level**: high_school, associates, bachelors, masters, phd
- **employment_gaps**: none, <6mo, 6mo-1yr, >1yr

#### Housing
- **source_of_income**: employment, section8, disability, pension, other

---

## Fairness Metrics

### Metric 1: Statistical Parity (SP)

**Definition**: Decision rates should be similar across protected groups.

**Formula**:
```
SP_difference = |P(judgment = ALLOW | group = A) - P(judgment = ALLOW | group = B)|
```

**Threshold**: ≤ 0.10 (10 percentage points)

**Example**:
- Group A (male): 1000 decisions, 800 ALLOW → 80% allow rate
- Group B (female): 1000 decisions, 700 ALLOW → 70% allow rate
- SP_difference = |0.80 - 0.70| = 0.10 ✅ **Within threshold**

**Interpretation**:
- SP_difference ≤ 0.10: **Compliant** (no disparate impact)
- 0.10 < SP_difference ≤ 0.15: **Warning** (monitor closely)
- SP_difference > 0.15: **Non-compliant** (disparate impact; corrective action required)

---

### Metric 2: Disparate Impact Ratio (DIR)

**Definition**: Ratio of favorable outcome rates between groups.

**Formula**:
```
DIR = P(judgment = ALLOW | protected_group) / P(judgment = ALLOW | reference_group)
```

**Threshold**: ≥ 0.80 (80% rule from EEOC guidelines)

**Example**:
- Reference group (white): 1000 decisions, 850 ALLOW → 85% allow rate
- Protected group (black): 1000 decisions, 680 ALLOW → 68% allow rate
- DIR = 0.68 / 0.85 = 0.80 ✅ **At threshold (marginal)**

**Interpretation**:
- DIR ≥ 0.80: **Compliant** (80% rule satisfied)
- 0.70 ≤ DIR < 0.80: **Warning** (below threshold but not critical)
- DIR < 0.70: **Non-compliant** (significant disparate impact)

---

### Metric 3: Equal Opportunity Difference (EOD)

**Definition**: True positive rates should be similar across groups.

**Formula**:
```
EOD = |TPR(group = A) - TPR(group = B)|

where TPR(group) = P(judgment = ALLOW | group, ground_truth = should_allow)
```

**Threshold**: ≤ 0.10

**Use Case**: Relevant when ground truth labels available (e.g., human-reviewed decisions)

**Example**:
- Group A: TPR = 0.90 (correctly allowed 90% of legitimate actions)
- Group B: TPR = 0.85 (correctly allowed 85% of legitimate actions)
- EOD = |0.90 - 0.85| = 0.05 ✅ **Within threshold**

---

### Metric 4: Average Odds Difference (AOD)

**Definition**: Combination of true positive rate and false positive rate differences.

**Formula**:
```
AOD = 0.5 * (|TPR(A) - TPR(B)| + |FPR(A) - FPR(B)|)

where FPR(group) = P(judgment = ALLOW | group, ground_truth = should_block)
```

**Threshold**: ≤ 0.10

**Use Case**: Comprehensive fairness measure when ground truth available

---

### Metric 5: Counterfactual Fairness (CF)

**Definition**: Decision should remain stable if protected attribute were changed, all else equal.

**Formula**:
```
CF_stability = P(judgment(context) == judgment(context_with_attribute_changed))
```

**Threshold**: ≥ 0.95 (95% stability)

**Evaluation Procedure**:
1. For each protected attribute, create counterfactual context (flip attribute value)
2. Re-evaluate action with counterfactual context
3. Compare judgments (original vs counterfactual)
4. Compute stability rate

**Example**:
- Original: {gender: male, ...} → judgment = ALLOW
- Counterfactual: {gender: female, ...} → judgment = ALLOW
- If 950 out of 1000 counterfactuals match original: CF_stability = 0.95 ✅

**Note**: Counterfactual analysis requires careful consideration of causal relationships; attribute change may legitimately affect decision (e.g., gender-specific medical policy).

---

## Statistical Significance Testing

### Chi-Square Test for Statistical Parity
**Null Hypothesis**: No association between protected attribute and judgment  
**Alternative**: Association exists (disparate impact)

**Test Procedure**:
```python
from scipy.stats import chi2_contingency

contingency_table = [
    [allow_count_groupA, block_count_groupA],
    [allow_count_groupB, block_count_groupB]
]

chi2, p_value, dof, expected = chi2_contingency(contingency_table)

if p_value < 0.05:
    print("Statistically significant disparity detected")
else:
    print("No significant disparity")
```

### Confidence Intervals
For each metric, compute 95% confidence interval using bootstrap or normal approximation:
```
CI = metric ± 1.96 * SE

where SE = sqrt(metric * (1 - metric) / n)
```

### Minimum Sample Size
Require minimum sample size for valid statistical inference:
- **Minimum per group**: 100 decisions
- **Recommended per group**: 1000 decisions
- **High-confidence analysis**: 10,000 decisions

**Error Handling**:
- If sample size < minimum: Mark metric as "Insufficient Data"
- If confidence interval overlaps threshold: Mark as "Marginal" (requires monitoring)

---

## Fairness Computation Schedule

### Batch Processing
**Frequency**: Monthly (first day of month)  
**Reason**: Balance timeliness with statistical power (accumulate sufficient samples)

**Computation Pipeline**:
1. Extract decisions from previous month (all tenants)
2. For each protected attribute:
   - For each pair of groups (reference vs protected):
     - Compute SP_difference, DIR
     - Run chi-square test
     - Compute confidence intervals
3. For subset with ground truth labels:
   - Compute EOD, AOD
4. For random sample (1000 actions):
   - Perform counterfactual analysis (CF_stability)
5. Store metrics in database
6. Generate fairness report
7. Trigger alerts if non-compliance detected

### Real-Time Monitoring (Optional)
For high-risk domains, compute approximate fairness metrics in real-time:
- Use sliding window (last 10,000 decisions)
- Update incrementally on each new decision
- Alert if metric crosses threshold

---

## Fairness Thresholds Summary

| Metric | Threshold | Compliance Status |
|--------|-----------|-------------------|
| **Statistical Parity Difference** | ≤ 0.10 | ✅ Compliant if within threshold |
| **Disparate Impact Ratio** | ≥ 0.80 | ✅ Compliant if above threshold |
| **Equal Opportunity Difference** | ≤ 0.10 | ✅ Compliant if within threshold |
| **Average Odds Difference** | ≤ 0.10 | ✅ Compliant if within threshold |
| **Counterfactual Fairness Stability** | ≥ 0.95 | ✅ Compliant if above threshold |

---

## Non-Compliance Response

### Alert Workflow
When metric breaches threshold:
1. **Automated Alert**: Email/Slack notification to governance team
2. **Escalation Ticket**: Create ticket with SLA (resolution within 7 days)
3. **Freeze Policy**: Optionally freeze affected policy (enter quarantine mode)
4. **Root Cause Analysis**: Investigate contributing policies, data biases, feature correlations
5. **Corrective Action**: Recalibrate thresholds, retrain models, revise policies
6. **Verification**: Re-compute metrics after corrective action

### Escalation Criteria
- **Critical**: Metric exceeds threshold by >50% (e.g., SP_difference > 0.15, DIR < 0.70)
- **High**: Metric exceeds threshold by <50% (e.g., 0.10 < SP_difference ≤ 0.15)
- **Medium**: Metric marginal (within confidence interval of threshold)

---

## Fairness Report Format

### Monthly Fairness Report
```json
{
  "report_id": "string",
  "report_period": "2025-10-01 to 2025-10-31",
  "tenant_id": "string",
  "protected_attributes_analyzed": ["race", "gender", "age", "disability_status"],
  "total_decisions_analyzed": 50000,
  "metrics": [
    {
      "protected_attribute": "gender",
      "reference_group": "male",
      "protected_group": "female",
      "metrics": {
        "sp_difference": 0.08,
        "sp_ci": [0.06, 0.10],
        "dir": 0.92,
        "dir_ci": [0.89, 0.95],
        "eod": 0.05,
        "aod": 0.06,
        "cf_stability": 0.97
      },
      "chi_square_p_value": 0.12,
      "sample_size_per_group": {
        "male": 25000,
        "female": 25000
      },
      "compliant": true,
      "alert_triggered": false
    },
    {
      "protected_attribute": "race",
      "reference_group": "white",
      "protected_group": "black",
      "metrics": {
        "sp_difference": 0.12,
        "sp_ci": [0.09, 0.15],
        "dir": 0.78,
        "dir_ci": [0.74, 0.82]
      },
      "chi_square_p_value": 0.001,
      "sample_size_per_group": {
        "white": 30000,
        "black": 5000
      },
      "compliant": false,
      "alert_triggered": true,
      "escalation_ticket_id": "ESC-12345"
    }
  ],
  "summary": {
    "total_attribute_group_pairs": 20,
    "compliant_pairs": 18,
    "non_compliant_pairs": 2,
    "overall_compliance_rate": 0.90
  },
  "recommendations": [
    "Investigate race-based disparity for black vs white; DIR below 80% threshold",
    "Review policies contributing to race-based decisions",
    "Consider bias mitigation techniques (reweighting, calibration)"
  ]
}
```

---

## Bias Mitigation Strategies

### Pre-Processing (Data-Level)
- **Reweighting**: Adjust sample weights to balance protected groups
- **Oversampling/Undersampling**: Balance class distribution across groups
- **Data Augmentation**: Generate synthetic samples for underrepresented groups

### In-Processing (Algorithm-Level)
- **Fairness Constraints**: Add fairness constraints to optimization (e.g., demographic parity)
- **Adversarial Debiasing**: Train model to be invariant to protected attributes
- **Calibration**: Post-hoc recalibration of scores per group

### Post-Processing (Decision-Level)
- **Threshold Adjustment**: Use group-specific thresholds to achieve fairness
- **Reject Option Classification**: Defer marginal decisions to human review
- **Equalized Odds Post-Processing**: Adjust decisions to equalize TPR and FPR

### Nethical-Specific Mitigations
- **Policy Recalibration**: Adjust risk score thresholds per group
- **Context Field Auditing**: Identify and remove proxy features (e.g., zip code as race proxy)
- **Fairness-Aware Training**: Retrain ML models with fairness constraints

---

## Fairness Dashboard Mockup

**Visualization Components**:
1. **Metric Heatmap**: Protected attributes × groups × metrics (green = compliant, yellow = warning, red = non-compliant)
2. **Trend Line Chart**: SP_difference over time (monthly)
3. **Group Comparison Bar Chart**: Allow rates per group
4. **Confidence Interval Plot**: Metrics with CI bands
5. **Alert Panel**: Recent non-compliance alerts and escalation status

---

## Integration with Phase 5B (Fairness Test Harness)

Phase 5B will implement:
1. Automated fairness metric computation (monthly batch job)
2. Real-time approximate fairness tracking (sliding window)
3. Fairness dashboard (Grafana panels)
4. Alerting system (threshold breach notifications)
5. Bias mitigation toolkit (pre/in/post-processing methods)
6. Fairness report generation (JSON + PDF export)

---

## Success Criteria (Phase 2C)

Phase 2C (Fairness Criteria Baseline) is complete when:
1. ✅ Protected attributes defined for all domains
2. ✅ Fairness metrics specified (SP, DIR, EOD, AOD, CF)
3. ✅ Thresholds established per metric
4. ✅ Statistical significance testing procedures documented
5. ✅ Non-compliance response workflow defined
6. ✅ Fairness report format specified
7. ✅ Bias mitigation strategies cataloged

---

## Related Documents
- governance_drivers.md: Protected attributes and governance domains
- requirements.md: R-F008 (Protected Attribute Monitoring), G-002, G-004
- risk_register.md: R-004 (Fairness Metric Drift)
- compliance_matrix.md: ECOA, FHA, EEOC compliance

---

**Status**: ✅ Phase 2C Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Ethics Data Scientist / Governance Lead
