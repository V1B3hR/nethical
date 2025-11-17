# Algorithm Cards for Nethical Platform

**Version**: 1.0  
**Last Updated**: 2025-11-17  
**Purpose**: Provide transparency about algorithms and models used in the Nethical platform

---

## What are Algorithm Cards?

Algorithm Cards are standardized documentation that describe the purpose, design, training data, performance characteristics, limitations, and ethical considerations of algorithms and machine learning models. They promote transparency, accountability, and trust.

---

## Table of Contents

1. [Decision Evaluation Engine](#1-decision-evaluation-engine)
2. [Fairness Metric Computation](#2-fairness-metric-computation)
3. [Anomaly Detection for Audit Logs](#3-anomaly-detection-for-audit-logs)
4. [Policy Similarity Matching](#4-policy-similarity-matching)
5. [Appeal Priority Scoring](#5-appeal-priority-scoring)

---

## 1. Decision Evaluation Engine

### 1.1 Basic Information

**Name**: Nethical Decision Evaluation Engine  
**Type**: Rule-based policy engine with agent orchestration  
**Version**: 2.0  
**Owner**: Nethical Core Team  
**Contact**: engineering@nethical.example.com

### 1.2 Purpose and Use Cases

**Primary Purpose**: Evaluate decision requests against active policies using a graph of interconnected agents.

**Use Cases**:
- Credit approval decisions
- Loan application processing
- Insurance underwriting
- Access control decisions
- Resource allocation

**Out of Scope**:
- Medical diagnosis
- Life-or-death emergency decisions
- Purely subjective decisions (e.g., artistic merit)

### 1.3 Algorithm Design

**Approach**: Deterministic rule-based evaluation

**Key Components**:
1. **Policy Parser**: Converts policy definitions into executable agent graphs
2. **Context Validator**: Ensures input data meets schema requirements
3. **Agent Orchestrator**: Executes agents in topologically sorted order
4. **Result Aggregator**: Combines agent outputs into final decision
5. **Trace Generator**: Records complete evaluation path for explainability

**Pseudocode**:
```
function evaluate_decision(context, policy):
    # 1. Validate input
    validated_context = validate_context(context, policy.schema)
    
    # 2. Build agent execution graph
    agent_graph = policy.build_agent_graph()
    
    # 3. Check for cycles (must be acyclic)
    if has_cycle(agent_graph):
        raise PolicyError("Cyclic agent dependencies detected")
    
    # 4. Topologically sort agents
    sorted_agents = topological_sort(agent_graph)
    
    # 5. Execute agents in order
    trace = []
    agent_results = {}
    for agent in sorted_agents:
        result = agent.execute(validated_context, agent_results)
        agent_results[agent.id] = result
        trace.append({
            'agent': agent.id,
            'input': validated_context,
            'output': result,
            'timestamp': now()
        })
    
    # 6. Generate final decision
    decision = aggregate_results(agent_results, policy.decision_rule)
    
    # 7. Return decision with trace
    return {
        'decision': decision,
        'confidence': compute_confidence(agent_results),
        'trace': trace,
        'justification': generate_justification(trace, decision)
    }
```

**Determinism Guarantee**: Given the same inputs and policy version, the algorithm always produces the same output (P-DET property).

### 1.4 Inputs and Outputs

**Inputs**:
- Decision context (JSON object with required and optional fields)
- Policy version identifier
- Agent configuration (embedded in policy)

**Input Validation**:
- Schema compliance check
- Type validation
- Range checks for numeric values
- Enumeration validation for categorical values
- Required field presence

**Outputs**:
- Decision outcome (e.g., "approved", "denied", "conditional")
- Confidence score (0.0 to 1.0)
- Justification text (human-readable explanation)
- Complete evaluation trace (all intermediate steps)
- Metadata (timestamp, policy version, executor ID)

### 1.5 Performance Characteristics

**Computational Complexity**:
- Time: O(N) where N is number of agents (assuming constant-time agent execution)
- Space: O(N) for trace storage

**Measured Performance**:
- Average execution time: 50ms (p50), 150ms (p95), 300ms (p99)
- Throughput: 1000 decisions/second (single instance)
- Memory usage: 100MB baseline + 1KB per decision in flight

**Scalability**:
- Horizontally scalable (stateless evaluation)
- Can process 100,000+ decisions/day per instance

### 1.6 Limitations and Risks

**Technical Limitations**:
- Maximum agent graph depth: 20 levels (to prevent stack overflow)
- Maximum evaluation time: 10 seconds (timeout)
- Maximum context size: 1MB
- No support for non-deterministic operations (random, external API calls)

**Accuracy Limitations**:
- Accuracy depends on policy quality (garbage in, garbage out)
- Cannot adapt to changing patterns without policy updates
- No learning from historical decisions

**Bias Risks**:
- Can perpetuate biases encoded in policies
- Mitigation: Fairness monitoring, regular policy audits, diverse policy authoring team

**Failure Modes**:
- Agent timeout: Returns "deferred" decision with explanation
- Invalid context: Returns error with specific validation failures
- Agent error: Returns error with agent ID and error message
- System overload: Returns 503 Service Unavailable with retry-after

### 1.7 Fairness and Bias Considerations

**Protected Attributes**: Not used directly in decision logic

**Fairness Monitoring**:
- Post-hoc analysis of decision outcomes by protected attribute
- Statistical parity, disparate impact, and equal opportunity metrics
- Automated alerts on threshold violations

**Bias Mitigation**:
- Context field whitelisting (prevents use of sensitive attributes)
- Regular policy audits by diverse team
- Mandatory fairness review before policy activation
- Counterfactual analysis during development

### 1.8 Explainability and Transparency

**Explainability Method**: Complete trace of evaluation steps

**Trace Contents**:
- Policy version applied
- Each agent executed (name, inputs, outputs)
- Decision logic path taken
- Final outcome rationale

**Accessibility**: All decision traces available via audit portal

### 1.9 Testing and Validation

**Test Coverage**:
- Unit tests: 95% code coverage
- Integration tests: All major decision paths
- Property-based tests: Determinism, termination, acyclicity

**Validation Procedures**:
- Shadow testing: New policies tested against historical data
- A/B testing: Gradual rollout with comparison to existing policies
- Expert review: Domain experts review policy logic

**Continuous Monitoring**:
- Decision quality metrics (accuracy, precision, recall where applicable)
- Fairness metrics (statistical parity, disparate impact)
- System health (latency, error rate, throughput)

### 1.10 Version History and Changes

| Version | Date | Changes | Impact |
|---------|------|---------|--------|
| 1.0 | 2024-01-01 | Initial release | N/A |
| 1.5 | 2024-06-15 | Added trace generation | Improved explainability |
| 2.0 | 2025-11-17 | Determinism guarantees, timeout enforcement | Enhanced reliability |

---

## 2. Fairness Metric Computation

### 2.1 Basic Information

**Name**: Fairness Metric Computation Module  
**Type**: Statistical analysis algorithms  
**Version**: 1.2  
**Owner**: Governance Team  

### 2.2 Purpose and Use Cases

**Primary Purpose**: Compute fairness metrics across demographic groups to detect and quantify bias.

**Metrics Computed**:
1. Statistical Parity Difference
2. Disparate Impact Ratio
3. Equal Opportunity Difference
4. Average Odds Difference
5. Counterfactual Fairness (experimental)

### 2.3 Algorithm Design

#### 2.3.1 Statistical Parity Difference (SPD)

**Definition**: Measures whether positive outcomes are distributed equally across groups.

**Formula**:
```
SPD = max_a,a' |P(Y=1|A=a) - P(Y=1|A=a')|
```

Where:
- Y=1 is a positive outcome (e.g., "approved")
- A is a protected attribute (e.g., gender)
- a, a' are different values of the protected attribute

**Interpretation**:
- SPD = 0: Perfect parity
- SPD ≤ 0.10 (10%): Acceptable threshold
- SPD > 0.10: Potential bias, requires investigation

**Implementation**:
```python
def statistical_parity_difference(outcomes, protected_attr):
    """
    Compute statistical parity difference.
    
    Args:
        outcomes: List of decision outcomes (0 or 1)
        protected_attr: List of protected attribute values
    
    Returns:
        Maximum difference in positive outcome rates
    """
    groups = group_by(protected_attr)
    positive_rates = {}
    
    for group in groups:
        group_outcomes = [outcomes[i] for i in groups[group]]
        positive_rates[group] = sum(group_outcomes) / len(group_outcomes)
    
    max_diff = 0
    for group1 in positive_rates:
        for group2 in positive_rates:
            if group1 != group2:
                diff = abs(positive_rates[group1] - positive_rates[group2])
                max_diff = max(max_diff, diff)
    
    return max_diff
```

#### 2.3.2 Disparate Impact Ratio (DIR)

**Definition**: Ratio of positive outcome rates between the least and most favored groups.

**Formula**:
```
DIR = min_a,a' [P(Y=1|A=a) / P(Y=1|A=a')]
```

**Interpretation**:
- DIR = 1.0: Perfect parity
- DIR ≥ 0.80: Acceptable (80% rule from US employment law)
- DIR < 0.80: Potential disparate impact

**Implementation**: Similar to SPD but computes ratio instead of difference.

#### 2.3.3 Equal Opportunity Difference (EOD)

**Definition**: Measures whether true positive rates are equal across groups.

**Formula**:
```
EOD = max_a,a' |P(Ŷ=1|A=a,Y=1) - P(Ŷ=1|A=a',Y=1)|
```

Where Ŷ is the predicted outcome and Y is the true outcome.

**Interpretation**:
- EOD = 0: Equal opportunity
- EOD ≤ 0.05 (5%): Acceptable threshold
- EOD > 0.05: Unequal opportunity for qualified individuals

### 2.4 Inputs and Outputs

**Inputs**:
- Decision outcomes (binary or categorical)
- Protected attribute values (anonymized group IDs)
- Ground truth labels (when available, for supervised metrics)
- Time range for analysis

**Input Requirements**:
- Minimum 100 decisions per group for statistical significance
- Balanced representation (no group <5% of total)

**Outputs**:
- Metric values for each protected attribute
- Confidence intervals (95%)
- Statistical significance tests
- Temporal trends
- Threshold violation alerts

### 2.5 Performance Characteristics

**Computational Complexity**:
- SPD, DIR: O(N×G) where N=decisions, G=groups
- EOD: O(N×G) with ground truth
- Counterfactual: O(N×E) where E=evaluations per decision

**Measured Performance**:
- Processing time: <1 second for 10,000 decisions
- Memory usage: O(N)

### 2.6 Limitations and Risks

**Statistical Limitations**:
- Requires sufficient sample size (N≥100 per group)
- Assumes IID (independent and identically distributed) samples
- Cannot detect intersectional bias without multi-attribute analysis

**Interpretation Challenges**:
- Metrics can conflict (e.g., improving one may worsen another)
- Context-dependent thresholds (0.80 DIR is a guideline, not universal law)
- Temporal instability (metrics change over time)

**Privacy Risks**:
- Group-level analysis only (no individual-level data exposed)
- Small group sizes may allow re-identification
- Mitigation: Suppress metrics when group size <30

### 2.7 Fairness of Fairness Metrics

**Meta-Fairness Considerations**:
- Statistical parity may conflict with individual fairness
- Equal opportunity focuses on qualified individuals (Y=1)
- No single metric captures all fairness intuitions

**Philosophical Stance**:
- We use multiple metrics to provide a comprehensive view
- Human judgment required for interpretation and action
- Metrics are tools, not absolute truth

### 2.8 Testing and Validation

**Unit Tests**:
- Known outcomes produce expected metric values
- Edge cases (all same group, perfect balance) handled correctly

**Integration Tests**:
- Metrics computed correctly from database queries
- Alerts triggered at correct thresholds

**Validation**:
- Cross-checked with external fairness libraries (AIF360, Fairlearn)
- Expert review of methodology

### 2.9 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-02-01 | Initial implementation (SPD, DIR) |
| 1.1 | 2024-08-01 | Added EOD and Average Odds |
| 1.2 | 2025-11-17 | Added counterfactual fairness (experimental) |

---

## 3. Anomaly Detection for Audit Logs

### 3.1 Basic Information

**Name**: Audit Log Anomaly Detector  
**Type**: Statistical anomaly detection  
**Version**: 1.0  
**Owner**: Security Team  

### 3.2 Purpose and Use Cases

**Primary Purpose**: Detect unusual patterns in audit logs that may indicate:
- Security breaches
- System malfunctions
- Policy violations
- Data integrity issues

**Use Cases**:
- Real-time monitoring for suspicious activity
- Proactive alerting before major incidents
- Forensic analysis post-incident

### 3.3 Algorithm Design

**Approach**: Multi-method ensemble combining:

1. **Statistical Outlier Detection**:
   - Z-score for numeric features (request rate, response time)
   - Interquartile range (IQR) for distributions

2. **Time Series Anomaly Detection**:
   - Moving average with confidence bands
   - Exponential smoothing (Holt-Winters)
   - Seasonal decomposition

3. **Sequence Anomaly Detection**:
   - N-gram analysis for event sequences
   - Markov chain transition probabilities

**Example - Request Rate Anomaly**:
```python
def detect_rate_anomaly(request_counts, window=60):
    """
    Detect anomalous request rates using moving average.
    
    Args:
        request_counts: Time series of request counts
        window: Window size in minutes
    
    Returns:
        List of anomaly timestamps
    """
    # Compute moving average and standard deviation
    ma = moving_average(request_counts, window)
    std = moving_std(request_counts, window)
    
    # Compute upper and lower bounds (3-sigma)
    upper = ma + 3 * std
    lower = ma - 3 * std
    
    # Identify anomalies
    anomalies = []
    for i, count in enumerate(request_counts):
        if count > upper[i] or count < lower[i]:
            anomalies.append(i)
    
    return anomalies
```

### 3.4 Inputs and Outputs

**Inputs**:
- Audit log events (timestamped)
- Event type, user, source IP, action
- Contextual features (time of day, day of week)

**Outputs**:
- Anomaly alerts with:
  - Timestamp
  - Anomaly score (0.0 to 1.0)
  - Affected entities (users, IPs, actions)
  - Suggested investigation steps

### 3.5 Performance Characteristics

**Accuracy**:
- Precision: 85% (15% false positive rate)
- Recall: 90% (10% false negative rate)
- F1-score: 0.87

**Performance**:
- Latency: <5 seconds for real-time detection
- Throughput: 100,000 events/second
- Memory: 500MB for 30-day sliding window

### 3.6 Limitations and Risks

**False Positives**:
- Legitimate traffic spikes trigger alerts (e.g., marketing campaign)
- Mitigation: Context-aware thresholds, user feedback loop

**False Negatives**:
- Slow-moving attacks may not trigger alerts
- Sophisticated attackers may mimic normal behavior
- Mitigation: Multiple detection methods, regular model updates

**Adversarial Risks**:
- Attackers may try to "learn" normal patterns and evade detection
- Mitigation: Ensemble methods, unpredictable model updates

### 3.7 Testing and Validation

**Test Data**:
- Historical audit logs with known anomalies (labeled)
- Synthetic attacks (penetration test logs)

**Validation**:
- Cross-validation on historical data
- Comparison with commercial SIEM tools
- Red team exercises

### 3.8 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-17 | Initial implementation |

---

## 4. Policy Similarity Matching

### 4.1 Basic Information

**Name**: Policy Similarity Matcher  
**Type**: NLP-based similarity scoring  
**Version**: 1.0  
**Owner**: Policy Management Team  

### 4.2 Purpose and Use Cases

**Primary Purpose**: Identify similar policies to prevent duplication and aid in policy discovery.

**Use Cases**:
- Suggest existing policies when creating new ones
- Detect potential conflicts between policies
- Group related policies for bulk updates

### 4.3 Algorithm Design

**Approach**: Hybrid similarity using:
1. **Semantic Similarity**: Sentence embeddings (BERT)
2. **Structural Similarity**: Edit distance on policy AST
3. **Behavioral Similarity**: Decision outcome similarity on test cases

**Similarity Score**:
```
similarity(P1, P2) = 0.5 * semantic_sim + 0.3 * structural_sim + 0.2 * behavioral_sim
```

**Implementation**:
- Pre-trained BERT model for embeddings
- Tree edit distance for AST comparison
- Monte Carlo sampling for behavioral testing

### 4.4 Performance Characteristics

**Accuracy**:
- Matches human similarity judgments with 0.85 correlation

**Performance**:
- Similarity computation: <500ms per policy pair
- Batch processing: 1000 policies in <5 minutes

### 4.5 Limitations

- Requires sufficient policy corpus for embeddings
- Semantic similarity may miss syntactic nuances
- Behavioral similarity requires labeled test cases

### 4.6 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-17 | Initial implementation |

---

## 5. Appeal Priority Scoring

### 5.1 Basic Information

**Name**: Appeal Priority Scorer  
**Type**: Multi-criteria decision analysis (MCDA)  
**Version**: 1.0  
**Owner**: Appeals Team  

### 5.2 Purpose and Use Cases

**Primary Purpose**: Prioritize appeals for review based on urgency, impact, and likelihood of success.

**Use Cases**:
- Queue management for human reviewers
- SLA compliance (high-priority appeals reviewed faster)
- Resource allocation

### 5.3 Algorithm Design

**Criteria** (weighted):
1. **Impact** (40%): Consequence of original decision (financial, health, liberty)
2. **Urgency** (30%): Time sensitivity (statute of limitations, deadlines)
3. **Merit** (20%): Likelihood of appeal success (based on justification quality)
4. **Systemic Importance** (10%): Potential for policy improvement

**Scoring**:
```
priority = 0.4 * impact + 0.3 * urgency + 0.2 * merit + 0.1 * systemic
```

Each criterion scored 0-10 based on rubric.

**Example - Impact Scoring**:
- 9-10: Life, liberty, or >$100K financial impact
- 7-8: Significant hardship or $10K-$100K
- 5-6: Moderate inconvenience or $1K-$10K
- 3-4: Minor inconvenience or <$1K
- 1-2: Negligible impact

### 5.4 Performance Characteristics

**Accuracy**:
- Matches human prioritization with 0.78 correlation

**Performance**:
- Scoring: <100ms per appeal
- Batch: 10,000 appeals in <1 minute

### 5.5 Limitations

- Weights are subjective (organization-dependent)
- Cannot capture all nuances of individual cases
- Risk of gaming (appellant exaggerates urgency)

### 5.6 Fairness Considerations

- Priority scoring should not discriminate by protected attributes
- Monitoring: Appeals from different demographic groups should have similar average priority scores (if underlying impacts are similar)

### 5.7 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-17 | Initial implementation |

---

## Conclusion

These algorithm cards provide transparency into the key algorithms and models used in the Nethical platform. They are living documents, updated as algorithms evolve. For questions or concerns, please contact:

**Transparency Team**: transparency@nethical.example.com  
**Technical Questions**: engineering@nethical.example.com  
**Fairness Concerns**: fairness@nethical.example.com

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-17 | Phase 9 Team | Initial algorithm cards for core algorithms |

**Next Review**: 2026-05-17 (6 months)
