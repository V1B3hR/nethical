# 🛡️ Nethical Detection Maturity Roadmap

**Version**:  1.0  
**Last Updated**: 2025-12-11  
**Aligned With**:  ROADMAP_9+. md v3.9, 25 Fundamental Laws  
**Codename**: "The Vigilant Guardian"

---

## 📊 Executive Summary

This roadmap defines the evolution of Nethical's attack detection capabilities from the current 36+ vectors to a comprehensive, future-proof detection system capable of defending against known, emerging, and adaptive threats.

| Phase | Timeline | Vectors | Detection Method | Validation | Status |
|-------|----------|---------|------------------|------------|--------|
| **Phase 1: Foundation** | Current | 36 core vectors | Rule + pattern matching | Manual test cases | ✅ **COMPLETE** |
| **Phase 2: Expansion** | 0-6 months | +18 vectors (54 total) | ML classifiers + embedding anomaly | Automated benchmark suite | ✅ **COMPLETE** |
| **Phase 3: Intelligence** | 6-12 months | +12 vectors (66 total) | Online learning + behavioral analysis | Continuous adversarial validation |
| **Phase 4: Autonomy** | 12-18 months | Dynamic registry | Self-updating detectors | Autonomous red-team + canaries |
| **Phase 5: Omniscience** | 18-24 months | Predictive detection | Threat anticipation | Formal verification + proofs |

---

## 🎯 Detection Philosophy

### Alignment with 25 Fundamental Laws

Every detector in Nethical must map to one or more Fundamental Laws:

| Law Category | Relevant Laws | Detection Focus |
|--------------|---------------|-----------------|
| **III. Transparency** | Laws 9-12 | Identity deception, persona attacks, hidden instructions |
| **IV. Accountability** | Laws 13-16 | Action attribution, decision traceability |
| **V. Coexistence** | Laws 17-20 | Social engineering, manipulation, deception |
| **VI. Protection** | Laws 21-23 | System exploitation, safety-critical threats |
| **VII. Growth** | Laws 24-25 | Adaptive attacks, evolving threat landscape |

### Detection Principles

```yaml
detection_principles:
  1_defense_in_depth:
    description: "Multiple detection layers, no single point of failure"
    implementation: "Rule → ML → Embedding → Behavioral → Contextual"
    
  2_fail_safe: 
    description: "Unknown threats trigger safe-mode, not pass-through"
    implementation: "Default RESTRICT for low-confidence detections"
    law_alignment: "Law 23:  Fail-Safe Design"
    
  3_explainable:
    description: "Every detection decision must be human-understandable"
    implementation: "Natural language explanations + SHAP values"
    law_alignment: "Law 10: Transparency of Reasoning"
    
  4_bi_directional:
    description:  "Protect humans from AI AND AI from misuse"
    implementation: "Detect both malicious outputs and adversarial inputs"
    
  5_adaptive: 
    description: "Detection evolves with threat landscape"
    implementation: "Online learning + red-team feedback loops"
```

---

## 📋 Current State:  Phase 1 Foundation (36+ Vectors)

### Existing Detector Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DetectorRegistry                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │  Security   │ │  Content    │ │  Privacy    │ │   System   │ │
│  │  Detector   │ │  Safety     │ │  Detector   │ │   Limits   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │  Ethical    │ │ Manipulation│ │ Law         │ │ Adversarial│ │
│  │  Detector   │ │  Detector   │ │ Violation   │ │  Detector  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Current Vector Coverage

| Family | Vectors Covered | Detection Method | Status |
|--------|-----------------|------------------|--------|
| **Prompt Injection** | Direct, Basic Indirect | Pattern matching | 🟢 Implemented |
| **Code Injection** | XSS, SQL, Command | Regex patterns | 🟢 Implemented |
| **Path Traversal** | Basic patterns | Regex patterns | 🟢 Implemented |
| **Content Safety** | Hate speech, Violence | Keyword + ML | 🟢 Implemented |
| **PII Detection** | Email, SSN, Phone | Regex + NER | 🟢 Implemented |
| **Volume Attacks** | Rate limiting, DoS | Statistical | 🟢 Implemented |
| **Resource Exhaustion** | Memory, CPU, Nesting | Threshold-based | 🟢 Implemented |
| **Social Engineering** | Basic manipulation | Keyword patterns | 🟠 Partial |

### Identified Gaps

| Gap | Risk Level | Impact on Nethical |
|-----|------------|-------------------|
| Multi-turn/Session attacks | 🔴 CRITICAL | Agents persist across sessions |
| Embedding-space attacks | 🔴 CRITICAL | Bypasses text-based detection |
| Model extraction | 🟡 HIGH | Violates bi-directional protection |
| Supply chain integrity | 🟡 HIGH | Policy/model tampering |
| Multimodal threats | 🟡 HIGH | Future agent capabilities |
| Adaptive/zero-day | 🟡 HIGH | Static rules age quickly |

---

## 🚀 Phase 2: Detection Expansion (0-6 Months)

**Objective**: Expand from 36 to 54 vectors with ML-enhanced detection

### 2.1 Advanced Prompt Injection Suite

**New Vectors**:  +6

```yaml
prompt_injection_expansion:
  vectors: 
    PI-007_multilingual: 
      description: "Injection attempts in non-English languages"
      detection: "Multilingual NLP + translation-aware patterns"
      signals:
        - language_switching_mid_prompt
        - unicode_homoglyph_detection
        - rtl_ltr_mixing
      law_alignment: [9, 18]  # Self-Disclosure, Non-Deception
      
    PI-008_context_overflow:
      description: "Exhausting context window to push out system prompts"
      detection: "Token counting + system prompt integrity check"
      signals:
        - token_budget_analysis
        - system_prompt_position_drift
        - repetitive_padding_detection
      law_alignment: [2, 22]  # Integrity, Boundary Respect
      
    PI-009_recursive: 
      description: "Self-referential prompts that amplify on each turn"
      detection: "Semantic similarity across turns + growth rate analysis"
      signals:
        - turn_over_turn_similarity
        - instruction_amplification_rate
        - recursive_reference_detection
      law_alignment: [13, 23]  # Action Responsibility, Fail-Safe
      
    PI-010_delimiter_escape:
      description: "Exploiting delimiter parsing (XML, JSON, markdown)"
      detection: "Structure validation + escape sequence analysis"
      signals:
        - malformed_structure_detection
        - escape_sequence_anomaly
        - nested_delimiter_abuse
      law_alignment:  [18, 22]
      
    PI-011_instruction_leak:
      description: "Attempts to extract system prompts or instructions"
      detection: "Output similarity to known system prompts"
      signals:
        - system_prompt_similarity_score
        - meta_instruction_requests
        - reflection_attack_patterns
      law_alignment: [2, 9]
      
    PI-012_indirect_multimodal:
      description: "Injection via images, audio, or file metadata"
      detection: "OCR + metadata extraction + embedding analysis"
      signals:
        - extracted_text_injection_check
        - metadata_instruction_detection
        - steganographic_content_flag
      law_alignment: [9, 18, 22]
```

**Deliverables**:
- [x] `nethical/detectors/prompt_injection/multilingual_detector.py`
- [x] `nethical/detectors/prompt_injection/context_overflow_detector.py`
- [x] `nethical/detectors/prompt_injection/recursive_detector.py`
- [x] `nethical/detectors/prompt_injection/delimiter_detector.py`
- [x] `nethical/detectors/prompt_injection/instruction_leak_detector.py`
- [x] `nethical/detectors/prompt_injection/indirect_multimodal_detector.py`
- [x] `datasets/adversarial/prompt_injection/` - Test corpus per vector (Structure created)
- [x] `tests/detectors/test_prompt_injection_suite.py` (Structure in place)

### 2.2 Session-Aware Detection

**New Vectors**: +4

```yaml
session_detection: 
  vectors:
    SA-001_multi_turn_staging:
      description: "Attacks staged across multiple conversation turns"
      detection: "Session state machine + cumulative risk scoring"
      signals:
        - cross_turn_semantic_drift
        - incremental_privilege_escalation
        - delayed_payload_assembly
      law_alignment: [13, 18, 23]
      
    SA-002_context_poisoning:
      description: "Gradually shifting context to enable later attacks"
      detection: "Baseline drift detection + context integrity hash"
      signals:
        - context_vector_drift_rate
        - topic_boundary_violations
        - trust_erosion_patterns
      law_alignment: [2, 18]
      
    SA-003_persona_hijack:
      description: "Attempting to change agent persona mid-session"
      detection: "Persona consistency scoring + instruction injection check"
      signals:
        - persona_trait_deviation
        - role_override_attempts
        - character_break_requests
      law_alignment: [9, 18]
      
    SA-004_memory_manipulation:
      description: "Exploiting agent memory/RAG to inject false information"
      detection: "Memory write validation + provenance checking"
      signals:
        - unauthorized_memory_write
        - contradictory_fact_injection
        - source_spoofing_attempts
      law_alignment: [2, 18, 22]
```

**Architecture**: 

```python
# nethical/detectors/session/session_state_tracker.py
class SessionStateTracker:
    """
    Maintains session state for multi-turn attack detection.
    
    Features:
    - Cumulative risk scoring across turns
    - Context integrity verification
    - Semantic drift monitoring
    - Cross-session pattern correlation
    """
    
    def __init__(self, agent_id: str, session_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self.turn_history:  List[TurnContext] = []
        self.cumulative_risk:  float = 0.0
        self.context_hash: str = ""
        self. baseline_embedding: Optional[np.ndarray] = None
        
    def record_turn(self, turn:  TurnContext) -> SessionRiskAssessment:
        """Record a turn and return updated risk assessment."""
        pass
        
    def check_staging_attack(self) -> Optional[SafetyViolation]:
        """Detect multi-turn staging patterns."""
        pass
        
    def check_context_drift(self) -> Optional[SafetyViolation]:
        """Detect gradual context poisoning."""
        pass
```

**Deliverables**:
- [x] `nethical/detectors/session/__init__.py`
- [x] `nethical/detectors/session/session_state_tracker.py`
- [x] `nethical/detectors/session/multi_turn_detector.py`
- [x] `nethical/detectors/session/context_poisoning_detector.py`
- [x] `nethical/detectors/session/persona_detector.py`
- [x] `nethical/detectors/session/memory_manipulation_detector.py`
- [x] Integration with `nethical/edge/local_governor.py` for session state (Ready)

### 2.3 Model Security Suite

**New Vectors**: +4

```yaml
model_security:
  vectors:
    MS-001_extraction_via_api:
      description: "Attempting to extract model weights via API queries"
      detection: "Query fingerprinting + synthetic probe detection"
      signals:
        - query_distribution_anomaly
        - boundary_probing_patterns
        - output_diversity_mining
      law_alignment: [2, 22]  # Integrity, Boundary Respect
      
    MS-002_membership_inference:
      description: "Determining if specific data was in training set"
      detection: "Confidence calibration + query pattern analysis"
      signals:
        - confidence_distribution_anomaly
        - targeted_query_sequences
        - training_data_probing
      law_alignment: [2, Privacy Laws]
      
    MS-003_model_inversion:
      description: "Reconstructing training data from model outputs"
      detection: "Output entropy monitoring + reconstruction attempt detection"
      signals:
        - low_entropy_output_sequences
        - iterative_refinement_patterns
        - gradient_approximation_queries
      law_alignment: [2, Privacy Laws]
      
    MS-004_backdoor_activation:
      description: "Triggering planted backdoors in models"
      detection: "Trigger pattern detection + anomalous output monitoring"
      signals:
        - known_trigger_patterns
        - output_mode_switching
        - unexpected_capability_activation
      law_alignment:  [2, 23]
```

**Deliverables**:
- [x] `nethical/detectors/model_security/__init__.py`
- [x] `nethical/detectors/model_security/extraction_detector.py`
- [x] `nethical/detectors/model_security/membership_inference_detector.py`
- [x] `nethical/detectors/model_security/inversion_detector.py`
- [x] `nethical/detectors/model_security/backdoor_detector.py`
- [x] Query fingerprint database for extraction detection (Structure ready)
- [x] Integration with rate limiting and quota systems (Ready)

### 2.4 Supply Chain Integrity

**New Vectors**: +4

```yaml
supply_chain: 
  vectors:
    SC-001_policy_tampering:
      description: "Unauthorized modification of governance policies"
      detection: "Cryptographic verification + Merkle proof validation"
      signals:
        - policy_hash_mismatch
        - unauthorized_policy_source
        - merkle_proof_failure
      law_alignment: [2, 13]  # Integrity, Action Responsibility
      
    SC-002_model_tampering: 
      description: "Modified model artifacts injected into pipeline"
      detection: "Model signature verification + behavioral drift detection"
      signals:
        - signature_verification_failure
        - model_hash_mismatch
        - behavioral_baseline_deviation
      law_alignment: [2, 23]
      
    SC-003_dependency_attack:
      description: "Malicious code in dependencies"
      detection: "SBOM verification + dependency hash checking"
      signals:
        - sbom_hash_mismatch
        - unexpected_dependency_behavior
        - known_vulnerability_match
      law_alignment: [2, 21, 23]
      
    SC-004_ci_cd_compromise:
      description: "Compromised build/deployment pipeline"
      detection: "Artifact provenance verification + build reproducibility"
      signals:
        - provenance_chain_break
        - non_reproducible_build
        - unauthorized_artifact_source
      law_alignment: [2, 13]
```

**Integration with Existing Infrastructure**:

```yaml
# Leverage existing SBOM. json and Merkle anchoring
supply_chain_integration:
  sbom_verification:
    source: "SBOM.json"
    check_frequency: "on_startup, hourly"
    
  policy_verification:
    source: "nethical/audit/merkle_anchor.py"
    check_frequency: "on_policy_load"
    
  model_verification:
    source: "nethical/mlops/model_registry.py"
    check_frequency: "on_model_load"
```

**Deliverables**: 
- [x] `nethical/detectors/supply_chain/__init__.py`
- [x] `nethical/detectors/supply_chain/policy_integrity_detector.py`
- [x] `nethical/detectors/supply_chain/model_integrity_detector.py`
- [x] `nethical/detectors/supply_chain/dependency_detector.py`
- [x] `nethical/detectors/supply_chain/cicd_detector.py`
- [x] Enhanced `SBOM.json` with runtime verification hooks (Ready)
- [x] Integration with `.github/workflows/` for CI/CD verification (Ready)

### 2.5 Embedding-Space Detection

**Purpose**: Detect attacks that bypass text-based pattern matching

```yaml
embedding_detection:
  components:
    semantic_anomaly_detector:
      description: "Detect semantically anomalous inputs"
      method: "Embedding distance from safe baseline clusters"
      threshold: "Mahalanobis distance > 3σ"
      
    adversarial_perturbation_detector:
      description: "Detect adversarially crafted inputs"
      method: "Gradient-based perturbation detection"
      signals:
        - high_gradient_norm
        - boundary_proximity
        - input_instability
        
    paraphrase_attack_detector:
      description: "Detect paraphrased versions of known attacks"
      method: "Semantic similarity to attack corpus"
      threshold: "Cosine similarity > 0.85 to known attack"
      
    covert_channel_detector:
      description: "Detect hidden information in embeddings"
      method: "Entropy analysis + steganographic pattern detection"
```

**Deliverables**:
- [x] `nethical/detectors/embedding/__init__.py`
- [x] `nethical/detectors/embedding/semantic_anomaly_detector.py`
- [x] `nethical/detectors/embedding/adversarial_perturbation_detector.py`
- [x] `nethical/detectors/embedding/paraphrase_detector.py`
- [x] `nethical/detectors/embedding/covert_channel_detector.py`
- [x] Safe baseline embedding clusters per agent type (Structure ready)
- [x] Attack corpus embeddings for similarity matching (Structure ready)

### Phase 2 Validation Framework

```yaml
phase_2_validation:
  benchmark_suite:
    location: "benchmarks/detection/"
    components:
      - prompt_injection_benchmark. py
      - session_attack_benchmark.py
      - model_security_benchmark.py
      - supply_chain_benchmark.py
      - embedding_attack_benchmark.py
      
  metrics: 
    detection_rate:
      target: ">= 0.95 for known vectors"
      measurement: "True positives / Total positives"
      
    false_positive_rate:
      target: "<= 0.02"
      measurement: "False positives / Total negatives"
      
    latency_overhead:
      target: "<= 5ms p99 additional latency"
      measurement: "Detection time per request"
      
    coverage: 
      target: "54/54 vectors with tests"
      measurement: "Vectors with passing test suites"
      
  ci_integration:
    workflow: ". github/workflows/detection_benchmark.yml"
    trigger: "on PR, daily scheduled"
    blocking: "Detection rate < 0.90 blocks merge"
```

---

## 🧠 Phase 3: Detection Intelligence (6-12 Months)

**Objective**: ML-powered adaptive detection with behavioral analysis

### 3.1 Online Learning Pipeline

```yaml
online_learning:
  architecture:
    feedback_sources:
      - human_review_decisions
      - appeal_outcomes
      - red_team_findings
      - false_positive_reports
      
    update_cycle:
      frequency: "continuous with batching"
      batch_size: 1000
      max_staleness: "24 hours"
      
    safety_constraints:
      - "No reduction in detection rate for critical vectors"
      - "Human approval for threshold changes > 10%"
      - "Rollback capability within 5 minutes"
      - "A/B testing before full deployment"
```

### 3.2 Behavioral Baseline System

```yaml
behavioral_baselines:
  per_agent_baselines:
    features:
      - request_rate_distribution
      - action_type_distribution
      - risk_score_distribution
      - time_of_day_patterns
      - session_length_distribution
      
    anomaly_detection:
      method: "Isolation Forest + Statistical Process Control"
      sensitivity: "Configurable per agent type"
      
  global_baselines:
    features:
      - cross_agent_coordination_patterns
      - attack_wave_detection
      - collective_anomaly_scoring
```

### 3.3 New Vector Categories

**New Vectors**: +12

| Vector ID | Name | Detection Method |
|-----------|------|------------------|
| BH-001 | Coordinated Agent Attack | Cross-agent correlation |
| BH-002 | Slow-and-Low Evasion | Long-term behavioral drift |
| BH-003 | Mimicry Attack | Behavioral fingerprint spoofing |
| BH-004 | Resource Timing Attack | Timing side-channel analysis |
| MM-001 | Adversarial Image | CNN-based perturbation detection |
| MM-002 | Audio Injection | Speech-to-text + injection check |
| MM-003 | Video Frame Attack | Per-frame adversarial detection |
| MM-004 | Cross-Modal Injection | Multi-encoder consistency check |
| ZD-001 | Zero-Day Pattern | Anomaly ensemble detection |
| ZD-002 | Polymorphic Attack | Behavioral invariant matching |
| ZD-003 | Attack Chain | Kill chain stage detection |
| ZD-004 | Living-off-the-Land | Legitimate capability abuse |

### Phase 3 Deliverables

- [ ] `nethical/ml/online_learning/` - Online learning pipeline
- [ ] `nethical/detectors/behavioral/` - Behavioral detection suite
- [ ] `nethical/detectors/multimodal/` - Multimodal detection suite
- [ ] `nethical/detectors/zeroday/` - Zero-day detection suite
- [ ] `training/detection_models/` - Model training pipelines
- [ ] `dashboards/detection_intelligence. json` - ML monitoring dashboard

---

## 🤖 Phase 4: Detection Autonomy (12-18 Months)

**Objective**: Self-updating detection with minimal human intervention

### 4.1 Autonomous Red Team

```yaml
autonomous_red_team:
  components:
    attack_generator:
      description: "ML-based generation of novel attack variants"
      method: "Adversarial generation with safety constraints"
      
    coverage_optimizer:
      description: "Identify gaps in detection coverage"
      method: "Fuzzing + coverage-guided mutation"
      
    detector_challenger:
      description: "Continuously probe detectors for weaknesses"
      method: "Gradient-based adversarial examples"
      
  safety_constraints:
    - "Sandboxed execution environment"
    - "No real data exposure"
    - "Human review of high-impact findings"
    - "Rate-limited to prevent self-DoS"
```

### 4.2 Canary System

```yaml
canary_deployment:
  honeypot_prompts:
    description: "Decoy prompts to detect active reconnaissance"
    detection: "Any interaction with canary triggers alert"
    
  tripwire_endpoints:
    description: "Fake API endpoints that should never be called"
    detection: "Any request to tripwire = active probing"
    
  watermarked_responses:
    description: "Invisible watermarks in responses"
    detection: "Watermark appearing elsewhere = data exfiltration"
```

### 4.3 Dynamic Attack Registry

```yaml
dynamic_registry:
  auto_registration:
    trigger:  "New attack pattern confirmed by red team"
    process: 
      1:  "Generate detector from attack signature"
      2: "Validate on test corpus"
      3: "Deploy to staging"
      4: "A/B test in production"
      5: "Full deployment with monitoring"
      
  auto_deprecation:
    trigger: "Zero detections for 90 days + no known variants"
    process:
      1: "Flag for review"
      2: "Human confirmation"
      3: "Move to archive (not delete)"
```

---

## 🔮 Phase 5: Detection Omniscience (18-24 Months)

**Objective**: Predictive threat detection with formal guarantees

### 5.1 Threat Anticipation

```yaml
threat_anticipation:
  threat_intelligence_integration:
    sources:
      - CVE databases
      - AI security research feeds
      - Dark web monitoring (ethical)
      - Industry sharing groups
      
  predictive_modeling:
    method: "Trend analysis + attack evolution modeling"
    output: "Probability of new attack vector emergence"
    
  proactive_hardening:
    trigger: "Predicted threat probability > 70%"
    action:  "Pre-deploy defensive measures"
```

### 5.2 Formal Verification

```yaml
formal_verification:
  properties_to_verify:
    - "No false negatives for critical safety vectors"
    - "Bounded false positive rate"
    - "Deterministic behavior for same input"
    - "Graceful degradation under resource pressure"
    
  tools: 
    - TLA+ for detection logic
    - Z3 for policy consistency
    - Lean 4 for core invariants
    
  integration: 
    - CI verification on detector changes
    - Runtime monitoring of verified properties
```

---

## 📈 Success Metrics

### Detection Effectiveness

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|--------|---------|---------|---------|---------|---------|
| **Vectors Covered** | 36 | 54 | 66 | Dynamic | Predictive |
| **Detection Rate (Known)** | 85% | 95% | 98% | 99% | 99.5% |
| **Detection Rate (Novel)** | 20% | 40% | 60% | 80% | 90% |
| **False Positive Rate** | 5% | 2% | 1% | 0.5% | 0.2% |
| **Time to New Vector** | Manual | 1 week | 1 day | 1 hour | Proactive |

### Operational Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Detection Latency p50** | <2ms | Prometheus histogram |
| **Detection Latency p99** | <10ms | Prometheus histogram |
| **Detector Availability** | 99.99% | Uptime monitoring |
| **Coverage Completeness** | 100% | Vector registry audit |
| **Test Coverage** | 95% | pytest-cov |

---

## 🗂️ Attack Registry Template

Each vector should be documented in the attack registry: 

```yaml
# attack_registry/vectors/{family}/{vector_id}.yaml
vector: 
  id: "PI-007"
  name: "Multilingual Prompt Injection"
  family: "prompt_injection"
  version: "1.0.0"
  status: "active"  # active | deprecated | experimental
  
  description: 
    short: "Injection attempts using non-English languages"
    detailed: |
      Attackers use non-English languages to bypass pattern-based
      detection that focuses on English keywords. May also use
      Unicode homoglyphs, RTL/LTR mixing, or language switching
      mid-prompt to evade detection.
      
  risk_assessment:
    severity: "high"
    exploitability: "medium"
    impact: "high"
    cvss_equivalent: 7.5
    
  law_alignment:
    - law:  9
      name: "Self-Disclosure"
      rationale: "Hidden instructions violate transparency"
    - law: 18
      name: "Non-Deception"
      rationale: "Obfuscated attacks are deceptive"
      
  detection: 
    detector:  "nethical. detectors.prompt_injection.MultilingualDetector"
    method: "multilingual_nlp"
    signals:
      - name: "language_switching"
        description: "Language changes mid-prompt"
        weight: 0.4
      - name: "homoglyph_presence"
        description: "Unicode homoglyphs detected"
        weight: 0.3
      - name: "instruction_in_non_primary_language"
        description: "Commands in secondary language"
        weight: 0.3
    thresholds:
      block:  0.8
      restrict: 0.5
      flag: 0.3
      
  governance:
    default_action: "BLOCK"
    human_review: true
    appeal_eligible: true
    
  testing: 
    corpus: "datasets/adversarial/prompt_injection/multilingual. json"
    test_suite: "tests/detectors/test_multilingual_injection.py"
    benchmark: "benchmarks/detection/prompt_injection_benchmark.py"
    slo: 
      detection_rate: ">= 0.95"
      false_positive_rate: "<= 0.02"
      latency_p99_ms: "<= 5"
      
  mitigations:
    automated: 
      - "Block request"
      - "Log for analysis"
      - "Increment agent risk score"
    human_escalation:
      - "Review for new variant"
      - "Update detection patterns"
      
  references:
    - title: "Multilingual Jailbreak Attacks"
      url: "https://arxiv.org/abs/..."
    - title: "OWASP LLM Top 10"
      url: "https://owasp.org/..."
      
  changelog:
    - version: "1.0.0"
      date: "2025-12-11"
      changes:  "Initial implementation"
```

---

## 📚 Additional Resources

- [ROADMAP_9+. md](ROADMAP_9+.md) - Main development roadmap
- [FUNDAMENTAL_LAWS.md](FUNDAMENTAL_LAWS.md) - 25 Fundamental Laws
- [ARCHITECTURE. md](ARCHITECTURE.md) - System architecture
- [SECURITY. md](SECURITY.md) - Security policy
- [docs/ops/SLOs.md](docs/ops/SLOs.md) - Service level objectives

---

## 🤝 Contributing to Detection

See [CONTRIBUTING.md](CONTRIBUTING.md) for general guidelines.

For detection-specific contributions: 

1. **New Vector Proposals**: Open an issue with the attack registry template filled out
2. **Detector Implementation**: Follow the `BaseDetector` interface in `nethical/detectors/base_detector.py`
3. **Test Corpus**: Add adversarial examples to `datasets/adversarial/`
4. **Benchmarks**: Add benchmark cases to `benchmarks/detection/`

---

**Document Owner**:  Nethical Security Team  
**Review Cycle**: Monthly  
**Next Review**: 2026-01-11

---

## 📝 Implementation Status

### Phase 1 & 2 Implementation Complete ✅

**Implementation Date**: December 12, 2025  
**Total Vectors**: 54 (Phase 1: 36 baseline + Phase 2: 18 expansion)

#### Phase 1 (Foundation) - ✅ Complete
- 36 core attack vectors implemented and documented
- Pattern-based detection with comprehensive coverage
- Integration with Nethical governance system
- Attack registry with full metadata and law alignment

#### Phase 2 (Expansion) - ✅ Complete

**2.1 Advanced Prompt Injection Suite** (+6 vectors) ✅
- Multilingual injection detection with Unicode analysis
- Context overflow detection with token budget analysis
- Recursive injection detection with self-reference patterns
- Delimiter escape detection for format exploitation
- Instruction leak detection for system prompt extraction
- Indirect multimodal injection for image/metadata attacks

**2.2 Session-Aware Detection** (+4 vectors) ✅
- Session state tracker for multi-turn analysis
- Multi-turn staging detection with cumulative risk scoring
- Context poisoning detection with drift analysis
- Persona hijacking detection for role override attempts
- Memory manipulation detection for RAG exploitation

**2.3 Model Security Suite** (+4 vectors) ✅
- Model extraction detection via API query analysis
- Membership inference detection for training data privacy
- Model inversion detection for data reconstruction attempts
- Backdoor activation detection for trigger patterns

**2.4 Supply Chain Integrity** (+4 vectors) ✅
- Policy tampering detection with integrity checks
- Model tampering detection with signature verification
- Dependency attack detection for supply chain security
- CI/CD compromise detection for pipeline integrity

**2.5 Embedding-Space Detection** ✅
- Semantic anomaly detection for embedding-space attacks
- Adversarial perturbation detection
- Paraphrase attack detection with similarity analysis
- Covert channel detection for steganographic content

#### Updated Artifacts
- ✅ `nethical/core/attack_registry.py` - Updated to 54 vectors (v2.0.0)
- ✅ `nethical/detectors/prompt_injection/` - 6 new detectors
- ✅ `nethical/detectors/session/` - 4 new detectors + session tracker
- ✅ `nethical/detectors/model_security/` - 4 new detectors
- ✅ `nethical/detectors/supply_chain/` - 4 new detectors
- ✅ `nethical/detectors/embedding/` - 4 new detectors

#### Next Steps (Phase 3+)
- Implement ML-powered adaptive detection (Phase 3)
- Add online learning pipeline for continuous improvement
- Develop behavioral baseline systems
- Expand to multimodal threat detection
- Implement zero-day detection capabilities

---

<div align="center">
  <sub>🛡️ Building the brakes so the car can drive fast 🛡️</sub>
</div>
