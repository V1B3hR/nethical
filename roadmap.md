# Detailed Job List: Phase 3, Phase 4, and Future Tracks

---
## Detailed Checklist by Phase

### Phase 1 Must-Haves
- 🟢 Detector registry abstraction (complete)
- 🟢 Core schemas authored + validated (complete)
- 🟢 Latency instrumentation (fast vs full path) (complete)
- 🟢 Basic risk score scaffold (complete)

### Phase 2
- 🟠 Event bus integration (producer/consumer libs) (ongoing)
- 🟠 Signed policy service + key rotation doc (ongoing)
- 🟡 Correlation MVP (rate + basic anomaly) (partially live; improvement ongoing)
- 🟡 Policy propagation metric (partially implemented, needs monitoring integration)

## Phase 3: Correlation & Adaptive Risk (Week 8–12)

**Subphase 3.1: Correlation & Multi-Agent Detection**
- Expand correlation: Add multi-agent pattern detection (payload entropy shifts, escalating multi-ID probes)
- Implement correlation rules (`correlation_rules.yaml`)
- Persist risk scores (Redis + periodic snapshot)

**Subphase 3.2: Risk Engine Evolution**
- Implement risk decay formula and multi-factor fusion
- Risk tier transitions and persistence (LOW, NORMAL, HIGH, ELEVATED)
- Introduce Elevated Tier triggers (invoke advanced detectors only when risk > threshold)

**Subphase 3.3: Fairness & Sampling**
- Begin fairness sampling job (nightly + on-demand)
- Fairness Sampler: selects stratified subset & stores for offline review

**Subphase 3.4: Ethical Drift and Reporting**
- Generate initial ethical drift report (difference in violation types by agent cohort)
- Fairness dashboard online (sampling coverage exposure)

**Subphase 3.5: Performance & Cost Optimization**
- Selective elevation: Reduce average CPU per action by >30% using risk-based gating

**Subphase 3.6: Quality, Testing & Exit**
- Human-labeled false positive rate <10% in HIGH severity violations
- Complete phase with successful fairness dashboard and correlation detection

---

## Phase 4: Integrity & Ethics Operationalization (Week 13–20)

**Subphase 4.1: Immutable Audit & Merkle Anchoring**
- Full event log chunking & Merkle root computation
- Anchor Merkle roots (S3 object lock or external notarization)
- Develop Merkle verification tool

**Subphase 4.2: Policy Auditing & Diff**
- Implement policy diff audit (show semantic diff risk)
- Build Policy diff CLI (`cli/policy_diff`)

**Subphase 4.3: Quarantine & Incident Response**
- Add Quarantine Mode: automatic global policy override for agent cohort with anomalies
- Develop quarantine workflow doc + API
- Quarantine scenario simulation (synthetic attack → cohort isolation <15 s)

**Subphase 4.4: Ethical Impact Layer**
- Tag violations with ethical dimension taxonomy (privacy, manipulation, fairness, safety)
- Integrate ethical taxonomy mapping (`ethics_taxonomy.json`)

**Subphase 4.5: SLA & Performance**
- Ensure P95 latency <220 ms under 2× nominal load
- SLA documentation and validation

**Subphase 4.6: Phase 4 Exit & Quality Gates**
- Merkle verification tool validates random historical segments
- Ethical taxonomy coverage >90% of violation categories

---

## Future Tracks (Preparations for 11–50 Systems)

**Subphase F1: Regionalization & Sharding**
- Introduce `region_id` and `logical_domain` fields in events
- Prepare for hierarchical aggregation and regional API gateways

**Subphase F2: Detector & Policy Extensibility**
- Detector externalization: encapsulate heavy detectors behind RPC/gRPC
- Policy DSL: draft abstract rule spec (YAML → compiled form)
- Version and metadata manifest for detectors

**Subphase F3: Privacy & Data Handling**
- Add redaction pipeline stage (hash & token classification)
- Differential privacy for federated analytics

**Subphase F4: Thresholds, Tuning, & Adaptivity**
- Store decision outcomes + human labels for ML-driven threshold tuning
- ML-based threshold tuner (offline)

**Subphase F5: Simulation & Replay**
- Persist full action streams with tags for replay engine (time-travel, what-if analysis)
- Simulation/replay engine implementation

**Subphase F6: Marketplace & Ecosystem**
- Monitor marketplace / plugin registry: support external detector contributions at scale
- Detector metadata manifest and governance

---
