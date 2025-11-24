# Nethical Production Readiness Checklist

## 1. Architecture
- [ ] Stateless API layer horizontally scalable
- [ ] Externalized DB (Postgres) with automated backups + PITR
- [ ] Redis cluster with auth + TLS
- [ ] Object storage for cold/archival audit tiers

## 2. Security
- [ ] SBOM generated per build (CycloneDX)
- [ ] Dependency audit pass (no Critical)
- [ ] Image scanning (Trivy) clean (Critical/High blocked)
- [ ] Secrets externalized (Vault / KMS)
- [ ] mTLS or JWT validated at gateway
- [ ] WAF rules (prompt injection / oversized payload) enabled

## 3. Governance & Ethics
- [ ] Policy grammar (EBNF) published
- [ ] Policy simulator & dry-run diff CLI
- [ ] Baseline ethics benchmark (precision/recall)
- [ ] Threshold configuration versioned

## 4. Observability
- [ ] Metrics: actions_total, latency histograms, violations_total
- [ ] Tracing: 10% sample baseline, 100% errors
- [ ] Log sanitization (PII redaction)
- [ ] Alert rules: latency, error rate, drift, quota saturation

## 5. Performance
- [ ] Sustained load test artifact (report & raw metrics)
- [ ] Burst test (5× baseline) passing
- [ ] Soak test (2h) with no memory leak >5%

## 6. Resilience
- [ ] Chaos test: pod kill, region failover
- [ ] Quorum / failover time < target
- [ ] Automated backup restore dry-run < target RTO

## 7. Data & Storage
- [ ] Tiered retention config documented
- [ ] Compression ratio >5:1 aggregate
- [ ] Projection for 12 months < budget threshold

## 8. Plugin Trust
- [ ] Signature verification enforced
- [ ] Trust score gating (threshold ≥80)
- [ ] Vulnerability scan per plugin load

## 9. Human Review
- [ ] Review queue SLA dashboard live
- [ ] Feedback taxonomy coverage report
- [ ] Reviewer drift metrics < 5%

## 10. Transparency
- [ ] Quarterly transparency report auto-generated
- [ ] Public methodology (risk scoring, detection)
- [ ] Anchored Merkle roots registry

## 11. Release & Change
- [ ] Versioned policy pack
- [ ] Rollback procedure tested
- [ ] Canary deployment config

## 12. Compliance
- [ ] Data residency mapping
- [ ] GDPR / CCPA data flow diagram
- [ ] Access request / deletion workflow tested
