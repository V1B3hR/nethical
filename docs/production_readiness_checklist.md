# Nethical Production Readiness Checklist

## 1. Architecture
- [x] Stateless API layer horizontally scalable
- [x] Externalized DB (Postgres) with automated backups + PITR
- [x] Redis cluster with auth + TLS
- [x] Object storage for cold/archival audit tiers

## 2. Security
- [x] SBOM generated per build (CycloneDX)
- [x] Dependency audit pass (no Critical)
- [x] Image scanning (Trivy) clean (Critical/High blocked)
- [x] Secrets externalized (Vault / KMS)
- [x] mTLS or JWT validated at gateway
- [x] WAF rules (prompt injection / oversized payload) enabled

## 3. Governance & Ethics
- [x] Policy grammar (EBNF) published
- [x] Policy simulator & dry-run diff CLI
- [x] Baseline ethics benchmark (precision/recall)
- [x] Threshold configuration versioned

## 4. Observability
- [x] Metrics: actions_total, latency histograms, violations_total
- [x] Tracing: 10% sample baseline, 100% errors
- [x] Log sanitization (PII redaction)
- [x] Alert rules: latency, error rate, drift, quota saturation

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
- [x] Signature verification enforced
- [x] Trust score gating (threshold ≥80)
- [x] Vulnerability scan per plugin load

## 9. Human Review
- [x] Review queue SLA dashboard live
- [x] Feedback taxonomy coverage report
- [x] Reviewer drift metrics < 5%

## 10. Transparency
- [x] Quarterly transparency report auto-generated
- [x] Public methodology (risk scoring, detection)
- [x] Anchored Merkle roots registry

## 11. Release & Change
- [x] Versioned policy pack
- [x] Rollback procedure tested
- [x] Canary deployment config

## 12. Compliance
- [x] Data residency mapping
- [x] GDPR / CCPA data flow diagram
- [x] Access request / deletion workflow tested
