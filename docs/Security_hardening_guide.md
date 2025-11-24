# Security Hardening Guide

## Layers
1. Perimeter: Load balancer + WAF (prompt injection regex, payload size cap).
2. AuthN/Z: JWT (RS256), RBAC matrix, scoped API keys.
3. Input Validation: Strict Pydantic schemas, length caps, Unicode normalization.
4. Secrets: Vault / KMS; no inline secrets in manifests.
5. Supply Chain: Signed container images, SBOM diff gate.
6. Runtime: Non-root user (already), seccomp/apparmor profiles, read-only FS (where possible).
7. Network: Namespace isolation, network policies (allow Redis/API DB only).
8. Logging: PII redaction filters; structured JSON logs.
9. Audit Integrity: Merkle anchoring + external timestamping.
10. Plugin Trust: Signature + static analysis + reputation gating.

## Mandatory Controls (World Class)
- [ ] mTLS between internal services OR service mesh (Istio/Linkerd)
- [ ] Automatic secret rotation (≤90 days)
- [ ] Vulnerability SLA: Critical <24h, High <72h
- [ ] Zero trust network segmentation (deny-all default)
- [ ] Attestation of build pipeline (SLSA level target)
