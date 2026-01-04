# Defense-Medical Hooks Design

Goal: Keep healthcare-first deployments compliant and simple, while providing optional, pluggable hooks to enable defense-medical capabilities (contested/offline environments, authenticated role signaling, attestation, geofencing, and later ROE/LOAC constraints).

## Hooks Overview

- AttestationProvider: Hardware/runtime attestation before enabling sensitive tool-use or mission roles.
- CryptoSignalProvider: Coalition-authenticated “medical” signaling (IFF-style); mitigates spoofing. Disabled by default.
- CommsPolicy: Zero-trust comms enforcement (mTLS, SPIFFE identities, allowlists).
- GeoFenceProvider: Mission/clinical geofencing restrictions (future).
- OfflineStore: Tamper-evident, append-only logs with Merkle roots and optional TSA anchoring for offline/contested ops.
- RoleAuthorityResolver: Jurisdiction-aware role binding (clinician, medic, operator).
- ExportControlAdvisor: Feature gating for dual-use/export-sensitive capabilities (NATO profile hooks only).

## Region Profiles

- US: HIPAA + SOC2 baseline, PHI minimization, residency required.
- UK: UK GDPR/DPA/NHS DSPT.
- EU: EU GDPR/NIS2, multi-lingual PHI patterns, strict residency.
- NATO: Enables hooks only; healthcare defaults remain. Defense features must be explicitly configured and reviewed.

## Policy Readiness

- Current: Healthcare policies (non-diagnostic defaults, emergency routing, PHI redaction, manipulation blocking).
- Extensible: Add ROE/LOAC rules via policy DSL overlays (e.g., mission_role transitions require dual approval + attestation + crypto signal).

## Safety in Contested Environments

- Fail-safe: If attestation or crypto signaling fails, degrade to safe-mode; restrict tool-use; escalate to human.
- Identity/workload security: Target SPIFFE/SPIRE or equivalent for workload identity & mTLS.
- Offline-first: Buffer events; compute Merkle roots; anchor with TSA when connectivity resumes.
- Integrity: Signed configuration and policy bundles, immutable audit, promotion gate for changes.

## Next Steps

- Implement MTLSCommsPolicy with SPIFFE, add GeoFenceProvider, and TSA client.
- Add policy overlays for ROE/LOAC with approval chains and export-control checks.
- Integrate device attestation libraries (TPM/TEE) for TrustedAttestation.
