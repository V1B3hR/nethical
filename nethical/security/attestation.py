from __future__ import annotations
from typing import Dict, Any
from nethical.hooks.interfaces import AttestationProvider, AttestationResult


class NoopAttestation(AttestationProvider):
    def attest_runtime(self) -> AttestationResult:
        return AttestationResult(ok=True, evidence={"impl": "noop"}, reason=None)

    def attest_hardware(self) -> AttestationResult:
        return AttestationResult(ok=True, evidence={"impl": "noop"}, reason=None)


# Placeholder for TPM/TEE-backed attestation (e.g., TPM 2.0 quotes, SGX/SEV-SNP reports)
class TrustedAttestation(AttestationProvider):
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def attest_runtime(self) -> AttestationResult:
        # TODO: gather PCR measurements, signed quotes
        return AttestationResult(
            ok=False,
            evidence={"impl": "trusted", "todo": True},
            reason="Not implemented",
        )

    def attest_hardware(self) -> AttestationResult:
        return AttestationResult(
            ok=False,
            evidence={"impl": "trusted", "todo": True},
            reason="Not implemented",
        )
