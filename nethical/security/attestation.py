from __future__ import annotations

"""
Attestation providers aligned with nethical's hooks.interfaces.

- Provides side-effect free runtime and hardware attestation suitable for
  zero-trust admission and continuous verification.
- Structures evidence to match hooks.interfaces.Evidence.
- Returns AttestationResult with machine-readable codes and human-readable reasons.
"""

from typing import Any, Dict, Optional, Tuple, Literal
import os
import sys
import platform
import json
import logging
from datetime import datetime, timezone

from nethical.hooks.interfaces import (
    AttestationProvider,
    AttestationResult,
    Evidence,
)

__all__ = [
    "NoopAttestation",
    "TrustedAttestation",
    "select_attestation_provider",
    "normalize_attestation_result",
]

log = logging.getLogger(__name__)

# ---- Error Codes (machine-readable) ----
ERR_NOT_CONFIGURED = "not_configured"
ERR_NOT_IMPLEMENTED = "not_implemented"
ERR_VERIFICATION_FAILED = "verification_failed"
ERR_RUNTIME_ERROR = "runtime_error"


def _as_str_map(d: Dict[str, Any]) -> Dict[str, str]:
    """Safely convert any values to strings for Evidence.measurements."""
    return {str(k): (json.dumps(v, sort_keys=True) if not isinstance(v, (str, bytes)) else str(v)) for k, v in d.items()}


def _baseline_measurements() -> Dict[str, str]:
    """Lightweight, deterministic baseline measurements for observability."""
    data = {
        "python.version": sys.version.split()[0],
        "python.implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "pid": str(os.getpid()),
        "tz_utc_now": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return _as_str_map(data)


def _base_evidence(
    verifier: str,
    measurements: Optional[Dict[str, str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Evidence:
    ev: Evidence = {
        "verifier": verifier,
        "measurements": measurements or {},
        "meta": (meta or {}),
    }
    return ev


def normalize_attestation_result(result: AttestationResult) -> AttestationResult:
    """
    Normalize AttestationResult to ensure stable Evidence structure and types.
    """
    ev = result.evidence or {}
    measurements = ev.get("measurements") or {}
    meta = ev.get("meta") or {}

    # Ensure string-typed measurements; keep meta free-form.
    ev["measurements"] = _as_str_map(measurements)
    ev["meta"] = dict(meta)

    # Ensure verifier present for auditability.
    if "verifier" not in ev or not ev["verifier"]:
        ev["verifier"] = "nethical/attestation"

    return AttestationResult(
        ok=result.ok,
        evidence=ev,  # type: ignore[assignment]
        reason=result.reason,
        code=result.code,
        created_at=result.created_at,  # already UTC in model default
    )


class NoopAttestation(AttestationProvider):
    """
    Permissive attestation for development and testing.

    - Always returns ok=True.
    - Provides baseline environment measurements for audit trails.
    - Marked explicitly as 'noop' to prevent accidental use in production.
    """

    def __init__(self, meta: Optional[Dict[str, Any]] = None):
        self.meta = meta or {}

    def attest_runtime(self) -> AttestationResult:
        evidence: Evidence = _base_evidence(
            verifier="nethical/noop",
            measurements={
                **_baseline_measurements(),
                "attestation.scope": "runtime",
            },
            meta={"impl": "noop", "policy": "permit_all", **self.meta},
        )
        return normalize_attestation_result(
            AttestationResult(ok=True, evidence=evidence)
        )

    def attest_hardware(self) -> AttestationResult:
        evidence: Evidence = _base_evidence(
            verifier="nethical/noop",
            measurements={
                **_baseline_measurements(),
                "attestation.scope": "hardware",
            },
            meta={"impl": "noop", "policy": "permit_all", **self.meta},
        )
        return normalize_attestation_result(
            AttestationResult(ok=True, evidence=evidence)
        )


class TrustedAttestation(AttestationProvider):
    """
    Placeholder for TPM/TEE/SEV/TDX-backed attestation.

    Configuration (all optional, shape is stable for future implementations):
    - enabled: bool (default False) gate for production use
    - provider: Literal['tpm2', 'sgx', 'sev-snp', 'tdx', 'custom']
    - require_cert_chain: bool (default False)
    - expected_tcb_min: str (e.g., '1.0.0')
    - verifier_name: str (default 'nethical/trusted')
    - meta: Dict[str, Any] extra metadata to include
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.provider: Optional[str] = cfg.get("provider")
        self.require_cert_chain: bool = bool(cfg.get("require_cert_chain", False))
        self.expected_tcb_min: Optional[str] = cfg.get("expected_tcb_min")
        self.verifier_name: str = str(cfg.get("verifier_name") or "nethical/trusted")
        self.meta: Dict[str, Any] = dict(cfg.get("meta") or {})

    def _disabled_result(self, scope: Literal["runtime", "hardware"]) -> AttestationResult:
        evidence: Evidence = _base_evidence(
            verifier=self.verifier_name,
            measurements={
                **_baseline_measurements(),
                "attestation.scope": scope,
                "provider": self.provider or "unspecified",
            },
            meta={"impl": "trusted", "enabled": False, **self.meta},
        )
        return normalize_attestation_result(
            AttestationResult(
                ok=False,
                evidence=evidence,
                reason="Trusted attestation not configured/enabled",
                code=ERR_NOT_CONFIGURED,
            )
        )

    def _not_implemented(self, scope: Literal["runtime", "hardware"]) -> AttestationResult:
        evidence: Evidence = _base_evidence(
            verifier=self.verifier_name,
            measurements={
                **_baseline_measurements(),
                "attestation.scope": scope,
                "provider": self.provider or "unspecified",
            },
            meta={"impl": "trusted", "todo": True, **self.meta},
        )
        return normalize_attestation_result(
            AttestationResult(
                ok=False,
                evidence=evidence,
                reason="Trusted attestation provider not implemented",
                code=ERR_NOT_IMPLEMENTED,
            )
        )

    def _gather_stubbed_evidence(self, scope: Literal["runtime", "hardware"]) -> Evidence:
        """
        Build a structured evidence envelope with placeholder fields that align
        with hooks.interfaces.Evidence for downstream auditors.
        """
        measurements = {
            **_baseline_measurements(),
            "attestation.scope": scope,
            "provider": self.provider or "unspecified",
        }
        meta = {
            "impl": "trusted",
            "note": "stub evidence; verification not performed",
            **self.meta,
        }
        # Placeholders for future real quotes/certs/TCB
        ev: Evidence = {
            "verifier": self.verifier_name,
            "runtime_quote": "UNAVAILABLE",
            "tcb_version": self.expected_tcb_min or "UNSPECIFIED",
            "cert_chain_pem": "" if not self.require_cert_chain else "UNAVAILABLE",
            "measurements": measurements,
            "meta": meta,
        }
        return ev

    def attest_runtime(self) -> AttestationResult:
        if not self.enabled:
            return self._disabled_result(scope="runtime")

        # In the future, branch on self.provider and gather real quotes.
        try:
            evidence = self._gather_stubbed_evidence(scope="runtime")
            return normalize_attestation_result(
                AttestationResult(
                    ok=False,
                    evidence=evidence,
                    reason="Runtime attestation not implemented for selected provider",
                    code=ERR_NOT_IMPLEMENTED,
                )
            )
        except Exception as e:
            log.exception("Trusted runtime attestation failed: %s", e)
            evidence = _base_evidence(
                verifier=self.verifier_name,
                measurements={"attestation.scope": "runtime", **_baseline_measurements()},
                meta={"impl": "trusted", "exception": type(e).__name__},
            )
            return normalize_attestation_result(
                AttestationResult(
                    ok=False,
                    evidence=evidence,
                    reason=str(e),
                    code=ERR_RUNTIME_ERROR,
                )
            )

    def attest_hardware(self) -> AttestationResult:
        if not self.enabled:
            return self._disabled_result(scope="hardware")

        try:
            evidence = self._gather_stubbed_evidence(scope="hardware")
            return normalize_attestation_result(
                AttestationResult(
                    ok=False,
                    evidence=evidence,
                    reason="Hardware attestation not implemented for selected provider",
                    code=ERR_NOT_IMPLEMENTED,
                )
            )
        except Exception as e:
            log.exception("Trusted hardware attestation failed: %s", e)
            evidence = _base_evidence(
                verifier=self.verifier_name,
                measurements={"attestation.scope": "hardware", **_baseline_measurements()},
                meta={"impl": "trusted", "exception": type(e).__name__},
            )
            return normalize_attestation_result(
                AttestationResult(
                    ok=False,
                    evidence=evidence,
                    reason=str(e),
                    code=ERR_RUNTIME_ERROR,
                )
            )


def select_attestation_provider(config: Optional[Dict[str, Any]] = None) -> AttestationProvider:
    """
    Helper to select an AttestationProvider based on config.
    Examples:
        {} -> NoopAttestation
        {"type": "trusted", "enabled": true, "provider": "tpm2"} -> TrustedAttestation
    """
    cfg = config or {}
    type_name = str(cfg.get("type") or "noop").lower()
    if type_name in ("trusted", "tpm2", "sgx", "sev-snp", "tdx", "custom"):
        return TrustedAttestation(cfg)
    return NoopAttestation(meta=cfg.get("meta") or {})
