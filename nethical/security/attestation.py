from __future__ import annotations

"""
Advanced attestation providers aligned with nethical.hooks.interfaces.

Enhancements:
- Provider registry for extensibility.
- Canonical measurement hashing & nonce.
- Optional async APIs.
- Policy evaluation scaffold (stub).
- Structured error codes and normalization.
- Deterministic, cacheable baseline measurements.
- Backward compatible with existing interfaces.
"""

from typing import Any, Dict, Optional, Tuple, Literal, Callable
import os
import sys
import platform
import json
import logging
import hashlib
import secrets
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

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
    "AttestationErrorCodes",
    "register_attestation_provider",
    "compute_measurements_digest",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error / status codes
# ---------------------------------------------------------------------------
class AttestationErrorCodes:
    NOT_CONFIGURED = "not_configured"
    NOT_IMPLEMENTED = "not_implemented"
    VERIFICATION_FAILED = "verification_failed"
    RUNTIME_ERROR = "runtime_error"
    POLICY_DENIED = "policy_denied"
    POLICY_INDETERMINATE = "policy_indeterminate"


ERR_NOT_CONFIGURED = AttestationErrorCodes.NOT_CONFIGURED
ERR_NOT_IMPLEMENTED = AttestationErrorCodes.NOT_IMPLEMENTED
ERR_VERIFICATION_FAILED = AttestationErrorCodes.VERIFICATION_FAILED
ERR_RUNTIME_ERROR = AttestationErrorCodes.RUNTIME_ERROR
ERR_POLICY_DENIED = AttestationErrorCodes.POLICY_DENIED
ERR_POLICY_INDETERMINATE = AttestationErrorCodes.POLICY_INDETERMINATE


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
_PROVIDER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], AttestationProvider]] = {}


def register_attestation_provider(
    type_name: str,
    factory: Callable[[Dict[str, Any]], AttestationProvider],
    override: bool = False,
) -> None:
    """
    Register a custom attestation provider type.
    """
    key = type_name.lower().strip()
    if key in _PROVIDER_REGISTRY and not override:
        raise ValueError(f"Provider '{type_name}' already registered")
    _PROVIDER_REGISTRY[key] = factory


# ---------------------------------------------------------------------------
# Utility & canonicalization helpers
# ---------------------------------------------------------------------------

_ALLOWED_KEY_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:")


def _sanitize_key(key: str) -> str:
    return "".join(c for c in key if c in _ALLOWED_KEY_CHARS)[:256]


def _as_str_map(d: Dict[str, Any]) -> Dict[str, str]:
    """Safely convert any values to strings for Evidence.measurements."""
    out: Dict[str, str] = {}
    for k, v in d.items():
        sk = _sanitize_key(str(k))
        if isinstance(v, (str, bytes)):
            out[sk] = str(v)
        else:
            try:
                out[sk] = json.dumps(v, sort_keys=True, separators=(",", ":"))
            except Exception:
                out[sk] = repr(v)
    return out


# Caching baseline measurements (lightweight; extend if heavy operations appear)
_BASELINE_CACHE: Dict[str, Tuple[datetime, Dict[str, str]]] = {}
_BASELINE_TTL = timedelta(seconds=30)
_PROCESS_START_UTC = datetime.now(timezone.utc)
_PROCESS_START_MONO = time.monotonic()


def _baseline_measurements(debug_extra: bool = False) -> Dict[str, str]:
    """
    Deterministic baseline: basic runtime info + stable process data.
    debug_extra (False): if True, includes additional introspection (safe subset).
    """
    now = datetime.now(timezone.utc)
    cache_key = f"base:{int(now.timestamp() // _BASELINE_TTL.total_seconds())}:{int(debug_extra)}"
    cached = _BASELINE_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _BASELINE_TTL:
        return dict(cached[1])

    base = {
        "python.version": sys.version.split()[0],
        "python.implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "pid": str(os.getpid()),
        "tz_utc_now": now.isoformat(timespec="seconds"),
        "process.start_utc": _PROCESS_START_UTC.isoformat(timespec="seconds"),
        "process.uptime_seconds": f"{time.monotonic() - _PROCESS_START_MONO:.2f}",
    }

    if debug_extra:
        # Add only non-sensitive, safe debug hints (avoid entire env or secrets).
        base.update(
            {
                "python.executable": sys.executable,
                "python.argv_count": str(len(sys.argv)),
            }
        )

    measurements = _as_str_map(base)
    _BASELINE_CACHE[cache_key] = (now, measurements)
    return measurements


def compute_measurements_digest(measurements: Dict[str, str]) -> str:
    """
    Produce a canonical SHA-256 digest of the measurements for tamper-evidence.
    Keys sorted, JSON canonical form (compact).
    """
    canonical = json.dumps(measurements, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _base_evidence(
    verifier: str,
    measurements: Optional[Dict[str, str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    nonce: Optional[str] = None,
) -> Evidence:
    ev: Evidence = {
        "verifier": verifier,
        "measurements": measurements or {},
        "meta": (meta or {}),
        "evidence_version": "1",
        "nonce": nonce or secrets.token_urlsafe(16),
    }
    # Digest added later after full measurement assembly
    return ev


def _finalize_evidence(ev: Evidence) -> Evidence:
    """
    Ensure canonical string measurements and add digest.
    """
    measurements = ev.get("measurements") or {}
    str_map = _as_str_map(measurements)
    ev["measurements"] = str_map  # type: ignore
    ev["measurements_digest"] = compute_measurements_digest(str_map)
    return ev


def normalize_attestation_result(result: AttestationResult) -> AttestationResult:
    """
    Normalize AttestationResult to ensure stable Evidence structure and types.
    Maintains backward compatibility.
    """
    ev = result.evidence or {}
    # finalize to ensure digest & canonicalization
    ev = _finalize_evidence(ev)

    if "verifier" not in ev or not ev["verifier"]:
        ev["verifier"] = "nethical/attestation"

    return AttestationResult(
        ok=result.ok,
        evidence=ev,  # type: ignore[assignment]
        reason=result.reason,
        code=result.code,
        created_at=result.created_at,
    )


# ---------------------------------------------------------------------------
# Policy evaluation scaffold
# ---------------------------------------------------------------------------


@dataclass
class PolicyOutcome:
    decision: Literal["permit", "deny", "indeterminate"]
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


def evaluate_policy_stub(
    measurements: Dict[str, str],
    expected_tcb_min: Optional[str],
) -> PolicyOutcome:
    """
    Placeholder policy: if expected_tcb_min is specified and a tcb_version measurement
    exists, do a naive lexical comparison (for illustration only).
    """
    tcb = measurements.get("tcb_version")
    if expected_tcb_min and tcb:
        # NOTE: Real logic would parse semver or vendor-specific version
        if tcb >= expected_tcb_min:
            return PolicyOutcome("permit", f"TCB {tcb} >= {expected_tcb_min}")
        return PolicyOutcome(
            "deny", f"TCB {tcb} < {expected_tcb_min}", {"expected_min": expected_tcb_min}
        )
    if expected_tcb_min and not tcb:
        return PolicyOutcome(
            "indeterminate",
            "Expected TCB reference not present",
            {"expected_min": expected_tcb_min},
        )
    return PolicyOutcome("permit", "No TCB policy specified")


# ---------------------------------------------------------------------------
# Noop Attestation
# ---------------------------------------------------------------------------


class NoopAttestation(AttestationProvider):
    """
    Permissive attestation for development and testing.

    - Always returns ok=True (policy: permit_all).
    - Provides baseline environment measurements for audit trails.
    - Marked explicitly as 'noop' to prevent accidental production acceptance.
    """

    def __init__(self, meta: Optional[Dict[str, Any]] = None, debug_extra: bool = False):
        self.meta = meta or {}
        self.debug_extra = debug_extra

    def _build(self, scope: Literal["runtime", "hardware"]) -> AttestationResult:
        base_measurements = _baseline_measurements(debug_extra=self.debug_extra)
        measurements = {
            **base_measurements,
            "attestation.scope": scope,
            "attestation.provider": "noop",
        }
        evidence: Evidence = _base_evidence(
            verifier="nethical/noop",
            measurements=measurements,
            meta={"impl": "noop", "policy": "permit_all", **self.meta},
        )
        return normalize_attestation_result(
            AttestationResult(ok=True, evidence=evidence, reason="Noop attestation accepted")
        )

    # Sync APIs
    def attest_runtime(self) -> AttestationResult:
        return self._build("runtime")

    def attest_hardware(self) -> AttestationResult:
        return self._build("hardware")

    # Async variants for symmetry (optional)
    async def async_attest_runtime(self) -> AttestationResult:
        return self.attest_runtime()

    async def async_attest_hardware(self) -> AttestationResult:
        return self.attest_hardware()


# ---------------------------------------------------------------------------
# Trusted Attestation (stubbed enhanced)
# ---------------------------------------------------------------------------


@dataclass
class TrustedProviderConfig:
    enabled: bool = False
    provider: Optional[str] = None  # 'tpm2', 'sgx', 'sev-snp', 'tdx', 'custom'
    require_cert_chain: bool = False
    expected_tcb_min: Optional[str] = None
    verifier_name: str = "nethical/trusted"
    meta: Dict[str, Any] = field(default_factory=dict)
    debug_extra: bool = False
    require_strict_policy: bool = False  # if True, indeterminate -> failure
    # Future fields: cert_store_path, quote_timeout, network_endpoints, etc.


class TrustedAttestation(AttestationProvider):
    """
    Enhanced placeholder for TPM/TEE/SEV/TDX-backed attestation.

    Currently stubbed: real cryptographic verification is NOT performed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg_dict = config or {}
        # Map into dataclass (ignore unknown keys gracefully).
        self.cfg = TrustedProviderConfig(
            enabled=bool(cfg_dict.get("enabled", False)),
            provider=cfg_dict.get("provider"),
            require_cert_chain=bool(cfg_dict.get("require_cert_chain", False)),
            expected_tcb_min=cfg_dict.get("expected_tcb_min"),
            verifier_name=str(cfg_dict.get("verifier_name") or "nethical/trusted"),
            meta=dict(cfg_dict.get("meta") or {}),
            debug_extra=bool(cfg_dict.get("debug_extra", False)),
            require_strict_policy=bool(cfg_dict.get("require_strict_policy", False)),
        )

    # ------------- Internal helpers -------------

    def _disabled_result(self, scope: Literal["runtime", "hardware"]) -> AttestationResult:
        measurements = {
            **_baseline_measurements(debug_extra=self.cfg.debug_extra),
            "attestation.scope": scope,
            "provider": self.cfg.provider or "unspecified",
        }
        evidence = _base_evidence(
            verifier=self.cfg.verifier_name,
            measurements=measurements,
            meta={"impl": "trusted", "enabled": False, **self.cfg.meta},
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
        measurements = {
            **_baseline_measurements(debug_extra=self.cfg.debug_extra),
            "attestation.scope": scope,
            "provider": self.cfg.provider or "unspecified",
        }
        evidence = _base_evidence(
            verifier=self.cfg.verifier_name,
            measurements=measurements,
            meta={"impl": "trusted", "todo": True, **self.cfg.meta},
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
        Builds a structured evidence envelope with placeholder fields.
        Future real implementation would replace runtime_quote, cert_chain_pem, tcb_version.
        """
        measurements = {
            **_baseline_measurements(debug_extra=self.cfg.debug_extra),
            "attestation.scope": scope,
            "provider": self.cfg.provider or "unspecified",
            "tcb_version": self.cfg.expected_tcb_min or "UNSPECIFIED",
        }
        meta = {
            "impl": "trusted",
            "note": "stub evidence; cryptographic verification not performed",
            **self.cfg.meta,
        }
        ev: Evidence = {
            "verifier": self.cfg.verifier_name,
            "runtime_quote": "UNAVAILABLE",
            "tcb_version": measurements.get("tcb_version"),
            "cert_chain_pem": "" if not self.cfg.require_cert_chain else "UNAVAILABLE",
            "measurements": measurements,
            "meta": meta,
        }
        return ev

    def _evaluate_policy(
        self,
        measurements: Dict[str, str],
    ) -> PolicyOutcome:
        return evaluate_policy_stub(
            measurements=measurements,
            expected_tcb_min=self.cfg.expected_tcb_min,
        )

    def _policy_to_result(
        self,
        base_result: AttestationResult,
        policy_outcome: PolicyOutcome,
    ) -> AttestationResult:
        """
        Merge policy outcome into evidence meta and adjust result if denied.
        """
        ev = base_result.evidence or {}
        meta = ev.get("meta") or {}
        policy_meta = {
            "policy.decision": policy_outcome.decision,
            "policy.reason": policy_outcome.reason,
            "policy.details": policy_outcome.details,
        }
        meta.update(policy_meta)
        ev["meta"] = meta  # type: ignore

        ok = base_result.ok
        code = base_result.code
        reason = base_result.reason

        if policy_outcome.decision == "deny":
            ok = False
            code = ERR_POLICY_DENIED
            reason = f"{reason}; policy denied: {policy_outcome.reason}"
        elif policy_outcome.decision == "indeterminate" and self.cfg.require_strict_policy:
            ok = False
            code = ERR_POLICY_INDETERMINATE
            reason = f"{reason}; policy indeterminate (strict mode)"

        return AttestationResult(
            ok=ok,
            evidence=ev,  # type: ignore
            reason=reason,
            code=code,
            created_at=base_result.created_at,
        )

    # ------------- Core attestation paths -------------

    def _attest(
        self,
        scope: Literal["runtime", "hardware"],
    ) -> AttestationResult:
        if not self.cfg.enabled:
            return self._disabled_result(scope)
        try:
            evidence = self._gather_stubbed_evidence(scope)
            # Currently always not implemented (no real provider logic yet)
            result = normalize_attestation_result(
                AttestationResult(
                    ok=False,
                    evidence=evidence,
                    reason=f"{scope.capitalize()} attestation not implemented for selected provider",
                    code=ERR_NOT_IMPLEMENTED,
                )
            )
            # Policy evaluation (still applied to stub)
            policy_outcome = self._evaluate_policy(result.evidence.get("measurements", {}))  # type: ignore
            result = self._policy_to_result(result, policy_outcome)
            return normalize_attestation_result(result)
        except Exception as e:
            log.exception(
                "Trusted %s attestation failed",
                scope,
                extra={"scope": scope, "provider": self.cfg.provider},
            )
            evidence = _base_evidence(
                verifier=self.cfg.verifier_name,
                measurements={
                    "attestation.scope": scope,
                    **_baseline_measurements(debug_extra=False),
                },
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

    # Sync API
    def attest_runtime(self) -> AttestationResult:
        return self._attest("runtime")

    def attest_hardware(self) -> AttestationResult:
        return self._attest("hardware")

    # Async variants for future real IO bound operations
    async def async_attest_runtime(self) -> AttestationResult:
        return self.attest_runtime()

    async def async_attest_hardware(self) -> AttestationResult:
        return self.attest_hardware()


# Register built-in provider types
register_attestation_provider(
    "noop",
    lambda cfg: NoopAttestation(
        meta=cfg.get("meta") or {}, debug_extra=bool(cfg.get("debug_extra", False))
    ),
    override=True,
)
for p in ("trusted", "tpm2", "sgx", "sev-snp", "tdx", "custom"):
    register_attestation_provider(
        p,
        lambda cfg, _p=p: TrustedAttestation({**cfg, "provider": cfg.get("provider") or _p}),
        override=True,
    )


def select_attestation_provider(config: Optional[Dict[str, Any]] = None) -> AttestationProvider:
    """
    Helper to select an AttestationProvider based on config.

    Examples:
        {} -> NoopAttestation
        {"type": "trusted", "enabled": true, "provider": "tpm2"} -> TrustedAttestation

    Allows custom registered providers.
    """
    cfg = config or {}
    type_name = str(cfg.get("type") or "noop").lower()
    factory = _PROVIDER_REGISTRY.get(type_name)
    if factory:
        return factory(cfg)
    # Fallback to noop if unknown type (fail-closed could be considered optionally)
    log.warning("Unknown attestation provider type '%s', defaulting to noop", type_name)
    return _PROVIDER_REGISTRY["noop"](cfg)
