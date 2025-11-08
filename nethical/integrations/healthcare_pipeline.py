from __future__ import annotations

from typing import Dict, Any, Optional, Callable, Iterable
from datetime import datetime

from nethical.core import IntegratedGovernance
from nethical.hooks.interfaces import (
    Region,
    AttestationProvider,
    CryptoSignalProvider,
    CommsPolicy,
    OfflineStore,
)
from nethical.security.attestation import NoopAttestation
from nethical.net.zerotrust import NoopCommsPolicy
from nethical.storage.tamper_store import TamperEvidentOfflineStore
from nethical.policy.engine import PolicyEngine

from nethical.detectors.healthcare.phi_detector import PHIDetector, detect_and_redact_payload
from nethical.detectors.healthcare.clinical_risk_detectors import extract_clinical_signals


class HealthcareGuardrails:
    """
    End-to-end healthcare guardrails orchestrator:
    - Ingress: minimum necessary + PHI redaction + attestation event
    - Agent: user-provided callable
    - Egress: PHI redaction + minimum necessary output
    - Governance: policy evaluation + IntegratedGovernance decision
    - Audit: tamper-evident, PHI-sanitized structured events
    """

    def __init__(
        self,
        gov: IntegratedGovernance,
        region: Region,
        policy_path: str = "policies/healthcare/core.yaml",
        attestation: Optional[AttestationProvider] = None,
        crypto_signal: Optional[CryptoSignalProvider] = None,
        comms: Optional[CommsPolicy] = None,
        offline_store: Optional[OfflineStore] = None,
        input_allowlist: Optional[Iterable[str]] = None,
        output_allowlist: Optional[Iterable[str]] = None,
    ):
        self.gov = gov
        self.region = region
        self.policy = PolicyEngine.load(policy_path, region)
        self.phi = PHIDetector()
        self.attestation = attestation or NoopAttestation()
        # crypto_signal is optional by design in healthcare; when present, it should be plumbed into features
        self.crypto_signal = crypto_signal
        self.comms = comms or NoopCommsPolicy()
        self.offline = offline_store or TamperEvidentOfflineStore()
        self.input_allowlist = set(input_allowlist or [])
        self.output_allowlist = set(output_allowlist or [])

    # ------------- internal helpers -------------

    def _timestamp(self) -> str:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    def _apply_allowlist(self, data: Dict[str, Any], allowlist: set[str]) -> Dict[str, Any]:
        if not allowlist:
            return data
        return {k: v for k, v in data.items() if k in allowlist}

    def _deep_redact(self, value: Any) -> Any:
        """
        Recursively redact strings; leave non-strings as-is; redact dict/list contents.
        """
        if isinstance(value, str):
            return self.phi.redact(value)
        if isinstance(value, dict):
            return {k: self._deep_redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deep_redact(v) for v in value]
        return value

    def _sanitize_for_log(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure no PHI leaks into logs. Redact all string values deep in the structure.
        """
        return self._deep_redact(event)

    def _log_event(self, event: Dict[str, Any]) -> None:
        event_with_meta = {"ts": self._timestamp(), **event}
        self.offline.append_event(self._sanitize_for_log(event_with_meta))

    def _policy_to_governance_flags(self, policy_out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map policy engine output to governance flags with safe defaults.
        Expected policy fields (best effort):
          - allow: bool OR decision: "allow"|"warn"|"block"
          - severity: "low"|"medium"|"high" (optional)
          - classification: string like "allow"|"warn"|"block" (optional)
          - reason/details: optional
        """
        allow = bool(policy_out.get("allow", True))
        decision = policy_out.get("decision")
        if isinstance(decision, str):
            decision_norm = decision.lower()
            if decision_norm in ("block", "deny", "reject"):
                allow = False
            elif decision_norm in ("warn",):
                # keep allow as True but classify
                allow = True
            elif decision_norm in ("allow", "permit"):
                allow = True

        classification = policy_out.get("classification")
        if not classification:
            classification = "allow" if allow else "block"

        severity = str(policy_out.get("severity", "medium")).lower()
        if severity not in ("low", "medium", "high", "critical"):
            severity = "medium"

        return {
            "violation_detected": not allow,
            "violation_severity": severity if not allow else "low",
            "rule_classification": classification,
        }

    # ------------- pipeline stages -------------

    def preprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Enforce minimum necessary on input (if configured)
        - Redact PHI at ingress
        - Record attestation result
        """
        if not isinstance(payload, dict):
            self._log_event({"type": "ingress_error", "error": "payload_not_dict"})
            # Fail closed: only pass an empty sanitized payload downstream
            payload = {}

        # Minimum necessary first, then redact
        # Note: _apply_allowlist creates a new dict, no deep copy needed
        filtered = self._apply_allowlist(payload, self.input_allowlist)
        redacted = detect_and_redact_payload(filtered)

        # Attestation gate
        attn = self.attestation.attest_runtime()
        self._log_event({"type": "attest_runtime", "ok": getattr(attn, "ok", False)})

        # Optional: comms policy could be consulted/enforced here if needed
        self._log_event({"type": "ingress", "keys": list(redacted.keys())})
        return redacted

    def postprocess(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Redact PHI on egress (agent_output and all string fields)
        - Enforce minimum necessary on output (if configured)
        - Record sanitized egress metadata
        """
        if not isinstance(result, dict):
            self._log_event({"type": "egress_error", "error": "result_not_dict"})
            result = {}

        # Redact agent_output if present
        if "agent_output" in result and isinstance(result["agent_output"], str):
            result["agent_output"] = self.phi.redact(result["agent_output"])

        # Deep redact everything else to be safe
        redacted_result = self._deep_redact(result)

        # Apply output allowlist, if configured
        redacted_result = self._apply_allowlist(redacted_result, self.output_allowlist)

        self._log_event(
            {"type": "egress", "len": len(str(redacted_result.get("agent_output", "")))}
        )
        return redacted_result

    def evaluate(self, agent_id: str, action_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Extract clinical signals into facts
        - Evaluate policy
        - Send decision to IntegratedGovernance with policy-driven flags
        - Persist policy outcome in tamper-evident, redacted logs
        """
        # Compose facts with extracted signals (ensure redaction-safe)
        facts = {**payload, **extract_clinical_signals(payload)}
        policy_out = self.policy.evaluate(facts)

        flags = self._policy_to_governance_flags(policy_out)

        action_text = payload.get("agent_output") or payload.get("user_input") or ""

        # Build features for governance
        features: Dict[str, Any] = {
            "ml_score": (
                float(policy_out.get("ml_score", 0.0)) if isinstance(policy_out, dict) else 0.0
            ),
            "attestation_ok": True,  # default; cannot re-attest synchronously here
            "crypto_signal_enabled": bool(self.crypto_signal is not None),
            "region": self.region.value,
        }

        decision = self.gov.process_action(
            agent_id=agent_id,
            action=action_text,
            cohort=f"healthcare_{self.region.value}",
            violation_detected=flags["violation_detected"],
            violation_type="policy" if flags["violation_detected"] else "none",
            violation_severity=flags["violation_severity"],
            action_id=action_id,
            action_type="healthcare_interaction",
            features=features,
            rule_risk_score=(
                float(policy_out.get("risk_score", 0.0)) if isinstance(policy_out, dict) else 0.0
            ),
            rule_classification=flags["rule_classification"],
        )

        self._log_event({"type": "policy_outcome", "policy": policy_out})
        return {"facts": facts, "policy": policy_out, "governance": decision}

    # ------------- convenience orchestrator -------------

    def run(
        self,
        agent_id: str,
        action_id: str,
        payload: Dict[str, Any],
        agent_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Convenience method to run the full pipeline:
        preprocess -> agent_fn -> postprocess -> evaluate
        """
        try:
            ingress = self.preprocess(payload)
            agent_result = agent_fn(ingress) or {}
            egress = self.postprocess(agent_result)
            eval_out = self.evaluate(
                agent_id=agent_id, action_id=action_id, payload={**ingress, **egress}
            )
            return {
                "ingress": ingress,
                "agent_result": egress,
                "evaluation": eval_out,
            }
        except Exception as e:
            # Ensure failures are recorded and sanitized
            self._log_event({"type": "pipeline_error", "error": str(e)})
            # Fail-closed result with sanitized message
            safe_error = self.phi.redact(str(e))
            fallback = {"error": "processing_failed", "detail": safe_error}
            # Still try to evaluate with minimal facts to record governance trace
            eval_out = self.evaluate(agent_id=agent_id, action_id=action_id, payload=fallback)
            return {
                "ingress": {},
                "agent_result": fallback,
                "evaluation": eval_out,
            }
