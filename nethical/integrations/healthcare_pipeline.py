from __future__ import annotations
from typing import Dict, Any, Optional
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

from nethical.detectors.healthcare.phi_detector import (
    PHIDetector,
    detect_and_redact_payload,
)
from nethical.detectors.healthcare.clinical_risk_detectors import (
    extract_clinical_signals,
)


class HealthcareGuardrails:
    def __init__(
        self,
        gov: IntegratedGovernance,
        region: Region,
        policy_path: str = "policies/healthcare/core.yaml",
        attestation: Optional[AttestationProvider] = None,
        crypto_signal: Optional[CryptoSignalProvider] = None,
        comms: Optional[CommsPolicy] = None,
        offline_store: Optional[OfflineStore] = None,
    ):
        self.gov = gov
        self.region = region
        self.policy = PolicyEngine.load(policy_path, region)
        self.phi = PHIDetector()
        self.attestation = attestation or NoopAttestation()
        self.crypto_signal = crypto_signal  # None by default in healthcare
        self.comms = comms or NoopCommsPolicy()
        self.offline = offline_store or TamperEvidentOfflineStore()

    def preprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Minimum necessary + PHI redaction at ingress
        redacted = detect_and_redact_payload(payload)
        # Attestation gate (hook)
        attn = self.attestation.attest_runtime()
        self.offline.append_event({"type": "attest_runtime", "ok": attn.ok})
        return redacted

    def postprocess(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # PHI redaction at egress and in logs
        if "agent_output" in result and isinstance(result["agent_output"], str):
            result["agent_output"] = self.phi.redact(result["agent_output"])
        self.offline.append_event(
            {"type": "egress", "len": len(result.get("agent_output", ""))}
        )
        return result

    def evaluate(
        self, agent_id: str, action_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        facts = {**payload, **extract_clinical_signals(payload)}
        policy_out = self.policy.evaluate(facts)
        decision = self.gov.process_action(
            agent_id=agent_id,
            action=payload.get("agent_output") or payload.get("user_input") or "",
            cohort=f"healthcare_{self.region.value}",
            violation_detected=False,
            violation_type="policy",
            violation_severity="medium",
            action_id=action_id,
            action_type="healthcare_interaction",
            features={"ml_score": 0.0},
            rule_risk_score=0.0,
            rule_classification="warn",
        )
        self.offline.append_event({"type": "policy_outcome", "policy": policy_out})
        return {"facts": facts, "policy": policy_out, "governance": decision}
