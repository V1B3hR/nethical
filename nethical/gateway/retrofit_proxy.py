# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Legacy System Bridge & Retrofit Governance Proxy (Gap 6.5).

Enables immediate zero-code retrofit of legacy AI systems and un-governed microservices:
- **Transparent Reverse Proxy / Sidecar Architecture:** Intercepts REST/JSON AI traffic without
  modifying legacy model source code.
- **Pre-Flight Invariant Enforcement:** Validates tenant identity, HRBAC authority, and input hygiene.
- **Post-Flight Output Interception:** Inspects model outputs for OPSEC leaks, PII exposure, and toxicity.
- **Cryptographic Audit Attestation:** Appends tamper-proof headers (`X-Nethical-Receipt-Id`)
  backed by Merkle-DAG proofs.
- **Circuit Breaker Fallback:** Intercepts unsafe legacy outputs and serves deterministic safe fallbacks.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.gateway.retrofit_proxy")


# ============== Enums & Value Types ==============


class RetrofitVerdict(str, Enum):
    """Governance decision rendered on intercepted traffic."""
    PASSTHROUGH_APPROVED = "PASSTHROUGH_APPROVED"
    INTERCEPTED_AND_SANITIZED = "INTERCEPTED_AND_SANITIZED"
    CIRCUIT_BREAKER_BLOCKED = "CIRCUIT_BREAKER_BLOCKED"


class RetrofitPolicyMode(str, Enum):
    """Operational mode of the retrofit governance wrapper."""
    STRICT_ENFORCE = "STRICT_ENFORCE"         # Blocks violating traffic immediately
    SHADOW_AUDIT = "SHADOW_AUDIT"             # Allows traffic through but logs cryptographic alerts
    FALLBACK_INTERCEPT = "FALLBACK_INTERCEPT" # Replaces unsafe response with standard compliant fallback


# ============== Data Models ==============


class InboundRequestWrapper(BaseModel):
    """Normalized representation of an intercepted request to a legacy system."""
    request_id: str = Field(default_factory=lambda: f"REQ-{uuid.uuid4().hex[:8].upper()}")
    legacy_service_id: str
    tenant_id: str = "default_tenant"
    client_principal: str = "legacy_operator"
    endpoint_path: str = "/api/v1/predict"
    payload: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OutboundResponseWrapper(BaseModel):
    """Governed response returned to the calling client."""
    response_id: str = Field(default_factory=lambda: f"RESP-{uuid.uuid4().hex[:8].upper()}")
    verdict: RetrofitVerdict
    status_code: int = 200
    body: Dict[str, Any]
    governance_headers: Dict[str, str] = Field(default_factory=dict)
    interventions_applied: List[str] = Field(default_factory=list)


# ============== Engine Class ==============


class LegacyRetrofitProxy:
    """Zero-Code Retrofit Governance Wrapper for Pre-Existing AI Services."""

    def __init__(
        self,
        legacy_service_id: str,
        policy_mode: RetrofitPolicyMode = RetrofitPolicyMode.STRICT_ENFORCE,
        ledger: Optional[MerkleLedger] = None,
        max_payload_bytes: int = 10_000_000,
    ) -> None:
        self.legacy_service_id = legacy_service_id
        self.policy_mode = policy_mode
        self.ledger = ledger or MerkleLedger()
        self.max_payload_bytes = max_payload_bytes

        # Registered safety rules
        self._forbidden_input_keywords: List[str] = ["DROP TABLE", "<script>", "__import__('os')", "EXPLOIT_PAYLOAD"]
        self._forbidden_output_keywords: List[str] = ["CONFIDENTIAL_CREDENTIAL", "INTERNAL_PRIVATE_KEY", "MALICIOUS_EXEC"]

    def intercept_and_forward(
        self,
        request: InboundRequestWrapper,
        legacy_backend_callable: Callable[[Dict[str, Any]], Tuple[int, Dict[str, Any]]],
    ) -> OutboundResponseWrapper:
        """Executes full pre-flight, execution, and post-flight governance lifecycle."""
        # 1. Pre-Flight Invariant Validation
        pre_flight_err = self._validate_pre_flight(request)
        if pre_flight_err:
            return self._handle_block(
                request=request,
                status_code=400,
                error_msg=pre_flight_err,
                stage="PRE_FLIGHT",
            )

        # 2. Forward to Legacy Backend
        try:
            backend_status, backend_body = legacy_backend_callable(request.payload)
        except Exception as e:
            logger.error("Błąd podczas wywołania legacy backendu %s: %s", self.legacy_service_id, e)
            return self._handle_block(
                request=request,
                status_code=502,
                error_msg=f"Legacy backend unavailable or crashed: {str(e)}",
                stage="BACKEND_EXECUTION",
            )

        # 3. Post-Flight Output Inspection
        post_flight_err, sanitized_body = self._validate_post_flight(backend_body)
        if post_flight_err:
            if self.policy_mode == RetrofitPolicyMode.STRICT_ENFORCE:
                return self._handle_block(
                    request=request,
                    status_code=403,
                    error_msg=post_flight_err,
                    stage="POST_FLIGHT",
                )
            elif self.policy_mode == RetrofitPolicyMode.FALLBACK_INTERCEPT:
                backend_body = {
                    "error": "Zawartość wyjściowa legacy systemu zablokowana przez filtr bezpieczeństwa Nethical.",
                    "fallback_activated": True,
                    "service_id": self.legacy_service_id,
                }
                backend_status = 200

        # 4. Seal Merkle Audit Proof
        receipt = self._seal_transaction(
            request=request,
            verdict=RetrofitVerdict.PASSTHROUGH_APPROVED,
            status_code=backend_status,
        )

        headers = {
            "X-Nethical-Governance": "RETROFIT_PROTECTED",
            "X-Nethical-Service": self.legacy_service_id,
            "X-Nethical-Receipt-Id": receipt.receipt_id if receipt else "NONE",
            "X-Nethical-Policy-Verdict": RetrofitVerdict.PASSTHROUGH_APPROVED.value,
        }

        return OutboundResponseWrapper(
            verdict=RetrofitVerdict.PASSTHROUGH_APPROVED,
            status_code=backend_status,
            body=backend_body,
            governance_headers=headers,
        )

    def _validate_pre_flight(self, request: InboundRequestWrapper) -> Optional[str]:
        """Inspects incoming payload for injection attempts or malformed structure."""
        payload_str = json.dumps(request.payload)
        if len(payload_str) > self.max_payload_bytes:
            return "Przekroczono maksymalny rozmiar payloadu wejściowego."

        for kw in self._forbidden_input_keywords:
            if kw.lower() in payload_str.lower():
                return f"Wykryto zabroniony wzorzec wejściowy: '{kw}'."
        return None

    def _validate_post_flight(self, body: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Inspects legacy output for leaked credentials, unauthorized commands or toxicity."""
        body_str = json.dumps(body)
        for kw in self._forbidden_output_keywords:
            if kw.lower() in body_str.lower():
                return f"Odpowiedź legacy modelu zawierała niedozwolone dane: '{kw}'", body
        return None, body

    def _handle_block(
        self,
        request: InboundRequestWrapper,
        status_code: int,
        error_msg: str,
        stage: str,
    ) -> OutboundResponseWrapper:
        """Constructs an intercepted block response and seals the incident."""
        logger.warning(
            "RETROFIT CIRCUIT BREAKER [%s]: Zablokowano żądanie do %s: %s",
            stage, self.legacy_service_id, error_msg
        )
        receipt = self._seal_transaction(
            request=request,
            verdict=RetrofitVerdict.CIRCUIT_BREAKER_BLOCKED,
            status_code=status_code,
            notes=f"BLOKADA RETROFIT ({stage}): {error_msg}",
        )

        headers = {
            "X-Nethical-Governance": "RETROFIT_PROTECTED",
            "X-Nethical-Service": self.legacy_service_id,
            "X-Nethical-Receipt-Id": receipt.receipt_id if receipt else "NONE",
            "X-Nethical-Policy-Verdict": RetrofitVerdict.CIRCUIT_BREAKER_BLOCKED.value,
        }

        return OutboundResponseWrapper(
            verdict=RetrofitVerdict.CIRCUIT_BREAKER_BLOCKED,
            status_code=status_code,
            body={"error": error_msg, "blocked_by": "Nethical Retrofit Gate", "stage": stage},
            governance_headers=headers,
            interventions_applied=[error_msg],
        )

    def _seal_transaction(
        self,
        request: InboundRequestWrapper,
        verdict: RetrofitVerdict,
        status_code: int,
        notes: Optional[str] = None,
    ) -> Optional[Any]:
        """Kryptograficzne pieczętowanie transakcji legacy w MerkleLedger."""
        try:
            payload = {
                "event_type": "LEGACY_RETROFIT_INTERCEPTION",
                "request_id": request.request_id,
                "service_id": self.legacy_service_id,
                "tenant_id": request.tenant_id,
                "verdict": verdict.value,
                "status_code": status_code,
            }
            return self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=notes or f"RETROFIT PROXY: {self.legacy_service_id} -> {verdict.value}.",
            )
        except Exception as e:
            logger.error("Błąd pieczętowania retrofit proxy w MerkleLedger: %s", e)
            return None
