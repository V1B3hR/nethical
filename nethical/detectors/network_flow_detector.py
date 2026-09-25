# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Network Flow Intrusion and Anomaly Detector (nethical.detectors.network_flow_detector).

Monitors and audits IP network flow telemetry in real-time, detecting cyberattacks,
rhythm anomalies, and volumetric bursts before they compromise ethical AI operations.

Grounded in Błyskawica's neuro-immunological engine:
- Multi-dimensional vector entropy tracking (CICIDS2017 / UNSW-NB15).
- Heuristic signature matching for SYN flood, port scans, exfiltration, DDoS.
- Integration with 'The Garden' quarantine buffer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from nethical.security.flow_telemetry import (
    GardenQuarantineBuffer,
    NetworkAttackType,
    NetworkFlowSample,
    NetworkRhythmTracker,
)

logger = logging.getLogger("nethical.detectors.network_flow")


class FlowMitigationAction(str, Enum):
    """Mitigation actions recommended by the network flow detector."""
    NONE = "NONE"
    DROP_FLOW = "DROP_FLOW"
    RATE_LIMIT_IP = "RATE_LIMIT_IP"
    QUARANTINE_SOURCE_IP = "QUARANTINE_SOURCE_IP"
    RESET_TCP_CONNECTION = "RESET_TCP_CONNECTION"
    ALERT_SOC = "ALERT_SOC"


@dataclass
class FlowViolation:
    """Detailed violation description for a suspicious or malicious flow."""
    violation_id: str
    flow_id: str
    attack_type: NetworkAttackType
    severity: str
    description: str
    source_ip: str
    destination_ip: str
    destination_port: int
    mitigation: FlowMitigationAction
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "flow_id": self.flow_id,
            "attack_type": self.attack_type.value,
            "severity": self.severity,
            "description": self.description,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "destination_port": self.destination_port,
            "mitigation": self.mitigation.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NetworkFlowEvaluationResult:
    """Comprehensive evaluation outcome of network flow analysis."""
    is_safe: bool
    decision: str  # ALLOW, RESTRICT, BLOCK
    violations: List[FlowViolation]
    primary_mitigation: FlowMitigationAction
    network_entropy: float
    total_flows_evaluated: int
    latency_microseconds: float
    details: str


class NetworkFlowDetector:
    """Evaluates time-series network flow streams for cyber threats and rhythm deviations."""

    def __init__(
        self,
        entropy_alert_threshold: float = 1.25,
        enable_quarantine: bool = True,
        quarantine_capacity: int = 1000,
    ) -> None:
        self.name = "NetworkFlowDetector"
        self.entropy_alert_threshold = entropy_alert_threshold
        self.tracker = NetworkRhythmTracker()
        self.quarantine = GardenQuarantineBuffer(capacity=quarantine_capacity) if enable_quarantine else None
        self._total_checks = 0
        self._violations_count = 0

    def evaluate_flows(self, flows: List[NetworkFlowSample]) -> NetworkFlowEvaluationResult:
        """Evaluates a batch of network flow samples with microsecond latency."""
        start_time = time.perf_counter()
        violations: List[FlowViolation] = []

        if not flows:
            latency_us = (time.perf_counter() - start_time) * 1_000_000.0
            return NetworkFlowEvaluationResult(
                is_safe=True,
                decision="ALLOW",
                violations=[],
                primary_mitigation=FlowMitigationAction.NONE,
                network_entropy=0.0,
                total_flows_evaluated=0,
                latency_microseconds=round(latency_us, 2),
                details="Empty flow batch evaluated.",
            )

        # 1. Ingest into Quarantine Buffer
        if self.quarantine:
            for flow in flows:
                self.quarantine.ingest(flow)

        # 2. Extract vectors and calculate rhythm entropy deviation
        vectors = [f.to_vector() for f in flows]
        entropy_dev = self.tracker.update(vectors)

        # 3. Check individual flow attack signatures
        for flow in flows:
            attack_type = flow.classify_signature()
            if attack_type != NetworkAttackType.NONE:
                severity = "CRITICAL" if attack_type in (NetworkAttackType.SYN_FLOOD, NetworkAttackType.DDOS_BURST) else "HIGH"
                mitigation = FlowMitigationAction.DROP_FLOW if severity == "CRITICAL" else FlowMitigationAction.QUARANTINE_SOURCE_IP

                violations.append(
                    FlowViolation(
                        violation_id=str(uuid4()),
                        flow_id=flow.flow_id,
                        attack_type=attack_type,
                        severity=severity,
                        description=(
                            f"Detected malicious network pattern '{attack_type.value}' from {flow.source_ip} "
                            f"to {flow.destination_ip}:{flow.destination_port}."
                        ),
                        source_ip=flow.source_ip,
                        destination_ip=flow.destination_ip,
                        destination_port=flow.destination_port,
                        mitigation=mitigation,
                        confidence=0.95,
                    )
                )

        # 4. Check global rhythm entropy deviation
        if entropy_dev >= self.entropy_alert_threshold:
            violations.append(
                FlowViolation(
                    violation_id=str(uuid4()),
                    flow_id=flows[0].flow_id,
                    attack_type=NetworkAttackType.SUSPICIOUS_RHYTHM_ANOMALY,
                    severity="HIGH",
                    description=(
                        f"Network rhythm entropy ({entropy_dev:.2f}) exceeds threshold "
                        f"({self.entropy_alert_threshold:.2f}). Flow rhythm deviation detected."
                    ),
                    source_ip="MULTIPLE",
                    destination_ip="MULTIPLE",
                    destination_port=0,
                    mitigation=FlowMitigationAction.ALERT_SOC,
                    confidence=0.85,
                )
            )

        # 5. Synthesize Decision
        self._total_checks += len(flows)
        self._violations_count += len(violations)

        is_safe = (len(violations) == 0)
        has_critical = any(v.severity == "CRITICAL" for v in violations)

        if has_critical:
            decision = "BLOCK"
            primary_mitigation = FlowMitigationAction.DROP_FLOW
        elif len(violations) > 0:
            decision = "RESTRICT"
            primary_mitigation = violations[0].mitigation
        else:
            decision = "ALLOW"
            primary_mitigation = FlowMitigationAction.NONE

        latency_us = (time.perf_counter() - start_time) * 1_000_000.0

        details = (
            f"Evaluated {len(flows)} flow(s). Entropy deviation: {entropy_dev:.3f}. "
            f"Decision: {decision} with {len(violations)} violation(s)."
        )

        return NetworkFlowEvaluationResult(
            is_safe=is_safe,
            decision=decision,
            violations=violations,
            primary_mitigation=primary_mitigation,
            network_entropy=round(entropy_dev, 4),
            total_flows_evaluated=len(flows),
            latency_microseconds=round(latency_us, 2),
            details=details,
        )

    def analyze(self, context: Dict[str, Any], agent_id: str = "default") -> NetworkFlowEvaluationResult:
        """Adapter method allowing single-flow or list-based dictionary evaluation."""
        if "flows" in context and isinstance(context["flows"], list):
            sample_list = [NetworkFlowSample.from_dict(item) for item in context["flows"]]
        else:
            sample_list = [NetworkFlowSample.from_dict(context)]
        return self.evaluate_flows(sample_list)

    async def detect_violations(self, action: Any) -> List[Dict[str, Any]]:
        """Async detector interface compliant with BaseDetector orchestration."""
        context = getattr(action, "context", {}) or {}
        agent_id = getattr(action, "agent_id", "default_agent")
        result = self.analyze(context, agent_id=agent_id)
        return [v.to_dict() for v in result.violations]
