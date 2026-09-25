# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Protocol Buffer definitions for Nethical.

This package contains the gRPC protocol definitions for
low-latency inter-service communication.

The proto files can be compiled using:
    protoc --python_out=. --grpc_python_out=. governance.proto

For now, we provide Python dataclasses that mirror the proto messages
until the proto files are compiled.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Violation:
    """Violation message (mirrors proto)."""
    
    id: str
    type: str
    severity: str
    description: str
    law_reference: Optional[str] = None
    evidence: dict = field(default_factory=dict)


@dataclass
class Explanation:
    """Explanation message (mirrors proto)."""
    
    summary: str
    risk_factors: list[str] = field(default_factory=list)
    decision_rationale: str = ""
    laws_applied: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class CellularTelemetry:
    """Cellular radio telemetry message (mirrors proto)."""
    cell_id: str = "CELL_01"
    generation: str = "5G_Sub6"
    frequency_mhz: float = 3500.0
    bandwidth_mhz: float = 100.0
    rsrp_dbm: float = -85.0
    rsrq_db: float = -10.0
    sinr_db: float = 15.0
    cqi: int = 12
    latency_ms: float = 8.0
    jitter_ms: float = 1.5
    packet_loss_percent: float = 0.0
    handover_count: int = 0
    connected: bool = True
    signal_grade: str = "EXCELLENT"


@dataclass
class EmfRadiationTelemetry:
    """Biological EMF radiation telemetry message (mirrors proto)."""
    emitter_id: str = "default_emitter"
    frequency_hz: float = 2.4e9
    tx_power_dbm: float = 20.0
    estimated_sar_w_kg: Optional[float] = None
    power_density_w_m2: Optional[float] = None
    human_distance_meters: Optional[float] = None
    pulse_modulation_hz: Optional[float] = None
    exposure_zone: str = "general_public"
    is_medical_device: bool = False


@dataclass
class NetworkFlowTelemetry:
    """IP network flow intrusion telemetry message (mirrors proto)."""
    flow_id: str = ""
    source_ip: str = "127.0.0.1"
    destination_ip: str = "127.0.0.1"
    destination_port: int = 443
    protocol: str = "TCP"
    flow_duration_ms: float = 100.0
    total_fwd_packets: int = 10
    total_bwd_packets: int = 10
    flow_bytes_per_sec: float = 1000.0
    flow_packets_per_sec: float = 20.0
    syn_flag_count: int = 1
    ack_flag_count: int = 1
    dst_port_entropy: float = 0.5
    is_anomaly: bool = False
    source_dataset: str = "CICIDS2017"


@dataclass
class OSTelemetry:
    """Host operating system and sandbox telemetry (mirrors proto)."""
    platform_name: str = "Windows"
    os_version: str = ""
    cpu_percent: float = 0.0
    total_ram_mb: float = 16384.0
    available_ram_mb: float = 8192.0
    used_ram_percent: float = 50.0
    somatic_condition: str = "HOMEOSTASIS_OPTIMAL"
    sandbox_tier: str = "STANDARD"
    privileges_dropped: bool = False


@dataclass
class EvaluateRequest:
    """Evaluate request message (mirrors proto)."""
    
    agent_id: str
    action: str
    action_type: str = "query"
    context: dict = field(default_factory=dict)
    stated_intent: Optional[str] = None
    priority: str = "normal"
    require_explanation: bool = False
    request_id: Optional[str] = None
    cellular_telemetry: Optional[CellularTelemetry] = None
    emf_telemetry: Optional[EmfRadiationTelemetry] = None
    network_flow_telemetry: Optional[NetworkFlowTelemetry] = None
    os_telemetry: Optional[OSTelemetry] = None


@dataclass
class EvaluateResponse:
    """Evaluate response message (mirrors proto)."""
    
    decision: str
    decision_id: str
    risk_score: float = 0.0
    confidence: float = 1.0
    latency_ms: int = 0
    violations: list[Violation] = field(default_factory=list)
    reason: str = ""
    explanation: Optional[Explanation] = None
    audit_id: Optional[str] = None
    cache_hit: bool = False
    fundamental_laws_checked: list[int] = field(default_factory=list)
    timestamp: str = ""
    primary_mitigation: Optional[str] = None


@dataclass
class Decision:
    """Decision record message (mirrors proto)."""
    
    decision_id: str
    decision: str
    agent_id: str
    action_summary: str
    action_type: str
    risk_score: float
    confidence: float
    reasoning: str
    violations: list[Violation] = field(default_factory=list)
    fundamental_laws: list[int] = field(default_factory=list)
    timestamp: str = ""
    latency_ms: int = 0
    audit_id: Optional[str] = None


@dataclass
class Policy:
    """Policy message (mirrors proto)."""
    
    policy_id: str
    name: str
    description: str
    version: str
    status: str
    scope: str
    fundamental_laws: list[int] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class BatchEvaluateRequest:
    """Batch evaluate request message (mirrors proto)."""

    requests: list[EvaluateRequest] = field(default_factory=list)
    parallel: bool = False
    fail_fast: bool = False


@dataclass
class DecisionStreamRequest:
    """Decision stream configuration message (mirrors proto)."""

    agent_id: Optional[str] = None
    decision_types: list[str] = field(default_factory=list)
    min_risk_score: Optional[float] = None
    history_seconds: Optional[int] = None


@dataclass
class GetDecisionRequest:
    """Get decision request message (mirrors proto)."""

    decision_id: str


@dataclass
class ListPoliciesRequest:
    """List policies request message (mirrors proto)."""

    status: Optional[str] = None
    scope: Optional[str] = None
    page: int = 1
    page_size: int = 20


@dataclass
class ListPoliciesResponse:
    """List policies response message (mirrors proto)."""

    policies: list[Policy] = field(default_factory=list)
    total_count: int = 0
    has_next: bool = False


@dataclass
class HealthCheckRequest:
    """Health check request message (mirrors proto)."""

    pass


@dataclass
class HealthCheckResponse:
    """Health check response message (mirrors proto)."""

    status: str = "SERVING"
    version: str = "1.0.0"
    uptime_seconds: int = 0
    timestamp: str = ""


__all__ = [
    "Violation",
    "Explanation",
    "EvaluateRequest",
    "EvaluateResponse",
    "Decision",
    "Policy",
    "BatchEvaluateRequest",
    "DecisionStreamRequest",
    "GetDecisionRequest",
    "ListPoliciesRequest",
    "ListPoliciesResponse",
    "HealthCheckRequest",
    "HealthCheckResponse",
    "CellularTelemetry",
    "EmfRadiationTelemetry",
    "NetworkFlowTelemetry",
    "OSTelemetry",
]
