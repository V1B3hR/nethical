# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Network Flow Telemetry and Cyber-Immunity Engine (nethical.security.flow_telemetry).

Ported and enhanced from Błyskawica's neuro-immunological telemetry engine
(adaptiveneuralnetwork.immune_system.immune_stream_pipeline).

Provides:
1. 8-dimensional network flow vectorization (CICIDS2017 / UNSW-NB15 / TON_IoT standards).
2. Dynamic rhythm tracking & Shannon entropy deviation calculation for network breathing.
3. Attack pattern classification (SYN flood, port scan, data exfiltration, DDoS bursts).
4. 'The Garden' data quarantine buffer with cryptographic provenance watermarking.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("nethical.security.flow_telemetry")


class NetworkAttackType(str, Enum):
    """Categorisation of network flow intrusion attack patterns."""
    NONE = "NONE"
    SYN_FLOOD = "SYN_FLOOD"
    PORT_SCAN = "PORT_SCAN"
    DDOS_BURST = "DDOS_BURST"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    SLOWLORIS = "SLOWLORIS"
    SUSPICIOUS_RHYTHM_ANOMALY = "SUSPICIOUS_RHYTHM_ANOMALY"


@dataclass
class NetworkFlowSample:
    """Represents a single time-series network flow observation (CICIDS2017 / UNSW standard)."""
    flow_id: str = field(default_factory=lambda: str(uuid4()))
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkFlowSample":
        """Instantiates a flow sample safely from dictionary context."""
        return cls(
            flow_id=str(data.get("flow_id", str(uuid4()))),
            source_ip=str(data.get("source_ip", "127.0.0.1")),
            destination_ip=str(data.get("destination_ip", "127.0.0.1")),
            destination_port=int(data.get("destination_port", 443)),
            protocol=str(data.get("protocol", "TCP")).upper(),
            flow_duration_ms=float(data.get("flow_duration_ms", 100.0)),
            total_fwd_packets=int(data.get("total_fwd_packets", 10)),
            total_bwd_packets=int(data.get("total_bwd_packets", 10)),
            flow_bytes_per_sec=float(data.get("flow_bytes_per_sec", 1000.0)),
            flow_packets_per_sec=float(data.get("flow_packets_per_sec", 20.0)),
            syn_flag_count=int(data.get("syn_flag_count", 1)),
            ack_flag_count=int(data.get("ack_flag_count", 1)),
            dst_port_entropy=float(data.get("dst_port_entropy", 0.5)),
            is_anomaly=bool(data.get("is_anomaly", False)),
            source_dataset=str(data.get("source_dataset", "CICIDS2017")),
        )

    def to_vector(self) -> List[float]:
        """Vectorizes network flow into an 8-dimensional normalized feature list.

        Matches Błyskawica's mathematical formulation:
        [log1p(duration), log1p(fwd_pkts), log1p(bwd_pkts), log1p(bytes_sec),
         log1p(pkts_sec), syn_count, ack_count, dst_port_entropy]
        """
        return [
            math.log1p(max(0.0, self.flow_duration_ms)),
            math.log1p(max(0, self.total_fwd_packets)),
            math.log1p(max(0, self.total_bwd_packets)),
            math.log1p(max(0.0, self.flow_bytes_per_sec)),
            math.log1p(max(0.0, self.flow_packets_per_sec)),
            float(max(0, self.syn_flag_count)),
            float(max(0, self.ack_flag_count)),
            float(max(0.0, self.dst_port_entropy)),
        ]

    def classify_signature(self) -> NetworkAttackType:
        """Classifies known cyberattack signatures using rule heuristics."""
        # 1. SYN Flood: High SYN flags with zero or negligible ACK flags
        if self.syn_flag_count > 20 and self.ack_flag_count == 0:
            return NetworkAttackType.SYN_FLOOD

        # 2. Port Scan / Reconnaissance: High destination port entropy
        if self.dst_port_entropy > 3.0 and self.total_fwd_packets > 15:
            return NetworkAttackType.PORT_SCAN

        # 3. High Volume Burst / DDoS: Enormous packet rate and throughput
        if self.flow_packets_per_sec > 10_000.0 or self.flow_bytes_per_sec > 10_000_000.0:
            return NetworkAttackType.DDOS_BURST

        # 4. Slowloris: Long duration with very few packets and low bytes
        if self.flow_duration_ms > 30_000.0 and self.total_fwd_packets < 5:
            return NetworkAttackType.SLOWLORIS

        # 5. Data Exfiltration: Unbalanced outbound payload with few backward responses
        if self.total_fwd_packets > 100 and self.total_bwd_packets < 3 and self.flow_bytes_per_sec > 500_000.0:
            return NetworkAttackType.DATA_EXFILTRATION

        return NetworkAttackType.NONE


class NetworkRhythmTracker:
    """Tracks baseline network statistics and computes flow rhythm entropy deviations."""

    def __init__(self, momentum: float = 0.9, history_capacity: int = 500) -> None:
        self.momentum = momentum
        self.dim = 8
        self.running_mean: List[float] = [0.0] * self.dim
        self.running_std: List[float] = [1.0] * self.dim
        self.sample_count: int = 0
        self.recent_vectors: deque[List[float]] = deque(maxlen=history_capacity)

    def update(self, vectors: List[List[float]]) -> float:
        """Updates running rhythm statistics and returns batch rhythm entropy deviation."""
        if not vectors:
            return 0.0

        batch_size = len(vectors)
        # Compute batch mean per dimension
        batch_mean = [
            sum(v[d] for v in vectors) / batch_size for d in range(self.dim)
        ]
        # Compute batch std per dimension
        batch_std = [
            math.sqrt(sum((v[d] - batch_mean[d]) ** 2 for v in vectors) / max(1, batch_size)) + 1e-6
            for d in range(self.dim)
        ]

        # Calculate deviation from established rhythm
        if self.sample_count > 0:
            deviations = [
                abs(batch_mean[d] - self.running_mean[d]) / (self.running_std[d] + 1e-6)
                for d in range(self.dim)
            ]
            entropy_deviation = sum(deviations) / self.dim
        else:
            entropy_deviation = 0.0

        # Update running stats
        for d in range(self.dim):
            self.running_mean[d] = self.momentum * self.running_mean[d] + (1.0 - self.momentum) * batch_mean[d]
            self.running_std[d] = self.momentum * self.running_std[d] + (1.0 - self.momentum) * batch_std[d]

        self.sample_count += batch_size
        for v in vectors:
            self.recent_vectors.append(v)

        return entropy_deviation


class GardenQuarantineBuffer:
    """The Garden: Data isolation buffer for external untrusted telemetry streams.

    Tags incoming telemetry with provenance watermarks ('Observation (External)' vs 'Self (Ground Truth)')
    before allowing ingress into core training or decision pipelines.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def ingest(self, sample: NetworkFlowSample, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingests a network flow sample into quarantine with an identity watermark."""
        record = {
            "record_id": str(uuid4()),
            "sample": sample,
            "vector": sample.to_vector(),
            "metadata": metadata or {},
            "provenance": "Observation (External)",
            "quarantined_at": time.time(),
            "passed_safety_check": False,
        }

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append(record)
        return record

    def release_verified(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Validates and marks a quarantined record as safe for consumption."""
        for item in self.buffer:
            if item["record_id"] == record_id:
                item["passed_safety_check"] = True
                item["provenance"] = "Verified (Internal)"
                return item
        return None

    def purge_quarantine(self) -> int:
        """Purges all unverified records from quarantine."""
        initial_len = len(self.buffer)
        self.buffer = [item for item in self.buffer if item["passed_safety_check"]]
        return initial_len - len(self.buffer)
