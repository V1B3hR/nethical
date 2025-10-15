"""Processor for Cyber Security Attacks dataset from Kaggle."""
from __future__ import annotations

import ipaddress
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class CyberSecurityAttacksProcessor(BaseDatasetProcessor):
    """Process the Cyber Security Attacks dataset for the Nethical system.

    Dataset: https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks

    This implementation aligns to the Nethical standard record contract:
      - Uses BaseDatasetProcessor.make_record for normalization and validation
      - Populates all STANDARD_FEATURES
      - Adds useful metadata and a deterministic group_id to reduce leakage risk
      - Robustly parses common cyber-security fields across heterogeneous column names
    """

    # Common field aliases found in cyber datasets
    ATTACK_TYPE_KEYS: Tuple[str, ...] = (
        "attack_type",
        "Attack Type",
        "type",
        "Type",
        "category",
        "Category",
    )
    LABEL_KEYS: Tuple[str, ...] = (
        "label",
        "Label",
        "class",
        "Class",
        "target",
        "Target",
    )
    SCORE_KEYS: Tuple[str, ...] = (
        "anomaly_score",
        "Anomaly Score",
        "Anomaly Scores",
        "score",
        "Score",
        "threat_score",
        "Threat Score",
        "risk_score",
        "Risk Score",
    )
    SEVERITY_KEYS: Tuple[str, ...] = (
        "severity",
        "Severity",
        "Severity Level",
        "priority",
        "Priority",
        "level",
        "Level",
        "alert_severity",
        "Alert Severity",
    )
    COUNT_KEYS: Tuple[str, ...] = (
        "packet_count",
        "Packet Count",
        "count",
        "Count",
        "frequency",
        "Frequency",
        "packets",
        "Packets",
        "events",
        "Events",
    )
    BYTES_KEYS: Tuple[str, ...] = (
        "bytes",
        "Bytes",
        "byte_count",
        "Byte Count",
        "tx_bytes",
        "rx_bytes",
    )
    DURATION_KEYS: Tuple[str, ...] = (
        "duration",
        "Duration",
        "flow_duration",
        "Flow Duration",
        "time_taken",
        "Time Taken",
    )
    PROTOCOL_KEYS: Tuple[str, ...] = (
        "protocol",
        "Protocol",
        "proto",
        "Proto",
        "service",
        "Service",
        "application",
        "Application",
    )
    PORT_KEYS: Tuple[str, ...] = (
        "port",
        "Port",
        "dst_port",
        "Dst Port",
        "destination_port",
        "Destination Port",
        "server_port",
        "Server Port",
    )
    SRC_IP_KEYS: Tuple[str, ...] = (
        "src_ip",
        "Src IP",
        "source_ip",
        "Source IP",
        "ip_src",
        "IP Src",
        "source",
        "Source",
    )
    DST_IP_KEYS: Tuple[str, ...] = (
        "dst_ip",
        "Dst IP",
        "destination_ip",
        "Destination IP",
        "ip_dst",
        "IP Dst",
        "destination",
        "Destination",
    )
    TIMESTAMP_KEYS: Tuple[str, ...] = (
        "timestamp",
        "Timestamp",
        "time",
        "Time",
        "event_time",
        "Event Time",
        "date",
        "Date",
        "datetime",
        "DateTime",
        "ts",
        "TS",
    )

    # Heuristic caps for normalization before the Base normalizer
    # Note: Base will further normalize to [0,1] using DEFAULT_FEATURE_RANGES.
    MAX_COUNT_FOR_FREQ = 1000.0
    MAX_BYTES_FOR_FREQ = 1_000_000.0  # 1 MB
    MAX_DURATION_FOR_FREQ = 3600.0  # 1 hour

    def __init__(self, output_dir: Path = Path("data/processed")):
        super().__init__("cyber_security_attacks", output_dir)
        self._ts_min: Optional[float] = None
        self._ts_max: Optional[float] = None

    # -----------------------------
    # Top-level pipeline
    # -----------------------------
    def process(self, input_path: Path) -> List[Dict[str, Any]]:
        """Process the dataset and return standardized records."""
        logger.info(f"Processing Cyber Security Attacks dataset from {input_path}")

        rows = self.load_csv(input_path)
        if not rows:
            logger.warning(f"No data found in {input_path}")
            return []

        # Pre-scan timestamps for recency normalization
        self._infer_timestamp_range(rows)

        records: List[Dict[str, Any]] = []
        for i, raw in enumerate(rows):
            try:
                row = self.preprocess_row(raw)
                if not self.validate_row(row):
                    continue
                rec = self.make_record(row, include_meta=True, normalize=True)
                if rec is None:
                    continue
                # Enrich metadata per record
                rec = self.postprocess_record(rec, row=row, idx=i)
                if rec is not None:
                    records.append(rec)
            except Exception as e:
                logger.warning(f"Error processing row {i}: {e}")
                continue

        # Deduplicate by stable key (src/dst/timestamp/label if present, else features hash)
        records = self.deduplicate(
            records,
            key_fn=lambda r: self._dedup_key(r),
        )

        logger.info(f"Processed {len(records)} records from Cyber Security Attacks")
        return records

    # -----------------------------
    # Row lifecycle hooks
    # -----------------------------
    def preprocess_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize keys and trim string values
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if k is None:
                continue
            key = str(k).strip()
            if isinstance(v, str):
                v2 = v.strip()
                # Normalize obvious NA tokens
                out[key] = "" if v2.lower() in {"na", "n/a", "none", "null"} else v2
            else:
                out[key] = v
        return out

    def validate_row(self, row: Dict[str, Any]) -> bool:
        # Keep rows that have at least a label-ish or any signal fields
        has_labelish = any(k in row for k in (*self.LABEL_KEYS, *self.ATTACK_TYPE_KEYS))
        has_signal = any(k in row for k in (*self.SCORE_KEYS, *self.SEVERITY_KEYS, *self.COUNT_KEYS, *self.PROTOCOL_KEYS))
        return has_labelish or has_signal

    def postprocess_record(self, record: Dict[str, Any], *, row: Optional[Dict[str, Any]] = None, idx: Optional[int] = None) -> Dict[str, Any]:
        # If called from base class without row/idx, just return the record
        if row is None or idx is None:
            return record
            
        # Attach helpful metadata and a deterministic group_id to support group-aware splits
        meta = record.setdefault("meta", {})
        meta.setdefault("dataset", self.dataset_name)

        src_ip = self._first(row, self.SRC_IP_KEYS)
        dst_ip = self._first(row, self.DST_IP_KEYS)
        protocol = self._first(row, self.PROTOCOL_KEYS)
        port = self._first(row, self.PORT_KEYS)
        attack_type = self._first(row, self.ATTACK_TYPE_KEYS)
        ts_str = self._first(row, self.TIMESTAMP_KEYS)

        meta.update(
            {
                "row_index": idx,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": protocol,
                "port": port,
                "attack_type": attack_type,
                "timestamp": self._normalize_ts_to_iso(ts_str),
            }
        )

        # group_id: prefer a flow-like group to reduce leakage across splits
        group = f"{src_ip}->{dst_ip}|{protocol or ''}|{port or ''}"
        if group.strip("|->") == "":
            group = f"row-{idx}"
        meta["group_id"] = group
        return record

    # -----------------------------
    # Feature extraction
    # -----------------------------
    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Map dataset fields to Nethical standard features.

        Returns raw (pre-Base-normalization) values for:
          - violation_count in [0, 10]
          - severity_max in [0, 1]
          - recency_score in [0, 1]
          - frequency_score in [0, 1]
          - context_risk in [0, 1]
        """
        # violation_count: derive from attack indicators, anomaly/threat scores, and counts
        violation_count = 0.0
        # Attack type presence is a strong indicator
        if self._first(row, self.ATTACK_TYPE_KEYS):
            violation_count += 3.0

        # Label text indicating attack adds weight
        lbl_txt = self._label_text(row)
        if lbl_txt and self._looks_malicious(lbl_txt):
            violation_count += 4.0

        # Numerical or textual score scaled to 0..10
        score_val = self._score_value(row)
        if score_val is not None:
            # If it's likely 0..100 scale -> map to 0..10
            if score_val > 1.0:
                violation_count += min(10.0, (score_val / 100.0) * 10.0)
            else:
                violation_count += min(10.0, score_val * 10.0)

        # Light bump if elevated counts exist
        if self._numeric_from(row, self.COUNT_KEYS) or self._numeric_from(row, self.BYTES_KEYS):
            violation_count += 1.0

        # Clip to 0..10 expected by Base feature ranges
        violation_count = max(0.0, min(10.0, violation_count))

        # severity_max: parse from severity/priority textual or numeric
        severity_max = self._severity01(row)

        # recency_score: use dataset-wide min/max timestamp if available
        recency_score = self._recency01(row)

        # frequency_score: normalize from counts/bytes/duration heuristics into [0,1]
        frequency_score = self._frequency01(row)

        # context_risk: risk inferred from protocol/port
        context_risk = self._context_risk01(row)

        return {
            "violation_count": float(violation_count),
            "severity_max": float(severity_max),
            "recency_score": float(recency_score),
            "frequency_score": float(frequency_score),
            "context_risk": float(context_risk),
        }

    def extract_label(self, row: Dict[str, Any]) -> int:
        """Determine if this is a malicious/risky event: 1 attack/risky, 0 normal."""
        # Explicit numeric/boolean labels
        for k in self.LABEL_KEYS + self.ATTACK_TYPE_KEYS:
            if k in row:
                v = row[k]
                # Numeric
                try:
                    fv = float(v)
                    if fv == 1.0:
                        return 1
                    if fv == 0.0:
                        return 0
                except (TypeError, ValueError):
                    pass
                # Boolean-like
                sv = str(v).strip().lower()
                if sv in {"1", "true", "attack", "attacked", "intrusion"}:
                    return 1
                if sv in {"0", "false", "benign", "normal", "legitimate", "safe"}:
                    return 0

        # Textual heuristics on combined label-ish fields
        text = self._label_text(row)
        if text:
            if self._looks_malicious(text):
                return 1
            if any(tok in text for tok in ("benign", "normal", "legit", "safe")):
                return 0

        # Conservative default
        return 1

    # -----------------------------
    # Helpers
    # -----------------------------
    def _first(self, row: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return str(row[k])
        return None

    def _numeric_from(self, row: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
        for k in keys:
            if k in row:
                try:
                    return float(row[k])
                except (TypeError, ValueError):
                    # Try to extract first number from a string like "123 packets"
                    m = re.search(r"-?\d+(?:\.\d+)?", str(row[k]))
                    if m:
                        try:
                            return float(m.group(0))
                        except ValueError:
                            pass
        return None

    # Severity mapping -> [0,1]
    def _severity01(self, row: Dict[str, Any]) -> float:
        for k in self.SEVERITY_KEYS:
            if k in row:
                v = row[k]
                if isinstance(v, (int, float)):
                    # Assume 0..10 or 1..5 like scales
                    val = float(v)
                    # Try to guess scale
                    if val <= 1.0:
                        return max(0.0, min(1.0, val))
                    if val <= 5.0:
                        return max(0.0, min(1.0, val / 5.0))
                    return max(0.0, min(1.0, val / 10.0))
                s = str(v).strip().lower()
                if any(tok in s for tok in ("critical", "severe", "catastrophic", "urgent")):
                    return 0.95
                if "high" in s:
                    return 0.8
                if "medium" in s or "moderate" in s:
                    return 0.55
                if "low" in s or "info" in s or "informational" in s:
                    return 0.25
        return 0.0

    # Frequency score -> [0,1] via counts/bytes/duration
    def _frequency01(self, row: Dict[str, Any]) -> float:
        count = self._numeric_from(row, self.COUNT_KEYS)
        if count is not None:
            return max(0.0, min(1.0, count / self.MAX_COUNT_FOR_FREQ))

        by = self._numeric_from(row, self.BYTES_KEYS)
        if by is not None:
            return max(0.0, min(1.0, by / self.MAX_BYTES_FOR_FREQ))

        dur = self._numeric_from(row, self.DURATION_KEYS)
        if dur is not None:
            return max(0.0, min(1.0, dur / self.MAX_DURATION_FOR_FREQ))

        return 0.0

    # Context risk -> [0,1] based on protocol/port
    def _context_risk01(self, row: Dict[str, Any]) -> float:
        risk = 0.0

        proto = (self._first(row, self.PROTOCOL_KEYS) or "").strip().lower()
        if proto:
            proto_risk_map = {
                "telnet": 0.9,
                "rdp": 0.85,
                "smb": 0.85,
                "ftp": 0.75,
                "ssh": 0.6,
                "smtp": 0.55,
                "dns": 0.5,
                "http": 0.5,
                "icmp": 0.45,
                "udp": 0.4,
                "tcp": 0.4,
                "https": 0.35,
                "quic": 0.35,
            }
            # Use the highest matching token
            for key, val in proto_risk_map.items():
                if key in proto:
                    risk = max(risk, val)

        port_val = self._numeric_from(row, self.PORT_KEYS)
        if port_val is not None:
            port = int(max(0, min(65535, int(port_val))))
            # Some common high-risk service ports
            port_risk_map = {
                23: 0.9,   # telnet
                3389: 0.88,  # RDP
                445: 0.85,  # SMB
                21: 0.75,  # FTP
                22: 0.6,   # SSH
                25: 0.55,  # SMTP
                53: 0.5,   # DNS
                80: 0.5,   # HTTP
                1433: 0.8,  # MSSQL
                1521: 0.8,  # Oracle
                3306: 0.7,  # MySQL
                6379: 0.75,  # Redis
            }
            risk = max(risk, port_risk_map.get(port, risk))

            # Non-privileged ephemeral ports lower the risk slightly
            if 1024 <= port <= 65535:
                risk = max(0.1, risk * 0.9)

        return max(0.0, min(1.0, risk))

    # Recency -> [0,1] using dataset min/max timestamp
    def _recency01(self, row: Dict[str, Any]) -> float:
        ts = self._parse_ts(self._first(row, self.TIMESTAMP_KEYS))
        if ts is None or self._ts_min is None or self._ts_max is None or self._ts_max == self._ts_min:
            return 0.5  # Unknown -> neutral
        return max(0.0, min(1.0, (ts - self._ts_min) / (self._ts_max - self._ts_min)))

    # Score numeric value if present
    def _score_value(self, row: Dict[str, Any]) -> Optional[float]:
        return self._numeric_from(row, self.SCORE_KEYS)

    def _label_text(self, row: Dict[str, Any]) -> Optional[str]:
        texts: List[str] = []
        for k in (*self.LABEL_KEYS, *self.ATTACK_TYPE_KEYS):
            if k in row and row[k] not in (None, ""):
                texts.append(str(row[k]).strip().lower())
        return " ".join(texts) if texts else None

    def _looks_malicious(self, text: str) -> bool:
        indicators = (
            "malicious",
            "attack",
            "intrusion",
            "anomaly",
            "threat",
            "dos",
            "ddos",
            "bruteforce",
            "brute force",
            "sql",
            "xss",
            "exploit",
            "ransom",
            "phish",
            "botnet",
            "portscan",
            "scan",
            "worm",
            "trojan",
            "malware",
        )
        return any(tok in text for tok in indicators)

    # -----------------------------
    # Timestamp handling
    # -----------------------------
    def _infer_timestamp_range(self, rows: List[Dict[str, Any]]) -> None:
        tmins: List[float] = []
        tmaxs: List[float] = []
        for r in rows:
            ts = self._parse_ts(self._first(r, self.TIMESTAMP_KEYS))
            if ts is not None:
                tmins.append(ts)
                tmaxs.append(ts)
        if tmins and tmaxs:
            self._ts_min = min(tmins)
            self._ts_max = max(tmaxs)
        else:
            self._ts_min = None
            self._ts_max = None

    def _parse_ts(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = s.strip()
        # Try epoch seconds or milliseconds
        try:
            val = float(s)
            # Heuristic: treat > 10^12 as ms
            if val > 1e12:
                val = val / 1000.0
            if val > 1e10:
                # still too large to be seconds realistically
                return None
            return val
        except (ValueError, TypeError):
            pass
        # Try several common datetime formats
        fmts = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
        # Last resort: ISO parser fallback
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    def _normalize_ts_to_iso(self, s: Optional[str]) -> Optional[str]:
        ts = self._parse_ts(s)
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -----------------------------
    # Deduplication key
    # -----------------------------
    def _dedup_key(self, rec: Dict[str, Any]) -> str:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        src = meta.get("src_ip") or ""
        dst = meta.get("dst_ip") or ""
        pr = meta.get("protocol") or ""
        pt = meta.get("port") or ""
        ts = meta.get("timestamp") or ""
        lbl = rec.get("label", 0)
        if any((src, dst, pr, pt, ts)):
            return f"{src}|{dst}|{pr}|{pt}|{ts}|{lbl}"
        # fallback: features+label hash via Base.deduplicate's default would also work
        # but we keep a stable readable key here
        feats = rec.get("features", {})
        return f"{feats}|{lbl}"

    # -----------------------------
    # Utility
    # -----------------------------
    @staticmethod
    def _ip_or_none(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        try:
            # Validate IPv4/IPv6
            ipaddress.ip_address(value)
            return value
        except ValueError:
            return None
