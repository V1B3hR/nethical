"""Generic processor for security datasets aligned with the Nethical standard.

This processor:
- Produces STANDARD_FEATURES expected by the Nethical system
- Uses BaseDatasetProcessor.make_record for normalization, validation, and metadata
- Applies robust heuristics to map arbitrary security dataset fields to standard features
"""

from __future__ import annotations

import logging
import re
import ipaddress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class GenericSecurityProcessor(BaseDatasetProcessor):
    """Generic processor for security datasets with flexible field mapping and Nethical alignment."""

    # Heuristic field hints (lowercased keys are used)
    _SEVERITY_KEYS = ("severity", "priority", "risk", "level", "grade", "impact")
    _COUNT_KEYS = ("count", "frequency", "freq", "number", "total", "hits", "events", "occurrences")
    _VIOLATION_KEYS = ("attack", "threat", "alert", "violation", "incident", "anomaly", "breach")
    _LABEL_KEYS = ("label", "class", "target", "prediction", "result", "classification", "is_malicious")
    _TIME_KEYS = ("timestamp", "time", "date", "datetime", "event_time", "ingest_time")
    _CONTEXT_KEYS = ("source", "destination", "src", "dst", "user", "username", "account",
                     "device", "host", "hostname", "ip", "src_ip", "dst_ip", "port",
                     "src_port", "dst_port", "protocol", "asset", "role")

    # Severity scale map
    _SEVERITY_MAP = {
        "critical": 1.0,
        "high": 0.8,
        "severe": 0.8,
        "medium": 0.5,
        "moderate": 0.5,
        "low": 0.2,
        "minor": 0.2,
    }

    # Label indicator lists
    _MALICIOUS_TOKENS = {
        "malicious", "attack", "threat", "anomaly", "intrusion", "breach", "suspicious",
        "true positive", "tp", "1", "yes", "y", "true", "bad", "harmful", "dangerous",
        "malware", "phishing", "ransomware", "c2", "command and control",
    }
    _BENIGN_TOKENS = {
        "benign", "normal", "safe", "legitimate", "false", "negative", "0", "no", "n",
        "false positive", "fp", "good", "clean", "allow", "whitelisted",
    }

    def __init__(
        self,
        dataset_name: str = "generic_security",
        output_dir: Path = Path("data/processed"),
        *,
        # Optional tuning for frequency normalization (raw -> [0,1] within this max)
        frequency_max_hint: float = 100.0,
        # Optional time horizon for recency scoring (seconds)
        recency_horizons: Tuple[int, int, int, int] = (3600, 86400, 7 * 86400, 30 * 86400),  # 1h, 1d, 7d, 30d
    ):
        super().__init__(dataset_name, output_dir)
        self.frequency_max_hint = max(1.0, float(frequency_max_hint))
        self.recency_horizons = tuple(int(x) for x in recency_horizons)

    # -----------------------------
    # Core API
    # -----------------------------
    def process(self, input_path: Path) -> List[Dict[str, Any]]:
        """Load, transform, and validate records from input_path into Nethical standard records."""
        logger.info(f"[{self.dataset_name}] Processing from {input_path}")

        # Input loading (CSV or JSONL handled by base I/O helpers)
        rows: List[Dict[str, Any]]
        suffix = input_path.suffix.lower()
        try:
            if suffix.endswith("jsonl"):
                rows = self.load_jsonl(input_path)
            else:
                rows = self.load_csv(input_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            return []

        if not rows:
            logger.warning(f"[{self.dataset_name}] No data found in {input_path}")
            return []

        records: List[Dict[str, Any]] = []
        for i, raw in enumerate(rows):
            try:
                rec = self.make_record(raw, include_meta=True, normalize=True)
                if rec is None:
                    continue
                # Enrich metadata
                meta = rec.setdefault("meta", {})
                meta.update({
                    "dataset": self.dataset_name,  # ensured by base, but keep explicit
                    "record_index": i,
                    "source_path": str(input_path),
                })
                records.append(rec)
            except Exception as e:
                logger.warning(f"[{self.dataset_name}] Error processing row {i}: {e}")
                continue

        # Summarize
        try:
            from .base_processor import DatasetStats  # local import to avoid cycles if any
            stats = DatasetStats.compute(records)
            logger.info(f"[{self.dataset_name}] Processed {stats.num_records} records "
                        f"(labels={stats.label_distribution})")
        except Exception:
            logger.info(f"[{self.dataset_name}] Processed {len(records)} records")

        return records

    # -----------------------------
    # Hooks
    # -----------------------------
    def preprocess_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize raw row values."""
        # Trim string fields, keep original keys
        cleaned: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str):
                v2 = v.strip()
                # normalize obvious 'null'/'none' strings to empty
                if v2.lower() in {"", "null", "none", "nan"}:
                    v2 = ""
                cleaned[k] = v2
            else:
                cleaned[k] = v
        return cleaned

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """Keep rows that have at least one indicative field for features or label."""
        kl = {k.lower() for k in row.keys()}
        # Must have at least one field we know how to map
        has_signal = any(k in kl for k in (
            *self._SEVERITY_KEYS, *self._COUNT_KEYS, *self._VIOLATION_KEYS,
            *self._TIME_KEYS, *self._CONTEXT_KEYS, *self._LABEL_KEYS
        ))
        return has_signal

    def postprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Optional post-processing. Currently no-op."""
        return record

    # -----------------------------
    # Feature extraction
    # -----------------------------
    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Map generic security dataset fields to Nethical STANDARD_FEATURES.

        Expected raw ranges before Base normalization:
        - violation_count: 0..10 (will be normalized using feature_ranges)
        - severity_max: 0..1
        - recency_score: 0..1
        - frequency_score: 0..1
        - context_risk: 0..1
        """
        rlow = {k.lower(): row[k] for k in row}

        # severity_max
        severity_max = self._extract_severity(rlow)

        # violation_count (raw 0..10)
        violation_count = self._extract_violation_count(rlow)

        # frequency_score (0..1)
        frequency_score = self._extract_frequency(rlow)

        # recency_score (0..1)
        recency_score = self._extract_recency(rlow)

        # context_risk (0..1)
        context_risk = self._extract_context_risk(rlow)

        return {
            "violation_count": violation_count,
            "severity_max": severity_max,
            "recency_score": recency_score,
            "frequency_score": frequency_score,
            "context_risk": context_risk,
        }

    def extract_label(self, row: Dict[str, Any]) -> int:
        """Extract label using robust heuristics. Returns 1 (malicious) or 0 (benign)."""
        rlow = {k.lower(): row[k] for k in row}

        # Prefer explicit label fields
        for key in self._LABEL_KEYS:
            if key in rlow:
                val = rlow[key]
                # numeric labels
                if isinstance(val, (int, float)):
                    return 1 if int(val) == 1 else 0
                s = str(val).strip().lower()
                if s in self._MALICIOUS_TOKENS:
                    return 1
                if s in self._BENIGN_TOKENS:
                    return 0
                # loose contains (e.g., "True Positive", "benign sample")
                if any(tok in s for tok in self._MALICIOUS_TOKENS):
                    return 1
                if any(tok in s for tok in self._BENIGN_TOKENS):
                    return 0

        # Fall back to severity-based rule if available
        sev = self._extract_severity(rlow)
        if sev >= 0.8:
            return 1

        # Conservative default: benign
        return 0

    # -----------------------------
    # Heuristic helpers
    # -----------------------------
    def _extract_severity(self, rlow: Dict[str, Any]) -> float:
        # Numeric severity or mapped textual levels
        for key in self._SEVERITY_KEYS:
            if key in rlow:
                val = rlow[key]
                # If numeric 0..10 or 1..5 scale
                if isinstance(val, (int, float)):
                    v = float(val)
                    # Heuristic: if >1 assume 0..10 or 1..5
                    if v <= 1.0:
                        return max(0.0, min(1.0, v))
                    # Map 1..5 -> 0..1; 0..10 -> 0..1
                    if 1.0 <= v <= 5.0:
                        return max(0.0, min(1.0, (v - 1.0) / 4.0))
                    # assume 0..10
                    return max(0.0, min(1.0, v / 10.0))
                s = str(val).lower()
                # direct tokens
                for tok, score in self._SEVERITY_MAP.items():
                    if tok in s:
                        return score
                # digits present (e.g., "P2", "sev-4")
                m = re.search(r"([0-9]+)", s)
                if m:
                    try:
                        num = float(m.group(1))
                        if 1 <= num <= 5:
                            return max(0.0, min(1.0, (num - 1.0) / 4.0))
                        return max(0.0, min(1.0, num / 10.0))
                    except Exception:
                        pass
                break
        return 0.0

    def _extract_violation_count(self, rlow: Dict[str, Any]) -> float:
        # Raw count clamped to 0..10 for normalization later
        total = 0.0
        # Numeric counter-style fields
        for key in self._COUNT_KEYS:
            if key in rlow:
                v = self.safe_float(rlow.get(key), 0.0)
                if v > 0:
                    total += v
        # Presence of explicit violation indicators increments baseline
        for key in self._VIOLATION_KEYS:
            if key in rlow:
                val = rlow[key]
                if isinstance(val, (int, float)):
                    if float(val) > 0:
                        total += 1.0
                else:
                    s = str(val).lower()
                    if self._is_truthy_string(s):
                        total += 1.0
        # clamp to expected raw range
        return max(0.0, min(10.0, total))

    def _extract_frequency(self, rlow: Dict[str, Any]) -> float:
        # Normalized by hint to [0,1]
        best = 0.0
        for key in self._COUNT_KEYS:
            if key in rlow:
                v = self.safe_float(rlow.get(key), 0.0)
                if v > 0:
                    best = max(best, self.normalize_feature(v, 0.0, self.frequency_max_hint))
        return best

    def _extract_recency(self, rlow: Dict[str, Any]) -> float:
        # Score newer events higher: now - event_time
        now = datetime.now(timezone.utc)
        event_ts: Optional[datetime] = None
        for key in self._TIME_KEYS:
            if key in rlow:
                ts = self._parse_datetime(rlow[key])
                if ts:
                    event_ts = ts
                    break
        if not event_ts:
            return 0.5  # unknown
        age = (now - event_ts).total_seconds()
        h1, d1, d7, d30 = self.recency_horizons
        if age <= 0:
            return 1.0
        if age <= h1:
            return 1.0
        if age <= d1:
            return 0.8
        if age <= d7:
            return 0.6
        if age <= d30:
            return 0.4
        return 0.2

    def _extract_context_risk(self, rlow: Dict[str, Any]) -> float:
        # Start with base if any context fields present
        has_context = any(k in rlow for k in self._CONTEXT_KEYS)
        risk = 0.4 if has_context else 0.0

        # IP-based hints: external IPs raise risk
        for ip_key in ("ip", "src_ip", "dst_ip"):
            if ip_key in rlow:
                ip_val = str(rlow[ip_key]).strip()
                try:
                    ip_obj = ipaddress.ip_address(ip_val)
                    if not self._is_private_ip(ip_obj):
                        risk = max(risk, 0.6)
                except ValueError:
                    pass

        # Privileged users / sensitive roles
        for user_key in ("user", "username", "account"):
            if user_key in rlow:
                s = str(rlow[user_key]).lower()
                if any(tok in s for tok in ("root", "admin", "administrator", "svc", "system")):
                    risk = max(risk, 0.7)

        # Services/ports often seen in lateral movement/exfil
        for port_key in ("port", "src_port", "dst_port"):
            if port_key in rlow:
                p = self.safe_float(rlow[port_key], 0.0)
                risky_ports = {22, 23, 80, 443, 445, 3389, 5900}
                if int(p) in risky_ports:
                    risk = max(risk, 0.65)

        # Protocol hints
        if "protocol" in rlow:
            proto = str(rlow["protocol"]).lower()
            if any(tok in proto for tok in ("smb", "rdp", "vnc", "ftp", "telnet", "http")):
                risk = max(risk, 0.6)

        # Asset roles
        if "role" in rlow or "asset" in rlow or "hostname" in rlow:
            val = (str(rlow.get("role", "")) + " " +
                   str(rlow.get("asset", "")) + " " +
                   str(rlow.get("hostname", ""))).lower()
            if any(tok in val for tok in ("dc", "domain controller", "prod", "database", "db", "gateway")):
                risk = max(risk, 0.7)

        return max(0.0, min(1.0, float(risk)))

    # -----------------------------
    # Parsing helpers
    # -----------------------------
    @staticmethod
    def _is_truthy_string(s: str) -> bool:
        s = s.strip().lower()
        return s in {"1", "true", "yes", "y", "t"} or any(tok in s for tok in ("true positive", "tp"))

    @staticmethod
    def _parse_datetime(val: Any) -> Optional[datetime]:
        """Parse various timestamp formats into timezone-aware UTC datetime."""
        if isinstance(val, datetime):
            return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        if not isinstance(val, str):
            return None
        s = val.strip()
        if not s:
            return None
        # Try ISO formats directly
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        # Common formats
        fmts = [
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
        return None

    @staticmethod
    def _is_private_ip(ip_obj: ipaddress._BaseAddress) -> bool:
        return ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
