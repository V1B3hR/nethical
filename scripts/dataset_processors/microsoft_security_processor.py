"""Processor for Microsoft Security Incident Prediction dataset."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_processor import (
    BaseDatasetProcessor,
    STANDARD_FEATURES,
    DEFAULT_FEATURE_RANGES,
)

logger = logging.getLogger(__name__)


class MicrosoftSecurityProcessor(BaseDatasetProcessor):
    """Process the Microsoft Security Incident Prediction dataset.

    Dataset: https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction

    This implementation aligns with Nethical's standardized record contract:
      - Uses BaseDatasetProcessor.make_record to normalize features and add metadata
      - Implements robust parsing (ints, floats, booleans, datetimes)
      - Adds flexible field fallbacks for common Microsoft Security schema variants
      - Provides dataset-specific feature ranges for better normalization
    """

    # Common field aliases seen across Microsoft security datasets
    _ALERT_COUNT_KEYS: Tuple[str, ...] = (
        "AlertCount",
        "NumberOfAlerts",
        "Alert Counts",
        "Alerts",
        "AlertsCount",
    )
    _SEVERITY_KEYS: Tuple[str, ...] = ("Severity", "AlertSeverity", "IncidentSeverity")
    _INCIDENT_GRADE_KEYS: Tuple[str, ...] = (
        "IncidentGrade",
        "Incident Grade",
        "Classification",
        "Label",
    )

    # Timestamps possibly present for recency computation
    _DATETIME_KEYS: Tuple[str, ...] = (
        "IncidentCreationTime",
        "CreationTimeUtc",
        "CreatedTimeUtc",
        "CreatedTime",
        "FirstSeen",
        "Timestamp",
        "TimeGenerated",
    )

    # Contextual identifiers that increase context_risk
    _CONTEXT_KEYS: Tuple[str, ...] = (
        "DeviceId",
        "OrgId",
        "DetectorId",
        "UserId",
        "AccountId",
        "TenantId",
    )

    def __init__(
        self,
        output_dir: Path = Path("data/processed"),
        *,
        seed: int = 42,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        # Provide dataset-tuned feature ranges; allow caller override
        msft_feature_ranges = {
            **DEFAULT_FEATURE_RANGES,
            # Allow higher raw counts; normalization will clamp to [0,1]
            "violation_count": (0.0, 50.0),
            "frequency_score": (0.0, 500.0),
            # Others already in [0,1] space
        }
        super().__init__(
            "microsoft_security",
            output_dir,
            seed=seed,
            feature_ranges=feature_ranges or msft_feature_ranges,
        )

    # -----------------------------
    # Parsing helpers
    # -----------------------------
    @staticmethod
    def _parse_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_int(v: Any, default: int = 0) -> int:
        try:
            # Accept floats-as-strings too
            return int(float(v))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_bool(v: Any) -> Optional[bool]:
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "t"}:
            return True
        if s in {"0", "false", "no", "n", "f"}:
            return False
        return None

    @staticmethod
    def _first_non_empty(row: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
        for k in keys:
            if k in row and str(row[k]).strip() != "":
                return row[k]
        return None

    @staticmethod
    def _parse_datetime_any(
        row: Dict[str, Any], keys: Tuple[str, ...]
    ) -> Optional[datetime]:
        raw = MicrosoftSecurityProcessor._first_non_empty(row, keys)
        if raw is None:
            return None
        s = str(raw).strip()

        # Try a few common formats before falling back to fromisoformat
        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %H:%M",
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # fromisoformat variants
        try:
            # Handle trailing Z
            if s.endswith("Z"):
                s = s[:-1]
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None

    # -----------------------------
    # Base hooks
    # -----------------------------
    def preprocess_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize keys and trim string values."""
        # Standardize keys by stripping whitespace
        cleaned: Dict[str, Any] = {}
        for k, v in row.items():
            nk = str(k).strip()
            if isinstance(v, str):
                cleaned[nk] = v.strip()
            else:
                cleaned[nk] = v
        return cleaned

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """Keep rows that have enough signal to produce features/labels."""
        has_any_signal = any(
            k in row for k in (self._SEVERITY_KEYS + self._INCIDENT_GRADE_KEYS + self._ALERT_COUNT_KEYS)  # type: ignore
        )
        return bool(row) and has_any_signal

    # -----------------------------
    # Feature extraction
    # -----------------------------
    def _severity_from_grade(self, row: Dict[str, Any]) -> float:
        """Map IncidentGrade/Classification/Label to a severity proxy in [0,1]."""
        raw = self._first_non_empty(row, self._INCIDENT_GRADE_KEYS)
        if raw is None:
            return 0.0
        s = str(raw).strip().lower()

        # Normalize shorthand and variants
        aliases = {
            "truepositive": "true positive",
            "benignpositive": "benign positive",
            "falsepositive": "false positive",
            "tp": "true positive",
            "bp": "benign positive",
            "fp": "false positive",
        }
        s = aliases.get(s, s)

        if "true" in s and "positive" in s:
            return 0.85
        if "benign" in s and "positive" in s:
            return 0.35
        if "false" in s and "positive" in s:
            return 0.15
        # Unknown grade: neutral/low
        return 0.2

    def _severity_from_severity_field(self, row: Dict[str, Any]) -> float:
        """Map Severity textual/numeric fields to [0,1]."""
        raw = self._first_non_empty(row, self._SEVERITY_KEYS)
        if raw is None:
            return 0.0
        s = str(raw).strip().lower()

        # Try numeric first
        try:
            # Many Microsoft severities are 0..5
            val = float(s)
            return self.normalize_feature(val, 0.0, 5.0)
        except ValueError:
            pass

        # Textual mapping
        if "critical" in s:
            return 1.0
        if "high" in s:
            return 0.9
        if "medium" in s or "moderate" in s:
            return 0.6
        if "low" in s:
            return 0.3
        return 0.0

    def _recency_score(self, row: Dict[str, Any], horizon_days: float = 90.0) -> float:
        """Compute a [0,1] recency score where 1 is most recent."""
        dt = self._parse_datetime_any(row, self._DATETIME_KEYS)
        if dt is None:
            # Neutral default if unknown
            return 0.5
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
        return max(0.0, min(1.0, 1.0 - (days / horizon_days)))

    def _alert_count(self, row: Dict[str, Any]) -> int:
        raw = self._first_non_empty(row, self._ALERT_COUNT_KEYS)
        return self._parse_int(raw, 0)

    def _context_risk(self, row: Dict[str, Any]) -> float:
        """Heuristic: increase risk if multiple contextual identifiers are present."""
        present = sum(
            1 for k in self._CONTEXT_KEYS if k in row and str(row[k]).strip() != ""
        )
        # 0..3+ presence -> 0.0..0.7
        if present <= 0:
            return 0.0
        if present == 1:
            return 0.25
        if present == 2:
            return 0.5
        if present == 3:
            return 0.7
        return 0.85

    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Map Microsoft Security dataset fields to standard features.

        Returns raw values consistent with feature_ranges where appropriate:
          - violation_count: raw count-like value (range handled by feature_ranges)
          - frequency_score: raw alert count (range handled by feature_ranges)
          - severity_max, recency_score, context_risk: values already in [0,1]
        """
        # Violation proxy: presence of notable fields or counts
        violation_count = 0.0
        if any(k in row for k in self._INCIDENT_GRADE_KEYS) or "AlertTitle" in row:
            violation_count += 1.0
        # Leverage alert count as additional violations
        cnt = self._alert_count(row)
        violation_count += max(0.0, float(cnt))

        # Severity: use the max of grade-derived and severity-field-derived
        sev_from_grade = self._severity_from_grade(row)
        sev_from_field = self._severity_from_severity_field(row)
        severity_max = max(sev_from_grade, sev_from_field)

        recency_score = self._recency_score(row)
        frequency_score = float(cnt)

        context_risk = self._context_risk(row)

        features = {
            "violation_count": violation_count,
            "severity_max": severity_max,
            "recency_score": recency_score,
            "frequency_score": frequency_score,
            "context_risk": context_risk,
        }

        # Ensure all standard features present
        for k in STANDARD_FEATURES:
            features.setdefault(k, 0.0)
        return features

    # -----------------------------
    # Label extraction
    # -----------------------------
    def extract_label(self, row: Dict[str, Any]) -> int:
        """Extract label: 1 for true positive/incidents, 0 for benign/false/no incident."""
        # Primary: grades/labels that contain "true positive"
        grade_raw = self._first_non_empty(row, self._INCIDENT_GRADE_KEYS)
        if grade_raw is not None:
            s = str(grade_raw).strip().lower()
            # Normalize known shorthands
            if s in {"tp", "truepositive"}:
                return 1
            if "true" in s and "positive" in s:
                return 1
            if s in {"fp", "falsepositive"}:
                return 0
            if "false" in s and "positive" in s:
                return 0
            if "benign" in s and "positive" in s:
                return 0

        # Secondary: boolean-like fields
        for k in ("HasIncident", "IsIncident", "Incident", "IsTruePositive"):
            if k in row:
                b = self._parse_bool(row.get(k))
                if b is not None:
                    return 1 if b else 0

        # Fallback: treat high/critical severity as positive if no explicit grade
        sev = self._severity_from_severity_field(row)
        if sev >= 0.85:
            return 1

        return 0

    # -----------------------------
    # Processing pipeline
    # -----------------------------
    def process(self, input_path: Path) -> List[Dict[str, Any]]:
        """Process the dataset at input_path and return standardized records.

        Returns a list of StandardRecord dicts with:
          - features: Dict[str, float] normalized to [0,1]
          - label: int (0 or 1)
          - meta: provenance (dataset, ingested_at)
        """
        logger.info(f"[{self.dataset_name}] Loading data from {input_path}")
        rows = self.load_csv(input_path)
        if not rows:
            logger.warning(f"[{self.dataset_name}] No data found in {input_path}")
            return []

        out: List[Dict[str, Any]] = []
        for i, raw in enumerate(rows):
            try:
                rec = self.make_record(raw, include_meta=True, normalize=True)
                if rec is None:
                    continue
                out.append(rec)
            except Exception as e:
                logger.warning(f"[{self.dataset_name}] Error processing row {i}: {e}")

        # Optional deduplication (keep first occurrence)
        before = len(out)
        out = self.deduplicate(out)
        after = len(out)
        if after < before:
            logger.info(
                f"[{self.dataset_name}] Deduplicated records: {before} -> {after}"
            )

        # Basic stats in logs
        logger.info(f"[{self.dataset_name}] Processed {len(out)} records")
        return out
