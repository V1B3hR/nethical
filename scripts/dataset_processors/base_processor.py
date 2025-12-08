"""Base dataset processor with common utilities."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import random
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Iterable, Tuple, TypedDict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard feature contract for the Nethical system
STANDARD_FEATURES: List[str] = [
    "violation_count",
    "severity_max",
    "recency_score",
    "frequency_score",
    "context_risk",
]

# Default expected feature ranges used for normalization/clipping
DEFAULT_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "violation_count": (0.0, 10.0),
    "severity_max": (0.0, 1.0),
    "recency_score": (0.0, 1.0),
    "frequency_score": (0.0, 1.0),
    "context_risk": (0.0, 1.0),
}


class StandardRecord(TypedDict, total=False):
    # Required keys
    features: Dict[str, float]
    label: int
    # Optional metadata
    meta: Dict[str, Any]


@dataclass
class DatasetStats:
    num_records: int
    label_distribution: Dict[str, int]
    feature_stats: Dict[str, Dict[str, float]]

    @staticmethod
    def compute(records: Iterable[StandardRecord]) -> "DatasetStats":
        feats: Dict[str, List[float]] = {k: [] for k in STANDARD_FEATURES}
        labels: Dict[str, int] = {"0": 0, "1": 0}
        n = 0
        for r in records:
            n += 1
            lbl = str(r.get("label", 0))
            if lbl not in labels:
                labels[lbl] = 0
            labels[lbl] += 1
            f = r.get("features", {})
            for k in STANDARD_FEATURES:
                v = f.get(k)
                if isinstance(v, (int, float)):
                    feats[k].append(float(v))
        feat_stats: Dict[str, Dict[str, float]] = {}
        for k, arr in feats.items():
            if arr:
                feat_stats[k] = {
                    "min": min(arr),
                    "max": max(arr),
                    "mean": statistics.fmean(arr),
                    "stdev": statistics.pstdev(arr) if len(arr) > 1 else 0.0,
                }
            else:
                feat_stats[k] = {"min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0}
        return DatasetStats(
            num_records=n, label_distribution=labels, feature_stats=feat_stats
        )


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BaseDatasetProcessor:
    """Base class for dataset processors with Nethical-aligned utilities."""

    def __init__(
        self,
        dataset_name: str,
        output_dir: Path = Path("data/processed"),
        *,
        seed: int = 42,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """Initialize processor.

        Args:
            dataset_name: Name of the dataset
            output_dir: Directory to save processed data
            seed: Random seed used for deterministic operations
            feature_ranges: Per-feature (min,max) ranges for normalization/clipping
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        random.seed(self.seed)
        self.feature_ranges = feature_ranges or DEFAULT_FEATURE_RANGES

    # -----------------------------
    # Abstracts (to be overridden)
    # -----------------------------
    def process(self, input_path: Path) -> List[StandardRecord]:
        """Process dataset and return standardized records.

        Must return records of shape:
            {
              "features": { ...STANDARD_FEATURES... },
              "label": 0 or 1,
              "meta": { optional free-form provenance }
            }
        """
        raise NotImplementedError("Subclasses must implement process()")

    # Optional hooks that subclasses may override
    def preprocess_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Hook to clean/transform a raw input row before feature extraction."""
        return row

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """Hook to quickly filter invalid rows. Return True to keep."""
        return True

    def postprocess_record(self, record: StandardRecord) -> StandardRecord:
        """Hook to adjust a record after extraction, before validation/saving."""
        return record

    # -----------------------------
    # I/O helpers
    # -----------------------------
    def _open_text_auto(
        self, path: Path, mode: str = "rt", encoding: Optional[str] = "utf-8"
    ):
        """Open text file with optional gzip support based on suffix."""
        if str(path).endswith(".gz"):
            # gzip.open accepts 'rt'/'wt' with encoding
            return gzip.open(path, mode=mode, encoding=encoding or "utf-8", newline="")
        return open(path, mode=mode, encoding=encoding or "utf-8", newline="")

    def load_csv(self, path: Path, encoding: str = "utf-8") -> List[Dict[str, Any]]:
        """Load CSV/CSV.GZ into list of dicts with encoding fallbacks and dialect sniffing."""
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        def read_with(enc: str) -> List[Dict[str, Any]]:
            rows_local: List[Dict[str, Any]] = []
            with self._open_text_auto(path, "rt", encoding=enc) as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(f, dialect=dialect)
                rows_local = list(reader)
            return rows_local

        encodings_to_try = [encoding, "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]
        for enc in encodings_to_try:
            try:
                rows = read_with(enc)
                if enc != encoding:
                    logger.info(f"Successfully read {path} with encoding {enc}")
                return rows
            except UnicodeDecodeError:
                continue
        # Last resort: open in binary and try to decode per line (may still fail)
        try:
            with (
                gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")
            ) as fb:
                text = fb.read().decode(errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            return list(reader)
        except Exception as e:
            logger.error(f"Failed to read CSV {path}: {e}")
            return []

    def load_jsonl(self, path: Path, encoding: str = "utf-8") -> List[Dict[str, Any]]:
        """Load JSONL (optionally .gz) into a list of dicts."""
        if not path.exists():
            raise FileNotFoundError(f"JSONL not found: {path}")
        records: List[Dict[str, Any]] = []
        with self._open_text_auto(path, "rt", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line")
        return records

    # -----------------------------
    # Normalization & validation
    # -----------------------------
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def normalize_feature(
        self, value: Any, min_val: float = 0.0, max_val: float = 1.0
    ) -> float:
        """Normalize a feature value to [0, 1] range with clipping."""
        try:
            val = float(value)
            if max_val == min_val:
                return 0.5
            normalized = (val - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))
        except (ValueError, TypeError):
            return 0.0

    def normalize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Normalize all STANDARD_FEATURES using configured ranges."""
        out: Dict[str, float] = {}
        for k in STANDARD_FEATURES:
            v = features.get(k, 0.0)
            fmin, fmax = self.feature_ranges.get(k, (0.0, 1.0))
            out[k] = self.normalize_feature(v, fmin, fmax)
        return out

    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Extract standard features from a row.

        Override in subclasses. The base implementation returns zeroed features.
        """
        return {k: 0.0 for k in STANDARD_FEATURES}

    def extract_label(self, row: Dict[str, Any]) -> int:
        """Extract binary label (0 or 1) from a row. Override in subclasses."""
        return 0

    def make_record(
        self,
        row: Dict[str, Any],
        *,
        include_meta: bool = True,
        normalize: bool = True,
    ) -> Optional[StandardRecord]:
        """Transform a raw row into a validated StandardRecord or None if invalid."""
        if not self.validate_row(row):
            return None
        row = self.preprocess_row(row)

        features_raw = self.extract_standard_features(row)
        features = self.normalize_features(features_raw) if normalize else features_raw
        label = int(self.extract_label(row))

        rec: StandardRecord = {"features": features, "label": label}
        if include_meta:
            rec["meta"] = {
                "dataset": self.dataset_name,
                "ingested_at": _now_utc_iso(),
            }
        rec = self.postprocess_record(rec)

        if not self.validate_record(rec):
            return None
        return rec

    def validate_record(self, record: StandardRecord) -> bool:
        """Validate feature keys, ranges, and label."""
        feats = record.get("features")
        if not isinstance(feats, dict):
            return False
        # Ensure all standard features present
        for k in STANDARD_FEATURES:
            if k not in feats:
                logger.debug(f"Missing feature {k}")
                return False
            v = feats[k]
            if not isinstance(v, (int, float)):
                logger.debug(f"Non-numeric feature {k}: {v}")
                return False
            if not (0.0 <= float(v) <= 1.0):
                logger.debug(f"Out-of-range feature {k}: {v}")
                return False
        label = record.get("label")
        if label not in (0, 1):
            logger.debug(f"Invalid label: {label}")
            return False
        return True

    # -----------------------------
    # Saving & metadata
    # -----------------------------
    def _primary_output_stem(self) -> str:
        return f"{self.dataset_name}_processed"

    def _write_json(
        self, path: Path, records: List[StandardRecord], compress: bool = False
    ) -> Path:
        target = path.with_suffix(path.suffix + ".gz") if compress else path
        opener = gzip.open if compress else open
        with opener(target, "wt", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        return target

    def _write_jsonl(
        self, path: Path, records: Iterable[StandardRecord], compress: bool = False
    ) -> Path:
        target = path.with_suffix(path.suffix + ".gz") if compress else path
        opener = gzip.open if compress else open
        with opener(target, "wt", encoding="utf-8", newline="") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        return target

    def _write_metadata(self, base_path: Path, records: List[StandardRecord]) -> Path:
        stats = DatasetStats.compute(records)
        meta = {
            "dataset": self.dataset_name,
            "created_at": _now_utc_iso(),
            "seed": self.seed,
            "num_records": stats.num_records,
            "label_distribution": stats.label_distribution,
            "feature_stats": stats.feature_stats,
            "standard_features": STANDARD_FEATURES,
            "feature_ranges": {
                k: {"min": v[0], "max": v[1]} for k, v in self.feature_ranges.items()
            },
        }
        meta_path = base_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta_path

    def save_processed_data(
        self,
        records: List[StandardRecord],
        *,
        formats: Tuple[str, ...] = ("json",),
        compress: bool = False,
        with_metadata: bool = True,
    ) -> Path:
        """Save processed records to one or more formats.

        Args:
            records: List of processed records
            formats: Any of ("json", "jsonl")
            compress: If True, write gzip (.gz) variants
            with_metadata: If True, write a .meta.json alongside outputs

        Returns:
            Path to the primary file (first format) for backward compatibility
        """
        stem = self._primary_output_stem()
        primary_path: Optional[Path] = None

        for fmt in formats:
            if fmt not in ("json", "jsonl"):
                raise ValueError(f"Unsupported format: {fmt}")
            out_path = self.output_dir / f"{stem}.{fmt}"
            if fmt == "json":
                out_written = self._write_json(out_path, records, compress=compress)
            else:
                out_written = self._write_jsonl(out_path, records, compress=compress)
            logger.info(f"Saved {len(records)} records to {out_written}")
            if primary_path is None:
                primary_path = out_written

        if with_metadata:
            meta_path = self._write_metadata(self.output_dir / stem, records)
            logger.info(f"Wrote metadata to {meta_path}")

        assert primary_path is not None
        return primary_path

    # -----------------------------
    # Splitting & deduplication
    # -----------------------------
    def _stable_hash(self, s: str) -> int:
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

    def deduplicate(
        self,
        records: List[StandardRecord],
        key_fn: Optional[Callable[[StandardRecord], str]] = None,
    ) -> List[StandardRecord]:
        """Deduplicate records using a key function, keeping first occurrence."""
        if key_fn is None:
            # Default: hash features + label
            def key_fn_default(r: StandardRecord) -> str:
                feats = r.get("features", {})
                payload = json.dumps(
                    {"f": feats, "l": r.get("label", 0)}, sort_keys=True
                )
                return hashlib.md5(payload.encode("utf-8")).hexdigest()

            key_fn = key_fn_default

        seen: set[str] = set()
        out: List[StandardRecord] = []
        for r in records:
            k = key_fn(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    def stratified_split(
        self,
        records: List[StandardRecord],
        *,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        group_key: Optional[Callable[[StandardRecord], str]] = None,
    ) -> Dict[str, List[StandardRecord]]:
        """Create deterministic stratified splits by label; supports grouping to avoid leakage.

        Args:
            records: dataset
            train/val/test: fractions summing to 1.0
            group_key: if provided, items with same key will go to the same split
        """
        eps = abs((train + val + test) - 1.0)
        if eps > 1e-6:
            raise ValueError("train + val + test must sum to 1.0")

        # Group by label for stratification
        buckets: Dict[int, List[StandardRecord]] = {0: [], 1: []}
        for r in records:
            buckets[int(r.get("label", 0))].append(r)

        def assign_split(
            items: List[StandardRecord],
        ) -> Dict[str, List[StandardRecord]]:
            if not group_key:
                # Simple deterministic shuffle by hash of repr + seed
                def sort_key(x: StandardRecord) -> int:
                    payload = json.dumps(x, sort_keys=True, ensure_ascii=False)
                    return self._stable_hash(payload + str(self.seed))

                sorted_items = sorted(items, key=sort_key)
                n = len(sorted_items)
                n_train = int(n * train)
                n_val = int(n * val)
                return {
                    "train": sorted_items[:n_train],
                    "val": sorted_items[n_train : n_train + n_val],
                    "test": sorted_items[n_train + n_val :],
                }
            # Group-aware assignment
            groups: Dict[str, List[StandardRecord]] = {}
            for it in items:
                g = group_key(it)
                groups.setdefault(g, []).append(it)

            # Deterministic group ordering
            def gkey(k: str) -> int:
                return self._stable_hash(k + str(self.seed))

            ordered_groups = sorted(groups.items(), key=lambda kv: gkey(kv[0]))
            train_set: List[StandardRecord] = []
            val_set: List[StandardRecord] = []
            test_set: List[StandardRecord] = []
            total = sum(len(v) for _, v in ordered_groups)
            target_train = total * train
            target_val = total * val

            c_train = c_val = 0
            for _, members in ordered_groups:
                if c_train < target_train:
                    train_set.extend(members)
                    c_train += len(members)
                elif c_val < target_val:
                    val_set.extend(members)
                    c_val += len(members)
                else:
                    test_set.extend(members)
            return {"train": train_set, "val": val_set, "test": test_set}

        # Merge per-label splits to preserve ratio
        out: Dict[str, List[StandardRecord]] = {"train": [], "val": [], "test": []}
        for lbl, items in buckets.items():
            split_lbl = assign_split(items)
            for k in out:
                out[k].extend(split_lbl[k])

        # Final deterministic shuffle within each split (by hash)
        for k in out:
            out[k] = sorted(
                out[k],
                key=lambda r: self._stable_hash(
                    json.dumps(r, sort_keys=True) + k + str(self.seed)
                ),
            )
        return out

    def save_splits(
        self,
        splits: Dict[str, List[StandardRecord]],
        *,
        base_name: Optional[str] = None,
        fmt: str = "jsonl",
        compress: bool = True,
        with_metadata: bool = True,
    ) -> Dict[str, Path]:
        """Save precomputed splits to disk, returns mapping of split name to path."""
        if fmt not in ("json", "jsonl"):
            raise ValueError("fmt must be 'json' or 'jsonl'")
        stem = base_name or self._primary_output_stem()
        paths: Dict[str, Path] = {}
        for split_name, records in splits.items():
            out = self.output_dir / f"{stem}.{split_name}.{fmt}"
            written = (
                self._write_json(out, records, compress)
                if fmt == "json"
                else self._write_jsonl(out, records, compress)
            )
            logger.info(
                f"Saved split '{split_name}' with {len(records)} records to {written}"
            )
            paths[split_name] = written
            if with_metadata:
                meta_stem = self.output_dir / f"{stem}.{split_name}"
                meta_path = self._write_metadata(meta_stem, records)
                logger.info(f"Wrote metadata for split '{split_name}' to {meta_path}")
        return paths
