"""
Benchmark Comparison Tool

Compares benchmark results against baselines to detect performance regressions.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing a metric against baseline."""

    metric: str
    current_value: float
    baseline_value: float
    difference_percent: float
    regression: bool
    threshold_percent: float


@dataclass
class BenchmarkComparison:
    """Overall comparison of benchmark results."""

    scenario_name: str
    passed: bool
    comparisons: List[ComparisonResult]
    summary: str


class BenchmarkComparer:
    """Compares benchmark results against baselines."""

    # Default regression thresholds (percentage increase that triggers failure)
    DEFAULT_THRESHOLDS = {
        "avg_latency_ms": 10.0,  # 10% increase triggers regression
        "p95_latency_ms": 15.0,
        "p99_latency_ms": 20.0,
        "throughput_rps": -10.0,  # 10% decrease triggers regression
    }

    def __init__(
        self,
        baselines_dir: str = "benchmarks/baselines",
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.baselines_dir = Path(baselines_dir)
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def load_baseline(self, version: str) -> Optional[Dict[str, Any]]:
        """Load baseline results for a specific version."""
        baseline_path = self.baselines_dir / f"{version}.json"
        if not baseline_path.exists():
            logger.warning(f"Baseline not found: {baseline_path}")
            return None

        with open(baseline_path) as f:
            return json.load(f)

    def compare_metric(
        self,
        metric: str,
        current: float,
        baseline: float,
        threshold: Optional[float] = None,
    ) -> ComparisonResult:
        """Compare a single metric against baseline."""
        if threshold is None:
            threshold = self.thresholds.get(metric, 10.0)

        if baseline == 0:
            difference_percent = 0.0 if current == 0 else 100.0
        else:
            difference_percent = ((current - baseline) / baseline) * 100

        # For throughput, regression is when value decreases
        if metric == "throughput_rps":
            regression = difference_percent < threshold  # threshold is negative
        else:
            regression = difference_percent > threshold

        return ComparisonResult(
            metric=metric,
            current_value=current,
            baseline_value=baseline,
            difference_percent=difference_percent,
            regression=regression,
            threshold_percent=threshold,
        )

    def compare_scenario(
        self,
        current_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        scenario_name: str,
    ) -> BenchmarkComparison:
        """Compare results for a single scenario against baseline."""
        comparisons: List[ComparisonResult] = []
        metrics_to_compare = [
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "throughput_rps",
        ]

        for metric in metrics_to_compare:
            current = current_results.get(metric, 0)
            baseline = baseline_results.get(metric, 0)
            comparison = self.compare_metric(metric, current, baseline)
            comparisons.append(comparison)

        regressions = [c for c in comparisons if c.regression]
        passed = len(regressions) == 0

        if passed:
            summary = f"✅ {scenario_name}: All metrics within thresholds"
        else:
            regression_names = [c.metric for c in regressions]
            summary = (
                f"❌ {scenario_name}: Regressions detected in: "
                f"{', '.join(regression_names)}"
            )

        return BenchmarkComparison(
            scenario_name=scenario_name,
            passed=passed,
            comparisons=comparisons,
            summary=summary,
        )

    def compare_results(
        self,
        current_path: str,
        baseline_version: str = "v2.2.0",
    ) -> List[BenchmarkComparison]:
        """Compare current benchmark results against a baseline version."""
        baseline = self.load_baseline(baseline_version)
        if baseline is None:
            logger.error(f"Cannot compare: baseline {baseline_version} not found")
            return []

        with open(current_path) as f:
            current = json.load(f)

        comparisons: List[BenchmarkComparison] = []

        # Map results by scenario name
        baseline_by_name = {r["name"]: r for r in baseline.get("results", [])}
        current_by_name = {r["name"]: r for r in current.get("results", [])}

        for name, current_result in current_by_name.items():
            if name not in baseline_by_name:
                logger.warning(f"No baseline for scenario: {name}")
                continue

            comparison = self.compare_scenario(
                current_result, baseline_by_name[name], name
            )
            comparisons.append(comparison)

        return comparisons

    def generate_report(self, comparisons: List[BenchmarkComparison]) -> str:
        """Generate a human-readable comparison report."""
        lines = [
            "# Benchmark Comparison Report",
            "",
            "## Summary",
            "",
        ]

        all_passed = all(c.passed for c in comparisons)
        if all_passed:
            lines.append("✅ **All benchmarks passed** - no regressions detected")
        else:
            failed = [c for c in comparisons if not c.passed]
            lines.append(
                f"❌ **{len(failed)}/{len(comparisons)} benchmarks failed** - "
                "regressions detected"
            )

        lines.extend(["", "## Details", ""])

        for comparison in comparisons:
            lines.append(f"### {comparison.scenario_name}")
            lines.append("")
            lines.append(comparison.summary)
            lines.append("")
            lines.append("| Metric | Current | Baseline | Change | Status |")
            lines.append("|--------|---------|----------|--------|--------|")

            for c in comparison.comparisons:
                status = "❌ REGRESSION" if c.regression else "✅ OK"
                change = f"{c.difference_percent:+.2f}%"
                lines.append(
                    f"| {c.metric} | {c.current_value:.2f} | "
                    f"{c.baseline_value:.2f} | {change} | {status} |"
                )
            lines.append("")

        return "\n".join(lines)


def compare_and_report(
    current_path: str,
    baseline_version: str = "v2.2.0",
    fail_on_regression: bool = True,
) -> int:
    """Compare benchmarks and return exit code."""
    comparer = BenchmarkComparer()
    comparisons = comparer.compare_results(current_path, baseline_version)

    if not comparisons:
        logger.error("No comparisons could be made")
        return 1

    report = comparer.generate_report(comparisons)
    print(report)

    all_passed = all(c.passed for c in comparisons)
    if fail_on_regression and not all_passed:
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python compare.py <current_results.json> [baseline_version]")
        sys.exit(1)

    current_path = sys.argv[1]
    baseline_version = sys.argv[2] if len(sys.argv) > 2 else "v2.2.0"

    exit_code = compare_and_report(current_path, baseline_version)
    sys.exit(exit_code)
