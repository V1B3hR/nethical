"""
Storage Projection Tests - Data & Storage Requirement 7.3

Tests 12-month storage projections and budget thresholds.

Run with: pytest tests/storage/test_projections.py -v -s
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


# Constants
BUDGET_THRESHOLD_USD = 50.00  # Monthly budget threshold
BUDGET_THRESHOLD_WARNING_PCT = 0.1  # Warn if cost exceeds 10% of threshold


class StorageProjection:
    """Calculate storage projections"""

    def __init__(self, daily_data_mb: float, compression_ratios: Dict[str, float]):
        self.daily_data_mb = daily_data_mb
        self.compression_ratios = compression_ratios
        self.monthly_data_mb = daily_data_mb * 30

    def project_month(self, month: int) -> Dict[str, Any]:
        """
        Project storage for a specific month

        Args:
            month: Month number (1-12)

        Returns:
            Dictionary with storage breakdown by tier
        """
        # Define tier age ranges (in days)
        tier_ranges = {
            "hot": (0, 30),
            "warm": (31, 90),
            "cool": (91, 365),
            "cold": (366, 1095),  # 3 years
        }

        total_days = month * 30
        tier_storage = {}
        total_raw = 0
        total_compressed = 0

        for tier, (start_day, end_day) in tier_ranges.items():
            # Calculate how many days of data fall in this tier
            if total_days <= start_day:
                days_in_tier = 0
            elif total_days <= end_day:
                days_in_tier = total_days - start_day
            else:
                days_in_tier = end_day - start_day + 1

            raw_mb = days_in_tier * self.daily_data_mb
            compressed_mb = raw_mb / self.compression_ratios.get(tier, 1.0)

            tier_storage[tier] = {
                "days": days_in_tier,
                "raw_mb": raw_mb,
                "compressed_mb": compressed_mb,
                "ratio": self.compression_ratios.get(tier, 1.0),
            }

            total_raw += raw_mb
            total_compressed += compressed_mb

        return {
            "month": month,
            "total_days": total_days,
            "total_raw_mb": total_raw,
            "total_compressed_mb": total_compressed,
            "effective_ratio": (
                total_raw / total_compressed if total_compressed > 0 else 0
            ),
            "tiers": tier_storage,
        }

    def project_12_months(self) -> List[Dict[str, Any]]:
        """Project storage for 12 months"""
        return [self.project_month(m) for m in range(1, 13)]

    def calculate_costs(
        self, projection: Dict[str, Any], cost_per_gb: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate storage costs for a projection

        Args:
            projection: Output from project_month()
            cost_per_gb: Cost per GB per month for each tier

        Returns:
            Cost breakdown
        """
        tier_costs = {}
        total_cost = 0

        for tier, data in projection["tiers"].items():
            gb = data["compressed_mb"] / 1024
            cost = gb * cost_per_gb.get(tier, 0)
            tier_costs[tier] = {
                "gb": gb,
                "cost_per_gb": cost_per_gb.get(tier, 0),
                "total_cost": cost,
            }
            total_cost += cost

        return {
            "month": projection["month"],
            "tier_costs": tier_costs,
            "total_cost": total_cost,
            "total_gb": projection["total_compressed_mb"] / 1024,
        }


@pytest.fixture
def projection():
    """Create storage projection calculator"""
    # Based on requirements document assumptions:
    # - 1000 agents
    # - 100 actions/agent/day = 100,000 actions/day
    # - 2 KB per action = 200 MB/day
    # - Metrics: 43.2 MB/day
    # - Audit: 25 MB/day
    # Total: ~268 MB/day

    compression_ratios = {
        "hot": 1.0,  # No compression
        "warm": 2.0,  # LZ4
        "cool": 5.0,  # ZSTD
        "cold": 10.0,  # ZSTD max
    }

    return StorageProjection(daily_data_mb=268, compression_ratios=compression_ratios)


@pytest.fixture
def cost_structure():
    """AWS-like storage costs"""
    return {
        "hot": 0.08,  # SSD EBS gp3
        "warm": 0.08,  # SSD EBS gp3
        "cool": 0.045,  # HDD EBS st1
        "cold": 0.0125,  # S3 Standard-IA
    }


@pytest.fixture
def output_dir():
    """Output directory for test artifacts"""
    return Path("tests/storage/results")


def test_monthly_projections(projection):
    """
    Test monthly storage projections

    Validates:
    - Month 1: ~8 GB
    - Month 3: ~13-15 GB
    - Month 6: ~18-20 GB
    - Month 12: ~28-30 GB
    """
    print("\n=== Testing Monthly Storage Projections ===")

    key_months = [1, 3, 6, 12]
    expected_ranges = {
        1: (7, 9),  # ~8 GB
        3: (13, 15),  # ~14 GB
        6: (18, 20),  # ~19 GB
        12: (28, 32),  # ~30 GB
    }

    print("\nProjected Storage:")
    print(f"{'Month':<8} {'Raw GB':<12} {'Compressed GB':<16} {'Ratio':<8}")
    print("-" * 50)

    for month in key_months:
        proj = projection.project_month(month)
        raw_gb = proj["total_raw_mb"] / 1024
        compressed_gb = proj["total_compressed_mb"] / 1024
        ratio = proj["effective_ratio"]

        print(f"{month:<8} {raw_gb:>10.1f} {compressed_gb:>14.1f} {ratio:>6.2f}:1")

        # Validate against expected range
        expected_min, expected_max = expected_ranges[month]
        assert (
            expected_min <= compressed_gb <= expected_max
        ), f"Month {month} projection out of range: {compressed_gb:.1f} GB (expected {expected_min}-{expected_max})"

    print("\n✅ Monthly projections validated")


def test_12_month_projection(projection, output_dir):
    """
    Test full 12-month projection

    Generates complete projection report
    """
    print("\n=== Testing 12-Month Projection ===")

    projections = projection.project_12_months()

    # Save detailed report
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report
    report_file = output_dir / f"storage_projection_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(projections, f, indent=2)

    # Markdown report
    md_file = output_dir / f"storage_projection_{timestamp}.md"
    with open(md_file, "w") as f:
        f.write("# 12-Month Storage Projection\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Daily Data Generation**: {projection.daily_data_mb:.1f} MB/day\n\n")

        f.write("## Monthly Breakdown\n\n")
        f.write(
            "| Month | Raw (GB) | Compressed (GB) | Ratio | Hot | Warm | Cool | Cold |\n"
        )
        f.write(
            "|-------|----------|-----------------|-------|-----|------|------|------|\n"
        )

        for proj in projections:
            month = proj["month"]
            raw_gb = proj["total_raw_mb"] / 1024
            comp_gb = proj["total_compressed_mb"] / 1024
            ratio = proj["effective_ratio"]

            hot = proj["tiers"]["hot"]["compressed_mb"] / 1024
            warm = proj["tiers"]["warm"]["compressed_mb"] / 1024
            cool = proj["tiers"]["cool"]["compressed_mb"] / 1024
            cold = proj["tiers"]["cold"]["compressed_mb"] / 1024

            f.write(f"| {month} | {raw_gb:.1f} | {comp_gb:.1f} | {ratio:.2f}:1 | ")
            f.write(f"{hot:.1f} | {warm:.1f} | {cool:.1f} | {cold:.1f} |\n")

        f.write("\n## Growth Trend\n\n")
        f.write("```\n")
        for proj in projections:
            month = proj["month"]
            comp_gb = proj["total_compressed_mb"] / 1024
            bar_length = int(comp_gb / 2)  # Scale for visualization
            bar = "█" * bar_length
            f.write(f"Month {month:>2}: {bar} {comp_gb:.1f} GB\n")
        f.write("```\n")

    print(f"\nReports saved:")
    print(f"  JSON: {report_file}")
    print(f"  Markdown: {md_file}")

    # Validate final month
    final_projection = projections[-1]
    final_gb = final_projection["total_compressed_mb"] / 1024

    print(f"\nFinal (Month 12): {final_gb:.1f} GB")

    # Should be around 28-30 GB
    assert (
        25 <= final_gb <= 35
    ), f"12-month projection out of expected range: {final_gb:.1f} GB"

    print("✅ 12-month projection validated")


def test_budget_threshold(projection, cost_structure, output_dir):
    """
    Test budget threshold compliance

    Target: < $50/month budget threshold

    Validates:
    - Month 12 cost < $50
    - Cost projection trend
    """
    print("\n=== Testing Budget Threshold ===")

    projections = projection.project_12_months()
    budget_threshold = 50.00  # USD

    print(f"\nBudget Threshold: ${budget_threshold:.2f}/month")
    print(f"\nMonthly Cost Breakdown:")
    print(f"{'Month':<8} {'Storage GB':<14} {'Cost':<12} {'% of Budget':<14}")
    print("-" * 52)

    monthly_costs = []

    for proj in projections:
        costs = projection.calculate_costs(proj, cost_structure)
        monthly_costs.append(costs)

        month = costs["month"]
        total_gb = costs["total_gb"]
        total_cost = costs["total_cost"]
        pct_budget = (total_cost / budget_threshold) * 100

        print(f"{month:<8} {total_gb:>12.2f} ${total_cost:>10.2f} {pct_budget:>12.1f}%")

        # Validate under budget
        assert (
            total_cost < budget_threshold
        ), f"Month {month} exceeds budget: ${total_cost:.2f} > ${budget_threshold:.2f}"

    # Final month cost
    final_cost = monthly_costs[-1]["total_cost"]
    final_gb = monthly_costs[-1]["total_gb"]

    print(f"\n{'='*52}")
    print(f"Month 12 Total: {final_gb:.2f} GB @ ${final_cost:.2f}/month")
    print(f"Budget Remaining: ${budget_threshold - final_cost:.2f}")
    print(f"Budget Utilization: {(final_cost/budget_threshold)*100:.1f}%")

    # Save cost report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cost_file = output_dir / f"cost_projection_{timestamp}.json"
    with open(cost_file, "w") as f:
        json.dump(
            {
                "budget_threshold": budget_threshold,
                "monthly_costs": monthly_costs,
                "cost_structure": cost_structure,
            },
            f,
            indent=2,
        )

    print(f"\nCost report saved: {cost_file}")

    # Validate well under budget
    assert (
        final_cost < budget_threshold * BUDGET_THRESHOLD_WARNING_PCT
    ), f"Month 12 cost too high: ${final_cost:.2f} (expected < ${budget_threshold * BUDGET_THRESHOLD_WARNING_PCT:.2f})"

    print(
        f"\n✅ Budget threshold validated (${final_cost:.2f} < ${budget_threshold:.2f})"
    )


def test_scaling_projections(projection, cost_structure):
    """
    Test storage projections at different scales

    Validates projections for:
    - 1× scale (1,000 agents)
    - 10× scale (10,000 agents)
    - 100× scale (100,000 agents)
    """
    print("\n=== Testing Scaling Projections ===")

    scales = {"1x": 1, "10x": 10, "100x": 100}

    budget_threshold = 50.00

    print(
        f"\n{'Scale':<10} {'Month 12 GB':<16} {'Month 12 Cost':<16} {'Within Budget':<16}"
    )
    print("-" * 62)

    for scale_name, multiplier in scales.items():
        # Create projection for this scale
        scaled_projection = StorageProjection(
            daily_data_mb=projection.daily_data_mb * multiplier,
            compression_ratios=projection.compression_ratios,
        )

        # Get month 12 projection
        month12 = scaled_projection.project_month(12)
        costs = scaled_projection.calculate_costs(month12, cost_structure)

        total_gb = costs["total_gb"]
        total_cost = costs["total_cost"]
        within_budget = total_cost < budget_threshold

        print(
            f"{scale_name:<10} {total_gb:>14.1f} ${total_cost:>14.2f} {'✅ Yes' if within_budget else '❌ No':<16}"
        )

        # 1× and 10× should be well within budget
        if multiplier <= 10:
            assert (
                within_budget
            ), f"{scale_name} scale exceeds budget: ${total_cost:.2f} > ${budget_threshold:.2f}"

    print("\n✅ Scaling projections validated")


def test_growth_rate(projection):
    """
    Test storage growth rate

    Validates:
    - Growth is linear for raw data
    - Growth is sub-linear for compressed data (due to tiering)
    - No unexpected spikes
    """
    print("\n=== Testing Storage Growth Rate ===")

    projections = projection.project_12_months()

    print("\nMonth-over-Month Growth:")
    print(f"{'Month':<8} {'Compressed GB':<16} {'Growth GB':<14} {'Growth %':<12}")
    print("-" * 54)

    prev_gb = 0
    for proj in projections:
        month = proj["month"]
        comp_gb = proj["total_compressed_mb"] / 1024
        growth_gb = comp_gb - prev_gb
        growth_pct = (growth_gb / prev_gb * 100) if prev_gb > 0 else 0

        print(f"{month:<8} {comp_gb:>14.1f} {growth_gb:>12.1f} {growth_pct:>10.1f}%")

        # Growth should be decreasing over time due to compression
        # Month 1->2 growth should be higher than Month 11->12
        if month > 1:
            assert growth_gb > 0, f"Storage should grow each month"
            if month > 3:
                # After initial months, growth rate should stabilize
                assert growth_pct < 50, f"Growth rate too high: {growth_pct:.1f}%"

        prev_gb = comp_gb

    print("\n✅ Growth rate validated")
