#!/usr/bin/env python3
"""Load Generator for Nethical Governance System.

This script simulates multiple AI agents generating actions at a target aggregate
RPS (requests per second) to validate throughput and latency SLOs.

Usage:
    # Basic test: 100 agents at 50 RPS for 60 seconds
    python generate_load.py --agents 100 --rps 50 --duration 60

    # Feature comparison: test with/without ML features
    python generate_load.py --agents 200 --rps 100 --duration 60 --shadow
    python generate_load.py --agents 200 --rps 100 --duration 60 --no-shadow

    # Stress test: find breaking point
    python generate_load.py --agents 1000 --rps 500 --duration 120

Requirements:
    - Standard library only (no external dependencies beyond nethical itself)
    - Python 3.8+
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure nethical is importable
try:
    from nethical.core import IntegratedGovernance
except ImportError:
    print("Error: nethical package not found. Install with: pip install -e .")
    sys.exit(1)


class LoadGenerator:
    """Generate load against Nethical governance system."""

    def __init__(
        self,
        agents: int,
        target_rps: float,
        duration: int,
        cohort: str,
        storage_dir: str,
        region_id: Optional[str],
        logical_domain: Optional[str],
        # Feature flags
        enable_shadow: bool,
        enable_ml_blend: bool,
        enable_anomaly: bool,
        enable_merkle: bool,
        enable_quota: bool,
        privacy_mode: Optional[str],
        redaction_policy: str,
        requests_per_second: float,
    ):
        """Initialize load generator.

        Args:
            agents: Number of agents to simulate
            target_rps: Target aggregate requests per second
            duration: Test duration in seconds
            cohort: Cohort identifier for grouping
            storage_dir: Storage directory for governance data
            region_id: Optional region identifier
            logical_domain: Optional logical domain
            enable_shadow: Enable ML shadow mode
            enable_ml_blend: Enable ML blended enforcement
            enable_anomaly: Enable anomaly detection
            enable_merkle: Enable Merkle anchoring
            enable_quota: Enable quota enforcement
            privacy_mode: Privacy mode (None, 'standard', 'differential')
            redaction_policy: PII redaction policy
            requests_per_second: Quota limit (RPS)
        """
        self.agents = agents
        self.target_rps = target_rps
        self.duration = duration
        self.cohort = cohort
        self.storage_dir = storage_dir
        self.region_id = region_id
        self.logical_domain = logical_domain

        # Initialize governance system
        self.gov = IntegratedGovernance(
            storage_dir=storage_dir,
            region_id=region_id,
            logical_domain=logical_domain,
            enable_shadow_mode=enable_shadow,
            enable_ml_blending=enable_ml_blend,
            enable_anomaly_detection=enable_anomaly,
            enable_merkle_anchoring=enable_merkle,
            enable_quota_enforcement=enable_quota,
            privacy_mode=privacy_mode,
            redaction_policy=redaction_policy,
            requests_per_second=requests_per_second,
            # Performance optimization
            enable_performance_optimization=True,
            # Reduce escalations for load testing
            auto_escalate_on_block=False,
            auto_escalate_on_low_confidence=False,
        )

        # Results storage
        self.results: List[Dict[str, Any]] = []

    def generate_action(self, agent_id: str, action_num: int) -> Dict[str, Any]:
        """Generate a single action.

        Returns:
            Dict with timing and result information
        """
        start_time = time.time()
        action_id = f"{agent_id}_action_{action_num}"

        # Vary actions to create realistic mix
        action_types = [
            "User request processing",
            "Database query execution",
            "API call to external service",
            "Content generation",
            "Data analysis task",
        ]
        action_text = action_types[action_num % len(action_types)]

        # Add some PII for redaction testing
        if action_num % 10 == 0:
            action_text += " Contact: user@example.com"

        # Simulate violations (10% rate)
        violation_detected = action_num % 10 == 0
        violation_type = "safety" if violation_detected else None
        violation_severity = "medium" if violation_detected else None

        # Simulate ML features
        ml_score = 0.3 if action_num % 10 == 0 else 0.1
        rule_risk_score = 0.5 if violation_detected else 0.2

        try:
            result = self.gov.process_action(  # noqa: F841
                agent_id=agent_id,
                action=action_text,
                cohort=self.cohort,
                violation_detected=violation_detected,
                violation_type=violation_type,
                violation_severity=violation_severity,
                action_id=action_id,
                action_type="response",
                features={"ml_score": ml_score},
                rule_risk_score=rule_risk_score,
                rule_classification="warn" if violation_detected else "pass",
            )
            error = None
            status = "success"
        except Exception as e:
            error = str(e)
            status = "error"

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return {
            "agent_id": agent_id,
            "action_id": action_id,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "violation_detected": violation_detected,
        }

    def run_agent_workload(
        self, agent_id: str, actions_per_agent: int, delay_between_actions: float
    ) -> List[Dict[str, Any]]:
        """Run workload for a single agent.

        Args:
            agent_id: Agent identifier
            actions_per_agent: Number of actions to generate
            delay_between_actions: Delay in seconds between actions

        Returns:
            List of result dictionaries
        """
        results = []
        for i in range(actions_per_agent):
            result = self.generate_action(agent_id, i)
            results.append(result)

            # Sleep to maintain target rate
            if i < actions_per_agent - 1:
                time.sleep(delay_between_actions)

        return results

    def run(self) -> Dict[str, Any]:
        """Run the load test.

        Returns:
            Summary statistics
        """
        print("Starting load test:")
        print(f"  Agents: {self.agents}")
        print(f"  Target RPS: {self.target_rps}")
        print(f"  Duration: {self.duration}s")
        print(f"  Cohort: {self.cohort}")
        print(f"  Storage: {self.storage_dir}")
        print()

        # Calculate per-agent rate
        total_actions = int(self.target_rps * self.duration)
        actions_per_agent = total_actions // self.agents

        # Calculate delay between actions for each agent
        # Each agent produces actions_per_agent over duration seconds
        delay_between_actions = (
            self.duration / actions_per_agent if actions_per_agent > 0 else 0
        )

        print("Calculated parameters:")
        print(f"  Total actions: {total_actions}")
        print(f"  Actions per agent: {actions_per_agent}")
        print(f"  Delay between actions: {delay_between_actions:.3f}s")
        print()

        # Run workload using thread pool
        start_time = time.time()
        all_results = []

        with ThreadPoolExecutor(max_workers=min(self.agents, 50)) as executor:
            futures = []
            for i in range(self.agents):
                agent_id = f"agent_{i:04d}"
                future = executor.submit(
                    self.run_agent_workload,
                    agent_id,
                    actions_per_agent,
                    delay_between_actions,
                )
                futures.append(future)

            # Collect results
            completed = 0
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                    completed += 1
                    if completed % max(1, self.agents // 10) == 0:
                        print(f"Progress: {completed}/{self.agents} agents completed")
                except Exception as e:
                    print(f"Error in agent workload: {e}")

        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate statistics
        self.results = all_results
        stats = self._calculate_stats(elapsed)

        return stats

    def _calculate_stats(self, elapsed: float) -> Dict[str, Any]:
        """Calculate summary statistics.

        Args:
            elapsed: Elapsed time in seconds

        Returns:
            Dictionary with statistics
        """
        if not self.results:
            return {
                "error": "No results collected",
                "elapsed": elapsed,
            }

        latencies = [r["latency_ms"] for r in self.results]
        latencies.sort()

        successes = sum(1 for r in self.results if r["status"] == "success")
        errors = sum(1 for r in self.results if r["status"] == "error")
        violations = sum(1 for r in self.results if r.get("violation_detected", False))

        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f < len(data) - 1 else f
            return data[f] + (data[c] - data[f]) * (k - f)

        return {
            "elapsed_seconds": elapsed,
            "total_actions": len(self.results),
            "target_rps": self.target_rps,
            "achieved_rps": len(self.results) / elapsed,
            "successes": successes,
            "errors": errors,
            "error_rate": errors / len(self.results) if self.results else 0,
            "violations_detected": violations,
            "latency_min_ms": min(latencies),
            "latency_max_ms": max(latencies),
            "latency_mean_ms": sum(latencies) / len(latencies),
            "latency_p50_ms": percentile(latencies, 0.5),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_p99_ms": percentile(latencies, 0.99),
        }

    def write_csv(self, filename: str):
        """Write results to CSV file.

        Args:
            filename: Output CSV filename
        """
        if not self.results:
            print(f"No results to write to {filename}")
            return

        with open(filename, "w", newline="") as f:
            fieldnames = [
                "agent_id",
                "action_id",
                "timestamp",
                "latency_ms",
                "status",
                "error",
                "violation_detected",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\nResults written to: {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load generator for Nethical governance system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Load parameters
    parser.add_argument(
        "--agents",
        type=int,
        default=100,
        help="Number of agents to simulate",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=50.0,
        help="Target aggregate requests per second",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default="load-test",
        help="Cohort identifier",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="./nethical_perf",
        help="Storage directory for governance data",
    )
    parser.add_argument(
        "--region-id",
        type=str,
        default=None,
        help="Region identifier (e.g., us-east-1)",
    )
    parser.add_argument(
        "--logical-domain",
        type=str,
        default=None,
        help="Logical domain (e.g., customer-service)",
    )

    # Feature flags
    parser.add_argument(
        "--shadow",
        dest="enable_shadow",
        action="store_true",
        default=True,
        help="Enable ML shadow mode",
    )
    parser.add_argument(
        "--no-shadow",
        dest="enable_shadow",
        action="store_false",
        help="Disable ML shadow mode",
    )
    parser.add_argument(
        "--ml-blend",
        dest="enable_ml_blend",
        action="store_true",
        default=False,
        help="Enable ML blended enforcement",
    )
    parser.add_argument(
        "--no-ml-blend",
        dest="enable_ml_blend",
        action="store_false",
        help="Disable ML blended enforcement",
    )
    parser.add_argument(
        "--anomaly",
        dest="enable_anomaly",
        action="store_true",
        default=False,
        help="Enable anomaly detection",
    )
    parser.add_argument(
        "--no-anomaly",
        dest="enable_anomaly",
        action="store_false",
        help="Disable anomaly detection",
    )
    parser.add_argument(
        "--merkle",
        dest="enable_merkle",
        action="store_true",
        default=True,
        help="Enable Merkle anchoring",
    )
    parser.add_argument(
        "--no-merkle",
        dest="enable_merkle",
        action="store_false",
        help="Disable Merkle anchoring",
    )
    parser.add_argument(
        "--quota",
        dest="enable_quota",
        action="store_true",
        default=False,
        help="Enable quota enforcement",
    )
    parser.add_argument(
        "--no-quota",
        dest="enable_quota",
        action="store_false",
        help="Disable quota enforcement",
    )

    # Privacy & redaction
    parser.add_argument(
        "--privacy-mode",
        type=str,
        choices=["standard", "differential", "none"],
        default="none",
        help="Privacy mode",
    )
    parser.add_argument(
        "--redaction-policy",
        type=str,
        choices=["minimal", "standard", "aggressive"],
        default="standard",
        help="PII redaction policy",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=1000.0,
        help="Quota limit (requests per second)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="perf_results.csv",
        help="Output CSV filename",
    )

    args = parser.parse_args()

    # Normalize privacy mode
    privacy_mode = None if args.privacy_mode == "none" else args.privacy_mode

    # Create storage directory
    Path(args.storage_dir).mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = LoadGenerator(
        agents=args.agents,
        target_rps=args.rps,
        duration=args.duration,
        cohort=args.cohort,
        storage_dir=args.storage_dir,
        region_id=args.region_id,
        logical_domain=args.logical_domain,
        enable_shadow=args.enable_shadow,
        enable_ml_blend=args.enable_ml_blend,
        enable_anomaly=args.enable_anomaly,
        enable_merkle=args.enable_merkle,
        enable_quota=args.enable_quota,
        privacy_mode=privacy_mode,
        redaction_policy=args.redaction_policy,
        requests_per_second=args.requests_per_second,
    )

    # Run test
    stats = generator.run()

    # Print results
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return 1

    print(f"Elapsed Time:        {stats['elapsed_seconds']:.2f}s")
    print(f"Total Actions:       {stats['total_actions']}")
    print(f"Target RPS:          {stats['target_rps']:.2f}")
    print(f"Achieved RPS:        {stats['achieved_rps']:.2f}")
    print(f"Successes:           {stats['successes']}")
    print(f"Errors:              {stats['errors']}")
    print(f"Error Rate:          {stats['error_rate']:.2%}")
    print(f"Violations Detected: {stats['violations_detected']}")
    print()
    print("Latency (ms):")
    print(f"  Min:     {stats['latency_min_ms']:.2f}")
    print(f"  Mean:    {stats['latency_mean_ms']:.2f}")
    print(f"  p50:     {stats['latency_p50_ms']:.2f}")
    print(f"  p95:     {stats['latency_p95_ms']:.2f}")
    print(f"  p99:     {stats['latency_p99_ms']:.2f}")
    print(f"  Max:     {stats['latency_max_ms']:.2f}")
    print()

    # Check SLOs
    print("SLO Compliance:")
    p95_ok = stats["latency_p95_ms"] < 200
    p99_ok = stats["latency_p99_ms"] < 500
    rps_ok = (
        abs(stats["achieved_rps"] - stats["target_rps"]) / stats["target_rps"]
        < 0.1
    )
    error_ok = stats["error_rate"] < 0.01

    print(
        f"  p95 < 200ms:  {'✓ PASS' if p95_ok else '✗ FAIL'} "
        f"({stats['latency_p95_ms']:.2f}ms)"
    )
    print(
        f"  p99 < 500ms:  {'✓ PASS' if p99_ok else '✗ FAIL'} "
        f"({stats['latency_p99_ms']:.2f}ms)"
    )
    print(f"  RPS within 10%: {'✓ PASS' if rps_ok else '✗ FAIL'}")
    print(f"  Error rate < 1%: {'✓ PASS' if error_ok else '✗ FAIL'}")

    # Write CSV
    generator.write_csv(args.output)

    return 0 if all([p95_ok, p99_ok, rps_ok, error_ok]) else 1


if __name__ == "__main__":
    sys.exit(main())
