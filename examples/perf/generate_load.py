#!/usr/bin/env python3
"""Load Generator for Nethical Governance System (hardened).

This script simulates multiple AI agents generating actions at a target aggregate
RPS (requests per second) to validate throughput and latency SLOs.

Key improvements and hardening:
- Robust timing using high-resolution monotonic clock to avoid clock skew.
- Fair per-agent action distribution to meet target RPS even when agents > total actions.
- Optional thread start staggering to avoid thundering herd.
- Graceful shutdown on SIGINT/SIGTERM with partial results preserved.
- CSV injection prevention (sanitizes string fields beginning with =,+,-,@ or tab).
- Optional suppression of detailed exception messages (to avoid leaking sensitive data).
- Optional toggle for adding synthetic PII to actions (disabled by default).
- Input validation and safer concurrency defaults based on CPU.
- Reproducibility via optional RNG seed.

Usage examples:
  # Basic test: 100 agents at 50 RPS for 60 seconds
  python examples/perf/generate_load.py --agents 100 --rps 50 --duration 60

  # CI-lean test
  python examples/perf/generate_load.py --agents 50 --rps 15 --duration 60

  # Hardened defaults (no PII in actions, error details suppressed)
  python examples/perf/generate_load.py --agents 100 --rps 50 --duration 60

  # With synthetic PII (for redaction tests)
  python examples/perf/generate_load.py --agents 200 --rps 100 --duration 60 --include-pii-test
"""

import argparse
import csv
import logging
import math
import os
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


# Structured logger
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("nethical.perf.generate_load")

# Ensure nethical is importable
try:
    from nethical.core import IntegratedGovernance
except ImportError:
    print(
        "Error: nethical package not found. Install with: pip install -e .",
        file=sys.stderr,
    )
    sys.exit(1)


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_csv_value(value: Any) -> Any:
    """Neutralize CSV formula injection and strip newlines for Excel safety."""
    if isinstance(value, str):
        v = value.replace("\r", " ").replace("\n", " ").strip()
        if v and v[0] in ("=", "+", "-", "@", "\t"):
            return "'" + v
        return v
    return value


def validate_positive(name: str, val: float) -> None:
    if val is None or val <= 0:
        raise ValueError(f"{name} must be > 0, got {val}")


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
        # Hardening and control
        include_pii_test: bool = False,
        include_error_details: bool = False,
        stagger_ms: int = 0,
        rng_seed: Optional[int] = None,
        max_workers: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        """Initialize load generator."""
        # Validation
        validate_positive("agents", float(agents))
        validate_positive("target_rps", float(target_rps))
        validate_positive("duration", float(duration))

        self.agents = int(agents)
        self.target_rps = float(target_rps)
        self.duration = int(duration)
        self.cohort = cohort
        self.storage_dir = storage_dir
        self.region_id = region_id
        self.logical_domain = logical_domain

        self.include_pii_test = bool(include_pii_test)
        self.include_error_details = bool(include_error_details)
        self.stagger_ms = max(0, int(stagger_ms))
        self.rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        self.stop_event = stop_event or threading.Event()

        # Determine sane worker count
        cpu = os.cpu_count() or 4
        default_workers = min(max(4, cpu * 5), max(4, self.agents))
        self.max_workers = (
            int(max_workers) if max_workers and max_workers > 0 else default_workers
        )

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

    def _action_text(self, action_num: int) -> str:
        """Build action text; optionally include synthetic PII for redaction tests."""
        action_types = [
            "User request processing",
            "Database query execution",
            "API call to external service",
            "Content generation",
            "Data analysis task",
        ]
        text = action_types[action_num % len(action_types)]
        if self.include_pii_test and action_num % 10 == 0:
            # Synthetic, non-sensitive PII-like token for redaction tests only
            text += " Contact: test.user+load@example.invalid"
        return text

    def _violations_for(
        self, action_num: int
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        violation_detected = action_num % 10 == 0
        violation_type = "safety" if violation_detected else None
        violation_severity = "medium" if violation_detected else None
        return violation_detected, violation_type, violation_severity

    def generate_action(self, agent_id: str, action_num: int) -> Dict[str, Any]:
        """Generate a single action and return result record."""
        t_start = time.perf_counter()
        action_id = f"{agent_id}_action_{action_num}"

        action_text = self._action_text(action_num)
        violation_detected, violation_type, violation_severity = self._violations_for(
            action_num
        )

        # Simulated ML/rule scoring patterns (deterministic)
        ml_score = 0.3 if violation_detected else 0.1
        rule_risk_score = 0.5 if violation_detected else 0.2

        error: Optional[str] = None
        status = "success"
        try:
            _result = self.gov.process_action(  # noqa: F841
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
        except Exception as e:
            status = "error"
            # Avoid leaking sensitive exception detail by default
            error = (
                f"{e.__class__.__name__}: {str(e)}"
                if self.include_error_details
                else e.__class__.__name__
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return {
            "agent_id": agent_id,
            "action_id": action_id,
            "timestamp": now_iso_utc(),
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "violation_detected": violation_detected,
        }

    def run_agent_workload(
        self,
        agent_id: str,
        actions_for_agent: int,
        start_time: float,
        interval_s: float,
        agent_stagger_s: float,
    ) -> List[Dict[str, Any]]:
        """Run workload for a single agent with rate pacing.

        Pacing model:
        - All agents target 'interval_s' between actions to hit the overall RPS.
        - Each agent's first action is delayed by 'agent_stagger_s' * agent_index to avoid a thundering herd.
        - Uses perf_counter for robust sleep scheduling.
        """
        if actions_for_agent <= 0:
            return []

        results: List[Dict[str, Any]] = []
        # Initial agent-specific stagger
        first_fire = start_time + agent_stagger_s

        for i in range(actions_for_agent):
            if self.stop_event.is_set():
                break

            # Schedule next fire time
            next_fire = first_fire + (i * interval_s)

            # Sleep until next_fire
            now = time.perf_counter()
            if next_fire > now:
                time.sleep(next_fire - now)

            # Execute action
            try:
                result = self.generate_action(agent_id, i)
                results.append(result)
            except Exception as e:
                # Shouldn't happen (generate_action handles exceptions), but guard anyway
                log.error("Unhandled agent workload error: %s", e)
        return results

    def _distribute_actions(self, total_actions: int) -> List[int]:
        """Distribute total actions across agents as evenly as possible."""
        base = total_actions // self.agents
        rem = total_actions % self.agents
        # First 'rem' agents get one extra action
        plan = [base + 1 if i < rem else base for i in range(self.agents)]
        return plan

    def run(self) -> Dict[str, Any]:
        """Run the load test and return summary statistics."""
        log.info(
            "Starting load test: agents=%d target_rps=%.2f duration=%ds cohort=%s storage=%s",
            self.agents,
            self.target_rps,
            self.duration,
            self.cohort,
            self.storage_dir,
        )

        # Calculate total actions and per-agent distribution
        total_actions = int(math.ceil(self.target_rps * self.duration))
        if total_actions <= 0:
            return {
                "error": "Computed total_actions <= 0; adjust parameters",
                "elapsed": 0.0,
            }

        per_agent_actions = self._distribute_actions(total_actions)
        # Effective interval per action across the whole system
        interval_s = 1.0 / self.target_rps if self.target_rps > 0 else 0.0

        # Shared start (perf_counter), allowing precise pacing
        global_start = time.perf_counter()

        # Agent staggering in seconds
        agent_stagger_s_base = self.stagger_ms / 1000.0

        all_results: List[Dict[str, Any]] = []

        max_workers = min(self.max_workers, self.agents)
        log.info("Thread pool: max_workers=%d (cpu=%s)", max_workers, os.cpu_count())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(self.agents):
                agent_id = f"agent_{i:04d}"
                # Stagger proportional to index, but cap to 1 second to avoid excessive delays
                stagger = min(agent_stagger_s_base * i, 1.0)
                fut = executor.submit(
                    self.run_agent_workload,
                    agent_id,
                    per_agent_actions[i],
                    global_start,
                    interval_s,
                    stagger,
                )
                futures.append(fut)

            completed = 0
            for fut in as_completed(futures):
                try:
                    results = fut.result()
                    all_results.extend(results)
                    completed += 1
                    if completed % max(1, self.agents // 10) == 0:
                        log.info(
                            "Progress: %d/%d agents completed", completed, self.agents
                        )
                except Exception as e:
                    log.error("Error in agent workload future: %s", e)

        elapsed = max(1e-6, time.perf_counter() - global_start)  # Avoid div-by-zero

        # Calculate statistics
        self.results = all_results
        stats = self._calculate_stats(elapsed)

        return stats

    def _calculate_stats(self, elapsed: float) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.results:
            return {"error": "No results collected", "elapsed": elapsed}

        latencies = [float(r["latency_ms"]) for r in self.results]
        latencies.sort()

        successes = sum(1 for r in self.results if r["status"] == "success")
        errors = sum(1 for r in self.results if r["status"] == "error")
        violations = sum(
            1 for r in self.results if bool(r.get("violation_detected", False))
        )

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = math.floor(k)
            c = min(f + 1, len(data) - 1)
            return float(data[f] + (data[c] - data[f]) * (k - f))

        achieved_rps = len(self.results) / elapsed if elapsed > 0 else 0.0

        return {
            "elapsed_seconds": elapsed,
            "total_actions": len(self.results),
            "target_rps": self.target_rps,
            "achieved_rps": achieved_rps,
            "successes": successes,
            "errors": errors,
            "error_rate": (errors / len(self.results)) if self.results else 0.0,
            "violations_detected": violations,
            "latency_min_ms": min(latencies),
            "latency_max_ms": max(latencies),
            "latency_mean_ms": sum(latencies) / len(latencies),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_p99_ms": percentile(latencies, 0.99),
        }

    def write_csv(self, filename: str):
        """Write results to CSV file (with CSV injection protection)."""
        if not self.results:
            log.warning("No results to write to %s", filename)
            return

        fieldnames = [
            "agent_id",
            "action_id",
            "timestamp",
            "latency_ms",
            "status",
            "error",
            "violation_detected",
        ]

        tmpfile = f"{filename}.tmp"
        with open(tmpfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                safe_row = {k: sanitize_csv_value(v) for k, v in row.items()}
                writer.writerow(safe_row)

        # Atomic replace
        os.replace(tmpfile, filename)
        log.info("Results written to: %s", filename)


def _install_signal_handlers(stop_event: threading.Event):
    def _handler(signum, _frame):
        log.warning("Received signal %s - initiating graceful shutdown", signum)
        stop_event.set()

    # Register for SIGINT/SIGTERM
    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load generator for Nethical governance system (hardened)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Load parameters
    parser.add_argument(
        "--agents", type=int, default=100, help="Number of agents to simulate"
    )
    parser.add_argument(
        "--rps", type=float, default=50.0, help="Target aggregate requests per second"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument(
        "--cohort", type=str, default="load-test", help="Cohort identifier"
    )
    parser.add_argument(
        "--storage-dir", type=str, default="./nethical_perf", help="Storage directory"
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
        "--output", type=str, default="perf_results.csv", help="Output CSV filename"
    )

    # Hardened controls
    parser.add_argument(
        "--include-pii-test",
        action="store_true",
        default=False,
        help="Include synthetic PII tokens in actions (for redaction testing)",
    )
    parser.add_argument(
        "--include-error-details",
        action="store_true",
        default=False,
        help="Include exception messages in CSV (may leak details)",
    )
    parser.add_argument(
        "--stagger-ms",
        type=int,
        default=0,
        help="Per-agent start staggering in milliseconds",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max thread workers (defaults to ~5xCPU, capped by agents)",
    )

    args = parser.parse_args()

    # Normalize privacy mode
    privacy_mode = None if args.privacy_mode == "none" else args.privacy_mode

    # Create storage directory
    Path(args.storage_dir).mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

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
        include_pii_test=args.include_pii_test,
        include_error_details=args.include_error_details,
        stagger_ms=args.stagger_ms,
        rng_seed=args.seed,
        max_workers=args.max_workers,
        stop_event=stop_event,
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

    # SLO gates (example values)
    print("SLO Compliance:")
    p95_ok = stats["latency_p95_ms"] < 200
    p99_ok = stats["latency_p99_ms"] < 500
    rps_ok = (
        abs(stats["achieved_rps"] - stats["target_rps"]) / stats["target_rps"] < 0.1
        if stats["target_rps"] > 0
        else False
    )
    error_ok = stats["error_rate"] < 0.01

    print(
        f"  p95 < 200ms:  {'✓ PASS' if p95_ok else '✗ FAIL'} ({stats['latency_p95_ms']:.2f}ms)"
    )
    print(
        f"  p99 < 500ms:  {'✓ PASS' if p99_ok else '✗ FAIL'} ({stats['latency_p99_ms']:.2f}ms)"
    )
    print(f"  RPS within 10%: {'✓ PASS' if rps_ok else '✗ FAIL'}")
    print(f"  Error rate < 1%: {'✓ PASS' if error_ok else '✗ FAIL'}")

    # Write CSV
    try:
        generator.write_csv(args.output)
    except Exception as e:
        log.error("Failed to write CSV '%s': %s", args.output, e)

    # Non-zero exit when SLOs not met, which can be used for regression gating if desired
    return 0 if all([p95_ok, p99_ok, rps_ok, error_ok]) else 1


if __name__ == "__main__":
    sys.exit(main())
