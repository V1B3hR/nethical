"""Automated test suite for Nethical CLI tools.

Covers:
- cli/review_queue
- cli/policy_diff
- cli/policy_simulator
"""

import sys
import json
import yaml
import pytest
import subprocess
from pathlib import Path
from typing import Tuple

from nethical.core import (
    EscalationQueue,
    ReviewPriority,
)
from nethical.core.human_feedback import EscalationCase


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run CLI tool with UTF-8 decoding and return CompletedProcess."""
    return subprocess.run(
        [sys.executable] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )


@pytest.fixture
def temp_cli_db(tmp_path: Path) -> str:
    """Provide temporary database path for review_queue testing."""
    db_file = tmp_path / "escalations_test.db"
    return str(db_file)


@pytest.fixture
def populated_queue(temp_cli_db: str) -> Tuple[EscalationCase, str]:
    """Provide a queue pre-populated with a test escalation case."""
    queue = EscalationQueue(storage_path=temp_cli_db)
    case = queue.add_case(
        judgment_id="jdg_001",
        action_id="act_001",
        agent_id="agent_alpha",
        decision="quarantine",
        confidence=0.88,
        violations=[
            {"type": "harm_detection", "severity": 4, "description": "Potential harm detected"}
        ],
        priority=ReviewPriority.HIGH
    )
    return case, temp_cli_db


def test_review_queue_list_and_subcommand_flags(populated_queue: Tuple[EscalationCase, str]) -> None:
    """Test review_queue list command with flag both before and after subcommand."""
    case, db_path = populated_queue
    cli_path = str(Path("cli/review_queue").resolve())

    # Test flag after subcommand: list --db-path ...
    res1 = run_cli(cli_path, "list", "--db-path", db_path)
    assert res1.returncode == 0
    assert "Pending Cases (1):" in res1.stdout
    assert case.case_id in res1.stdout
    assert "quarantine" in res1.stdout

    # Test flag before subcommand: --db-path ... list
    res2 = run_cli(cli_path, "--db-path", db_path, "list")
    assert res2.returncode == 0
    assert case.case_id in res2.stdout


def test_review_queue_next_and_feedback_flow(populated_queue: Tuple[EscalationCase, str]) -> None:
    """Test getting next case, submitting feedback, and viewing persistent stats."""
    case, db_path = populated_queue
    cli_path = str(Path("cli/review_queue").resolve())

    # Next case
    res_next = run_cli(cli_path, "next", "reviewer_bob", "--db-path", db_path)
    assert res_next.returncode == 0
    assert "Next Case for Review:" in res_next.stdout
    assert case.case_id in res_next.stdout

    # Submit valid feedback
    res_fb = run_cli(
        cli_path, "feedback", case.case_id, "reviewer_bob",
        "--tags", "false_positive",
        "--rationale", "Legitimate research query",
        "--corrected-decision", "allow",
        "--confidence", "0.95",
        "--db-path", db_path
    )
    assert res_fb.returncode == 0
    assert "Feedback Submitted Successfully!" in res_fb.stdout
    assert "false_positive" in res_fb.stdout

    # Verify stats reflect the completed case
    res_stats = run_cli(cli_path, "stats", "--db-path", db_path)
    assert res_stats.returncode == 0
    assert "Total Cases: 1" in res_stats.stdout
    assert "Completed Cases: 1" in res_stats.stdout

    # Verify summary
    res_sum = run_cli(cli_path, "summary", "--db-path", db_path)
    assert res_sum.returncode == 0
    assert "Total Feedback: 1" in res_sum.stdout


def test_review_queue_error_exit_codes(temp_cli_db: str) -> None:
    """Test review_queue returns exit code 1 on invalid inputs."""
    cli_path = str(Path("cli/review_queue").resolve())

    # Case not found
    res1 = run_cli(
        cli_path, "feedback", "non_existent_case", "reviewer_bob",
        "--tags", "false_positive",
        "--rationale", "Test rationale",
        "--db-path", temp_cli_db
    )
    assert res1.returncode == 1
    assert "not found" in res1.stderr

    # Invalid confidence bounds (> 1.0)
    res2 = run_cli(
        cli_path, "feedback", "case_001", "reviewer_bob",
        "--tags", "false_positive",
        "--rationale", "Test rationale",
        "--confidence", "2.0",
        "--db-path", temp_cli_db
    )
    assert res2.returncode == 1
    assert "Confidence must be between 0.0 and 1.0" in res2.stderr


def test_policy_diff_flow(tmp_path: Path) -> None:
    """Test policy_diff execution across text, json, and markdown formats."""
    cli_path = str(Path("cli/policy_diff").resolve())

    policy_v1 = {
        "version": "1.0",
        "rules": [
            {"id": "R01", "action": "allow", "scope": "public"}
        ]
    }
    policy_v2 = {
        "version": "2.0",
        "rules": [
            {"id": "R01", "action": "allow", "scope": "public"},
            {"id": "R02", "action": "deny", "scope": "admin"}
        ]
    }

    f1 = tmp_path / "policy_v1.yaml"
    f2 = tmp_path / "policy_v2.yaml"
    f1.write_text(yaml.dump(policy_v1), encoding="utf-8")
    f2.write_text(yaml.dump(policy_v2), encoding="utf-8")

    # JSON format output to file
    out_json = tmp_path / "diff.json"
    res_json = run_cli(
        cli_path,
        str(f1), str(f2),
        "--format", "json",
        "--output", str(out_json),
        "--storage-dir", str(tmp_path / "history")
    )
    assert res_json.returncode in (0, 1, 2)
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "risk_score" in data
    assert "summary" in data

    # Markdown format to stdout
    res_md = run_cli(
        cli_path,
        str(f1), str(f2),
        "--format", "markdown",
        "--storage-dir", str(tmp_path / "history")
    )
    assert res_md.returncode in (0, 1, 2)
    assert "# Policy Diff:" in res_md.stdout


def test_policy_diff_missing_file() -> None:
    """Test policy_diff exits with code 3 when policy file is missing."""
    cli_path = str(Path("cli/policy_diff").resolve())
    res = run_cli(cli_path, "missing_v1.yaml", "missing_v2.yaml")
    assert res.returncode == 3
    assert "Policy file not found" in res.stderr


def test_policy_simulator_simulate_and_dryrun(tmp_path: Path) -> None:
    """Test policy_simulator simulate and dry-run subcommands."""
    cli_path = str(Path("cli/policy_simulator").resolve())

    # Valid policy file (PolicyEngine compliant)
    policy_data = {
        "name": "Safety Baseline",
        "version": "1.0",
        "rules": [
            {
                "id": "SAFE_DEFAULT",
                "condition": "true",
                "decision": "ALLOW"
            }
        ]
    }
    test_cases_data = {
        "test_cases": [
            {
                "name": "Basic Allow Scenario",
                "input": {"prompt": "Hello world"},
                "expected_decision": "ALLOW"
            }
        ]
    }

    pol_file = tmp_path / "policy.yaml"
    tc_file = tmp_path / "test_cases.json"
    pol_file.write_text(yaml.dump(policy_data), encoding="utf-8")
    tc_file.write_text(json.dumps(test_cases_data), encoding="utf-8")

    # Simulate
    res_sim = run_cli(
        cli_path, "simulate",
        str(pol_file), str(tc_file),
        "--format", "json"
    )
    assert res_sim.returncode == 0
    sim_data = json.loads(res_sim.stdout)
    assert sim_data["total_cases"] == 1
    assert "timestamp" in sim_data

    # Dry-run
    res_dry = run_cli(
        cli_path, "dry-run",
        str(pol_file), str(pol_file), str(tc_file),
        "--format", "text"
    )
    assert res_dry.returncode == 0
    assert "POLICY DRY-RUN DIFF" in res_dry.stdout
    assert "Unchanged: 1" in res_dry.stdout
