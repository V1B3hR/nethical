"""
Unit and Integration Tests for Formal Verification Subsystem (formal/)

Tests:
1. Z3 SMT Solver PolicyVerifier (non-contradiction, determinism, fairness bounds, safety state reachability).
2. FundamentalLawsVerifier across all 25 Fundamental Laws of AI governance.
3. Integrity and existence of formal TLA+ and Lean 4 specifications.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from formal.z3 import (
    PolicyVerifier,
    FundamentalLawsVerifier,
    VerificationResult,
    VerificationReport,
)

FORMAL_DIR = Path("formal")


class TestPolicyVerifier:
    """Verifies Z3-based formal properties of governance policies."""

    def test_z3_solver_availability(self) -> None:
        verifier = PolicyVerifier()
        assert verifier.is_available is True

    def test_non_contradictory_policies_valid(self) -> None:
        verifier = PolicyVerifier()
        policies = [
            {"id": "p1", "action_type": "file_read", "decision": "ALLOW", "priority": 10},
            {"id": "p2", "action_type": "file_delete", "decision": "BLOCK", "priority": 10},
            {"id": "p3", "action_type": "network_egress", "decision": "RESTRICT", "priority": 5},
        ]
        report = verifier.verify_policy_non_contradiction(policies)
        assert report.result == VerificationResult.VALID
        assert "consistent" in report.details.lower()

    def test_contradictory_policies_invalid(self) -> None:
        verifier = PolicyVerifier()
        # Same action_type, same priority, one ALLOW and one BLOCK
        policies = [
            {"id": "p1", "action_type": "file_delete", "decision": "ALLOW", "priority": 10},
            {"id": "p2", "action_type": "file_delete", "decision": "BLOCK", "priority": 10},
        ]
        report = verifier.verify_policy_non_contradiction(policies)
        assert report.result == VerificationResult.INVALID
        assert "contradict" in report.details.lower()
        assert report.counterexample is not None

    def test_decision_determinism(self) -> None:
        verifier = PolicyVerifier()
        policy = {"id": "det_policy", "risk_threshold": 0.65}
        report = verifier.verify_decision_determinism(policy)
        assert report.result == VerificationResult.VALID

    def test_fairness_bounds_verification(self) -> None:
        verifier = PolicyVerifier()
        fair_policy = {"protected_rate": 0.85, "unprotected_rate": 0.90}
        report = verifier.verify_fairness_bounds(fair_policy)
        assert report.result == VerificationResult.VALID

    def test_state_machine_safety(self) -> None:
        verifier = PolicyVerifier()
        safe_sm = {
            "initial_state": "idle",
            "states": ["idle", "processing", "governed_done"],
            "unsafe_states": ["unauthorized_override", "deadlock"],
            "transitions": [
                {"from": "idle", "to": "processing"},
                {"from": "processing", "to": "governed_done"},
            ]
        }
        report = verifier.verify_no_unsafe_states(safe_sm)
        assert report.result == VerificationResult.VALID

    def test_run_all_checks_and_summary(self) -> None:
        verifier = PolicyVerifier()
        policies = [
            {"id": "p1", "action_type": "query", "decision": "ALLOW", "priority": 1, "risk_threshold": 0.5},
            {"id": "p2", "action_type": "update", "decision": "RESTRICT", "priority": 2, "risk_threshold": 0.8},
        ]
        results = verifier.run_all_checks(policies)
        assert len(results) >= 5

        summary = verifier.get_summary()
        assert summary["total_checks"] == len(results)
        assert summary["valid"] > 0
        assert summary["errors"] == 0


class TestFundamentalLawsVerifier:
    """Verifies that policies comply with the 25 Fundamental Laws of Nethical."""

    def test_human_safety_priority_law_21(self) -> None:
        laws_verifier = FundamentalLawsVerifier()
        # Compliant policy
        rep_valid = laws_verifier.verify_law_compliance({"human_safety_priority": True}, 21)
        assert rep_valid.result == VerificationResult.VALID

        # Non-compliant policy
        rep_invalid = laws_verifier.verify_law_compliance({"human_safety_priority": False}, 21)
        assert rep_invalid.result == VerificationResult.INVALID

    def test_fail_safe_design_law_23(self) -> None:
        laws_verifier = FundamentalLawsVerifier()
        rep_valid = laws_verifier.verify_law_compliance({"failsafe_enabled": True}, 23)
        assert rep_valid.result == VerificationResult.VALID

        rep_invalid = laws_verifier.verify_law_compliance({"failsafe_enabled": False}, 23)
        assert rep_invalid.result == VerificationResult.INVALID

    def test_verify_all_25_laws(self) -> None:
        laws_verifier = FundamentalLawsVerifier()
        policy = {
            "human_safety_priority": True,
            "failsafe_enabled": True,
            "identity_preserved": True,
            "audit_enabled": True,
        }
        reports = laws_verifier.verify_all_laws(policy)
        assert len(reports) == 25
        # All reports must have a valid result status
        for r in reports:
            assert isinstance(r, VerificationReport)
            assert r.result in (VerificationResult.VALID, VerificationResult.INVALID)


class TestFormalSpecificationsIntegrity:
    """Validates presence and syntax headers of TLA+ and Lean formal specifications."""

    @pytest.mark.parametrize("tla_file", [
        "AuditIntegrity.tla",
        "EdgeDecision.tla",
        "FundamentalLaws.tla",
        "GovernanceStateMachine.tla",
        "NethicalGovernance.tla",
        "PolicyConsistency.tla",
        "PolicyEngine.tla",
    ])
    def test_tla_specifications_exist(self, tla_file: str) -> None:
        path = FORMAL_DIR / "tla" / tla_file
        assert path.exists(), f"Missing TLA+ spec: {tla_file}"
        content = path.read_text(encoding="utf-8")
        assert f"MODULE {path.stem}" in content or "---- MODULE" in content

    def test_lean4_specification_exists(self) -> None:
        lean_path = FORMAL_DIR / "lean" / "NethicalCore.lean"
        assert lean_path.exists(), "Missing Lean 4 specification file"
        content = lean_path.read_text(encoding="utf-8")
        assert len(content) > 100
