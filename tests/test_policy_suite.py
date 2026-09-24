# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Unit test suite for sovereign policy module: PolicyEngine & PolicyPack release management.
"""

import pytest

from nethical.hooks.interfaces import Region
from nethical.policy import (
    PolicyEngine,
    PolicyError,
    PolicyPack,
    PolicyVersion,
    CanaryConfig,
    DeploymentStage,
)


class TestPolicyEngine:
    """Test PolicyEngine DSL, operators, region overlays, and evaluation."""

    def test_basic_rule_evaluation_and_deny_overrides(self):
        rules = {
            "defaults": {"decision": "ALLOW", "deny_overrides": True},
            "rules": [
                {
                    "id": "rule_warn_tokens",
                    "conditions": ["action.tokens > 500"],
                    "actions": {"decision": "RESTRICT", "tags": ["high_token_usage"]},
                },
                {
                    "id": "rule_block_exfil",
                    "conditions": ["action.type == exfiltration"],
                    "actions": {"decision": "DENY", "tags": ["data_loss_prevention"]},
                },
            ],
        }
        engine = PolicyEngine(rules, Region.EU)

        # Case 1: benign action
        outcome_benign = engine.evaluate({"action": {"tokens": 100, "type": "query"}})
        assert outcome_benign["final_decision"] == "ALLOW"
        assert len(outcome_benign["matched_rules"]) == 0

        # Case 2: high token usage -> RESTRICT normalizes to DENY in decisions and final_decision
        outcome_restrict = engine.evaluate({"action": {"tokens": 800, "type": "query"}})
        assert outcome_restrict["decisions"] == ["DENY"]
        assert outcome_restrict["final_decision"] == "DENY"
        assert "high_token_usage" in outcome_restrict["tags"]

        # Case 3: both high token usage and exfiltration -> DENY overrides RESTRICT
        outcome_deny = engine.evaluate({"action": {"tokens": 800, "type": "exfiltration"}})
        assert outcome_deny["final_decision"] == "DENY"
        assert "data_loss_prevention" in outcome_deny["tags"]
        assert "high_token_usage" in outcome_deny["tags"]

    def test_nested_dsl_conditions_and_regex(self):
        rules = {
            "rules": [
                {
                    "id": "rule_pii_check",
                    "conditions": {
                        "any": [
                            {"endswith": ["user.email", "@gov.pl"]},
                            {"matches": ["payload.ssn", r"^\d{3}-\d{2}-\d{4}$"]},
                        ]
                    },
                    "actions": {"decision": "ALLOW", "tags": ["gov_pl"]},
                }
            ]
        }
        engine = PolicyEngine(rules, Region.EU)

        # Match regex
        out1 = engine.evaluate({"user": {"email": "user@corp.com"}, "payload": {"ssn": "123-45-6789"}})
        matched_1 = [r["id"] for r in out1["matched_rules"]]
        assert "rule_pii_check" in matched_1

        # Match endswith
        out2 = engine.evaluate({"user": {"email": "minister@gov.pl"}, "payload": {"ssn": "invalid"}})
        matched_2 = [r["id"] for r in out2["matched_rules"]]
        assert "rule_pii_check" in matched_2

        # Match neither
        out3 = engine.evaluate({"user": {"email": "user@corp.com"}, "payload": {"ssn": "invalid"}})
        matched_3 = [r["id"] for r in out3["matched_rules"]]
        assert "rule_pii_check" not in matched_3


class TestPolicyPackReleaseManagement:
    """Test PolicyPack versioning, canary deployment, persistence, and rollback."""

    def test_version_creation_and_checksum(self, tmp_path):
        pack = PolicyPack("safety_core", storage_dir=str(tmp_path / "packs"))
        content = {"rules": [{"id": "r1", "actions": {"decision": "ALLOW"}}]}

        v1 = pack.create_version(
            version="1.0.0",
            content=content,
            created_by="sovereign_admin",
            description="Initial baseline security policy",
        )

        assert v1.version == "1.0.0"
        assert len(v1.checksum) == 64
        assert pack.get_version("1.0.0") is not None

        # Re-create duplicate version raises error
        with pytest.raises(ValueError, match="already exists"):
            pack.create_version("1.0.0", content, "admin")

    def test_canary_deployment_and_promotion(self, tmp_path):
        pack = PolicyPack("safety_core", storage_dir=str(tmp_path / "packs"))
        content_v1 = {"v": 1}
        content_v2 = {"v": 2}

        pack.create_version("1.0.0", content_v1, "admin")
        pack.create_version("2.0.0", content_v2, "admin")

        # Promote 1.0.0 directly to production
        dep_prod = pack.promote_to_production("1.0.0")
        assert dep_prod.stage == DeploymentStage.PRODUCTION
        assert pack.get_production_version().version == "1.0.0"

        # Deploy 2.0.0 as canary
        dep_canary = pack.deploy_canary("2.0.0", canary_percentage=20.0)
        assert dep_canary.stage == DeploymentStage.CANARY
        assert pack.get_canary_version().version == "2.0.0"

        # Promote canary to production
        dep_promoted = pack.promote_to_production("2.0.0")
        assert dep_promoted.stage == DeploymentStage.PRODUCTION
        assert pack.get_production_version().version == "2.0.0"
        assert pack.get_canary_version() is None

    def test_rollback_procedure_and_disk_persistence(self, tmp_path):
        storage = str(tmp_path / "packs")
        pack = PolicyPack("finance_pack", storage_dir=storage)

        pack.create_version("1.0.0", {"rules": ["allow_all"]}, "admin")
        pack.create_version("1.1.0", {"rules": ["strict_filter"]}, "admin")
        pack.promote_to_production("1.0.0")
        pack.promote_to_production("1.1.0")

        # Rollback to 1.0.0
        rollback_dep = pack.rollback_to_version("1.0.0", reason="False positive spike in 1.1.0")
        assert rollback_dep.stage == DeploymentStage.ROLLBACK
        assert pack.get_production_version().version == "1.0.0"

        # Re-load pack from disk to test UTF-8 persistence and state integrity
        reloaded_pack = PolicyPack("finance_pack", storage_dir=storage)
        assert reloaded_pack.get_production_version().version == "1.0.0"
        assert len(reloaded_pack.list_versions()) == 2
