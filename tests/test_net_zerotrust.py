# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Unit test suite for sovereign zero-trust network policies: NoopCommsPolicy & MTLSCommsPolicy.
"""

import pytest

from nethical.net import NoopCommsPolicy, MTLSCommsPolicy


class TestNoopCommsPolicy:
    """Test permissive NoopCommsPolicy."""

    def test_noop_allows_all(self):
        policy = NoopCommsPolicy()
        assert policy.connection_allowed("any_peer", {}) is True
        assert policy.connection_allowed("attacker", {"malicious": True}) is True
        ident = policy.identities()
        assert ident["impl"] == "noop"
        assert ident["version"] == 1


class TestMTLSCommsPolicy:
    """Test mTLS + SPIFFE Zero-Trust Policy with layered verification."""

    def test_trust_domain_normalization_and_validation(self):
        policy1 = MTLSCommsPolicy(trust_domain="spiffe://sov.nethical.org/ns/prod")
        assert policy1.trust_domain == "sov.nethical.org"

        policy2 = MTLSCommsPolicy(trust_domain="  SOV.NETHICAL.ORG  ")
        assert policy2.trust_domain == "sov.nethical.org"

        with pytest.raises(ValueError, match="trust_domain must be a non-empty string"):
            MTLSCommsPolicy(trust_domain="")

    def test_deny_pattern_takes_precedence(self):
        policy = MTLSCommsPolicy(
            trust_domain="sov.nethical.org",
            allowed_identities=["node-*"],
            denied_identities=["node-untrusted-*"],
        )

        # Allowed pattern matches, but deny pattern matches -> DENY
        ctx = {"spiffe_id": "spiffe://sov.nethical.org/node-untrusted-1"}
        allowed, reasons = policy.evaluate("node-untrusted-1", ctx, explain=True)
        assert allowed is False
        assert any("DENY: Matched denied identity pattern" in r for r in reasons)

    def test_allow_pattern_matching(self):
        policy = MTLSCommsPolicy(
            trust_domain="sov.nethical.org",
            allowed_identities=["node-worker-*", "spiffe://sov.nethical.org/gateway"],
        )

        ctx_allowed = {"spiffe_id": "spiffe://sov.nethical.org/gateway"}
        assert policy.connection_allowed("gateway", ctx_allowed) is True

        ctx_denied = {"spiffe_id": "spiffe://sov.nethical.org/unauthorized"}
        allowed, reasons = policy.evaluate("unauthorized", ctx_denied, explain=True)
        assert allowed is False
        assert any("No candidate matched any allowed identity pattern" in r for r in reasons)

    def test_spiffe_trust_domain_verification(self):
        policy = MTLSCommsPolicy(trust_domain="sov.nethical.org")

        # Matching trust domain
        valid_ctx = {"spiffe_id": "spiffe://sov.nethical.org/workload/db"}
        assert policy.connection_allowed("db_worker", valid_ctx) is True

        # External / hostile trust domain
        rogue_ctx = {"spiffe_id": "spiffe://hostile.domain.net/workload/db"}
        allowed, reasons = policy.evaluate("db_worker", rogue_ctx, explain=True)
        assert allowed is False
        assert any("SPIFFE trust domain mismatch" in r for r in reasons)

        # Malformed SPIFFE ID
        malformed_ctx = {"spiffe_id": "spiffe://malformed_no_path"}
        allowed_m, reasons_m = policy.evaluate("bad_spiffe", malformed_ctx, explain=True)
        assert allowed_m is False
        assert any("Invalid SPIFFE ID format" in r for r in reasons_m)

    def test_attestation_requirements(self):
        policy = MTLSCommsPolicy(
            trust_domain="sov.nethical.org",
            require_device_attested=True,
            require_runtime_attested=True,
        )

        # Missing both
        allowed, reasons = policy.evaluate("peer_1", {}, explain=True)
        assert allowed is False
        assert any("Device attestation required" in r for r in reasons)

        # Device attested only
        allowed_d, reasons_d = policy.evaluate("peer_1", {"device_attested": True}, explain=True)
        assert allowed_d is False
        assert any("Runtime attestation required" in r for r in reasons_d)

        # Both attested
        ctx_ok = {"device_attested": True, "runtime_attested": True}
        assert policy.connection_allowed("peer_1", ctx_ok) is True

    def test_region_and_purpose_constraints(self):
        policy = MTLSCommsPolicy(
            trust_domain="sov.nethical.org",
            allowed_regions=["EU", "PL"],
            allowed_purposes=["audit", "inference"],
        )

        # Valid region and purpose
        valid_ctx = {"region": "PL", "purpose": "audit"}
        assert policy.connection_allowed("peer", valid_ctx) is True

        # Invalid region
        bad_region_ctx = {"region": "US", "purpose": "audit"}
        allowed_r, reasons_r = policy.evaluate("peer", bad_region_ctx, explain=True)
        assert allowed_r is False
        assert any("Region 'US' not in allowed regions" in r for r in reasons_r)

        # Invalid purpose
        bad_purpose_ctx = {"region": "PL", "purpose": "scraping"}
        allowed_p, reasons_p = policy.evaluate("peer", bad_purpose_ctx, explain=True)
        assert allowed_p is False
        assert any("Purpose 'scraping' not in allowed purposes" in r for r in reasons_p)

    def test_role_requirements_all_vs_any(self):
        # require_all_roles = True
        policy_all = MTLSCommsPolicy(
            trust_domain="sov.nethical.org",
            required_roles=["admin", "auditor"],
            require_all_roles=True,
        )

        ctx_subset = {"peer_roles": ["admin"]}
        allowed, reasons = policy_all.evaluate("peer", ctx_subset, explain=True)
        assert allowed is False
        assert any("Missing required roles (all-of)" in r for r in reasons)

        ctx_all = {"peer_roles": ["admin", "auditor", "operator"]}
        assert policy_all.connection_allowed("peer", ctx_all) is True

        # require_all_roles = False (any-of)
        policy_any = MTLSCommsPolicy(
            trust_domain="sov.nethical.org",
            required_roles=["admin", "auditor"],
            require_all_roles=False,
        )
        assert policy_any.connection_allowed("peer", {"peer_roles": ["auditor"]}) is True
        assert policy_any.connection_allowed("peer", {"peer_roles": ["guest"]}) is False

    def test_from_config_and_identities(self):
        cfg = {
            "trust_domain": "sov.nethical.org",
            "allowed_identities": ["agent-*"],
            "allowed_regions": ["PL"],
            "require_device_attested": True,
        }
        policy = MTLSCommsPolicy.from_config(cfg)
        ident = policy.identities()

        assert ident["impl"] == "mtls"
        assert ident["trust_domain"] == "sov.nethical.org"
        assert ident["require_device_attested"] is True
        assert "agent-*" in ident["allowed_identities"]
        assert "PL" in ident["allowed_regions"]

        with pytest.raises(ValueError, match="trust_domain is required"):
            MTLSCommsPolicy.from_config({})
