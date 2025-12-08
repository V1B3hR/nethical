"""
Tests for Phase 9 Audit Portal API

This module tests the audit portal REST API including:
- Rate limiting functionality
- Decision trace explorer endpoints
- Policy lineage viewer endpoints
- Fairness metrics endpoints
- Audit log browser endpoints
- Appeals tracking endpoints
"""

import pytest
from datetime import datetime, timedelta
from portal.api import AuditPortalAPI, RateLimitTier, RateLimiter, RateLimitStatus


class TestRateLimiter:
    """Test suite for rate limiting functionality"""

    def test_rate_limiter_initialization(self):
        """Test rate limiter can be initialized"""
        limiter = RateLimiter()
        assert limiter is not None
        assert limiter._buckets == {}

    def test_anonymous_tier_limits(self):
        """Test anonymous tier has correct limits"""
        tier = RateLimitTier.ANONYMOUS
        assert tier.requests_per_hour == 100
        assert tier.burst == 20
        assert tier.concurrency == 5

    def test_authenticated_tier_limits(self):
        """Test authenticated tier has correct limits"""
        tier = RateLimitTier.AUTHENTICATED
        assert tier.requests_per_hour == 1000
        assert tier.burst == 100
        assert tier.concurrency == 20

    def test_premium_tier_limits(self):
        """Test premium tier has correct limits"""
        tier = RateLimitTier.PREMIUM
        assert tier.requests_per_hour == 10000
        assert tier.burst == 500
        assert tier.concurrency == 50

    def test_first_request_allowed(self):
        """Test first request is always allowed within burst"""
        limiter = RateLimiter()
        allowed, status = limiter.check_rate_limit(
            "test_client", RateLimitTier.AUTHENTICATED
        )

        assert allowed is True
        assert status.limit == 1000
        assert status.remaining > 0
        assert status.tier == "authenticated"

    def test_burst_limit_enforcement(self):
        """Test burst limit is enforced"""
        limiter = RateLimiter()
        tier = RateLimitTier.AUTHENTICATED

        # Exhaust burst capacity
        for _ in range(tier.burst):
            allowed, status = limiter.check_rate_limit("test_client", tier)
            assert allowed is True

        # Next request should be denied
        allowed, status = limiter.check_rate_limit("test_client", tier)
        assert allowed is False

    def test_different_clients_independent(self):
        """Test different clients have independent rate limits"""
        limiter = RateLimiter()
        tier = RateLimitTier.AUTHENTICATED

        # Client 1 exhausts burst
        for _ in range(tier.burst):
            limiter.check_rate_limit("client_1", tier)

        # Client 2 should still have full burst available
        allowed, status = limiter.check_rate_limit("client_2", tier)
        assert allowed is True
        assert status.remaining == tier.burst - 1


class TestAuditPortalAPI:
    """Test suite for Audit Portal API endpoints"""

    @pytest.fixture
    def api(self):
        """Create API instance for testing"""
        return AuditPortalAPI()

    @pytest.fixture
    def sample_decision(self):
        """Create sample decision for testing"""
        return {
            "decision_id": "dec_001",
            "policy_id": "pol_001",
            "agent_id": "agent_001",
            "outcome": "approved",
            "timestamp": datetime.utcnow(),
            "trace": {
                "steps": [
                    {"step": 1, "action": "evaluate_context"},
                    {"step": 2, "action": "apply_policy"},
                ],
                "justification": "Decision approved based on policy rules",
            },
        }

    def test_api_initialization(self, api):
        """Test API can be initialized"""
        assert api is not None
        assert api.rate_limiter is not None
        assert isinstance(api._decisions_db, dict)
        assert isinstance(api._policies_db, dict)
        assert isinstance(api._audit_logs_db, dict)
        assert isinstance(api._appeals_db, dict)

    def test_rate_limit_headers_included(self, api):
        """Test rate limit headers are included in responses"""
        response = api.search_decisions("test_client")

        assert "headers" in response
        assert "X-RateLimit-Limit" in response["headers"]
        assert "X-RateLimit-Remaining" in response["headers"]
        assert "X-RateLimit-Reset" in response["headers"]
        assert "X-RateLimit-Tier" in response["headers"]

    def test_search_decisions_empty_db(self, api):
        """Test searching decisions with empty database"""
        response = api.search_decisions("test_client")

        assert response["status"] == 200
        assert "data" in response
        assert response["data"]["decisions"] == []
        assert response["data"]["pagination"]["total"] == 0

    def test_search_decisions_with_data(self, api, sample_decision):
        """Test searching decisions with populated database"""
        # Add decision to database
        api._decisions_db["dec_001"] = sample_decision

        response = api.search_decisions("test_client")

        assert response["status"] == 200
        assert len(response["data"]["decisions"]) == 1
        assert response["data"]["pagination"]["total"] == 1

    def test_search_decisions_filter_by_policy(self, api, sample_decision):
        """Test filtering decisions by policy ID"""
        api._decisions_db["dec_001"] = sample_decision

        # Search with matching policy
        response = api.search_decisions("test_client", policy_id="pol_001")
        assert response["status"] == 200
        assert len(response["data"]["decisions"]) == 1

        # Search with non-matching policy
        response = api.search_decisions("test_client", policy_id="pol_999")
        assert response["status"] == 200
        assert len(response["data"]["decisions"]) == 0

    def test_search_decisions_pagination(self, api):
        """Test decision search pagination"""
        # Add multiple decisions
        for i in range(75):
            api._decisions_db[f"dec_{i:03d}"] = {
                "decision_id": f"dec_{i:03d}",
                "policy_id": "pol_001",
                "outcome": "approved",
                "timestamp": datetime.utcnow(),
            }

        # Get first page
        response = api.search_decisions("test_client", page=1, per_page=50)
        assert response["status"] == 200
        assert len(response["data"]["decisions"]) == 50
        assert response["data"]["pagination"]["page"] == 1
        assert response["data"]["pagination"]["total"] == 75
        assert response["data"]["pagination"]["pages"] == 2

        # Get second page
        response = api.search_decisions("test_client", page=2, per_page=50)
        assert response["status"] == 200
        assert len(response["data"]["decisions"]) == 25
        assert response["data"]["pagination"]["page"] == 2

    def test_get_decision_not_found(self, api):
        """Test getting non-existent decision"""
        response = api.get_decision("test_client", "dec_999")

        assert response["status"] == 404
        assert "error" in response
        assert response["error"] == "Decision not found"

    def test_get_decision_success(self, api, sample_decision):
        """Test getting existing decision"""
        api._decisions_db["dec_001"] = sample_decision

        response = api.get_decision("test_client", "dec_001")

        assert response["status"] == 200
        assert "data" in response
        assert response["data"]["decision_id"] == "dec_001"

    def test_get_decision_trace(self, api, sample_decision):
        """Test getting decision evaluation trace"""
        api._decisions_db["dec_001"] = sample_decision

        response = api.get_decision_trace("test_client", "dec_001")

        assert response["status"] == 200
        assert "data" in response
        assert "trace" in response["data"]
        assert "steps" in response["data"]
        assert len(response["data"]["steps"]) == 2

    def test_list_policies_empty(self, api):
        """Test listing policies with empty database"""
        response = api.list_policies("test_client")

        assert response["status"] == 200
        assert response["data"]["policies"] == []
        assert response["data"]["pagination"]["total"] == 0

    def test_get_policy_versions(self, api):
        """Test getting policy versions"""
        # Add policy with versions
        api._policies_db["pol_001"] = {
            "policy_id": "pol_001",
            "name": "Test Policy",
            "current_version": "2.0.0",
            "versions": [
                {"version": "1.0.0", "hash": "abc123"},
                {"version": "2.0.0", "hash": "def456", "prev_hash": "abc123"},
            ],
        }

        response = api.get_policy_versions("test_client", "pol_001")

        assert response["status"] == 200
        assert "data" in response
        assert len(response["data"]["versions"]) == 2
        assert response["data"]["current_version"] == "2.0.0"

    def test_get_policy_lineage(self, api):
        """Test getting policy lineage"""
        api._policies_db["pol_001"] = {
            "policy_id": "pol_001",
            "lineage": {
                "hash_chain": [
                    {"version": "1.0.0", "hash": "abc123"},
                    {"version": "2.0.0", "hash": "def456"},
                ]
            },
        }

        response = api.get_policy_lineage("test_client", "pol_001")

        assert response["status"] == 200
        assert "data" in response
        assert "hash_chain" in response["data"]
        assert len(response["data"]["hash_chain"]) == 2

    def test_verify_policy_chain_valid(self, api):
        """Test verifying valid policy hash chain"""
        api._policies_db["pol_001"] = {
            "policy_id": "pol_001",
            "versions": [
                {"version": "1.0.0", "hash": "abc123"},
                {"version": "2.0.0", "hash": "def456", "prev_hash": "abc123"},
                {"version": "3.0.0", "hash": "ghi789", "prev_hash": "def456"},
            ],
        }

        response = api.verify_policy_chain("test_client", "pol_001")

        assert response["status"] == 200
        assert response["data"]["chain_valid"] is True
        assert response["data"]["total_versions"] == 3
        assert len(response["data"]["verification_details"]) == 2

    def test_verify_policy_chain_invalid(self, api):
        """Test verifying invalid policy hash chain"""
        api._policies_db["pol_001"] = {
            "policy_id": "pol_001",
            "versions": [
                {"version": "1.0.0", "hash": "abc123"},
                {"version": "2.0.0", "hash": "def456", "prev_hash": "wrong"},
            ],
        }

        response = api.verify_policy_chain("test_client", "pol_001")

        assert response["status"] == 200
        assert response["data"]["chain_valid"] is False

    def test_get_fairness_metrics(self, api):
        """Test getting fairness metrics"""
        api._fairness_metrics_db["metrics"] = [
            {
                "attribute": "gender",
                "statistical_parity": 0.05,
                "timestamp": datetime.utcnow(),
            }
        ]

        response = api.get_fairness_metrics("test_client")

        assert response["status"] == 200
        assert "data" in response
        assert len(response["data"]["metrics"]) == 1

    def test_get_audit_logs(self, api):
        """Test getting audit logs"""
        api._audit_logs_db["log_001"] = {
            "log_id": "log_001",
            "event_type": "decision_made",
            "timestamp": datetime.utcnow(),
        }

        response = api.get_audit_logs("test_client")

        assert response["status"] == 200
        assert "data" in response
        assert len(response["data"]["logs"]) == 1

    def test_get_merkle_root(self, api):
        """Test getting Merkle tree root"""
        api._audit_logs_db["merkle_root"] = {
            "hash": "abc123def456",
            "timestamp": datetime.utcnow().isoformat(),
            "log_count": 1000,
            "anchored": True,
            "anchor_locations": ["s3", "blockchain"],
        }

        response = api.get_merkle_root("test_client")

        assert response["status"] == 200
        assert "data" in response
        assert response["data"]["merkle_root"] == "abc123def456"
        assert response["data"]["anchored"] is True
        assert len(response["data"]["anchor_locations"]) == 2

    def test_create_appeal(self, api):
        """Test creating an appeal"""
        response = api.create_appeal(
            "test_client", "dec_001", "I believe the decision was incorrect"
        )

        assert response["status"] == 201
        assert "data" in response
        assert response["data"]["decision_id"] == "dec_001"
        assert response["data"]["status"] == "submitted"
        assert "appeal_id" in response["data"]

    def test_get_appeals_empty(self, api):
        """Test getting appeals with empty database"""
        response = api.get_appeals("test_client")

        assert response["status"] == 200
        assert response["data"]["appeals"] == []
        assert response["data"]["pagination"]["total"] == 0

    def test_get_appeals_with_data(self, api):
        """Test getting appeals with data"""
        # Create an appeal first
        api.create_appeal("test_client", "dec_001", "Test appeal")

        response = api.get_appeals("test_client")

        assert response["status"] == 200
        assert len(response["data"]["appeals"]) == 1

    def test_rate_limit_enforcement(self, api):
        """Test rate limiting is enforced across requests"""
        tier = RateLimitTier.ANONYMOUS

        # Make requests up to burst limit
        for i in range(tier.burst):
            response = api.search_decisions("limited_client", tier)
            assert response["status"] == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = api.search_decisions("limited_client", tier)
        assert response["status"] == 429
        assert "error" in response
        assert response["error"] == "Rate limit exceeded"
        assert "retry_after" in response


class TestAPIIntegration:
    """Integration tests for API workflows"""

    @pytest.fixture
    def api(self):
        """Create API instance for testing"""
        return AuditPortalAPI()

    def test_complete_decision_workflow(self, api):
        """Test complete workflow: create decision, search, retrieve, trace"""
        # Create decision
        decision = {
            "decision_id": "dec_integration",
            "policy_id": "pol_001",
            "agent_id": "agent_001",
            "outcome": "approved",
            "timestamp": datetime.utcnow(),
            "trace": {
                "steps": [{"step": 1, "action": "evaluate"}],
                "justification": "Approved",
            },
        }
        api._decisions_db["dec_integration"] = decision

        # Search for decision
        search_response = api.search_decisions("test_client", policy_id="pol_001")
        assert search_response["status"] == 200
        assert len(search_response["data"]["decisions"]) == 1

        # Get decision details
        get_response = api.get_decision("test_client", "dec_integration")
        assert get_response["status"] == 200
        assert get_response["data"]["decision_id"] == "dec_integration"

        # Get decision trace
        trace_response = api.get_decision_trace("test_client", "dec_integration")
        assert trace_response["status"] == 200
        assert "trace" in trace_response["data"]

    def test_complete_appeal_workflow(self, api):
        """Test complete appeal workflow: create, list, filter"""
        # Create multiple appeals
        api.create_appeal("test_client", "dec_001", "Appeal 1")
        api.create_appeal("test_client", "dec_002", "Appeal 2")
        api.create_appeal("test_client", "dec_003", "Appeal 3")

        # List all appeals
        response = api.get_appeals("test_client")
        assert response["status"] == 200
        assert len(response["data"]["appeals"]) == 3

        # Test pagination
        response = api.get_appeals("test_client", page=1, per_page=2)
        assert len(response["data"]["appeals"]) == 2
        assert response["data"]["pagination"]["pages"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
