"""
Tests for Production Readiness Checklist Implementations

This test suite validates all implementations for sections 8-12 of the
production readiness checklist.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Section 8: Plugin Trust
from nethical.marketplace.plugin_trust import (
    PluginTrustSystem,
    TrustGatingResult,
    PluginTrustCheck,
)
from nethical.marketplace.plugin_governance import PluginGovernance
from nethical.marketplace.community import CommunityManager

# Section 9: Human Review
from nethical.governance.human_review import (
    HumanReviewQueue,
    ReviewPriority,
    FeedbackCategory,
    ReviewStatus,
)

# Section 10: Transparency
from nethical.explainability.quarterly_transparency import (
    QuarterlyTransparencyReportGenerator,
    MerkleRootsRegistry,
)

# Section 11: Release & Change
from nethical.policy.release_management import PolicyPack, DeploymentStage, CanaryConfig

# Section 12: Compliance
from nethical.security.data_compliance import (
    DataResidencyMapper,
    DataSubjectRequestHandler,
    DataRegion,
    DataCategory,
    ProcessingPurpose,
    RequestType,
    RequestStatus,
)


class TestPluginTrust:
    """Test Section 8: Plugin Trust"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def trust_system(self, temp_dir):
        """Create trust system instance"""
        governance = PluginGovernance(storage_dir=temp_dir)
        community = CommunityManager(storage_dir=temp_dir)
        return PluginTrustSystem(
            governance=governance,
            community=community,
            trust_threshold=80.0,
            storage_dir=temp_dir,
        )

    def test_trust_system_initialization(self, trust_system):
        """Test trust system initializes correctly"""
        assert trust_system.trust_threshold == 80.0
        assert trust_system.enforce_signature is True
        assert trust_system.max_vulnerabilities == 0

    def test_plugin_trust_verification(self, trust_system):
        """Test complete plugin trust verification"""
        # Create a simple plugin for testing
        plugin_code = """
def safe_function():
    return "Hello World"
"""

        check = trust_system.verify_plugin_trust(
            plugin_id="test-plugin-001", plugin_code=plugin_code
        )

        assert isinstance(check, PluginTrustCheck)
        assert check.plugin_id == "test-plugin-001"
        assert check.trust_score_threshold == 80.0
        # Note: Will likely fail trust score due to no reviews

    def test_trust_metrics(self, trust_system):
        """Test trust system metrics collection"""
        metrics = trust_system.get_trust_metrics()

        assert "total_checks" in metrics
        assert "passed_checks" in metrics
        assert "failed_checks" in metrics
        assert "pass_rate" in metrics
        assert "trust_threshold" in metrics
        assert metrics["trust_threshold"] == 80.0

    def test_trust_cache_management(self, trust_system):
        """Test trust check caching"""
        plugin_code = "def test(): pass"

        # First check
        check1 = trust_system.verify_plugin_trust(
            "cached-plugin", plugin_code=plugin_code
        )

        # Second check (should use cache)
        check2 = trust_system.verify_plugin_trust(
            "cached-plugin", plugin_code=plugin_code
        )

        # Timestamps should be the same (cached)
        assert check1.timestamp == check2.timestamp

        # Clear cache
        trust_system.clear_cache("cached-plugin")

        # Third check (should be new)
        check3 = trust_system.verify_plugin_trust(
            "cached-plugin", plugin_code=plugin_code, force_scan=True
        )

        # Timestamp should be different (new scan)
        assert check3.timestamp > check1.timestamp


class TestHumanReview:
    """Test Section 9: Human Review"""

    @pytest.fixture
    def review_queue(self):
        """Create review queue instance"""
        return HumanReviewQueue(drift_threshold=0.05)

    def test_queue_initialization(self, review_queue):
        """Test review queue initializes correctly"""
        assert review_queue.drift_threshold == 0.05
        assert review_queue.sla_hours[ReviewPriority.CRITICAL] == 4
        assert review_queue.sla_hours[ReviewPriority.HIGH] == 24

    def test_add_review_item(self, review_queue):
        """Test adding items to review queue"""
        item = review_queue.add_item(
            "plugin-001",
            "plugin",
            ReviewPriority.HIGH,
            metadata={"reason": "security review"},
        )

        assert item.item_id == "plugin-001"
        assert item.priority == ReviewPriority.HIGH
        assert item.status == ReviewStatus.PENDING
        assert item.sla_deadline is not None

    def test_assign_and_complete_item(self, review_queue):
        """Test assigning and completing review items"""
        # Add item
        item = review_queue.add_item("action-001", "action", ReviewPriority.MEDIUM)

        # Assign to reviewer
        assigned = review_queue.assign_item(item.item_id, "reviewer-1")
        assert assigned is True

        updated_item = review_queue._items[item.item_id]
        assert updated_item.status == ReviewStatus.IN_PROGRESS
        assert updated_item.assigned_to == "reviewer-1"

        # Complete review
        completed = review_queue.complete_item(
            item.item_id,
            FeedbackCategory.QUALITY_CONCERN,
            "Needs improvement in error handling",
        )
        assert completed is True

        final_item = review_queue._items[item.item_id]
        assert final_item.status == ReviewStatus.COMPLETED
        assert final_item.feedback_category == FeedbackCategory.QUALITY_CONCERN

    def test_sla_metrics(self, review_queue):
        """Test SLA metrics collection"""
        # Add several items
        for i in range(5):
            review_queue.add_item(f"item-{i}", "plugin", ReviewPriority.MEDIUM)

        # Complete some items
        for i in range(3):
            review_queue.assign_item(f"item-{i}", "reviewer-1")
            review_queue.complete_item(
                f"item-{i}", FeedbackCategory.QUALITY_CONCERN, "Review complete"
            )

        metrics = review_queue.get_sla_metrics()

        assert metrics.total_items == 5
        assert metrics.completed_items == 3
        assert metrics.pending_items == 2
        assert metrics.sla_compliance_rate >= 0

    def test_feedback_taxonomy_coverage(self, review_queue):
        """Test feedback taxonomy coverage reporting"""
        # Add and complete items with different feedback categories
        categories = [
            FeedbackCategory.SECURITY_ISSUE,
            FeedbackCategory.QUALITY_CONCERN,
            FeedbackCategory.PERFORMANCE_ISSUE,
        ]

        for i, category in enumerate(categories):
            item_id = f"item-{i}"
            review_queue.add_item(item_id, "plugin", ReviewPriority.MEDIUM)
            review_queue.assign_item(item_id, "reviewer-1")
            review_queue.complete_item(item_id, category, "Feedback")

        coverage = review_queue.get_feedback_taxonomy_coverage()

        assert coverage["total_reviews"] == 3
        assert coverage["covered_categories"] == 3
        assert "coverage_percentage" in coverage
        assert "coverage_by_category" in coverage

    def test_reviewer_drift_metrics(self, review_queue):
        """Test reviewer drift detection"""
        # Create reviews from multiple reviewers
        reviewers = ["reviewer-1", "reviewer-2"]

        for reviewer_idx, reviewer in enumerate(reviewers):
            for i in range(5):
                item_id = f"{reviewer}-item-{i}"
                review_queue.add_item(item_id, "plugin", ReviewPriority.MEDIUM)
                review_queue.assign_item(item_id, reviewer)

                # reviewer-1 focuses on security, reviewer-2 on quality
                category = (
                    FeedbackCategory.SECURITY_ISSUE
                    if reviewer_idx == 0
                    else FeedbackCategory.QUALITY_CONCERN
                )
                review_queue.complete_item(item_id, category, "Review")

        drift_report = review_queue.get_drift_report()

        assert "total_reviewers" in drift_report
        assert "max_drift_score" in drift_report
        assert "avg_drift_score" in drift_report
        assert drift_report["drift_threshold"] == 0.05

        # Check that drift is detected
        assert drift_report["max_drift_percentage"] >= 0


class TestTransparency:
    """Test Section 10: Transparency"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def merkle_registry(self, temp_dir):
        """Create Merkle roots registry"""
        return MerkleRootsRegistry(storage_dir=temp_dir)

    @pytest.fixture
    def report_generator(self, temp_dir, merkle_registry):
        """Create quarterly report generator"""
        return QuarterlyTransparencyReportGenerator(
            output_dir=temp_dir, merkle_registry=merkle_registry
        )

    def test_merkle_registry_initialization(self, merkle_registry):
        """Test Merkle registry initializes correctly"""
        stats = merkle_registry.get_statistics()
        assert "total_events" in stats
        assert "total_anchors" in stats
        assert stats["total_events"] >= 0

    def test_register_event_and_anchor(self, merkle_registry):
        """Test registering events and anchoring Merkle root"""
        # Register events
        leaf1 = merkle_registry.register_event(
            "policy_change", {"policy_id": "P001", "version": "1.0"}
        )
        assert leaf1 is not None

        leaf2 = merkle_registry.register_event(
            "violation_detected", {"severity": "high", "type": "security"}
        )
        assert leaf2 is not None

        # Anchor the root
        anchor = merkle_registry.anchor_merkle_root(
            anchor_type="file", receipt="Test anchor"
        )

        assert anchor.root is not None
        assert anchor.type == "file"

        # Verify statistics
        stats = merkle_registry.get_statistics()
        assert stats["total_events"] == 2
        assert stats["total_anchors"] == 1

    def test_quarterly_report_generation(self, report_generator):
        """Test quarterly transparency report generation"""
        # Sample data
        decisions = [
            {"timestamp": "2024-01-15T10:00:00Z", "decision": "ALLOW"},
            {"timestamp": "2024-02-20T14:30:00Z", "decision": "BLOCK"},
            {"timestamp": "2024-03-10T09:15:00Z", "decision": "ALLOW"},
        ]

        violations = [
            {
                "timestamp": "2024-02-20T14:30:00Z",
                "severity": "high",
                "type": "security",
            }
        ]

        policies = [{"policy_id": "P001", "name": "Safety Policy"}]

        # Generate report for Q1 2024
        report = report_generator.generate_quarterly_report(
            quarter=1,
            year=2024,
            decisions=decisions,
            violations=violations,
            policies=policies,
        )

        assert report.report_id == "QTR-2024Q1"
        assert report.quarter == 1
        assert report.year == 2024
        assert report.base_report is not None
        assert "description" in report.risk_methodology
        assert "factors" in report.risk_methodology
        assert "description" in report.detection_methodology
        assert "layers" in report.detection_methodology

    def test_auto_generate_current_quarter(self, report_generator):
        """Test auto-generation for current quarter"""
        decisions = []
        violations = []
        policies = []

        report = report_generator.auto_generate_for_current_quarter(
            decisions=decisions, violations=violations, policies=policies
        )

        assert report.quarter >= 1 and report.quarter <= 4
        assert report.year == datetime.now().year


class TestReleaseManagement:
    """Test Section 11: Release & Change"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def policy_pack(self, temp_dir):
        """Create policy pack instance"""
        return PolicyPack("test_policies", storage_dir=temp_dir)

    def test_policy_pack_initialization(self, policy_pack):
        """Test policy pack initializes correctly"""
        assert policy_pack.pack_name == "test_policies"
        assert len(policy_pack._versions) == 0

    def test_create_policy_version(self, policy_pack):
        """Test creating policy versions"""
        policy_content = {
            "rules": [{"id": "R001", "condition": "severity > 7", "action": "BLOCK"}]
        }

        version = policy_pack.create_version(
            version="1.0.0",
            content=policy_content,
            created_by="admin",
            description="Initial release",
        )

        assert version.version == "1.0.0"
        assert version.checksum is not None
        assert version.content == policy_content

    def test_canary_deployment(self, policy_pack):
        """Test canary deployment configuration"""
        # Create versions
        policy_pack.create_version("1.0.0", {"rules": []}, "admin", "Initial")
        policy_pack.create_version(
            "1.1.0", {"rules": [{"new": "rule"}]}, "admin", "Update"
        )

        # Deploy to canary
        deployment = policy_pack.deploy_canary(
            version="1.1.0", canary_percentage=10.0, duration_minutes=30
        )

        assert deployment.policy_version == "1.1.0"
        assert deployment.stage == DeploymentStage.CANARY
        assert deployment.canary_config.canary_percentage == 10.0

    def test_promote_to_production(self, policy_pack):
        """Test promoting version to production"""
        # Create and promote version
        policy_pack.create_version("2.0.0", {"rules": []}, "admin")

        deployment = policy_pack.promote_to_production("2.0.0")

        assert deployment.policy_version == "2.0.0"
        assert deployment.stage == DeploymentStage.PRODUCTION
        assert policy_pack._production_version == "2.0.0"

    def test_rollback_procedure(self, policy_pack):
        """Test rollback to previous version"""
        # Create versions
        policy_pack.create_version("1.0.0", {"v": "1.0"}, "admin")
        policy_pack.promote_to_production("1.0.0")

        policy_pack.create_version("1.1.0", {"v": "1.1"}, "admin")
        policy_pack.promote_to_production("1.1.0")

        # Rollback
        deployment = policy_pack.rollback_to_version(
            "1.0.0", reason="Critical bug in 1.1.0"
        )

        assert deployment.stage == DeploymentStage.ROLLBACK
        assert policy_pack._production_version == "1.0.0"

    def test_rollback_testing(self, policy_pack):
        """Test rollback procedure testing"""
        # Need at least 2 versions for rollback test
        policy_pack.create_version("1.0.0", {"v": "1.0"}, "admin")
        policy_pack.promote_to_production("1.0.0")

        policy_pack.create_version("1.1.0", {"v": "1.1"}, "admin")
        policy_pack.promote_to_production("1.1.0")

        # Test rollback
        result = policy_pack.test_rollback()
        assert result is True


class TestCompliance:
    """Test Section 12: Compliance"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def residency_mapper(self, temp_dir):
        """Create data residency mapper"""
        return DataResidencyMapper(storage_dir=temp_dir)

    @pytest.fixture
    def request_handler(self, temp_dir):
        """Create data subject request handler"""
        return DataSubjectRequestHandler(storage_dir=temp_dir)

    def test_register_data_store(self, residency_mapper):
        """Test registering data stores"""
        store = residency_mapper.register_data_store(
            store_id="db-001",
            name="Primary Database",
            region=DataRegion.US_EAST,
            data_categories={
                DataCategory.PERSONAL_IDENTIFIABLE,
                DataCategory.BEHAVIORAL,
            },
            retention_days=365,
            encryption_enabled=True,
        )

        assert store.store_id == "db-001"
        assert store.region == DataRegion.US_EAST
        assert len(store.data_categories) == 2

    def test_register_data_flow(self, residency_mapper):
        """Test registering data flows"""
        # Register stores first
        residency_mapper.register_data_store(
            "store-1",
            "Store 1",
            DataRegion.US_EAST,
            {DataCategory.PERSONAL_IDENTIFIABLE},
            365,
        )
        residency_mapper.register_data_store(
            "store-2",
            "Store 2",
            DataRegion.EU_WEST,
            {DataCategory.PERSONAL_IDENTIFIABLE},
            365,
        )

        # Register flow
        flow = residency_mapper.register_data_flow(
            flow_id="flow-001",
            source="store-1",
            destination="store-2",
            data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
            purpose=ProcessingPurpose.CONTRACT,
            encryption_in_transit=True,
        )

        assert flow.flow_id == "flow-001"
        assert flow.cross_border is True  # US to EU
        assert flow.encryption_in_transit is True

    def test_generate_data_flow_diagram(self, residency_mapper):
        """Test data flow diagram generation"""
        # Register some stores and flows
        residency_mapper.register_data_store(
            "db-1",
            "Database",
            DataRegion.US_EAST,
            {DataCategory.PERSONAL_IDENTIFIABLE},
            365,
        )

        diagram = residency_mapper.generate_data_flow_diagram()

        assert "metadata" in diagram
        assert "data_stores" in diagram
        assert "data_flows" in diagram
        assert len(diagram["data_stores"]) > 0

    def test_submit_access_request(self, request_handler):
        """Test submitting an access request"""
        request = request_handler.submit_request(
            RequestType.ACCESS,
            subject_id="user-123",
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
            verification_method="email",
        )

        assert request.request_type == RequestType.ACCESS
        assert request.subject_id == "user-123"
        assert request.status == RequestStatus.PENDING

    def test_process_access_request(self, request_handler):
        """Test processing an access request"""
        # Submit request
        request = request_handler.submit_request(RequestType.ACCESS, "user-456")

        # Process request
        data = request_handler.process_access_request(request.request_id)

        assert data is not None
        assert "subject_id" in data
        assert data["subject_id"] == "user-456"

        # Check status
        updated = request_handler.get_request_status(request.request_id)
        assert updated.status == RequestStatus.COMPLETED

    def test_process_deletion_request(self, request_handler):
        """Test processing a deletion request"""
        # Submit deletion request
        request = request_handler.submit_request(RequestType.ERASURE, "user-789")

        # Process request (dry run)
        result = request_handler.process_deletion_request(
            request.request_id, dry_run=True
        )

        assert result is not None
        assert result["dry_run"] is True
        assert "deleted_records" in result

        # Check status
        updated = request_handler.get_request_status(request.request_id)
        assert updated.status == RequestStatus.COMPLETED

    def test_workflow_testing(self, request_handler):
        """Test access and deletion workflows"""
        results = request_handler.test_workflow()

        assert "access_request" in results
        assert "deletion_request" in results
        assert results["access_request"] is True
        assert results["deletion_request"] is True
        assert results["all_passed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
