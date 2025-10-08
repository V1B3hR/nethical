"""Tests for Regionalization & Sharding (F1) features."""

import pytest
from nethical.core import IntegratedGovernance
from nethical.core.models import (
    AgentAction,
    SafetyViolation,
    JudgmentResult,
    ActionType,
    ViolationType,
    Severity,
    Decision
)


class TestRegionalDataModels:
    """Test regional fields in data models."""
    
    def test_agent_action_with_region(self):
        """Test AgentAction with regional fields."""
        action = AgentAction(
            agent_id="agent_001",
            action_type=ActionType.QUERY,
            content="Test action",
            region_id="eu-west-1",
            logical_domain="customer-service"
        )
        
        assert action.region_id == "eu-west-1"
        assert action.logical_domain == "customer-service"
        assert action.agent_id == "agent_001"
    
    def test_agent_action_without_region(self):
        """Test AgentAction without regional fields (backwards compatibility)."""
        action = AgentAction(
            agent_id="agent_002",
            action_type=ActionType.RESPONSE,
            content="Test response"
        )
        
        assert action.region_id is None
        assert action.logical_domain is None
        assert action.agent_id == "agent_002"
    
    def test_safety_violation_with_region(self):
        """Test SafetyViolation with regional fields."""
        violation = SafetyViolation(
            action_id="action_123",
            violation_type=ViolationType.PRIVACY,
            severity=Severity.HIGH,
            description="Privacy violation",
            confidence=0.9,
            region_id="us-east-1",
            logical_domain="data-processing"
        )
        
        assert violation.region_id == "us-east-1"
        assert violation.logical_domain == "data-processing"
        assert violation.violation_type == ViolationType.PRIVACY
    
    def test_judgment_result_with_region(self):
        """Test JudgmentResult with regional fields."""
        # Create a violation first since BLOCK decisions require violations
        violation = SafetyViolation(
            action_id="action_456",
            violation_type=ViolationType.SECURITY,
            severity=Severity.HIGH,
            description="Security issue",
            confidence=0.9
        )
        
        judgment = JudgmentResult(
            action_id="action_456",
            decision=Decision.BLOCK,
            confidence=0.95,
            reasoning="Blocked due to policy",
            violations=[violation],
            region_id="ap-south-1",
            logical_domain="payment-processing"
        )
        
        assert judgment.region_id == "ap-south-1"
        assert judgment.logical_domain == "payment-processing"
        assert judgment.decision == Decision.BLOCK


class TestRegionalGovernance:
    """Test regional governance initialization and configuration."""
    
    def test_governance_with_regional_config(self):
        """Test IntegratedGovernance with regional configuration."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data1",
            region_id="eu-west-1",
            logical_domain="customer-service",
            data_residency_policy="EU_GDPR"
        )
        
        assert gov.region_id == "eu-west-1"
        assert gov.logical_domain == "customer-service"
        assert gov.data_residency_policy == "EU_GDPR"
        assert 'compliance_requirements' in gov.regional_policies
        assert 'GDPR' in gov.regional_policies['compliance_requirements']
    
    def test_governance_with_us_ccpa(self):
        """Test IntegratedGovernance with US CCPA policy."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data2",
            region_id="us-west-2",
            data_residency_policy="US_CCPA"
        )
        
        assert gov.region_id == "us-west-2"
        assert gov.data_residency_policy == "US_CCPA"
        assert 'CCPA' in gov.regional_policies['compliance_requirements']
        assert gov.regional_policies['cross_border_transfer_allowed'] is True
    
    def test_governance_with_ai_act(self):
        """Test IntegratedGovernance with AI_ACT policy."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data3",
            region_id="eu-central-1",
            data_residency_policy="AI_ACT"
        )
        
        assert gov.data_residency_policy == "AI_ACT"
        assert 'AI_ACT' in gov.regional_policies['compliance_requirements']
        assert gov.regional_policies['human_oversight_required'] is True
    
    def test_governance_without_regional_config(self):
        """Test IntegratedGovernance without regional configuration (backwards compatibility)."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data4"
        )
        
        assert gov.region_id is None
        assert gov.logical_domain is None
        assert gov.data_residency_policy is None


class TestRegionalActionProcessing:
    """Test action processing with regional information."""
    
    def test_process_action_with_region(self):
        """Test processing action with regional information."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data5",
            region_id="eu-west-1",
            logical_domain="customer-service",
            data_residency_policy="EU_GDPR"
        )
        
        result = gov.process_action(
            agent_id="agent_123",
            action="test action",
            region_id="eu-west-1",
            compliance_requirements=["GDPR", "data_protection"]
        )
        
        assert result['region_id'] == "eu-west-1"
        assert result['logical_domain'] == "customer-service"
        assert result['compliance_requirements'] == ["GDPR", "data_protection"]
        assert 'data_residency' in result
        assert result['data_residency']['compliant'] is True
    
    def test_process_action_with_different_region(self):
        """Test processing action from different region."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data6",
            region_id="us-east-1",
            data_residency_policy="US_CCPA"
        )
        
        result = gov.process_action(
            agent_id="agent_456",
            action="test action",
            region_id="us-west-2"
        )
        
        # CCPA allows cross-border transfers within US
        assert result['region_id'] == "us-west-2"
        assert result['data_residency']['compliant'] is True
    
    def test_process_action_cross_border_blocked(self):
        """Test processing action with cross-border transfer blocked."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data7",
            region_id="eu-west-1",
            data_residency_policy="EU_GDPR"
        )
        
        result = gov.process_action(
            agent_id="agent_789",
            action="test action",
            region_id="us-east-1"
        )
        
        # GDPR does not allow cross-border transfers
        assert result['data_residency']['compliant'] is False
        assert len(result['data_residency']['violations']) > 0


class TestDataResidencyValidation:
    """Test data residency compliance validation."""
    
    def test_validate_same_region(self):
        """Test validation within same region."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data8",
            region_id="eu-west-1",
            data_residency_policy="EU_GDPR"
        )
        
        validation = gov.validate_data_residency("eu-west-1")
        
        assert validation['compliant'] is True
        assert validation['region_id'] == "eu-west-1"
        assert len(validation['violations']) == 0
    
    def test_validate_cross_border_allowed(self):
        """Test validation with cross-border transfer allowed."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data9",
            region_id="us-east-1",
            data_residency_policy="US_CCPA"
        )
        
        validation = gov.validate_data_residency("us-west-2")
        
        assert validation['compliant'] is True
    
    def test_validate_cross_border_blocked(self):
        """Test validation with cross-border transfer blocked."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data10",
            region_id="eu-west-1",
            data_residency_policy="EU_GDPR"
        )
        
        validation = gov.validate_data_residency("us-east-1")
        
        assert validation['compliant'] is False
        assert 'Cross-border data transfer' in validation['violations'][0]


class TestCrossRegionAggregation:
    """Test cross-region reporting and aggregation."""
    
    def test_aggregate_by_region(self):
        """Test aggregating metrics by region."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data11"
        )
        
        metrics = [
            {
                'action_id': 'action_1',
                'region_id': 'eu-west-1',
                'risk_score': 0.5,
                'violation_detected': False
            },
            {
                'action_id': 'action_2',
                'region_id': 'eu-west-1',
                'risk_score': 0.7,
                'violation_detected': True
            },
            {
                'action_id': 'action_3',
                'region_id': 'us-east-1',
                'risk_score': 0.3,
                'violation_detected': False
            },
            {
                'action_id': 'action_4',
                'region_id': 'us-east-1',
                'risk_score': 0.9,
                'violation_detected': True
            },
        ]
        
        aggregated = gov.aggregate_by_region(metrics, group_by='region_id')
        
        assert 'eu-west-1' in aggregated
        assert 'us-east-1' in aggregated
        
        assert aggregated['eu-west-1']['count'] == 2
        assert aggregated['eu-west-1']['violations'] == 1
        assert aggregated['eu-west-1']['avg_risk_score'] == 0.6
        
        assert aggregated['us-east-1']['count'] == 2
        assert aggregated['us-east-1']['violations'] == 1
        assert aggregated['us-east-1']['avg_risk_score'] == 0.6
    
    def test_aggregate_by_logical_domain(self):
        """Test aggregating metrics by logical domain."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data12"
        )
        
        metrics = [
            {
                'action_id': 'action_1',
                'logical_domain': 'customer-service',
                'risk_score': 0.4,
                'violation_detected': False
            },
            {
                'action_id': 'action_2',
                'logical_domain': 'customer-service',
                'risk_score': 0.6,
                'violation_detected': False
            },
            {
                'action_id': 'action_3',
                'logical_domain': 'payment-processing',
                'risk_score': 0.8,
                'violation_detected': True
            },
        ]
        
        aggregated = gov.aggregate_by_region(metrics, group_by='logical_domain')
        
        assert 'customer-service' in aggregated
        assert 'payment-processing' in aggregated
        
        assert aggregated['customer-service']['count'] == 2
        assert aggregated['customer-service']['violations'] == 0
        assert aggregated['customer-service']['avg_risk_score'] == 0.5
        
        assert aggregated['payment-processing']['count'] == 1
        assert aggregated['payment-processing']['violations'] == 1
        assert aggregated['payment-processing']['avg_risk_score'] == 0.8


class TestMultiRegionPerformance:
    """Test performance with multiple regions."""
    
    def test_multiple_regions_basic(self):
        """Test basic operations with multiple regions."""
        regions = ['eu-west-1', 'us-east-1', 'ap-south-1', 'ap-northeast-1', 'sa-east-1']
        
        for region in regions:
            gov = IntegratedGovernance(
                storage_dir=f"./test_regional_data_{region}",
                region_id=region,
                logical_domain="test-domain"
            )
            
            result = gov.process_action(
                agent_id=f"agent_{region}",
                action=f"test action for {region}",
                region_id=region
            )
            
            assert result['region_id'] == region
            assert 'phase3' in result
            assert 'phase4' in result
    
    def test_cross_region_aggregation_with_5_regions(self):
        """Test cross-region aggregation with 5+ regions."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data_multi"
        )
        
        regions = ['eu-west-1', 'us-east-1', 'ap-south-1', 'ap-northeast-1', 'sa-east-1', 'ca-central-1']
        
        metrics = []
        for i, region in enumerate(regions):
            metrics.append({
                'action_id': f'action_{i}',
                'region_id': region,
                'risk_score': 0.1 * (i + 1),
                'violation_detected': i % 2 == 0
            })
        
        aggregated = gov.aggregate_by_region(metrics, group_by='region_id')
        
        assert len(aggregated) == len(regions)
        
        for region in regions:
            assert region in aggregated
            assert aggregated[region]['count'] == 1


class TestRegionalPolicyProfiles:
    """Test different regional policy profiles."""
    
    def test_gdpr_policy_profile(self):
        """Test GDPR policy profile configuration."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data_gdpr",
            data_residency_policy="EU_GDPR"
        )
        
        assert gov.regional_policies['compliance_requirements'] == ['GDPR', 'data_protection', 'right_to_erasure']
        assert gov.regional_policies['cross_border_transfer_allowed'] is False
        assert gov.regional_policies['encryption_required'] is True
        assert gov.regional_policies['audit_trail_required'] is True
        assert gov.regional_policies['consent_required'] is True
        assert gov.regional_policies['data_retention_days'] == 30
    
    def test_ccpa_policy_profile(self):
        """Test CCPA policy profile configuration."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data_ccpa",
            data_residency_policy="US_CCPA"
        )
        
        assert gov.regional_policies['compliance_requirements'] == ['CCPA', 'consumer_privacy']
        assert gov.regional_policies['cross_border_transfer_allowed'] is True
        assert gov.regional_policies['encryption_required'] is True
        assert gov.regional_policies['data_retention_days'] == 90
    
    def test_ai_act_policy_profile(self):
        """Test AI_ACT policy profile configuration."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data_ai_act",
            data_residency_policy="AI_ACT"
        )
        
        assert 'AI_ACT' in gov.regional_policies['compliance_requirements']
        assert gov.regional_policies['human_oversight_required'] is True
        assert gov.regional_policies['data_retention_days'] == 180
    
    def test_global_default_policy_profile(self):
        """Test default policy profile for unknown policies."""
        gov = IntegratedGovernance(
            storage_dir="./test_regional_data_unknown",
            data_residency_policy="UNKNOWN_POLICY"
        )
        
        assert gov.regional_policies['compliance_requirements'] == ['basic_safety']
        assert gov.regional_policies['cross_border_transfer_allowed'] is True
        assert gov.regional_policies['data_retention_days'] == 365
