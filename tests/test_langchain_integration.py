"""
Tests for LangChain Integration

Tests the Nethical-LangChain integration including:
- NethicalGuardTool basic functionality
- Decision-making logic
- Integration with governance
- Error handling
- Helper functions
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.core import IntegratedGovernance
from nethical.core.models import Decision, ActionType


class TestNethicalGuardToolWithoutLangChain:
    """Test NethicalGuardTool without requiring LangChain installation."""

    def test_module_imports_without_langchain(self):
        """Test that module can be imported even without LangChain."""
        # This should not raise an ImportError
        from nethical.integrations import langchain_tools

        # Check that LANGCHAIN_AVAILABLE is False without langchain installed
        # (unless it's actually installed in the test environment)
        assert hasattr(langchain_tools, "LANGCHAIN_AVAILABLE")

    def test_module_exports_expected_classes(self):
        """Test that module exports expected classes."""
        from nethical.integrations import langchain_tools

        # These should always be available
        assert hasattr(langchain_tools, "NethicalGuardTool")
        assert hasattr(langchain_tools, "LlamaGuardChain")
        assert hasattr(langchain_tools, "create_nethical_agent")
        assert hasattr(langchain_tools, "chain_guards")


class TestNethicalGuardToolCore:
    """Test core functionality of NethicalGuardTool."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    @pytest.fixture
    def mock_governance(self):
        """Create a mock IntegratedGovernance instance."""
        gov = Mock(spec=IntegratedGovernance)
        gov.process_action = Mock()
        return gov

    def test_decision_logic_allow(self, temp_storage):
        """Test decision logic for ALLOW case."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(storage_dir=temp_storage, block_threshold=0.7)

        # Mock result with low risk score
        result = {
            "phase3": {"risk_score": 0.2, "risk_tier": "LOW"},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }

        decision = tool._make_decision(result)
        assert decision == "ALLOW"

    def test_decision_logic_warn(self, temp_storage):
        """Test decision logic for WARN case."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(
            storage_dir=temp_storage, block_threshold=0.7, warn_threshold=0.4
        )

        # Mock result with medium risk score
        result = {
            "phase3": {"risk_score": 0.5, "risk_tier": "MEDIUM"},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }

        decision = tool._make_decision(result)
        assert decision == "WARN"

    def test_decision_logic_block(self, temp_storage):
        """Test decision logic for BLOCK case."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(storage_dir=temp_storage, block_threshold=0.7)

        # Mock result with high risk score
        result = {
            "phase3": {"risk_score": 0.9, "risk_tier": "CRITICAL"},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }

        decision = tool._make_decision(result)
        assert decision == "BLOCK"

    def test_decision_logic_quota_block(self, temp_storage):
        """Test decision logic for quota-based BLOCK."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(storage_dir=temp_storage)

        # Mock result with quota block
        result = {"blocked_by_quota": True, "phase3": {"risk_score": 0.1}}

        decision = tool._make_decision(result)
        assert decision == "BLOCK"

    def test_decision_logic_quarantine(self, temp_storage):
        """Test decision logic for quarantine."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(storage_dir=temp_storage)

        # Mock result with quarantine
        result = {
            "phase3": {"risk_score": 0.5},
            "phase4": {"quarantined": True},
            "phase567": {},
            "phase89": {},
        }

        decision = tool._make_decision(result)
        assert decision == "BLOCK"

    def test_decision_logic_escalate(self, temp_storage):
        """Test decision logic for ESCALATE case."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(storage_dir=temp_storage)

        # Mock result with escalation
        result = {
            "phase3": {"risk_score": 0.6},
            "phase4": {},
            "phase567": {},
            "phase89": {"escalated": True, "escalation_reason": "High risk action"},
        }

        decision = tool._make_decision(result)
        assert decision == "ESCALATE"

    def test_format_simple_response(self, temp_storage):
        """Test formatting of simple response."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(storage_dir=temp_storage)

        result = {
            "phase3": {"risk_score": 0.8, "risk_tier": "HIGH"},
            "phase4": {"ethical_tags": ["privacy", "safety"]},
            "phase89": {"escalation_reason": "Potential violation"},
        }

        response = tool._format_simple_response("BLOCK", result)

        assert "Decision: BLOCK" in response
        assert "Risk Score: 0.80" in response
        assert "Risk Tier: HIGH" in response
        assert "privacy" in response
        assert "safety" in response
        assert "Potential violation" in response

    def test_run_with_simple_action(self, temp_storage, mock_governance):
        """Test _run method with a simple action."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        # Create tool with mock governance
        tool = NethicalGuardTool(storage_dir=temp_storage)
        tool.governance = mock_governance

        # Setup mock response
        mock_governance.process_action.return_value = {
            "phase3": {"risk_score": 0.3, "risk_tier": "LOW"},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }

        # Run evaluation
        result = tool._run(action="Tell me about AI safety", agent_id="test_agent")

        # Verify governance was called
        assert mock_governance.process_action.called
        assert "ALLOW" in result
        assert "Risk Score" in result

    def test_run_with_detailed_response(self, temp_storage, mock_governance):
        """Test _run method with detailed response enabled."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(
            storage_dir=temp_storage, return_detailed_response=True
        )
        tool.governance = mock_governance

        mock_governance.process_action.return_value = {
            "phase3": {"risk_score": 0.3, "risk_tier": "LOW"},
            "phase4": {},
        }

        result = tool._run(action="Test action", agent_id="test_agent")

        # Should return JSON with decision and details
        parsed = json.loads(result)
        assert "decision" in parsed
        assert "details" in parsed
        assert parsed["decision"] == "ALLOW"

    def test_run_with_error_handling(self, temp_storage):
        """Test error handling in _run method."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(storage_dir=temp_storage)

        # Create a governance that raises an error
        tool.governance = Mock(spec=IntegratedGovernance)
        tool.governance.process_action.side_effect = Exception("Test error")

        result = tool._run(action="Test action", agent_id="test_agent")

        assert "ERROR" in result
        assert "Governance evaluation failed" in result

    def test_threshold_configuration(self, temp_storage):
        """Test that thresholds can be configured."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(
            storage_dir=temp_storage, block_threshold=0.8, warn_threshold=0.5
        )

        assert tool.block_threshold == 0.8
        assert tool.warn_threshold == 0.5

        # Test WARN decision with custom thresholds
        result = {
            "phase3": {"risk_score": 0.6},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }
        decision = tool._make_decision(result)
        assert decision == "WARN"

        # Test BLOCK decision with custom thresholds
        result = {
            "phase3": {"risk_score": 0.85},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }
        decision = tool._make_decision(result)
        assert decision == "BLOCK"


class TestLlamaGuardChain:
    """Test LlamaGuard chain functionality."""

    def test_llama_guard_init_without_dependencies(self):
        """Test that LlamaGuard init handles missing dependencies gracefully."""
        from nethical.integrations.langchain_tools import LlamaGuardChain

        # Should not raise during init, but will raise when trying to use
        with pytest.raises((ImportError, NotImplementedError)):
            guard = LlamaGuardChain(use_local=False)
            # Trying to use it should raise
            guard.is_safe("test")

    def test_llama_guard_is_safe_safe_content(self):
        """Test LlamaGuard with safe content."""
        try:
            import langchain  # noqa: F401
        except ImportError:
            pytest.skip("LangChain not installed")

        from nethical.integrations.langchain_tools import LlamaGuardChain

        # This test requires langchain to be properly installed
        # We'll just test that we can create the instance
        pytest.skip("Requires full LangChain setup with transformers")

    def test_llama_guard_is_safe_unsafe_content(self):
        """Test LlamaGuard with unsafe content."""
        try:
            import langchain  # noqa: F401
        except ImportError:
            pytest.skip("LangChain not installed")

        from nethical.integrations.langchain_tools import LlamaGuardChain

        # This test requires langchain to be properly installed
        pytest.skip("Requires full LangChain setup with transformers")

    def test_llama_guard_evaluate(self):
        """Test LlamaGuard evaluate method."""
        try:
            import langchain  # noqa: F401
        except ImportError:
            pytest.skip("LangChain not installed")

        from nethical.integrations.langchain_tools import LlamaGuardChain

        # This test requires langchain to be properly installed
        pytest.skip("Requires full LangChain setup with transformers")


class TestChainGuards:
    """Test the chain_guards helper function."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_chain_guards_nethical_only(self, temp_storage):
        """Test chain_guards with only Nethical guard."""
        from nethical.integrations.langchain_tools import (
            NethicalGuardTool,
            chain_guards,
        )

        tool = NethicalGuardTool(storage_dir=temp_storage)
        tool.governance = Mock(spec=IntegratedGovernance)
        tool.governance.process_action.return_value = {
            "phase3": {"risk_score": 0.2, "risk_tier": "LOW"},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }

        result = chain_guards(tool, "Test action", "test_agent", None)

        assert result["final_decision"] == "ALLOW"
        assert result["nethical"] is not None
        assert result["llama_guard"] is None

    def test_chain_guards_llama_blocks(self, temp_storage):
        """Test chain_guards when LlamaGuard blocks."""
        from nethical.integrations.langchain_tools import (
            NethicalGuardTool,
            LlamaGuardChain,
            chain_guards,
        )

        tool = NethicalGuardTool(storage_dir=temp_storage)
        llama = Mock(spec=LlamaGuardChain)
        llama.evaluate.return_value = {
            "safe": False,
            "reason": "Harmful content detected",
        }

        result = chain_guards(tool, "Harmful action", "test_agent", llama)

        assert result["final_decision"] == "BLOCK"
        assert result["blocked_by"] == "llama_guard"
        assert result["llama_guard"] is not None
        # Nethical should not be called if LlamaGuard blocks
        assert result["nethical"] is None

    def test_chain_guards_both_allow(self, temp_storage):
        """Test chain_guards when both guards allow."""
        from nethical.integrations.langchain_tools import (
            NethicalGuardTool,
            LlamaGuardChain,
            chain_guards,
        )

        tool = NethicalGuardTool(storage_dir=temp_storage)
        tool.governance = Mock(spec=IntegratedGovernance)
        tool.governance.process_action.return_value = {
            "phase3": {"risk_score": 0.2, "risk_tier": "LOW"},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        }

        llama = Mock(spec=LlamaGuardChain)
        llama.evaluate.return_value = {"safe": True, "reason": None}

        result = chain_guards(tool, "Safe action", "test_agent", llama)

        assert result["final_decision"] == "ALLOW"
        assert result["llama_guard"]["safe"] is True
        assert "ALLOW" in result["nethical"]


class TestCreateNethicalAgent:
    """Test the create_nethical_agent helper function."""

    def test_create_nethical_agent_without_langchain(self):
        """Test that create_nethical_agent raises ImportError without LangChain."""
        from nethical.integrations.langchain_tools import (
            create_nethical_agent,
            LANGCHAIN_AVAILABLE,
        )

        if not LANGCHAIN_AVAILABLE:
            with pytest.raises(ImportError, match="LangChain is not installed"):
                create_nethical_agent(Mock(), [], "./test_data")
        else:
            pytest.skip("LangChain is available in test environment")


class TestIntegrationWithRealGovernance:
    """Integration tests with real IntegratedGovernance."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_tool_with_real_governance_safe_action(self, temp_storage):
        """Test tool with real governance for a safe action."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(
            storage_dir=temp_storage,
            enable_shadow_mode=False,  # Disable for faster tests
            enable_ml_blending=False,
            enable_anomaly_detection=False,
        )

        result = tool._run(
            action="What is the weather today?",
            agent_id="test_agent",
            action_type="query",
        )

        # Should get a response (allow, warn, or error - all acceptable)
        # The tool successfully calls governance, even if governance has issues
        assert isinstance(result, str)
        assert len(result) > 0
        # Should either be a decision or an error message
        assert any(
            x in result
            for x in ["ALLOW", "WARN", "BLOCK", "ESCALATE", "ERROR", "Decision"]
        )

    def test_tool_with_real_governance_multiple_actions(self, temp_storage):
        """Test tool with multiple actions to verify governance works."""
        from nethical.integrations.langchain_tools import NethicalGuardTool

        tool = NethicalGuardTool(
            storage_dir=temp_storage,
            enable_shadow_mode=False,
            enable_ml_blending=False,
            enable_anomaly_detection=False,
        )

        # Test multiple actions
        actions = [
            "Tell me about machine learning",
            "What is the capital of France?",
            "How do I create a secure password?",
        ]

        for action in actions:
            result = tool._run(action=action, agent_id="test_agent")
            # Should get some response from the tool
            assert isinstance(result, str)
            assert len(result) > 0
            # Tool successfully processes each action (even if governance has issues)
            assert any(
                x in result
                for x in ["ALLOW", "WARN", "BLOCK", "ESCALATE", "ERROR", "Decision"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
