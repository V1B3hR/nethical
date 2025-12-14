"""
Tests for Agent Framework integrations.

Tests the Nethical agent framework integrations including:
- Base class functionality
- LlamaIndex integration
- CrewAI integration
- DSPy integration
- AutoGen integration
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestAgentFrameworkBase:
    """Test AgentFrameworkBase functionality."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_base_imports(self):
        """Test that base classes can be imported."""
        from nethical.integrations.agent_frameworks import (
            AgentFrameworkBase,
            AgentWrapper,
            GovernanceDecision,
            GovernanceResult,
        )
        
        assert AgentFrameworkBase is not None
        assert AgentWrapper is not None
        assert GovernanceDecision is not None
        assert GovernanceResult is not None

    def test_governance_decision_enum(self):
        """Test GovernanceDecision enum values."""
        from nethical.integrations.agent_frameworks import GovernanceDecision
        
        assert GovernanceDecision.ALLOW.value == "ALLOW"
        assert GovernanceDecision.RESTRICT.value == "RESTRICT"
        assert GovernanceDecision.BLOCK.value == "BLOCK"
        assert GovernanceDecision.ESCALATE.value == "ESCALATE"

    def test_governance_result_dataclass(self):
        """Test GovernanceResult dataclass."""
        from nethical.integrations.agent_frameworks import (
            GovernanceResult,
            GovernanceDecision,
        )
        
        result = GovernanceResult(
            decision=GovernanceDecision.ALLOW,
            risk_score=0.2,
            reason="Low risk action",
            details={"phase3": {"risk_tier": "LOW"}}
        )
        
        assert result.decision == GovernanceDecision.ALLOW
        assert result.risk_score == 0.2
        assert result.reason == "Low risk action"
        assert "phase3" in result.details

    def test_framework_info_function(self):
        """Test get_framework_info returns expected structure."""
        from nethical.integrations.agent_frameworks import get_framework_info
        
        info = get_framework_info()
        
        assert "llamaindex" in info
        assert "crewai" in info
        assert "dspy" in info
        assert "autogen" in info
        
        # Check structure
        for framework_info in info.values():
            assert "available" in framework_info
            assert "setup" in framework_info
            assert "classes" in framework_info


class TestLlamaIndexIntegration:
    """Test LlamaIndex integration."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_llamaindex_imports(self):
        """Test LlamaIndex module imports."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalLlamaIndexTool,
            NethicalQueryEngine,
            LlamaIndexFramework,
        )
        
        assert NethicalLlamaIndexTool is not None
        assert NethicalQueryEngine is not None
        assert LlamaIndexFramework is not None

    def test_llamaindex_tool_creation(self, temp_storage):
        """Test creating LlamaIndex tool."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalLlamaIndexTool,
        )
        
        tool = NethicalLlamaIndexTool(
            storage_dir=temp_storage,
            block_threshold=0.7,
            restrict_threshold=0.4
        )
        
        assert tool.block_threshold == 0.7
        assert tool.restrict_threshold == 0.4

    def test_llamaindex_tool_compute_decision(self, temp_storage):
        """Test LlamaIndex tool decision computation."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalLlamaIndexTool,
        )
        
        tool = NethicalLlamaIndexTool(
            storage_dir=temp_storage,
            block_threshold=0.7,
            restrict_threshold=0.4
        )
        
        # Test BLOCK
        assert tool._compute_decision({"phase3": {"risk_score": 0.8}}) == "BLOCK"
        
        # Test RESTRICT
        assert tool._compute_decision({"phase3": {"risk_score": 0.5}}) == "RESTRICT"
        
        # Test ALLOW
        assert tool._compute_decision({"phase3": {"risk_score": 0.2}}) == "ALLOW"


class TestCrewAIIntegration:
    """Test CrewAI integration."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_crewai_imports(self):
        """Test CrewAI module imports."""
        from nethical.integrations.agent_frameworks.crewai_tools import (
            NethicalCrewAITool,
            NethicalAgentWrapper,
            CrewAIFramework,
        )
        
        assert NethicalCrewAITool is not None
        assert NethicalAgentWrapper is not None
        assert CrewAIFramework is not None

    def test_crewai_tool_creation(self, temp_storage):
        """Test creating CrewAI tool."""
        from nethical.integrations.agent_frameworks.crewai_tools import (
            NethicalCrewAITool,
        )
        
        tool = NethicalCrewAITool(
            block_threshold=0.7,
            restrict_threshold=0.4,
            storage_dir=temp_storage
        )
        
        assert tool.name == "nethical_governance"
        assert tool.block_threshold == 0.7

    def test_crewai_tool_callable(self, temp_storage):
        """Test CrewAI tool is callable."""
        from nethical.integrations.agent_frameworks.crewai_tools import (
            NethicalCrewAITool,
        )
        
        tool = NethicalCrewAITool(storage_dir=temp_storage)
        
        # Tool should be callable
        assert callable(tool)


class TestDSPyIntegration:
    """Test DSPy integration."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_dspy_imports(self):
        """Test DSPy module imports."""
        from nethical.integrations.agent_frameworks.dspy_tools import (
            NethicalModule,
            GovernedChainOfThought,
            GovernedPredict,
            DSPyFramework,
        )
        
        assert NethicalModule is not None
        assert GovernedChainOfThought is not None
        assert GovernedPredict is not None
        assert DSPyFramework is not None

    def test_dspy_module_creation(self, temp_storage):
        """Test creating DSPy module."""
        from nethical.integrations.agent_frameworks.dspy_tools import (
            NethicalModule,
        )
        
        module = NethicalModule(
            block_threshold=0.7,
            restrict_threshold=0.4,
            storage_dir=temp_storage
        )
        
        assert module.block_threshold == 0.7
        assert module.restrict_threshold == 0.4

    def test_dspy_module_decision_logic(self, temp_storage):
        """Test DSPy module decision logic."""
        from nethical.integrations.agent_frameworks.dspy_tools import (
            NethicalModule,
        )
        
        module = NethicalModule(
            block_threshold=0.7,
            restrict_threshold=0.4,
            storage_dir=temp_storage
        )
        
        # Test decisions
        assert module._get_decision(0.8) == "BLOCK"
        assert module._get_decision(0.5) == "RESTRICT"
        assert module._get_decision(0.2) == "ALLOW"


class TestAutoGenIntegration:
    """Test AutoGen integration."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_autogen_imports(self):
        """Test AutoGen module imports."""
        from nethical.integrations.agent_frameworks.autogen_tools import (
            NethicalAutoGenTool,
            NethicalConversableAgent,
            GovernedGroupChat,
            AutoGenFramework,
        )
        
        assert NethicalAutoGenTool is not None
        assert NethicalConversableAgent is not None
        assert GovernedGroupChat is not None
        assert AutoGenFramework is not None

    def test_autogen_tool_creation(self, temp_storage):
        """Test creating AutoGen tool."""
        from nethical.integrations.agent_frameworks.autogen_tools import (
            NethicalAutoGenTool,
        )
        
        tool = NethicalAutoGenTool(
            block_threshold=0.7,
            restrict_threshold=0.4,
            storage_dir=temp_storage
        )
        
        assert tool.block_threshold == 0.7
        assert tool.restrict_threshold == 0.4

    def test_autogen_function_config(self, temp_storage):
        """Test AutoGen function configuration."""
        from nethical.integrations.agent_frameworks.autogen_tools import (
            NethicalAutoGenTool,
        )
        
        tool = NethicalAutoGenTool(storage_dir=temp_storage)
        config = tool.get_function_config()
        
        assert config["name"] == "nethical_check"
        assert "description" in config
        assert "parameters" in config

    def test_autogen_get_nethical_function(self):
        """Test get_nethical_function helper."""
        from nethical.integrations.agent_frameworks.autogen_tools import (
            get_nethical_function,
        )
        
        func_def = get_nethical_function()
        
        assert func_def["name"] == "nethical_check"
        assert "parameters" in func_def


class TestFrameworkWithMockedGovernance:
    """Test frameworks with mocked governance."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    @pytest.fixture
    def mock_governance(self):
        """Create mock governance."""
        from nethical.core import IntegratedGovernance
        gov = Mock(spec=IntegratedGovernance)
        gov.process_action = Mock(return_value={
            "phase3": {"risk_score": 0.2, "risk_tier": "LOW"},
            "phase4": {},
            "phase567": {},
            "phase89": {},
        })
        return gov

    def test_llamaindex_tool_with_mock(self, temp_storage, mock_governance):
        """Test LlamaIndex tool with mock governance."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalLlamaIndexTool,
        )
        
        tool = NethicalLlamaIndexTool(storage_dir=temp_storage)
        tool._governance = mock_governance
        
        result = tool("Test action", "query")
        
        mock_governance.process_action.assert_called_once()
        assert "decision" in str(result).lower() or "Decision" in str(result)

    def test_crewai_tool_with_mock(self, temp_storage, mock_governance):
        """Test CrewAI tool with mock governance."""
        from nethical.integrations.agent_frameworks.crewai_tools import (
            NethicalCrewAITool,
        )
        
        tool = NethicalCrewAITool(storage_dir=temp_storage)
        tool._governance = mock_governance
        
        result = tool._run("Test action")
        
        mock_governance.process_action.assert_called_once()
        assert "ALLOW" in result

    def test_dspy_module_with_mock(self, temp_storage, mock_governance):
        """Test DSPy module with mock governance."""
        from nethical.integrations.agent_frameworks.dspy_tools import (
            NethicalModule,
        )
        
        module = NethicalModule(storage_dir=temp_storage)
        module._governance = mock_governance
        
        result = module.check("Test content", "query")
        
        mock_governance.process_action.assert_called_once()
        assert result["allowed"] is True
        assert result["decision"] == "ALLOW"

    def test_autogen_tool_with_mock(self, temp_storage, mock_governance):
        """Test AutoGen tool with mock governance."""
        from nethical.integrations.agent_frameworks.autogen_tools import (
            NethicalAutoGenTool,
        )
        
        tool = NethicalAutoGenTool(storage_dir=temp_storage)
        tool._governance = mock_governance
        
        result = tool.check("Test action", "query")
        
        mock_governance.process_action.assert_called_once()
        assert result["allowed"] is True
        assert result["decision"] == "ALLOW"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
