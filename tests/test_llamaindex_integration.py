"""
Tests for LlamaIndex integration specifically.

Tests the LlamaIndex-specific features including:
- Tool creation and metadata
- Query engine wrapping
- Index wrapping utilities
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestNethicalLlamaIndexTool:
    """Test NethicalLlamaIndexTool functionality."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_tool_creation(self, temp_storage):
        """Test creating the LlamaIndex tool."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalLlamaIndexTool,
        )
        
        tool = NethicalLlamaIndexTool(
            storage_dir=temp_storage,
            block_threshold=0.7,
            restrict_threshold=0.4
        )
        
        assert tool.storage_dir == temp_storage
        assert tool.block_threshold == 0.7
        assert tool.restrict_threshold == 0.4

    def test_tool_threshold_decisions(self, temp_storage):
        """Test tool decision logic with different risk scores."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalLlamaIndexTool,
        )
        
        tool = NethicalLlamaIndexTool(
            storage_dir=temp_storage,
            block_threshold=0.7,
            restrict_threshold=0.4
        )
        
        # High risk -> BLOCK
        result = {"phase3": {"risk_score": 0.85}}
        assert tool._compute_decision(result) == "BLOCK"
        
        # Medium risk -> RESTRICT
        result = {"phase3": {"risk_score": 0.55}}
        assert tool._compute_decision(result) == "RESTRICT"
        
        # Low risk -> ALLOW
        result = {"phase3": {"risk_score": 0.2}}
        assert tool._compute_decision(result) == "ALLOW"
        
        # Zero risk -> ALLOW
        result = {"phase3": {"risk_score": 0.0}}
        assert tool._compute_decision(result) == "ALLOW"
        
        # Boundary tests
        result = {"phase3": {"risk_score": 0.7}}
        assert tool._compute_decision(result) == "RESTRICT"  # Exactly at block threshold
        
        result = {"phase3": {"risk_score": 0.4}}
        assert tool._compute_decision(result) == "ALLOW"  # Exactly at restrict threshold

    def test_tool_callable_with_mock(self, temp_storage):
        """Test tool is callable and returns expected format."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalLlamaIndexTool,
        )
        from nethical.core import IntegratedGovernance
        
        tool = NethicalLlamaIndexTool(storage_dir=temp_storage)
        
        # Mock governance
        mock_gov = Mock(spec=IntegratedGovernance)
        mock_gov.process_action.return_value = {
            "phase3": {"risk_score": 0.2, "risk_tier": "LOW"},
            "phase4": {},
        }
        tool._governance = mock_gov
        
        result = tool("Test action", "query")
        
        mock_gov.process_action.assert_called_once()
        
        # Check result format (dict since LlamaIndex not installed)
        assert "decision" in result or hasattr(result, 'content')


class TestNethicalQueryEngine:
    """Test NethicalQueryEngine functionality."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    @pytest.fixture
    def mock_query_engine(self):
        """Create mock query engine."""
        engine = Mock()
        engine.query.return_value = Mock(
            response="Test response",
            source_nodes=[],
            metadata={}
        )
        return engine

    def test_query_engine_wrapper_creation(self, temp_storage, mock_query_engine):
        """Test creating query engine wrapper."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalQueryEngine,
        )
        
        wrapper = NethicalQueryEngine(
            query_engine=mock_query_engine,
            check_query=True,
            check_response=True,
            block_threshold=0.7,
            storage_dir=temp_storage
        )
        
        assert wrapper.check_query is True
        assert wrapper.check_response is True
        assert wrapper.block_threshold == 0.7

    def test_query_engine_query_passthrough(self, temp_storage, mock_query_engine):
        """Test query passthrough when governance disabled."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalQueryEngine,
        )
        
        wrapper = NethicalQueryEngine(
            query_engine=mock_query_engine,
            check_query=False,
            check_response=False,
            storage_dir=temp_storage
        )
        
        result = wrapper.query("Test query")
        
        mock_query_engine.query.assert_called_once_with("Test query")

    def test_query_engine_with_governance(self, temp_storage, mock_query_engine):
        """Test query engine with governance enabled."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            NethicalQueryEngine,
        )
        from nethical.core import IntegratedGovernance
        
        wrapper = NethicalQueryEngine(
            query_engine=mock_query_engine,
            check_query=True,
            check_response=True,
            block_threshold=0.7,
            storage_dir=temp_storage
        )
        
        # Mock governance with low risk
        mock_gov = Mock(spec=IntegratedGovernance)
        mock_gov.process_action.return_value = {
            "phase3": {"risk_score": 0.2, "risk_tier": "LOW"},
        }
        wrapper._governance = mock_gov
        
        result = wrapper.query("Test query")
        
        # Should have called governance for query check
        assert mock_gov.process_action.called


class TestLlamaIndexFramework:
    """Test LlamaIndexFramework functionality."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_framework_creation(self, temp_storage):
        """Test creating framework instance."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            LlamaIndexFramework,
        )
        
        framework = LlamaIndexFramework(
            block_threshold=0.7,
            restrict_threshold=0.4,
            storage_dir=temp_storage
        )
        
        assert framework.block_threshold == 0.7
        assert framework.restrict_threshold == 0.4

    def test_framework_get_tool(self, temp_storage):
        """Test framework get_tool method."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            LlamaIndexFramework,
            NethicalLlamaIndexTool,
        )
        
        framework = LlamaIndexFramework(
            block_threshold=0.8,
            storage_dir=temp_storage
        )
        
        tool = framework.get_tool()
        
        assert isinstance(tool, NethicalLlamaIndexTool)
        assert tool.block_threshold == 0.8

    def test_framework_wrap_query_engine(self, temp_storage):
        """Test framework wrap_query_engine method."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            LlamaIndexFramework,
            NethicalQueryEngine,
        )
        
        framework = LlamaIndexFramework(
            block_threshold=0.8,
            storage_dir=temp_storage
        )
        
        mock_engine = Mock()
        
        wrapped = framework.wrap_query_engine(
            mock_engine,
            check_query=True,
            check_response=False
        )
        
        assert isinstance(wrapped, NethicalQueryEngine)
        assert wrapped.check_query is True
        assert wrapped.check_response is False


class TestCreateSafeIndex:
    """Test create_safe_index utility function."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_create_safe_index_function(self, temp_storage):
        """Test create_safe_index utility."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            create_safe_index,
            NethicalQueryEngine,
        )
        
        # Mock index
        mock_index = Mock()
        mock_index.as_query_engine.return_value = Mock()
        
        result = create_safe_index(
            mock_index,
            check_query=True,
            check_response=True,
            block_threshold=0.7,
            storage_dir=temp_storage
        )
        
        mock_index.as_query_engine.assert_called_once()
        assert isinstance(result, NethicalQueryEngine)


class TestLlamaIndexAvailability:
    """Test LlamaIndex availability detection."""

    def test_llamaindex_availability_flag(self):
        """Test LLAMAINDEX_AVAILABLE flag exists."""
        from nethical.integrations.agent_frameworks.llamaindex_tools import (
            LLAMAINDEX_AVAILABLE,
        )
        
        # Should be a boolean
        assert isinstance(LLAMAINDEX_AVAILABLE, bool)

    def test_module_exports(self):
        """Test module exports expected classes."""
        from nethical.integrations.agent_frameworks import llamaindex_tools
        
        assert hasattr(llamaindex_tools, 'NethicalLlamaIndexTool')
        assert hasattr(llamaindex_tools, 'NethicalQueryEngine')
        assert hasattr(llamaindex_tools, 'LlamaIndexFramework')
        assert hasattr(llamaindex_tools, 'create_safe_index')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
