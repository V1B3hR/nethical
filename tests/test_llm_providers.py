"""
Tests for LLM Provider integrations.

Tests the Nethical LLM provider integrations including:
- Base class functionality
- Provider implementations
- Governance checks
- Error handling
"""

import pytest
from pathlib import Path
from unittest.mock import Mock


class TestLLMProviderBase:
    """Test LLMProviderBase functionality."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_base_imports(self):
        """Test that base classes can be imported."""
        from nethical.integrations.llm_providers import LLMProviderBase, LLMResponse
        
        assert LLMProviderBase is not None
        assert LLMResponse is not None

    def test_llm_response_dataclass(self):
        """Test LLMResponse dataclass."""
        from nethical.integrations.llm_providers import LLMResponse
        
        response = LLMResponse(
            content="Test content",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20}
        )
        
        assert response.content == "Test content"
        assert response.model == "test-model"
        assert response.usage["input_tokens"] == 10
        assert response.risk_score == 0.0
        assert response.governance_result is None

    def test_custom_provider_implementation(self, temp_storage):
        """Test implementing a custom provider."""
        from nethical.integrations.llm_providers import LLMProviderBase, LLMResponse
        
        class TestProvider(LLMProviderBase):
            @property
            def model_name(self) -> str:
                return "test-model"
            
            def _generate(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(
                    content=f"Response to: {prompt}",
                    model=self.model_name,
                    usage={"input_tokens": len(prompt), "output_tokens": 10}
                )
        
        provider = TestProvider(
            check_input=False,
            check_output=False,
            storage_dir=temp_storage
        )
        
        assert provider.model_name == "test-model"
        
        response = provider._generate("Hello")
        assert "Response to: Hello" in response.content

    def test_provider_info_function(self):
        """Test get_provider_info returns expected structure."""
        from nethical.integrations.llm_providers import get_provider_info
        
        info = get_provider_info()
        
        assert "cohere" in info
        assert "mistral" in info
        assert "together" in info
        assert "fireworks" in info
        assert "groq" in info
        assert "replicate" in info
        
        # Check structure
        for provider_info in info.values():
            assert "available" in provider_info
            assert "setup" in provider_info
            assert "class" in provider_info


class TestCohereProvider:
    """Test Cohere provider functionality."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage_dir = tmp_path / "nethical_test_data"
        storage_dir.mkdir()
        return str(storage_dir)

    def test_cohere_imports(self):
        """Test Cohere module imports."""
        from nethical.integrations.llm_providers.cohere_tools import (
            CohereProvider,
            get_nethical_tool,
            handle_nethical_tool,
        )
        
        assert CohereProvider is not None
        assert callable(get_nethical_tool)
        assert callable(handle_nethical_tool)

    def test_cohere_tool_definition(self):
        """Test Cohere tool definition format."""
        from nethical.integrations.llm_providers.cohere_tools import get_nethical_tool
        
        tool = get_nethical_tool()
        
        assert tool["name"] == "nethical_governance"
        assert "description" in tool
        assert "parameter_definitions" in tool
        assert "action" in tool["parameter_definitions"]

    def test_cohere_provider_model_name(self, temp_storage):
        """Test Cohere provider model name."""
        from nethical.integrations.llm_providers.cohere_tools import CohereProvider
        
        # Create provider (will fail to init client but that's OK)
        provider = CohereProvider(
            api_key="test-key",
            model="command-r-plus",
            check_input=False,
            check_output=False,
            storage_dir=temp_storage
        )
        
        assert provider.model_name == "cohere-command-r-plus"


class TestMistralProvider:
    """Test Mistral provider functionality."""

    def test_mistral_imports(self):
        """Test Mistral module imports."""
        from nethical.integrations.llm_providers.mistral_tools import (
            MistralProvider,
            get_nethical_tool,
            handle_nethical_tool,
        )
        
        assert MistralProvider is not None
        assert callable(get_nethical_tool)

    def test_mistral_tool_definition(self):
        """Test Mistral tool definition format."""
        from nethical.integrations.llm_providers.mistral_tools import get_nethical_tool
        
        tool = get_nethical_tool()
        
        assert tool["type"] == "function"
        assert "function" in tool
        assert tool["function"]["name"] == "nethical_governance"


class TestTogetherProvider:
    """Test Together AI provider functionality."""

    def test_together_imports(self):
        """Test Together module imports."""
        from nethical.integrations.llm_providers.together_tools import (
            TogetherProvider,
            get_nethical_tool,
        )
        
        assert TogetherProvider is not None
        assert callable(get_nethical_tool)


class TestFireworksProvider:
    """Test Fireworks AI provider functionality."""

    def test_fireworks_imports(self):
        """Test Fireworks module imports."""
        from nethical.integrations.llm_providers.fireworks_tools import (
            FireworksProvider,
            get_nethical_tool,
        )
        
        assert FireworksProvider is not None
        assert callable(get_nethical_tool)


class TestGroqProvider:
    """Test Groq provider functionality."""

    def test_groq_imports(self):
        """Test Groq module imports."""
        from nethical.integrations.llm_providers.groq_tools import (
            GroqProvider,
            get_nethical_tool,
        )
        
        assert GroqProvider is not None
        assert callable(get_nethical_tool)


class TestReplicateProvider:
    """Test Replicate provider functionality."""

    def test_replicate_imports(self):
        """Test Replicate module imports."""
        from nethical.integrations.llm_providers.replicate_tools import (
            ReplicateProvider,
            get_nethical_tool,
        )
        
        assert ReplicateProvider is not None
        assert callable(get_nethical_tool)


class TestProviderWithMockedGovernance:
    """Test providers with mocked governance."""

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

    def test_handle_cohere_tool_with_mock(self, temp_storage):
        """Test handle_nethical_tool with mock governance."""
        from nethical.integrations.llm_providers.cohere_tools import CohereProvider
        
        provider = CohereProvider(
            api_key="test-key",
            model="command-r-plus",
            check_input=False,
            check_output=False,
            storage_dir=temp_storage
        )
        
        tool_def = provider.get_tool_definition()
        
        assert tool_def["name"] == "nethical_governance"
        assert "parameter_definitions" in tool_def


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
