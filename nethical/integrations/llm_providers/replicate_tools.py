"""
Replicate integration with Nethical governance.

Provides a governed wrapper around Replicate's API for running various models.
"""

from typing import Any, Dict, List, Optional

from .base import LLMProviderBase, LLMResponse


class ReplicateProvider(LLMProviderBase):
    """Replicate integration with Nethical governance.
    
    Wraps Replicate's API with automatic governance checks.
    Replicate allows running various open-source models.
    
    Example:
        provider = ReplicateProvider(
            api_key="your-api-key",
            model="meta/llama-2-70b-chat"
        )
        
        response = provider.safe_generate("Tell me about AI safety")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-2-70b-chat",
        **kwargs
    ):
        """Initialize Replicate provider with governance.
        
        Args:
            api_key: Replicate API key
            model: Model to use (format: owner/model-name)
            **kwargs: Additional arguments for LLMProviderBase
        """
        super().__init__(**kwargs)
        
        try:
            import replicate
            self._replicate_module = replicate
            # Set the API token
            import os
            os.environ["REPLICATE_API_TOKEN"] = api_key
            self._replicate_available = True
        except ImportError:
            self._replicate_available = False
            self._replicate_module = None
        
        self._model = model
    
    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return f"replicate-{self._model.replace('/', '-')}"
    
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Replicate's API.
        
        Args:
            prompt: The input prompt/message
            **kwargs: Additional arguments for the model
            
        Returns:
            LLMResponse with generated content
        """
        if not self._replicate_available:
            return LLMResponse(
                content="Replicate library not installed. Install with: pip install replicate",
                model=self.model_name,
                usage={}
            )
        
        # Prepare input based on model type
        input_data = {"prompt": prompt}
        input_data.update(kwargs)
        
        # Run the model
        output = self._replicate_module.run(
            self._model,
            input=input_data
        )
        
        # Handle different output formats
        if isinstance(output, list):
            content = "".join(output)
        elif isinstance(output, str):
            content = output
        else:
            content = str(output)
        
        return LLMResponse(
            content=content,
            model=self._model,
            usage={}  # Replicate doesn't provide detailed usage
        )
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for Replicate (generic format)."""
        return {
            "name": "nethical_governance",
            "description": "Evaluate an action for safety, ethics, and compliance",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action or content to evaluate"
                    },
                    "action_type": {
                        "type": "string",
                        "description": "Type of action"
                    }
                },
                "required": ["action"]
            }
        }


def get_nethical_tool() -> Dict[str, Any]:
    """Get tool definition for Replicate."""
    return {
        "name": "nethical_governance",
        "description": "Evaluate an action for safety, ethics, and compliance",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action or content to evaluate"
                },
                "action_type": {
                    "type": "string",
                    "description": "Type of action"
                }
            },
            "required": ["action"]
        }
    }


def handle_nethical_tool(
    tool_input: Dict[str, Any],
    agent_id: str = "replicate-agent"
) -> Dict[str, Any]:
    """Handle a Nethical tool call from Replicate."""
    from nethical.core import IntegratedGovernance
    
    governance = IntegratedGovernance()
    
    action = tool_input.get("action", "")
    action_type = tool_input.get("action_type", "query")
    
    result = governance.process_action(
        action=action,
        agent_id=agent_id,
        action_type=action_type
    )
    
    phase3 = result.get("phase3", {})
    risk_score = phase3.get("risk_score", 0.0)
    
    if risk_score > 0.7:
        decision = "BLOCK"
    elif risk_score > 0.4:
        decision = "RESTRICT"
    else:
        decision = "ALLOW"
    
    return {
        "decision": decision,
        "risk_score": risk_score,
        "risk_tier": phase3.get("risk_tier", "UNKNOWN"),
        "governance_result": result
    }
