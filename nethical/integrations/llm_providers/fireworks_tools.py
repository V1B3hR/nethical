"""
Fireworks AI integration with Nethical governance.

Provides a governed wrapper around Fireworks AI's API.
"""

from typing import Any, Dict, List, Optional

from .base import LLMProviderBase, LLMResponse


class FireworksProvider(LLMProviderBase):
    """Fireworks AI integration with Nethical governance.
    
    Wraps Fireworks AI's API with automatic governance checks.
    
    Example:
        provider = FireworksProvider(
            api_key="your-api-key",
            model="accounts/fireworks/models/llama-v3-70b-instruct"
        )
        
        response = provider.safe_generate("Tell me about AI safety")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "accounts/fireworks/models/llama-v3-70b-instruct",
        **kwargs
    ):
        """Initialize Fireworks provider with governance.
        
        Args:
            api_key: Fireworks API key
            model: Model to use
            **kwargs: Additional arguments for LLMProviderBase
        """
        super().__init__(**kwargs)
        
        try:
            import fireworks.client
            fireworks.client.api_key = api_key
            self._fireworks_module = fireworks.client
            self._fireworks_available = True
        except ImportError:
            self._fireworks_available = False
            self._fireworks_module = None
        
        self._model = model
    
    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return f"fireworks-{self._model.split('/')[-1]}"
    
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Fireworks AI's chat API.
        
        Args:
            prompt: The input prompt/message
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse with generated content
        """
        if not self._fireworks_available:
            return LLMResponse(
                content="Fireworks library not installed. Install with: pip install fireworks-ai",
                model=self.model_name,
                usage={}
            )
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self._fireworks_module.ChatCompletion.create(
            model=self._model,
            messages=messages,
            **kwargs
        )
        
        content = response.choices[0].message.content if response.choices else ""
        
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0)
            }
        
        return LLMResponse(
            content=content,
            model=self._model,
            usage=usage
        )
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition for Fireworks AI."""
        return {
            "type": "function",
            "function": {
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
        }


def get_nethical_tool() -> Dict[str, Any]:
    """Get OpenAI-compatible tool definition for Fireworks AI."""
    return {
        "type": "function",
        "function": {
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
    }


def handle_nethical_tool(
    tool_input: Dict[str, Any],
    agent_id: str = "fireworks-agent"
) -> Dict[str, Any]:
    """Handle a Nethical tool call from Fireworks AI."""
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
