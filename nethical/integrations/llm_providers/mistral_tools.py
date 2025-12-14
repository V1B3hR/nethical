"""
Mistral AI integration with Nethical governance.

Provides a governed wrapper around Mistral AI's API.
"""

from typing import Any, Dict, List, Optional

from .base import LLMProviderBase, LLMResponse


class MistralProvider(LLMProviderBase):
    """Mistral AI integration with Nethical governance.
    
    Wraps Mistral's API with automatic governance checks on inputs and outputs.
    
    Example:
        provider = MistralProvider(
            api_key="your-api-key",
            model="mistral-large-latest"
        )
        
        response = provider.safe_generate("Tell me about AI safety")
        print(f"Response: {response.content}")
        print(f"Risk Score: {response.risk_score}")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "mistral-large-latest",
        **kwargs
    ):
        """Initialize Mistral provider with governance.
        
        Args:
            api_key: Mistral API key
            model: Model to use (default: mistral-large-latest)
            **kwargs: Additional arguments for LLMProviderBase
        """
        super().__init__(**kwargs)
        
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=api_key)
            self._mistral_available = True
        except ImportError:
            self._mistral_available = False
            self.client = None
        
        self._model = model
    
    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return f"mistral-{self._model}"
    
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Mistral's chat API.
        
        Args:
            prompt: The input prompt/message
            **kwargs: Additional arguments for Mistral chat
            
        Returns:
            LLMResponse with generated content
        """
        if not self._mistral_available:
            return LLMResponse(
                content="Mistral library not installed. Install with: pip install mistralai",
                model=self.model_name,
                usage={}
            )
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.complete(
            model=self._model,
            messages=messages,
            **kwargs
        )
        
        # Extract content and usage
        content = response.choices[0].message.content if response.choices else ""
        
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return LLMResponse(
            content=content,
            model=self._model,
            usage=usage
        )
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get Mistral-compatible tool definition for Nethical.
        
        Returns:
            Mistral-compatible tool definition
        """
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
    """Get Mistral-compatible tool definition for Nethical."""
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
    agent_id: str = "mistral-agent"
) -> Dict[str, Any]:
    """Handle a Nethical tool call from Mistral."""
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
