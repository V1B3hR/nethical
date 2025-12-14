"""
Together AI integration with Nethical governance.

Provides a governed wrapper around Together AI's API.
"""


from .base import LLMProviderBase, LLMResponse


class TogetherProvider(LLMProviderBase):
    """Together AI integration with Nethical governance.
    
    Wraps Together AI's API with automatic governance checks.
    
    Example:
        provider = TogetherProvider(
            api_key="your-api-key",
            model="meta-llama/Llama-3-70b-chat-hf"
        )
        
        response = provider.safe_generate("Tell me about AI safety")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
        **kwargs
    ):
        """Initialize Together provider with governance.
        
        Args:
            api_key: Together API key
            model: Model to use
            **kwargs: Additional arguments for LLMProviderBase
        """
        super().__init__(**kwargs)
        
        try:
            from together import Together
            self.client = Together(api_key=api_key)
            self._together_available = True
        except ImportError:
            self._together_available = False
            self.client = None
        
        self._model = model
    
    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return f"together-{self._model}"
    
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Together AI's chat API.
        
        Args:
            prompt: The input prompt/message
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse with generated content
        """
        if not self._together_available:
            return LLMResponse(
                content="Together library not installed. Install with: pip install together",
                model=self.model_name,
                usage={}
            )
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs
        )
        
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
        """Get OpenAI-compatible tool definition for Together AI."""
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
    """Get OpenAI-compatible tool definition for Together AI."""
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
    agent_id: str = "together-agent"
) -> Dict[str, Any]:
    """Handle a Nethical tool call from Together AI."""
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
