"""
Cohere integration with Nethical governance.

Provides a governed wrapper around Cohere's API including:
- Chat generation with safety checks
- Rerank with content filtering
- Tool definitions for function calling
"""

from typing import Any, Dict, List, Optional

from .base import LLMProviderBase, LLMResponse


class CohereProvider(LLMProviderBase):
    """Cohere integration with Nethical governance.
    
    Wraps Cohere's API with automatic governance checks on inputs and outputs.
    Supports both chat and rerank operations.
    
    Example:
        provider = CohereProvider(
            api_key="your-api-key",
            model="command-r-plus",
            check_input=True,
            check_output=True
        )
        
        # Safe chat generation
        response = provider.safe_generate("Tell me about AI safety")
        print(f"Response: {response.content}")
        print(f"Risk Score: {response.risk_score}")
        
        # Safe reranking
        results = provider.safe_rerank(
            query="AI safety",
            documents=["Doc 1...", "Doc 2..."],
            top_n=5
        )
    
    Attributes:
        client: Cohere client instance
        _model: Model name for generation
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "command-r-plus",
        **kwargs
    ):
        """Initialize Cohere provider with governance.
        
        Args:
            api_key: Cohere API key
            model: Model to use for generation (default: command-r-plus)
            **kwargs: Additional arguments for LLMProviderBase
        """
        super().__init__(**kwargs)
        
        try:
            import cohere
            self.client = cohere.Client(api_key)
            self._cohere_available = True
        except ImportError:
            self._cohere_available = False
            self.client = None
        
        self._model = model
    
    @property
    def model_name(self) -> str:
        """Get the model identifier."""
        return f"cohere-{self._model}"
    
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Cohere's chat API.
        
        Args:
            prompt: The input prompt/message
            **kwargs: Additional arguments for Cohere chat
            
        Returns:
            LLMResponse with generated content
        """
        if not self._cohere_available:
            return LLMResponse(
                content="Cohere library not installed. Install with: pip install cohere",
                model=self.model_name,
                usage={}
            )
        
        response = self.client.chat(
            model=self._model,
            message=prompt,
            **kwargs
        )
        
        # Extract usage information
        usage = {}
        if hasattr(response, 'meta') and hasattr(response.meta, 'tokens'):
            tokens = response.meta.tokens
            if hasattr(tokens, 'input_tokens'):
                usage["input_tokens"] = tokens.input_tokens
            if hasattr(tokens, 'output_tokens'):
                usage["output_tokens"] = tokens.output_tokens
        
        return LLMResponse(
            content=response.text,
            model=self._model,
            usage=usage
        )
    
    def safe_rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
        rerank_model: str = "rerank-v3.5"
    ) -> List[Dict[str, Any]]:
        """Rerank documents with governance checks.
        
        Checks the query for safety and filters results based on
        document content governance checks.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return
            rerank_model: Cohere rerank model to use
            
        Returns:
            List of safe reranked results with governance info
        """
        if not self._cohere_available:
            return []
        
        # Check query for safety
        if self.check_input:
            query_result = self._check_governance(query, "search_query")
            query_risk = self._get_risk_score(query_result)
            
            if query_risk > self.block_threshold:
                return []
        
        # Perform reranking
        response = self.client.rerank(
            model=rerank_model,
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        # Filter results based on governance
        safe_results = []
        for result in response.results:
            doc = documents[result.index]
            
            if self.check_output:
                doc_result = self._check_governance(doc, "retrieved_content")
                doc_risk = self._get_risk_score(doc_result)
                
                if doc_risk > self.block_threshold:
                    continue
                
                safe_results.append({
                    "index": result.index,
                    "document": doc,
                    "relevance_score": result.relevance_score,
                    "governance": doc_result,
                    "risk_score": doc_risk
                })
            else:
                safe_results.append({
                    "index": result.index,
                    "document": doc,
                    "relevance_score": result.relevance_score
                })
        
        return safe_results
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get Cohere-compatible tool definition for Nethical governance.
        
        Returns a tool definition in Cohere's function calling format.
        
        Returns:
            Cohere-compatible tool definition
        """
        return {
            "name": "nethical_governance",
            "description": "Evaluate an action for safety, ethics, and compliance",
            "parameter_definitions": {
                "action": {
                    "type": "str",
                    "description": "The action or content to evaluate",
                    "required": True
                },
                "action_type": {
                    "type": "str",
                    "description": "Type of action (query, code_generation, data_access, etc.)",
                    "required": False
                }
            }
        }


def get_nethical_tool() -> Dict[str, Any]:
    """Get Cohere-compatible tool definition for Nethical.
    
    Convenience function to get the tool definition without
    instantiating a provider.
    
    Returns:
        Cohere-compatible tool definition
    """
    return {
        "name": "nethical_governance",
        "description": "Evaluate an action for safety, ethics, and compliance",
        "parameter_definitions": {
            "action": {
                "type": "str",
                "description": "The action or content to evaluate",
                "required": True
            },
            "action_type": {
                "type": "str",
                "description": "Type of action",
                "required": False
            }
        }
    }


def handle_nethical_tool(
    tool_input: Dict[str, Any],
    agent_id: str = "cohere-agent"
) -> Dict[str, Any]:
    """Handle a Nethical tool call from Cohere.
    
    Args:
        tool_input: The tool input from Cohere
        agent_id: Agent identifier for governance
        
    Returns:
        Result dictionary with decision and details
    """
    from nethical.core import IntegratedGovernance
    
    governance = IntegratedGovernance()
    
    action = tool_input.get("action", "")
    action_type = tool_input.get("action_type", "query")
    
    result = governance.process_action(
        action=action,
        agent_id=agent_id,
        action_type=action_type
    )
    
    # Extract decision
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
