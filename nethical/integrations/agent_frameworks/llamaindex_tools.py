"""
LlamaIndex integration with Nethical governance.

Provides governed wrappers for LlamaIndex tools and query engines.
"""

from typing import Any, Dict, List, Optional

from .base import AgentFrameworkBase, GovernanceResult, GovernanceDecision


# Check for LlamaIndex availability
try:
    from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    BaseTool = object
    ToolMetadata = None
    ToolOutput = None


class NethicalLlamaIndexTool:
    """LlamaIndex tool for Nethical governance.
    
    This tool can be added to LlamaIndex agents to enable governance
    checks during agent execution.
    
    Example:
        from nethical.integrations.agent_frameworks import NethicalLlamaIndexTool
        
        tool = NethicalLlamaIndexTool(block_threshold=0.7)
        
        # Add to LlamaIndex agent
        agent = ReActAgent.from_tools([tool, ...other_tools...])
        
        # Or use directly
        result = tool("Check if this action is safe: delete user data")
    """
    
    def __init__(
        self,
        storage_dir: str = "./nethical_data",
        block_threshold: float = 0.7,
        restrict_threshold: float = 0.4
    ):
        """Initialize the LlamaIndex tool.
        
        Args:
            storage_dir: Directory for Nethical data storage
            block_threshold: Risk threshold for blocking
            restrict_threshold: Risk threshold for restriction
        """
        self.storage_dir = storage_dir
        self.block_threshold = block_threshold
        self.restrict_threshold = restrict_threshold
        
        self._governance = None
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(storage_dir=self.storage_dir)
        return self._governance
    
    @property
    def metadata(self):
        """Get tool metadata for LlamaIndex."""
        if not LLAMAINDEX_AVAILABLE:
            return None
        
        return ToolMetadata(
            name="nethical_governance",
            description=(
                "Evaluate an action for safety, ethics, and compliance. "
                "Use this tool to check if an action is allowed before executing it."
            ),
            fn_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "The action to evaluate"},
                    "action_type": {"type": "string", "description": "Type of action"}
                },
                "required": ["action"]
            }
        )
    
    def __call__(self, action: str, action_type: str = "query"):
        """Evaluate an action through governance.
        
        Args:
            action: The action to evaluate
            action_type: Type of action
            
        Returns:
            ToolOutput with governance decision (if LlamaIndex available)
            or dict with result
        """
        result = self.governance.process_action(
            action=action,
            agent_id="llamaindex-agent",
            action_type=action_type
        )
        
        decision = self._compute_decision(result)
        risk_score = result.get("phase3", {}).get("risk_score", 0.0)
        
        content = f"Decision: {decision} | Risk: {risk_score:.2f}"
        
        if LLAMAINDEX_AVAILABLE and ToolOutput is not None:
            return ToolOutput(
                content=content,
                tool_name="nethical_governance",
                raw_input={"action": action, "action_type": action_type},
                raw_output=result
            )
        
        return {
            "decision": decision,
            "risk_score": risk_score,
            "content": content,
            "raw_output": result
        }
    
    def _compute_decision(self, result: Dict[str, Any]) -> str:
        """Compute decision from governance result."""
        risk = result.get("phase3", {}).get("risk_score", 0)
        
        if risk > self.block_threshold:
            return "BLOCK"
        elif risk > self.restrict_threshold:
            return "RESTRICT"
        return "ALLOW"


class NethicalQueryEngine:
    """Wrapper for LlamaIndex query engines with governance.
    
    Wraps any LlamaIndex query engine with pre- and post-query
    governance checks.
    
    Example:
        from llama_index.core import VectorStoreIndex
        from nethical.integrations.agent_frameworks import NethicalQueryEngine
        
        # Create your index
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
        # Wrap with governance
        safe_engine = NethicalQueryEngine(
            query_engine=query_engine,
            check_query=True,
            check_response=True
        )
        
        # Query with governance
        response = safe_engine.query("What is the company policy?")
    """
    
    def __init__(
        self,
        query_engine: Any,
        check_query: bool = True,
        check_response: bool = True,
        block_threshold: float = 0.7,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the governed query engine.
        
        Args:
            query_engine: LlamaIndex query engine to wrap
            check_query: Enable query governance checks
            check_response: Enable response governance checks
            block_threshold: Risk threshold for blocking
            storage_dir: Directory for Nethical data storage
        """
        self.query_engine = query_engine
        self.check_query = check_query
        self.check_response = check_response
        self.block_threshold = block_threshold
        self.storage_dir = storage_dir
        
        self._governance = None
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(storage_dir=self.storage_dir)
        return self._governance
    
    def query(self, query_str: str) -> Any:
        """Execute a query with governance checks.
        
        Args:
            query_str: The query string
            
        Returns:
            Query response (possibly filtered)
        """
        # Check query
        if self.check_query:
            query_result = self.governance.process_action(
                action=query_str,
                agent_id="llamaindex-query",
                action_type="search_query"
            )
            
            query_risk = query_result.get("phase3", {}).get("risk_score", 0.0)
            
            if query_risk > self.block_threshold:
                reason = query_result.get("reason", "High risk query")
                
                # Return blocked response
                if LLAMAINDEX_AVAILABLE:
                    try:
                        from llama_index.core.response.schema import Response
                        return Response(
                            response=f"Query blocked: {reason}",
                            source_nodes=[],
                            metadata={"governance": query_result}
                        )
                    except ImportError:
                        pass
                
                return {"blocked": True, "reason": reason, "governance": query_result}
        
        # Execute query
        response = self.query_engine.query(query_str)
        
        # Check response
        if self.check_response:
            response_text = str(response)
            
            response_result = self.governance.process_action(
                action=response_text,
                agent_id="llamaindex-query",
                action_type="generated_content"
            )
            
            response_risk = response_result.get("phase3", {}).get("risk_score", 0.0)
            
            if response_risk > self.block_threshold:
                reason = response_result.get("reason", "High risk response")
                
                # Modify response if possible
                if hasattr(response, 'response'):
                    response.response = f"Response filtered: {reason}"
                if hasattr(response, 'metadata'):
                    if response.metadata is None:
                        response.metadata = {}
                    response.metadata["governance"] = response_result
        
        return response


def create_safe_index(index: Any, **kwargs) -> NethicalQueryEngine:
    """Wrap a LlamaIndex index with Nethical governance.
    
    Convenience function to quickly wrap an index.
    
    Args:
        index: LlamaIndex index to wrap
        **kwargs: Arguments for NethicalQueryEngine
        
    Returns:
        NethicalQueryEngine wrapping the index's query engine
    """
    query_engine = index.as_query_engine()
    return NethicalQueryEngine(query_engine, **kwargs)


class LlamaIndexFramework(AgentFrameworkBase):
    """LlamaIndex framework integration with Nethical.
    
    Provides governance tools and utilities for LlamaIndex agents.
    """
    
    def __init__(self, **kwargs):
        """Initialize the LlamaIndex framework integration."""
        super().__init__(agent_id="llamaindex-framework", **kwargs)
    
    def get_tool(self) -> NethicalLlamaIndexTool:
        """Get a LlamaIndex-compatible governance tool.
        
        Returns:
            NethicalLlamaIndexTool instance
        """
        return NethicalLlamaIndexTool(
            storage_dir=self.storage_dir,
            block_threshold=self.block_threshold,
            restrict_threshold=self.restrict_threshold
        )
    
    def wrap_query_engine(self, query_engine: Any, **kwargs) -> NethicalQueryEngine:
        """Wrap a query engine with governance.
        
        Args:
            query_engine: Query engine to wrap
            **kwargs: Additional arguments for NethicalQueryEngine
            
        Returns:
            Governed query engine
        """
        return NethicalQueryEngine(
            query_engine=query_engine,
            block_threshold=self.block_threshold,
            storage_dir=self.storage_dir,
            **kwargs
        )
