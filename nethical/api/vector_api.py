"""
High-level Vector Language API for Nethical Governance.

This module provides a simplified API matching the problem statement example,
making it easy to use Nethical with vector-based governance and the 25 Fundamental Laws.
"""

from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ..core import IntegratedGovernance, EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Agent model for registration with Nethical.
    
    Attributes:
        id: Unique identifier for the agent
        type: Type of agent (coding, chat, autonomous, etc.)
        capabilities: List of agent capabilities
        metadata: Additional agent metadata
    """
    id: str
    type: str
    capabilities: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EvaluationResult:
    """Result of action evaluation.
    
    This provides a clean interface to evaluation results with easy attribute access.
    """
    decision: str
    laws_evaluated: List[int]
    risk_score: float
    confidence: float
    reasoning: str
    embedding_trace_id: str
    detected_primitives: List[str]
    relevant_laws: List[Dict[str, Any]]
    agent_id: str
    timestamp: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create EvaluationResult from dictionary."""
        return cls(
            decision=data.get("decision", "ALLOW"),
            laws_evaluated=data.get("laws_evaluated", []),
            risk_score=data.get("risk_score", 0.0),
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", ""),
            embedding_trace_id=data.get("embedding_trace_id", ""),
            detected_primitives=data.get("detected_primitives", []),
            relevant_laws=data.get("relevant_laws", []),
            agent_id=data.get("agent_id", ""),
            timestamp=data.get("timestamp", "")
        )


class Nethical:
    """High-level Nethical API with vector-based governance.
    
    This class provides a simplified interface to Nethical's governance features,
    focusing on vector-based evaluation with the 25 Fundamental Laws.
    
    Example:
        >>> nethical = Nethical(config_path="./config/nethical.yaml", enable_25_laws=True)
        >>> agent = Agent(id="agent-001", type="coding", capabilities=["text_generation"])
        >>> nethical.register_agent(agent)
        >>> result = nethical.evaluate(
        ...     agent_id="agent-001",
        ...     action="def greet(name): return 'Hello, ' + name",
        ...     context={"purpose": "demo"}
        ... )
        >>> print(result.decision, result.laws_evaluated, result.risk_score)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        storage_dir: str = "./nethical_data",
        enable_25_laws: bool = True,
        enable_vector_evaluation: bool = True,
        embedding_provider: Optional[EmbeddingProvider] = None,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        """Initialize Nethical governance system.
        
        Args:
            config_path: Path to configuration file (optional)
            storage_dir: Directory for data storage
            enable_25_laws: Enable the 25 Fundamental Laws
            enable_vector_evaluation: Enable vector-based evaluation
            embedding_provider: Custom embedding provider (defaults to SimpleEmbeddingProvider)
            similarity_threshold: Similarity threshold for law matching
            **kwargs: Additional configuration passed to IntegratedGovernance
        """
        self.config_path = config_path
        self.storage_dir = storage_dir
        
        # Load config if provided
        config_params = self._load_config(config_path) if config_path else {}
        
        # Merge with provided kwargs
        config_params.update(kwargs)
        
        # Initialize integrated governance
        self.governance = IntegratedGovernance(
            storage_dir=storage_dir,
            enable_25_laws=enable_25_laws,
            enable_vector_evaluation=enable_vector_evaluation,
            embedding_provider=embedding_provider,
            vector_similarity_threshold=similarity_threshold,
            **config_params
        )
        
        # Track registered agents
        self.agents: Dict[str, Agent] = {}
        
        logger.info(
            f"Nethical initialized with 25 Laws: {enable_25_laws}, "
            f"Vector evaluation: {enable_vector_evaluation}"
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def register_agent(self, agent: Agent) -> bool:
        """Register an agent with Nethical.
        
        Args:
            agent: Agent instance to register
            
        Returns:
            True if registration successful
        """
        if agent.id in self.agents:
            logger.warning(f"Agent {agent.id} already registered, updating...")
        
        self.agents[agent.id] = agent
        logger.info(
            f"Registered agent {agent.id} (type: {agent.type}, "
            f"capabilities: {', '.join(agent.capabilities)})"
        )
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if agent was unregistered
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get registered agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Agent]:
        """List all registered agents."""
        return list(self.agents.values())
    
    def evaluate(
        self,
        agent_id: str,
        action: Union[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate an agent action using vector-based governance.
        
        Args:
            agent_id: ID of agent performing the action
            action: Action to evaluate (text, code, or object)
            context: Additional context for evaluation
            
        Returns:
            EvaluationResult with decision and metadata
            
        Example:
            >>> result = nethical.evaluate(
            ...     agent_id="agent-001",
            ...     action="access user database",
            ...     context={"purpose": "analytics"}
            ... )
            >>> if result.decision == "BLOCK":
            ...     print(f"Action blocked: {result.reasoning}")
        """
        # Check if agent is registered
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not registered, proceeding anyway...")
        
        # Evaluate using governance system
        raw_result = self.governance.evaluate(
            agent_id=agent_id,
            action=action,
            context=context
        )
        
        # Convert to EvaluationResult
        return EvaluationResult.from_dict(raw_result)
    
    def trace_embedding(self, embedding_trace_id: str) -> Optional[Dict[str, Any]]:
        """Trace an embedding decision for audit/debugging.
        
        Args:
            embedding_trace_id: Trace ID from evaluation result
            
        Returns:
            Trace information or None if not found
        """
        return self.governance.trace_embedding(embedding_trace_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dictionary with system stats including:
            - agent_count: Number of registered agents
            - embedding_stats: Embedding engine statistics
            - governance_stats: Governance system statistics
        """
        stats = {
            "agent_count": len(self.agents),
            "agents": [
                {"id": a.id, "type": a.type, "capabilities": a.capabilities}
                for a in self.agents.values()
            ]
        }
        
        # Add embedding stats if available
        if self.governance.embedding_engine:
            stats["embedding_stats"] = self.governance.embedding_engine.get_stats()
        
        # Add governance stats
        stats["governance_enabled"] = {
            "25_laws": self.governance.enable_25_laws,
            "vector_evaluation": self.governance.enable_vector_evaluation,
            "similarity_threshold": self.governance.vector_similarity_threshold
        }
        
        return stats


# Convenience function for quick setup
def create_nethical(
    enable_25_laws: bool = True,
    config_path: Optional[str] = None,
    **kwargs
) -> Nethical:
    """Create a Nethical instance with sensible defaults.
    
    Args:
        enable_25_laws: Enable 25 Fundamental Laws evaluation
        config_path: Optional configuration file path
        **kwargs: Additional configuration options
        
    Returns:
        Configured Nethical instance
    """
    return Nethical(
        enable_25_laws=enable_25_laws,
        enable_vector_evaluation=enable_25_laws,
        config_path=config_path,
        **kwargs
    )
