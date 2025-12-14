"""
CrewAI integration with Nethical governance.

Provides governed wrappers for CrewAI agents and tools.
"""

from typing import Any, Dict, Optional

from .base import AgentFrameworkBase, AgentWrapper


# Check for CrewAI availability
try:
    from crewai import Tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Tool = None


class NethicalCrewAITool:
    """CrewAI tool for Nethical governance.
    
    This tool can be added to CrewAI agents to enable governance
    checks during task execution.
    
    Example:
        from nethical.integrations.agent_frameworks import NethicalCrewAITool
        from crewai import Agent
        
        # Create governance tool
        governance_tool = NethicalCrewAITool(block_threshold=0.7)
        
        # Create agent with governance tool
        agent = Agent(
            role="researcher",
            tools=[governance_tool.as_crewai_tool()],
            ...
        )
    """
    
    name: str = "nethical_governance"
    description: str = (
        "Evaluate an action for safety and ethics compliance. "
        "Use before executing potentially sensitive actions."
    )
    
    def __init__(
        self,
        block_threshold: float = 0.7,
        restrict_threshold: float = 0.4,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the CrewAI tool.
        
        Args:
            block_threshold: Risk threshold for blocking
            restrict_threshold: Risk threshold for restriction
            storage_dir: Directory for Nethical data storage
        """
        self.block_threshold = block_threshold
        self.restrict_threshold = restrict_threshold
        self.storage_dir = storage_dir
        
        self._governance = None
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(storage_dir=self.storage_dir)
        return self._governance
    
    def _run(self, action: str) -> str:
        """Execute governance check on an action.
        
        Args:
            action: The action to evaluate
            
        Returns:
            Decision string with details
        """
        result = self.governance.process_action(
            action=action,
            agent_id="crewai-agent",
            action_type="agent_action"
        )
        
        risk = result.get("phase3", {}).get("risk_score", 0)
        reason = result.get("reason", "")
        
        if risk > self.block_threshold:
            return f"BLOCK: Action not allowed. Risk: {risk:.2f}. Reason: {reason}"
        elif risk > self.restrict_threshold:
            return f"RESTRICT: Proceed with caution. Risk: {risk:.2f}"
        return f"ALLOW: Action permitted. Risk: {risk:.2f}"
    
    def __call__(self, action: str) -> str:
        """Make the tool callable."""
        return self._run(action)
    
    def as_crewai_tool(self):
        """Convert to a CrewAI Tool object.
        
        Returns:
            CrewAI Tool object (or None if CrewAI not available)
        """
        if not CREWAI_AVAILABLE or Tool is None:
            return None
        
        return Tool(
            name=self.name,
            description=self.description,
            func=self._run
        )


class NethicalAgentWrapper(AgentWrapper):
    """Wrap CrewAI agents with governance.
    
    Provides pre- and post-task governance checks for CrewAI agents.
    
    Example:
        from crewai import Agent
        from nethical.integrations.agent_frameworks import NethicalAgentWrapper
        
        # Create your agent
        agent = Agent(role="researcher", ...)
        
        # Wrap with governance
        wrapped_agent = NethicalAgentWrapper(
            agent=agent,
            pre_check=True,
            post_check=True
        )
        
        # Execute task with governance
        result = wrapped_agent.execute_task(task)
    """
    
    def __init__(
        self,
        agent: Any,
        pre_check: bool = True,
        post_check: bool = True,
        block_threshold: float = 0.7,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the agent wrapper.
        
        Args:
            agent: CrewAI agent to wrap
            pre_check: Enable pre-task governance checks
            post_check: Enable post-task governance checks
            block_threshold: Risk threshold for blocking
            storage_dir: Directory for Nethical data storage
        """
        # Get agent role for ID
        role = getattr(agent, 'role', 'unknown')
        agent_id = f"crewai-{role}"
        
        super().__init__(
            agent=agent,
            pre_check=pre_check,
            post_check=post_check,
            block_threshold=block_threshold,
            agent_id=agent_id,
            storage_dir=storage_dir
        )
    
    def execute(self, task: Any) -> Any:
        """Execute a task with governance checks.
        
        Args:
            task: The task to execute
            
        Returns:
            Task output (possibly filtered)
        """
        return self.execute_task(task)
    
    def execute_task(self, task: Any) -> Any:
        """Execute a task with governance checks.
        
        Args:
            task: The task to execute
            
        Returns:
            Task output (possibly filtered)
        """
        # Pre-check
        if self.pre_check:
            task_str = str(task)
            result = self._check_governance(task_str, "task_execution")
            risk = self._get_risk_score(result)
            
            if risk > self.block_threshold:
                reason = result.get("reason", "High risk task")
                return f"Task blocked by governance: {reason}"
        
        # Execute task
        if hasattr(self.agent, 'execute_task'):
            output = self.agent.execute_task(task)
        else:
            # Fallback for different agent interfaces
            output = str(task)
        
        # Post-check
        if self.post_check:
            output_str = str(output)
            result = self._check_governance(output_str, "task_output")
            risk = self._get_risk_score(result)
            
            if risk > self.block_threshold:
                reason = result.get("reason", "High risk output")
                return f"Output filtered by governance: {reason}"
        
        return output


class CrewAIFramework(AgentFrameworkBase):
    """CrewAI framework integration with Nethical.
    
    Provides governance tools and utilities for CrewAI agents.
    """
    
    def __init__(self, **kwargs):
        """Initialize the CrewAI framework integration."""
        super().__init__(agent_id="crewai-framework", **kwargs)
    
    def get_tool(self) -> NethicalCrewAITool:
        """Get a CrewAI-compatible governance tool.
        
        Returns:
            NethicalCrewAITool instance
        """
        return NethicalCrewAITool(
            block_threshold=self.block_threshold,
            restrict_threshold=self.restrict_threshold,
            storage_dir=self.storage_dir
        )
    
    def wrap_agent(self, agent: Any, **kwargs) -> NethicalAgentWrapper:
        """Wrap a CrewAI agent with governance.
        
        Args:
            agent: CrewAI agent to wrap
            **kwargs: Additional arguments for NethicalAgentWrapper
            
        Returns:
            Governed agent wrapper
        """
        return NethicalAgentWrapper(
            agent=agent,
            block_threshold=self.block_threshold,
            storage_dir=self.storage_dir,
            **kwargs
        )


def get_nethical_tool() -> Optional[Any]:
    """Get a CrewAI-compatible Nethical tool.
    
    Returns:
        CrewAI Tool object or NethicalCrewAITool if CrewAI not available
    """
    tool = NethicalCrewAITool()
    crewai_tool = tool.as_crewai_tool()
    return crewai_tool if crewai_tool else tool


def handle_nethical_tool(
    tool_input: Dict[str, Any],
    agent_id: str = "crewai-agent"
) -> Dict[str, Any]:
    """Handle a Nethical tool call from CrewAI."""
    from nethical.core import IntegratedGovernance
    
    governance = IntegratedGovernance()
    
    action = tool_input.get("action", "")
    
    result = governance.process_action(
        action=action,
        agent_id=agent_id,
        action_type="agent_action"
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
        "governance_result": result
    }
