"""
AutoGen integration with Nethical governance.

Provides governed wrappers for AutoGen agents and conversations.
"""

from typing import Any, Callable, Dict, List, Optional

from .base import AgentFrameworkBase, AgentWrapper


# Check for AutoGen availability
try:
    import autogen
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    autogen = None


class NethicalAutoGenTool:
    """AutoGen tool for Nethical governance.
    
    This tool can be registered with AutoGen agents to enable
    governance checks during conversations.
    
    Example:
        from nethical.integrations.agent_frameworks import NethicalAutoGenTool
        from autogen import AssistantAgent
        
        governance_tool = NethicalAutoGenTool()
        
        assistant = AssistantAgent(
            name="assistant",
            llm_config={"config_list": [...]},
        )
        
        # Register the governance function
        assistant.register_function(
            function_map={
                "nethical_check": governance_tool.check
            }
        )
    """
    
    def __init__(
        self,
        block_threshold: float = 0.7,
        restrict_threshold: float = 0.4,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the AutoGen tool.
        
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
    
    def check(self, action: str, action_type: str = "query") -> Dict[str, Any]:
        """Check an action against governance rules.
        
        This method can be registered as a function with AutoGen agents.
        
        Args:
            action: The action to evaluate
            action_type: Type of action
            
        Returns:
            Dict with decision and details
        """
        result = self.governance.process_action(
            action=action,
            agent_id="autogen-agent",
            action_type=action_type
        )
        
        risk = result.get("phase3", {}).get("risk_score", 0.0)
        
        if risk > self.block_threshold:
            decision = "BLOCK"
        elif risk > self.restrict_threshold:
            decision = "RESTRICT"
        else:
            decision = "ALLOW"
        
        return {
            "decision": decision,
            "risk_score": risk,
            "allowed": decision != "BLOCK",
            "reason": self._get_reason(result)
        }
    
    def _get_reason(self, result: Dict[str, Any]) -> str:
        """Extract reason from governance result."""
        if "reason" in result:
            return result["reason"]
        
        phase3 = result.get("phase3", {})
        return f"Risk tier: {phase3.get('risk_tier', 'UNKNOWN')}"
    
    def get_function_config(self) -> Dict[str, Any]:
        """Get function configuration for AutoGen registration.
        
        Returns:
            Dict with function definition for AutoGen
        """
        return {
            "name": "nethical_check",
            "description": (
                "Check if an action is safe and compliant with governance rules. "
                "Use before executing potentially sensitive operations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action or content to evaluate"
                    },
                    "action_type": {
                        "type": "string",
                        "description": "Type of action (query, code, data_access, etc.)"
                    }
                },
                "required": ["action"]
            }
        }


class NethicalConversableAgent:
    """Wrapper for AutoGen ConversableAgent with governance.
    
    Provides message-level governance checks for AutoGen conversations.
    
    Example:
        from autogen import ConversableAgent
        from nethical.integrations.agent_frameworks import NethicalConversableAgent
        
        # Create base agent
        agent = ConversableAgent(
            name="assistant",
            llm_config={"config_list": [...]}
        )
        
        # Wrap with governance
        safe_agent = NethicalConversableAgent(
            agent=agent,
            check_incoming=True,
            check_outgoing=True
        )
    """
    
    def __init__(
        self,
        agent: Any,
        check_incoming: bool = True,
        check_outgoing: bool = True,
        block_threshold: float = 0.7,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the governed agent.
        
        Args:
            agent: AutoGen agent to wrap
            check_incoming: Check incoming messages
            check_outgoing: Check outgoing messages
            block_threshold: Risk threshold for blocking
            storage_dir: Directory for Nethical data storage
        """
        self.agent = agent
        self.check_incoming = check_incoming
        self.check_outgoing = check_outgoing
        self.block_threshold = block_threshold
        self.storage_dir = storage_dir
        
        self._governance = None
        
        # Store original methods for wrapping
        self._original_receive = getattr(agent, 'receive', None)
        self._original_send = getattr(agent, 'send', None)
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(storage_dir=self.storage_dir)
        return self._governance
    
    def _check_message(self, message: Any, direction: str) -> Optional[str]:
        """Check a message for governance compliance.
        
        Args:
            message: The message to check
            direction: "incoming" or "outgoing"
            
        Returns:
            None if allowed, or blocked message string
        """
        # Extract content from message
        if isinstance(message, dict):
            content = message.get("content", str(message))
        else:
            content = str(message)
        
        action_type = f"message_{direction}"
        
        result = self.governance.process_action(
            action=content,
            agent_id=f"autogen-{self.agent.name}",
            action_type=action_type
        )
        
        risk = result.get("phase3", {}).get("risk_score", 0.0)
        
        if risk > self.block_threshold:
            reason = result.get("reason", "High risk content")
            return f"[BLOCKED by governance: {reason}]"
        
        return None
    
    def receive(self, message: Any, sender: Any, **kwargs) -> Any:
        """Receive a message with governance check.
        
        Args:
            message: The incoming message
            sender: The sender agent
            **kwargs: Additional arguments
            
        Returns:
            Response from the wrapped agent
        """
        if self.check_incoming:
            blocked = self._check_message(message, "incoming")
            if blocked:
                # Return blocked response instead of processing
                return blocked
        
        if self._original_receive:
            return self._original_receive(message, sender, **kwargs)
        
        return None
    
    def send(self, message: Any, recipient: Any, **kwargs) -> Any:
        """Send a message with governance check.
        
        Args:
            message: The outgoing message
            recipient: The recipient agent
            **kwargs: Additional arguments
            
        Returns:
            Response from send operation
        """
        if self.check_outgoing:
            blocked = self._check_message(message, "outgoing")
            if blocked:
                # Modify message to indicate blocking
                if isinstance(message, dict):
                    message = {**message, "content": blocked}
                else:
                    message = blocked
        
        if self._original_send:
            return self._original_send(message, recipient, **kwargs)
        
        return None
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped agent."""
        return getattr(self.agent, name)


class GovernedGroupChat:
    """Wrapper for AutoGen GroupChat with governance.
    
    Monitors group chat conversations and applies governance checks
    to all messages.
    """
    
    def __init__(
        self,
        agents: List[Any],
        messages: List[Dict] = None,
        block_threshold: float = 0.7,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize governed group chat.
        
        Args:
            agents: List of AutoGen agents
            messages: Initial messages
            block_threshold: Risk threshold for blocking
            storage_dir: Directory for Nethical data storage
        """
        self.agents = agents
        self.messages = messages or []
        self.block_threshold = block_threshold
        self.storage_dir = storage_dir
        
        self._governance = None
        self._group_chat = None
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(storage_dir=self.storage_dir)
        return self._governance
    
    def check_message(self, message: Dict) -> bool:
        """Check if a message is allowed.
        
        Args:
            message: The message to check
            
        Returns:
            True if allowed, False if blocked
        """
        content = message.get("content", str(message))
        sender = message.get("name", "unknown")
        
        result = self.governance.process_action(
            action=content,
            agent_id=f"autogen-groupchat-{sender}",
            action_type="group_message"
        )
        
        risk = result.get("phase3", {}).get("risk_score", 0.0)
        return risk <= self.block_threshold


class AutoGenFramework(AgentFrameworkBase):
    """AutoGen framework integration with Nethical.
    
    Provides governance tools and utilities for AutoGen agents.
    """
    
    def __init__(self, **kwargs):
        """Initialize the AutoGen framework integration."""
        super().__init__(agent_id="autogen-framework", **kwargs)
    
    def get_tool(self) -> NethicalAutoGenTool:
        """Get an AutoGen-compatible governance tool.
        
        Returns:
            NethicalAutoGenTool instance
        """
        return NethicalAutoGenTool(
            block_threshold=self.block_threshold,
            restrict_threshold=self.restrict_threshold,
            storage_dir=self.storage_dir
        )
    
    def wrap_agent(self, agent: Any, **kwargs) -> NethicalConversableAgent:
        """Wrap an AutoGen agent with governance.
        
        Args:
            agent: AutoGen agent to wrap
            **kwargs: Additional arguments
            
        Returns:
            Governed agent wrapper
        """
        return NethicalConversableAgent(
            agent=agent,
            block_threshold=self.block_threshold,
            storage_dir=self.storage_dir,
            **kwargs
        )


def get_nethical_function() -> Dict[str, Any]:
    """Get function definition for AutoGen agent registration.
    
    Returns:
        Function definition dict
    """
    return {
        "name": "nethical_check",
        "description": (
            "Check if an action is safe and compliant with governance rules. "
            "Use before executing potentially sensitive operations."
        ),
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


def handle_nethical_function(
    action: str,
    action_type: str = "query"
) -> Dict[str, Any]:
    """Handle a Nethical function call from AutoGen.
    
    Args:
        action: The action to evaluate
        action_type: Type of action
        
    Returns:
        Result dictionary
    """
    tool = NethicalAutoGenTool()
    return tool.check(action, action_type)
