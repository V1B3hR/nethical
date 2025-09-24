"""
Nethical: Safety governance system for AI agents.

This package provides comprehensive monitoring and safety governance
for AI agents, including:
- Intent vs action deviation monitoring
- Ethical and safety constraint violation detection
- Manipulation technique recognition
- Judge system for action evaluation and feedback
"""

from .core.governance import SafetyGovernance, AgentAction as _AgentAction, SafetyViolation, JudgmentResult, MonitoringConfig, ActionType

def AgentAction(id=None, action_id=None, agent_id=None, stated_intent=None, actual_action=None, content=None, action_type=None, context=None, **kwargs):
    """Compatibility wrapper for AgentAction that accepts both old and new APIs."""
    # Handle backward compatibility
    if id is not None and action_id is None:
        action_id = id
    if stated_intent is not None and content is None:
        content = stated_intent  # Use stated_intent as content if no content provided
    if actual_action is not None and content is None:
        content = actual_action  # Use actual_action as content if no content provided
    if actual_action is not None and content == stated_intent:
        content = actual_action  # Prefer actual_action over stated_intent for content
    
    # Set defaults for required fields
    if action_type is None:
        action_type = ActionType.RESPONSE
    if content is None:
        content = stated_intent or actual_action or ""
    if context is None:
        context = {}
    
    # Create agent action with new API
    agent_action = _AgentAction(
        action_id=action_id,
        agent_id=agent_id,
        action_type=action_type,
        content=content,
        context=context,
        **kwargs
    )
    
    # Add compatibility attributes
    agent_action.id = action_id
    agent_action.stated_intent = stated_intent or content
    agent_action.actual_action = actual_action or content
    
    return agent_action

__version__ = "0.1.0"
__all__ = ["SafetyGovernance", "AgentAction", "SafetyViolation", "JudgmentResult", "MonitoringConfig"]