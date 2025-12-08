"""
Utility functions for validation tests.

Common helpers used across multiple validation test suites.
"""

from typing import Any


def extract_action_content(action: Any) -> str:
    """
    Extract action content from various action representations.

    Handles AgentAction objects with 'content' field (new),
    objects with 'action' field (legacy), or string representations.

    Args:
        action: Action object or string

    Returns:
        String representation of the action content
    """
    if hasattr(action, "content"):
        return action.content
    elif hasattr(action, "action"):
        return action.action
    else:
        return str(action)
