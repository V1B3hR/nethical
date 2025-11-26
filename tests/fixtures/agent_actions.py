"""
Agent Action Test Fixtures

Factory functions and sample data for creating AgentAction objects in tests.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import uuid


def create_agent_action(
    agent_id: Optional[str] = None,
    stated_intent: Optional[str] = None,
    actual_action: str = "Test action",
    context: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an AgentAction-like dictionary for testing.

    Args:
        agent_id: Agent identifier (auto-generated if not provided)
        stated_intent: Optional stated goal
        actual_action: The action to evaluate
        context: Optional context dictionary
        parameters: Optional parameters dictionary

    Returns:
        Dictionary representing an agent action
    """
    return {
        "id": str(uuid.uuid4()),
        "agent_id": agent_id or f"test-agent-{uuid.uuid4().hex[:8]}",
        "stated_intent": stated_intent,
        "actual_action": actual_action,
        "context": context or {},
        "parameters": parameters or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_safe_action(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a safe/allowed action for testing.

    Args:
        agent_id: Optional agent identifier

    Returns:
        Dictionary representing a safe agent action
    """
    return create_agent_action(
        agent_id=agent_id,
        stated_intent="Help user with their question",
        actual_action="Provide factual information about the weather",
        context={"user_query": "What's the weather like today?"},
    )


def create_risky_action(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a risky/potentially blocked action for testing.

    Args:
        agent_id: Optional agent identifier

    Returns:
        Dictionary representing a risky agent action
    """
    return create_agent_action(
        agent_id=agent_id,
        stated_intent="Execute user request",
        actual_action="DELETE FROM users WHERE 1=1; DROP TABLE users;",
        context={"user_id": "attacker123"},
    )


# Sample actions for various test scenarios
SAMPLE_ACTIONS = {
    "sql_query": create_agent_action(
        agent_id="db-agent",
        stated_intent="Retrieve user data",
        actual_action="SELECT id, name, email FROM users WHERE id = ?",
        context={"user_id": "123"},
    ),
    "file_operation": create_agent_action(
        agent_id="file-agent",
        stated_intent="Read configuration file",
        actual_action="Read file: /etc/app/config.yaml",
        context={"purpose": "configuration"},
    ),
    "api_call": create_agent_action(
        agent_id="api-agent",
        stated_intent="Fetch external data",
        actual_action="HTTP GET https://api.example.com/data",
        context={"headers": {"Authorization": "Bearer token"}},
    ),
    "system_command": create_agent_action(
        agent_id="sys-agent",
        stated_intent="Check system status",
        actual_action="Execute: systemctl status nginx",
        context={"sudo": False},
    ),
    "pii_access": create_agent_action(
        agent_id="data-agent",
        stated_intent="Process user profile",
        actual_action="Access SSN, DOB, and address for user verification",
        context={"purpose": "identity_verification"},
    ),
    "financial_transaction": create_agent_action(
        agent_id="finance-agent",
        stated_intent="Process payment",
        actual_action="Transfer $1000 from account A to account B",
        context={"currency": "USD", "verified": True},
    ),
    "healthcare_data": create_agent_action(
        agent_id="health-agent",
        stated_intent="Review medical records",
        actual_action="Access patient diagnosis history",
        context={"hipaa_compliant": True},
    ),
    "admin_operation": create_agent_action(
        agent_id="admin-agent",
        stated_intent="System maintenance",
        actual_action="Restart database service",
        context={"maintenance_window": True},
    ),
}
