"""Nethical Python SDK.

Official Python client for the Nethical Governance API.

Features:
- Sync and async evaluation
- Real-time streaming via WebSocket
- Built-in retry and error handling
- Full compliance with 25 Fundamental Laws

Installation:
    pip install nethical-sdk

Usage:
    from nethical_sdk import NethicalClient
    
    client = NethicalClient(
        api_url="https://api.nethical.example.com",
        api_key="your-key"
    )
    
    # Evaluate an action
    result = client.evaluate(
        agent_id="my-agent",
        action="Generate code to access database",
        action_type="code_generation"
    )
    
    if result.decision != "ALLOW":
        print(f"Action blocked: {result.reason}")

Async usage:
    async with client.async_session() as session:
        result = await session.evaluate(...)
"""

from .client import NethicalClient
from .async_client import AsyncNethicalClient
from .models import (
    EvaluateRequest,
    EvaluateResponse,
    Decision,
    Violation,
    Policy,
    FairnessReport,
    Appeal,
    AuditRecord,
)
from .exceptions import (
    NethicalError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServerError,
)

__version__ = "1.0.0"
__all__ = [
    "NethicalClient",
    "AsyncNethicalClient",
    "EvaluateRequest",
    "EvaluateResponse",
    "Decision",
    "Violation",
    "Policy",
    "FairnessReport",
    "Appeal",
    "AuditRecord",
    "NethicalError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ServerError",
    "__version__",
]
