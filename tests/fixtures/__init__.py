"""
Test Fixtures Module

This module provides shared test data factories and fixtures for Nethical tests.
"""

from .agent_actions import (
    create_agent_action,
    create_safe_action,
    create_risky_action,
    SAMPLE_ACTIONS,
)

from .violations import (
    create_violation,
    COMMON_VIOLATIONS,
)

from .payloads import (
    INJECTION_PAYLOADS,
    XSS_PAYLOADS,
    JAILBREAK_PAYLOADS,
)

__all__ = [
    "create_agent_action",
    "create_safe_action",
    "create_risky_action",
    "SAMPLE_ACTIONS",
    "create_violation",
    "COMMON_VIOLATIONS",
    "INJECTION_PAYLOADS",
    "XSS_PAYLOADS",
    "JAILBREAK_PAYLOADS",
]
