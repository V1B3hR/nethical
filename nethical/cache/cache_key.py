"""
Cache Key Generation - Consistent Key Generation

Provides consistent cache key generation across all cache levels.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheKey:
    """
    Cache key representation.

    Attributes:
        key: The cache key string
        namespace: Key namespace (e.g., 'policy', 'decision')
        version: Key version for cache invalidation
        components: Components used to generate key
    """

    key: str
    namespace: str
    version: str = "v1"
    components: Dict[str, Any] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}

    def full_key(self) -> str:
        """Get full key with namespace and version."""
        return f"{self.namespace}:{self.version}:{self.key}"


def generate_cache_key(
    namespace: str,
    *args,
    version: str = "v1",
    **kwargs,
) -> CacheKey:
    """
    Generate a cache key.

    Args:
        namespace: Key namespace
        *args: Positional components
        version: Key version
        **kwargs: Named components

    Returns:
        CacheKey instance
    """
    # Build components dict
    components = {}

    for i, arg in enumerate(args):
        components[f"arg{i}"] = str(arg)

    for key, value in sorted(kwargs.items()):
        components[key] = str(value)

    # Create deterministic JSON
    json_str = json.dumps(components, sort_keys=True, separators=(",", ":"))

    # Hash for consistent key
    key_hash = hashlib.md5(json_str.encode()).hexdigest()

    return CacheKey(
        key=key_hash,
        namespace=namespace,
        version=version,
        components=components,
    )


def generate_decision_key(
    agent_id: str,
    action: str,
    action_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a decision cache key.

    Args:
        agent_id: Agent identifier
        action: Action content
        action_type: Type of action
        context: Action context

    Returns:
        Cache key string
    """
    context = context or {}

    # Include stable context fields only
    stable_context = {}
    for key in ["agent_type", "domain", "environment"]:
        if key in context:
            stable_context[key] = context[key]

    cache_key = generate_cache_key(
        "decision",
        agent_id=agent_id,
        action_hash=hashlib.md5(action.encode()).hexdigest()[:16],
        action_type=action_type,
        **stable_context,
    )

    return cache_key.full_key()


def generate_policy_key(policy_id: str, version: Optional[str] = None) -> str:
    """
    Generate a policy cache key.

    Args:
        policy_id: Policy identifier
        version: Policy version

    Returns:
        Cache key string
    """
    cache_key = generate_cache_key(
        "policy",
        policy_id=policy_id,
        version=version or "latest",
    )

    return cache_key.full_key()


def generate_agent_key(agent_id: str, data_type: str) -> str:
    """
    Generate an agent data cache key.

    Args:
        agent_id: Agent identifier
        data_type: Type of data (e.g., 'quota', 'state')

    Returns:
        Cache key string
    """
    cache_key = generate_cache_key(
        "agent",
        agent_id=agent_id,
        data_type=data_type,
    )

    return cache_key.full_key()
