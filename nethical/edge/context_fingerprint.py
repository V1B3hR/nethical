"""
Context Fingerprint - Context Hashing for Cache Keys

Provides consistent context fingerprinting for cache key generation.
Target: <0.1ms fingerprint computation
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextFingerprint:
    """
    Context fingerprint for cache key generation.

    Attributes:
        hash: The fingerprint hash
        components: Components used in hash
        version: Fingerprint algorithm version
    """

    hash: str
    components: Dict[str, str]
    version: str = "1.0"


def compute_fingerprint(
    action: str,
    action_type: str,
    context: Optional[Dict[str, Any]] = None,
    include_timestamp: bool = False,
) -> str:
    """
    Compute context fingerprint for cache key.

    Target: <0.1ms

    Args:
        action: The action content
        action_type: Type of action
        context: Additional context
        include_timestamp: Whether to include timestamp (makes unique)

    Returns:
        Fingerprint hash string
    """
    context = context or {}

    # Build components for hashing
    components = {
        "action": action,
        "action_type": action_type,
    }

    # Add relevant context fields (excluding volatile data)
    stable_context_keys = [
        "agent_id",
        "agent_type",
        "domain",
        "environment",
        "user_role",
    ]

    for key in stable_context_keys:
        if key in context:
            components[key] = str(context[key])

    # Optional timestamp for unique fingerprints
    if include_timestamp:
        import time

        components["timestamp"] = str(int(time.time()))

    # Create deterministic JSON
    json_str = json.dumps(components, sort_keys=True, separators=(",", ":"))

    # Use xxhash if available for speed, fallback to md5
    try:
        import xxhash

        return xxhash.xxh64(json_str.encode()).hexdigest()
    except ImportError:
        return hashlib.md5(json_str.encode()).hexdigest()


def compute_detailed_fingerprint(
    action: str,
    action_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> ContextFingerprint:
    """
    Compute detailed context fingerprint with metadata.

    Args:
        action: The action content
        action_type: Type of action
        context: Additional context

    Returns:
        ContextFingerprint with hash and components
    """
    context = context or {}

    components = {
        "action_hash": hashlib.md5(action.encode()).hexdigest()[:16],
        "action_type": action_type,
    }

    stable_context_keys = [
        "agent_id",
        "agent_type",
        "domain",
        "environment",
    ]

    for key in stable_context_keys:
        if key in context:
            components[key] = str(context[key])

    hash_value = compute_fingerprint(action, action_type, context)

    return ContextFingerprint(
        hash=hash_value,
        components=components,
        version="1.0",
    )


def normalize_action(action: str) -> str:
    """
    Normalize action for consistent fingerprinting.

    Args:
        action: The action content

    Returns:
        Normalized action string
    """
    # Lowercase
    normalized = action.lower()

    # Remove extra whitespace
    normalized = " ".join(normalized.split())

    # Remove common prefixes
    prefixes_to_remove = [
        "please ",
        "could you ",
        "can you ",
        "i want to ",
        "i need to ",
    ]

    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    return normalized.strip()


def action_similarity_hash(action: str, granularity: str = "medium") -> str:
    """
    Compute similarity hash for action clustering.

    Actions with similar semantic meaning should produce similar hashes.

    Args:
        action: The action content
        granularity: Hash granularity (coarse, medium, fine)

    Returns:
        Similarity hash
    """
    normalized = normalize_action(action)

    if granularity == "coarse":
        # Use first few words only
        words = normalized.split()[:3]
        key_text = " ".join(words)
    elif granularity == "fine":
        # Use full normalized text
        key_text = normalized
    else:  # medium
        # Use first words + length bucket
        words = normalized.split()[:5]
        length_bucket = len(normalized) // 50
        key_text = f"{' '.join(words)}:{length_bucket}"

    return hashlib.md5(key_text.encode()).hexdigest()[:12]
