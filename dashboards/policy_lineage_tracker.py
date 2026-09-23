"""
Policy Lineage Tracker

Tracks policy version history, cryptographic hash chain integrity, and multi-signature
compliance for sovereign governance dashboard monitoring and audit verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Set


def _ensure_utc(dt: Optional[datetime]) -> datetime:
    """Ensure datetime has UTC timezone."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class PolicyVersion:
    """A version in policy lineage."""
    policy_id: str
    version: int
    content_hash: str
    parent_hash: Optional[str]
    signatures: List[Dict[str, str]]
    timestamp: datetime
    author: str


class PolicyLineageTracker:
    """
    Policy Lineage Tracker

    Monitors policy version history and validates cryptographic hash chain integrity
    for sovereign governance and tamper detection.
    """

    def __init__(self) -> None:
        """Initialise policy lineage tracker."""
        self._policies: Dict[str, List[PolicyVersion]] = {}
        self._active_policies: Set[str] = set()

    def record_policy_version(
        self,
        policy_id: str,
        version: int,
        content: str,
        parent_hash: Optional[str],
        signatures: List[Dict[str, str]],
        author: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a new policy version.

        Args:
            policy_id: Policy identifier
            version: Version number
            content: Policy content
            parent_hash: Hash of parent version
            signatures: List of signatures
            author: Policy author
            timestamp: Optional creation timestamp
        """
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        policy_version = PolicyVersion(
            policy_id=policy_id,
            version=version,
            content_hash=content_hash,
            parent_hash=parent_hash,
            signatures=signatures,
            timestamp=_ensure_utc(timestamp),
            author=author,
        )

        if policy_id not in self._policies:
            self._policies[policy_id] = []

        self._policies[policy_id].append(policy_version)
        self._active_policies.add(policy_id)

    def get_chain_integrity(self) -> Dict[str, Any]:
        """
        Get policy chain integrity metrics.

        Returns:
            Chain integrity statistics dictionary
        """
        total_policies = len(self._policies)
        verified_chains = 0
        broken_chains = 0

        for policy_id, versions in self._policies.items():
            if self._verify_chain(versions):
                verified_chains += 1
            else:
                broken_chains += 1

        integrity_rate = verified_chains / float(total_policies) if total_policies > 0 else 1.0

        return {
            "total_policies": total_policies,
            "verified_chains": verified_chains,
            "broken_chains": broken_chains,
            "integrity_rate": integrity_rate,
            "status": "healthy" if broken_chains == 0 else "critical",
        }

    def get_version_metrics(self) -> Dict[str, Any]:
        """
        Get policy version tracking metrics.

        Returns:
            Version statistics dictionary
        """
        total_versions = sum(len(versions) for versions in self._policies.values())
        active_policies = len(self._active_policies)

        # Count recent changes in last 24h
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        recent_changes = 0
        for versions in self._policies.values():
            recent_changes += sum(1 for v in versions if _ensure_utc(v.timestamp) >= cutoff)

        avg_versions = total_versions / float(active_policies) if active_policies > 0 else 0.0

        return {
            "total_versions": total_versions,
            "active_policies": active_policies,
            "recent_changes_24h": recent_changes,
            "average_versions_per_policy": avg_versions,
        }

    def get_multi_sig_metrics(self, min_required: int = 2) -> Dict[str, Any]:
        """
        Get multi-signature compliance metrics.

        Args:
            min_required: Minimum signatures required per policy change

        Returns:
            Multi-sig compliance statistics dictionary
        """
        total_changes = sum(len(versions) for versions in self._policies.values())
        properly_signed = 0

        for versions in self._policies.values():
            for version in versions:
                if len(version.signatures) >= min_required:
                    properly_signed += 1

        compliance_rate = properly_signed / float(total_changes) if total_changes > 0 else 1.0

        return {
            "total_changes": total_changes,
            "properly_signed": properly_signed,
            "compliance_rate": compliance_rate,
            "min_signatures_required": min_required,
            "status": "healthy" if compliance_rate >= 1.0 else "warning",
        }

    def _verify_chain(self, versions: List[PolicyVersion]) -> bool:
        """
        Verify hash chain integrity for policy versions.

        Args:
            versions: List of policy versions

        Returns:
            True if chain is cryptographically valid
        """
        if not versions or len(versions) < 2:
            return True

        # Sort by version ascending
        sorted_versions = sorted(versions, key=lambda v: v.version)

        for i in range(1, len(sorted_versions)):
            prev = sorted_versions[i - 1]
            curr = sorted_versions[i]

            # Verify current parent_hash links to previous content_hash
            if curr.parent_hash != prev.content_hash:
                return False

        return True
