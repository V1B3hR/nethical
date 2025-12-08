"""
Policy Lineage Tracker

Tracks policy version history, hash chain integrity, and multi-signature
compliance for governance dashboard visualization.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib


@dataclass
class PolicyVersion:
    """A version in policy lineage"""

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

    Monitors policy version history and validates hash chain integrity
    for the governance dashboard.
    """

    def __init__(self):
        """Initialize policy lineage tracker"""
        self._policies: Dict[str, List[PolicyVersion]] = {}
        self._active_policies: set = set()

    def record_policy_version(
        self,
        policy_id: str,
        version: int,
        content: str,
        parent_hash: Optional[str],
        signatures: List[Dict[str, str]],
        author: str,
    ):
        """
        Record a new policy version.

        Args:
            policy_id: Policy identifier
            version: Version number
            content: Policy content
            parent_hash: Hash of parent version
            signatures: List of signatures
            author: Policy author
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        policy_version = PolicyVersion(
            policy_id=policy_id,
            version=version,
            content_hash=content_hash,
            parent_hash=parent_hash,
            signatures=signatures,
            timestamp=datetime.utcnow(),
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
            Chain integrity statistics
        """
        total_policies = len(self._policies)
        verified_chains = 0
        broken_chains = 0

        for policy_id, versions in self._policies.items():
            if self._verify_chain(versions):
                verified_chains += 1
            else:
                broken_chains += 1

        integrity_rate = verified_chains / total_policies if total_policies > 0 else 1.0

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
            Version statistics
        """
        total_versions = sum(len(versions) for versions in self._policies.values())
        active_policies = len(self._active_policies)

        # Count recent changes
        cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0)
        recent_changes = 0
        for versions in self._policies.values():
            recent_changes += sum(1 for v in versions if v.timestamp >= cutoff)

        avg_versions = total_versions / active_policies if active_policies > 0 else 0

        return {
            "total_versions": total_versions,
            "active_policies": active_policies,
            "recent_changes_24h": recent_changes,
            "average_versions_per_policy": avg_versions,
        }

    def get_multi_sig_metrics(self) -> Dict[str, Any]:
        """
        Get multi-signature compliance metrics.

        Returns:
            Multi-sig compliance statistics
        """
        total_changes = sum(len(versions) for versions in self._policies.values())
        properly_signed = 0
        min_required = 2  # From governance config

        for versions in self._policies.values():
            for version in versions:
                if len(version.signatures) >= min_required:
                    properly_signed += 1

        compliance_rate = properly_signed / total_changes if total_changes > 0 else 1.0

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
            True if chain is valid
        """
        if not versions or len(versions) < 2:
            return True

        # Sort by version
        sorted_versions = sorted(versions, key=lambda v: v.version)

        for i in range(1, len(sorted_versions)):
            prev = sorted_versions[i - 1]
            curr = sorted_versions[i]

            # Verify current links to previous
            if curr.parent_hash != prev.content_hash:
                return False

        return True
