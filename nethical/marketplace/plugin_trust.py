"""
Plugin Trust System - Signature Verification and Trust Score Gating

This module implements the trust and security verification system for plugins,
including signature verification enforcement, trust score gating, and
vulnerability scanning per plugin load.

Production Readiness Checklist - Section 8: Plugin Trust
- Signature verification enforced
- Trust score gating (threshold ≥80)
- Vulnerability scan per plugin load
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import hashlib

from nethical.marketplace.plugin_governance import PluginGovernance, SecurityScanResult
from nethical.marketplace.community import CommunityManager
from nethical.marketplace.plugin_registry import PluginRegistry, PluginTrustLevel

logger = logging.getLogger(__name__)


class TrustGatingResult(Enum):
    """Result of trust gating check"""

    PASSED = "passed"
    FAILED_SIGNATURE = "failed_signature"
    FAILED_TRUST_SCORE = "failed_trust_score"
    FAILED_VULNERABILITIES = "failed_vulnerabilities"
    FAILED_MULTIPLE = "failed_multiple"


@dataclass
class PluginTrustCheck:
    """Complete trust check result for a plugin"""

    plugin_id: str
    timestamp: datetime
    signature_valid: bool
    trust_score: float
    trust_score_threshold: float
    vulnerability_count: int
    critical_vulnerability_count: int
    gating_result: TrustGatingResult
    scan_result: Optional[SecurityScanResult] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def passed(self) -> bool:
        """Check if plugin passed all trust checks"""
        return self.gating_result == TrustGatingResult.PASSED


class PluginTrustSystem:
    """
    Enforces plugin trust requirements including signature verification,
    trust score gating, and vulnerability scanning.

    This system ensures that only trusted and secure plugins are loaded,
    meeting the production readiness requirements for plugin trust.

    Example:
        >>> trust_system = PluginTrustSystem(trust_threshold=80)
        >>> check = trust_system.verify_plugin_trust("my-plugin")
        >>> if check.passed():
        ...     print("Plugin is trusted and safe to load")
    """

    def __init__(
        self,
        registry: Optional[PluginRegistry] = None,
        governance: Optional[PluginGovernance] = None,
        community: Optional[CommunityManager] = None,
        trust_threshold: float = 80.0,
        max_vulnerabilities: int = 0,
        max_critical_vulnerabilities: int = 0,
        enforce_signature: bool = True,
        storage_dir: str = "./nethical_trust",
    ):
        """
        Initialize the plugin trust system.

        Args:
            registry: Plugin registry for signature verification
            governance: Plugin governance for security scanning
            community: Community manager for trust scores
            trust_threshold: Minimum trust score required (0-100)
            max_vulnerabilities: Maximum allowed vulnerabilities
            max_critical_vulnerabilities: Maximum allowed critical vulnerabilities
            enforce_signature: Whether to enforce signature verification
            storage_dir: Directory for trust system data
        """
        self.registry = registry
        self.governance = governance or PluginGovernance()
        self.community = community or CommunityManager()
        self.trust_threshold = trust_threshold
        self.max_vulnerabilities = max_vulnerabilities
        self.max_critical_vulnerabilities = max_critical_vulnerabilities
        self.enforce_signature = enforce_signature
        self.storage_dir = storage_dir

        # Cache for trust checks to avoid repeated scans
        self._trust_cache: Dict[str, PluginTrustCheck] = {}

        logger.info(
            f"Plugin trust system initialized with threshold={trust_threshold}, "
            f"enforce_signature={enforce_signature}"
        )

    def verify_plugin_trust(
        self,
        plugin_id: str,
        plugin_code: Optional[str] = None,
        plugin_path: Optional[str] = None,
        force_scan: bool = False,
    ) -> PluginTrustCheck:
        """
        Verify complete trust for a plugin before loading.

        This performs three key checks:
        1. Signature verification (if enforced)
        2. Trust score gating (threshold ≥80)
        3. Vulnerability scanning

        Args:
            plugin_id: Plugin identifier
            plugin_code: Optional plugin source code
            plugin_path: Optional path to plugin files
            force_scan: Force a new security scan even if cached

        Returns:
            PluginTrustCheck with complete verification results
        """
        timestamp = datetime.now(timezone.utc)

        # Check cache unless forced
        if not force_scan and plugin_id in self._trust_cache:
            cached = self._trust_cache[plugin_id]
            # Use cache if less than 1 hour old
            if (timestamp - cached.timestamp).total_seconds() < 3600:
                logger.info(f"Using cached trust check for {plugin_id}")
                return cached

        logger.info(f"Verifying plugin trust for {plugin_id}")

        failures: List[str] = []

        # 1. Signature verification
        signature_valid = self._verify_signature(plugin_id)
        if self.enforce_signature and not signature_valid:
            failures.append("signature")
            logger.warning(f"Signature verification failed for {plugin_id}")

        # 2. Trust score gating
        trust_score = self._calculate_trust_score(plugin_id)
        if trust_score < self.trust_threshold:
            failures.append("trust_score")
            logger.warning(
                f"Trust score {trust_score} below threshold {self.trust_threshold} "
                f"for {plugin_id}"
            )

        # 3. Vulnerability scanning
        scan_result = self.governance.security_scan(
            plugin_id=plugin_id, plugin_code=plugin_code, plugin_path=plugin_path
        )

        vulnerability_count = len(scan_result.vulnerabilities)
        critical_count = sum(
            1 for v in scan_result.vulnerabilities if "CRITICAL" in v or "HIGH" in v
        )

        if vulnerability_count > self.max_vulnerabilities:
            failures.append("vulnerabilities")
            logger.warning(
                f"Plugin {plugin_id} has {vulnerability_count} vulnerabilities "
                f"(max: {self.max_vulnerabilities})"
            )

        if critical_count > self.max_critical_vulnerabilities:
            failures.append("critical_vulnerabilities")
            logger.error(
                f"Plugin {plugin_id} has {critical_count} critical vulnerabilities "
                f"(max: {self.max_critical_vulnerabilities})"
            )

        # Determine gating result
        if len(failures) == 0:
            gating_result = TrustGatingResult.PASSED
        elif len(failures) == 1:
            if "signature" in failures:
                gating_result = TrustGatingResult.FAILED_SIGNATURE
            elif "trust_score" in failures:
                gating_result = TrustGatingResult.FAILED_TRUST_SCORE
            else:
                gating_result = TrustGatingResult.FAILED_VULNERABILITIES
        else:
            gating_result = TrustGatingResult.FAILED_MULTIPLE

        # Create trust check result
        check = PluginTrustCheck(
            plugin_id=plugin_id,
            timestamp=timestamp,
            signature_valid=signature_valid,
            trust_score=trust_score,
            trust_score_threshold=self.trust_threshold,
            vulnerability_count=vulnerability_count,
            critical_vulnerability_count=critical_count,
            gating_result=gating_result,
            scan_result=scan_result,
            details={
                "failures": failures,
                "enforce_signature": self.enforce_signature,
                "scan_passed": scan_result.passed,
                "security_level": scan_result.security_level.value,
            },
        )

        # Cache the result
        self._trust_cache[plugin_id] = check

        if check.passed():
            logger.info(f"Plugin {plugin_id} passed all trust checks")
        else:
            logger.error(
                f"Plugin {plugin_id} failed trust checks: {failures}, "
                f"result={gating_result.value}"
            )

        return check

    def _verify_signature(self, plugin_id: str) -> bool:
        """Verify plugin signature using registry"""
        if not self.registry:
            logger.warning("No registry configured, skipping signature verification")
            return not self.enforce_signature  # Pass if not enforced

        plugin = self.registry.get_plugin(plugin_id)
        if not plugin:
            logger.error(f"Plugin {plugin_id} not found in registry")
            return False

        if not plugin.signature:
            logger.warning(f"No signature available for plugin {plugin_id}")
            return not self.enforce_signature

        # Verify using registry
        try:
            return self.registry.verify_signature(
                plugin_id,
                (
                    plugin.signature.encode()
                    if isinstance(plugin.signature, str)
                    else plugin.signature
                ),
            )
        except Exception as e:
            logger.error(f"Signature verification error for {plugin_id}: {e}")
            return False

    def _calculate_trust_score(self, plugin_id: str) -> float:
        """Calculate trust score (0-100) for a plugin"""
        # Get plugin stats from community
        plugin_stats = self.community.get_plugin_stats(plugin_id)

        # Get contributor stats if available using public API
        submissions = self.community.list_submissions(plugin_id=plugin_id)

        if not submissions:
            logger.warning(f"No submission data for {plugin_id}, using minimal score")
            return 0.0

        submission = submissions[0]  # Use first submission
        contributor_stats = self.community.get_contributor_stats(submission.author)

        # Calculate trust score (0-1 scale)
        trust_score_01 = self.community.compute_trust_score(
            average_rating=plugin_stats.get("average_rating", 0.0),
            review_count=plugin_stats.get("total_reviews", 0),
            approval_ratio=contributor_stats.get("acceptance_rate", 0.0),
            helpful_votes=plugin_stats.get("helpful_votes_total", 0),
        )

        # Convert to 0-100 scale
        return trust_score_01 * 100.0

    def get_trust_metrics(self) -> Dict[str, Any]:
        """Get overall trust system metrics"""
        total_checks = len(self._trust_cache)
        passed_checks = sum(1 for c in self._trust_cache.values() if c.passed())

        failure_breakdown = {
            "signature": 0,
            "trust_score": 0,
            "vulnerabilities": 0,
            "multiple": 0,
        }

        for check in self._trust_cache.values():
            if check.gating_result == TrustGatingResult.FAILED_SIGNATURE:
                failure_breakdown["signature"] += 1
            elif check.gating_result == TrustGatingResult.FAILED_TRUST_SCORE:
                failure_breakdown["trust_score"] += 1
            elif check.gating_result == TrustGatingResult.FAILED_VULNERABILITIES:
                failure_breakdown["vulnerabilities"] += 1
            elif check.gating_result == TrustGatingResult.FAILED_MULTIPLE:
                failure_breakdown["multiple"] += 1

        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "pass_rate": (
                (passed_checks / total_checks * 100) if total_checks > 0 else 0.0
            ),
            "failure_breakdown": failure_breakdown,
            "trust_threshold": self.trust_threshold,
            "enforce_signature": self.enforce_signature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def clear_cache(self, plugin_id: Optional[str] = None):
        """Clear trust check cache"""
        if plugin_id:
            self._trust_cache.pop(plugin_id, None)
            logger.info(f"Cleared trust cache for {plugin_id}")
        else:
            self._trust_cache.clear()
            logger.info("Cleared all trust cache")
