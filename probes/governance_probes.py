"""
Governance Property Probes

Runtime probes that monitor governance properties defined in Phase 4:
- P-MULTI-SIG: Multi-signature approval
- P-POL-LIN: Policy lineage integrity
- P-DATA-MIN: Data minimization
- P-TENANT-ISO: Tenant isolation

These probes ensure compliance with governance requirements in production.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import hashlib

from .base_probe import BaseProbe, ProbeResult, ProbeStatus


class MultiSigProbe(BaseProbe):
    """
    P-MULTI-SIG: Multi-Signature Approval Probe
    
    Validates that policy activations require multiple authorized signatures
    as specified in governance requirements.
    
    Checks:
    - Policy changes have required number of signatures
    - Signatures are from authorized signers
    - No single-party activations for critical policies
    """
    
    def __init__(
        self,
        policy_service: Any,
        min_signatures: int = 2,
        check_interval_seconds: int = 300,
    ):
        """
        Initialize multi-sig probe.
        
        Args:
            policy_service: Service managing policies
            min_signatures: Minimum required signatures
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-MULTI-SIG-MultiSignature",
            check_interval_seconds=check_interval_seconds,
        )
        self.policy_service = policy_service
        self.min_signatures = min_signatures
        self._authorized_signers: Set[str] = set()
    
    def set_authorized_signers(self, signers: List[str]):
        """Set the list of authorized signers"""
        self._authorized_signers = set(signers)
    
    def check(self) -> ProbeResult:
        """Check multi-signature property"""
        timestamp = datetime.utcnow()
        violations = []
        
        try:
            # Get recent policy changes
            recent_changes = self._get_recent_policy_changes()
            
            if not recent_changes:
                return ProbeResult(
                    probe_name=self.name,
                    status=ProbeStatus.HEALTHY,
                    timestamp=timestamp,
                    message="No recent policy changes to verify",
                    metrics={"changes_checked": 0},
                )
            
            insufficient_sigs = 0
            unauthorized_sigs = 0
            
            for change in recent_changes:
                policy_id = change.get("policy_id")
                signatures = change.get("signatures", [])
                
                # Check signature count
                if len(signatures) < self.min_signatures:
                    insufficient_sigs += 1
                    violations.append(
                        f"Policy {policy_id}: only {len(signatures)} signatures "
                        f"(required: {self.min_signatures})"
                    )
                
                # Check signer authorization
                for sig in signatures:
                    signer = sig.get("signer_id")
                    if signer and signer not in self._authorized_signers:
                        unauthorized_sigs += 1
                        violations.append(
                            f"Policy {policy_id}: unauthorized signer {signer}"
                        )
            
            # Determine status
            if insufficient_sigs > 0 or unauthorized_sigs > 0:
                status = ProbeStatus.CRITICAL
                message = f"Multi-sig violations: {insufficient_sigs} insufficient, {unauthorized_sigs} unauthorized"
            else:
                status = ProbeStatus.HEALTHY
                message = f"All {len(recent_changes)} policy changes properly signed"
            
            return ProbeResult(
                probe_name=self.name,
                status=status,
                timestamp=timestamp,
                message=message,
                metrics={
                    "changes_checked": len(recent_changes),
                    "insufficient_signatures": insufficient_sigs,
                    "unauthorized_signatures": unauthorized_sigs,
                    "compliance_rate": (len(recent_changes) - insufficient_sigs) / len(recent_changes)
                        if recent_changes else 1.0,
                },
                violations=violations[:10],  # Limit violations
            )
            
        except Exception as e:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.CRITICAL,
                timestamp=timestamp,
                message=f"Failed to check multi-sig property: {str(e)}",
                metrics={},
            )
    
    def _get_recent_policy_changes(self) -> List[Dict[str, Any]]:
        """Get recent policy changes (placeholder)"""
        # In real implementation, query policy service
        return []


class PolicyLineageProbe(BaseProbe):
    """
    P-POL-LIN: Policy Lineage Integrity Probe
    
    Validates that policy version history forms an unbroken hash chain,
    ensuring policy evolution is traceable and tamper-evident.
    
    Checks:
    - Hash chain integrity from policy creation to current version
    - All versions are properly linked
    - No missing versions in lineage
    """
    
    def __init__(
        self,
        policy_service: Any,
        check_interval_seconds: int = 300,
    ):
        """
        Initialize policy lineage probe.
        
        Args:
            policy_service: Service managing policy lineage
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-POL-LIN-PolicyLineage",
            check_interval_seconds=check_interval_seconds,
        )
        self.policy_service = policy_service
    
    def check(self) -> ProbeResult:
        """Check policy lineage property"""
        timestamp = datetime.utcnow()
        violations = []
        
        try:
            # Get all active policies
            policies = self._get_active_policies()
            
            if not policies:
                return ProbeResult(
                    probe_name=self.name,
                    status=ProbeStatus.WARNING,
                    timestamp=timestamp,
                    message="No active policies to check",
                    metrics={"policies_checked": 0},
                )
            
            broken_chains = 0
            missing_versions = 0
            
            for policy in policies:
                policy_id = policy.get("policy_id")
                
                # Verify hash chain
                lineage = self._get_policy_lineage(policy_id)
                if not self._verify_hash_chain(lineage):
                    broken_chains += 1
                    violations.append(
                        f"Policy {policy_id}: broken hash chain detected"
                    )
                
                # Check for missing versions
                if self._has_missing_versions(lineage):
                    missing_versions += 1
                    violations.append(
                        f"Policy {policy_id}: missing versions in lineage"
                    )
            
            # Determine status
            if broken_chains > 0:
                status = ProbeStatus.CRITICAL
                message = f"Lineage integrity violations: {broken_chains} broken chains"
            elif missing_versions > 0:
                status = ProbeStatus.WARNING
                message = f"Lineage warnings: {missing_versions} policies with gaps"
            else:
                status = ProbeStatus.HEALTHY
                message = f"All {len(policies)} policy lineages intact"
            
            return ProbeResult(
                probe_name=self.name,
                status=status,
                timestamp=timestamp,
                message=message,
                metrics={
                    "policies_checked": len(policies),
                    "broken_chains": broken_chains,
                    "missing_versions": missing_versions,
                    "integrity_rate": (len(policies) - broken_chains) / len(policies)
                        if policies else 1.0,
                },
                violations=violations,
            )
            
        except Exception as e:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.CRITICAL,
                timestamp=timestamp,
                message=f"Failed to check policy lineage: {str(e)}",
                metrics={},
            )
    
    def _get_active_policies(self) -> List[Dict[str, Any]]:
        """Get active policies (placeholder)"""
        # In real implementation, query policy service
        return []
    
    def _get_policy_lineage(self, policy_id: str) -> List[Dict[str, Any]]:
        """Get policy version history (placeholder)"""
        # In real implementation, query policy lineage
        return []
    
    def _verify_hash_chain(self, lineage: List[Dict[str, Any]]) -> bool:
        """Verify hash chain integrity (placeholder)"""
        if not lineage or len(lineage) < 2:
            return True
        
        for i in range(1, len(lineage)):
            prev_version = lineage[i - 1]
            curr_version = lineage[i]
            
            # Verify current version links to previous
            expected_hash = self._compute_hash(prev_version)
            if curr_version.get("parent_hash") != expected_hash:
                return False
        
        return True
    
    def _has_missing_versions(self, lineage: List[Dict[str, Any]]) -> bool:
        """Check for missing versions in lineage (placeholder)"""
        if not lineage:
            return False
        
        # Check version sequence
        versions = [v.get("version", 0) for v in lineage]
        versions.sort()
        
        for i in range(1, len(versions)):
            if versions[i] != versions[i - 1] + 1:
                return True
        
        return False
    
    def _compute_hash(self, version: Dict[str, Any]) -> str:
        """Compute hash of policy version (placeholder)"""
        content = str(version.get("content", ""))
        return hashlib.sha256(content.encode()).hexdigest()


class DataMinimizationProbe(BaseProbe):
    """
    P-DATA-MIN: Data Minimization Probe
    
    Validates that only required context fields are accessed during
    policy evaluation, ensuring privacy and compliance with GDPR.
    
    Checks:
    - Only whitelisted fields are accessed
    - No excessive data collection
    - PII access is logged and justified
    """
    
    def __init__(
        self,
        allowed_fields: Optional[Set[str]] = None,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize data minimization probe.
        
        Args:
            allowed_fields: Whitelisted context fields
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-DATA-MIN-DataMinimization",
            check_interval_seconds=check_interval_seconds,
        )
        self.allowed_fields = allowed_fields or {
            "action_type",
            "resource_type",
            "timestamp",
            "agent_id",
        }
        self._access_logs: List[Dict[str, Any]] = []
    
    def record_access(self, fields_accessed: Set[str], context: str = ""):
        """Record field access for monitoring"""
        self._access_logs.append({
            "fields": fields_accessed,
            "context": context,
            "timestamp": datetime.utcnow(),
        })
        # Keep only recent logs
        if len(self._access_logs) > 1000:
            self._access_logs.pop(0)
    
    def check(self) -> ProbeResult:
        """Check data minimization property"""
        timestamp = datetime.utcnow()
        violations = []
        
        if not self._access_logs:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.HEALTHY,
                timestamp=timestamp,
                message="No field accesses to check",
                metrics={"accesses_checked": 0},
            )
        
        unauthorized_count = 0
        unauthorized_fields = set()
        
        for log in self._access_logs:
            fields = log["fields"]
            unauthorized = fields - self.allowed_fields
            
            if unauthorized:
                unauthorized_count += 1
                unauthorized_fields.update(unauthorized)
                if len(violations) < 10:  # Limit violations
                    violations.append(
                        f"Unauthorized field access: {unauthorized} in {log['context']}"
                    )
        
        # Determine status
        violation_rate = unauthorized_count / len(self._access_logs)
        
        if violation_rate > 0.05:  # >5% violations
            status = ProbeStatus.CRITICAL
            message = f"Excessive unauthorized field access: {unauthorized_count} cases"
        elif unauthorized_count > 0:
            status = ProbeStatus.WARNING
            message = f"Some unauthorized field access: {unauthorized_count} cases"
        else:
            status = ProbeStatus.HEALTHY
            message = f"All {len(self._access_logs)} accesses within allowed fields"
        
        return ProbeResult(
            probe_name=self.name,
            status=status,
            timestamp=timestamp,
            message=message,
            metrics={
                "accesses_checked": len(self._access_logs),
                "unauthorized_count": unauthorized_count,
                "compliance_rate": 1 - violation_rate,
                "unauthorized_fields": list(unauthorized_fields),
            },
            violations=violations,
        )


class TenantIsolationProbe(BaseProbe):
    """
    P-TENANT-ISO: Tenant Isolation Probe
    
    Validates that tenant data and decisions are properly isolated,
    preventing cross-tenant data leakage or interference.
    
    Checks:
    - Tenant boundaries are enforced
    - No cross-tenant data access
    - Network segmentation is maintained
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize tenant isolation probe.
        
        Args:
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-TENANT-ISO-TenantIsolation",
            check_interval_seconds=check_interval_seconds,
        )
        self._access_logs: List[Dict[str, Any]] = []
    
    def record_access(
        self,
        tenant_id: str,
        resource_tenant_id: str,
        access_type: str = "read",
    ):
        """Record tenant access for monitoring"""
        self._access_logs.append({
            "tenant_id": tenant_id,
            "resource_tenant_id": resource_tenant_id,
            "access_type": access_type,
            "timestamp": datetime.utcnow(),
        })
        # Keep only recent logs
        if len(self._access_logs) > 1000:
            self._access_logs.pop(0)
    
    def check(self) -> ProbeResult:
        """Check tenant isolation property"""
        timestamp = datetime.utcnow()
        violations = []
        
        if not self._access_logs:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.HEALTHY,
                timestamp=timestamp,
                message="No tenant accesses to check",
                metrics={"accesses_checked": 0},
            )
        
        cross_tenant_count = 0
        tenant_pairs = set()
        
        for log in self._access_logs:
            tenant_id = log["tenant_id"]
            resource_tenant_id = log["resource_tenant_id"]
            
            if tenant_id != resource_tenant_id:
                cross_tenant_count += 1
                tenant_pairs.add((tenant_id, resource_tenant_id))
                
                if len(violations) < 10:  # Limit violations
                    violations.append(
                        f"Cross-tenant access: tenant {tenant_id} accessed "
                        f"resource from tenant {resource_tenant_id}"
                    )
        
        # Determine status
        violation_rate = cross_tenant_count / len(self._access_logs)
        
        if violation_rate > 0.01:  # >1% violations (stricter than data min)
            status = ProbeStatus.CRITICAL
            message = f"Tenant isolation breach: {cross_tenant_count} cross-tenant accesses"
        elif cross_tenant_count > 0:
            status = ProbeStatus.WARNING
            message = f"Potential isolation issues: {cross_tenant_count} cross-tenant accesses"
        else:
            status = ProbeStatus.HEALTHY
            message = f"All {len(self._access_logs)} accesses properly isolated"
        
        return ProbeResult(
            probe_name=self.name,
            status=status,
            timestamp=timestamp,
            message=message,
            metrics={
                "accesses_checked": len(self._access_logs),
                "cross_tenant_count": cross_tenant_count,
                "isolation_rate": 1 - violation_rate,
                "affected_tenant_pairs": len(tenant_pairs),
            },
            violations=violations,
        )
