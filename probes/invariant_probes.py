"""
Invariant Monitoring Probes

Runtime probes that monitor formal invariants defined in Phase 3:
- P-DET: Determinism
- P-TERM: Termination
- P-ACYCLIC: Acyclicity
- P-AUD: Audit Completeness
- P-NONREP: Non-repudiation

These probes provide continuous validation that the system maintains its
formal guarantees in production.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import time

from .base_probe import BaseProbe, ProbeResult, ProbeStatus


class DeterminismProbe(BaseProbe):
    """
    P-DET: Determinism Probe
    
    Validates that identical inputs produce identical outputs, ensuring
    reproducibility of decisions. This is critical for appeals and audits.
    
    Checks:
    - Same policy + context → same decision
    - Decision hashes match for repeated evaluations
    - No non-deterministic randomness in decision path
    """
    
    def __init__(
        self,
        evaluation_service: Any,
        check_interval_seconds: int = 300,
        sample_size: int = 10,
    ):
        """
        Initialize determinism probe.
        
        Args:
            evaluation_service: Service to test for determinism
            check_interval_seconds: Check interval
            sample_size: Number of test cases to validate
        """
        super().__init__(
            name="P-DET-Determinism",
            check_interval_seconds=check_interval_seconds,
        )
        self.evaluation_service = evaluation_service
        self.sample_size = sample_size
        self._test_cases: List[Dict[str, Any]] = []
    
    def add_test_case(self, policy_id: str, context: Dict[str, Any]):
        """Add a test case for determinism validation"""
        self._test_cases.append({
            "policy_id": policy_id,
            "context": context,
        })
    
    def check(self) -> ProbeResult:
        """Check determinism property"""
        timestamp = datetime.utcnow()
        violations = []
        test_results = []
        
        if not self._test_cases:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.WARNING,
                timestamp=timestamp,
                message="No test cases configured for determinism checking",
                metrics={"test_cases": 0},
            )
        
        # Test a sample of cases
        for i, test_case in enumerate(self._test_cases[:self.sample_size]):
            try:
                # Evaluate twice with same inputs
                result1 = self._evaluate_case(test_case)
                result2 = self._evaluate_case(test_case)
                
                # Compare results
                hash1 = self._hash_result(result1)
                hash2 = self._hash_result(result2)
                
                if hash1 != hash2:
                    violations.append(
                        f"Non-deterministic result for policy {test_case['policy_id']}: "
                        f"hash1={hash1[:8]}, hash2={hash2[:8]}"
                    )
                
                test_results.append({
                    "policy_id": test_case["policy_id"],
                    "deterministic": hash1 == hash2,
                    "hash": hash1,
                })
                
            except Exception as e:
                violations.append(f"Error evaluating test case {i}: {str(e)}")
        
        # Determine status
        if violations:
            status = ProbeStatus.CRITICAL
            message = f"Determinism violations detected: {len(violations)} cases"
        else:
            status = ProbeStatus.HEALTHY
            message = f"All {len(test_results)} test cases deterministic"
        
        return ProbeResult(
            probe_name=self.name,
            status=status,
            timestamp=timestamp,
            message=message,
            metrics={
                "test_cases_checked": len(test_results),
                "violations_count": len(violations),
                "determinism_rate": (len(test_results) - len(violations)) / len(test_results) 
                    if test_results else 0,
            },
            violations=violations,
            details={"test_results": test_results},
        )
    
    def _evaluate_case(self, test_case: Dict[str, Any]) -> Any:
        """Evaluate a single test case (placeholder for actual evaluation)"""
        # In real implementation, this would call the actual evaluation service
        # For now, return a mock result
        return {
            "policy_id": test_case["policy_id"],
            "decision": "allow",
            "context": test_case["context"],
            "timestamp": time.time(),
        }
    
    def _hash_result(self, result: Any) -> str:
        """Hash a result for comparison"""
        # Normalize result for hashing (remove timestamps, etc.)
        normalized = {
            "policy_id": result.get("policy_id"),
            "decision": result.get("decision"),
            "context": str(sorted(result.get("context", {}).items())),
        }
        return hashlib.sha256(str(normalized).encode()).hexdigest()


class TerminationProbe(BaseProbe):
    """
    P-TERM: Termination Probe
    
    Validates that all policy evaluations complete within bounded time,
    preventing infinite loops or resource exhaustion.
    
    Checks:
    - Evaluation duration < timeout threshold
    - No hanging evaluations
    - Resource consumption within bounds
    """
    
    def __init__(
        self,
        max_evaluation_time_ms: int = 5000,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize termination probe.
        
        Args:
            max_evaluation_time_ms: Maximum allowed evaluation time
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-TERM-Termination",
            check_interval_seconds=check_interval_seconds,
        )
        self.max_evaluation_time_ms = max_evaluation_time_ms
        self._recent_evaluations: List[Dict[str, Any]] = []
    
    def record_evaluation(
        self,
        policy_id: str,
        duration_ms: float,
        completed: bool,
    ):
        """Record an evaluation for monitoring"""
        self._recent_evaluations.append({
            "policy_id": policy_id,
            "duration_ms": duration_ms,
            "completed": completed,
            "timestamp": datetime.utcnow(),
        })
        # Keep only recent evaluations
        if len(self._recent_evaluations) > 1000:
            self._recent_evaluations.pop(0)
    
    def check(self) -> ProbeResult:
        """Check termination property"""
        timestamp = datetime.utcnow()
        violations = []
        
        if not self._recent_evaluations:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.HEALTHY,
                timestamp=timestamp,
                message="No recent evaluations to check",
                metrics={"evaluations_checked": 0},
            )
        
        # Check for timeout violations
        timeout_count = 0
        incomplete_count = 0
        total_duration = 0
        
        for eval_data in self._recent_evaluations:
            duration = eval_data["duration_ms"]
            total_duration += duration
            
            if not eval_data["completed"]:
                incomplete_count += 1
                violations.append(
                    f"Incomplete evaluation for policy {eval_data['policy_id']}"
                )
            elif duration > self.max_evaluation_time_ms:
                timeout_count += 1
                violations.append(
                    f"Timeout for policy {eval_data['policy_id']}: "
                    f"{duration:.2f}ms > {self.max_evaluation_time_ms}ms"
                )
        
        total = len(self._recent_evaluations)
        avg_duration = total_duration / total if total > 0 else 0
        
        # Determine status
        if incomplete_count > 0 or timeout_count > total * 0.05:
            status = ProbeStatus.CRITICAL
            message = f"Termination violations: {incomplete_count} incomplete, {timeout_count} timeouts"
        elif timeout_count > 0:
            status = ProbeStatus.WARNING
            message = f"Some evaluations exceeded timeout: {timeout_count} cases"
        else:
            status = ProbeStatus.HEALTHY
            message = f"All {total} evaluations completed within bounds"
        
        return ProbeResult(
            probe_name=self.name,
            status=status,
            timestamp=timestamp,
            message=message,
            metrics={
                "evaluations_checked": total,
                "timeout_violations": timeout_count,
                "incomplete_evaluations": incomplete_count,
                "avg_duration_ms": avg_duration,
                "max_duration_ms": max(e["duration_ms"] for e in self._recent_evaluations),
                "termination_rate": (total - incomplete_count) / total if total > 0 else 0,
            },
            violations=violations[:10],  # Limit violation list
        )


class AcyclicityProbe(BaseProbe):
    """
    P-ACYCLIC: Acyclicity Probe
    
    Validates that policy dependencies form a directed acyclic graph (DAG),
    preventing circular dependencies that could cause evaluation loops.
    
    Checks:
    - No cycles in policy dependency graph
    - Policy evaluation order is valid
    - Dependency depth within bounds
    """
    
    def __init__(
        self,
        policy_graph: Optional[Dict[str, List[str]]] = None,
        max_depth: int = 10,
        check_interval_seconds: int = 300,
    ):
        """
        Initialize acyclicity probe.
        
        Args:
            policy_graph: Adjacency list of policy dependencies
            max_depth: Maximum allowed dependency depth
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-ACYCLIC-Acyclicity",
            check_interval_seconds=check_interval_seconds,
        )
        self.policy_graph = policy_graph or {}
        self.max_depth = max_depth
    
    def update_graph(self, policy_graph: Dict[str, List[str]]):
        """Update the policy dependency graph"""
        self.policy_graph = policy_graph
    
    def check(self) -> ProbeResult:
        """Check acyclicity property"""
        timestamp = datetime.utcnow()
        violations = []
        
        if not self.policy_graph:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.WARNING,
                timestamp=timestamp,
                message="No policy graph configured",
                metrics={"policies": 0},
            )
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        cycles_found = []
        
        def has_cycle(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.policy_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path[:]):
                        return True
                elif neighbor in rec_stack:
                    cycle = path[path.index(neighbor):] + [neighbor]
                    cycles_found.append(" -> ".join(cycle))
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check each policy
        for policy_id in self.policy_graph:
            if policy_id not in visited:
                if has_cycle(policy_id, []):
                    violations.append(f"Cycle detected involving {policy_id}")
        
        # Check depth
        max_depth_found = self._calculate_max_depth()
        if max_depth_found > self.max_depth:
            violations.append(
                f"Dependency depth {max_depth_found} exceeds maximum {self.max_depth}"
            )
        
        # Determine status
        if cycles_found:
            status = ProbeStatus.CRITICAL
            message = f"Cycles detected in policy graph: {len(cycles_found)} cycles"
        elif violations:
            status = ProbeStatus.WARNING
            message = "Policy graph warnings detected"
        else:
            status = ProbeStatus.HEALTHY
            message = f"Policy graph is acyclic with {len(self.policy_graph)} policies"
        
        return ProbeResult(
            probe_name=self.name,
            status=status,
            timestamp=timestamp,
            message=message,
            metrics={
                "policies_count": len(self.policy_graph),
                "cycles_found": len(cycles_found),
                "max_depth": max_depth_found,
            },
            violations=violations,
            details={"cycles": cycles_found[:5]},  # Limit cycles shown
        )
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of policy dependency graph"""
        if not self.policy_graph:
            return 0
        
        def dfs_depth(node: str, visited: set) -> int:
            if node in visited:
                return 0
            visited.add(node)
            max_child_depth = 0
            for neighbor in self.policy_graph.get(node, []):
                max_child_depth = max(max_child_depth, dfs_depth(neighbor, visited.copy()))
            return 1 + max_child_depth
        
        return max(dfs_depth(node, set()) for node in self.policy_graph)


class AuditCompletenessProbe(BaseProbe):
    """
    P-AUD: Audit Completeness Probe
    
    Validates that all decisions are fully audited with complete information
    for accountability and compliance.
    
    Checks:
    - All decisions have audit entries
    - Audit entries contain required fields
    - Audit log is append-only and monotonic
    """
    
    def __init__(
        self,
        audit_service: Any,
        required_fields: List[str] = None,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize audit completeness probe.
        
        Args:
            audit_service: Service providing audit logs
            required_fields: Required fields in audit entries
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-AUD-AuditCompleteness",
            check_interval_seconds=check_interval_seconds,
        )
        self.audit_service = audit_service
        self.required_fields = required_fields or [
            "timestamp",
            "policy_id",
            "decision",
            "context",
            "agent_id",
        ]
        self._last_audit_count = 0
    
    def check(self) -> ProbeResult:
        """Check audit completeness property"""
        timestamp = datetime.utcnow()
        violations = []
        
        try:
            # Get recent audit entries (placeholder)
            audit_entries = self._get_recent_audits()
            current_count = len(audit_entries)
            
            # Check monotonicity
            if current_count < self._last_audit_count:
                violations.append(
                    f"Audit log not monotonic: count decreased from "
                    f"{self._last_audit_count} to {current_count}"
                )
            
            # Check completeness
            incomplete_count = 0
            for entry in audit_entries[-100:]:  # Check recent 100
                missing_fields = [
                    field for field in self.required_fields
                    if field not in entry or entry[field] is None
                ]
                if missing_fields:
                    incomplete_count += 1
                    if len(violations) < 10:  # Limit violations
                        violations.append(
                            f"Incomplete audit entry: missing {missing_fields}"
                        )
            
            self._last_audit_count = current_count
            
            # Determine status
            if violations and "not monotonic" in violations[0]:
                status = ProbeStatus.CRITICAL
                message = "Audit log integrity violation detected"
            elif incomplete_count > 0:
                status = ProbeStatus.WARNING
                message = f"{incomplete_count} incomplete audit entries"
            else:
                status = ProbeStatus.HEALTHY
                message = f"All audit entries complete ({current_count} total)"
            
            return ProbeResult(
                probe_name=self.name,
                status=status,
                timestamp=timestamp,
                message=message,
                metrics={
                    "total_audit_entries": current_count,
                    "incomplete_entries": incomplete_count,
                    "completeness_rate": (len(audit_entries) - incomplete_count) / len(audit_entries)
                        if audit_entries else 1.0,
                },
                violations=violations,
            )
            
        except Exception as e:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.CRITICAL,
                timestamp=timestamp,
                message=f"Failed to check audit completeness: {str(e)}",
                metrics={},
            )
    
    def _get_recent_audits(self) -> List[Dict[str, Any]]:
        """Get recent audit entries (placeholder)"""
        # In real implementation, this would query the audit service
        return []


class NonRepudiationProbe(BaseProbe):
    """
    P-NONREP: Non-repudiation Probe
    
    Validates that audit logs are cryptographically signed and tamper-evident,
    ensuring decisions cannot be repudiated.
    
    Checks:
    - Audit entries have valid signatures
    - Merkle tree roots match expected values
    - No evidence of tampering
    """
    
    def __init__(
        self,
        audit_service: Any,
        check_interval_seconds: int = 300,
    ):
        """
        Initialize non-repudiation probe.
        
        Args:
            audit_service: Service providing signed audit logs
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="P-NONREP-NonRepudiation",
            check_interval_seconds=check_interval_seconds,
        )
        self.audit_service = audit_service
        self._last_merkle_root: Optional[str] = None
    
    def check(self) -> ProbeResult:
        """Check non-repudiation property"""
        timestamp = datetime.utcnow()
        violations = []
        
        try:
            # Get Merkle root and verify signatures
            current_root = self._get_merkle_root()
            signature_valid = self._verify_signature(current_root)
            
            if not signature_valid:
                violations.append("Merkle root signature verification failed")
            
            # Check for tampering (root should only grow, never change historical)
            if self._last_merkle_root and not self._is_valid_successor(current_root):
                violations.append(
                    f"Potential tampering detected: invalid Merkle root transition"
                )
            
            self._last_merkle_root = current_root
            
            # Verify sample of audit entries
            sample_results = self._verify_audit_sample()
            invalid_signatures = sum(1 for r in sample_results if not r["valid"])
            
            if invalid_signatures > 0:
                violations.append(
                    f"{invalid_signatures} audit entries have invalid signatures"
                )
            
            # Determine status
            if violations:
                status = ProbeStatus.CRITICAL
                message = f"Non-repudiation violations: {len(violations)}"
            else:
                status = ProbeStatus.HEALTHY
                message = "All audit signatures valid, no tampering detected"
            
            return ProbeResult(
                probe_name=self.name,
                status=status,
                timestamp=timestamp,
                message=message,
                metrics={
                    "merkle_root": current_root[:16] if current_root else None,
                    "signature_valid": signature_valid,
                    "sample_size": len(sample_results),
                    "invalid_signatures": invalid_signatures,
                },
                violations=violations,
            )
            
        except Exception as e:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.CRITICAL,
                timestamp=timestamp,
                message=f"Failed to check non-repudiation: {str(e)}",
                metrics={},
            )
    
    def _get_merkle_root(self) -> str:
        """Get current Merkle root (placeholder)"""
        # In real implementation, get from audit service
        return hashlib.sha256(b"mock_root").hexdigest()
    
    def _verify_signature(self, root: str) -> bool:
        """Verify Merkle root signature (placeholder)"""
        # In real implementation, verify cryptographic signature
        return True
    
    def _is_valid_successor(self, new_root: str) -> bool:
        """Check if new root is valid successor to previous root (placeholder)"""
        # In real implementation, verify Merkle tree consistency
        return True
    
    def _verify_audit_sample(self) -> List[Dict[str, Any]]:
        """Verify signatures of sample audit entries (placeholder)"""
        # In real implementation, sample and verify audit entries
        return []
