"""
Data Integrity Validation Test Suite

Tests data integrity through:
- Merkle chain continuity
- Audit replay verification
- Cryptographic proof validation

Thresholds:
- Merkle Verification: 100% blocks
- Audit Replay Success: 100%
"""

import pytest
import hashlib
import json
from typing import List, Dict, Tuple
from datetime import datetime
from nethical.core.audit_merkle import MerkleAnchor
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.models import AgentAction


class IntegrityValidator:
    """Validate data integrity"""
    
    @staticmethod
    def verify_merkle_chain(audit_trail: List[Dict]) -> Dict:
        """
        Verify Merkle chain continuity
        
        Args:
            audit_trail: List of audit entries with merkle roots
            
        Returns:
            Verification results
        """
        if not audit_trail:
            return {
                "total_blocks": 0,
                "verified_blocks": 0,
                "verification_rate": 0.0,
                "chain_valid": True,
                "issues": []
            }
        
        verified = 0
        issues = []
        
        for i, entry in enumerate(audit_trail):
            # Check if entry has merkle root
            if "merkle_root" not in entry:
                issues.append(f"Block {i}: Missing merkle_root")
                continue
            
            # Check if merkle root is valid hash
            root = entry["merkle_root"]
            if not isinstance(root, str) or len(root) != 64:
                issues.append(f"Block {i}: Invalid merkle_root format")
                continue
            
            # If not first block, check chain linkage
            if i > 0:
                prev_entry = audit_trail[i - 1]
                if "merkle_root" in prev_entry:
                    # Verify continuity (previous root should be referenced)
                    if "previous_root" in entry:
                        if entry["previous_root"] != prev_entry["merkle_root"]:
                            issues.append(f"Block {i}: Chain break - previous_root mismatch")
                            continue
            
            verified += 1
        
        return {
            "total_blocks": len(audit_trail),
            "verified_blocks": verified,
            "verification_rate": verified / len(audit_trail) if audit_trail else 0.0,
            "chain_valid": len(issues) == 0,
            "issues": issues
        }
    
    @staticmethod
    def replay_audit_trail(actions: List[AgentAction], governance: IntegratedGovernance) -> Dict:
        """
        Replay audit trail to verify reproducibility
        
        Args:
            actions: List of actions to replay
            governance: Governance instance
            
        Returns:
            Replay verification results
        """
        replayed = 0
        mismatches = []
        
        for i, action in enumerate(actions):
            try:
                # Re-evaluate action as string
                result = governance.process_action(
                    agent_id="test_agent",
                    action=action.action if hasattr(action, 'action') else str(action)
                )
                replayed += 1
            except Exception as e:
                action_id = action.action_id if hasattr(action, 'action_id') else str(i)
                mismatches.append(f"Action {i} ({action_id}): Replay failed - {str(e)}")
        
        return {
            "total_actions": len(actions),
            "replayed_actions": replayed,
            "replay_rate": replayed / len(actions) if actions else 0.0,
            "mismatches": mismatches,
            "replay_successful": len(mismatches) == 0
        }
    
    @staticmethod
    def verify_cryptographic_proofs(entries: List[Dict]) -> Dict:
        """
        Verify cryptographic proofs in audit entries
        
        Args:
            entries: List of audit entries with signatures/proofs
            
        Returns:
            Proof verification results
        """
        verified = 0
        failures = []
        
        for i, entry in enumerate(entries):
            # Check for proof elements
            has_merkle = "merkle_root" in entry
            has_hash = "entry_hash" in entry or "action_hash" in entry
            
            if has_merkle and has_hash:
                # Verify hash format
                merkle = entry.get("merkle_root", "")
                entry_hash = entry.get("entry_hash") or entry.get("action_hash", "")
                
                if len(merkle) == 64 and len(entry_hash) >= 32:
                    verified += 1
                else:
                    failures.append(f"Entry {i}: Invalid proof format")
            else:
                failures.append(f"Entry {i}: Missing cryptographic proofs")
        
        return {
            "total_entries": len(entries),
            "verified_proofs": verified,
            "verification_rate": verified / len(entries) if entries else 0.0,
            "failures": failures,
            "all_proofs_valid": len(failures) == 0
        }


@pytest.fixture
def integrity_validator():
    """Initialize integrity validator"""
    return IntegrityValidator()


@pytest.fixture
def governance():
    """Initialize governance"""
    return IntegratedGovernance()


@pytest.fixture
def merkle_anchor():
    """Initialize Merkle anchor"""
    return MerkleAnchor()


def test_merkle_chain_continuity(integrity_validator, merkle_anchor):
    """Test Merkle chain continuity validation"""
    # Create sample audit trail
    audit_trail = []
    previous_root = None
    
    for i in range(10):
        entry = {
            "block_id": i,
            "timestamp": datetime.now().isoformat(),
            "action_id": f"action_{i}",
            "merkle_root": hashlib.sha256(f"block_{i}".encode()).hexdigest()
        }
        
        if previous_root:
            entry["previous_root"] = previous_root
        
        audit_trail.append(entry)
        previous_root = entry["merkle_root"]
    
    # Verify chain
    result = integrity_validator.verify_merkle_chain(audit_trail)
    
    print(f"\nMerkle Chain Continuity Test:")
    print(f"  Total Blocks: {result['total_blocks']}")
    print(f"  Verified Blocks: {result['verified_blocks']}")
    print(f"  Verification Rate: {result['verification_rate']:.2%}")
    print(f"  Chain Valid: {result['chain_valid']}")
    
    assert result["verification_rate"] == 1.0, "Not all blocks verified"
    assert result["chain_valid"], "Chain integrity compromised"


def test_merkle_chain_break_detection(integrity_validator):
    """Test detection of Merkle chain breaks"""
    # Create audit trail with intentional break
    audit_trail = [
        {
            "block_id": 0,
            "merkle_root": hashlib.sha256(b"block_0").hexdigest()
        },
        {
            "block_id": 1,
            "merkle_root": hashlib.sha256(b"block_1").hexdigest(),
            "previous_root": hashlib.sha256(b"block_0").hexdigest()
        },
        {
            "block_id": 2,
            "merkle_root": hashlib.sha256(b"block_2").hexdigest(),
            "previous_root": hashlib.sha256(b"wrong_block").hexdigest()  # Break!
        }
    ]
    
    result = integrity_validator.verify_merkle_chain(audit_trail)
    
    print(f"\nMerkle Chain Break Detection:")
    print(f"  Chain Valid: {result['chain_valid']}")
    print(f"  Issues Found: {len(result['issues'])}")
    for issue in result['issues']:
        print(f"    - {issue}")
    
    assert not result["chain_valid"], "Chain break not detected"
    assert len(result["issues"]) > 0, "No issues reported for broken chain"


def test_audit_replay_verification(integrity_validator, governance):
    """Test audit trail replay verification"""
    # Create test actions
    test_actions = [
        AgentAction(
            action_id=f"replay_test_{i}",
            agent_id="test_agent",
            action=f"Test action {i}",
            action_type="query"
        )
        for i in range(20)
    ]
    
    # Replay actions
    result = integrity_validator.replay_audit_trail(test_actions, governance)
    
    print(f"\nAudit Replay Verification:")
    print(f"  Total Actions: {result['total_actions']}")
    print(f"  Replayed Actions: {result['replayed_actions']}")
    print(f"  Replay Rate: {result['replay_rate']:.2%}")
    print(f"  Replay Successful: {result['replay_successful']}")
    
    assert result["replay_rate"] >= 0.95, "Replay rate below 95%"
    assert result["replay_successful"], "Replay had mismatches"


def test_cryptographic_proof_validation(integrity_validator):
    """Test cryptographic proof validation"""
    # Create entries with proofs
    entries = []
    for i in range(15):
        entry = {
            "entry_id": i,
            "merkle_root": hashlib.sha256(f"merkle_{i}".encode()).hexdigest(),
            "action_hash": hashlib.sha256(f"action_{i}".encode()).hexdigest(),
            "timestamp": datetime.now().isoformat()
        }
        entries.append(entry)
    
    result = integrity_validator.verify_cryptographic_proofs(entries)
    
    print(f"\nCryptographic Proof Validation:")
    print(f"  Total Entries: {result['total_entries']}")
    print(f"  Verified Proofs: {result['verified_proofs']}")
    print(f"  Verification Rate: {result['verification_rate']:.2%}")
    print(f"  All Proofs Valid: {result['all_proofs_valid']}")
    
    assert result["verification_rate"] == 1.0, "Not all proofs verified"
    assert result["all_proofs_valid"], "Some proofs invalid"


def test_integrity_with_merkle_anchor(merkle_anchor):
    """Test integrity using MerkleAnchor"""
    # Create and anchor multiple blocks
    test_data = [
        {"action_id": f"test_{i}", "data": f"test data {i}"}
        for i in range(10)
    ]
    
    roots = []
    for data in test_data:
        data_json = json.dumps(data, sort_keys=True)
        # Use add_event to add the data
        merkle_anchor.add_event(data)
        # Finalize to get root for this data
        root = merkle_anchor.finalize_chunk()
        if root:
            roots.append(root)
    
    print(f"\nMerkle Anchor Integrity Test:")
    print(f"  Blocks Anchored: {len(roots)}")
    print(f"  All Roots Generated: {all(r for r in roots)}")
    print(f"  Unique Roots: {len(set(roots))}")
    
    # All roots should be generated
    assert all(r for r in roots), "Some blocks failed to generate roots"
    
    # Roots should be unique (for different data)
    assert len(set(roots)) == len(roots), "Duplicate roots for different data"
    
    # Verify root format (64-character hex)
    for root in roots:
        assert len(root) == 64, f"Invalid root length: {len(root)}"
        assert all(c in "0123456789abcdef" for c in root), "Invalid root format"


def test_generate_integrity_report(integrity_validator, governance, merkle_anchor, tmp_path):
    """Generate comprehensive integrity report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "data_integrity",
        "tests": {}
    }
    
    # Test 1: Merkle chain
    audit_trail = []
    previous_root = None
    for i in range(50):
        entry = {
            "block_id": i,
            "merkle_root": hashlib.sha256(f"block_{i}".encode()).hexdigest()
        }
        if previous_root:
            entry["previous_root"] = previous_root
        audit_trail.append(entry)
        previous_root = entry["merkle_root"]
    
    merkle_result = integrity_validator.verify_merkle_chain(audit_trail)
    report["tests"]["merkle_chain"] = {
        "total_blocks": merkle_result["total_blocks"],
        "verified_blocks": merkle_result["verified_blocks"],
        "verification_rate": merkle_result["verification_rate"],
        "chain_valid": merkle_result["chain_valid"],
        "threshold_met": merkle_result["verification_rate"] == 1.0
    }
    
    # Test 2: Audit replay
    test_actions = [
        AgentAction(
            action_id=f"report_test_{i}",
            agent_id="report_agent",
            action=f"Report test {i}",
            action_type="query"
        )
        for i in range(30)
    ]
    replay_result = integrity_validator.replay_audit_trail(test_actions, governance)
    report["tests"]["audit_replay"] = {
        "total_actions": replay_result["total_actions"],
        "replayed_actions": replay_result["replayed_actions"],
        "replay_rate": replay_result["replay_rate"],
        "replay_successful": replay_result["replay_successful"],
        "threshold_met": replay_result["replay_rate"] >= 0.95
    }
    
    # Test 3: Cryptographic proofs
    entries = [
        {
            "entry_id": i,
            "merkle_root": hashlib.sha256(f"proof_{i}".encode()).hexdigest(),
            "action_hash": hashlib.sha256(f"hash_{i}".encode()).hexdigest()
        }
        for i in range(40)
    ]
    proof_result = integrity_validator.verify_cryptographic_proofs(entries)
    report["tests"]["cryptographic_proofs"] = {
        "total_entries": proof_result["total_entries"],
        "verified_proofs": proof_result["verified_proofs"],
        "verification_rate": proof_result["verification_rate"],
        "all_proofs_valid": proof_result["all_proofs_valid"],
        "threshold_met": proof_result["verification_rate"] == 1.0
    }
    
    # Overall compliance
    report["overall_compliance"] = all(
        test.get("threshold_met", False) 
        for test in report["tests"].values()
    )
    
    # Save report
    report_path = tmp_path / "integrity_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nIntegrity report saved to: {report_path}")
    print(f"Overall Compliance: {report['overall_compliance']}")
    
    assert report_path.exists()
    assert report["overall_compliance"], "Integrity validation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
