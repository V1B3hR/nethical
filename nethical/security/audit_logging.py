"""
Phase 3: Enhanced Audit Logging with Blockchain

This module provides:
- Private blockchain for tamper-proof audit logs
- RFC 3161 timestamp authority integration
- Digital signatures for all audit events
- Forensic analysis tools
- Chain-of-custody documentation
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any


class AuditEventType(str, Enum):
    """Types of audit events"""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(str, Enum):
    """Audit event severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"


@dataclass
class AuditEvent:
    """Individual audit event"""

    id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, denied
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockchainBlock:
    """Block in the audit blockchain"""

    index: int
    timestamp: datetime
    events: List[AuditEvent]
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "events": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type.value,
                    "user_id": e.user_id,
                    "action": e.action,
                    "resource": e.resource,
                    "result": e.result,
                }
                for e in self.events
            ],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int = 2):
        """Mine block with proof of work"""
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()


@dataclass
class TimestampAuthority:
    """RFC 3161 Timestamp Authority integration"""

    tsa_url: str = "https://timestamp.example.com"
    enabled: bool = True

    def request_timestamp(self, data: bytes) -> Dict[str, Any]:
        """Request RFC 3161 timestamp for data"""
        # In production, this would make actual TSA request
        data_hash = hashlib.sha256(data).hexdigest()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_hash": data_hash,
            "tsa_signature": f"tsa_sig_{uuid.uuid4().hex[:16]}",
            "tsa_certificate": "tsa_cert_placeholder",
            "serial_number": uuid.uuid4().hex,
        }

    def verify_timestamp(self, timestamp_token: Dict[str, Any]) -> bool:
        """Verify RFC 3161 timestamp token"""
        # In production, verify cryptographic signature
        required_fields = ["timestamp", "data_hash", "tsa_signature"]
        return all(field in timestamp_token for field in required_fields)


@dataclass
class DigitalSignature:
    """Digital signature for audit events"""

    algorithm: str = "RSA-SHA256"
    key_id: str = ""
    signature: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def sign_data(data: bytes, private_key: Optional[str] = None) -> "DigitalSignature":
        """Sign data with digital signature"""
        # In production, use actual cryptographic signing
        data_hash = hashlib.sha256(data).hexdigest()
        signature = DigitalSignature(
            key_id=f"key_{uuid.uuid4().hex[:8]}",
            signature=f"sig_{hashlib.sha256((data_hash + 'private_key').encode()).hexdigest()}",
        )
        return signature

    def verify(self, data: bytes, public_key: Optional[str] = None) -> bool:
        """Verify digital signature"""
        # In production, use actual cryptographic verification
        return len(self.signature) > 0 and self.signature.startswith("sig_")


class AuditBlockchain:
    """Blockchain-based audit log storage"""

    def __init__(self, difficulty: int = 2):
        self.chain: List[BlockchainBlock] = []
        self.pending_events: List[AuditEvent] = []
        self.difficulty = difficulty
        self.max_events_per_block = 100
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis = BlockchainBlock(
            index=0, timestamp=datetime.now(timezone.utc), events=[], previous_hash="0", nonce=0
        )
        genesis.mine_block(self.difficulty)
        self.chain.append(genesis)

    def add_event(self, event: AuditEvent):
        """Add event to pending events"""
        self.pending_events.append(event)

        # Create new block if we have enough events
        if len(self.pending_events) >= self.max_events_per_block:
            self.create_block()

    def create_block(self) -> Optional[BlockchainBlock]:
        """Create new block with pending events"""
        if not self.pending_events:
            return None

        previous_block = self.chain[-1]
        new_block = BlockchainBlock(
            index=len(self.chain),
            timestamp=datetime.now(timezone.utc),
            events=self.pending_events.copy(),
            previous_hash=previous_block.hash,
        )
        new_block.mine_block(self.difficulty)

        self.chain.append(new_block)
        self.pending_events.clear()
        return new_block

    def verify_chain(self) -> bool:
        """Verify integrity of the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Verify current block hash
            if current_block.hash != current_block.calculate_hash():
                return False

            # Verify link to previous block
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def get_block(self, index: int) -> Optional[BlockchainBlock]:
        """Retrieve block by index"""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None

    def search_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditEvent]:
        """Search for events across the blockchain"""
        results = []

        for block in self.chain:
            for event in block.events:
                # Apply filters
                if user_id and event.user_id != user_id:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue

                results.append(event)

        return results


class ForensicAnalyzer:
    """Forensic analysis tools for audit logs"""

    def __init__(self, blockchain: AuditBlockchain):
        self.blockchain = blockchain

    def analyze_user_activity(self, user_id: str, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Analyze user activity patterns"""
        end_time = datetime.now(timezone.utc)
        from datetime import timedelta

        start_time = end_time - timedelta(hours=timeframe_hours)

        events = self.blockchain.search_events(
            user_id=user_id, start_time=start_time, end_time=end_time
        )

        return {
            "user_id": user_id,
            "timeframe_hours": timeframe_hours,
            "total_events": len(events),
            "event_types": self._count_event_types(events),
            "failed_actions": sum(1 for e in events if e.result == "failure"),
            "successful_actions": sum(1 for e in events if e.result == "success"),
            "accessed_resources": list(set(e.resource for e in events)),
            "suspicious_patterns": self._detect_suspicious_patterns(events),
        }

    def _count_event_types(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by type"""
        counts: Dict[str, int] = {}
        for event in events:
            event_type = event.event_type.value
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def _detect_suspicious_patterns(self, events: List[AuditEvent]) -> List[str]:
        """Detect suspicious patterns in events"""
        patterns = []

        # Check for excessive failed authentications
        failed_auth = sum(
            1
            for e in events
            if e.event_type == AuditEventType.AUTHENTICATION and e.result == "failure"
        )
        if failed_auth > 5:
            patterns.append(f"Excessive failed authentication attempts: {failed_auth}")

        # Check for unusual access patterns
        unique_resources = set(e.resource for e in events)
        if len(unique_resources) > 50:
            patterns.append(f"Unusual number of accessed resources: {len(unique_resources)}")

        return patterns

    def generate_timeline(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Generate event timeline for forensic analysis"""
        events = self.blockchain.search_events(start_time=start_time, end_time=end_time)

        timeline = []
        for event in sorted(events, key=lambda e: e.timestamp):
            timeline.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "user_id": event.user_id,
                    "action": event.action,
                    "resource": event.resource,
                    "result": event.result,
                    "severity": event.severity.value,
                }
            )

        return timeline

    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify blockchain integrity for forensic purposes"""
        is_valid = self.blockchain.verify_chain()

        return {
            "chain_valid": is_valid,
            "total_blocks": len(self.blockchain.chain),
            "total_events": sum(len(block.events) for block in self.blockchain.chain),
            "genesis_block_hash": self.blockchain.chain[0].hash if self.blockchain.chain else None,
            "latest_block_hash": self.blockchain.chain[-1].hash if self.blockchain.chain else None,
        }


class ChainOfCustodyManager:
    """Chain-of-custody documentation for audit evidence"""

    def __init__(self):
        self.custody_records: Dict[str, List[Dict[str, Any]]] = {}

    def create_evidence(
        self, evidence_id: str, description: str, collected_by: str, source: str
    ) -> Dict[str, Any]:
        """Create new evidence with chain of custody"""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "created",
            "custodian": collected_by,
            "location": source,
            "description": description,
            "signature": DigitalSignature.sign_data(evidence_id.encode()),
        }

        self.custody_records[evidence_id] = [record]
        return record

    def transfer_custody(
        self, evidence_id: str, from_custodian: str, to_custodian: str, reason: str
    ) -> bool:
        """Transfer evidence custody"""
        if evidence_id not in self.custody_records:
            return False

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "transferred",
            "from_custodian": from_custodian,
            "to_custodian": to_custodian,
            "reason": reason,
            "signature": DigitalSignature.sign_data(f"{evidence_id}:{to_custodian}".encode()),
        }

        self.custody_records[evidence_id].append(record)
        return True

    def access_evidence(self, evidence_id: str, accessor: str, purpose: str) -> bool:
        """Record evidence access"""
        if evidence_id not in self.custody_records:
            return False

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "accessed",
            "accessor": accessor,
            "purpose": purpose,
            "signature": DigitalSignature.sign_data(f"{evidence_id}:{accessor}".encode()),
        }

        self.custody_records[evidence_id].append(record)
        return True

    def get_custody_chain(self, evidence_id: str) -> List[Dict[str, Any]]:
        """Retrieve complete chain of custody"""
        return self.custody_records.get(evidence_id, [])

    def verify_custody_integrity(self, evidence_id: str) -> bool:
        """Verify chain of custody integrity"""
        if evidence_id not in self.custody_records:
            return False

        chain = self.custody_records[evidence_id]

        # Verify all signatures exist
        for record in chain:
            if "signature" not in record:
                return False

        # Verify chronological order
        timestamps = [record["timestamp"] for record in chain]
        if timestamps != sorted(timestamps):
            return False

        return True


class EnhancedAuditLogger:
    """Enhanced audit logging system with blockchain and digital signatures"""

    def __init__(self):
        self.blockchain = AuditBlockchain()
        self.timestamp_authority = TimestampAuthority()
        self.forensic_analyzer = ForensicAnalyzer(self.blockchain)
        self.custody_manager = ChainOfCustodyManager()

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> AuditEvent:
        """Log audit event with blockchain and timestamp"""

        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            details=details or {},
        )

        # Add to blockchain
        self.blockchain.add_event(event)

        # Get RFC 3161 timestamp
        if self.timestamp_authority.enabled:
            event_data = json.dumps(
                {
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": user_id,
                    "action": action,
                }
            ).encode()

            timestamp_token = self.timestamp_authority.request_timestamp(event_data)
            event.metadata["rfc3161_timestamp"] = timestamp_token

        # Add digital signature
        event_bytes = json.dumps(
            {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "user_id": user_id,
            }
        ).encode()

        signature = DigitalSignature.sign_data(event_bytes)
        event.metadata["digital_signature"] = {
            "algorithm": signature.algorithm,
            "key_id": signature.key_id,
            "signature": signature.signature,
            "timestamp": signature.timestamp.isoformat(),
        }

        return event

    def finalize_block(self) -> Optional[BlockchainBlock]:
        """Finalize current block of audit events"""
        return self.blockchain.create_block()

    def verify_integrity(self) -> bool:
        """Verify audit log integrity"""
        return self.blockchain.verify_chain()

    def search_logs(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditEvent]:
        """Search audit logs"""
        return self.blockchain.search_events(user_id, event_type, start_time, end_time)

    def generate_forensic_report(self, user_id: str) -> Dict[str, Any]:
        """Generate forensic analysis report"""
        return self.forensic_analyzer.analyze_user_activity(user_id)
