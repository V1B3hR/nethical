"""
Unit tests for Phase 3: Enhanced Audit Logging with Blockchain
"""

import pytest
from datetime import datetime, timedelta, timezone
from nethical.security.audit_logging import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    BlockchainBlock,
    TimestampAuthority,
    DigitalSignature,
    AuditBlockchain,
    ForensicAnalyzer,
    ChainOfCustodyManager,
    EnhancedAuditLogger,
)


class TestAuditEvent:
    """Test audit event model"""
    
    def test_event_creation(self):
        """Test creating an audit event"""
        event = AuditEvent(
            id="evt-123",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.AUTHENTICATION,
            severity=AuditSeverity.MEDIUM,
            user_id="user123",
            action="login",
            resource="system",
            result="success"
        )
        
        assert event.id == "evt-123"
        assert event.event_type == AuditEventType.AUTHENTICATION
        assert event.user_id == "user123"
        assert event.result == "success"
    
    def test_event_with_details(self):
        """Test event with additional details"""
        event = AuditEvent(
            id="evt-456",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.HIGH,
            user_id="user456",
            action="read",
            resource="patient_record_123",
            result="success",
            ip_address="192.168.1.100",
            details={"record_count": 1, "fields_accessed": ["name", "dob"]}
        )
        
        assert event.ip_address == "192.168.1.100"
        assert event.details["record_count"] == 1
        assert len(event.details["fields_accessed"]) == 2


class TestBlockchainBlock:
    """Test blockchain block"""
    
    def test_block_creation(self):
        """Test creating a blockchain block"""
        event = AuditEvent(
            id="evt-1",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.SECURITY_EVENT,
            severity=AuditSeverity.HIGH,
            user_id="user1",
            action="test",
            resource="resource1",
            result="success"
        )
        
        block = BlockchainBlock(
            index=1,
            timestamp=datetime.now(timezone.utc),
            events=[event],
            previous_hash="0000000000000000"
        )
        
        assert block.index == 1
        assert len(block.events) == 1
        assert block.hash is not None
        assert len(block.hash) == 64  # SHA-256 hash
    
    def test_block_hash_calculation(self):
        """Test block hash calculation"""
        event = AuditEvent(
            id="evt-1",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.AUTHENTICATION,
            severity=AuditSeverity.INFO,
            user_id="user1",
            action="login",
            resource="system",
            result="success"
        )
        
        block = BlockchainBlock(
            index=1,
            timestamp=datetime.now(timezone.utc),
            events=[event],
            previous_hash="prev_hash"
        )
        
        calculated_hash = block.calculate_hash()
        assert calculated_hash == block.hash
    
    def test_block_mining(self):
        """Test block mining with proof of work"""
        event = AuditEvent(
            id="evt-1",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.SYSTEM_EVENT,
            severity=AuditSeverity.LOW,
            user_id="system",
            action="startup",
            resource="system",
            result="success"
        )
        
        block = BlockchainBlock(
            index=1,
            timestamp=datetime.now(timezone.utc),
            events=[event],
            previous_hash="0"
        )
        
        block.mine_block(difficulty=2)
        assert block.hash.startswith("00")
        assert block.nonce > 0


class TestTimestampAuthority:
    """Test RFC 3161 timestamp authority"""
    
    def test_initialization(self):
        """Test TSA initialization"""
        tsa = TimestampAuthority()
        assert tsa.enabled is True
        assert tsa.tsa_url is not None
    
    def test_request_timestamp(self):
        """Test requesting timestamp"""
        tsa = TimestampAuthority()
        data = b"test data to timestamp"
        
        timestamp_token = tsa.request_timestamp(data)
        
        assert "timestamp" in timestamp_token
        assert "data_hash" in timestamp_token
        assert "tsa_signature" in timestamp_token
        assert "serial_number" in timestamp_token
    
    def test_verify_timestamp(self):
        """Test verifying timestamp token"""
        tsa = TimestampAuthority()
        
        valid_token = {
            "timestamp": "2025-01-01T00:00:00Z",
            "data_hash": "abc123",
            "tsa_signature": "sig_xyz"
        }
        
        assert tsa.verify_timestamp(valid_token) is True
        
        invalid_token = {"timestamp": "2025-01-01T00:00:00Z"}
        assert tsa.verify_timestamp(invalid_token) is False


class TestDigitalSignature:
    """Test digital signatures"""
    
    def test_sign_data(self):
        """Test signing data"""
        data = b"important audit data"
        signature = DigitalSignature.sign_data(data)
        
        assert signature.algorithm == "RSA-SHA256"
        assert signature.key_id is not None
        assert signature.signature is not None
        assert signature.timestamp is not None
    
    def test_verify_signature(self):
        """Test verifying signature"""
        data = b"test data"
        signature = DigitalSignature.sign_data(data)
        
        assert signature.verify(data) is True


class TestAuditBlockchain:
    """Test audit blockchain"""
    
    def test_initialization(self):
        """Test blockchain initialization"""
        blockchain = AuditBlockchain()
        
        assert len(blockchain.chain) == 1  # Genesis block
        assert blockchain.chain[0].index == 0
        assert blockchain.chain[0].previous_hash == "0"
    
    def test_add_event(self):
        """Test adding event to blockchain"""
        blockchain = AuditBlockchain()
        
        event = AuditEvent(
            id="evt-1",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.AUTHENTICATION,
            severity=AuditSeverity.INFO,
            user_id="user1",
            action="login",
            resource="system",
            result="success"
        )
        
        blockchain.add_event(event)
        assert len(blockchain.pending_events) == 1
    
    def test_create_block(self):
        """Test creating block from pending events"""
        blockchain = AuditBlockchain()
        
        for i in range(5):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.DATA_ACCESS,
                severity=AuditSeverity.MEDIUM,
                user_id=f"user{i}",
                action="read",
                resource=f"resource{i}",
                result="success"
            )
            blockchain.add_event(event)
        
        block = blockchain.create_block()
        
        assert block is not None
        assert len(block.events) == 5
        assert len(blockchain.pending_events) == 0
        assert len(blockchain.chain) == 2  # Genesis + new block
    
    def test_auto_block_creation(self):
        """Test automatic block creation when threshold reached"""
        blockchain = AuditBlockchain()
        blockchain.max_events_per_block = 10
        
        for i in range(15):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.LOW,
                user_id="system",
                action="test",
                resource="test",
                result="success"
            )
            blockchain.add_event(event)
        
        # Should have auto-created one block at 10 events
        assert len(blockchain.chain) == 2
        assert len(blockchain.pending_events) == 5
    
    def test_verify_chain(self):
        """Test blockchain verification"""
        blockchain = AuditBlockchain()
        
        # Add some events and create blocks
        for i in range(10):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                severity=AuditSeverity.HIGH,
                user_id="admin",
                action="update",
                resource="config",
                result="success"
            )
            blockchain.add_event(event)
        
        blockchain.create_block()
        
        assert blockchain.verify_chain() is True
    
    def test_verify_tampered_chain(self):
        """Test detection of tampered blockchain"""
        blockchain = AuditBlockchain()
        
        # Add events and create block
        for i in range(5):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.DATA_MODIFICATION,
                severity=AuditSeverity.CRITICAL,
                user_id="user1",
                action="delete",
                resource="data",
                result="success"
            )
            blockchain.add_event(event)
        
        blockchain.create_block()
        
        # Tamper with block
        blockchain.chain[1].events[0].user_id = "hacker"
        
        assert blockchain.verify_chain() is False
    
    def test_get_block(self):
        """Test retrieving block by index"""
        blockchain = AuditBlockchain()
        
        genesis = blockchain.get_block(0)
        assert genesis is not None
        assert genesis.index == 0
        
        invalid = blockchain.get_block(999)
        assert invalid is None
    
    def test_search_events(self):
        """Test searching events"""
        blockchain = AuditBlockchain()
        
        # Add events with different users
        for i in range(5):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.AUTHENTICATION,
                severity=AuditSeverity.INFO,
                user_id="user1" if i < 3 else "user2",
                action="login",
                resource="system",
                result="success"
            )
            blockchain.add_event(event)
        
        blockchain.create_block()
        
        # Search for user1 events
        user1_events = blockchain.search_events(user_id="user1")
        assert len(user1_events) == 3
    
    def test_search_events_by_type(self):
        """Test searching events by type"""
        blockchain = AuditBlockchain()
        
        for event_type in [AuditEventType.AUTHENTICATION, AuditEventType.DATA_ACCESS]:
            for i in range(3):
                event = AuditEvent(
                    id=f"evt-{event_type.value}-{i}",
                    timestamp=datetime.now(timezone.utc),
                    event_type=event_type,
                    severity=AuditSeverity.INFO,
                    user_id="user1",
                    action="test",
                    resource="test",
                    result="success"
                )
                blockchain.add_event(event)
        
        blockchain.create_block()
        
        auth_events = blockchain.search_events(event_type=AuditEventType.AUTHENTICATION)
        assert len(auth_events) == 3


class TestForensicAnalyzer:
    """Test forensic analyzer"""
    
    def test_initialization(self):
        """Test analyzer initialization"""
        blockchain = AuditBlockchain()
        analyzer = ForensicAnalyzer(blockchain)
        assert analyzer.blockchain is not None
    
    def test_analyze_user_activity(self):
        """Test user activity analysis"""
        blockchain = AuditBlockchain()
        analyzer = ForensicAnalyzer(blockchain)
        
        # Add some user events
        for i in range(10):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.DATA_ACCESS,
                severity=AuditSeverity.MEDIUM,
                user_id="user123",
                action="read",
                resource=f"resource{i}",
                result="success"
            )
            blockchain.add_event(event)
        
        blockchain.create_block()
        
        analysis = analyzer.analyze_user_activity("user123", timeframe_hours=24)
        
        assert analysis["user_id"] == "user123"
        assert analysis["total_events"] == 10
        assert analysis["successful_actions"] == 10
        assert len(analysis["accessed_resources"]) == 10
    
    def test_detect_suspicious_patterns(self):
        """Test detection of suspicious patterns"""
        blockchain = AuditBlockchain()
        analyzer = ForensicAnalyzer(blockchain)
        
        # Add failed authentication attempts
        for i in range(10):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.AUTHENTICATION,
                severity=AuditSeverity.HIGH,
                user_id="user123",
                action="login",
                resource="system",
                result="failure"
            )
            blockchain.add_event(event)
        
        blockchain.create_block()
        
        analysis = analyzer.analyze_user_activity("user123")
        assert len(analysis["suspicious_patterns"]) > 0
        assert any("failed authentication" in p.lower() for p in analysis["suspicious_patterns"])
    
    def test_generate_timeline(self):
        """Test timeline generation"""
        blockchain = AuditBlockchain()
        analyzer = ForensicAnalyzer(blockchain)
        
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        for i in range(5):
            event = AuditEvent(
                id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.INFO,
                user_id="system",
                action="test",
                resource="test",
                result="success"
            )
            blockchain.add_event(event)
        
        blockchain.create_block()
        
        end_time = datetime.now(timezone.utc) + timedelta(hours=1)
        timeline = analyzer.generate_timeline(start_time, end_time)
        
        assert len(timeline) == 5
        assert all("timestamp" in item for item in timeline)
    
    def test_verify_chain_integrity(self):
        """Test chain integrity verification"""
        blockchain = AuditBlockchain()
        analyzer = ForensicAnalyzer(blockchain)
        
        result = analyzer.verify_chain_integrity()
        
        assert result["chain_valid"] is True
        assert result["total_blocks"] >= 1
        assert result["genesis_block_hash"] is not None


class TestChainOfCustodyManager:
    """Test chain of custody manager"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = ChainOfCustodyManager()
        assert len(manager.custody_records) == 0
    
    def test_create_evidence(self):
        """Test creating evidence with custody"""
        manager = ChainOfCustodyManager()
        
        record = manager.create_evidence(
            evidence_id="ev-123",
            description="Security logs",
            collected_by="investigator@example.com",
            source="production_server"
        )
        
        assert record["action"] == "created"
        assert record["custodian"] == "investigator@example.com"
        assert "signature" in record
        
        assert "ev-123" in manager.custody_records
        assert len(manager.custody_records["ev-123"]) == 1
    
    def test_transfer_custody(self):
        """Test custody transfer"""
        manager = ChainOfCustodyManager()
        
        manager.create_evidence(
            "ev-456",
            "Audit logs",
            "analyst1@example.com",
            "server1"
        )
        
        success = manager.transfer_custody(
            "ev-456",
            "analyst1@example.com",
            "analyst2@example.com",
            "Forensic analysis"
        )
        
        assert success is True
        chain = manager.get_custody_chain("ev-456")
        assert len(chain) == 2
        assert chain[1]["action"] == "transferred"
        assert chain[1]["to_custodian"] == "analyst2@example.com"
    
    def test_access_evidence(self):
        """Test evidence access recording"""
        manager = ChainOfCustodyManager()
        
        manager.create_evidence("ev-789", "Config files", "admin", "system")
        
        success = manager.access_evidence(
            "ev-789",
            "auditor@example.com",
            "Compliance review"
        )
        
        assert success is True
        chain = manager.get_custody_chain("ev-789")
        assert len(chain) == 2
        assert chain[1]["action"] == "accessed"
    
    def test_get_custody_chain(self):
        """Test retrieving custody chain"""
        manager = ChainOfCustodyManager()
        
        manager.create_evidence("ev-999", "Data", "user1", "src")
        manager.transfer_custody("ev-999", "user1", "user2", "review")
        manager.access_evidence("ev-999", "user3", "audit")
        
        chain = manager.get_custody_chain("ev-999")
        
        assert len(chain) == 3
        assert chain[0]["action"] == "created"
        assert chain[1]["action"] == "transferred"
        assert chain[2]["action"] == "accessed"
    
    def test_verify_custody_integrity(self):
        """Test custody chain integrity verification"""
        manager = ChainOfCustodyManager()
        
        manager.create_evidence("ev-111", "Test", "user1", "src")
        manager.transfer_custody("ev-111", "user1", "user2", "test")
        
        assert manager.verify_custody_integrity("ev-111") is True
        
        # Test non-existent evidence
        assert manager.verify_custody_integrity("non-existent") is False


class TestEnhancedAuditLogger:
    """Test enhanced audit logger"""
    
    def test_initialization(self):
        """Test logger initialization"""
        logger = EnhancedAuditLogger()
        
        assert logger.blockchain is not None
        assert logger.timestamp_authority is not None
        assert logger.forensic_analyzer is not None
        assert logger.custody_manager is not None
    
    def test_log_event(self):
        """Test logging audit event"""
        logger = EnhancedAuditLogger()
        
        event = logger.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id="user123",
            action="login",
            resource="system",
            result="success",
            severity=AuditSeverity.INFO,
            ip_address="192.168.1.100"
        )
        
        assert event.id is not None
        assert event.user_id == "user123"
        assert "rfc3161_timestamp" in event.metadata
        assert "digital_signature" in event.metadata
    
    def test_log_event_with_details(self):
        """Test logging event with details"""
        logger = EnhancedAuditLogger()
        
        event = logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="user456",
            action="read",
            resource="patient_records",
            result="success",
            details={"record_count": 5, "fields": ["name", "dob"]}
        )
        
        assert event.details["record_count"] == 5
        assert len(event.details["fields"]) == 2
    
    def test_finalize_block(self):
        """Test finalizing audit block"""
        logger = EnhancedAuditLogger()
        
        # Add some events
        for i in range(5):
            logger.log_event(
                AuditEventType.SYSTEM_EVENT,
                f"user{i}",
                "test",
                "test",
                "success"
            )
        
        block = logger.finalize_block()
        
        assert block is not None
        assert len(block.events) == 5
    
    def test_verify_integrity(self):
        """Test verifying audit log integrity"""
        logger = EnhancedAuditLogger()
        
        # Log some events
        for i in range(3):
            logger.log_event(
                AuditEventType.CONFIGURATION_CHANGE,
                "admin",
                "update",
                "config",
                "success"
            )
        
        logger.finalize_block()
        
        assert logger.verify_integrity() is True
    
    def test_search_logs(self):
        """Test searching audit logs"""
        logger = EnhancedAuditLogger()
        
        # Log events for different users
        logger.log_event(
            AuditEventType.AUTHENTICATION,
            "user1",
            "login",
            "system",
            "success"
        )
        logger.log_event(
            AuditEventType.DATA_ACCESS,
            "user1",
            "read",
            "data",
            "success"
        )
        logger.log_event(
            AuditEventType.AUTHENTICATION,
            "user2",
            "login",
            "system",
            "success"
        )
        
        logger.finalize_block()
        
        user1_events = logger.search_logs(user_id="user1")
        assert len(user1_events) == 2
    
    def test_generate_forensic_report(self):
        """Test generating forensic report"""
        logger = EnhancedAuditLogger()
        
        # Log some activity
        for i in range(10):
            logger.log_event(
                AuditEventType.DATA_ACCESS,
                "user123",
                "read",
                f"resource{i}",
                "success"
            )
        
        logger.finalize_block()
        
        report = logger.generate_forensic_report("user123")
        
        assert report["user_id"] == "user123"
        assert report["total_events"] == 10
        assert "event_types" in report
