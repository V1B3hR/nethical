"""
Unit tests for Phase 2.2: SOC Integration
"""

import pytest
import json
from datetime import datetime, timedelta, timezone
from nethical.security.soc_integration import (
    SIEMFormat,
    AlertSeverity,
    IncidentStatus,
    SIEMEvent,
    Incident,
    SIEMConnector,
    IncidentManager,
    ThreatHuntingEngine,
    AlertingEngine,
    ForensicCollector,
    SOCIntegrationHub,
)


class TestSIEMEvent:
    """Test SIEM Event"""
    
    def test_creation(self):
        """Test event creation"""
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.HIGH,
            event_type="security_violation",
            source="nethical",
            description="Test security event",
            agent_id="agent1",
        )
        
        assert event.severity == AlertSeverity.HIGH
        assert event.event_type == "security_violation"
        assert event.agent_id == "agent1"
    
    def test_to_cef(self):
        """Test CEF format conversion"""
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.CRITICAL,
            event_type="apt_detected",
            source="nethical",
            description="APT behavior detected",
            agent_id="agent1",
            resource="server1",
            action="block",
        )
        
        cef = event.to_cef()
        
        assert cef.startswith("CEF:0|Nethical|")
        assert "apt_detected" in cef
        assert "suser=agent1" in cef
        assert "dst=server1" in cef
        assert "act=block" in cef
    
    def test_to_leef(self):
        """Test LEEF format conversion"""
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.HIGH,
            event_type="insider_threat",
            source="nethical",
            description="Insider threat detected",
            agent_id="user1",
        )
        
        leef = event.to_leef()
        
        assert leef.startswith("LEEF:2.0|Nethical|")
        assert "insider_threat" in leef
        assert "usrName=user1" in leef
    
    def test_to_json(self):
        """Test JSON format conversion"""
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.MEDIUM,
            event_type="anomaly",
            source="nethical",
            description="Behavioral anomaly",
        )
        
        json_str = event.to_json()
        data = json.loads(json_str)
        
        assert data["severity"] == "medium"
        assert data["event_type"] == "anomaly"
        assert data["source"] == "nethical"


class TestIncident:
    """Test Incident"""
    
    def test_creation(self):
        """Test incident creation"""
        incident = Incident(
            incident_id="INC-001",
            title="Security Breach",
            description="Unauthorized access detected",
            severity=AlertSeverity.CRITICAL,
            status=IncidentStatus.NEW,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        assert incident.incident_id == "INC-001"
        assert incident.severity == AlertSeverity.CRITICAL
        assert incident.status == IncidentStatus.NEW
    
    def test_to_dict(self):
        """Test incident serialization"""
        incident = Incident(
            incident_id="INC-002",
            title="Test Incident",
            description="Test description",
            severity=AlertSeverity.HIGH,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        data = incident.to_dict()
        
        assert data["incident_id"] == "INC-002"
        assert data["severity"] == "high"
        assert data["status"] == "open"
        assert "created_at" in data


class TestSIEMConnector:
    """Test SIEM Connector"""
    
    def test_initialization(self):
        """Test connector initialization"""
        connector = SIEMConnector(
            siem_endpoint="https://siem.example.com",
            default_format=SIEMFormat.CEF,
        )
        
        assert connector.siem_endpoint == "https://siem.example.com"
        assert connector.default_format == SIEMFormat.CEF
    
    @pytest.mark.asyncio
    async def test_send_event_cef(self):
        """Test sending event in CEF format"""
        connector = SIEMConnector(default_format=SIEMFormat.CEF)
        
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.HIGH,
            event_type="test_event",
            source="test",
            description="Test event",
        )
        
        result = await connector.send_event(event)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_send_event_leef(self):
        """Test sending event in LEEF format"""
        connector = SIEMConnector(default_format=SIEMFormat.LEEF)
        
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.MEDIUM,
            event_type="test_event",
            source="test",
            description="Test event",
        )
        
        result = await connector.send_event(event, format=SIEMFormat.LEEF)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_send_batch(self):
        """Test sending batch of events"""
        connector = SIEMConnector()
        
        events = [
            SIEMEvent(
                timestamp=datetime.now(timezone.utc),
                severity=AlertSeverity.LOW,
                event_type=f"event_{i}",
                source="test",
                description=f"Event {i}",
            )
            for i in range(5)
        ]
        
        count = await connector.send_batch(events)
        
        assert count == 5
    
    def test_buffer_event(self):
        """Test event buffering"""
        connector = SIEMConnector(batch_size=3)
        
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.INFO,
            event_type="test",
            source="test",
            description="Test",
        )
        
        connector.buffer_event(event)
        
        assert len(connector._event_buffer) == 1
    
    @pytest.mark.asyncio
    async def test_flush_buffer(self):
        """Test buffer flushing"""
        connector = SIEMConnector()
        
        # Add events to buffer
        for i in range(3):
            event = SIEMEvent(
                timestamp=datetime.now(timezone.utc),
                severity=AlertSeverity.INFO,
                event_type=f"event_{i}",
                source="test",
                description=f"Event {i}",
            )
            connector.buffer_event(event)
        
        count = await connector.flush_buffer()
        
        assert count == 3
        assert len(connector._event_buffer) == 0


class TestIncidentManager:
    """Test Incident Manager"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = IncidentManager(
            ticketing_api_url="https://tickets.example.com",
            auto_create_threshold=AlertSeverity.HIGH,
        )
        
        assert manager.auto_create_threshold == AlertSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_create_incident(self):
        """Test incident creation"""
        manager = IncidentManager()
        
        incident = await manager.create_incident(
            title="Test Incident",
            description="Test description",
            severity=AlertSeverity.CRITICAL,
        )
        
        assert incident.incident_id.startswith("INC-")
        assert incident.title == "Test Incident"
        assert incident.severity == AlertSeverity.CRITICAL
        assert incident.status == IncidentStatus.NEW
    
    @pytest.mark.asyncio
    async def test_update_incident(self):
        """Test incident update"""
        manager = IncidentManager()
        
        incident = await manager.create_incident(
            title="Test",
            description="Test",
            severity=AlertSeverity.HIGH,
        )
        
        updated = await manager.update_incident(
            incident.incident_id,
            status=IncidentStatus.IN_PROGRESS,
            assigned_to="analyst1",
        )
        
        assert updated is not None
        assert updated.status == IncidentStatus.IN_PROGRESS
        assert updated.assigned_to == "analyst1"
    
    @pytest.mark.asyncio
    async def test_should_create_incident_high_severity(self):
        """Test incident creation threshold"""
        manager = IncidentManager(auto_create_threshold=AlertSeverity.HIGH)
        
        high_event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.HIGH,
            event_type="test",
            source="test",
            description="Test",
        )
        
        should_create = await manager.should_create_incident(high_event)
        assert should_create is True
        
        low_event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.LOW,
            event_type="test",
            source="test",
            description="Test",
        )
        
        should_create = await manager.should_create_incident(low_event)
        assert should_create is False
    
    @pytest.mark.asyncio
    async def test_list_incidents(self):
        """Test incident listing"""
        manager = IncidentManager()
        
        # Create multiple incidents
        await manager.create_incident("Inc 1", "Desc 1", AlertSeverity.HIGH)
        await manager.create_incident("Inc 2", "Desc 2", AlertSeverity.CRITICAL)
        
        incidents = manager.list_incidents()
        
        assert len(incidents) == 2
    
    @pytest.mark.asyncio
    async def test_list_incidents_filtered(self):
        """Test filtered incident listing"""
        manager = IncidentManager()
        
        # Create incidents with different severities
        inc1 = await manager.create_incident("Inc 1", "Desc 1", AlertSeverity.HIGH)
        inc2 = await manager.create_incident("Inc 2", "Desc 2", AlertSeverity.CRITICAL)
        
        critical_incidents = manager.list_incidents(severity=AlertSeverity.CRITICAL)
        
        assert len(critical_incidents) == 1
        assert critical_incidents[0].severity == AlertSeverity.CRITICAL


class TestThreatHuntingEngine:
    """Test Threat Hunting Engine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = ThreatHuntingEngine()
        
        assert len(engine._query_templates) > 0
    
    def test_list_hunt_types(self):
        """Test listing hunt types"""
        engine = ThreatHuntingEngine()
        
        hunt_types = engine.list_hunt_types()
        
        assert len(hunt_types) > 0
        assert any(h["type"] == "lateral_movement" for h in hunt_types)
        assert any(h["type"] == "data_exfiltration" for h in hunt_types)
    
    @pytest.mark.asyncio
    async def test_execute_hunt(self):
        """Test hunt execution"""
        engine = ThreatHuntingEngine()
        
        result = await engine.execute_hunt("lateral_movement")
        
        assert result["hunt_type"] == "lateral_movement"
        assert "findings_count" in result
        assert "findings" in result
    
    @pytest.mark.asyncio
    async def test_execute_hunt_with_filters(self):
        """Test hunt with filters"""
        engine = ThreatHuntingEngine()
        
        time_range = (
            datetime.now(timezone.utc) - timedelta(hours=24),
            datetime.now(timezone.utc),
        )
        
        result = await engine.execute_hunt(
            "privilege_escalation",
            time_range=time_range,
        )
        
        assert result["hunt_type"] == "privilege_escalation"
    
    def test_get_hunt_history(self):
        """Test hunt history retrieval"""
        engine = ThreatHuntingEngine()
        
        history = engine.get_hunt_history()
        
        assert isinstance(history, list)


class TestAlertingEngine:
    """Test Alerting Engine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = AlertingEngine()
        
        assert len(engine._alert_channels) == 0
    
    def test_register_channel(self):
        """Test channel registration"""
        engine = AlertingEngine()
        
        async def mock_handler(alert_data):
            return True
        
        engine.register_channel("email", mock_handler)
        
        assert "email" in engine._alert_channels
    
    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alert"""
        engine = AlertingEngine()
        
        # Register mock channel
        async def mock_handler(alert_data):
            return True
        
        engine.register_channel("test_channel", mock_handler)
        
        results = await engine.send_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.HIGH,
            channels=["test_channel"],
        )
        
        assert results["test_channel"] is True
    
    @pytest.mark.asyncio
    async def test_send_alert_all_channels(self):
        """Test sending to all channels"""
        engine = AlertingEngine()
        
        # Register multiple channels
        async def mock_handler(alert_data):
            return True
        
        engine.register_channel("email", mock_handler)
        engine.register_channel("slack", mock_handler)
        
        results = await engine.send_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.CRITICAL,
        )
        
        assert len(results) == 2
        assert results["email"] is True
        assert results["slack"] is True
    
    def test_get_alert_history(self):
        """Test alert history retrieval"""
        engine = AlertingEngine()
        
        history = engine.get_alert_history()
        
        assert isinstance(history, list)


class TestForensicCollector:
    """Test Forensic Collector"""
    
    def test_initialization(self):
        """Test collector initialization"""
        collector = ForensicCollector(storage_path="/tmp/forensics")
        
        assert collector.storage_path == "/tmp/forensics"
    
    @pytest.mark.asyncio
    async def test_collect_forensics(self):
        """Test forensic collection"""
        collector = ForensicCollector()
        
        collection = await collector.collect_forensics(
            incident_id="INC-001",
            collection_type="logs",
            target="server1",
        )
        
        assert collection["collection_id"].startswith("FOR-")
        assert collection["incident_id"] == "INC-001"
        assert collection["collection_type"] == "logs"
        assert collection["status"] == "collected"
    
    @pytest.mark.asyncio
    async def test_collect_with_metadata(self):
        """Test collection with metadata"""
        collector = ForensicCollector()
        
        metadata = {"source": "server1", "user": "admin"}
        
        collection = await collector.collect_forensics(
            incident_id="INC-002",
            collection_type="memory",
            target="server1",
            metadata=metadata,
        )
        
        assert collection["metadata"]["source"] == "server1"
        assert len(collection["chain_of_custody"]) > 0
    
    def test_get_collection(self):
        """Test collection retrieval"""
        collector = ForensicCollector()
        
        # Non-existent collection
        result = collector.get_collection("FOR-999")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_collections(self):
        """Test collection listing"""
        collector = ForensicCollector()
        
        # Create collections
        await collector.collect_forensics("INC-001", "logs", "server1")
        await collector.collect_forensics("INC-001", "memory", "server1")
        await collector.collect_forensics("INC-002", "logs", "server2")
        
        all_collections = collector.list_collections()
        assert len(all_collections) == 3
        
        # Filter by incident
        inc1_collections = collector.list_collections(incident_id="INC-001")
        assert len(inc1_collections) == 2


class TestSOCIntegrationHub:
    """Test SOC Integration Hub"""
    
    def test_initialization(self):
        """Test hub initialization"""
        hub = SOCIntegrationHub()
        
        assert hub.siem_connector is not None
        assert hub.incident_manager is not None
        assert hub.threat_hunting is not None
        assert hub.alerting is not None
        assert hub.forensics is not None
    
    @pytest.mark.asyncio
    async def test_process_security_event(self):
        """Test event processing workflow"""
        hub = SOCIntegrationHub()
        
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.CRITICAL,
            event_type="apt_detected",
            source="nethical",
            description="APT behavior detected",
        )
        
        result = await hub.process_security_event(event)
        
        assert "actions" in result
        assert "sent_to_siem" in result["actions"]
    
    @pytest.mark.asyncio
    async def test_process_high_severity_event(self):
        """Test processing of high severity event"""
        hub = SOCIntegrationHub()
        
        event = SIEMEvent(
            timestamp=datetime.now(timezone.utc),
            severity=AlertSeverity.CRITICAL,
            event_type="data_breach",
            source="nethical",
            description="Data breach detected",
        )
        
        result = await hub.process_security_event(event, auto_incident=True)
        
        assert "created_incident" in result["actions"]
        assert "incident_id" in result
    
    @pytest.mark.asyncio
    async def test_get_soc_status(self):
        """Test SOC status retrieval"""
        hub = SOCIntegrationHub()
        
        status = await hub.get_soc_status()
        
        assert status["status"] == "operational"
        assert "components" in status
        assert "capabilities" in status
        assert len(status["capabilities"]) > 0


class TestEnums:
    """Test enum types"""
    
    def test_siem_format(self):
        """Test SIEM format enum"""
        assert SIEMFormat.CEF == "cef"
        assert SIEMFormat.LEEF == "leef"
        assert SIEMFormat.JSON == "json"
    
    def test_alert_severity(self):
        """Test alert severity enum"""
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.LOW == "low"
        assert AlertSeverity.MEDIUM == "medium"
        assert AlertSeverity.HIGH == "high"
        assert AlertSeverity.CRITICAL == "critical"
    
    def test_incident_status(self):
        """Test incident status enum"""
        assert IncidentStatus.NEW == "new"
        assert IncidentStatus.OPEN == "open"
        assert IncidentStatus.IN_PROGRESS == "in_progress"
        assert IncidentStatus.RESOLVED == "resolved"
        assert IncidentStatus.CLOSED == "closed"
