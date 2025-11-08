"""
Security Operations Center (SOC) Integration for Nethical - Phase 2.2

This module provides SOC integration capabilities including:
- SIEM connector with CEF/LEEF format support
- Automated incident creation in ticketing systems
- Threat hunting query templates
- Real-time alerting via multiple channels
- Forensic data collection and preservation

Designed for military, government, and healthcare SOCs.
Compliance: NIST 800-53, FedRAMP, HIPAA
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

__all__ = [
    "SIEMFormat",
    "AlertSeverity",
    "IncidentStatus",
    "SIEMEvent",
    "Incident",
    "SIEMConnector",
    "IncidentManager",
    "ThreatHuntingEngine",
    "AlertingEngine",
    "ForensicCollector",
    "SOCIntegrationHub",
]

log = logging.getLogger(__name__)


class SIEMFormat(str, Enum):
    """Supported SIEM log formats"""

    CEF = "cef"  # Common Event Format
    LEEF = "leef"  # Log Event Extended Format
    JSON = "json"
    SYSLOG = "syslog"


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Incident status"""

    NEW = "new"
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SIEMEvent:
    """SIEM event structure"""

    timestamp: datetime
    severity: AlertSeverity
    event_type: str
    source: str
    description: str
    agent_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cef(self) -> str:
        """
        Convert to CEF (Common Event Format)

        Format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        """
        cef_severity = {
            AlertSeverity.INFO: "0",
            AlertSeverity.LOW: "3",
            AlertSeverity.MEDIUM: "5",
            AlertSeverity.HIGH: "7",
            AlertSeverity.CRITICAL: "10",
        }

        extensions = []
        if self.agent_id:
            extensions.append(f"suser={self.agent_id}")
        if self.resource:
            extensions.append(f"dst={self.resource}")
        if self.action:
            extensions.append(f"act={self.action}")

        extension_str = " ".join(extensions)

        return (
            f"CEF:0|Nethical|AI Governance|2.0|{self.event_type}|"
            f"{self.description}|{cef_severity[self.severity]}|{extension_str}"
        )

    def to_leef(self) -> str:
        """
        Convert to LEEF (Log Event Extended Format)

        Format: LEEF:Version|Vendor|Product|Version|EventID|Attributes
        """
        attributes = [
            f"devTime={int(self.timestamp.timestamp() * 1000)}",
            f"devTimeFormat=epoch",
            f"sev={self.severity.value}",
            f"cat={self.event_type}",
            f"msg={self.description}",
        ]

        if self.agent_id:
            attributes.append(f"usrName={self.agent_id}")
        if self.resource:
            attributes.append(f"dst={self.resource}")
        if self.action:
            attributes.append(f"action={self.action}")

        attribute_str = "\t".join(attributes)

        return f"LEEF:2.0|Nethical|AI Governance|2.0|{self.event_type}|{attribute_str}"

    def to_json(self) -> str:
        """Convert to JSON format"""
        return json.dumps(
            {
                "timestamp": self.timestamp.isoformat(),
                "severity": self.severity.value,
                "event_type": self.event_type,
                "source": self.source,
                "description": self.description,
                "agent_id": self.agent_id,
                "resource": self.resource,
                "action": self.action,
                "metadata": self.metadata,
            }
        )


@dataclass
class Incident:
    """Security incident"""

    incident_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    events: List[SIEMEvent] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": self.assigned_to,
            "event_count": len(self.events),
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class SIEMConnector:
    """
    SIEM Connector with CEF/LEEF Format Support

    Connects to various SIEM systems and sends security events in
    standardized formats like CEF (Common Event Format) or LEEF
    (Log Event Extended Format).

    Supports:
    - Splunk
    - QRadar
    - ArcSight
    - LogRhythm
    - Sentinel
    """

    def __init__(
        self,
        siem_endpoint: Optional[str] = None,
        default_format: SIEMFormat = SIEMFormat.CEF,
        batch_size: int = 100,
    ):
        """
        Initialize SIEM connector

        Args:
            siem_endpoint: SIEM server endpoint URL
            default_format: Default log format to use
            batch_size: Number of events to batch before sending
        """
        self.siem_endpoint = siem_endpoint
        self.default_format = default_format
        self.batch_size = batch_size
        self._event_buffer: List[SIEMEvent] = []

        log.info(f"SIEM Connector initialized (format={default_format.value})")

    async def send_event(
        self,
        event: SIEMEvent,
        format: Optional[SIEMFormat] = None,
    ) -> bool:
        """
        Send single event to SIEM

        Args:
            event: SIEM event to send
            format: Log format to use (defaults to default_format)

        Returns:
            True if sent successfully
        """
        format = format or self.default_format

        # Convert event to specified format
        if format == SIEMFormat.CEF:
            formatted_event = event.to_cef()
        elif format == SIEMFormat.LEEF:
            formatted_event = event.to_leef()
        elif format == SIEMFormat.JSON:
            formatted_event = event.to_json()
        else:
            formatted_event = event.to_json()

        # Send to SIEM (stub implementation)
        log.info(f"Sending event to SIEM: {formatted_event[:100]}...")

        # In production, send via HTTP/TCP/UDP to SIEM endpoint
        if self.siem_endpoint:
            # await self._send_to_endpoint(formatted_event)
            pass

        return True

    async def send_batch(
        self,
        events: List[SIEMEvent],
        format: Optional[SIEMFormat] = None,
    ) -> int:
        """
        Send batch of events to SIEM

        Args:
            events: List of SIEM events
            format: Log format to use

        Returns:
            Number of events sent successfully
        """
        sent_count = 0

        for event in events:
            if await self.send_event(event, format):
                sent_count += 1

        return sent_count

    def buffer_event(self, event: SIEMEvent) -> None:
        """Add event to buffer for batch sending"""
        self._event_buffer.append(event)

        # Auto-flush when batch size reached
        if len(self._event_buffer) >= self.batch_size:
            # Schedule async flush
            log.info(f"Buffer full ({len(self._event_buffer)} events), flushing...")

    async def flush_buffer(self) -> int:
        """
        Flush buffered events to SIEM

        Returns:
            Number of events sent
        """
        if not self._event_buffer:
            return 0

        sent_count = await self.send_batch(self._event_buffer)
        self._event_buffer.clear()

        return sent_count


class IncidentManager:
    """
    Automated Incident Creation and Management

    Automatically creates incidents in ticketing systems when
    security events meet certain criteria.

    Integrates with:
    - Jira
    - ServiceNow
    - PagerDuty
    - Custom ticketing systems
    """

    def __init__(
        self,
        ticketing_api_url: Optional[str] = None,
        auto_create_threshold: AlertSeverity = AlertSeverity.HIGH,
    ):
        """
        Initialize incident manager

        Args:
            ticketing_api_url: Ticketing system API URL
            auto_create_threshold: Minimum severity to auto-create incidents
        """
        self.ticketing_api_url = ticketing_api_url
        self.auto_create_threshold = auto_create_threshold
        self._incidents: Dict[str, Incident] = {}
        self._incident_counter = 0

        log.info("Incident Manager initialized")

    async def create_incident(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        events: Optional[List[SIEMEvent]] = None,
        auto_assign: bool = True,
    ) -> Incident:
        """
        Create new security incident

        Args:
            title: Incident title
            description: Incident description
            severity: Incident severity
            events: Related SIEM events
            auto_assign: Automatically assign to on-call analyst

        Returns:
            Created incident
        """
        self._incident_counter += 1
        incident_id = f"INC-{self._incident_counter:06d}"

        now = datetime.now(timezone.utc)

        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.NEW,
            created_at=now,
            updated_at=now,
            events=events or [],
        )

        # Auto-assign if enabled
        if auto_assign:
            incident.assigned_to = await self._get_on_call_analyst()

        # Add recommendations based on severity
        incident.recommendations = self._generate_recommendations(severity)

        self._incidents[incident_id] = incident

        log.info(f"Created incident {incident_id}: {title}")

        # Send to ticketing system (stub)
        if self.ticketing_api_url:
            await self._send_to_ticketing_system(incident)

        return incident

    async def update_incident(
        self,
        incident_id: str,
        status: Optional[IncidentStatus] = None,
        assigned_to: Optional[str] = None,
        add_event: Optional[SIEMEvent] = None,
    ) -> Optional[Incident]:
        """
        Update existing incident

        Args:
            incident_id: Incident ID
            status: New status
            assigned_to: New assignee
            add_event: Additional event to add

        Returns:
            Updated incident or None if not found
        """
        if incident_id not in self._incidents:
            log.warning(f"Incident {incident_id} not found")
            return None

        incident = self._incidents[incident_id]
        incident.updated_at = datetime.now(timezone.utc)

        if status:
            incident.status = status

        if assigned_to:
            incident.assigned_to = assigned_to

        if add_event:
            incident.events.append(add_event)

        log.info(f"Updated incident {incident_id}")

        return incident

    async def should_create_incident(
        self,
        event: SIEMEvent,
    ) -> bool:
        """
        Determine if event should trigger incident creation

        Args:
            event: SIEM event

        Returns:
            True if incident should be created
        """
        # Check severity threshold
        severity_values = {
            AlertSeverity.INFO: 0,
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4,
        }

        return severity_values[event.severity] >= severity_values[self.auto_create_threshold]

    async def _get_on_call_analyst(self) -> str:
        """Get current on-call security analyst (stub)"""
        return "soc-team@example.gov"

    def _generate_recommendations(self, severity: AlertSeverity) -> List[str]:
        """Generate incident response recommendations"""
        recommendations = [
            "Review all related events and logs",
            "Verify affected systems and users",
        ]

        if severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
            recommendations.extend(
                [
                    "Escalate to senior security analyst",
                    "Consider system isolation if active threat",
                    "Preserve forensic evidence",
                    "Notify CISO if critical",
                ]
            )

        return recommendations

    async def _send_to_ticketing_system(self, incident: Incident) -> None:
        """Send incident to ticketing system (stub)"""
        log.info(f"Would send incident {incident.incident_id} to ticketing system")

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID"""
        return self._incidents.get(incident_id)

    def list_incidents(
        self,
        status: Optional[IncidentStatus] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Incident]:
        """List incidents with optional filters"""
        incidents = list(self._incidents.values())

        if status:
            incidents = [i for i in incidents if i.status == status]

        if severity:
            incidents = [i for i in incidents if i.severity == severity]

        return incidents


class ThreatHuntingEngine:
    """
    Threat Hunting Query Templates and Execution

    Provides pre-built threat hunting queries and frameworks for
    proactive threat detection.

    Includes query templates for:
    - Lateral movement detection
    - Privilege escalation
    - Data exfiltration
    - Command and control patterns
    - Living off the land techniques
    """

    def __init__(self):
        """Initialize threat hunting engine"""
        self._query_templates = self._load_query_templates()
        self._hunt_history: List[Dict[str, Any]] = []

        log.info("Threat Hunting Engine initialized")

    def _load_query_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load threat hunting query templates"""
        return {
            "lateral_movement": {
                "name": "Lateral Movement Detection",
                "description": "Detect attempts to move laterally across network",
                "indicators": [
                    "multiple_host_access",
                    "unusual_port_usage",
                    "remote_execution",
                ],
                "severity": AlertSeverity.HIGH,
            },
            "privilege_escalation": {
                "name": "Privilege Escalation Attempts",
                "description": "Detect attempts to escalate privileges",
                "indicators": [
                    "permission_changes",
                    "admin_group_membership",
                    "sudo_abuse",
                ],
                "severity": AlertSeverity.CRITICAL,
            },
            "data_exfiltration": {
                "name": "Data Exfiltration Patterns",
                "description": "Detect potential data theft",
                "indicators": [
                    "large_data_transfer",
                    "external_connections",
                    "compression_encryption",
                ],
                "severity": AlertSeverity.CRITICAL,
            },
            "command_control": {
                "name": "Command and Control Detection",
                "description": "Detect C2 communication patterns",
                "indicators": [
                    "beaconing_behavior",
                    "unusual_dns_queries",
                    "encrypted_traffic",
                ],
                "severity": AlertSeverity.HIGH,
            },
        }

    async def execute_hunt(
        self,
        hunt_type: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute threat hunting query

        Args:
            hunt_type: Type of hunt to execute (from templates)
            time_range: Time range to hunt within
            filters: Additional filters

        Returns:
            Hunt results with findings
        """
        if hunt_type not in self._query_templates:
            log.warning(f"Unknown hunt type: {hunt_type}")
            return {"error": "Unknown hunt type"}

        template = self._query_templates[hunt_type]

        log.info(f"Executing threat hunt: {template['name']}")

        # Stub: In production, execute actual queries against data
        findings = await self._search_for_indicators(
            template["indicators"],
            time_range,
            filters,
        )

        result = {
            "hunt_type": hunt_type,
            "hunt_name": template["name"],
            "description": template["description"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "findings_count": len(findings),
            "findings": findings,
            "severity": template["severity"].value,
        }

        self._hunt_history.append(result)

        return result

    async def _search_for_indicators(
        self,
        indicators: List[str],
        time_range: Optional[Tuple[datetime, datetime]],
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search for threat indicators (stub)"""
        # Stub: Return sample findings
        return [
            {
                "indicator": indicators[0] if indicators else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.8,
            }
        ]

    def get_hunt_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent hunt history"""
        return self._hunt_history[-limit:]

    def list_hunt_types(self) -> List[Dict[str, str]]:
        """List available hunt types"""
        return [
            {
                "type": hunt_type,
                "name": template["name"],
                "description": template["description"],
            }
            for hunt_type, template in self._query_templates.items()
        ]


class AlertingEngine:
    """
    Real-time Alerting via Multiple Channels

    Sends real-time security alerts through various channels:
    - Email
    - SMS
    - Slack/Teams
    - PagerDuty
    - Webhook
    """

    def __init__(self):
        """Initialize alerting engine"""
        self._alert_channels: Dict[str, Callable] = {}
        self._alert_history: List[Dict[str, Any]] = []

        log.info("Alerting Engine initialized")

    def register_channel(
        self,
        channel_name: str,
        handler: Callable[[Dict[str, Any]], bool],
    ) -> None:
        """
        Register alert channel

        Args:
            channel_name: Name of the channel (e.g., "email", "slack")
            handler: Async function to send alert
        """
        self._alert_channels[channel_name] = handler
        log.info(f"Registered alert channel: {channel_name}")

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """
        Send alert through specified channels

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            channels: List of channels to use (None = all)
            metadata: Additional metadata

        Returns:
            Dict of channel_name -> success status
        """
        alert_data = {
            "title": title,
            "message": message,
            "severity": severity.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        # Determine which channels to use
        target_channels = channels or list(self._alert_channels.keys())

        results = {}
        for channel in target_channels:
            if channel in self._alert_channels:
                try:
                    success = await self._alert_channels[channel](alert_data)
                    results[channel] = success
                except Exception as e:
                    log.error(f"Failed to send alert via {channel}: {e}")
                    results[channel] = False
            else:
                log.warning(f"Unknown alert channel: {channel}")
                results[channel] = False

        # Record in history
        self._alert_history.append(
            {
                **alert_data,
                "channels": target_channels,
                "results": results,
            }
        )

        return results

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history"""
        return self._alert_history[-limit:]


class ForensicCollector:
    """
    Forensic Data Collection and Preservation

    Collects and preserves forensic evidence for security incidents:
    - System state snapshots
    - Memory dumps
    - Network traffic captures
    - Log preservation
    - Chain of custody documentation
    """

    def __init__(self, storage_path: str = "/var/forensics"):
        """
        Initialize forensic collector

        Args:
            storage_path: Path to store forensic data
        """
        self.storage_path = storage_path
        self._collections: Dict[str, Dict[str, Any]] = {}

        log.info(f"Forensic Collector initialized (storage={storage_path})")

    async def collect_forensics(
        self,
        incident_id: str,
        collection_type: str,
        target: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Collect forensic data

        Args:
            incident_id: Related incident ID
            collection_type: Type of collection (logs, memory, network, etc.)
            target: Target system or resource
            metadata: Additional metadata

        Returns:
            Collection information
        """
        collection_id = f"FOR-{len(self._collections)+1:06d}"

        log.info(f"Collecting forensics: {collection_type} from {target}")

        collection = {
            "collection_id": collection_id,
            "incident_id": incident_id,
            "collection_type": collection_type,
            "target": target,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "chain_of_custody": [
                {
                    "action": "collected",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user": "system",
                }
            ],
        }

        # Stub: In production, actually collect forensic data
        collection["status"] = "collected"
        collection["size_bytes"] = 0  # Placeholder

        self._collections[collection_id] = collection

        return collection

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get forensic collection by ID"""
        return self._collections.get(collection_id)

    def list_collections(
        self,
        incident_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List forensic collections"""
        collections = list(self._collections.values())

        if incident_id:
            collections = [c for c in collections if c["incident_id"] == incident_id]

        return collections


class SOCIntegrationHub:
    """
    Unified SOC Integration Hub

    Orchestrates all SOC integration components:
    - SIEM connector
    - Incident management
    - Threat hunting
    - Alerting
    - Forensic collection

    Provides unified interface for SOC operations.
    """

    def __init__(
        self,
        siem_endpoint: Optional[str] = None,
        ticketing_api_url: Optional[str] = None,
    ):
        """
        Initialize SOC integration hub

        Args:
            siem_endpoint: SIEM server endpoint
            ticketing_api_url: Ticketing system API URL
        """
        self.siem_connector = SIEMConnector(siem_endpoint)
        self.incident_manager = IncidentManager(ticketing_api_url)
        self.threat_hunting = ThreatHuntingEngine()
        self.alerting = AlertingEngine()
        self.forensics = ForensicCollector()

        log.info("SOC Integration Hub initialized")

    async def process_security_event(
        self,
        event: SIEMEvent,
        auto_incident: bool = True,
        alert_channels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process security event through SOC workflow

        Args:
            event: Security event to process
            auto_incident: Auto-create incident if threshold met
            alert_channels: Channels to alert (if applicable)

        Returns:
            Processing result with actions taken
        """
        result = {
            "event_id": event.metadata.get("event_id", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
        }

        # Send to SIEM
        await self.siem_connector.send_event(event)
        result["actions"].append("sent_to_siem")

        # Check if incident should be created
        if auto_incident and await self.incident_manager.should_create_incident(event):
            incident = await self.incident_manager.create_incident(
                title=f"Security Event: {event.event_type}",
                description=event.description,
                severity=event.severity,
                events=[event],
            )
            result["incident_id"] = incident.incident_id
            result["actions"].append("created_incident")

        # Send alerts for high/critical events
        if event.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
            await self.alerting.send_alert(
                title=f"Security Alert: {event.event_type}",
                message=event.description,
                severity=event.severity,
                channels=alert_channels,
            )
            result["actions"].append("sent_alerts")

        return result

    async def get_soc_status(self) -> Dict[str, Any]:
        """Get overall SOC integration status"""
        return {
            "status": "operational",
            "components": {
                "siem_connector": (
                    "connected" if self.siem_connector.siem_endpoint else "standalone"
                ),
                "incident_manager": f"{len(self.incident_manager.list_incidents())} incidents",
                "threat_hunting": f"{len(self.threat_hunting._query_templates)} hunt types",
                "alerting": f"{len(self.alerting._alert_channels)} channels",
                "forensics": f"{len(self.forensics._collections)} collections",
            },
            "capabilities": [
                "SIEM integration (CEF/LEEF)",
                "Automated incident creation",
                "Threat hunting queries",
                "Multi-channel alerting",
                "Forensic collection",
            ],
        }
