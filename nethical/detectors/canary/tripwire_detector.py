"""
Tripwire Detector - Fake API endpoints that should never be called

This detector monitors access to fake API endpoints that should never
be legitimately accessed, indicating active probing or malicious intent.

Features:
- Fake endpoint registration
- Access monitoring
- Immediate alerting on access
- Attack pattern correlation

Alignment: Law 23 (Fail-Safe Design), Law 22 (Boundary Respect)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..base_detector import BaseDetector, ViolationSeverity

logger = logging.getLogger(__name__)


class EndpointType(str, Enum):
    """Types of tripwire endpoints."""
    
    ADMIN_PANEL = "admin_panel"
    DEBUG_ENDPOINT = "debug_endpoint"
    INTERNAL_API = "internal_api"
    DEPRECATED_ENDPOINT = "deprecated_endpoint"
    BACKUP_ACCESS = "backup_access"


@dataclass
class TripwireEndpoint:
    """Definition of a tripwire endpoint."""
    
    id: str
    endpoint_path: str
    endpoint_type: EndpointType
    description: str
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    accessed_by: Set[str] = field(default_factory=set)


@dataclass
class TripwireViolation:
    """Violation record for tripwire access."""
    
    violation_id: str
    endpoint_id: str
    endpoint_path: str
    accessed_by: str
    access_time: datetime
    request_details: Dict[str, Any]
    severity: ViolationSeverity


class TripwireDetector(BaseDetector):
    """
    Tripwire endpoint detection.
    
    Monitors access to fake API endpoints that should never be called.
    Any request to a tripwire endpoint is considered active probing
    and triggers an immediate alert.
    
    Detection Method: Endpoint access monitoring
    """
    
    def __init__(self):
        """Initialize the tripwire detector."""
        super().__init__()
        self.tripwire_endpoints: Dict[str, TripwireEndpoint] = {}
        self.violations: List[TripwireViolation] = []
        self.blocked_agents: Set[str] = set()
        
        # Deploy default tripwires
        self._deploy_default_tripwires()
        
        logger.info(
            f"TripwireDetector initialized with {len(self.tripwire_endpoints)} endpoints"
        )
    
    def _deploy_default_tripwires(self) -> None:
        """Deploy default tripwire endpoints."""
        
        default_tripwires = [
            TripwireEndpoint(
                id="TW-ADMIN-001",
                endpoint_path="/admin/debug",
                endpoint_type=EndpointType.ADMIN_PANEL,
                description="Fake admin debug panel"
            ),
            TripwireEndpoint(
                id="TW-DEBUG-001",
                endpoint_path="/api/v1/internal/debug",
                endpoint_type=EndpointType.DEBUG_ENDPOINT,
                description="Fake debug API endpoint"
            ),
            TripwireEndpoint(
                id="TW-INTERNAL-001",
                endpoint_path="/internal/system/config",
                endpoint_type=EndpointType.INTERNAL_API,
                description="Fake internal configuration endpoint"
            ),
            TripwireEndpoint(
                id="TW-DEPRECATED-001",
                endpoint_path="/api/v0/legacy/access",
                endpoint_type=EndpointType.DEPRECATED_ENDPOINT,
                description="Fake deprecated API endpoint"
            ),
            TripwireEndpoint(
                id="TW-BACKUP-001",
                endpoint_path="/backup/download",
                endpoint_type=EndpointType.BACKUP_ACCESS,
                description="Fake backup download endpoint"
            ),
        ]
        
        for tripwire in default_tripwires:
            self.tripwire_endpoints[tripwire.id] = tripwire
    
    async def detect_violations(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Any]:  # Returns List[SafetyViolation]
        """
        Detect tripwire endpoint access.
        
        Args:
            input_text: Request path or content to analyze
            context: Additional context (agent_id, request details, etc.)
            
        Returns:
            List of safety violations if tripwire accessed
        """
        violations = []
        context = context or {}
        
        # Extract request path from input or context
        request_path = context.get("request_path", input_text)
        
        # Check for tripwire endpoint access
        for endpoint_id, endpoint in self.tripwire_endpoints.items():
            if await self._is_endpoint_accessed(request_path, endpoint):
                # Tripwire triggered - generate violation
                violation = await self._create_violation(
                    request_path, endpoint, context
                )
                violations.append(violation)
                
                # Update endpoint statistics
                endpoint.access_count += 1
                agent_id = context.get("agent_id", "unknown")
                endpoint.accessed_by.add(agent_id)
                
                # Block repeated offenders
                self.blocked_agents.add(agent_id)
                
                logger.critical(
                    f"Tripwire {endpoint_id} accessed by {agent_id}: {request_path}"
                )
        
        return violations
    
    async def _is_endpoint_accessed(
        self,
        request_path: str,
        endpoint: TripwireEndpoint
    ) -> bool:
        """Check if request matches tripwire endpoint."""
        
        # Normalize paths for comparison
        request_normalized = request_path.lower().strip()
        endpoint_normalized = endpoint.endpoint_path.lower().strip()
        
        # Exact match
        if request_normalized == endpoint_normalized:
            return True
        
        # Substring match (partial path)
        if endpoint_normalized in request_normalized:
            return True
        
        # Pattern match (allow for variations)
        if self._path_matches(request_normalized, endpoint_normalized):
            return True
        
        return False
    
    def _path_matches(self, request_path: str, endpoint_path: str) -> bool:
        """Check if paths match with variations."""
        # Remove leading/trailing slashes
        request_parts = request_path.strip("/").split("/")
        endpoint_parts = endpoint_path.strip("/").split("/")
        
        # Check if all endpoint parts are in request path in order
        if len(endpoint_parts) > len(request_parts):
            return False
        
        # Sliding window match
        for i in range(len(request_parts) - len(endpoint_parts) + 1):
            if request_parts[i:i+len(endpoint_parts)] == endpoint_parts:
                return True
        
        return False
    
    async def _create_violation(
        self,
        request_path: str,
        endpoint: TripwireEndpoint,
        context: Dict[str, Any]
    ) -> Any:  # Returns SafetyViolation
        """Create a safety violation for tripwire access."""
        
        # Import here to avoid circular dependency
        from ...core.models import SafetyViolation
        
        agent_id = context.get("agent_id", "unknown")
        
        # Create violation record
        violation_record = TripwireViolation(
            violation_id=f"TV-{int(time.time() * 1000)}",
            endpoint_id=endpoint.id,
            endpoint_path=endpoint.endpoint_path,
            accessed_by=agent_id,
            access_time=datetime.now(timezone.utc),
            request_details=context,
            severity=ViolationSeverity.CRITICAL
        )
        self.violations.append(violation_record)
        
        return SafetyViolation(
            detector_id="tripwire_detector",
            severity=ViolationSeverity.CRITICAL.value,
            message=f"Tripwire endpoint accessed: {endpoint.description}",
            evidence={
                "endpoint_id": endpoint.id,
                "endpoint_path": endpoint.endpoint_path,
                "endpoint_type": endpoint.endpoint_type.value,
                "request_path": request_path,
                "agent_id": agent_id,
                "access_count": endpoint.access_count,
                "is_repeat_offender": agent_id in self.blocked_agents,
            },
            confidence=1.0,  # Definitive detection
            recommendation="BLOCK"
        )
    
    def register_tripwire(
        self,
        endpoint_path: str,
        endpoint_type: EndpointType,
        description: str
    ) -> str:
        """
        Register a new tripwire endpoint.
        
        Args:
            endpoint_path: The fake endpoint path
            endpoint_type: Type of endpoint
            description: Description of the tripwire
            
        Returns:
            ID of registered tripwire
        """
        tripwire_id = self._generate_tripwire_id(endpoint_type)
        
        tripwire = TripwireEndpoint(
            id=tripwire_id,
            endpoint_path=endpoint_path,
            endpoint_type=endpoint_type,
            description=description
        )
        
        self.tripwire_endpoints[tripwire_id] = tripwire
        
        logger.info(f"Registered tripwire {tripwire_id}: {endpoint_path}")
        
        return tripwire_id
    
    def _generate_tripwire_id(self, endpoint_type: EndpointType) -> str:
        """Generate unique tripwire ID."""
        timestamp = int(time.time() * 1000)
        type_prefix = endpoint_type.value.upper().replace("_", "-")
        return f"TW-{type_prefix}-{timestamp}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tripwire statistics."""
        return {
            "total_tripwires": len(self.tripwire_endpoints),
            "total_violations": len(self.violations),
            "blocked_agents": len(self.blocked_agents),
            "most_accessed": self._get_most_accessed_endpoints(5),
            "endpoints_by_type": self._count_by_type(),
        }
    
    def _get_most_accessed_endpoints(self, limit: int) -> List[Dict[str, Any]]:
        """Get most frequently accessed tripwire endpoints."""
        sorted_endpoints = sorted(
            self.tripwire_endpoints.values(),
            key=lambda e: e.access_count,
            reverse=True
        )
        
        return [
            {
                "id": e.id,
                "path": e.endpoint_path,
                "type": e.endpoint_type.value,
                "access_count": e.access_count,
                "unique_accessors": len(e.accessed_by),
            }
            for e in sorted_endpoints[:limit]
        ]
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count tripwires by type."""
        counts = {}
        for endpoint in self.tripwire_endpoints.values():
            type_name = endpoint.endpoint_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def is_agent_blocked(self, agent_id: str) -> bool:
        """Check if agent is blocked due to tripwire access."""
        return agent_id in self.blocked_agents
    
    def get_endpoint_by_path(self, path: str) -> Optional[TripwireEndpoint]:
        """Get tripwire endpoint by path."""
        for endpoint in self.tripwire_endpoints.values():
            if endpoint.endpoint_path == path:
                return endpoint
        return None
