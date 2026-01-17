"""Real-time threat notification routes for API v1.

Provides WebSocket and SSE endpoints for real-time threat notifications.

Endpoints:
- WS /api/v1/ws/threats - WebSocket endpoint for real-time threats
- GET /api/v1/sse/threats - Server-Sent Events endpoint for real-time threats
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


router = APIRouter(tags=["Real-time Threats"])


class ThreatEvent(BaseModel):
    """Threat event model."""
    
    event_type: str  # threat_detected, action_blocked, kill_switch_alarm
    timestamp: str
    agent_id: str
    threat_type: str  # prompt_injection, deepfake, etc.
    severity: str  # low, medium, high, critical
    action_taken: str  # blocked, allowed, restricted
    details: dict


# Global connection manager for WebSocket connections
class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[tuple[WebSocket, Optional[str], Optional[str]]] = []
    
    async def connect(self, websocket: WebSocket, agent_id: Optional[str] = None, threat_type: Optional[str] = None):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections.append((websocket, agent_id, threat_type))
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections = [
            conn for conn in self.active_connections if conn[0] != websocket
        ]
    
    async def broadcast(self, event: ThreatEvent):
        """Broadcast threat event to all matching connections."""
        disconnected = []
        for websocket, agent_filter, threat_filter in self.active_connections:
            # Check filters
            if agent_filter and event.agent_id != agent_filter:
                continue
            if threat_filter and event.threat_type != threat_filter:
                continue
            
            try:
                await websocket.send_json(event.model_dump())
            except Exception:
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)


manager = ConnectionManager()


@router.websocket("/ws/threats")
async def websocket_threats(
    websocket: WebSocket,
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    threat_type: Optional[str] = Query(None, description="Filter by threat type"),
):
    """WebSocket endpoint for real-time threat notifications.
    
    **Authentication:** Pass JWT token as `token` query parameter
    
    Args:
        websocket: WebSocket connection
        agent_id: Optional filter for specific agent
        threat_type: Optional filter for specific threat type
        
    Query Parameters:
        - token: JWT authentication token (required)
        - agent_id: Filter events for specific agent
        - threat_type: Filter events for specific threat type
        
    Message Format:
        ```json
        {
            "event_type": "threat_detected",
            "timestamp": "2026-01-09T12:34:56Z",
            "agent_id": "agent-123",
            "threat_type": "prompt_injection",
            "severity": "high",
            "action_taken": "blocked",
            "details": {...}
        }
        ```
    """
    # TODO: Implement WebSocket authentication
    # For now, accept all connections (should validate token in production)
    
    await manager.connect(websocket, agent_id, threat_type)
    
    try:
        # Keep connection alive and listen for messages
        while True:
            # Receive message (client can send ping or filters update)
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection keep-alive
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.get("/sse/threats")
async def sse_threats(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    threat_type: Optional[str] = Query(None, description="Filter by threat type"),
):
    """Server-Sent Events endpoint for real-time threat notifications.
    
    **Note:** SSE is simpler than WebSocket but less flexible.
    Consider using WebSocket for production.
    
    Args:
        agent_id: Optional filter for specific agent
        threat_type: Optional filter for specific threat type
        
    Returns:
        SSE stream
        
    Event Format:
        ```
        event: threat
        data: {"event_type":"threat_detected","timestamp":"2026-01-09T12:34:56Z",...}
        
        ```
    """
    async def event_generator():
        """Generate SSE events."""
        # TODO: Implement actual event subscription
        # For now, send heartbeat every 30 seconds
        
        while True:
            # Send heartbeat
            yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
            
            await asyncio.sleep(30)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# Helper function to broadcast threat events (to be called from other modules)
async def broadcast_threat_event(
    event_type: str,
    agent_id: str,
    threat_type: str,
    severity: str,
    action_taken: str,
    details: dict
):
    """Broadcast threat event to all connected clients.
    
    Args:
        event_type: Event type (threat_detected, action_blocked, kill_switch_alarm)
        agent_id: Agent identifier
        threat_type: Threat type (prompt_injection, deepfake, etc.)
        severity: Severity level (low, medium, high, critical)
        action_taken: Action taken (blocked, allowed, restricted)
        details: Additional event details
    """
    event = ThreatEvent(
        event_type=event_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_id=agent_id,
        threat_type=threat_type,
        severity=severity,
        action_taken=action_taken,
        details=details
    )
    
    await manager.broadcast(event)
