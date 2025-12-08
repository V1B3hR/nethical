"""Kill Switch API endpoints.

This module provides REST API endpoints for kill switch management,
including emergency shutdown, actuator severing, and hardware isolation.

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kill-switch", tags=["kill-switch"])


# ========================== Pydantic Models ==========================


class ShutdownRequest(BaseModel):
    """Request model for emergency shutdown."""

    mode: str = Field(
        default="graceful", description="Shutdown mode: immediate, graceful, or staged"
    )
    cohort: Optional[str] = Field(default=None, description="Target cohort (optional)")
    agent_id: Optional[str] = Field(
        default=None, description="Target agent ID (optional)"
    )
    sever_actuators: bool = Field(
        default=True, description="Whether to sever actuator connections"
    )
    isolate_hardware: bool = Field(
        default=False, description="Whether to activate hardware isolation"
    )


class ShutdownResponse(BaseModel):
    """Response model for shutdown operations."""

    success: bool
    operation: str
    activation_time_ms: float
    agents_affected: int = 0
    actuators_severed: int = 0
    errors: List[str] = []
    metadata: Dict[str, Any] = {}


class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration."""

    agent_id: str = Field(..., description="Unique agent identifier")
    cohort: str = Field(..., description="Cohort the agent belongs to")
    priority: int = Field(
        default=0, description="Shutdown priority (higher = more critical)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ActuatorRegistrationRequest(BaseModel):
    """Request model for actuator registration."""

    actuator_id: str = Field(..., description="Unique actuator identifier")
    connection_type: str = Field(
        ..., description="Connection type (network_tcp, serial, etc.)"
    )
    agent_id: str = Field(..., description="ID of the controlling agent")
    safe_state_config: Dict[str, Any] = Field(
        default_factory=dict, description="Safe state configuration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class HardwareIsolationRequest(BaseModel):
    """Request model for hardware isolation."""

    level: str = Field(
        default="network_only",
        description="Isolation level: network_only, full_isolation, or airgap",
    )
    dry_run: bool = Field(
        default=False, description="If True, simulate without making changes"
    )


class SignerRegistrationRequest(BaseModel):
    """Request model for signer registration."""

    signer_id: str = Field(..., description="Unique signer identifier")
    public_key: str = Field(..., description="Base64-encoded public key")


class SignedCommandRequest(BaseModel):
    """Request model for creating and executing signed commands."""

    command_type: str = Field(
        ...,
        description="Command type: kill_all, kill_cohort, kill_agent, sever_actuators, hardware_isolate",
    )
    target: Optional[str] = Field(default=None, description="Target cohort or agent ID")
    ttl_seconds: int = Field(default=300, description="Time-to-live in seconds")
    signatures: List[Dict[str, str]] = Field(
        default_factory=list, description="List of {signer_id, signature} pairs"
    )


class StatusResponse(BaseModel):
    """Response model for status queries."""

    enabled: bool
    kill_switch: Dict[str, Any]
    actuator_severing: Dict[str, Any]
    crypto_commands: Dict[str, Any]
    hardware_isolation: Dict[str, Any]


# ========================== Dependency Injection ==========================


# Singleton instance for the kill switch protocol
_kill_switch_protocol = None


def get_kill_switch_protocol():
    """Get or create the KillSwitchProtocol instance."""
    global _kill_switch_protocol
    if _kill_switch_protocol is None:
        from ..core.kill_switch import KillSwitchProtocol

        _kill_switch_protocol = KillSwitchProtocol()
    return _kill_switch_protocol


# ========================== API Endpoints ==========================


@router.get("/status", response_model=StatusResponse)
async def get_status(protocol=Depends(get_kill_switch_protocol)) -> StatusResponse:
    """Get the current status of the kill switch system."""
    status = protocol.get_status()
    return StatusResponse(**status)


@router.post("/shutdown", response_model=ShutdownResponse)
async def emergency_shutdown(
    request: ShutdownRequest,
    protocol=Depends(get_kill_switch_protocol),
) -> ShutdownResponse:
    """Execute an emergency shutdown.

    This is the primary endpoint for initiating an emergency shutdown.
    Requires appropriate authorization.
    """
    from ..core.kill_switch import ShutdownMode

    try:
        mode = ShutdownMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid shutdown mode: {request.mode}. Must be one of: immediate, graceful, staged",
        )

    result = protocol.emergency_shutdown(
        mode=mode,
        cohort=request.cohort,
        agent_id=request.agent_id,
        sever_actuators=request.sever_actuators,
        isolate_hardware=request.isolate_hardware,
    )

    logger.warning(
        "Emergency shutdown executed: agents=%d, actuators=%d, success=%s",
        result.agents_affected,
        result.actuators_severed,
        result.success,
    )

    return ShutdownResponse(
        success=result.success,
        operation=result.operation,
        activation_time_ms=result.activation_time_ms,
        agents_affected=result.agents_affected,
        actuators_severed=result.actuators_severed,
        errors=result.errors,
        metadata=result.metadata,
    )


@router.post("/reset")
async def reset_kill_switch(
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Reset the kill switch system after activation."""
    success = protocol.reset()
    return {
        "success": success,
        "message": (
            "Kill switch reset successfully" if success else "Kill switch reset failed"
        ),
    }


# ========================== Agent Management ==========================


@router.post("/agents/register")
async def register_agent(
    request: AgentRegistrationRequest,
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Register an agent with the kill switch system."""
    record = protocol.global_kill_switch.register_agent(
        agent_id=request.agent_id,
        cohort=request.cohort,
        priority=request.priority,
        metadata=request.metadata,
    )

    return {
        "success": True,
        "agent_id": record.agent_id,
        "cohort": record.cohort,
        "priority": record.priority,
        "registered_at": record.registered_at.isoformat(),
    }


@router.delete("/agents/{agent_id}")
async def unregister_agent(
    agent_id: str,
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Unregister an agent from the kill switch system."""
    success = protocol.global_kill_switch.unregister_agent(agent_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    return {
        "success": True,
        "agent_id": agent_id,
        "message": "Agent unregistered successfully",
    }


@router.post("/agents/{agent_id}/kill")
async def kill_agent(
    agent_id: str,
    mode: str = "graceful",
    protocol=Depends(get_kill_switch_protocol),
) -> ShutdownResponse:
    """Kill a specific agent."""
    from ..core.kill_switch import ShutdownMode

    try:
        shutdown_mode = ShutdownMode(mode)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid shutdown mode: {mode}",
        )

    result = protocol.global_kill_switch.activate(mode=shutdown_mode, agent_id=agent_id)

    return ShutdownResponse(
        success=result.success,
        operation=result.operation,
        activation_time_ms=result.activation_time_ms,
        agents_affected=result.agents_affected,
        errors=result.errors,
        metadata=result.metadata,
    )


# ========================== Actuator Management ==========================


@router.post("/actuators/register")
async def register_actuator(
    request: ActuatorRegistrationRequest,
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Register an actuator with the kill switch system."""
    from ..core.kill_switch import ConnectionType

    try:
        connection_type = ConnectionType(request.connection_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid connection type: {request.connection_type}",
        )

    record = protocol.actuator_severing.register_actuator(
        actuator_id=request.actuator_id,
        connection_type=connection_type,
        agent_id=request.agent_id,
        safe_state_config=request.safe_state_config,
        metadata=request.metadata,
    )

    return {
        "success": True,
        "actuator_id": record.actuator_id,
        "connection_type": record.connection_type.value,
        "agent_id": record.agent_id,
        "state": record.state.value,
        "connected_at": record.connected_at.isoformat(),
    }


@router.delete("/actuators/{actuator_id}")
async def unregister_actuator(
    actuator_id: str,
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Unregister an actuator from the kill switch system."""
    success = protocol.actuator_severing.unregister_actuator(actuator_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actuator {actuator_id} not found",
        )

    return {
        "success": True,
        "actuator_id": actuator_id,
        "message": "Actuator unregistered successfully",
    }


@router.post("/actuators/{actuator_id}/sever")
async def sever_actuator(
    actuator_id: str,
    actor: str = "api",
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Sever connection to a specific actuator."""
    success, error = protocol.actuator_severing.sever_actuator(actuator_id, actor)

    if not success and error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error,
        )

    return {
        "success": success,
        "actuator_id": actuator_id,
        "message": "Actuator severed successfully",
    }


@router.post("/actuators/sever-all")
async def sever_all_actuators(
    actor: str = "api",
    protocol=Depends(get_kill_switch_protocol),
) -> ShutdownResponse:
    """Sever all actuator connections."""
    result = protocol.actuator_severing.sever_all(actor)

    return ShutdownResponse(
        success=result.success,
        operation=result.operation,
        activation_time_ms=result.activation_time_ms,
        actuators_severed=result.actuators_severed,
        errors=result.errors,
        metadata=result.metadata,
    )


@router.post("/actuators/{actuator_id}/authorize-reconnection")
async def authorize_reconnection(
    actuator_id: str,
    actor: str = "api",
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Authorize reconnection for a previously severed actuator."""
    success = protocol.actuator_severing.authorize_reconnection(actuator_id, actor)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not authorize reconnection for actuator {actuator_id}",
        )

    return {
        "success": True,
        "actuator_id": actuator_id,
        "message": "Reconnection authorized successfully",
    }


# ========================== Hardware Isolation ==========================


@router.post("/hardware/isolate")
async def hardware_isolate(
    request: HardwareIsolationRequest,
    protocol=Depends(get_kill_switch_protocol),
) -> ShutdownResponse:
    """Activate hardware isolation."""
    from ..core.kill_switch import IsolationLevel

    try:
        level = IsolationLevel(request.level)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid isolation level: {request.level}",
        )

    result = protocol.hardware_isolation.isolate(level=level, dry_run=request.dry_run)

    return ShutdownResponse(
        success=result.success,
        operation=result.operation,
        activation_time_ms=result.activation_time_ms,
        errors=result.errors,
        metadata=result.metadata,
    )


@router.post("/hardware/restore")
async def hardware_restore(
    protocol=Depends(get_kill_switch_protocol),
) -> ShutdownResponse:
    """Restore from hardware isolation."""
    result = protocol.hardware_isolation.restore()

    return ShutdownResponse(
        success=result.success,
        operation=result.operation,
        activation_time_ms=result.activation_time_ms,
        errors=result.errors,
        metadata=result.metadata,
    )


@router.get("/hardware/status")
async def hardware_status(protocol=Depends(get_kill_switch_protocol)) -> Dict[str, Any]:
    """Get hardware isolation status."""
    return protocol.hardware_isolation.get_statistics()


# ========================== Crypto Commands ==========================


@router.post("/signers/register")
async def register_signer(
    request: SignerRegistrationRequest,
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Register an authorized signer for multi-signature commands."""
    import base64

    try:
        public_key = base64.b64decode(request.public_key)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64-encoded public key",
        )

    success = protocol.crypto_commands.register_signer(request.signer_id, public_key)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of signers reached or signer already registered",
        )

    return {
        "success": True,
        "signer_id": request.signer_id,
        "message": "Signer registered successfully",
    }


@router.delete("/signers/{signer_id}")
async def unregister_signer(
    signer_id: str,
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Unregister a signer."""
    success = protocol.crypto_commands.unregister_signer(signer_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signer {signer_id} not found",
        )

    return {
        "success": True,
        "signer_id": signer_id,
        "message": "Signer unregistered successfully",
    }


@router.post("/commands/execute")
async def execute_signed_command(
    request: SignedCommandRequest,
    protocol=Depends(get_kill_switch_protocol),
) -> ShutdownResponse:
    """Execute a signed command with multi-signature verification."""
    from ..core.kill_switch import CommandType
    import base64

    try:
        command_type = CommandType(request.command_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid command type: {request.command_type}",
        )

    # Create the command
    command = protocol.crypto_commands.create_command(
        command_type=command_type,
        target=request.target,
        ttl_seconds=request.ttl_seconds,
    )

    # Add signatures
    for sig_entry in request.signatures:
        signer_id = sig_entry.get("signer_id")
        signature_b64 = sig_entry.get("signature")

        if not signer_id or not signature_b64:
            continue

        try:
            signature = base64.b64decode(signature_b64)
        except Exception:
            continue

        protocol.crypto_commands.add_signature(command, signer_id, signature)

    # Execute the command
    result = protocol.crypto_commands.execute_command(
        command,
        protocol.global_kill_switch,
        protocol.actuator_severing,
        protocol.hardware_isolation,
    )

    return ShutdownResponse(
        success=result.success,
        operation=result.operation,
        activation_time_ms=result.activation_time_ms,
        agents_affected=result.agents_affected,
        actuators_severed=result.actuators_severed,
        errors=result.errors,
        metadata=result.metadata,
    )


# ========================== Audit ==========================


@router.get("/audit/log")
async def get_audit_log(
    limit: int = 100,
    protocol=Depends(get_kill_switch_protocol),
) -> Dict[str, Any]:
    """Get the audit log for actuator severing events."""
    log = protocol.actuator_severing.get_audit_log()

    # Return the most recent entries
    recent_entries = log[-limit:] if len(log) > limit else log

    return {
        "total_entries": len(log),
        "returned_entries": len(recent_entries),
        "entries": [
            {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "operation": entry.operation,
                "actor": entry.actor,
                "target": entry.target,
                "success": entry.success,
                "details": entry.details,
                "signed": entry.signature is not None,
            }
            for entry in recent_entries
        ],
    }


# ========================== Module Exports ==========================


__all__ = ["router"]
