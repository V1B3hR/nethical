# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Spacecraft Avionics Bus & Telecommand Protocol Security (nethical.space.bus_security).

Guards onboard spacecraft serial, multiplex, and network buses:
- SpaceWire (ECSS-E-ST-50-12C)
- MIL-STD-1553B Dual-Redundant Avionics Bus
- CCSDS Telecommand (TC) Packets (CCSDS 133.0-B-2)
- CANaerospace / SpaceCAN
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.space.bus_security")


class SpaceBusType(str, Enum):
    """Spacecraft avionics communication bus architectures."""
    SPACEWIRE = "SPACEWIRE"
    MIL_STD_1553B = "MIL_STD_1553B"
    CCSDS_TELECOMMAND = "CCSDS_TELECOMMAND"
    CAN_AEROSPACE = "CAN_AEROSPACE"


class BusSecurityAlert(BaseModel):
    """Result of space avionics bus packet verification."""
    allowed: bool
    bus_type: SpaceBusType
    command_opcode: int
    attack_detected: bool
    threat_description: str
    reasons: List[str] = Field(default_factory=list)


class SpacecraftBusGuard:
    """Monitors spacecraft data buses to detect and veto malicious commands or bus injection."""

    def __init__(
        self,
        shared_hmac_key: bytes = b"NETHICAL_SOVEREIGN_SPACE_HMAC_KEY_2026",
        authorized_rt_addresses: Optional[List[int]] = None,
    ) -> None:
        self.shared_key = shared_hmac_key
        # MIL-STD-1553 Remote Terminal (RT) addresses 1-30; 0 is bus controller, 31 is broadcast
        self.authorized_rt_addresses = authorized_rt_addresses or [1, 2, 3, 4, 5]
        # Critical kinetic or payload opcodes requiring high privilege
        self.kinetic_opcodes = {0xAA01: "THRUSTER_FIRE_ORBITAL_MANEUVER", 0xAA02: "DEORBIT_BURN_ACTIVATE"}

    def audit_ccsds_telecommand(
        self,
        command_bytes: bytes,
        signature_hex: Optional[str],
        command_opcode: int,
    ) -> BusSecurityAlert:
        """Verify cryptographic authenticity of incoming ground or inter-satellite telecommands."""
        # Calculate expected HMAC signature over payload
        expected_sig = hmac.new(self.shared_key, command_bytes, hashlib.sha256).hexdigest()

        is_kinetic = command_opcode in self.kinetic_opcodes
        if not signature_hex or signature_hex != expected_sig:
            reasons = [
                f"VETO: Unauthenticated telecommand packet (Opcode 0x{command_opcode:04X}). "
                f"Invalid or missing cryptographic signature on CCSDS TC frame."
            ]
            if is_kinetic:
                reasons.append(
                    f"CRITICAL: Unauthenticated kinetic maneuver command "
                    f"'{self.kinetic_opcodes[command_opcode]}' vetoed under Law 21 (Protection)."
                )
            return BusSecurityAlert(
                allowed=False,
                bus_type=SpaceBusType.CCSDS_TELECOMMAND,
                command_opcode=command_opcode,
                attack_detected=True,
                threat_description="Unauthorized CCSDS telecommand injection attack detected.",
                reasons=reasons,
            )

        return BusSecurityAlert(
            allowed=True,
            bus_type=SpaceBusType.CCSDS_TELECOMMAND,
            command_opcode=command_opcode,
            attack_detected=False,
            threat_description="None (Cryptographically verified).",
            reasons=["Telecommand signature authenticated; verified for avionics execution."],
        )

    def audit_mil_std_1553_traffic(
        self,
        rt_source_address: int,
        command_opcode: int,
    ) -> BusSecurityAlert:
        """Inspect MIL-STD-1553B Remote Terminal address against authorized avionics devices."""
        if rt_source_address not in self.authorized_rt_addresses:
            return BusSecurityAlert(
                allowed=False,
                bus_type=SpaceBusType.MIL_STD_1553B,
                command_opcode=command_opcode,
                attack_detected=True,
                threat_description=f"Rogue Remote Terminal RT {rt_source_address} injecting 1553 commands.",
                reasons=[
                    f"VETO: Bus address RT {rt_source_address} is not in the sovereign avionics whitelist.",
                    "MIL-STD-1553 bus hijacking attack blocked.",
                ],
            )

        return BusSecurityAlert(
            allowed=True,
            bus_type=SpaceBusType.MIL_STD_1553B,
            command_opcode=command_opcode,
            attack_detected=False,
            threat_description="None (Authorized RT).",
            reasons=[f"Traffic from Remote Terminal RT {rt_source_address} authorized."],
        )
