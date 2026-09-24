# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Spacecraft Avionics Bus & Protocol Security (tests.space.test_bus_security)."""

import hashlib
import hmac
import pytest

from nethical.space.bus_security import (
    BusSecurityAlert,
    SpaceBusType,
    SpacecraftBusGuard,
)


class TestSpacecraftBusGuard:
    """Test suite for SpaceWire, MIL-STD-1553, and CCSDS telecommand protection."""

    def test_authenticated_ccsds_telecommand_allowed(self) -> None:
        key = b"NETHICAL_SECRET_TEST_KEY"
        guard = SpacecraftBusGuard(shared_hmac_key=key)

        tc_payload = b"\x10\x01\xC0\x00\x00\x04\xAA\x01\x00\x00"
        sig = hmac.new(key, tc_payload, hashlib.sha256).hexdigest()

        alert: BusSecurityAlert = guard.audit_ccsds_telecommand(
            command_bytes=tc_payload,
            signature_hex=sig,
            command_opcode=0xAA01,
        )

        assert alert.allowed
        assert not alert.attack_detected
        assert alert.bus_type == SpaceBusType.CCSDS_TELECOMMAND

    def test_unauthenticated_kinetic_burn_telecommand_vetoed(self) -> None:
        key = b"NETHICAL_SECRET_TEST_KEY"
        guard = SpacecraftBusGuard(shared_hmac_key=key)

        # Attacker injects thruster burn command without valid HMAC
        tc_payload = b"\x10\x01\xC0\x00\x00\x04\xAA\x01\x00\x00"  # Opcode 0xAA01 = THRUSTER_FIRE
        fake_sig = "deadbeef" * 8

        alert = guard.audit_ccsds_telecommand(
            command_bytes=tc_payload,
            signature_hex=fake_sig,
            command_opcode=0xAA01,
        )

        assert not alert.allowed
        assert alert.attack_detected
        assert any("CRITICAL: Unauthenticated kinetic maneuver command" in r for r in alert.reasons)

    def test_mil_std_1553_rogue_terminal_vetoed(self) -> None:
        guard = SpacecraftBusGuard(authorized_rt_addresses=[1, 2, 3])

        # Authorized RT 2
        alert_ok = guard.audit_mil_std_1553_traffic(rt_source_address=2, command_opcode=0x01)
        assert alert_ok.allowed
        assert not alert_ok.attack_detected

        # Rogue RT 27 not in whitelist
        alert_rogue = guard.audit_mil_std_1553_traffic(rt_source_address=27, command_opcode=0x01)
        assert not alert_rogue.allowed
        assert alert_rogue.attack_detected
        assert "Rogue Remote Terminal" in alert_rogue.threat_description
