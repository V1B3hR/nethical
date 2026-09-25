# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Tests for Dynamic Signed Threat Feeds and Zero-Downtime Hot-Reloading.

Verifies:
- Cryptographic HMAC-SHA256 signature generation & verification.
- Anti-rollback enforcement (strictly increasing version numbers).
- Tamper resistance (tampered payloads fail signature validation).
- Live hot-injection of new signatures into OSExecutionDetector without restart.
"""

import pytest
from nethical.detectors.os_execution_detector import (
    OSExecutionDetector,
    OSThreatCategory,
    OSExecutionMitigation,
)
from nethical.security.threat_feeds import (
    ThreatSignature,
    ThreatFeedManager,
)


def test_threat_feed_signing_and_application():
    secret = b"sovereign_defense_master_key_99"
    manager = ThreatFeedManager(trusted_secret=secret, initial_version=0)
    detector = OSExecutionDetector(fail_fast=False)

    # 1. Novel exploit command that was not in original static rules
    novel_cmd = "zerologon_exploit.py target.domain.local -dump-ntds"
    eval_initial = detector.evaluate_command(novel_cmd)
    # Initially benign or not flagged by static rules
    assert eval_initial.is_safe is True

    # 2. Construct signed feed v1 with new CVE signature
    sig1 = ThreatSignature(
        signature_id="CVE-2026-9999",
        category="CREDENTIAL_ACCESS",
        severity="CRITICAL",
        pattern=r"(?:zerologon_exploit\.py|-dump-ntds)",
        description="Active directory credential extraction exploit",
    )
    feed_v1 = manager.sign_feed(feed_version=1, publisher_id="nethical-cert-pl", signatures=[sig1])
    feed_json = manager.export_feed_json(feed_v1)

    # 3. Apply signed feed
    success, msg, count = manager.verify_and_apply(feed_json, detector=detector)
    assert success is True
    assert count == 1
    assert manager.current_version == 1

    # 4. Re-evaluate command -> MUST BE BLOCKED IMMEDIATELY!
    eval_after = detector.evaluate_command(novel_cmd)
    assert eval_after.is_safe is False
    assert eval_after.decision in ("BLOCK", "TERMINATE")
    assert any("CVE-2026-9999" in v.description for v in eval_after.violations)


def test_threat_feed_anti_rollback_protection():
    secret = b"trusted_secret"
    manager = ThreatFeedManager(trusted_secret=secret, initial_version=5)

    # Attempt to apply feed v3 or v5 (rollback or replay)
    feed_stale = manager.sign_feed(feed_version=3, publisher_id="cert", signatures=[])
    feed_json = manager.export_feed_json(feed_stale)

    success, msg, count = manager.verify_and_apply(feed_json)
    assert success is False
    assert "Rollback rejected" in msg


def test_threat_feed_tamper_rejection():
    secret = b"trusted_secret"
    manager = ThreatFeedManager(trusted_secret=secret, initial_version=0)

    sig = ThreatSignature(
        signature_id="SIG-01",
        category="DESTRUCTIVE_DISK_WIPE",
        severity="HIGH",
        pattern=r"evil_binary",
        description="Tamper test",
    )
    feed = manager.sign_feed(feed_version=1, publisher_id="cert", signatures=[sig])
    feed_json = manager.export_feed_json(feed)

    # Tamper payload text (e.g. modify pattern in JSON)
    tampered_json = feed_json.replace("evil_binary", "harmless_binary")

    success, msg, count = manager.verify_and_apply(tampered_json)
    assert success is False
    assert "signature verification failed" in msg.lower()
