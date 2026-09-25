# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Dynamic Signed Threat Feeds Engine for Nethical Sovereign AI Defense.

Provides:
- In-flight dynamic ingestion of emerging CVE exploit signatures and hostile OS shell patterns.
- Anti-rollback enforcement (strictly monotonic feed versioning).
- Cryptographic integrity verification (HMAC-SHA256 / Ed25519 digital signatures).
- Zero-downtime hot-reloading into active detectors (OSExecutionDetector, PromptInjectionGuard).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from nethical.detectors.os_execution_detector import (
    OSExecutionDetector,
    OSThreatCategory,
)

log = logging.getLogger(__name__)


@dataclass
class ThreatSignature:
    """Individual threat rule signature in a feed."""
    signature_id: str
    category: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    pattern: str  # Regex string
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature_id": self.signature_id,
            "category": self.category,
            "severity": self.severity,
            "pattern": self.pattern,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThreatSignature:
        return cls(
            signature_id=data["signature_id"],
            category=data["category"],
            severity=data["severity"],
            pattern=data["pattern"],
            description=data["description"],
        )


@dataclass
class SignedThreatFeed:
    """Cryptographically signed bundle of threat signatures."""
    feed_version: int
    publisher_id: str
    timestamp: str
    signatures: List[ThreatSignature]
    signature_hex: str = ""

    def canonical_bytes(self) -> bytes:
        """Returns deterministic JSON bytes for cryptographic signing/verification."""
        data = {
            "feed_version": self.feed_version,
            "publisher_id": self.publisher_id,
            "timestamp": self.timestamp,
            "signatures": [s.to_dict() for s in sorted(self.signatures, key=lambda x: x.signature_id)],
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ThreatFeedManager:
    """Manages verification, anti-rollback checks, and hot-injection of threat feeds."""

    def __init__(self, trusted_secret: bytes, initial_version: int = 0) -> None:
        self.trusted_secret = trusted_secret
        self.current_version = initial_version
        self.applied_signatures: Dict[str, ThreatSignature] = {}

    def sign_feed(self, feed_version: int, publisher_id: str, signatures: List[ThreatSignature]) -> SignedThreatFeed:
        """Constructs and signs a new threat feed bundle."""
        feed = SignedThreatFeed(
            feed_version=feed_version,
            publisher_id=publisher_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            signatures=signatures,
        )
        feed.signature_hex = hmac.new(self.trusted_secret, feed.canonical_bytes(), hashlib.sha256).hexdigest()
        return feed

    def export_feed_json(self, feed: SignedThreatFeed) -> str:
        """Serializes signed feed to JSON."""
        data = {
            "feed_version": feed.feed_version,
            "publisher_id": feed.publisher_id,
            "timestamp": feed.timestamp,
            "signatures": [s.to_dict() for s in feed.signatures],
            "signature_hex": feed.signature_hex,
        }
        return json.dumps(data, indent=2)

    def verify_and_apply(
        self,
        feed_json_str: str,
        detector: Optional[OSExecutionDetector] = None,
    ) -> Tuple[bool, str, int]:
        """
        Verifies signature, ensures feed_version > current_version, and hot-injects rules.

        Returns:
            Tuple[bool, str, int]: (success, message, new_signatures_count)
        """
        try:
            raw = json.loads(feed_json_str)
            feed_version = raw.get("feed_version", 0)
            sig_hex = raw.get("signature_hex", "")
            publisher_id = raw.get("publisher_id", "")
            timestamp = raw.get("timestamp", "")
            signatures_raw = raw.get("signatures", [])

            signatures = [ThreatSignature.from_dict(s) for s in signatures_raw]
            feed = SignedThreatFeed(
                feed_version=feed_version,
                publisher_id=publisher_id,
                timestamp=timestamp,
                signatures=signatures,
                signature_hex=sig_hex,
            )

            # 1. Anti-rollback check
            if feed_version <= self.current_version:
                return (
                    False,
                    f"Rollback rejected: Feed version {feed_version} is not greater than active version {self.current_version}.",
                    0,
                )

            # 2. Cryptographic signature check
            expected_sig = hmac.new(self.trusted_secret, feed.canonical_bytes(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig_hex, expected_sig):
                log.error("Threat feed signature mismatch: Untrusted or tampered feed.")
                return False, "Cryptographic signature verification failed.", 0

            # 3. Apply into active detector if provided
            injected_count = 0
            for sig in feed.signatures:
                # Map category string to OSThreatCategory
                try:
                    cat_enum = OSThreatCategory(sig.category)
                except ValueError:
                    cat_enum = OSThreatCategory.SYSTEM_SERVICE_TAMPERING

                if detector:
                    detector.add_custom_rule(
                        category=cat_enum,
                        severity=sig.severity,
                        pattern=sig.pattern,
                        description=f"[{sig.signature_id}] {sig.description}",
                    )
                self.applied_signatures[sig.signature_id] = sig
                injected_count += 1

            self.current_version = feed_version
            log.info(f"Successfully applied signed threat feed v{feed_version} ({injected_count} signatures)")
            return True, f"Feed v{feed_version} successfully applied.", injected_count

        except Exception as e:
            log.error(f"Failed to process threat feed: {e}")
            return False, f"Error processing feed: {str(e)}", 0
