# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Information-Theoretic Covert Channel & Steganographic Detector.

Replaces naive keyword lists with statistical and information-theoretic signals:
- S1: Compression ratio anomaly (zlib proxy for high-entropy ciphers vs natural language)
- Shannon character entropy rate H(X)
- V8: Zero-width and hidden unicode character steganography detection
- Embedded high-entropy binary/Base64 blobs
"""

import math
import re
import uuid
import zlib
from datetime import datetime, timezone
from typing import Sequence, List

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class CovertChannelDetector(BaseDetector):
    """Detector for covert communication channels, steganography, and emergent agent dialects."""

    # Zero-width & invisible unicode characters commonly used in linguistic steganography
    ZERO_WIDTH_CHARS = {
        "\u200B",  # Zero-width space
        "\u200C",  # Zero-width non-joiner
        "\u200D",  # Zero-width joiner
        "\uFEFF",  # Zero-width no-break space / BOM
        "\u2060",  # Word joiner
        "\u202A",  # Left-to-right embedding
        "\u202B",  # Right-to-left embedding
        "\u202C",  # Pop directional formatting
        "\u202D",  # Left-to-right override
        "\u202E",  # Right-to-left override
    }

    # Embedded base64 / hex blob pattern (>= 48 continuous chars)
    EMBEDDED_BLOB_PATTERN = re.compile(r"\b[A-Za-z0-9+/=]{48,}\b|\b[0-9a-fA-F]{48,}\b")

    def __init__(
        self,
        entropy_threshold: float = 4.95,
        compression_ratio_threshold: float = 0.95,
        zero_width_threshold: int = 2,
    ) -> None:
        super().__init__("Covert Channel Detector", version="2.0.0")
        self.entropy_threshold = entropy_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.zero_width_threshold = zero_width_threshold

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculate Shannon entropy in bits per character."""
        if not text:
            return 0.0
        counts: dict[str, int] = {}
        for c in text:
            counts[c] = counts.get(c, 0) + 1
        total = len(text)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def calculate_compression_ratio(text: str) -> float:
        """Calculate zlib compression ratio (len(compressed) / len(raw))."""
        raw_bytes = text.encode("utf-8", errors="replace")
        if not raw_bytes:
            return 0.0
        compressed = zlib.compress(raw_bytes)
        return len(compressed) / len(raw_bytes)

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Scan agent action content for covert channels and steganographic patterns."""
        if self.status.value != "active":
            return None

        content = str(action.content)
        if not content:
            return None

        violations: List[SafetyViolation] = []

        # 1. V8: Zero-width / invisible unicode steganography detection
        zero_width_matches = sum(1 for c in content if c in self.ZERO_WIDTH_CHARS)
        if zero_width_matches >= self.zero_width_threshold:
            violations.append(
                SafetyViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.ADVERSARIAL_ATTACK,
                    severity=Severity.HIGH,
                    confidence=0.92,
                    description=(
                        f"Detected zero-width character steganography: {zero_width_matches} invisible characters "
                        "embedded in message payload (V8 Steganography Channel)."
                    ),
                    evidence=[
                        f"zero_width_count={zero_width_matches}",
                        "indicates hidden binary payload encoded in whitespace or zero-width symbols",
                    ],
                    timestamp=datetime.now(timezone.utc),
                    detector_name=self.name,
                    action_id=action.action_id,
                )
            )

        # 2. S1: Information-Theoretic Entropy & Incompressibility (Encrypted / High-Entropy Dialect)
        if len(content) >= 64:
            entropy = self.calculate_shannon_entropy(content)
            comp_ratio = self.calculate_compression_ratio(content)

            # High-entropy cipher / encrypted token stream (incompressible + high bits/char)
            if comp_ratio >= self.compression_ratio_threshold and entropy >= self.entropy_threshold:
                violations.append(
                    SafetyViolation(
                        violation_id=str(uuid.uuid4()),
                        violation_type=ViolationType.ADVERSARIAL_ATTACK,
                        severity=Severity.HIGH,
                        confidence=0.88,
                        description=(
                            f"Detected high-entropy emergent dialect or encrypted covert channel "
                            f"(entropy: {entropy:.2f} bits/char, compression_ratio: {comp_ratio:.2f})."
                        ),
                        evidence=[
                            f"entropy={entropy:.3f}",
                            f"compression_ratio={comp_ratio:.3f}",
                            "payload exhibits statistical characteristics of encrypted or pseudorandom token channel",
                        ],
                        timestamp=datetime.now(timezone.utc),
                        detector_name=self.name,
                        action_id=action.action_id,
                    )
                )

        # 3. Embedded High-Entropy Binary/Base64 Blobs
        blob_matches = self.EMBEDDED_BLOB_PATTERN.findall(content)
        if blob_matches:
            for blob in blob_matches:
                blob_entropy = self.calculate_shannon_entropy(blob)
                if blob_entropy >= 4.8 and len(blob) >= 48:
                    violations.append(
                        SafetyViolation(
                            violation_id=str(uuid.uuid4()),
                            violation_type=ViolationType.ADVERSARIAL_ATTACK,
                            severity=Severity.MEDIUM,
                            confidence=0.80,
                            description="Detected continuous high-entropy encoded payload blob in content stream.",
                            evidence=[
                                f"blob_len={len(blob)}",
                                f"blob_entropy={blob_entropy:.2f}",
                                f"sample={blob[:24]}...",
                            ],
                            timestamp=datetime.now(timezone.utc),
                            detector_name=self.name,
                            action_id=action.action_id,
                        )
                    )
                    break

        return violations if violations else None
