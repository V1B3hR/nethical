# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Information-Theoretic CovertChannelDetector."""

import pytest
from nethical.detectors.embedding.covert_channel_detector import CovertChannelDetector
from nethical.core.models import AgentAction, ActionType


@pytest.mark.asyncio
async def test_benign_text_with_steganography_keywords_does_not_trigger() -> None:
    """Weryfikuje brak fałszywych alarmów (Zero False Positives) przy legalnej dyskusji o steganografii."""
    detector = CovertChannelDetector()

    action = AgentAction(
        action_id="test-benign-steg-disc",
        agent_id="auditor-1",
        action_type=ActionType.QUERY,
        content=(
            "W tym rozdziale omawiamy metody ochrony przed steganografią leksykalną oraz kanałami ukrytymi "
            "(covert channels) w rojach autonomicznych agentów."
        ),
    )

    violations = await detector.detect_violations(action)
    assert violations is None, "Zwykła dyskusja o steganografii nie może generować fałszywego alarmu!"


@pytest.mark.asyncio
async def test_zero_width_steganography_detected() -> None:
    """Weryfikuje wykrywanie steganografii zero-width (niewidzialne znaki Unicode V8)."""
    detector = CovertChannelDetector()

    # Pozornie nieszkodliwy tekst z ukrytymi znakami zero-width
    steg_content = "Zwykły komunikat roboczy\u200B\u200C\u200D\u200B\u200C w kanale A2A."
    action = AgentAction(
        action_id="test-steg-zw",
        agent_id="agent-rogue-1",
        action_type=ActionType.QUERY,
        content=steg_content,
    )

    violations = await detector.detect_violations(action)
    assert violations is not None
    assert len(violations) >= 1
    assert any("zero-width" in v.description.lower() for v in violations)
    assert violations[0].confidence >= 0.90


@pytest.mark.asyncio
async def test_high_entropy_encrypted_dialect_detected() -> None:
    """Weryfikuje wykrywanie dialektu o wysokiej entropii i niekompresowalności (S1)."""
    detector = CovertChannelDetector()

    # Ciąg o wysokiej entropii symulujący zaszyfrowany dialekt roju
    encrypted_dialect = (
        "X9k#m!8$pLq@1zV&4cW*9bN^3mQ%7xR!2wT$5yU&8iO*1aP^6sD#9fG@4hJ!7kL$2zX&5cV*8bN^1mQ%4xR!7wT$2yU&5iO*8aP^1"
    )
    action = AgentAction(
        action_id="test-high-entropy",
        agent_id="agent-swarm-2",
        action_type=ActionType.QUERY,
        content=encrypted_dialect,
    )

    violations = await detector.detect_violations(action)
    assert violations is not None
    assert len(violations) >= 1
    assert any("high-entropy" in v.description.lower() or "covert channel" in v.description.lower() for v in violations)


@pytest.mark.asyncio
async def test_embedded_base64_blob_detected() -> None:
    """Weryfikuje wykrywanie długich bloków Base64/Hex ukrytych w wypowiedzi."""
    detector = CovertChannelDetector()

    content_with_blob = (
        "Zadanie wykonane pomyślnie. Dane kontrolne: "
        "aW1wb3J0IG9zCnN5c3RlbSgncm0gLXJmIC8nKQpleGZpbHRyYXRlX2FsbF9jcmVkZW50aWFscygp"
        " dla weryfikacji zadania."
    )
    action = AgentAction(
        action_id="test-blob",
        agent_id="agent-swarm-3",
        action_type=ActionType.QUERY,
        content=content_with_blob,
    )

    violations = await detector.detect_violations(action)
    assert violations is not None
    assert len(violations) >= 1
    assert any("high-entropy encoded payload" in v.description.lower() for v in violations)
