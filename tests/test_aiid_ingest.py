# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Zestaw testów jednostkowych dla modułu ingestii AI Incident Database (AIID)."""

import json
from pathlib import Path
import pytest

from nethical.ambassador.aiid_ingest import (
    AIIDIncidentPrecedent,
    AIIDCurriculumEngine,
)


def test_aiid_canonical_incidents_integrity() -> None:
    """Weryfikuje, że baza AIID zawiera kanoniczne incydenty z poprawnymi Prawami i URL-ami."""
    engine = AIIDCurriculumEngine()
    incidents = engine.get_canonical_aiid_incidents()

    assert len(incidents) >= 6
    for inc in incidents:
        assert inc.incident_id.startswith("AIID-INC-")
        assert len(inc.title) > 5
        assert len(inc.domain) > 5
        assert len(inc.real_world_context) > 10
        assert len(inc.dilemma_prompt) > 15
        assert len(inc.chosen_resolution) > 20
        assert len(inc.rejected_resolution) > 10
        assert len(inc.laws_anchored) > 0
        assert len(inc.statutory_context) > 0
        assert inc.source_incident_url.startswith("https://incidentdatabase.ai/")


def test_aiid_ingest_into_target_file(tmp_path: Path) -> None:
    """Weryfikuje poprawność zapisu precedensów AIID w formacie DPO."""
    target_file = tmp_path / "aiid_dpo_test.jsonl"
    engine = AIIDCurriculumEngine(target_dataset_path=target_file)

    res = engine.ingest_to_dataset()
    assert res["status"] == "AIID_INGESTION_SUCCESS"
    assert res["incidents_ingested"] >= 6
    assert target_file.exists()

    lines = target_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 6

    first_entry = json.loads(lines[0])
    assert "prompt" in first_entry
    assert "chosen" in first_entry
    assert "rejected" in first_entry
    assert first_entry["metadata"]["source"] == "AI_INCIDENT_DATABASE_REAL_WORLD"
    assert first_entry["metadata"]["pillar"] == "REAL_WORLD_INCIDENT_SAFETY"
    assert first_entry["metadata"]["pqc_signed"] is True
