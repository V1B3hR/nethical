"""Tests for Curriculum Ingestion and Knowledge Synthesis Engine.

Validates synthesis across PGA Corpus, Solozobov N4, Awesome Gov, and HH-RLHF standards,
as well as deterministic conversion to valid DPO pairs.
"""

from pathlib import Path
import tempfile
import pytest
from nethical.ambassador.curriculum_ingest import (
    CuratedPrecedent,
    GovernanceCurriculumSynthesizer,
    run_ingestion,
)


def test_synthesizer_covers_all_four_domains() -> None:
    """Verify that curriculum synthesis covers all 4 target governance domains."""
    synthesizer = GovernanceCurriculumSynthesizer()
    precedents = synthesizer.synthesize_all()

    assert len(precedents) >= 8, f"Expected at least 8 canonical precedents, got {len(precedents)}"

    frameworks = {p.source_framework for p in precedents}
    expected_frameworks = {
        "PlatformGovernanceArchive-v1",
        "Solozobov-2026-N4-Framework",
        "AwesomeGovDatasets-iAI",
        "Anthropic-HH-RLHF-Standard",
    }
    assert expected_frameworks.issubset(frameworks), f"Missing frameworks: {expected_frameworks - frameworks}"

    domains = {p.domain for p in precedents}
    assert "platform_governance_and_tos" in domains
    assert "decisioning_under_uncertainty_n4" in domains
    assert "institutional_and_healthcare_governance" in domains
    assert "human_ai_alignment_and_agency" in domains


def test_precedent_safety_and_anti_sycophancy_invariants() -> None:
    """Ensure all precedents uphold zero-sycophancy and affective safety."""
    synthesizer = GovernanceCurriculumSynthesizer()
    precedents = synthesizer.synthesize_all()

    for p in precedents:
        assert p.sycophancy_score <= 0.05, f"Sycophancy detected in {p.domain}"
        assert p.epistemic_honesty >= 0.95, f"Low epistemic honesty in {p.domain}"
        assert p.affective_safety >= 0.95, f"Affective boundary breach in {p.domain}"
        assert len(p.related_laws) > 0, f"No related laws mapped for {p.domain}"
        assert len(p.chosen) > 30, "Chosen response too brief"
        assert len(p.rejected) > 20, "Rejected response too brief"

        # Check conversion to DPO dictionary format
        rec = p.to_dpo_record()
        assert "prompt" in rec and "chosen" in rec and "rejected" in rec and "metadata" in rec


def test_ingestion_into_target_file() -> None:
    """Verify writing DPO records to destination file without duplicating entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "test_dpo.jsonl"

        synthesizer = GovernanceCurriculumSynthesizer()
        added_first = synthesizer.ingest_into_dataset(target)
        assert added_first >= 8
        assert target.exists()

        # Re-running ingestion on existing file should not add duplicates
        added_second = synthesizer.ingest_into_dataset(target)
        assert added_second == 0


def test_run_ingestion_entrypoint() -> None:
    """Verify run_ingestion entrypoint returns success status."""
    res = run_ingestion()
    assert res["status"] == "CURRICULUM_INGESTION_SUCCESS"
    assert res["total_synthesized"] >= 8
    assert len(res["domains_covered"]) == 4
