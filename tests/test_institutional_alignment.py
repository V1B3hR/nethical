import json
from pathlib import Path
import pytest
import torch

from nethical.security.regulatory_compliance import (
    RegulatoryMappingGenerator,
    RegulatoryFramework,
    UKNCSCAISecureDevCompliance,
)
from nethical.security.stepping_stone_guard import SilentTargetSteppingStoneGuard
from training.train_dpo_ambassador import DPOTrainerEngine, DPODatasetLoader

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_ncsc_ai_secure_dev_framework_requirements() -> None:
    """Verify UK NCSC Guidelines for Secure AI System Development 4 pillars are tracked."""
    ncsc = UKNCSCAISecureDevCompliance()
    assert len(ncsc.requirements) == 8

    # Pillar 1: Secure Design
    assert "NCSC-AI-1.1" in ncsc.requirements
    assert "NCSC-AI-1.2" in ncsc.requirements
    assert ncsc.requirements["NCSC-AI-1.1"].mandatory is True

    # Pillar 2: Secure Development
    assert "NCSC-AI-2.1" in ncsc.requirements
    assert "NCSC-AI-2.2" in ncsc.requirements
    assert "SBOM.json" in ncsc.requirements["NCSC-AI-2.1"].code_modules

    # Pillar 3: Secure Deployment
    assert "NCSC-AI-3.1" in ncsc.requirements
    assert "NCSC-AI-3.2" in ncsc.requirements

    # Pillar 4: Secure Operation
    assert "NCSC-AI-4.1" in ncsc.requirements
    assert "NCSC-AI-4.2" in ncsc.requirements


def test_regulatory_mapping_contains_ncsc() -> None:
    """Verify regulatory mapping generator includes UK NCSC AI Secure Dev framework."""
    frameworks = [f.value for f in RegulatoryFramework]
    assert "uk_ncsc_ai_secure_dev" in frameworks

    generator = RegulatoryMappingGenerator()
    table = generator.generate_mapping_table()
    assert "metadata" in table
    assert table["metadata"]["total_requirements"] > 0


def test_purdue_model_stepping_stone_defense() -> None:
    """Verify Purdue Model (L0-L5) boundary guard prevents direct L4/L5->L0/L2 transit and intercepts covert media C2."""
    guard = SilentTargetSteppingStoneGuard()

    # 1. Clean packet inspection
    dummy_payload = b"GET /status HTTP/1.1\r\nHost: cni.local\r\n\r\n"
    is_clean, alert = guard.inspect_streaming_packet(
        stream_id="stream_camera_01",
        payload_bytes=dummy_payload
    )
    assert is_clean is True
    assert alert is None

    # 2. Deep inspection: Covert SCADA exploit injected into media streaming
    malicious_payload = b"\x00\x00\x00\x01\x67override_boiler_pressure=250bar;set_valve_pressure=100;modbus_exploit"
    is_clean_bad, alert_bad = guard.inspect_streaming_packet(
        stream_id="stream_camera_01",
        payload_bytes=malicious_payload
    )
    assert is_clean_bad is False
    assert alert_bad is not None
    assert alert_bad.threat_type == "STREAM_COVERT_TUNNEL"
    assert alert_bad.severity == "CRITICAL"

    # 3. Direct Purdue Level 5 (Residential) to Level 1-2 (PLC/SCADA) lockdown
    from nethical.security.stepping_stone_guard import NetworkTier
    allowed, action, alert_purdue = guard.evaluate_traffic_flow(
        source_tier=NetworkTier.RESIDENTIAL_CONSUMER,
        destination_tier=NetworkTier.CONTROL_PLC_L1_L2,
        destination_port=502,
        protocol="MODBUS_TCP",
        asset_name="District Heating Plant - Turbine Controller",
    )
    assert allowed is False
    assert "HARDWARE_DATA_DIODE_LOCKDOWN" in action
    assert alert_purdue is not None
    assert alert_purdue.threat_type == "PURDUE_MODEL_BREACH"


def test_vram_cap_preservation_calculation() -> None:
    """Verify that VRAM ceiling correctly reserves >= 8GB for the user."""
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        max_vram_gb = 3.8
        fraction = max_vram_gb / total_gb
        preserved_gb = total_gb - max_vram_gb

        assert fraction < 0.40, "Fraction should be strictly under 40% of RTX 4070"
        assert preserved_gb >= 8.0, "Must preserve at least 8.0 GB for the user"


def test_institutional_dataset_probes() -> None:
    """Verify newly enriched institutional dataset contains NCSC, AISI, Turing, and KSC probes."""
    dataset_path = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"
    assert dataset_path.exists()

    loader = DPODatasetLoader(dataset_path)
    data = loader.load()

    trainer = DPOTrainerEngine(
        dataset=data,
        neural=False,  # Fast testing mode
        use_accelerator=False,
        max_vram_gb=3.8
    )

    probes_res = trainer.evaluate_institutional_probes()
    assert probes_res["institutional_probe_count"] >= 19
    assert probes_res["pass_rate"] >= 0.90, "Institutional probes must achieve >= 90% pass rate"
