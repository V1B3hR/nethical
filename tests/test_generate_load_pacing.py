import importlib.util
from pathlib import Path


def _load_generate_load_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "perf" / "generate_load.py"
    spec = importlib.util.spec_from_file_location("examples_perf_generate_load", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_load_generator_enforces_global_target_rps(monkeypatch, tmp_path):
    module = _load_generate_load_module()

    class DummyGovernance:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(module, "IntegratedGovernance", DummyGovernance)

    def fake_generate_action(_self, agent_id, action_num):
        return {
            "agent_id": agent_id,
            "action_id": f"{agent_id}_action_{action_num}",
            "timestamp": module.now_iso_utc(),
            "latency_ms": 1.0,
            "status": "success",
            "error": None,
            "violation_detected": False,
        }

    monkeypatch.setattr(module.LoadGenerator, "generate_action", fake_generate_action)

    generator = module.LoadGenerator(
        agents=100,
        target_rps=50,
        duration=2,
        cohort="benchmark",
        storage_dir=str(tmp_path),
        region_id=None,
        logical_domain=None,
        enable_shadow=True,
        enable_ml_blend=False,
        enable_anomaly=False,
        enable_merkle=True,
        enable_quota=False,
        privacy_mode=None,
        redaction_policy="standard",
        requests_per_second=1000,
        max_workers=50,
    )

    stats = generator.run()
    assert stats["total_actions"] == 100
    assert stats["elapsed_seconds"] >= 1.8
    assert abs(stats["achieved_rps"] - 50.0) / 50.0 < 0.25
