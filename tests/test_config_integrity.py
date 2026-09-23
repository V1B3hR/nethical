# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Test suite for configuration integrity and loader validation.

Verifies:
- All 57 configuration files in config/ (YAML, JSON, INI, ENV)
- Syntax validity and absence of corrupted schemas
- Regional .env configuration completeness and residency policies
- KillSwitchConfig.from_yaml() programmatic loading
- LawEnforcer.from_yaml() programmatic loading
- CorrelationEngine loading from config/correlation_rules.yaml
- ValidationConfig loading from config/
- Security constraints (no default passwords, localhost metrics bind)
"""

import configparser
import json
import os
from pathlib import Path
from typing import Dict, List
import pytest
import yaml

from nethical.core.correlation_engine import CorrelationEngine
from nethical.core.fundamental_laws import LawEnforcer
from nethical.core.kill_switch import KeyType, KillSwitchConfig, ShutdownMode
from validation_modules.config_loader import ValidationConfig


REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def test_all_yaml_and_json_files_syntax() -> None:
    """Ensure every YAML and JSON file in config/ is syntactically valid."""
    assert CONFIG_DIR.exists() and CONFIG_DIR.is_dir(), f"Config directory missing: {CONFIG_DIR}"

    yaml_count = 0
    json_count = 0

    for root, _, files in os.walk(CONFIG_DIR):
        for f in files:
            file_path = Path(root) / f
            if f.endswith((".yaml", ".yml")):
                with open(file_path, "r", encoding="utf-8") as fp:
                    content = yaml.safe_load(fp)
                assert content is not None, f"YAML file is empty or invalid: {file_path}"
                yaml_count += 1
            elif f.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as fp:
                    content = json.load(fp)
                assert content is not None, f"JSON file is empty or invalid: {file_path}"
                json_count += 1

    assert yaml_count >= 15, f"Expected at least 15 YAML config files, found {yaml_count}"
    assert json_count >= 10, f"Expected at least 10 JSON manifests, found {json_count}"


def test_ini_configuration_syntax() -> None:
    """Ensure example INI file parses cleanly."""
    ini_path = CONFIG_DIR / "example_config.ini"
    assert ini_path.exists()

    parser = configparser.ConfigParser()
    parser.read(str(ini_path), encoding="utf-8")
    assert "default" in parser.sections()
    assert "governance" in parser.sections()
    assert parser.get("default", "target") == "agent_example_001"


def test_all_regional_env_files_completeness() -> None:
    """Ensure all 16 regional .env files define required institutional parameters."""
    env_files: List[Path] = sorted(CONFIG_DIR.glob("*.env"))
    assert len(env_files) >= 16, f"Expected at least 16 regional .env files, found {len(env_files)}"

    required_keys = [
        "NETHICAL_REGION_ID",
        "NETHICAL_LOGICAL_DOMAIN",
        "NETHICAL_DATA_RESIDENCY_POLICY",
    ]

    for env_file in env_files:
        lines = env_file.read_text(encoding="utf-8").splitlines()
        env_vars: Dict[str, str] = {}
        for line in lines:
            trimmed = line.strip()
            if trimmed and not trimmed.startswith("#") and "=" in trimmed:
                key, val = trimmed.split("=", 1)
                env_vars[key.strip()] = val.strip()

        for req in required_keys:
            assert req in env_vars, f"Missing {req} in {env_file.name}"
            assert env_vars[req], f"Empty {req} in {env_file.name}"

        # Ensure region ID in file matches filename prefix
        expected_region = env_file.stem
        assert env_vars["NETHICAL_REGION_ID"] == expected_region, (
            f"Region mismatch in {env_file.name}: expected {expected_region}, "
            f"got {env_vars['NETHICAL_REGION_ID']}"
        )


def test_kill_switch_config_loader() -> None:
    """Verify KillSwitchConfig.from_yaml loads production config/kill_switch.yaml."""
    ks_path = CONFIG_DIR / "kill_switch.yaml"
    assert ks_path.exists()

    config = KillSwitchConfig.from_yaml(ks_path)
    assert config.enabled is True
    assert config.sla_target_ms == 1000
    assert config.default_mode == ShutdownMode.GRACEFUL
    assert config.graceful_timeout_s == 5.0
    assert config.multi_sig_enabled is True
    assert config.threshold == 2
    assert config.total_signers == 3
    assert config.key_type == KeyType.ED25519
    assert config.hardware_isolation_enabled is True
    assert config.enforce_safe_state is True
    assert config.audit_enabled is True
    assert config.retention_days == 365


def test_fundamental_laws_config_loader() -> None:
    """Verify LawEnforcer.from_yaml loads production config/fundamental_laws.yaml."""
    fl_path = CONFIG_DIR / "fundamental_laws.yaml"
    assert fl_path.exists()

    enforcer = LawEnforcer.from_yaml(fl_path)
    assert enforcer.registry is not None
    assert enforcer.category_weights.get("existence") == 0.9
    assert enforcer.category_weights.get("protection") == 0.95
    assert enforcer.category_weights.get("growth") == 0.6
    assert enforcer.config.get("fundamental_laws", {}).get("total_laws") == 25


def test_correlation_rules_loader() -> None:
    """Verify CorrelationEngine loads and parses config/correlation_rules.yaml."""
    cr_path = CONFIG_DIR / "correlation_rules.yaml"
    assert cr_path.exists()

    engine = CorrelationEngine(config_path=cr_path)
    assert engine.config is not None
    patterns = engine.config.get("multi_agent_patterns", [])
    pattern_names = {p.get("name") for p in patterns}

    assert "escalating_multi_id_probes" in pattern_names
    assert "payload_entropy_shift" in pattern_names
    assert "coordinated_attack" in pattern_names
    assert "distributed_reconnaissance" in pattern_names
    assert "anomalous_agent_cluster" in pattern_names


def test_validation_config_discovery() -> None:
    """Verify ValidationConfig discovers configuration from standard locations."""
    vc = ValidationConfig()
    assert vc.config is not None
    assert "global" in vc.config or "metrics" in vc.config


def test_security_constraints_in_monitoring_and_cache() -> None:
    """Assert security requirements: no plaintext changeme credentials and localhost bind defaults."""
    mon_path = CONFIG_DIR / "monitoring.yaml"
    mon_content = mon_path.read_text(encoding="utf-8")
    assert "changeme" not in mon_content, "Default insecure password 'changeme' found in monitoring.yaml"
    assert "0.0.0.0" not in mon_content, "Unsafe wildcard bind '0.0.0.0' found in monitoring.yaml"

    l2_path = CONFIG_DIR / "cache" / "l2_config.yaml"
    with open(l2_path, "r", encoding="utf-8") as fp:
        l2_cfg = yaml.safe_load(fp)

    # When Redis Cluster is enabled, Sentinel should not be concurrently active
    cluster_enabled = l2_cfg.get("redis", {}).get("cluster", {}).get("enabled", False)
    sentinel_enabled = l2_cfg.get("high_availability", {}).get("sentinel", {}).get("enabled", False)
    assert not (cluster_enabled and sentinel_enabled), (
        "Redis cluster and Sentinel are mutually exclusive and should not both be active."
    )
