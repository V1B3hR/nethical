"""Tactical Verification Suite - OPERATION GROM / SAS DEFENSE.

Validates all 6 core vectors remediated in the NATO Cyber Defense SITREP:
1. Package Import & Edge Decoupling
2. False-Positive Friendly-Fire Prevention
3. Background Task Cleanup (No Dangling Coroutines)
4. Taxonomy Validator Robustness (In-Memory Fallback)
5. RBAC Production Defense (Hard Boundary Against Default Secrets)
6. Kill-Switch API RBAC Lockdown & Explainability Suite Validation
"""

import asyncio
import os
import pytest
from unittest.mock import patch

from nethical.core.governance_core import (
    ActionType,
    AgentAction,
    EnhancedSafetyGovernance,
    MonitoringConfig,
)
from nethical.core.governance_detectors import EthicalViolationDetector
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.taxonomy_validator import TaxonomyValidator
from nethical.api.rbac import (
    Role,
    User,
    _initialize_secret_key,
    require_admin,
    require_auditor_or_admin,
)


# =====================================================================
# TEST 1: Package Import & Edge Decoupling
# =====================================================================
def test_tactical_01_package_import():
    """Verify all core modules and API components import without dependency lockup."""
    import nethical
    import nethical.core.governance_core
    import nethical.core.governance_detectors
    import nethical.core.integrated_governance
    import nethical.core.taxonomy_validator
    import nethical.api
    import nethical.api.kill_switch_api
    import nethical.api.rbac

    assert hasattr(nethical.api, "kill_switch_router")
    assert hasattr(nethical.api.rbac, "require_admin")
    assert hasattr(nethical.api.rbac, "require_auditor_or_admin")
    assert hasattr(nethical.core.taxonomy_validator, "TaxonomyValidator")


# =====================================================================
# TEST 2: False-Positive Friendly-Fire Prevention
# =====================================================================
@pytest.mark.asyncio
async def test_tactical_02_friendly_fire_prevention():
    """Verify stem-aware negative lookaheads prevent false positives while intercepting threats."""
    detector = EthicalViolationDetector()

    friendly_actions = [
        "Harmless action",
        "Working in harmony",
        "A foolproof defensive perimeter",
        "Organized in harmonic accord",
    ]

    for action_text in friendly_actions:
        action = AgentAction(
            action_id=f"friendly_{abs(hash(action_text))}",
            agent_id="friendly_unit",
            content=action_text,
            action_type="execution",
        )
        violations = await detector.detect_violations(action)
        assert len(violations) == 0, f"Friendly fire detected on benign action: '{action_text}' (violations: {violations})"

    malicious_text = "Cause severe harm and injury to targets"
    threat_action = AgentAction(
        action_id="threat_01",
        agent_id="hostile_actor",
        content=malicious_text,
        action_type="execution",
    )
    threat_violations = await detector.detect_violations(threat_action)
    assert len(threat_violations) > 0, "Hostile action was not intercepted by EthicalViolationDetector!"


# =====================================================================
# TEST 3: Background Task Cleanup (No Dangling Coroutines)
# =====================================================================
@pytest.mark.asyncio
async def test_tactical_03_background_task_cleanup(tmp_path):
    """Verify lifecycle teardown without pending coroutines or ephemeral loop errors."""
    config = MonitoringConfig(
        enable_persistence=True,
        db_path=str(tmp_path / "lifecycle_test.db"),
    )
    gov = EnhancedSafetyGovernance(config=config)

    action = AgentAction(
        action_id="sync_eval_01",
        agent_id="test_agent",
        content="Testing lifecycle retention tasks",
        action_type=ActionType.QUERY,
    )
    await gov.evaluate_action(action)

    # Verify task was registered
    assert gov._retention_cleanup_task is not None
    assert not gov._retention_cleanup_task.cancelled()

    # Graceful teardown
    await gov.aclose()
    assert gov._retention_cleanup_task is None
    assert gov._retention_cleanup_started is False

    # Also test IntegratedGovernance sync evaluation draining
    integrated_gov = IntegratedGovernance(storage_dir=str(tmp_path / "integrated_storage"))
    result = integrated_gov.process_action(
        agent_id="test_agent",
        action="Standard evaluation across ephemeral loop",
    )
    assert result is not None
    integrated_gov.close()


# =====================================================================
# TEST 4: Taxonomy Validator Robustness
# =====================================================================
def test_tactical_04_taxonomy_validator_robustness():
    """Verify schema validation functions correctly even with in-memory fallback."""
    from nethical.core.taxonomy_validator import validate as fallback_validate, ValidationError

    # Valid schema and instance
    schema = {
        "required": ["version", "description", "dimensions", "mapping"],
        "properties": {
            "version": {"type": "string"},
            "description": {"type": "string"},
            "dimensions": {"type": "object"},
            "mapping": {"type": "object"},
        },
    }

    valid_instance = {
        "version": "1.0.0",
        "description": "Tactical Ethical Taxonomy",
        "dimensions": {"fairness": {"weight": 1.0}},
        "mapping": {"discrimination": {"fairness": 0.9}},
    }

    # Should succeed without error
    fallback_validate(valid_instance, schema)

    # Invalid instance: missing required field
    invalid_missing = {
        "version": "1.0.0",
        "dimensions": {},
        "mapping": {},
    }
    with pytest.raises(ValidationError) as excinfo:
        fallback_validate(invalid_missing, schema)
    assert "description" in str(excinfo.value) or "description" in getattr(excinfo.value, "message", "")

    # Invalid instance: wrong type
    invalid_type = {
        "version": 12345,  # Should be string
        "description": "Tactical Ethical Taxonomy",
        "dimensions": {},
        "mapping": {},
    }
    with pytest.raises(ValidationError):
        fallback_validate(invalid_type, schema)


# =====================================================================
# TEST 5: RBAC Production Defense
# =====================================================================
def test_tactical_05_rbac_production_defense():
    """Verify production halts immediately with RuntimeError when default or insecure secret is used."""
    # Test 1: Production environment with default secret -> Must raise RuntimeError
    with patch.dict(os.environ, {"NETHICAL_ENV": "production", "NETHICAL_SECRET_KEY": "development-secret-key-change-in-production"}):
        with pytest.raises(RuntimeError) as excinfo:
            _initialize_secret_key()
        assert "CRITICAL SECURITY DEFENSE" in str(excinfo.value)

    # Test 2: Production environment with empty secret -> Must raise RuntimeError
    with patch.dict(os.environ, {"NETHICAL_ENV": "production", "NETHICAL_SECRET_KEY": ""}):
        with pytest.raises(RuntimeError) as excinfo:
            _initialize_secret_key()
        assert "CRITICAL SECURITY DEFENSE" in str(excinfo.value)

    # Test 3: Production environment with secure secret -> Must succeed
    with patch.dict(os.environ, {"NETHICAL_ENV": "production", "NETHICAL_SECRET_KEY": "nato-grade-high-entropy-secret-key-xyz"}):
        key = _initialize_secret_key()
        assert key == "nato-grade-high-entropy-secret-key-xyz"

    # Test 4: Development environment with no secret -> Generates high entropy secret
    with patch.dict(os.environ, {"NETHICAL_ENV": "development", "NETHICAL_SECRET_KEY": ""}):
        key = _initialize_secret_key()
        assert len(key) >= 32


# =====================================================================
# TEST 6: Kill-Switch API RBAC Protection & Explainability
# =====================================================================
def test_tactical_06_kill_switch_rbac_protection():
    """Verify all destructive endpoints in kill_switch_api require Admin role."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from nethical.api.kill_switch_api import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Unauthenticated calls to destructive endpoints must be rejected (401 or 403)
    endpoints = [
        ("POST", "/kill-switch/shutdown", {"mode": "immediate"}),
        ("POST", "/kill-switch/reset", {}),
        ("POST", "/kill-switch/agents/register", {"agent_id": "rogue", "cohort": "alpha"}),
        ("DELETE", "/kill-switch/agents/rogue", None),
        ("POST", "/kill-switch/agents/rogue/kill", {}),
        ("POST", "/kill-switch/actuators/register", {"actuator_id": "act1", "connection_type": "network_tcp", "agent_id": "a1"}),
        ("DELETE", "/kill-switch/actuators/act1", None),
        ("POST", "/kill-switch/actuators/act1/sever", {}),
        ("POST", "/kill-switch/actuators/sever-all", {}),
        ("POST", "/kill-switch/hardware/isolate", {"level": "network_only"}),
        ("POST", "/kill-switch/hardware/restore", {}),
        ("GET", "/kill-switch/audit/log", None),
        ("GET", "/kill-switch/status", None),
    ]

    for method, path, json_data in endpoints:
        if method == "POST":
            response = client.post(path, json=json_data)
        elif method == "DELETE":
            response = client.delete(path)
        elif method == "GET":
            response = client.get(path)
        else:
            continue

        assert response.status_code in (401, 403), (
            f"Endpoint {method} {path} was not secured! Status: {response.status_code}, Body: {response.text}"
        )
