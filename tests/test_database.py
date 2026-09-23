# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for database connectivity, session lifecycle, and ORM models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nethical.database.models import Base, User, Agent, Policy, AuditLog
from nethical.database.database import get_db, init_db


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_user_model_lifecycle(test_db_session):
    """Test User creation, queries, to_dict, and updates."""
    user = User(
        username="auditor_test",
        email="auditor@sovereign.local",
        hashed_password="pbkdf2_sha256$mocked_hash",
        full_name="Chief Compliance Auditor",
        role="auditor",
        is_active=True,
    )
    test_db_session.add(user)
    test_db_session.commit()
    test_db_session.refresh(user)

    assert user.id is not None
    data = user.to_dict()
    assert data["username"] == "auditor_test"
    assert data["role"] == "auditor"
    assert data["created_at"] is not None


def test_agent_model_lifecycle(test_db_session):
    """Test Agent creation, metadata mapping, and to_dict."""
    agent = Agent(
        agent_id="agt-edge-001",
        name="Tactical Perimeter Monitor",
        agent_type="edge_autonomous",
        description="Autonomous node under NATO AEP-107 doctrine",
        trust_level=0.95,
        status="active",
        configuration={"timeout_ms": 25, "failsafe_mode": "cut_power"},
        meta_data={"jurisdiction": "UK", "classification": "restricted"},
        region_id="uk-south",
        dock_status="docked",
        visibility=True,
    )
    test_db_session.add(agent)
    test_db_session.commit()
    test_db_session.refresh(agent)

    data = agent.to_dict()
    assert data["agent_id"] == "agt-edge-001"
    assert data["trust_level"] == 0.95
    assert data["dock_status"] == "docked"
    assert data["metadata"]["jurisdiction"] == "UK"
    assert data["configuration"]["failsafe_mode"] == "cut_power"


def test_policy_model_lifecycle(test_db_session):
    """Test Policy model persistence, rules, and to_dict representation."""
    policy = Policy(
        policy_id="pol-safety-eu-01",
        name="EU AI Act High-Risk Policy",
        description="Enforces strict Article 9 risk management controls",
        version="2.1.0",
        policy_type="regulatory",
        priority=10,
        status="active",
        rules=[{"rule_id": "r1", "action": "block_on_biometric"}],
        scope="regional",
        fundamental_laws=["preserve_human_dignity", "prevent_kinetic_harm"],
        meta_data={"standards": ["EU_2024_1689"]},
    )
    test_db_session.add(policy)
    test_db_session.commit()
    test_db_session.refresh(policy)

    data = policy.to_dict()
    assert data["policy_id"] == "pol-safety-eu-01"
    assert data["version"] == "2.1.0"
    assert "preserve_human_dignity" in data["fundamental_laws"]
    assert data["metadata"]["standards"] == ["EU_2024_1689"]


def test_audit_log_model_lifecycle(test_db_session):
    """Test AuditLog persistence, merkle verification fields, and to_dict."""
    audit_log = AuditLog(
        log_id="log-event-999",
        event_type="threat_detected",
        agent_id="agt-edge-001",
        action="intercept_unauthorised_override",
        outcome="blocked",
        threat_type="adversarial_perturbation",
        threat_level="critical",
        risk_score=0.98,
        details={"vector": "gradient_ascent", "target_tensor": "actuator_throttle"},
        merkle_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        previous_hash="0000000000000000000000000000000000000000000000000000000000000000",
        verified=True,
    )
    test_db_session.add(audit_log)
    test_db_session.commit()
    test_db_session.refresh(audit_log)

    data = audit_log.to_dict()
    assert data["log_id"] == "log-event-999"
    assert data["threat_level"] == "critical"
    assert data["risk_score"] == 0.98
    assert data["verified"] is True
    assert data["timestamp"] is not None


def test_get_db_generator():
    """Test that get_db yields a session and closes it cleanly upon exit."""
    generator = get_db()
    session = next(generator)
    assert session is not None
    # Verify session is open
    assert session.is_active
    try:
        next(generator)
    except StopIteration:
        pass


def test_init_db_execution():
    """Test init_db executes without error."""
    init_db()
