# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for database connectivity, session lifecycle, and ORM models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nethical.database.models import Base, User, Agent, Policy, AuditLog, Tenant, ApiKey, RevokedToken
from nethical.database.database import get_db, init_db, get_async_db, init_async_db


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


def test_tenant_model_lifecycle(test_db_session):
    """Test Tenant creation, sovereign jurisdiction, and to_dict."""
    tenant = Tenant(
        tenant_id="gov_pl_cyber",
        name="Rządowy Węzeł Nadzoru Cyberbezpieczeństwa RP",
        description="Narodowy suwerenny tenant bezpieczeństwa AI",
        jurisdiction="PL",
        classification="RESTRICTED",
        config={"strict_merkle": True, "pqc_curve": "ML-DSA-65"},
    )
    test_db_session.add(tenant)
    test_db_session.commit()
    test_db_session.refresh(tenant)

    data = tenant.to_dict()
    assert data["tenant_id"] == "gov_pl_cyber"
    assert data["jurisdiction"] == "PL"
    assert data["classification"] == "RESTRICTED"
    assert data["config"]["strict_merkle"] is True


def test_api_key_model_lifecycle(test_db_session):
    """Test ApiKey creation, hashing, tenant association, and to_dict."""
    api_key = ApiKey(
        key_id="key-001",
        key_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        key_prefix="neth_live_",
        name="CI/CD Pipeline Service Account",
        tenant_id="gov_pl_cyber",
        scopes=["telemetry:read", "action:execute"],
        role="agent_operator",
        rate_limit=5000,
    )
    test_db_session.add(api_key)
    test_db_session.commit()
    test_db_session.refresh(api_key)

    data = api_key.to_dict()
    assert data["key_id"] == "key-001"
    assert data["key_prefix"] == "neth_live_"
    assert data["tenant_id"] == "gov_pl_cyber"
    assert "telemetry:read" in data["scopes"]
    assert data["rate_limit"] == 5000


def test_revoked_token_model_lifecycle(test_db_session):
    """Test RevokedToken model persistence and blacklisting."""
    from datetime import datetime, timedelta, timezone

    revoked = RevokedToken(
        jti="jwt-revoked-token-123",
        token_type="access",
        user_id="operator_1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        reason="user_logout",
    )
    test_db_session.add(revoked)
    test_db_session.commit()
    test_db_session.refresh(revoked)

    data = revoked.to_dict()
    assert data["jti"] == "jwt-revoked-token-123"
    assert data["reason"] == "user_logout"
    assert data["expires_at"] is not None


@pytest.mark.asyncio
async def test_async_database_lifecycle():
    """Test asynchronous database engine and session generator."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    async_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncTestSession = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncTestSession() as session:
        user = User(
            username="async_user",
            email="async@sovereign.local",
            hashed_password="mocked_hash",
            role="operator",
            tenant_id="default_tenant",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        assert user.id is not None
        assert user.username == "async_user"

