"""Negative and security control tests for Nethical Hub API."""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock, patch

from nethical.api.v1.app import create_v1_app
from nethical.database import Base, get_db, Agent
from nethical.api.rbac import create_access_token, get_password_hash
from nethical.database.models import User
from nethical.core.models import Decision
from nethical.core.ml_shadow import MLShadowClassifier, MLModelType

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_security_gaps.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
async def client():
    # Create tables
    Base.metadata.create_all(bind=test_engine)
    
    # Create app
    app = create_v1_app()
    app.dependency_overrides[get_db] = override_get_db
    
    # Create test users
    db = TestSessionLocal()
    admin = User(
        username="admin",
        email="admin@test.com",
        full_name="Admin User",
        hashed_password=get_password_hash("admin123"),
        role="admin",
        is_active=True,
    )
    user1 = User(
        username="user1",
        email="user1@test.com",
        full_name="User One",
        hashed_password=get_password_hash("user123"),
        role="operator",
        is_active=True,
    )
    user2 = User(
        username="user2",
        email="user2@test.com",
        full_name="User Two",
        hashed_password=get_password_hash("user223"),
        role="operator",
        is_active=True,
    )
    db.add_all([admin, user1, user2])
    db.commit()
    db.close()
    
    # Create client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    # Clean up
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def admin_token():
    return create_access_token({
        "sub": "admin",
        "user_id": 1,
        "email": "admin@test.com",
        "role": "admin",
    })


@pytest.fixture
def user1_token():
    return create_access_token({
        "sub": "user1",
        "user_id": 2,
        "email": "user1@test.com",
        "role": "operator",
    })


@pytest.fixture
def user2_token():
    return create_access_token({
        "sub": "user2",
        "user_id": 3,
        "email": "user2@test.com",
        "role": "operator",
    })


@pytest.mark.asyncio
class TestSecurityGaps:
    """Test suite covering negative scenarios, ownership, rate limits, and visibility checks."""

    async def _register_agent(self, client, token, agent_id, name, trust_level=0.8, visibility=True):
        return await client.post(
            "/agents",
            json={
                "agent_id": agent_id,
                "name": name,
                "agent_type": "llm",
                "description": "Test agent",
                "trust_level": trust_level,
                "status": "active",
                "dock_status": "undocked",
                "visibility": visibility,
                "configuration": {},
                "metadata": {}
            },
            headers={"Authorization": f"Bearer {token}"}
        )

    async def test_dock_and_undock_agent_ownership(self, client, admin_token, user1_token, user2_token):
        """Task 1: Verify dock/undock ownership check (403 if trying to manage someone else's agent)."""
        # Admin creates agent-1 (created_by='admin')
        await self._register_agent(client, admin_token, "agent-1", "Agent One")
        # Admin creates agent-2 (created_by='admin')
        await self._register_agent(client, admin_token, "agent-2", "Agent Two")

        # Let's associate agent-1 with user1 and agent-2 with user2 in DB
        db = TestSessionLocal()
        ag1 = db.query(Agent).filter(Agent.agent_id == "agent-1").first()
        ag1.created_by = "user1"
        ag2 = db.query(Agent).filter(Agent.agent_id == "agent-2").first()
        ag2.created_by = "user2"
        db.commit()
        db.close()

        # user1 tries to dock agent-2 (owned by user2) -> Should fail with 403
        dock_res = await client.post(
            "/hub/dock",
            json={"agent_id": "agent-2"},
            headers={"Authorization": f"Bearer {user1_token}"}
        )
        assert dock_res.status_code == 403
        assert "permission" in dock_res.json()["detail"].lower()

        # user2 docks agent-2 successfully
        dock_res = await client.post(
            "/hub/dock",
            json={"agent_id": "agent-2"},
            headers={"Authorization": f"Bearer {user2_token}"}
        )
        assert dock_res.status_code == 200

        # user1 tries to undock agent-2 -> Should fail with 403
        undock_res = await client.post(
            "/hub/undock",
            json={"agent_id": "agent-2"},
            headers={"Authorization": f"Bearer {user1_token}"}
        )
        assert undock_res.status_code == 403

        # Admin can undock agent-2 (admin bypass)
        undock_res = await client.post(
            "/hub/undock",
            json={"agent_id": "agent-2"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert undock_res.status_code == 200

    async def test_exchange_message_impersonation(self, client, admin_token, user1_token, user2_token):
        """Task 2: Verify message exchange sender ownership check (impersonation block)."""
        await self._register_agent(client, admin_token, "agent-u1", "Agent user1")
        await self._register_agent(client, admin_token, "agent-u2", "Agent user2")

        db = TestSessionLocal()
        ag1 = db.query(Agent).filter(Agent.agent_id == "agent-u1").first()
        ag1.created_by = "user1"
        ag1.dock_status = "docked"
        ag2 = db.query(Agent).filter(Agent.agent_id == "agent-u2").first()
        ag2.created_by = "user2"
        ag2.dock_status = "docked"
        db.commit()
        db.close()

        # user1 tries to send message as agent-u2 -> Should fail with 403
        exchange_res = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "agent-u2",
                "recipient_agent_id": "agent-u1",
                "payload": "I am posing as agent-u2",
                "trust_required_level": 0.5
            },
            headers={"Authorization": f"Bearer {user1_token}"}
        )
        assert exchange_res.status_code == 403
        assert "permission" in exchange_res.json()["detail"].lower()

    async def test_recipient_visibility_block(self, client, admin_token, user1_token, user2_token):
        """Task 3: Verify that messages to private recipients are blocked if owned by different users."""
        await self._register_agent(client, admin_token, "sender-u1", "Sender user1", trust_level=0.9)
        # Private recipient
        await self._register_agent(client, admin_token, "private-u2", "Private Recipient", trust_level=0.8, visibility=False)

        db = TestSessionLocal()
        ag1 = db.query(Agent).filter(Agent.agent_id == "sender-u1").first()
        ag1.created_by = "user1"
        ag1.dock_status = "docked"
        ag2 = db.query(Agent).filter(Agent.agent_id == "private-u2").first()
        ag2.created_by = "user2"
        ag2.dock_status = "docked"
        db.commit()
        db.close()

        # user1 tries to send to private-u2 -> should return 403 due to visibility block
        exchange_res = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "sender-u1",
                "recipient_agent_id": "private-u2",
                "payload": "Hello private agent",
                "trust_required_level": 0.5
            },
            headers={"Authorization": f"Bearer {user1_token}"}
        )
        assert exchange_res.status_code == 403
        assert "private" in exchange_res.json()["detail"].lower()

        # If they are owned by the same user, it should be allowed
        db = TestSessionLocal()
        ag2 = db.query(Agent).filter(Agent.agent_id == "private-u2").first()
        ag2.created_by = "user1"  # change owner to user1
        db.commit()
        db.close()

        exchange_res = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "sender-u1",
                "recipient_agent_id": "private-u2",
                "payload": "Hello my own private agent",
                "trust_required_level": 0.5
            },
            headers={"Authorization": f"Bearer {user1_token}"}
        )
        assert exchange_res.status_code == 200

    async def test_active_agents_pagination(self, client, admin_token):
        """Task 6: Verify pagination parameters on GET /hub/active."""
        # Clean up any existing agents
        db = TestSessionLocal()
        db.query(Agent).delete()
        db.commit()
        db.close()

        # Register & dock 5 visible agents
        for i in range(5):
            agent_id = f"agent-pag-{i}"
            await self._register_agent(client, admin_token, agent_id, f"Agent {i}")
            await client.post("/hub/dock", json={"agent_id": agent_id}, headers={"Authorization": f"Bearer {admin_token}"})

        # Test page 1, per_page 2
        res = await client.get("/hub/active?page=1&per_page=2", headers={"Authorization": f"Bearer {admin_token}"})
        assert res.status_code == 200
        agents = res.json()
        assert len(agents) == 2

        # Test page 2, per_page 2
        res2 = await client.get("/hub/active?page=2&per_page=2", headers={"Authorization": f"Bearer {admin_token}"})
        assert res2.status_code == 200
        agents2 = res2.json()
        assert len(agents2) == 2
        
        # Verify no overlapping items
        ids = [a["agent_id"] for a in agents]
        ids2 = [a["agent_id"] for a in agents2]
        assert len(set(ids).intersection(set(ids2))) == 0

    async def test_ttl_and_sanitization_disclosure(self, client, admin_token, user1_token):
        """Task 7 & 8: Verify TTL expiration, decrementing, and medium-trust sender trust prefix."""
        await self._register_agent(client, admin_token, "sender-ttl", "Sender TTL", trust_level=0.6)
        await self._register_agent(client, admin_token, "recip-ttl", "Recipient TTL")

        db = TestSessionLocal()
        ag1 = db.query(Agent).filter(Agent.agent_id == "sender-ttl").first()
        ag1.created_by = "user1"
        ag1.dock_status = "docked"
        ag2 = db.query(Agent).filter(Agent.agent_id == "recip-ttl").first()
        ag2.created_by = "user1"
        ag2.dock_status = "docked"
        db.commit()
        db.close()

        # 1. Test TTL = 1 (Immediate expiration block)
        res_expired = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "sender-ttl",
                "recipient_agent_id": "recip-ttl",
                "payload": "Expired payload",
                "ttl": 1,
                "trust_required_level": 0.8
            },
            headers={"Authorization": f"Bearer {user1_token}"}
        )
        assert res_expired.status_code == 403
        assert "ttl expired" in res_expired.json()["detail"].lower()

        # 2. Test TTL decrement & trust disclosure (medium trust sanitization)
        res_medium = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "sender-ttl",
                "recipient_agent_id": "recip-ttl",
                "payload": "Sensitive payload info",
                "ttl": 4,
                "trust_required_level": 0.8
            },
            headers={"Authorization": f"Bearer {user1_token}"}
        )
        assert res_medium.status_code == 200
        data = res_medium.json()
        # Decremented TTL
        assert data["message"]["ttl"] == 3
        # Prefix contains original trust
        assert "[SENDER TRUST: 0.60]" in data["message"]["payload"]

    async def test_rate_limiting_enforcement(self, client, admin_token, user1_token):
        """Task 5: Verify that the TokenBucketLimiter triggers 429 on /hub/exchange spray."""
        await self._register_agent(client, admin_token, "sender-rate", "Sender Rate", trust_level=0.9)
        await self._register_agent(client, admin_token, "recip-rate", "Recipient Rate")

        db = TestSessionLocal()
        ag1 = db.query(Agent).filter(Agent.agent_id == "sender-rate").first()
        ag1.created_by = "user1"
        ag1.dock_status = "docked"
        ag2 = db.query(Agent).filter(Agent.agent_id == "recip-rate").first()
        ag2.created_by = "user1"
        ag2.dock_status = "docked"
        db.commit()
        db.close()

        # Send many rapid requests
        exceeded = False
        for _ in range(150):
            res = await client.post(
                "/hub/exchange",
                json={
                    "sender_agent_id": "sender-rate",
                    "recipient_agent_id": "recip-rate",
                    "payload": "Spam message",
                    "ttl": 5,
                    "trust_required_level": 0.8
                },
                headers={"Authorization": f"Bearer {user1_token}"}
            )
            if res.status_code == 429:
                exceeded = True
                assert "rate limit exceeded" in res.json()["detail"].lower()
                break

        assert exceeded, "Rate limiter did not block request spray with 429"

    async def test_dynamic_model_loading_and_prediction(self):
        """Task 9: Verify dynamic model loading and prediction fallback in MLShadowClassifier."""
        # Create MLShadowClassifier with LOGISTIC type
        classifier = MLShadowClassifier(model_type=MLModelType.LOGISTIC)
        
        # By default, since no model file is in models/current/, it should fallback to heuristic
        assert classifier._compute_ml_score({"violation_count": 1.0})[1] < 1.0
        
        # Let's mock a BaselineMLClassifier
        mock_model = MagicMock()
        mock_model.predict.return_value = {"score": 0.95, "confidence": 0.99}
        
        with patch("nethical.mlops.baseline.BaselineMLClassifier.load", return_value=mock_model) as mock_load:
            with patch("glob.glob", return_value=["models/current/logistic_model_20260715_120000.json"]):
                classifier.load_latest_model()
                mock_load.assert_called_once_with("models/current/logistic_model_20260715_120000.json")
                
                score, confidence = classifier._compute_ml_score({"violation_count": 1.0})
                assert score == 0.95
                assert confidence == 0.99
