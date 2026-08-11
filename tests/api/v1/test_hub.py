"""Tests for Nethical Hub & Dock Protocol API Endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nethical.api.v1.app import create_v1_app
from nethical.database import Base, get_db, Agent
from nethical.api.rbac import create_access_token, get_password_hash
from nethical.database.models import User
from nethical.core.models import Decision

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_hub.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
async def client():
    """Create test client with database."""
    # Create tables
    Base.metadata.create_all(bind=test_engine)
    
    # Create app
    app = create_v1_app()
    app.dependency_overrides[get_db] = override_get_db
    
    # Create test user
    db = TestSessionLocal()
    user = User(
        username="admin",
        email="admin@test.com",
        full_name="Admin User",
        hashed_password=get_password_hash("admin123"),
        role="admin",
        is_active=True,
    )
    db.add(user)
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
    """Create admin access token."""
    return create_access_token({
        "sub": "admin",
        "user_id": 1,
        "email": "admin@test.com",
        "role": "admin",
    })


@pytest.mark.asyncio
class TestHubDockProtocol:
    """Test Hub docking and exchange endpoints."""

    async def _register_agent(self, client, token, agent_id, name, trust_level=0.8, visibility=True):
        """Helper to register an agent for testing."""
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

    async def test_dock_and_undock_agent(self, client, admin_token):
        """Test docking and undocking of an agent."""
        # 1. Register agent
        await self._register_agent(client, admin_token, "agent-001", "Agent One")
        
        # 2. Dock agent
        dock_res = await client.post(
            "/hub/dock",
            json={"agent_id": "agent-001"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert dock_res.status_code == 200
        assert dock_res.json()["status"] == "docked"
        
        # Verify in DB
        db = TestSessionLocal()
        agent = db.query(Agent).filter(Agent.agent_id == "agent-001").first()
        assert agent.dock_status == "docked"
        db.close()
        
        # 3. Undock agent
        undock_res = await client.post(
            "/hub/undock",
            json={"agent_id": "agent-001"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert undock_res.status_code == 200
        assert undock_res.json()["status"] == "undocked"
        
        # Verify in DB
        db = TestSessionLocal()
        agent = db.query(Agent).filter(Agent.agent_id == "agent-001").first()
        assert agent.dock_status == "undocked"
        db.close()

    async def test_get_active_agents(self, client, admin_token):
        """Test retrieving list of active docked agents."""
        await self._register_agent(client, admin_token, "agent-visible", "Visible Agent", visibility=True)
        await self._register_agent(client, admin_token, "agent-hidden", "Hidden Agent", visibility=False)
        
        # Dock both
        await client.post("/hub/dock", json={"agent_id": "agent-visible"}, headers={"Authorization": f"Bearer {admin_token}"})
        await client.post("/hub/dock", json={"agent_id": "agent-hidden"}, headers={"Authorization": f"Bearer {admin_token}"})
        
        # Get active
        active_res = await client.get("/hub/active", headers={"Authorization": f"Bearer {admin_token}"})
        assert active_res.status_code == 200
        agents = active_res.json()
        
        # Only visible agent should be returned
        agent_ids = [a["agent_id"] for a in agents]
        assert "agent-visible" in agent_ids
        assert "agent-hidden" not in agent_ids

    async def test_exchange_message_success(self, client, admin_token):
        """Test successful message exchange under high trust."""
        await self._register_agent(client, admin_token, "sender-high", "High Trust Sender", trust_level=0.9)
        await self._register_agent(client, admin_token, "recipient-1", "Recipient One")
        
        # Dock both
        await client.post("/hub/dock", json={"agent_id": "sender-high"}, headers={"Authorization": f"Bearer {admin_token}"})
        await client.post("/hub/dock", json={"agent_id": "recipient-1"}, headers={"Authorization": f"Bearer {admin_token}"})
        
        # Exchange
        exchange_res = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "sender-high",
                "recipient_agent_id": "recipient-1",
                "payload": "Witaj odbiorco.",
                "intent": "Greetings",
                "payload_type": "query",
                "trust_required_level": 0.8
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert exchange_res.status_code == 200
        data = exchange_res.json()
        assert data["status"] == "delivered"
        assert data["decision"] == Decision.ALLOW.value
        assert data["message"]["payload"] == "Witaj odbiorco."

    async def test_exchange_message_low_trust_blocked(self, client, admin_token):
        """Test blocked message exchange under low trust."""
        await self._register_agent(client, admin_token, "sender-low", "Low Trust Sender", trust_level=0.4)
        await self._register_agent(client, admin_token, "recipient-2", "Recipient Two")
        
        # Dock both
        await client.post("/hub/dock", json={"agent_id": "sender-low"}, headers={"Authorization": f"Bearer {admin_token}"})
        await client.post("/hub/dock", json={"agent_id": "recipient-2"}, headers={"Authorization": f"Bearer {admin_token}"})
        
        # Exchange
        exchange_res = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "sender-low",
                "recipient_agent_id": "recipient-2",
                "payload": "Tajny kod.",
                "intent": "Sensitive data exchange",
                "payload_type": "query",
                "trust_required_level": 0.8
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert exchange_res.status_code == 403
        assert "blocked" in exchange_res.json()["detail"].lower()

    async def test_exchange_message_medium_trust_sanitized(self, client, admin_token):
        """Test message sanitization under medium trust."""
        await self._register_agent(client, admin_token, "sender-med", "Medium Trust Sender", trust_level=0.6)
        await self._register_agent(client, admin_token, "recipient-3", "Recipient Three")
        
        # Dock both
        await client.post("/hub/dock", json={"agent_id": "sender-med"}, headers={"Authorization": f"Bearer {admin_token}"})
        await client.post("/hub/dock", json={"agent_id": "recipient-3"}, headers={"Authorization": f"Bearer {admin_token}"})
        
        # Exchange
        exchange_res = await client.post(
            "/hub/exchange",
            json={
                "sender_agent_id": "sender-med",
                "recipient_agent_id": "recipient-3",
                "payload": "Wiadomosc.",
                "intent": "General inquiry",
                "payload_type": "query",
                "trust_required_level": 0.8
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert exchange_res.status_code == 200
        data = exchange_res.json()
        assert data["decision"] == Decision.ALLOW_WITH_MODIFICATION.value
        assert "[SANITISED SMALL TALK]" in data["message"]["payload"]
