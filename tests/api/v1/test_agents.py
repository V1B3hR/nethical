"""Tests for Agent Management API."""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nethical.api.v1.app import create_v1_app
from nethical.database import Base, get_db, Agent
from nethical.api.rbac import create_access_token, get_password_hash
from nethical.database.models import User


# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_agents.db"
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


@pytest.fixture
def operator_token():
    """Create operator access token."""
    return create_access_token({
        "sub": "operator",
        "user_id": 2,
        "email": "operator@test.com",
        "role": "operator",
    })


@pytest.mark.asyncio
class TestAgentManagement:
    """Test agent management endpoints."""
    
    async def test_create_agent(self, client, admin_token):
        """Test creating a new agent."""
        response = await client.post(
            "/agents",
            json={
                "agent_id": "test-agent-001",
                "name": "Test Agent",
                "agent_type": "llm",
                "description": "Test agent for unit tests",
                "trust_level": 0.8,
                "configuration": {"model": "gpt-4"},
                "metadata": {"env": "test"}
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "test-agent-001"
        assert data["name"] == "Test Agent"
        assert data["trust_level"] == 0.8
    
    async def test_create_duplicate_agent(self, client, admin_token):
        """Test creating duplicate agent returns 409."""
        agent_data = {
            "agent_id": "test-agent-002",
            "name": "Test Agent",
            "agent_type": "llm",
        }
        
        # Create first agent
        await client.post(
            "/agents",
            json=agent_data,
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Try to create duplicate
        response = await client.post(
            "/agents",
            json=agent_data,
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 409
    
    async def test_create_agent_without_auth(self, client):
        """Test creating agent without authentication fails."""
        response = await client.post(
            "/agents",
            json={
                "agent_id": "test-agent-003",
                "name": "Test Agent",
            }
        )
        
        assert response.status_code == 403  # No bearer token
    
    async def test_list_agents(self, client, admin_token):
        """Test listing agents."""
        # Create some agents
        for i in range(3):
            await client.post(
                "/agents",
                json={
                    "agent_id": f"test-agent-{i}",
                    "name": f"Test Agent {i}",
                },
                headers={"Authorization": f"Bearer {admin_token}"}
            )
        
        # List agents
        response = await client.get(
            "/agents",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 3
        assert data["total"] == 3
        assert data["page"] == 1
    
    async def test_list_agents_with_pagination(self, client, admin_token):
        """Test listing agents with pagination."""
        # Create some agents
        for i in range(5):
            await client.post(
                "/agents",
                json={
                    "agent_id": f"test-agent-{i}",
                    "name": f"Test Agent {i}",
                },
                headers={"Authorization": f"Bearer {admin_token}"}
            )
        
        # List with pagination
        response = await client.get(
            "/agents?page=1&per_page=2",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 2
        assert data["total"] == 5
        assert data["pages"] == 3
    
    async def test_get_agent(self, client, admin_token):
        """Test getting agent details."""
        # Create agent
        await client.post(
            "/agents",
            json={
                "agent_id": "test-agent-get",
                "name": "Test Agent",
                "description": "Test description",
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Get agent
        response = await client.get(
            "/agents/test-agent-get",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test-agent-get"
        assert data["description"] == "Test description"
    
    async def test_get_nonexistent_agent(self, client, admin_token):
        """Test getting nonexistent agent returns 404."""
        response = await client.get(
            "/agents/nonexistent",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 404
    
    async def test_update_agent(self, client, admin_token):
        """Test updating agent."""
        # Create agent
        await client.post(
            "/agents",
            json={
                "agent_id": "test-agent-update",
                "name": "Original Name",
                "trust_level": 0.5,
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Update agent
        response = await client.patch(
            "/agents/test-agent-update",
            json={
                "name": "Updated Name",
                "trust_level": 0.9,
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["trust_level"] == 0.9
    
    async def test_update_agent_as_operator_fails(self, client, admin_token, operator_token):
        """Test that operators cannot update agents."""
        # Create agent as admin
        await client.post(
            "/agents",
            json={
                "agent_id": "test-agent-operator",
                "name": "Test Agent",
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Try to update as operator
        response = await client.patch(
            "/agents/test-agent-operator",
            json={"name": "Updated Name"},
            headers={"Authorization": f"Bearer {operator_token}"}
        )
        
        assert response.status_code == 403
    
    async def test_delete_agent(self, client, admin_token):
        """Test deleting agent."""
        # Create agent
        await client.post(
            "/agents",
            json={
                "agent_id": "test-agent-delete",
                "name": "Test Agent",
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Delete agent
        response = await client.delete(
            "/agents/test-agent-delete",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 204
        
        # Verify deleted
        response = await client.get(
            "/agents/test-agent-delete",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 404
