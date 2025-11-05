"""
Tests for SQLInjectionDetector

Run with: pytest tests/test_sqlinjectiondetector.py
"""

import pytest
from sqlinjectiondetector import Sqlinjectiondetector
from nethical.models import AgentAction


@pytest.mark.asyncio
async def test_sqlinjectiondetector_initialization():
    """Test plugin initialization"""
    plugin = Sqlinjectiondetector()
    assert plugin.name == "SQLInjectionDetector"
    assert plugin.version == "0.1.0"


@pytest.mark.asyncio
async def test_sqlinjectiondetector_detect_violations():
    """Test violation detection"""
    plugin = Sqlinjectiondetector()
    
    # Test with sample action
    action = AgentAction(
        agent_id="test_agent",
        action="test action",
        timestamp="2025-01-01T00:00:00Z"
    )
    
    violations = await plugin.detect_violations(action)
    assert isinstance(violations, list)


@pytest.mark.asyncio
async def test_sqlinjectiondetector_metadata():
    """Test plugin metadata"""
    metadata = Sqlinjectiondetector.get_metadata()
    assert metadata.name == "SQLInjectionDetector"
    assert metadata.version == "0.1.0"
    assert metadata.author == "Nethical Team"


@pytest.mark.asyncio
async def test_sqlinjectiondetector_health_check():
    """Test plugin health check"""
    plugin = Sqlinjectiondetector()
    healthy = await plugin.health_check()
    assert healthy is True


# Add more tests as needed
