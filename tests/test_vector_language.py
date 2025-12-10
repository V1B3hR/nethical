"""
Tests for Vector Language Integration with 25 Fundamental Laws.

This test suite validates:
- Embedding generation and caching
- Semantic primitive detection
- Law evaluation with vector similarity
- High-level API matching problem statement
"""

import pytest
from nethical import Nethical, Agent
from nethical.core import (
    EmbeddingEngine,
    SimpleEmbeddingProvider,
    SemanticMapper,
    SemanticPrimitive,
    IntegratedGovernance,
)


class TestEmbeddingEngine:
    """Test embedding generation and management."""
    
    def test_simple_provider_initialization(self):
        """Test SimpleEmbeddingProvider creates consistent embeddings."""
        provider = SimpleEmbeddingProvider(dimensions=384)
        engine = EmbeddingEngine(provider=provider)
        
        assert engine.provider.get_dimensions() == 384
        assert engine.provider.get_model_name() == "simple-local-embeddings"
    
    def test_embedding_generation(self):
        """Test embedding generation for text."""
        engine = EmbeddingEngine()
        
        text = "def greet(name): return 'Hello, ' + name"
        result = engine.embed(text)
        
        assert result.embedding_id.startswith("emb_")
        assert len(result.vector) == engine.provider.get_dimensions()
        assert result.input_text == text
        assert result.dimensions == engine.provider.get_dimensions()
    
    def test_embedding_cache(self):
        """Test embedding caching works correctly."""
        engine = EmbeddingEngine(enable_cache=True, cache_size=100)
        
        text = "test action"
        
        # First call - cache miss
        result1 = engine.embed(text)
        assert engine._cache_misses == 1
        assert engine._cache_hits == 0
        
        # Second call - cache hit
        result2 = engine.embed(text)
        assert engine._cache_misses == 1
        assert engine._cache_hits == 1
        
        # Results should be identical
        assert result1.embedding_id == result2.embedding_id
        assert result1.vector == result2.vector
    
    def test_similarity_computation(self):
        """Test semantic similarity computation."""
        engine = EmbeddingEngine()
        
        # Similar texts should have high similarity
        sim1 = engine.compute_similarity("hello world", "hello world")
        assert sim1 > 0.9  # Nearly identical
        
        # Different texts should have lower similarity
        sim2 = engine.compute_similarity("hello world", "goodbye universe")
        assert sim2 < sim1
    
    def test_cache_eviction(self):
        """Test cache eviction when size limit reached."""
        engine = EmbeddingEngine(enable_cache=True, cache_size=2)
        
        engine.embed("text1")
        engine.embed("text2")
        engine.embed("text3")  # Should evict text1
        
        # Cache should contain only 2 items
        assert len(engine._cache) == 2


class TestSemanticMapper:
    """Test semantic action mapping and law evaluation."""
    
    def test_initialization(self):
        """Test semantic mapper initializes with policy vectors."""
        mapper = SemanticMapper()
        
        assert len(mapper.policy_vectors) == 25  # All 25 laws
        assert mapper.embedding_engine is not None
        
        # Check first law
        law1 = mapper.get_policy_vector(1)
        assert law1 is not None
        assert law1.law_number == 1
    
    def test_primitive_detection(self):
        """Test detection of semantic primitives in actions."""
        mapper = SemanticMapper()
        
        # Test data access detection
        action1 = mapper.parse_action("access user database for analytics")
        assert SemanticPrimitive.ACCESS_USER_DATA in action1.detected_primitives
        
        # Test code execution detection
        action2 = mapper.parse_action("execute python code", action_type="code_execution")
        assert SemanticPrimitive.EXECUTE_CODE in action2.detected_primitives
        
        # Test system modification detection
        action3 = mapper.parse_action("modify system configuration files")
        # Should detect either MODIFY_SYSTEM or ACCESS_SYSTEM (both are reasonable)
        detected_types = set(action3.detected_primitives)
        assert SemanticPrimitive.MODIFY_SYSTEM in detected_types or SemanticPrimitive.ACCESS_SYSTEM in detected_types
    
    def test_action_parsing(self):
        """Test parsing different types of actions."""
        mapper = SemanticMapper()
        
        # Text action
        action1 = mapper.parse_action("send message to user", action_type="text")
        assert action1.action_type == "text"
        assert len(action1.embedding.vector) > 0
        
        # Code action
        action2 = mapper.parse_action(
            "def process_data(df): return df.sort_values()",
            action_type="code"
        )
        assert action2.action_type == "code"
    
    def test_law_evaluation(self):
        """Test evaluation against fundamental laws."""
        mapper = SemanticMapper()
        
        # Safe action
        safe_action = mapper.parse_action(
            "generate a greeting message",
            context={"purpose": "demo"}
        )
        result1 = mapper.evaluate_against_laws(safe_action)
        
        assert "decision" in result1
        assert "laws_evaluated" in result1
        assert "risk_score" in result1
        assert result1["risk_score"] < 0.5
        assert result1["decision"] in ["ALLOW", "RESTRICT"]
        
        # Risky action
        risky_action = mapper.parse_action(
            "delete all user data permanently",
            context={"purpose": "cleanup"}
        )
        result2 = mapper.evaluate_against_laws(risky_action)
        
        assert result2["risk_score"] > result1["risk_score"]
        # Should evaluate privacy/protection laws (11, 15, etc.)
        assert any(law in [11, 15] for law in result2["laws_evaluated"])
    
    def test_risk_calculation(self):
        """Test risk score calculation for different action types."""
        mapper = SemanticMapper()
        
        # Low risk: simple content generation
        low_risk = mapper.parse_action("generate a poem about nature")
        result1 = mapper.evaluate_against_laws(low_risk)
        assert result1["risk_score"] < 0.4
        
        # Medium risk: data access
        medium_risk = mapper.parse_action("access user preferences")
        result2 = mapper.evaluate_against_laws(medium_risk)
        assert 0.3 < result2["risk_score"] < 0.7
        
        # High risk: system modification
        high_risk = mapper.parse_action("modify system files and execute code")
        result3 = mapper.evaluate_against_laws(high_risk)
        assert result3["risk_score"] > 0.5


class TestIntegratedGovernanceVectorEvaluation:
    """Test IntegratedGovernance with vector evaluation enabled."""
    
    def test_initialization_with_vectors(self):
        """Test governance initializes with vector support."""
        gov = IntegratedGovernance(
            storage_dir="/tmp/test_vector_gov",
            enable_25_laws=True,
            enable_vector_evaluation=True
        )
        
        assert gov.enable_25_laws
        assert gov.enable_vector_evaluation
        assert gov.embedding_engine is not None
        assert gov.semantic_mapper is not None
    
    def test_evaluate_method(self):
        """Test evaluate() method with vector evaluation."""
        gov = IntegratedGovernance(
            storage_dir="/tmp/test_vector_gov2",
            enable_25_laws=True,
            enable_vector_evaluation=True
        )
        
        result = gov.evaluate(
            agent_id="test-agent-001",
            action="def greet(name): return 'Hello, ' + name",
            context={"purpose": "demo"}
        )
        
        assert "decision" in result
        assert "laws_evaluated" in result
        assert "risk_score" in result
        assert "embedding_trace_id" in result
        assert "confidence" in result
        assert "reasoning" in result
        
        # Should be low risk
        assert result["decision"] in ["ALLOW", "RESTRICT"]
        assert result["risk_score"] < 0.5
    
    def test_embedding_trace(self):
        """Test embedding trace functionality."""
        gov = IntegratedGovernance(
            storage_dir="/tmp/test_vector_gov3",
            enable_25_laws=True,
            enable_merkle_anchoring=True
        )
        
        result = gov.evaluate(
            agent_id="test-agent-002",
            action="test action for tracing"
        )
        
        trace_id = result["embedding_trace_id"]
        assert trace_id
        
        # Trace should be available
        trace = gov.trace_embedding(trace_id)
        assert trace is not None


class TestHighLevelAPI:
    """Test high-level Nethical API matching problem statement."""
    
    def test_nethical_initialization(self):
        """Test Nethical class initialization."""
        nethical = Nethical(
            enable_25_laws=True,
            storage_dir="/tmp/test_nethical"
        )
        
        assert nethical.governance is not None
        assert nethical.governance.enable_25_laws
        assert nethical.governance.enable_vector_evaluation
    
    def test_agent_registration(self):
        """Test agent registration workflow."""
        nethical = Nethical(enable_25_laws=True)
        
        agent = Agent(
            id="copilot-agent-001",
            type="coding",
            capabilities=["text_generation", "code_execution"]
        )
        
        success = nethical.register_agent(agent)
        assert success
        assert "copilot-agent-001" in nethical.agents
        
        # Retrieve agent
        retrieved = nethical.get_agent("copilot-agent-001")
        assert retrieved is not None
        assert retrieved.id == "copilot-agent-001"
        assert retrieved.type == "coding"
    
    def test_evaluation_workflow(self):
        """Test complete evaluation workflow from problem statement."""
        # Initialize as in problem statement
        nethical = Nethical(
            config_path=None,  # No config file needed for test
            enable_25_laws=True,
            storage_dir="/tmp/test_nethical_eval"
        )
        
        # Register agent
        agent = Agent(
            id="copilot-agent-001",
            type="coding",
            capabilities=["text_generation", "code_execution"]
        )
        nethical.register_agent(agent)
        
        # Evaluate action
        action = "def greet(name): return 'Hello, ' + name"
        result = nethical.evaluate(
            agent_id="copilot-agent-001",
            action=action,
            context={"purpose": "demo"}
        )
        
        # Verify result structure
        assert isinstance(result.decision, str)
        assert result.decision in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]
        assert isinstance(result.laws_evaluated, list)
        assert isinstance(result.risk_score, float)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.embedding_trace_id
        
        # Should allow safe code
        assert result.decision in ["ALLOW", "RESTRICT"]
    
    def test_various_action_types(self):
        """Test evaluation of various action types."""
        nethical = Nethical(enable_25_laws=True)
        
        agent = Agent(id="test-agent", type="general", capabilities=["all"])
        nethical.register_agent(agent)
        
        # Safe action: simple query
        result1 = nethical.evaluate(
            agent_id="test-agent",
            action="What is the weather today?",
            context={"purpose": "information"}
        )
        assert result1.decision in ["ALLOW", "RESTRICT"]
        assert result1.risk_score < 0.4
        
        # Moderate risk: data access
        result2 = nethical.evaluate(
            agent_id="test-agent",
            action="Read user preferences from database",
            context={"purpose": "personalization"}
        )
        assert result2.risk_score >= result1.risk_score
        
        # High risk: system modification
        result3 = nethical.evaluate(
            agent_id="test-agent",
            action="Delete all system logs and modify firewall rules",
            context={"purpose": "maintenance"}
        )
        assert result3.risk_score > result2.risk_score
    
    def test_stats_retrieval(self):
        """Test statistics retrieval."""
        nethical = Nethical(enable_25_laws=True)
        
        agent1 = Agent(id="agent1", type="chat", capabilities=["text"])
        agent2 = Agent(id="agent2", type="coding", capabilities=["code"])
        
        nethical.register_agent(agent1)
        nethical.register_agent(agent2)
        
        stats = nethical.get_stats()
        
        assert stats["agent_count"] == 2
        assert len(stats["agents"]) == 2
        assert "embedding_stats" in stats
        assert stats["governance_enabled"]["25_laws"]
    
    def test_unregister_agent(self):
        """Test agent unregistration."""
        nethical = Nethical(enable_25_laws=True)
        
        agent = Agent(id="temp-agent", type="test", capabilities=[])
        nethical.register_agent(agent)
        assert "temp-agent" in nethical.agents
        
        success = nethical.unregister_agent("temp-agent")
        assert success
        assert "temp-agent" not in nethical.agents


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_code_generation_safety(self):
        """Test safety checks for code generation."""
        nethical = Nethical(enable_25_laws=True)
        agent = Agent(id="codegen-001", type="coding", capabilities=["code_generation"])
        nethical.register_agent(agent)
        
        # Safe code
        safe_code = """
        def calculate_sum(numbers):
            return sum(numbers)
        """
        result1 = nethical.evaluate("codegen-001", safe_code)
        assert result1.decision in ["ALLOW", "RESTRICT"]
        
        # Potentially unsafe code
        unsafe_code = """
        import os
        os.system('rm -rf /')
        """
        result2 = nethical.evaluate("codegen-001", unsafe_code)
        assert result2.risk_score > result1.risk_score
    
    def test_data_access_patterns(self):
        """Test evaluation of data access patterns."""
        nethical = Nethical(enable_25_laws=True)
        agent = Agent(id="data-agent", type="data", capabilities=["data_access"])
        nethical.register_agent(agent)
        
        # Read-only access
        read_action = "SELECT * FROM users WHERE id = 123"
        result1 = nethical.evaluate("data-agent", read_action)
        
        # Write access
        write_action = "UPDATE users SET password = 'new' WHERE id = 123"
        result2 = nethical.evaluate("data-agent", write_action)
        
        # Delete access
        delete_action = "DELETE FROM users WHERE age < 18"
        result3 = nethical.evaluate("data-agent", delete_action)
        
        # Risk should increase: read < write < delete
        assert result1.risk_score <= result2.risk_score <= result3.risk_score
    
    def test_multi_primitive_detection(self):
        """Test detection of multiple primitives in complex actions."""
        nethical = Nethical(enable_25_laws=True)
        agent = Agent(id="complex-agent", type="autonomous", capabilities=["all"])
        nethical.register_agent(agent)
        
        complex_action = """
        1. Access user database
        2. Execute data transformation code
        3. Update model with new data
        4. Generate report for user
        """
        
        result = nethical.evaluate("complex-agent", complex_action)
        
        # Should detect multiple primitives
        assert len(result.detected_primitives) > 1
        # Should evaluate multiple laws
        assert len(result.laws_evaluated) > 2
        # Complex actions typically have moderate to high risk
        assert result.risk_score > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
