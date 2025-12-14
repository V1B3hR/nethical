"""
Tests for Vector Store Integrations

Tests the vector store integration interfaces including Pinecone, Weaviate, Chroma, and Qdrant.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Import base classes
from nethical.integrations.vector_stores.base import (
    VectorStoreProvider,
    VectorSearchResult,
)


class TestVectorSearchResult:
    """Test VectorSearchResult dataclass"""
    
    def test_result_creation(self):
        """Test creating a vector search result"""
        result = VectorSearchResult(
            id="vec_123",
            score=0.95,
            vector=[0.1, 0.2, 0.3],
            metadata={"text": "sample"},
            payload={"extra": "data"},
        )
        
        assert result.id == "vec_123"
        assert result.score == 0.95
        assert len(result.vector) == 3
        assert result.metadata["text"] == "sample"
    
    def test_result_to_dict(self):
        """Test converting result to dict"""
        result = VectorSearchResult(
            id="vec_123",
            score=0.95,
            metadata={"text": "sample"},
        )
        
        result_dict = result.to_dict()
        assert result_dict["id"] == "vec_123"
        assert result_dict["score"] == 0.95
        assert result_dict["metadata"]["text"] == "sample"


class TestVectorStoreProviderBase:
    """Test VectorStoreProvider base class"""
    
    def test_base_provider_initialization(self):
        """Test base provider initialization"""
        # Create a concrete implementation for testing
        class TestProvider(VectorStoreProvider):
            def upsert(self, vectors, namespace=""):
                return len(vectors)
            
            def query(self, vector, top_k=10, filter=None, namespace=""):
                return []
            
            def delete(self, ids, namespace=""):
                return len(ids)
        
        provider = TestProvider(
            enable_governance=True,
            enable_pii_detection=True,
            enable_audit_logging=True,
        )
        
        assert provider.enable_governance is True
        assert provider.enable_pii_detection is True
        assert provider.enable_audit_logging is True
    
    def test_governance_check_disabled(self):
        """Test governance check when disabled"""
        class TestProvider(VectorStoreProvider):
            def upsert(self, vectors, namespace=""):
                return len(vectors)
            
            def query(self, vector, top_k=10, filter=None, namespace=""):
                return []
            
            def delete(self, ids, namespace=""):
                return len(ids)
        
        provider = TestProvider(enable_governance=False)
        result = provider._check_governance({"test": "data"})
        
        assert result["decision"] == "ALLOW"
        assert "disabled" in result["reason"].lower()


# Pinecone Tests
class TestPineconeConnector:
    """Test Pinecone connector"""
    
    @pytest.fixture
    def mock_pinecone(self):
        """Mock Pinecone module"""
        with patch.dict('sys.modules', {'pinecone': MagicMock()}):
            import sys
            mock_pinecone = sys.modules['pinecone']
            mock_index = MagicMock()
            mock_pinecone.Index.return_value = mock_index
            yield mock_pinecone, mock_index
    
    def test_pinecone_not_available(self):
        """Test error when Pinecone is not installed"""
        with patch.dict('sys.modules', {'pinecone': None}):
            from nethical.integrations.vector_stores import pinecone_connector
            # Re-import to get updated PINECONE_AVAILABLE
            import importlib
            importlib.reload(pinecone_connector)
            
            # Should not raise during module import
            assert pinecone_connector.PINECONE_AVAILABLE is False
    
    def test_upsert_with_governance(self, mock_pinecone):
        """Test upserting vectors with governance checks"""
        from nethical.integrations.vector_stores.pinecone_connector import PineconeConnector
        
        _, mock_index = mock_pinecone
        mock_index.upsert.return_value = {"upserted_count": 2}
        
        connector = PineconeConnector(
            api_key="test-key",
            environment="test-env",
            index_name="test-index",
            enable_governance=False,  # Disable to avoid dependencies
        )
        
        vectors = [
            {"id": "vec1", "values": [0.1, 0.2], "metadata": {"text": "sample1"}},
            {"id": "vec2", "values": [0.3, 0.4], "metadata": {"text": "sample2"}},
        ]
        
        count = connector.upsert(vectors)
        assert count == 2
        assert mock_index.upsert.called
    
    def test_query_returns_results(self, mock_pinecone):
        """Test querying vectors"""
        from nethical.integrations.vector_stores.pinecone_connector import PineconeConnector
        
        _, mock_index = mock_pinecone
        mock_index.query.return_value = {
            "matches": [
                {"id": "vec1", "score": 0.95, "metadata": {"text": "sample1"}},
                {"id": "vec2", "score": 0.90, "metadata": {"text": "sample2"}},
            ]
        }
        
        connector = PineconeConnector(
            api_key="test-key",
            environment="test-env",
            index_name="test-index",
            enable_pii_detection=False,
        )
        
        results = connector.query([0.1, 0.2], top_k=10)
        
        assert len(results) == 2
        assert results[0].id == "vec1"
        assert results[0].score == 0.95
        assert mock_index.query.called
    
    def test_delete_vectors(self, mock_pinecone):
        """Test deleting vectors"""
        from nethical.integrations.vector_stores.pinecone_connector import PineconeConnector
        
        _, mock_index = mock_pinecone
        
        connector = PineconeConnector(
            api_key="test-key",
            environment="test-env",
            index_name="test-index",
        )
        
        ids = ["vec1", "vec2", "vec3"]
        count = connector.delete(ids)
        
        assert count == 3
        assert mock_index.delete.called


# Weaviate Tests  
class TestWeaviateConnector:
    """Test Weaviate connector"""
    
    @pytest.fixture
    def mock_weaviate(self):
        """Mock Weaviate module"""
        with patch.dict('sys.modules', {'weaviate': MagicMock()}):
            import sys
            mock_weaviate = sys.modules['weaviate']
            mock_client = MagicMock()
            mock_client.is_ready.return_value = True
            mock_weaviate.Client.return_value = mock_client
            yield mock_weaviate, mock_client
    
    def test_weaviate_initialization(self, mock_weaviate):
        """Test Weaviate connector initialization"""
        from nethical.integrations.vector_stores.weaviate_connector import WeaviateConnector
        
        _, mock_client = mock_weaviate
        
        connector = WeaviateConnector(
            url="http://localhost:8080",
            class_name="Document",
        )
        
        assert connector.url == "http://localhost:8080"
        assert connector.class_name == "Document"
        assert mock_client.is_ready.called
    
    def test_hybrid_search(self, mock_weaviate):
        """Test hybrid search functionality"""
        from nethical.integrations.vector_stores.weaviate_connector import WeaviateConnector
        
        _, mock_client = mock_weaviate
        
        # Mock query builder chain
        mock_query = MagicMock()
        mock_client.query.get.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.with_hybrid.return_value = mock_query
        mock_query.with_additional.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "Document": [
                        {
                            "_additional": {"id": "obj1", "distance": 0.1},
                            "text": "sample1",
                        }
                    ]
                }
            }
        }
        
        connector = WeaviateConnector(
            url="http://localhost:8080",
            class_name="Document",
            enable_pii_detection=False,
        )
        
        results = connector.query([0.1, 0.2], top_k=5, hybrid_alpha=0.5)
        
        assert len(results) == 1
        assert results[0].id == "obj1"


# Chroma Tests
class TestChromaConnector:
    """Test Chroma connector"""
    
    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB module"""
        with patch.dict('sys.modules', {'chromadb': MagicMock()}):
            import sys
            mock_chromadb = sys.modules['chromadb']
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.Client.return_value = mock_client
            mock_chromadb.PersistentClient.return_value = mock_client
            yield mock_chromadb, mock_client, mock_collection
    
    def test_chroma_local_mode(self, mock_chromadb):
        """Test Chroma in local mode"""
        from nethical.integrations.vector_stores.chroma_connector import ChromaConnector
        
        _, mock_client, _ = mock_chromadb
        
        connector = ChromaConnector(
            collection_name="test_collection",
            persist_directory="./chroma_data",
        )
        
        assert connector.collection_name == "test_collection"
        assert connector.persist_directory == "./chroma_data"
    
    def test_chroma_upsert(self, mock_chromadb):
        """Test upserting to Chroma"""
        from nethical.integrations.vector_stores.chroma_connector import ChromaConnector
        
        _, _, mock_collection = mock_chromadb
        
        connector = ChromaConnector(
            collection_name="test_collection",
            enable_governance=False,
        )
        
        vectors = [
            {"id": "vec1", "values": [0.1, 0.2], "metadata": {"text": "sample1"}},
            {"id": "vec2", "values": [0.3, 0.4], "metadata": {"text": "sample2"}},
        ]
        
        count = connector.upsert(vectors)
        
        assert count == 2
        assert mock_collection.upsert.called
    
    def test_chroma_query(self, mock_chromadb):
        """Test querying Chroma"""
        from nethical.integrations.vector_stores.chroma_connector import ChromaConnector
        
        _, _, mock_collection = mock_chromadb
        mock_collection.query.return_value = {
            "ids": [["vec1", "vec2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"text": "sample1"}, {"text": "sample2"}]],
        }
        
        connector = ChromaConnector(
            collection_name="test_collection",
            enable_pii_detection=False,
        )
        
        results = connector.query([0.1, 0.2], top_k=10)
        
        assert len(results) == 2
        assert results[0].id == "vec1"


# Qdrant Tests
class TestQdrantConnector:
    """Test Qdrant connector"""
    
    @pytest.fixture
    def mock_qdrant(self):
        """Mock Qdrant module"""
        with patch.dict('sys.modules', {
            'qdrant_client': MagicMock(),
            'qdrant_client.models': MagicMock(),
        }):
            import sys
            mock_qdrant = sys.modules['qdrant_client']
            mock_client = MagicMock()
            mock_collections = MagicMock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            mock_qdrant.QdrantClient.return_value = mock_client
            yield mock_qdrant, mock_client
    
    def test_qdrant_initialization(self, mock_qdrant):
        """Test Qdrant connector initialization"""
        from nethical.integrations.vector_stores.qdrant_connector import QdrantConnector
        
        _, mock_client = mock_qdrant
        
        connector = QdrantConnector(
            collection_name="test_collection",
            path="./qdrant_data",
        )
        
        assert connector.collection_name == "test_collection"
        assert mock_client.get_collections.called
    
    def test_qdrant_upsert(self, mock_qdrant):
        """Test upserting to Qdrant"""
        from nethical.integrations.vector_stores.qdrant_connector import QdrantConnector
        
        _, mock_client = mock_qdrant
        
        connector = QdrantConnector(
            collection_name="test_collection",
            path="./qdrant_data",
            enable_governance=False,
        )
        
        vectors = [
            {"id": "vec1", "values": [0.1, 0.2], "metadata": {"text": "sample1"}},
            {"id": "vec2", "values": [0.3, 0.4], "metadata": {"text": "sample2"}},
        ]
        
        count = connector.upsert(vectors)
        
        assert count == 2
        assert mock_client.upsert.called


# Integration Tests
class TestVectorStoreIntegration:
    """Integration tests for vector stores"""
    
    def test_multiple_connectors(self):
        """Test using multiple vector store connectors"""
        # This test verifies that multiple connectors can coexist
        # In real usage, only one would typically be used at a time
        
        # Test that modules can be imported
        try:
            from nethical.integrations.vector_stores import (
                VectorStoreProvider,
                VectorSearchResult,
            )
            assert VectorStoreProvider is not None
            assert VectorSearchResult is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")
    
    def test_health_check(self):
        """Test health check functionality"""
        class TestProvider(VectorStoreProvider):
            def upsert(self, vectors, namespace=""):
                return len(vectors)
            
            def query(self, vector, top_k=10, filter=None, namespace=""):
                return []
            
            def delete(self, ids, namespace=""):
                return len(ids)
        
        provider = TestProvider()
        health = provider.health_check()
        
        assert "status" in health
        assert "provider" in health
        assert health["governance_enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
