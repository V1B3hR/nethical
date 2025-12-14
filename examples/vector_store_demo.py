"""
Vector Store Integration Demo

This demo shows how to use Nethical's vector store integrations with
built-in governance, PII detection, and audit logging.

Requirements:
    # Install one or more vector stores
    pip install pinecone-client>=3.0.0
    pip install weaviate-client>=4.0.0
    pip install chromadb>=0.4.0
    pip install qdrant-client>=1.7.0

Usage:
    python examples/vector_store_demo.py
"""

import logging
import sys
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_chroma():
    """Demonstrate Chroma integration with governance."""
    print("\n" + "=" * 60)
    print("Chroma Vector Store Demo")
    print("=" * 60)
    
    try:
        from nethical.integrations.vector_stores import ChromaConnector
        
        # Initialize Chroma connector (local mode)
        connector = ChromaConnector(
            collection_name="nethical_demo",
            persist_directory="./demo_chroma_data",
            enable_governance=True,
            enable_pii_detection=True,
            enable_audit_logging=True,
        )
        
        print("✓ Chroma connector initialized")
        
        # 1. Upsert vectors with governance checks
        print("\n1. Upserting vectors with governance checks...")
        
        vectors = [
            {
                "id": "doc1",
                "values": [0.1, 0.2, 0.3, 0.4] * 96,  # 384 dimensions
                "metadata": {"text": "Machine learning is transforming technology"},
                "document": "Machine learning is transforming technology",
            },
            {
                "id": "doc2",
                "values": [0.2, 0.3, 0.4, 0.5] * 96,
                "metadata": {"text": "AI safety requires careful governance"},
                "document": "AI safety requires careful governance",
            },
            {
                "id": "doc3",
                "values": [0.3, 0.4, 0.5, 0.6] * 96,
                "metadata": {"text": "Vector databases enable semantic search"},
                "document": "Vector databases enable semantic search",
            },
        ]
        
        count = connector.upsert(vectors)
        print(f"  ✓ Upserted {count} vectors")
        
        # 2. Query with PII redaction
        print("\n2. Querying vectors (results will have PII redacted)...")
        
        query_vector = [0.2, 0.3, 0.4, 0.5] * 96
        results = connector.query(query_vector, top_k=3)
        
        print(f"  ✓ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. ID: {result.id}, Score: {result.score:.4f}")
            print(f"       Metadata: {result.metadata}")
        
        # 3. Delete with audit logging
        print("\n3. Deleting vectors with audit logging...")
        
        deleted = connector.delete(["doc1"])
        print(f"  ✓ Deleted {deleted} vectors")
        
        # 4. Health check
        print("\n4. Checking connector health...")
        
        health = connector.health_check()
        print(f"  ✓ Status: {health['status']}")
        print(f"  ✓ Collection: {health.get('collection')}")
        print(f"  ✓ Vector count: {health.get('vector_count', 'N/A')}")
        
        print("\n✓ Chroma demo completed successfully")
        
    except ImportError:
        print("✗ ChromaDB not installed. Install with: pip install chromadb>=0.4.0")
    except Exception as e:
        print(f"✗ Chroma demo failed: {e}")
        logger.exception("Chroma demo error")


def demo_qdrant():
    """Demonstrate Qdrant integration with governance."""
    print("\n" + "=" * 60)
    print("Qdrant Vector Store Demo")
    print("=" * 60)
    
    try:
        from nethical.integrations.vector_stores import QdrantConnector
        
        # Initialize Qdrant connector (local mode)
        connector = QdrantConnector(
            collection_name="nethical_demo",
            path="./demo_qdrant_data",
            vector_size=384,
            enable_governance=True,
            enable_pii_detection=True,
            enable_audit_logging=True,
        )
        
        print("✓ Qdrant connector initialized")
        
        # 1. Upsert vectors
        print("\n1. Upserting vectors with payload filtering...")
        
        vectors = [
            {
                "id": "vec1",
                "values": [0.1, 0.2, 0.3, 0.4] * 96,
                "metadata": {
                    "category": "technology",
                    "text": "Artificial intelligence and machine learning",
                },
            },
            {
                "id": "vec2",
                "values": [0.2, 0.3, 0.4, 0.5] * 96,
                "metadata": {
                    "category": "security",
                    "text": "Cybersecurity and data protection",
                },
            },
        ]
        
        count = connector.upsert(vectors)
        print(f"  ✓ Upserted {count} vectors")
        
        # 2. Query with filtering
        print("\n2. Querying vectors with payload filtering...")
        
        query_vector = [0.15, 0.25, 0.35, 0.45] * 96
        results = connector.query(
            query_vector,
            top_k=5,
            filter={"category": "technology"},
        )
        
        print(f"  ✓ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. ID: {result.id}, Score: {result.score:.4f}")
            print(f"       Category: {result.metadata.get('category')}")
        
        # 3. Health check
        print("\n3. Checking connector health...")
        
        health = connector.health_check()
        print(f"  ✓ Status: {health['status']}")
        print(f"  ✓ Collection: {health.get('collection')}")
        
        print("\n✓ Qdrant demo completed successfully")
        
    except ImportError:
        print("✗ Qdrant not installed. Install with: pip install qdrant-client>=1.7.0")
    except Exception as e:
        print(f"✗ Qdrant demo failed: {e}")
        logger.exception("Qdrant demo error")


def demo_pinecone():
    """Demonstrate Pinecone integration (requires API key)."""
    print("\n" + "=" * 60)
    print("Pinecone Vector Store Demo")
    print("=" * 60)
    
    print("Note: Pinecone requires API key and remote server")
    print("Set PINECONE_API_KEY and PINECONE_ENVIRONMENT env vars")
    print("This demo is informational only without credentials")
    
    try:
        from nethical.integrations.vector_stores import PineconeConnector
        
        print("\n✓ Pinecone connector available")
        print("\nUsage example:")
        print("""
        import os
        connector = PineconeConnector(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
            index_name="nethical-demo",
        )
        
        # Upsert with governance
        vectors = [
            {"id": "vec1", "values": [...], "metadata": {...}}
        ]
        connector.upsert(vectors, namespace="production")
        
        # Query with PII redaction
        results = connector.query([...], top_k=10, namespace="production")
        """)
        
    except ImportError:
        print("✗ Pinecone not installed. Install with: pip install pinecone-client>=3.0.0")


def demo_weaviate():
    """Demonstrate Weaviate integration (requires server)."""
    print("\n" + "=" * 60)
    print("Weaviate Vector Store Demo")
    print("=" * 60)
    
    print("Note: Weaviate requires a running server")
    print("Start server with: docker run -p 8080:8080 semitechnologies/weaviate")
    print("This demo is informational only without a running server")
    
    try:
        from nethical.integrations.vector_stores import WeaviateConnector
        
        print("\n✓ Weaviate connector available")
        print("\nUsage example:")
        print("""
        connector = WeaviateConnector(
            url="http://localhost:8080",
            class_name="Document",
        )
        
        # Upsert with schema-aware governance
        vectors = [
            {"id": "doc1", "values": [...], "metadata": {"text": "..."}}
        ]
        connector.upsert(vectors)
        
        # Hybrid search (vector + keyword)
        results = connector.query([...], top_k=10, hybrid_alpha=0.7)
        """)
        
    except ImportError:
        print("✗ Weaviate not installed. Install with: pip install weaviate-client>=4.0.0")


def demo_governance_features():
    """Demonstrate governance features across vector stores."""
    print("\n" + "=" * 60)
    print("Governance Features Demo")
    print("=" * 60)
    
    print("\n✓ All vector store connectors support:")
    print("  1. Governance checks on metadata before upsert")
    print("  2. PII detection and redaction on query results")
    print("  3. Audit logging for all operations")
    print("  4. Error handling and graceful degradation")
    
    print("\n✓ Governance configuration example:")
    print("""
    connector = VectorStoreConnector(
        # ... connection params ...
        enable_governance=True,       # Check metadata for violations
        enable_pii_detection=True,    # Detect and redact PII
        enable_audit_logging=True,    # Log all operations
    )
    
    # Governance automatically applied to:
    # - upsert(): Checks metadata before insertion
    # - query(): Redacts PII from results
    # - delete(): Logs deletions for audit
    """)
    
    print("\n✓ PII Detection example:")
    print("""
    # Metadata with PII
    metadata = {
        "name": "John Doe",
        "email": "john@example.com",  # <-- PII detected
        "phone": "555-1234",           # <-- PII detected
        "text": "Machine learning content"
    }
    
    # After query with PII detection enabled:
    # - PII fields are redacted or anonymized
    # - Detection is logged for audit
    # - Original structure is preserved
    """)


def main():
    """Run all demos."""
    print("=" * 60)
    print("Nethical Vector Store Integration Demo")
    print("=" * 60)
    print("\nThis demo showcases vector database integrations with")
    print("built-in governance, PII detection, and audit logging.")
    
    # Run demos
    demo_chroma()
    demo_qdrant()
    demo_pinecone()
    demo_weaviate()
    demo_governance_features()
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - docs/VECTOR_STORE_INTEGRATION_GUIDE.md")
    print("  - config/integrations/vector-stores-mcp.yaml")
    print("  - tests/test_vector_stores.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Demo failed")
        sys.exit(1)
