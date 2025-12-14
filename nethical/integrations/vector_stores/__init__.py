"""Vector Store Integrations with Governance.

This module provides integrations with popular vector databases,
all enhanced with Nethical's safety and governance features.

Supported vector stores:
- Pinecone: Cloud-native vector database
- Weaviate: Schema-aware vector search engine
- Chroma: Lightweight vector database
- Qdrant: High-performance vector search engine

All connectors provide:
- Governance checks on vector metadata
- PII detection and redaction
- Audit logging for all operations
- Error handling and graceful degradation
"""

from .base import VectorStoreProvider, VectorSearchResult

__all__ = [
    "VectorStoreProvider",
    "VectorSearchResult",
]

# Optional imports - check availability
try:
    from .pinecone_connector import PineconeConnector
    __all__.append("PineconeConnector")
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from .weaviate_connector import WeaviateConnector
    __all__.append("WeaviateConnector")
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    from .chroma_connector import ChromaConnector
    __all__.append("ChromaConnector")
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from .qdrant_connector import QdrantConnector
    __all__.append("QdrantConnector")
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
