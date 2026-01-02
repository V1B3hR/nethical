# Vector Store Integration Guide

Nethical provides production-ready integrations with popular vector databases, all enhanced with built-in safety governance, PII detection, and audit logging.

## Supported Vector Stores

- **Pinecone**: Cloud-native vector database with namespace support
- **Weaviate**: Schema-aware vector search engine with hybrid search
- **Chroma**: Lightweight vector database for local and client/server modes
- **Qdrant**: High-performance vector search engine with advanced filtering

## Core Features

All vector store connectors provide:

✅ **Governance Checks**: Validate metadata against policies before insertion  
✅ **PII Detection**: Automatically detect and redact PII in query results  
✅ **Audit Logging**: Track all operations for compliance and debugging  
✅ **Error Handling**: Graceful degradation when dependencies are unavailable  
✅ **Health Monitoring**: Connection and performance health checks

## Installation

### Pinecone

```bash
pip install pinecone-client>=3.0.0
```

### Weaviate

```bash
pip install weaviate-client>=4.0.0
```

### Chroma

```bash
pip install chromadb>=0.4.0
```

### Qdrant

```bash
pip install qdrant-client>=1.7.0
```

## Quick Start

### Chroma (Local Mode)

```python
from nethical.integrations.vector_stores import ChromaConnector

# Initialize connector
connector = ChromaConnector(
    collection_name="my_collection",
    persist_directory="./chroma_data",
    enable_governance=True,
    enable_pii_detection=True,
    enable_audit_logging=True,
)

# Upsert vectors with governance
vectors = [
    {
        "id": "doc1",
        "values": [0.1, 0.2, 0.3, ...],  # Your embedding
        "metadata": {"text": "Sample document"},
        "document": "Sample document content",
    }
]
connector.upsert(vectors)

# Query with PII redaction
results = connector.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={"category": "tech"},
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

### Qdrant (Local Mode)

```python
from nethical.integrations.vector_stores import QdrantConnector

connector = QdrantConnector(
    collection_name="my_collection",
    path="./qdrant_data",
    vector_size=384,  # Your embedding dimension
    enable_governance=True,
)

# Upsert with payload
vectors = [
    {
        "id": "vec1",
        "values": [0.1, 0.2, ...],
        "metadata": {
            "category": "technology",
            "text": "AI and ML content",
        },
    }
]
connector.upsert(vectors)

# Query with filtering
results = connector.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"category": "technology"},
    score_threshold=0.7,
)
```

### Pinecone (Cloud)

```python
from nethical.integrations.vector_stores import PineconeConnector
import os

connector = PineconeConnector(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
    index_name="my-index",
    enable_governance=True,
    enable_pii_detection=True,
)

# Upsert with namespace
vectors = [
    {
        "id": "vec1",
        "values": [0.1, 0.2, ...],
        "metadata": {"text": "Sample"},
    }
]
connector.upsert(vectors, namespace="production")

# Query with namespace
results = connector.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    namespace="production",
)
```

### Weaviate (Client/Server)

```python
from nethical.integrations.vector_stores import WeaviateConnector

connector = WeaviateConnector(
    url="http://localhost:8080",
    class_name="Document",
    enable_governance=True,
)

# Upsert objects
vectors = [
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],
        "metadata": {
            "text": "Sample",
            "category": "tech",
        },
    }
]
connector.upsert(vectors)

# Hybrid search (vector + keyword)
results = connector.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    hybrid_alpha=0.7,  # 0=keyword, 1=vector
)
```

## Governance Integration

### Enabling Governance

```python
connector = VectorStoreConnector(
    # ... connection params ...
    enable_governance=True,       # Check metadata for policy violations
    enable_pii_detection=True,    # Detect and redact PII
    enable_audit_logging=True,    # Log all operations
)
```

### Governance on Upsert

Governance checks are automatically applied to metadata before insertion:

```python
# This will be checked against governance policies
vectors = [
    {
        "id": "vec1",
        "values": [...],
        "metadata": {
            "text": "Content to check",
            "source": "user_input",
            "category": "general",
        },
    }
]

try:
    connector.upsert(vectors)
except ValueError as e:
    print(f"Governance check failed: {e}")
    # Handle policy violation
```

### PII Detection on Query

PII is automatically detected and logged when querying:

```python
# Query results will have PII detected and flagged
results = connector.query([...], top_k=10)

for result in results:
    # PII in metadata is logged and can be redacted
    print(result.metadata)  # PII-aware
```

## Best Practices

### 1. PII Handling in Vector Data

**Recommendation**: Never store raw PII in vector metadata

```python
# ❌ BAD: Storing PII directly
metadata = {
    "email": "user@example.com",  # PII!
    "phone": "555-1234",           # PII!
    "text": "Document content",
}

# ✅ GOOD: Use anonymized identifiers
metadata = {
    "user_id_hash": "abc123...",   # Hashed ID
    "doc_type": "email",
    "text": "Document content",
}
```

### 2. Namespace/Tenant Isolation

Use namespaces to isolate data:

```python
# Pinecone supports namespaces
connector.upsert(vectors, namespace="customer_a")
connector.query([...], namespace="customer_a")

# For Chroma/Qdrant, use separate collections
connector_a = ChromaConnector(collection_name="customer_a")
connector_b = ChromaConnector(collection_name="customer_b")
```

### 3. Error Handling

Always handle errors gracefully:

```python
try:
    count = connector.upsert(vectors)
    logger.info(f"Upserted {count} vectors")
except ValueError as e:
    logger.error(f"Governance check failed: {e}")
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    # Implement retry logic or fallback
```

### 4. Health Monitoring

Check connector health regularly:

```python
health = connector.health_check()

if health["status"] != "healthy":
    logger.warning(f"Connector unhealthy: {health}")
    # Alert or failover
```

### 5. Batch Operations

Use batch operations for efficiency:

```python
# Batch upsert
batch_size = 100
for i in range(0, len(all_vectors), batch_size):
    batch = all_vectors[i:i+batch_size]
    connector.upsert(batch)
```

## Advanced Usage

### Custom Governance Policies

Governance is integrated with Nethical's `IntegratedGovernance`:

```python
from nethical.core.integrated_governance import IntegratedGovernance

# The connector automatically uses IntegratedGovernance
# To customize, you can configure it before initializing connectors

governance = IntegratedGovernance(
    storage_dir="./custom_data",
    enable_performance_optimization=True,
)

# Then use connectors as usual
connector = ChromaConnector(enable_governance=True)
```

### Multi-Vector Store Setup

Use multiple vector stores for different purposes:

```python
# Fast local cache
local_cache = ChromaConnector(
    collection_name="cache",
    enable_governance=False,  # Fast path
)

# Production store with full governance
production_store = PineconeConnector(
    api_key=api_key,
    environment=environment,
    index_name="production",
    enable_governance=True,
    enable_pii_detection=True,
)

# Check cache first, then production
results = local_cache.query(vector, top_k=10)
if not results:
    results = production_store.query(vector, top_k=10)
```

### Custom PII Detection

PII detection integrates with Nethical's `PIIDetector`:

```python
from nethical.utils.pii import get_pii_detector

# The connector automatically uses PIIDetector
# Customize detection patterns if needed
pii_detector = get_pii_detector()

# Use with connector
connector = WeaviateConnector(enable_pii_detection=True)
```

## Troubleshooting

### Import Errors

If you get import errors:

```python
# Check if module is available
try:
    from nethical.integrations.vector_stores import PineconeConnector
    print("Pinecone available")
except ImportError:
    print("Install: pip install pinecone-client>=3.0.0")
```

### Connection Failures

For connection issues:

```python
# Test connection with health check
try:
    health = connector.health_check()
    print(f"Status: {health['status']}")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Governance Failures

If governance checks are failing:

```python
# Disable governance temporarily for debugging
connector = ChromaConnector(enable_governance=False)

# Or catch and log governance failures
try:
    connector.upsert(vectors)
except ValueError as e:
    logger.error(f"Governance: {e}")
    # Inspect which metadata failed
```

## Performance Considerations

### Vector Dimensions

- Pinecone: Supports up to 20,000 dimensions
- Weaviate: Configurable, typically 384-1536
- Chroma: Flexible
- Qdrant: Configurable, specify with `vector_size`

### Batch Sizes

Recommended batch sizes:
- Upsert: 100-1000 vectors per batch
- Query: 10-100 results (top_k)
- Delete: 100-1000 IDs per batch

### Governance Overhead

Governance adds minimal overhead (~1-5ms per operation):
- Metadata validation: <1ms
- PII detection: 1-3ms
- Audit logging: <1ms

For high-throughput scenarios, consider:
```python
# Fast path without governance
fast_connector = ChromaConnector(
    collection_name="high_throughput",
    enable_governance=False,
    enable_pii_detection=False,
)
```

## Configuration

See `config/integrations/vector-stores-mcp.yaml` for complete configuration options.

## Examples

- `examples/vector_store_demo.py` - Comprehensive demo
- `tests/test_vector_stores.py` - Test examples

## Support

For issues or questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/blob/main/docs/
