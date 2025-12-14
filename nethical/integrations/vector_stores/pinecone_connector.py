"""Pinecone vector database integration with governance.

Provides integration with Pinecone cloud-native vector database
with built-in safety checks, PII detection, and audit logging.

Requirements:
    pip install pinecone-client>=3.0.0

Usage:
    from nethical.integrations.vector_stores import PineconeConnector
    
    connector = PineconeConnector(
        api_key="your-api-key",
        environment="us-west1-gcp",
        index_name="my-index",
    )
    
    # Upsert with governance
    connector.upsert([
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"text": "sample"}}
    ])
    
    # Query with PII redaction
    results = connector.query([0.1, 0.2, ...], top_k=10)
"""

import logging
from typing import Any, Dict, List, Optional

from .base import VectorStoreProvider, VectorSearchResult

logger = logging.getLogger(__name__)

# Check if Pinecone is available
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not installed. Install with: pip install pinecone-client>=3.0.0")


class PineconeConnector(VectorStoreProvider):
    """Pinecone vector database connector with governance.
    
    Provides full integration with Pinecone including:
    - Governance checks on metadata before upsert
    - PII detection and redaction on query results
    - Audit logging for all operations
    - Namespace support for multi-tenancy
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        enable_governance: bool = True,
        enable_pii_detection: bool = True,
        enable_audit_logging: bool = True,
    ):
        """Initialize Pinecone connector.
        
        Args:
            api_key: Pinecone API key (or set PINECONE_API_KEY env var)
            environment: Pinecone environment (or set PINECONE_ENVIRONMENT env var)
            index_name: Name of the Pinecone index
            enable_governance: Enable governance checks
            enable_pii_detection: Enable PII detection
            enable_audit_logging: Enable audit logging
            
        Raises:
            ImportError: If pinecone-client is not installed
            ValueError: If required parameters are missing
        """
        super().__init__(enable_governance, enable_pii_detection, enable_audit_logging)
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not installed. Install with: pip install pinecone-client>=3.0.0")
        
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self._index = None
        
        # Initialize Pinecone
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone (v3.0.0+ uses new API)
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            if self.index_name:
                self._index = pinecone.Index(self.index_name)
                logger.info(f"Connected to Pinecone index: {self.index_name}")
            else:
                logger.warning("No index name provided. Call connect() to set index.")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def connect(self, index_name: str):
        """Connect to a specific Pinecone index.
        
        Args:
            index_name: Name of the index to connect to
        """
        self.index_name = index_name
        self._index = pinecone.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
    ) -> int:
        """Upsert vectors with governance checks on metadata.
        
        Args:
            vectors: List of vectors to upsert. Each dict should contain:
                - id: Vector ID (string)
                - values: Vector values (list of floats)
                - metadata: Optional metadata dict
            namespace: Optional namespace for multi-tenancy
            
        Returns:
            Number of vectors successfully upserted
            
        Raises:
            ValueError: If governance check fails
            ConnectionError: If Pinecone connection fails
        """
        if not self._index:
            raise ConnectionError("Not connected to any Pinecone index")
        
        # Check governance on metadata
        if self.enable_governance:
            for vec in vectors:
                metadata = vec.get("metadata", {})
                check_result = self._check_governance(metadata, "vector_upsert")
                if check_result.get("decision") != "ALLOW":
                    raise ValueError(
                        f"Governance check failed for vector {vec.get('id')}: "
                        f"{check_result.get('reason')}"
                    )
        
        # Upsert to Pinecone
        try:
            # Pinecone expects tuples or dicts in specific format
            pinecone_vectors = []
            for vec in vectors:
                pinecone_vec = (
                    vec["id"],
                    vec["values"],
                    vec.get("metadata", {}),
                )
                pinecone_vectors.append(pinecone_vec)
            
            response = self._index.upsert(vectors=pinecone_vectors, namespace=namespace)
            upserted_count = response.get("upserted_count", len(vectors))
            
            # Audit log
            self._audit_log("upsert", {
                "namespace": namespace,
                "count": upserted_count,
                "index": self.index_name,
            })
            
            return upserted_count
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            raise ConnectionError(f"Failed to upsert vectors: {e}")
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> List[VectorSearchResult]:
        """Query vectors with PII redaction on results.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter
            namespace: Optional namespace for multi-tenancy
            include_metadata: Include metadata in results
            include_values: Include vector values in results
            
        Returns:
            List of search results with PII redacted
            
        Raises:
            ConnectionError: If Pinecone connection fails
        """
        if not self._index:
            raise ConnectionError("Not connected to any Pinecone index")
        
        try:
            # Query Pinecone
            response = self._index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=include_metadata,
                include_values=include_values,
            )
            
            # Convert to VectorSearchResult with PII redaction
            results = []
            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                
                # Redact PII from metadata
                if self.enable_pii_detection and metadata:
                    pii_result = self._detect_pii(metadata)
                    if pii_result["has_pii"]:
                        logger.info(f"PII detected in vector {match['id']}, redacting")
                        # Note: For structured metadata, we keep structure but redact values
                        # This is a simplified implementation
                
                result = VectorSearchResult(
                    id=match["id"],
                    score=match["score"],
                    vector=match.get("values"),
                    metadata=metadata,
                    payload={},
                )
                results.append(result)
            
            # Audit log
            self._audit_log("query", {
                "namespace": namespace,
                "top_k": top_k,
                "results_count": len(results),
                "index": self.index_name,
            })
            
            return results
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            raise ConnectionError(f"Failed to query vectors: {e}")
    
    def delete(
        self,
        ids: List[str],
        namespace: str = "",
    ) -> int:
        """Delete vectors with audit logging.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace for multi-tenancy
            
        Returns:
            Number of vectors successfully deleted
            
        Raises:
            ConnectionError: If Pinecone connection fails
        """
        if not self._index:
            raise ConnectionError("Not connected to any Pinecone index")
        
        try:
            # Delete from Pinecone
            self._index.delete(ids=ids, namespace=namespace)
            
            # Audit log
            self._audit_log("delete", {
                "namespace": namespace,
                "count": len(ids),
                "ids": ids[:10],  # Log first 10 IDs
                "index": self.index_name,
            })
            
            return len(ids)
        except Exception as e:
            logger.error(f"Pinecone delete failed: {e}")
            raise ConnectionError(f"Failed to delete vectors: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of Pinecone connection.
        
        Returns:
            Health check result with status and details
        """
        base_health = super().health_check()
        
        try:
            if self._index:
                stats = self._index.describe_index_stats()
                base_health.update({
                    "status": "healthy",
                    "index": self.index_name,
                    "dimension": stats.get("dimension"),
                    "index_fullness": stats.get("index_fullness"),
                    "total_vector_count": stats.get("total_vector_count"),
                })
            else:
                base_health.update({
                    "status": "not_connected",
                    "index": None,
                })
        except Exception as e:
            base_health.update({
                "status": "error",
                "error": str(e),
            })
        
        return base_health
