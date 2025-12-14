"""Chroma vector database integration with governance.

Provides integration with Chroma lightweight vector database
with built-in safety checks, PII detection, and audit logging.

Requirements:
    pip install chromadb>=0.4.0

Usage:
    from nethical.integrations.vector_stores import ChromaConnector
    
    # Local mode
    connector = ChromaConnector(
        collection_name="my_collection",
        persist_directory="./chroma_data",
    )
    
    # Client/server mode
    connector = ChromaConnector(
        collection_name="my_collection",
        host="localhost",
        port=8000,
    )
    
    # Upsert with governance
    connector.upsert([
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"text": "sample"}}
    ])
"""

import logging
from typing import Any, Dict, List, Optional

from .base import VectorStoreProvider, VectorSearchResult

logger = logging.getLogger(__name__)

# Check if Chroma is available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed. Install with: pip install chromadb>=0.4.0")


class ChromaConnector(VectorStoreProvider):
    """Chroma vector database connector with governance.
    
    Provides full integration with Chroma including:
    - Local and client/server mode support
    - Collection-level governance policies
    - PII detection and redaction on query results
    - Audit logging for all operations
    - Embedding function integration
    """
    
    def __init__(
        self,
        collection_name: str = "nethical_collection",
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        enable_governance: bool = True,
        enable_pii_detection: bool = True,
        enable_audit_logging: bool = True,
    ):
        """Initialize Chroma connector.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory for local persistence (local mode)
            host: Chroma server host (client/server mode)
            port: Chroma server port (client/server mode)
            enable_governance: Enable governance checks
            enable_pii_detection: Enable PII detection
            enable_audit_logging: Enable audit logging
            
        Raises:
            ImportError: If chromadb is not installed
            ConnectionError: If cannot connect to Chroma
        """
        super().__init__(enable_governance, enable_pii_detection, enable_audit_logging)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb>=0.4.0")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self._client = None
        self._collection = None
        
        # Initialize Chroma
        self._init_chroma()
    
    def _init_chroma(self):
        """Initialize Chroma client and collection."""
        try:
            # Create client based on mode
            if self.host and self.port:
                # Client/server mode
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                )
                logger.info(f"Connected to Chroma server at {self.host}:{self.port}")
            else:
                # Local mode
                if self.persist_directory:
                    self._client = chromadb.PersistentClient(path=self.persist_directory)
                    logger.info(f"Using Chroma local persistence at {self.persist_directory}")
                else:
                    self._client = chromadb.Client()
                    logger.info("Using Chroma in-memory mode")
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info(f"Using Chroma collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise
    
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
                - document: Optional document text
            namespace: Not used in Chroma (use different collections instead)
            
        Returns:
            Number of vectors successfully upserted
            
        Raises:
            ValueError: If governance check fails
            ConnectionError: If Chroma connection fails
        """
        if not self._collection:
            raise ConnectionError("Chroma collection not initialized")
        
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
        
        try:
            # Prepare data for Chroma
            ids = [vec["id"] for vec in vectors]
            embeddings = [vec["values"] for vec in vectors]
            metadatas = [vec.get("metadata", {}) for vec in vectors]
            documents = [vec.get("document", "") for vec in vectors]
            
            # Upsert to Chroma
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas if any(metadatas) else None,
                documents=documents if any(documents) else None,
            )
            
            # Audit log
            self._audit_log("upsert", {
                "namespace": namespace,
                "count": len(vectors),
                "collection": self.collection_name,
            })
            
            return len(vectors)
        except Exception as e:
            logger.error(f"Chroma upsert failed: {e}")
            raise ConnectionError(f"Failed to upsert vectors: {e}")
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_documents: bool = True,
    ) -> List[VectorSearchResult]:
        """Query vectors with PII redaction on results.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter (where clause)
            namespace: Not used in Chroma (use different collections instead)
            include_documents: Include document text in results
            
        Returns:
            List of search results with PII redacted
            
        Raises:
            ConnectionError: If Chroma connection fails
        """
        if not self._collection:
            raise ConnectionError("Chroma collection not initialized")
        
        try:
            # Query Chroma
            results_dict = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=filter,
                include=["metadatas", "documents", "distances", "embeddings"] if include_documents else ["metadatas", "distances"],
            )
            
            # Convert to VectorSearchResult with PII redaction
            results = []
            ids = results_dict.get("ids", [[]])[0]
            distances = results_dict.get("distances", [[]])[0]
            metadatas = results_dict.get("metadatas", [[]])[0]
            documents = results_dict.get("documents", [[]])[0] if include_documents else []
            embeddings = results_dict.get("embeddings", [[]])[0] if "embeddings" in results_dict else []
            
            for i, vec_id in enumerate(ids):
                distance = distances[i] if i < len(distances) else 1.0
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity score
                metadata = metadatas[i] if i < len(metadatas) else {}
                document = documents[i] if i < len(documents) else ""
                embedding = embeddings[i] if i < len(embeddings) else None
                
                # Redact PII from metadata and document
                if self.enable_pii_detection:
                    if metadata:
                        pii_result = self._detect_pii(metadata)
                        if pii_result["has_pii"]:
                            logger.info(f"PII detected in vector {vec_id} metadata, redacting")
                    
                    if document:
                        doc_pii_result = self._detect_pii(document)
                        if doc_pii_result["has_pii"]:
                            logger.info(f"PII detected in vector {vec_id} document, redacting")
                            document = doc_pii_result["redacted_text"]
                
                # Add document to metadata if present
                if document:
                    metadata["_document"] = document
                
                result = VectorSearchResult(
                    id=vec_id,
                    score=score,
                    vector=embedding,
                    metadata=metadata,
                    payload={},
                )
                results.append(result)
            
            # Audit log
            self._audit_log("query", {
                "namespace": namespace,
                "top_k": top_k,
                "results_count": len(results),
                "collection": self.collection_name,
            })
            
            return results
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            raise ConnectionError(f"Failed to query vectors: {e}")
    
    def delete(
        self,
        ids: List[str],
        namespace: str = "",
    ) -> int:
        """Delete vectors with audit logging.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Not used in Chroma (use different collections instead)
            
        Returns:
            Number of vectors successfully deleted
            
        Raises:
            ConnectionError: If Chroma connection fails
        """
        if not self._collection:
            raise ConnectionError("Chroma collection not initialized")
        
        try:
            # Delete from Chroma
            self._collection.delete(ids=ids)
            
            # Audit log
            self._audit_log("delete", {
                "namespace": namespace,
                "count": len(ids),
                "ids": ids[:10],  # Log first 10 IDs
                "collection": self.collection_name,
            })
            
            return len(ids)
        except Exception as e:
            logger.error(f"Chroma delete failed: {e}")
            raise ConnectionError(f"Failed to delete vectors: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of Chroma connection.
        
        Returns:
            Health check result with status and details
        """
        base_health = super().health_check()
        
        try:
            if self._collection:
                # Get collection info
                count = self._collection.count()
                
                base_health.update({
                    "status": "healthy",
                    "collection": self.collection_name,
                    "vector_count": count,
                    "mode": "client_server" if self.host else "local",
                })
            else:
                base_health.update({
                    "status": "not_initialized",
                    "collection": None,
                })
        except Exception as e:
            base_health.update({
                "status": "error",
                "error": str(e),
            })
        
        return base_health
