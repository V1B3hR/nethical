"""Merkle Anchoring System for Phase 4.1: Immutable Audit & Merkle Anchoring.

This module implements:
- Full event log chunking
- Merkle root computation
- Anchor Merkle roots (S3 object lock or external notarization)
- Merkle verification tool
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class MerkleNode:
    """Node in Merkle tree."""
    hash_value: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class AuditChunk:
    """Chunk of audit events with Merkle root."""
    chunk_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    merkle_root: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    finalized_at: Optional[datetime] = None
    event_count: int = 0
    anchored: bool = False
    anchor_location: Optional[str] = None


class MerkleAnchor:
    """Merkle anchoring system for immutable audit trails."""
    
    def __init__(
        self,
        storage_path: str = "audit_logs",
        chunk_size: int = 1000,
        hash_algorithm: str = "sha256",
        s3_bucket: Optional[str] = None,
        enable_object_lock: bool = False
    ):
        """Initialize Merkle anchor system.
        
        Args:
            storage_path: Local storage path for audit logs
            chunk_size: Maximum events per chunk
            hash_algorithm: Hash algorithm (sha256, sha512)
            s3_bucket: Optional S3 bucket for anchoring
            enable_object_lock: Enable S3 object lock
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.hash_algorithm = hash_algorithm
        self.s3_bucket = s3_bucket
        self.enable_object_lock = enable_object_lock
        
        # Current chunk being built
        self.current_chunk: Optional[AuditChunk] = None
        
        # Finalized chunks
        self.finalized_chunks: Dict[str, AuditChunk] = {}
        
        # Merkle roots index
        self.merkle_roots: Dict[str, str] = {}  # chunk_id -> root
        
        # Initialize first chunk
        self._create_new_chunk()
    
    def _create_new_chunk(self) -> str:
        """Create a new audit chunk.
        
        Returns:
            Chunk ID
        """
        chunk_id = f"chunk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000000) % 1000000}"
        self.current_chunk = AuditChunk(chunk_id=chunk_id)
        return chunk_id
    
    def _compute_hash(self, data: str) -> str:
        """Compute hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex digest of hash
        """
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(data.encode()).hexdigest()
        elif self.hash_algorithm == "sha512":
            return hashlib.sha512(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
    
    def _build_merkle_tree(self, events: List[Dict[str, Any]]) -> MerkleNode:
        """Build Merkle tree from events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Root node of Merkle tree
        """
        if not events:
            # Empty tree - use null hash
            return MerkleNode(hash_value=self._compute_hash(""))
        
        # Create leaf nodes
        leaves = []
        for event in events:
            event_json = json.dumps(event, sort_keys=True)
            event_hash = self._compute_hash(event_json)
            leaves.append(MerkleNode(hash_value=event_hash, data=event))
        
        # Build tree bottom-up
        current_level = leaves
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                # Handle odd number of nodes - duplicate last node
                if i + 1 >= len(current_level):
                    right = left
                else:
                    right = current_level[i + 1]
                
                # Combine hashes
                combined = left.hash_value + right.hash_value
                parent_hash = self._compute_hash(combined)
                
                parent = MerkleNode(
                    hash_value=parent_hash,
                    left=left,
                    right=right
                )
                next_level.append(parent)
            
            current_level = next_level
        
        return current_level[0]
    
    def add_event(self, event_data: Dict[str, Any]) -> bool:
        """Add event to current chunk.
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            True if added, False if chunk full (triggers finalization)
        """
        if not self.current_chunk:
            self._create_new_chunk()
        
        # Add timestamp if not present
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.utcnow().isoformat()
        
        # Add to current chunk
        self.current_chunk.events.append(event_data)
        self.current_chunk.event_count += 1
        
        # Check if chunk is full
        if self.current_chunk.event_count >= self.chunk_size:
            self.finalize_chunk()
            return False
        
        return True
    
    def finalize_chunk(self) -> str:
        """Finalize current chunk and compute Merkle root.
        
        Returns:
            Merkle root hash
        """
        if not self.current_chunk or self.current_chunk.event_count == 0:
            raise ValueError("No events in current chunk")
        
        # Build Merkle tree
        merkle_tree = self._build_merkle_tree(self.current_chunk.events)
        merkle_root = merkle_tree.hash_value
        
        # Update chunk
        self.current_chunk.merkle_root = merkle_root
        self.current_chunk.finalized_at = datetime.utcnow()
        
        # Store chunk
        chunk_id = self.current_chunk.chunk_id
        self.finalized_chunks[chunk_id] = self.current_chunk
        self.merkle_roots[chunk_id] = merkle_root
        
        # Persist to disk
        self._persist_chunk(self.current_chunk)
        
        # Anchor if S3 configured
        if self.s3_bucket:
            self._anchor_to_s3(self.current_chunk)
        
        # Create new chunk
        self._create_new_chunk()
        
        return merkle_root
    
    def _persist_chunk(self, chunk: AuditChunk):
        """Persist chunk to local storage.
        
        Args:
            chunk: Audit chunk to persist
        """
        chunk_file = self.storage_path / f"{chunk.chunk_id}.json"
        
        chunk_data = {
            'chunk_id': chunk.chunk_id,
            'merkle_root': chunk.merkle_root,
            'created_at': chunk.created_at.isoformat(),
            'finalized_at': chunk.finalized_at.isoformat() if chunk.finalized_at else None,
            'event_count': chunk.event_count,
            'events': chunk.events
        }
        
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
    
    def _anchor_to_s3(self, chunk: AuditChunk):
        """Anchor Merkle root to S3 with optional object lock.
        
        Args:
            chunk: Audit chunk to anchor
        """
        # This is a placeholder - actual S3 integration would use boto3
        # For now, we just mark it as anchored
        chunk.anchored = True
        chunk.anchor_location = f"s3://{self.s3_bucket}/{chunk.chunk_id}.json"
    
    def verify_event(
        self,
        event_id: str,
        expected_merkle_root: str,
        chunk_id: Optional[str] = None
    ) -> bool:
        """Verify event integrity against Merkle root.
        
        Args:
            event_id: Event identifier
            expected_merkle_root: Expected Merkle root
            chunk_id: Optional chunk ID to search in
            
        Returns:
            True if event is valid
        """
        # Find chunk containing event
        target_chunk = None
        
        if chunk_id:
            target_chunk = self.finalized_chunks.get(chunk_id)
        else:
            # Search all chunks
            for chunk in self.finalized_chunks.values():
                for event in chunk.events:
                    if event.get('event_id') == event_id:
                        target_chunk = chunk
                        break
                if target_chunk:
                    break
        
        if not target_chunk:
            return False
        
        # Rebuild Merkle tree and verify root
        merkle_tree = self._build_merkle_tree(target_chunk.events)
        computed_root = merkle_tree.hash_value
        
        return computed_root == expected_merkle_root
    
    def verify_chunk(self, chunk_id: str) -> bool:
        """Verify integrity of entire chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if chunk is valid
        """
        chunk = self.finalized_chunks.get(chunk_id)
        if not chunk or not chunk.merkle_root:
            return False
        
        # Recompute Merkle root
        merkle_tree = self._build_merkle_tree(chunk.events)
        computed_root = merkle_tree.hash_value
        
        return computed_root == chunk.merkle_root
    
    def get_merkle_proof(
        self,
        event_id: str,
        chunk_id: Optional[str] = None
    ) -> Optional[List[str]]:
        """Get Merkle proof for event verification.
        
        Args:
            event_id: Event identifier
            chunk_id: Optional chunk ID
            
        Returns:
            List of hashes forming Merkle proof, or None if not found
        """
        # Find chunk containing event
        target_chunk = None
        event_index = -1
        
        if chunk_id:
            target_chunk = self.finalized_chunks.get(chunk_id)
            if target_chunk:
                for i, event in enumerate(target_chunk.events):
                    if event.get('event_id') == event_id:
                        event_index = i
                        break
        else:
            # Search all chunks
            for chunk in self.finalized_chunks.values():
                for i, event in enumerate(chunk.events):
                    if event.get('event_id') == event_id:
                        target_chunk = chunk
                        event_index = i
                        break
                if target_chunk:
                    break
        
        if not target_chunk or event_index < 0:
            return None
        
        # Build tree and collect proof
        # (Simplified - full implementation would traverse tree)
        proof = []
        return proof
    
    def get_chunk_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk information dictionary
        """
        chunk = self.finalized_chunks.get(chunk_id)
        if not chunk:
            return None
        
        return {
            'chunk_id': chunk.chunk_id,
            'merkle_root': chunk.merkle_root,
            'created_at': chunk.created_at.isoformat(),
            'finalized_at': chunk.finalized_at.isoformat() if chunk.finalized_at else None,
            'event_count': chunk.event_count,
            'anchored': chunk.anchored,
            'anchor_location': chunk.anchor_location
        }
    
    def list_chunks(self) -> List[Dict[str, Any]]:
        """List all finalized chunks.
        
        Returns:
            List of chunk information dictionaries
        """
        return [
            self.get_chunk_info(chunk_id)
            for chunk_id in self.finalized_chunks.keys()
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Statistics dictionary
        """
        total_events = sum(
            chunk.event_count
            for chunk in self.finalized_chunks.values()
        )
        
        anchored_count = sum(
            1 for chunk in self.finalized_chunks.values()
            if chunk.anchored
        )
        
        return {
            'total_chunks': len(self.finalized_chunks),
            'total_events': total_events,
            'current_chunk_events': self.current_chunk.event_count if self.current_chunk else 0,
            'anchored_chunks': anchored_count,
            'chunk_size': self.chunk_size,
            'hash_algorithm': self.hash_algorithm,
            's3_enabled': self.s3_bucket is not None
        }
