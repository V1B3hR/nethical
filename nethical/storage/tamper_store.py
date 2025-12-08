from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Any, Optional, Tuple, List, Iterable, Literal


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _digest(algorithm: str, data: bytes) -> bytes:
    h = hashlib.new(algorithm)
    h.update(data)
    return h.digest()


def _hex(b: bytes) -> str:
    return b.hex()


def _canonical_json_bytes(obj: Any) -> bytes:
    # Deterministic JSON: sort keys, compact separators, stable UTF-8
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class TamperStoreError(Exception):
    pass


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    ts_iso: str
    leaf: str
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    prev_root: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "type": "event",
            "seq": self.seq,
            "ts": self.ts,
            "ts_iso": self.ts_iso,
            "leaf": self.leaf,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "prev_root": self.prev_root,
        }


@dataclass(frozen=True)
class Anchor:
    type: Literal["tsa", "file", "custom"]
    root: str
    ts: float
    ts_iso: str
    url: Optional[str] = None
    receipt: Optional[str] = None  # opaque receipt/assertion (e.g., RFC3161 token)

    def to_record(self) -> Dict[str, Any]:
        return {
            "type": "anchor",
            "anchor_type": self.type,
            "root": self.root,
            "ts": self.ts,
            "ts_iso": self.ts_iso,
            "url": self.url,
            "receipt": self.receipt,
        }


class MerkleAppender:
    """
    Simple append-only Merkle tree with inclusion proof generation.
    Leaves and internal nodes are hex-encoded digests.
    Node hash: H(left_bytes || right_bytes)
    """

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self._leaves: List[str] = []

    @property
    def size(self) -> int:
        return len(self._leaves)

    @property
    def leaves(self) -> List[str]:
        return self._leaves

    def add_leaf(self, data: bytes) -> str:
        leaf = _hex(_digest(self.algorithm, data))
        self._leaves.append(leaf)
        return leaf

    def root(self) -> Optional[str]:
        nodes = self._leaves[:]
        if not nodes:
            return None
        while len(nodes) > 1:
            nxt: List[str] = []
            # pairwise combine, duplicate last if odd length
            for i in range(0, len(nodes), 2):
                a = nodes[i]
                b = nodes[i + 1] if i + 1 < len(nodes) else a
                nxt.append(
                    _hex(_digest(self.algorithm, bytes.fromhex(a) + bytes.fromhex(b)))
                )
            nodes = nxt
        return nodes[0]

    def _level_nodes(self, nodes: List[str]) -> List[str]:
        if len(nodes) == 1:
            return nodes
        nxt: List[str] = []
        for i in range(0, len(nodes), 2):
            a = nodes[i]
            b = nodes[i + 1] if i + 1 < len(nodes) else a
            nxt.append(
                _hex(_digest(self.algorithm, bytes.fromhex(a) + bytes.fromhex(b)))
            )
        return nxt

    def prove(self, index: int) -> List[Dict[str, Any]]:
        """
        Build an inclusion proof for leaf at index (0-based).
        Proof is a list of dicts: {"hash": hex, "left": bool}
        where "left" indicates the sibling is the left node at that level.
        """
        if index < 0 or index >= len(self._leaves):
            raise TamperStoreError("Leaf index out of range")

        nodes = self._leaves[:]
        i = index
        proof: List[Dict[str, Any]] = []

        while len(nodes) > 1:
            # sibling within this level
            if i % 2 == 0:
                # right sibling if exists else self
                sib_index = i + 1 if i + 1 < len(nodes) else i
                sib_left = False
            else:
                sib_index = i - 1
                sib_left = True

            sibling_hash = nodes[sib_index]
            if sib_index == i:
                # odd duplication: explicitly mark as right duplicate for determinism
                sib_left = False

            proof.append({"hash": sibling_hash, "left": sib_left})

            # build next level and move index up
            nodes = self._level_nodes(nodes)
            i //= 2

        return proof

    def verify(
        self,
        leaf: str,
        proof: Iterable[Dict[str, Any]],
        expected_root: Optional[str] = None,
    ) -> str:
        """
        Verify a proof against an optional expected_root.
        Returns the computed root hex. Raises on mismatch if expected_root provided.
        """
        acc = leaf
        for step in proof:
            sib = step["hash"]
            left = bool(step["left"])
            if left:
                acc = _hex(
                    _digest(self.algorithm, bytes.fromhex(sib) + bytes.fromhex(acc))
                )
            else:
                acc = _hex(
                    _digest(self.algorithm, bytes.fromhex(acc) + bytes.fromhex(sib))
                )
        if expected_root is not None and acc != expected_root:
            raise TamperStoreError("Proof verification failed: root mismatch")
        return acc


class TamperEvidentOfflineStore:
    """
    Append-only tamper-evident offline store with Merkle roots, inclusion proofs,
    optional anchoring records, deterministic JSON hashing, sequencing, and
    import/export for durability.

    Notes:
    - Payloads are hashed deterministically with canonical JSON.
    - Each event references the previous Merkle root (prev_root) to strengthen
      chain-of-custody semantics.
    - Anchors are just recorded metadata (e.g., for RFC3161 TSA). No network I/O.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        tsa_url: Optional[str] = None,
        digest: str = "sha256",
    ):
        self.tsa_url = tsa_url
        self._digest = digest
        self._merkle = MerkleAppender(algorithm=digest)
        self._events: List[Event] = []
        self._anchors: List[Anchor] = []
        self._lock = RLock()

    # --------------- Core API ---------------

    def append_event(
        self,
        payload: Dict[str, Any],
        *,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Append a payload as an event. Returns the hex leaf hash.
        """
        with self._lock:
            now = time.time()
            prev_root = self._merkle.root()
            seq = len(self._events) + 1

            # Build a canonical record for hashing without depending on internal fields
            record_for_hash = {
                "_seq": seq,
                "_ts": now,
                "_cid": correlation_id,
                "_prev_root": prev_root,
                "payload": payload,
            }
            leaf = self._merkle.add_leaf(_canonical_json_bytes(record_for_hash))

            ev = Event(
                seq=seq,
                ts=now,
                ts_iso=_utc_iso(now),
                leaf=leaf,
                payload=payload,
                correlation_id=correlation_id,
                prev_root=prev_root,
            )
            self._events.append(ev)
            return leaf

    def append_bytes(self, blob: bytes, *, correlation_id: Optional[str] = None) -> str:
        """
        Append arbitrary bytes as an event payload container under {"_raw_b64": ...} to preserve determinism.
        """
        # Keep an explicit wrapper so canonical JSON remains stable and extensible
        import base64

        return self.append_event(
            {"_raw_b64": base64.b64encode(blob).decode("ascii")},
            correlation_id=correlation_id,
        )

    def root(self) -> Optional[str]:
        with self._lock:
            return self._merkle.root()

    def size(self) -> int:
        with self._lock:
            return len(self._events)

    def get_event(self, seq: int) -> Event:
        with self._lock:
            if seq <= 0 or seq > len(self._events):
                raise TamperStoreError("Sequence out of range")
            return self._events[seq - 1]

    def prove(self, seq: int) -> Dict[str, Any]:
        """
        Build an inclusion proof for the event at sequence 'seq' (1-based).
        """
        with self._lock:
            index = seq - 1
            leaf = self._merkle.leaves[index]
            proof = self._merkle.prove(index)
            root = self._merkle.root()
            return {
                "seq": seq,
                "leaf": leaf,
                "root": root,
                "proof": proof,
                "algorithm": self._digest,
                "size": len(self._merkle.leaves),
            }

    @staticmethod
    def verify_proof(proof_bundle: Dict[str, Any]) -> bool:
        """
        Verify an inclusion proof bundle produced by prove().
        """
        algo = proof_bundle.get("algorithm", "sha256")
        leaf = proof_bundle["leaf"]
        proof = proof_bundle["proof"]
        root = proof_bundle["root"]

        merkle = MerkleAppender(algorithm=algo)
        # Reuse Merkle verification routine without relying on internal leaves
        computed = merkle.verify(leaf=leaf, proof=proof, expected_root=root)
        return computed == root

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": self.SCHEMA_VERSION,
                "events": len(self._events),
                "last_seq": len(self._events),
                "merkle_root": self._merkle.root(),
                "anchors": [asdict(a) for a in self._anchors],
                "digest": self._digest,
            }

    # --------------- Anchoring ---------------

    def flush_to_remote(
        self,
        *,
        anchor: bool = True,
        anchor_type: Literal["tsa", "file", "custom"] = "tsa",
        anchor_url: Optional[str] = None,
        receipt: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        No-op remote push placeholder. Optionally records an anchor.
        Returns (ok, root).
        """
        with self._lock:
            root = self._merkle.root()
            if not root:
                return True, None

            if anchor:
                at = anchor_type
                au = anchor_url or (self.tsa_url if at == "tsa" else None)
                now = time.time()
                self._anchors.append(
                    Anchor(
                        type=at,
                        url=au,
                        root=root,
                        ts=now,
                        ts_iso=_utc_iso(now),
                        receipt=receipt,
                    )
                )
            return True, root

    # --------------- Verification ---------------

    def verify_all(self) -> bool:
        """
        Full verification pass: recompute all leaves and the final root, compare with current root.
        """
        with self._lock:
            # Recompute leaves deterministically from event records
            tmp = MerkleAppender(algorithm=self._digest)
            for ev in self._events:
                record_for_hash = {
                    "_seq": ev.seq,
                    "_ts": ev.ts,
                    "_cid": ev.correlation_id,
                    "_prev_root": ev.prev_root,
                    "payload": ev.payload,
                }
                tmp.add_leaf(_canonical_json_bytes(record_for_hash))

            expected_root = tmp.root()
            return expected_root == self._merkle.root()

    # --------------- Export / Import ---------------

    def export_records(self) -> List[Dict[str, Any]]:
        """
        Export a sequence of records suitable for durable persistence (e.g., JSONL).
        """
        with self._lock:
            recs: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "schema_version": self.SCHEMA_VERSION,
                    "digest": self._digest,
                    "created_ts": time.time(),
                    "created_ts_iso": _utc_iso(time.time()),
                }
            ]
            recs.extend(ev.to_record() for ev in self._events)
            recs.extend(a.to_record() for a in self._anchors)
            return recs

    @classmethod
    def import_records(
        cls, records: Iterable[Dict[str, Any]]
    ) -> TamperEvidentOfflineStore:
        """
        Build a store from exported records. Verifies integrity as it loads.
        """
        digest = "sha256"
        tsa_url: Optional[str] = None

        # find header if present
        for r in records:
            if r.get("type") == "header":
                digest = r.get("digest", "sha256")
                break

        store = cls(tsa_url=tsa_url, digest=digest)

        seq_counter = 0
        for r in records:
            rtype = r.get("type")
            if rtype == "event":
                seq_counter += 1
                # Recreate hashing record and recompute leaf to verify integrity
                record_for_hash = {
                    "_seq": r["seq"],
                    "_ts": r["ts"],
                    "_cid": r.get("correlation_id"),
                    "_prev_root": r.get("prev_root"),
                    "payload": r.get("payload", {}),
                }
                recomputed_leaf = _hex(
                    _digest(digest, _canonical_json_bytes(record_for_hash))
                )
                if recomputed_leaf != r.get("leaf"):
                    raise TamperStoreError(f"Event leaf mismatch at seq {r.get('seq')}")

                # Append to internal state
                store._merkle._leaves.append(recomputed_leaf)
                store._events.append(
                    Event(
                        seq=r["seq"],
                        ts=r["ts"],
                        ts_iso=r.get("ts_iso") or _utc_iso(r["ts"]),
                        leaf=recomputed_leaf,
                        payload=r.get("payload", {}),
                        correlation_id=r.get("correlation_id"),
                        prev_root=r.get("prev_root"),
                    )
                )
            elif rtype == "anchor":
                store._anchors.append(
                    Anchor(
                        type=r.get("anchor_type", "custom"),
                        root=r["root"],
                        ts=r["ts"],
                        ts_iso=r.get("ts_iso") or _utc_iso(r["ts"]),
                        url=r.get("url"),
                        receipt=r.get("receipt"),
                    )
                )
            elif rtype == "header":
                # already handled above
                continue
            else:
                raise TamperStoreError(f"Unknown record type: {rtype}")

        # Post condition: verify chain correctness
        if not store.verify_all():
            raise TamperStoreError("Verification failed after import")

        return store
