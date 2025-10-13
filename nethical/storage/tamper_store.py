from __future__ import annotations
import json, hashlib, time
from typing import Dict, Any, Optional, Tuple, List

class MerkleAppender:
    def __init__(self):
        self.leaves: List[str] = []

    def add_leaf(self, data: bytes) -> str:
        leaf = hashlib.sha256(data).hexdigest()
        self.leaves.append(leaf)
        return leaf

    def root(self) -> Optional[str]:
        nodes = self.leaves[:]
        if not nodes:
            return None
        while len(nodes) > 1:
            nxt = []
            for i in range(0, len(nodes), 2):
                a = nodes[i]
                b = nodes[i+1] if i+1 < len(nodes) else a
                nxt.append(hashlib.sha256((a+b).encode()).hexdigest())
            nodes = nxt
        return nodes[0]

class TamperEvidentOfflineStore:
    def __init__(self, tsa_url: Optional[str] = None):
        self.tsa_url = tsa_url
        self.merkle = MerkleAppender()
        self.events: list[dict[str, Any]] = []
        self.anchors: list[dict[str, Any]] = []

    def append_event(self, event: Dict[str, Any]) -> str:
        event["_ts"] = time.time()
        blob = json.dumps(event, sort_keys=True).encode()
        leaf = self.merkle.add_leaf(blob)
        self.events.append(event)
        return leaf

    def snapshot(self) -> Dict[str, Any]:
        return {"events": len(self.events), "merkle_root": self.merkle.root(), "anchors": self.anchors}

    def flush_to_remote(self) -> Tuple[bool, Optional[str]]:
        # TODO: push to remote + optional RFC3161 TSA anchor
        root = self.merkle.root()
        if not root:
            return True, None
        if self.tsa_url:
            self.anchors.append({"type": "tsa", "url": self.tsa_url, "root": root, "ts": time.time()})
        # reset after flush (optionally keep)
        return True, root
