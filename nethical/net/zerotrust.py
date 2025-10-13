from __future__ import annotations
from typing import Dict, Any
from nethical.hooks.interfaces import CommsPolicy

class NoopCommsPolicy(CommsPolicy):
    def connection_allowed(self, peer_id: str, context: Dict[str, Any]) -> bool:
        return True
    def identities(self) -> Dict[str, Any]:
        return {"impl": "noop"}

# Placeholder for SPIFFE/SPIRE, mTLS identities, allowlists/deny-lists per region/mission
class MTLSCommsPolicy(CommsPolicy):
    def __init__(self, trust_domain: str, allowed_identities: list[str] | None = None):
        self.trust_domain = trust_domain
        self.allowed = set(allowed_identities or [])
    def connection_allowed(self, peer_id: str, context: Dict[str, Any]) -> bool:
        return (peer_id in self.allowed) if self.allowed else False
    def identities(self) -> Dict[str, Any]:
        return {"trust_domain": self.trust_domain, "allowed": list(self.allowed)}
