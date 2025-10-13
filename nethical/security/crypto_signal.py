from __future__ import annotations
import time
import hmac
import hashlib
import base64
from typing import Dict, Any
from nethical.hooks.interfaces import CryptoSignalProvider, SignalResult

class HmacRoleSignal(CryptoSignalProvider):
    def __init__(self, secret: bytes, issuer: str, audience: str):
        self.secret = secret
        self.issuer = issuer
        self.audience = audience

    def issue_medical_role_token(self, role: str, ttl_seconds: int = 60) -> SignalResult:
        now = int(time.time())
        payload = f"{self.issuer}|{self.audience}|{role}|{now}|{ttl_seconds}"
        sig = hmac.new(self.secret, payload.encode(), hashlib.sha256).digest()
        token = base64.urlsafe_b64encode(payload.encode() + b"." + sig).decode()
        return SignalResult(ok=True, token=token, meta={"role": role, "exp": now + ttl_seconds})

    def verify_peer_role_token(self, token: str) -> SignalResult:
        try:
            data = base64.urlsafe_b64decode(token.encode())
            payload, sig = data.rsplit(b".", 1)
            calc = hmac.new(self.secret, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(calc, sig):
                return SignalResult(ok=False, reason="bad signature")
            parts = payload.decode().split("|")
            issuer, audience, role, issued, ttl = parts
            if int(issued) + int(ttl) < int(time.time()):
                return SignalResult(ok=False, reason="expired")
            return SignalResult(ok=True, meta={"issuer": issuer, "audience": audience, "role": role})
        except Exception as e:
            return SignalResult(ok=False, reason=str(e))
