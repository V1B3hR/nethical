from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

from nethical.hooks.interfaces import (
    CryptoSignalProvider,
    SignalResult,
    RoleName,
    TokenStr,
    TokenMeta,
)


class HmacRoleSignal(CryptoSignalProvider):
    """
    Minimal, deterministic HMAC-based role signaling aligned with nethical interfaces.

    - Compact, JWT-like structure: base64url(header).base64url(payload).base64url(signature)
    - Header fields: alg, kid, typ="ROLE", v="1"
    - Payload fields: iss, sub, aud, role, iat, nbf, exp
    - Enforces TTL, audience, clock skew, and optional peer issuer check
    """

    def __init__(
        self,
        secret: bytes,
        issuer: str,
        audience: str,
        subject: Optional[str] = None,
        key_id: str = "default",
        algorithm: str = "HS256",
        max_ttl_seconds: int = 300,
        clock_skew_seconds: int = 5,
        enforce_peer_issuer: bool = True,
    ):
        """
        Args:
            secret: Shared secret for HMAC signing/verification.
            issuer: Our identity when issuing tokens (iss).
            audience: The peer's identity (aud) for issued tokens; also expected issuer when verifying if enforce_peer_issuer=True.
            subject: Subject bound to our issued tokens (defaults to issuer).
            key_id: Identifier of the key/secret used.
            algorithm: Only "HS256" is supported currently.
            max_ttl_seconds: Upper bound for token TTL to constrain issuance.
            clock_skew_seconds: Allowed clock skew for nbf/exp checks.
            enforce_peer_issuer: When verifying, require iss == audience (peer).
        """
        self.secret = secret
        self.issuer = issuer
        self.audience = audience
        self.subject = subject or issuer
        self.key_id = key_id
        self.algorithm = algorithm
        self.max_ttl_seconds = int(max_ttl_seconds)
        self.clock_skew_seconds = int(clock_skew_seconds)
        self.enforce_peer_issuer = bool(enforce_peer_issuer)

    # --------- helpers ---------

    def _now(self) -> int:
        return int(time.time())

    def _b64url(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _b64urldecode(self, s: str) -> bytes:
        pad = "=" * (-len(s) % 4)
        return base64.urlsafe_b64decode((s + pad).encode("ascii"))

    def _sign(self, message: bytes) -> bytes:
        if self.algorithm != "HS256":
            raise ValueError("unsupported algorithm")
        return hmac.new(self.secret, message, hashlib.sha256).digest()

    def _build_header(self) -> Dict[str, Any]:
        return {"alg": self.algorithm, "kid": self.key_id, "typ": "ROLE", "v": "1"}

    def _build_payload(
        self,
        role: str,
        iat: int,
        nbf: int,
        exp: int,
        iss: str,
        sub: str,
        aud: str,
    ) -> Dict[str, Any]:
        return {
            "iss": iss,
            "sub": sub,
            "aud": aud,
            "role": role,
            "iat": iat,
            "nbf": nbf,
            "exp": exp,
        }

    # --------- CryptoSignalProvider ---------

    def issue_medical_role_token(self, role: RoleName, ttl_seconds: int = 60) -> SignalResult:
        try:
            ttl = max(1, min(int(ttl_seconds), self.max_ttl_seconds))
            now = self._now()
            iat = now
            nbf = now - self.clock_skew_seconds
            exp = now + ttl

            header = self._build_header()
            payload = self._build_payload(
                role=str(role),
                iat=iat,
                nbf=nbf,
                exp=exp,
                iss=self.issuer,
                sub=self.subject,
                aud=self.audience,
            )

            header_b64 = self._b64url(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
            payload_b64 = self._b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
            signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
            sig = self._sign(signing_input)
            sig_b64 = self._b64url(sig)

            token: TokenStr = f"{header_b64}.{payload_b64}.{sig_b64}"

            meta: TokenMeta = {
                "issuer": payload["iss"],
                "subject": payload["sub"],
                "audience": payload["aud"],
                "issued_at": float(payload["iat"]),
                "not_before": float(payload["nbf"]),
                "expires_at": float(payload["exp"]),
                "key_id": self.key_id,
                "algorithm": self.algorithm,
                "meta": {"role": str(role)},
            }

            return SignalResult(ok=True, token=token, meta=meta)
        except Exception as e:
            return SignalResult(ok=False, reason=str(e), code="issue_error")

    def verify_peer_role_token(self, token: TokenStr) -> SignalResult:
        try:
            if not isinstance(token, str):
                return SignalResult(ok=False, reason="token must be str", code="bad_input")

            parts = token.split(".")
            if len(parts) != 3:
                return SignalResult(ok=False, reason="malformed token", code="bad_format")

            header_raw, payload_raw, sig_raw = parts
            header_bytes = self._b64urldecode(header_raw)
            payload_bytes = self._b64urldecode(payload_raw)
            sig = self._b64urldecode(sig_raw)

            header = json.loads(header_bytes.decode("utf-8"))
            payload = json.loads(payload_bytes.decode("utf-8"))

            # Algorithm
            alg = header.get("alg")
            if alg != self.algorithm:
                return SignalResult(ok=False, reason="algorithm mismatch", code="alg_mismatch")

            # Signature
            expected_sig = self._sign(f"{header_raw}.{payload_raw}".encode("ascii"))
            if not hmac.compare_digest(expected_sig, sig):
                return SignalResult(ok=False, reason="bad signature", code="bad_signature")

            # Claims presence
            for key in ("iss", "sub", "aud", "role", "iat", "nbf", "exp"):
                if key not in payload:
                    return SignalResult(ok=False, reason=f"missing claim: {key}", code="bad_claims")

            now = self._now()

            # Time checks with skew
            if now + self.clock_skew_seconds < int(payload["nbf"]):
                return SignalResult(ok=False, reason="token not yet valid", code="not_yet_valid")
            if int(payload["exp"]) < now - self.clock_skew_seconds:
                return SignalResult(ok=False, reason="token expired", code="expired")

            # Audience: incoming tokens must target us (our issuer identity)
            if str(payload["aud"]) != self.issuer:
                return SignalResult(ok=False, reason="audience mismatch", code="aud_mismatch")

            # Issuer: by default require the peer issuer to match configured audience
            if self.enforce_peer_issuer and str(payload["iss"]) != self.audience:
                return SignalResult(ok=False, reason="issuer mismatch", code="iss_mismatch")

            meta: TokenMeta = {
                "issuer": str(payload["iss"]),
                "subject": str(payload["sub"]),
                "audience": str(payload["aud"]),
                "issued_at": float(payload["iat"]),
                "not_before": float(payload["nbf"]),
                "expires_at": float(payload["exp"]),
                "key_id": str(header.get("kid", "")),
                "algorithm": str(alg),
                "meta": {"role": str(payload["role"])},
            }

            return SignalResult(ok=True, token=token, meta=meta)
        except Exception as e:
            return SignalResult(ok=False, reason=str(e), code="verify_error")
