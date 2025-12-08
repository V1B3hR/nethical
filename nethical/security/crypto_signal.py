from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    MutableSet,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from nethical.hooks.interfaces import (
    CryptoSignalProvider,
    SignalResult,
    RoleName,
    TokenStr,
    TokenMeta,
)


# =========================
# Constants & Type Aliases
# =========================

Header = Dict[str, Any]
Payload = Dict[str, Any]

SUPPORTED_HMAC_ALGS: Mapping[str, Callable[[bytes], "hashlib._Hash"]] = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}

DEFAULT_TYPE = "ROLE"
DEFAULT_VERSION = "1"

# Reason / code constants for consistency and reuse
REASON_ALG_MISMATCH = "algorithm mismatch"
REASON_SIGNATURE_BAD = "bad signature"
REASON_TOKEN_MALFORMED = "malformed token"
REASON_CLAIM_MISSING = "missing claim"
REASON_TOKEN_EXPIRED = "token expired"
REASON_TOKEN_NOT_YET_VALID = "token not yet valid"
REASON_AUDIENCE_MISMATCH = "audience mismatch"
REASON_ISSUER_MISMATCH = "issuer mismatch"
REASON_TYP_MISMATCH = "type mismatch"
REASON_VERSION_UNSUPPORTED = "version unsupported"
REASON_TTL_EXCESS = "ttl exceeds allowed maximum"
REASON_REPLAY = "replay detected"
REASON_ROLE_NOT_ALLOWED = "role not allowed"


class KeyResolver(Protocol):
    """
    Protocol for pluggable key resolution / rotation.
    Given a key id (kid) return the shared secret bytes.
    """

    def __call__(self, key_id: str) -> bytes: ...


@dataclass
class IssueOptions:
    """
    Options controlling issuance behavior.
    """

    ttl_seconds: int = 60
    not_before_leeway: int = 0  # Additional backdating leeway beyond core clock skew
    extra_claims: Dict[str, Any] | None = None
    role_claim_name: str = "role"
    # Provide an externally generated jti if required (e.g. deterministic); otherwise random.
    jti: Optional[str] = None


@dataclass
class VerifyOptions:
    """
    Options controlling verification behavior.
    """

    enforce_peer_issuer: bool = True
    max_ttl_seconds: Optional[int] = None  # Override provider default if set
    acceptable_clock_skew_seconds: Optional[int] = None
    required_type: str = DEFAULT_TYPE
    required_version: str = DEFAULT_VERSION
    role_claim_name: str = "role"
    allowed_roles: Optional[Set[str]] = None
    require_jti: bool = False
    reject_if_no_jti: bool = False
    # Callback to record accepted jti (for replay prevention).
    replay_register: Optional[Callable[[str], None]] = None
    # Callback returning True if jti already used.
    replay_already_seen: Optional[Callable[[str], bool]] = None
    # Allow injecting additional validation logic; raise Exception or return str for failure message.
    custom_validators: Optional[
        Sequence[Callable[[Header, Payload], Optional[str]]]
    ] = None


class InMemoryJtiStore:
    """
    Simple bounded in-memory replay detector.
    Not thread-safe for high concurrency without additional locking.
    """

    def __init__(self, max_entries: int = 10_000):
        self.max_entries = max_entries
        self._entries: MutableSet[str] = set()

    def already_seen(self, jti: str) -> bool:
        return jti in self._entries

    def register(self, jti: str) -> None:
        self._entries.add(jti)
        # Simple pruning policy - remove excess entries efficiently
        if len(self._entries) > self.max_entries:
            # Calculate how many to remove
            to_remove = len(self._entries) - self.max_entries
            # Random sampling removal for O(1) average; acceptable for a lightweight utility
            for _ in range(to_remove):
                self._entries.pop()


class HmacRoleSignal(CryptoSignalProvider):
    """
    Advanced, deterministic, HMAC-based role signaling compatible with nethical interfaces.

    Enhancements over the initial minimal implementation:
    - Multiple HMAC algorithms (HS256 / HS384 / HS512)
    - Optional key resolver for rotation (verification path)
    - Support for `jti` claim (optional replay detection)
    - Enforced TTL upper bound in verification
    - Configurable role claim name and allowed role set
    - Extensible verification via custom validators
    - Structured options dataclasses for issuance and verification
    - Extra claims injection at issuance
    - Type & version claims to enable protocol evolution
    - Defensive size limits & strict JSON parsing
    """

    def __init__(
        self,
        secret: bytes | None,
        issuer: str,
        audience: str,
        subject: Optional[str] = None,
        key_id: str = "default",
        algorithm: str = "HS256",
        max_ttl_seconds: int = 300,
        clock_skew_seconds: int = 5,
        enforce_peer_issuer: bool = True,
        key_resolver: Optional[KeyResolver] = None,
        enable_replay_store: bool = False,
    ):
        """
        Args:
            secret: Shared secret for signing (required if issuing). If using only verification with key_resolver, may be None.
            issuer: Identity we assert when issuing tokens (iss).
            audience: Peer identity (aud) we target in issued tokens.
            subject: Subject (sub) bound to our issued tokens; defaults to issuer.
            key_id: Identifier of the secret.
            algorithm: One of SUPPORTED_HMAC_ALGS keys.
            max_ttl_seconds: Upper bound for token TTL at issuance & optional verification enforcement.
            clock_skew_seconds: Allowed skew for time-based claim validation.
            enforce_peer_issuer: Default for verification if options not overridden.
            key_resolver: Optional resolver function for key rotation; if provided, used in verification.
            enable_replay_store: If True, enables an in-memory replay detection store.
        """
        if algorithm not in SUPPORTED_HMAC_ALGS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        if secret is None and key_resolver is None:
            raise ValueError("Either secret or key_resolver must be provided.")

        self._signing_secret = secret  # May be None if only verifying with resolver
        self.issuer = issuer
        self.audience = audience
        self.subject = subject or issuer
        self.key_id = key_id
        self.algorithm = algorithm
        self.max_ttl_seconds = int(max_ttl_seconds)
        self.clock_skew_seconds = int(clock_skew_seconds)
        self.default_enforce_peer_issuer = bool(enforce_peer_issuer)
        self.key_resolver = key_resolver
        self._replay_store = InMemoryJtiStore() if enable_replay_store else None

    # ========= Internal Helpers =========

    @staticmethod
    def _now() -> int:
        return int(time.time())

    @staticmethod
    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64urldecode(s: str) -> bytes:
        pad = "=" * (-len(s) % 4)
        return base64.urlsafe_b64decode((s + pad).encode("ascii"))

    def _resolve_secret_for_verification(self, kid: str) -> bytes:
        if self.key_resolver:
            return self.key_resolver(kid)
        # Fallback to static secret if key id matches
        if self._signing_secret is None:
            raise ValueError(
                "No key resolver configured and no signing secret available."
            )
        if kid != self.key_id:
            # Allow mismatch but warn (could also treat as error to tighten security)
            # For strictness, raise:
            raise ValueError("Key ID mismatch for static secret.")
        return self._signing_secret

    def _sign(self, secret: bytes, message: bytes) -> bytes:
        digest_constructor = SUPPORTED_HMAC_ALGS[self.algorithm]
        return hmac.new(secret, message, digest_constructor).digest()

    def _build_header(self) -> Header:
        return {
            "alg": self.algorithm,
            "kid": self.key_id,
            "typ": DEFAULT_TYPE,
            "v": DEFAULT_VERSION,
        }

    def _build_payload(
        self,
        *,
        role_claim_name: str,
        role_value: str,
        iat: int,
        nbf: int,
        exp: int,
        iss: str,
        sub: str,
        aud: str,
        jti: Optional[str],
        extra_claims: Optional[Dict[str, Any]],
    ) -> Payload:
        payload: Payload = {
            "iss": iss,
            "sub": sub,
            "aud": aud,
            role_claim_name: role_value,
            "iat": iat,
            "nbf": nbf,
            "exp": exp,
        }
        if jti is not None:
            payload["jti"] = jti
        if extra_claims:
            # Avoid overwriting core claims
            for k, v in extra_claims.items():
                if k in payload:
                    raise ValueError(f"Extra claim collides with reserved claim: {k}")
                payload[k] = v
        return payload

    @staticmethod
    def _json_dumps_canonical(obj: Any) -> bytes:
        return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @staticmethod
    def _safe_json_loads(raw: bytes) -> Any:
        # Additional safety check – limit size to mitigate pathological memory usage.
        if len(raw) > 64 * 1024:  # Arbitrary cap for safety
            raise ValueError("JSON segment too large")
        return json.loads(raw.decode("utf-8"))

    def _compute_ttl(self, payload: Payload) -> int:
        return int(payload["exp"]) - int(payload["iat"])

    # ========= Public (Issuance) =========

    def issue_medical_role_token(
        self,
        role: RoleName,
        ttl_seconds: int = 60,
        options: Optional[IssueOptions] = None,
    ) -> SignalResult:
        """
        Issue a signed role token.

        Args:
            role: The role value to embed.
            ttl_seconds: Requested TTL; will be clamped by configuration.
            options: Optional IssueOptions for advanced control.

        Returns:
            SignalResult with token and metadata.
        """
        if self._signing_secret is None:
            return SignalResult(
                ok=False, reason="signing not configured", code="issue_unavailable"
            )

        opts = options or IssueOptions(ttl_seconds=ttl_seconds)
        # ttl_seconds parameter still observed if options not provided
        if options is None:
            opts.ttl_seconds = ttl_seconds

        try:
            ttl = max(1, min(int(opts.ttl_seconds), self.max_ttl_seconds))
            now = self._now()
            iat = now
            nbf = now - self.clock_skew_seconds - max(0, int(opts.not_before_leeway))
            exp = now + ttl
            jti = opts.jti or secrets.token_urlsafe(12)

            header = self._build_header()
            payload = self._build_payload(
                role_claim_name=opts.role_claim_name,
                role_value=str(role),
                iat=iat,
                nbf=nbf,
                exp=exp,
                iss=self.issuer,
                sub=self.subject,
                aud=self.audience,
                jti=jti,
                extra_claims=opts.extra_claims,
            )

            header_b64 = self._b64url(self._json_dumps_canonical(header))
            payload_b64 = self._b64url(self._json_dumps_canonical(payload))
            signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
            sig = self._sign(self._signing_secret, signing_input)
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
                "meta": {
                    "role": payload[opts.role_claim_name],
                    "role_claim": opts.role_claim_name,
                    "jti": payload.get("jti"),
                },
            }

            return SignalResult(ok=True, token=token, meta=meta)
        except Exception as e:
            return SignalResult(ok=False, reason=str(e), code="issue_error")

    # ========= Public (Verification) =========

    def verify_peer_role_token(
        self,
        token: TokenStr,
        options: Optional[VerifyOptions] = None,
    ) -> SignalResult:
        """
        Verify a peer-provided token and extract role metadata.

        Args:
            token: The token string.
            options: Optional VerifyOptions for more granular control.

        Returns:
            SignalResult with metadata if valid.
        """
        opts = options or VerifyOptions(
            enforce_peer_issuer=self.default_enforce_peer_issuer
        )

        try:
            if not isinstance(token, str):
                return SignalResult(
                    ok=False, reason="token must be str", code="bad_input"
                )

            parts = token.split(".")
            if len(parts) != 3:
                return SignalResult(
                    ok=False, reason=REASON_TOKEN_MALFORMED, code="bad_format"
                )
            header_raw, payload_raw, sig_raw = parts

            # Basic sanity size guard
            if any(len(p) > 16_384 for p in parts):
                return SignalResult(
                    ok=False, reason="segment too large", code="bad_format"
                )

            header_bytes = self._b64urldecode(header_raw)
            payload_bytes = self._b64urldecode(payload_raw)
            sig = self._b64urldecode(sig_raw)

            header = self._safe_json_loads(header_bytes)
            payload = self._safe_json_loads(payload_bytes)

            # Header validations
            alg = header.get("alg")
            kid = header.get("kid")
            typ = header.get("typ")
            ver = header.get("v")

            if alg != self.algorithm:
                return SignalResult(
                    ok=False, reason=REASON_ALG_MISMATCH, code="alg_mismatch"
                )
            if typ != opts.required_type:
                return SignalResult(
                    ok=False, reason=REASON_TYP_MISMATCH, code="typ_mismatch"
                )
            if ver != opts.required_version:
                return SignalResult(
                    ok=False, reason=REASON_VERSION_UNSUPPORTED, code="version_mismatch"
                )

            # Resolve secret (supports rotation)
            secret = self._resolve_secret_for_verification(str(kid))

            # Signature check
            expected_sig = self._sign(
                secret, f"{header_raw}.{payload_raw}".encode("ascii")
            )
            if not hmac.compare_digest(expected_sig, sig):
                return SignalResult(
                    ok=False, reason=REASON_SIGNATURE_BAD, code="bad_signature"
                )

            role_claim_name = opts.role_claim_name
            required_claims = {
                "iss",
                "sub",
                "aud",
                role_claim_name,
                "iat",
                "nbf",
                "exp",
            }
            missing = [c for c in required_claims if c not in payload]
            if missing:
                return SignalResult(
                    ok=False,
                    reason=f"{REASON_CLAIM_MISSING}: {','.join(missing)}",
                    code="bad_claims",
                )

            now = self._now()
            clock_skew = (
                opts.acceptable_clock_skew_seconds
                if opts.acceptable_clock_skew_seconds is not None
                else self.clock_skew_seconds
            )

            if now + clock_skew < int(payload["nbf"]):
                return SignalResult(
                    ok=False, reason=REASON_TOKEN_NOT_YET_VALID, code="not_yet_valid"
                )
            if int(payload["exp"]) < now - clock_skew:
                return SignalResult(
                    ok=False, reason=REASON_TOKEN_EXPIRED, code="expired"
                )

            # TTL enforcement (exp - iat)
            ttl = self._compute_ttl(payload)
            limit_ttl = (
                opts.max_ttl_seconds
                if opts.max_ttl_seconds is not None
                else self.max_ttl_seconds
            )
            if ttl > limit_ttl:
                return SignalResult(
                    ok=False, reason=REASON_TTL_EXCESS, code="ttl_excess"
                )

            # Audience/Issuer semantics
            if str(payload["aud"]) != self.issuer:
                return SignalResult(
                    ok=False, reason=REASON_AUDIENCE_MISMATCH, code="aud_mismatch"
                )
            if opts.enforce_peer_issuer and str(payload["iss"]) != self.audience:
                return SignalResult(
                    ok=False, reason=REASON_ISSUER_MISMATCH, code="iss_mismatch"
                )

            # Role filtering
            role_value = str(payload[role_claim_name])
            if opts.allowed_roles is not None and role_value not in opts.allowed_roles:
                return SignalResult(
                    ok=False, reason=REASON_ROLE_NOT_ALLOWED, code="role_not_allowed"
                )

            # Replay protection
            jti = payload.get("jti")
            if opts.require_jti and jti is None:
                return SignalResult(ok=False, reason="missing jti", code="missing_jti")
            if opts.reject_if_no_jti and jti is None:
                return SignalResult(ok=False, reason="missing jti", code="missing_jti")
            if jti:
                already_seen = False
                if opts.replay_already_seen:
                    already_seen = opts.replay_already_seen(jti)
                elif self._replay_store:
                    already_seen = self._replay_store.already_seen(jti)
                if already_seen:
                    return SignalResult(ok=False, reason=REASON_REPLAY, code="replay")
                # Register use
                if opts.replay_register:
                    opts.replay_register(jti)
                elif self._replay_store:
                    self._replay_store.register(jti)

            # Custom validators
            if opts.custom_validators:
                for validator in opts.custom_validators:
                    try:
                        res = validator(header, payload)
                        if isinstance(res, str):
                            return SignalResult(
                                ok=False, reason=res, code="custom_validator_failed"
                            )
                    except Exception as ve:
                        return SignalResult(
                            ok=False, reason=str(ve), code="custom_validator_error"
                        )

            meta: TokenMeta = {
                "issuer": str(payload["iss"]),
                "subject": str(payload["sub"]),
                "audience": str(payload["aud"]),
                "issued_at": float(payload["iat"]),
                "not_before": float(payload["nbf"]),
                "expires_at": float(payload["exp"]),
                "key_id": str(kid),
                "algorithm": str(alg),
                "meta": {
                    "role": role_value,
                    "role_claim": role_claim_name,
                    "jti": jti,
                    "ttl": ttl,
                },
            }

            return SignalResult(ok=True, token=token, meta=meta)
        except Exception as e:
            return SignalResult(ok=False, reason=str(e), code="verify_error")

    # ========= Utility / Introspection =========

    def decode_unverified(self, token: TokenStr) -> Tuple[Header, Payload]:
        """
        Decode header & payload WITHOUT verifying signature. For debugging / logging ONLY.
        """
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("malformed token")
        header_raw, payload_raw, _ = parts
        header = self._safe_json_loads(self._b64urldecode(header_raw))
        payload = self._safe_json_loads(self._b64urldecode(payload_raw))
        return header, payload
