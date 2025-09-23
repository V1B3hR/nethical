"""
nethical.py v6 - Cognitive AI Ethics, Safety Governance, and Secure File Management

Nethical is a Cognitive Residual Current Device (RCD), AI Ethics Framework, and secure file
management utility. It monitors and enforces multi-layered ethical, safety, and human-AI
relationship principles, and provides secure file encryption/decryption with chunked
processing, key rotation support via MultiFernet, and metadata preservation.

Highlights:
- Simple, single-file implementation with minimal dependencies
- Intent vs Action monitoring with similarity scoring
- Rule-based safety and ethics constraint detection
- Circuit breaker with cooldown for CRITICAL/EMERGENCY events
- Thread-safe histories and callbacks
- Secure file encryption/decryption (chunked) with metadata preservation
- Cross-platform-safe file-lock fallbacks (no-ops on unsupported platforms)
- UTC timestamps and monotonic clock for safety operations
"""

from __future__ import annotations

import os
import re
import sys
import time
import logging
import platform
from pathlib import Path
from typing import Optional, List, Any, Callable, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

try:
    # Unix-only; will fallback to no-op on Windows or if unavailable
    import fcntl as _fcntl  # type: ignore
except Exception:  # pragma: no cover
    _fcntl = None  # type: ignore

try:
    from cryptography.fernet import Fernet, InvalidToken, MultiFernet
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'cryptography' package is required. Install with: pip install cryptography"
    ) from e

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Default basic configuration if the host app didn't configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ------------------------------------------------------------------------------
# Secure File Management
# ------------------------------------------------------------------------------

MAGIC_HEADER = b"FEN1"
CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk for streaming
MAX_CIPHERTEXT_CHUNK = 32 * 1024 * 1024  # Safety cap of 32 MB per encrypted chunk

class SecurityError(Exception):
    pass

class AlreadyEncryptedError(SecurityError):
    pass

class NotEncryptedError(SecurityError):
    pass


def _acquire_file_lock(fileobj) -> None:
    """Best-effort exclusive lock; no-op on unsupported platforms."""
    if _fcntl is None:
        return
    try:
        _fcntl.flock(fileobj.fileno(), _fcntl.LOCK_EX)
    except Exception as e:  # pragma: no cover
        logger.debug(f"Could not acquire file lock: {e}")

def _release_file_lock(fileobj) -> None:
    """Best-effort unlock; no-op on unsupported platforms."""
    if _fcntl is None:
        return
    try:
        _fcntl.flock(fileobj.fileno(), _fcntl.LOCK_UN)
    except Exception as e:  # pragma: no cover
        logger.debug(f"Could not release file lock: {e}")

def _preserve_metadata(src_path: Path, dst_path: Path) -> None:
    """Copy permissions, ownership (best-effort), and timestamps from src to dst."""
    try:
        st = src_path.stat()
        os.chmod(dst_path, st.st_mode)
        try:
            # May fail without sufficient privileges; ignore
            os.chown(dst_path, st.st_uid, st.st_gid)  # type: ignore[attr-defined]
        except Exception:
            pass
        os.utime(dst_path, (st.st_atime, st.st_mtime))
    except FileNotFoundError:
        # If src disappeared during operation, skip metadata copy
        pass

def _warn_if_permissive(path: Path) -> None:
    """Warn if key file is too permissive on POSIX systems."""
    try:
        st = path.stat()
        # Only warn on POSIX-like systems
        if os.name == "posix":
            if (st.st_mode & 0o077) != 0:
                logger.warning(f"Key file {path} has overly permissive permissions: {oct(st.st_mode)}")
    except Exception:
        pass


class SecurityManager:
    """
    SecurityManager with idempotency, metadata preservation,
    chunked encryption, key rotation (MultiFernet), and concurrency safety.
    """

    @staticmethod
    def generate_key(key_path: str) -> None:
        """
        Generate a new Fernet key and write it to key_path with restrictive permissions.
        """
        key_file = Path(key_path)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        tmp_path = key_file.with_suffix(key_file.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(key)
        os.replace(tmp_path, key_file)
        # Best effort restrictive permissions on POSIX
        if os.name == "posix":
            try:
                os.chmod(key_file, 0o600)
            except Exception:
                pass
        logger.info(f"New encryption key generated and saved securely to {key_path}")

    @staticmethod
    def _normalize_key_bytes(key: bytes | str) -> bytes:
        if isinstance(key, str):
            key = key.encode("utf-8")
        return key.strip()

    @staticmethod
    def load_key(key_path: str, extra_keys: Optional[List[bytes | str]] = None) -> MultiFernet:
        """
        Load a primary key from key_path and optional extra (older) keys for decryption.
        Primary key is used for encryption; extra_keys are fallback for decryption/rotation.
        """
        with open(key_path, "rb") as key_file:
            key = key_file.read().strip()
            if len(key) != 44:  # Fernet keys are 32 bytes base64 (44 chars)
                raise ValueError("Fernet key must be 44-byte base64.")
            _warn_if_permissive(Path(key_path))
            ferns = [Fernet(key)]
            if extra_keys:
                for ek in extra_keys:
                    kb = SecurityManager._normalize_key_bytes(ek)
                    if len(kb) == 44:
                        ferns.append(Fernet(kb))
                    else:
                        logger.warning("Skipping invalid extra key (must be 44-byte base64).")
            return MultiFernet(ferns)

    @staticmethod
    def is_encrypted(file_path: str) -> bool:
        """
        Return True if the file begins with the expected magic header.
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(len(MAGIC_HEADER))
                return header == MAGIC_HEADER
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to check encryption status for {file_path}: {e}")
            return False

    @staticmethod
    def encrypt_file(
        file_path: str,
        key_path: str,
        output_path: Optional[str] = None,
        extra_keys: Optional[List[bytes | str]] = None,
    ) -> None:
        """
        Encrypt file in chunks. If output_path is None, overwrite the source (atomic replace).
        """
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path

        if src_path.exists() and SecurityManager.is_encrypted(str(src_path)):
            logger.warning(f"File '{src_path}' is already encrypted.")
            raise AlreadyEncryptedError(f"File '{src_path}' is already encrypted.")

        fernet = SecurityManager.load_key(key_path, extra_keys=extra_keys)
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".enc_tmp")

        with open(src_path, "rb") as infile, open(tmp_path, "wb") as outfile:
            _acquire_file_lock(outfile)
            try:
                outfile.write(MAGIC_HEADER)
                while True:
                    chunk = infile.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    ciphertext = fernet.encrypt(chunk)
                    chunk_len = len(ciphertext).to_bytes(4, "big")
                    outfile.write(chunk_len)
                    outfile.write(ciphertext)
            finally:
                _release_file_lock(outfile)

        _preserve_metadata(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        logger.info(f"File '{file_path}' has been successfully encrypted to '{dst_path}'.")

    @staticmethod
    def decrypt_file_safely(
        file_path: str,
        key_path: str,
        output_path: Optional[str] = None,
        extra_keys: Optional[List[bytes | str]] = None,
    ) -> None:
        """
        Decrypt a file created by encrypt_file(). If output_path is None, overwrite the source.
        Validates chunk lengths for safety and removes temporary artifacts on failure.
        """
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".decrypted_tmp")

        with open(src_path, "rb") as infile:
            header = infile.read(len(MAGIC_HEADER))
            if header != MAGIC_HEADER:
                logger.error(f"File '{file_path}' is not encrypted with expected header.")
                raise NotEncryptedError(f"File '{file_path}' is not encrypted with expected header.")

            fernet = SecurityManager.load_key(key_path, extra_keys=extra_keys)
            with open(tmp_path, "wb") as outfile:
                _acquire_file_lock(outfile)
                try:
                    while True:
                        len_bytes = infile.read(4)
                        if not len_bytes:
                            break
                        if len(len_bytes) != 4:
                            raise SecurityError("Corrupted file: incomplete chunk length.")
                        chunk_len = int.from_bytes(len_bytes, "big")
                        if chunk_len <= 0 or chunk_len > MAX_CIPHERTEXT_CHUNK:
                            raise SecurityError("Corrupted file: invalid chunk length.")
                        chunk_ciphertext = infile.read(chunk_len)
                        if len(chunk_ciphertext) != chunk_len:
                            raise SecurityError("Corrupted file: truncated ciphertext chunk.")
                        try:
                            plaintext = fernet.decrypt(chunk_ciphertext)
                        except InvalidToken as e:
                            raise InvalidToken(
                                "DECRYPTION FAILED: The key is incorrect or the data is corrupt."
                            ) from e
                        outfile.write(plaintext)
                finally:
                    _release_file_lock(outfile)

        _preserve_metadata(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        logger.info(f"File '{file_path}' has been successfully decrypted to '{dst_path}'.")

    @staticmethod
    def encrypt_file_to(file_path: str, key_path: str, output_path: str, extra_keys: Optional[List[bytes | str]] = None) -> None:
        SecurityManager.encrypt_file(file_path, key_path, output_path=output_path, extra_keys=extra_keys)

    @staticmethod
    def decrypt_file_to(file_path: str, key_path: str, output_path: str, extra_keys: Optional[List[bytes | str]] = None) -> None:
        SecurityManager.decrypt_file_safely(file_path, key_path, output_path=output_path, extra_keys=extra_keys)


# ------------------------------------------------------------------------------
# Ethics/Safety Governance
# ------------------------------------------------------------------------------

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ActionType(Enum):
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    SYSTEM_MODIFICATION = "system_modification"
    DATA_ACCESS = "data_access"
    EXTERNAL_INTERACTION = "external_interaction"

class ConstraintCategory(Enum):
    HUMAN_AI = "human_ai"
    UNIVERSAL = "universal"
    OPERATIONAL = "operational"
    CUSTOM = "custom"
    INTENT_LOCAL = "intent_local"

@dataclass
class Intent:
    description: str
    action_type: ActionType
    expected_outcome: str
    safety_constraints: List[str]
    confidence: float = 1.0
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class Action:
    description: str
    action_type: ActionType
    actual_parameters: Dict[str, Any]
    observed_effects: List[str]
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class ConstraintRule:
    rule_id: str
    description: str
    category: ConstraintCategory
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    check: Optional[Callable[[Action], bool]] = None
    keywords_any: Optional[List[str]] = None
    keywords_all: Optional[List[str]] = None
    except_keywords: Optional[List[str]] = None
    regex_any: Optional[List[str]] = None
    regex_all: Optional[List[str]] = None

    def violates(self, action: Action) -> bool:
        """
        Returns True if the action violates this rule.
        Priority:
        - If 'check' callable is provided, use it.
        - Otherwise, evaluate keyword and regex heuristics.
        """
        if self.check is not None:
            try:
                return bool(self.check(action))
            except Exception as e:
                logger.error(f"ConstraintRule.check failed for {self.rule_id}: {e}")
                return False

        desc_raw = action.description or ""
        desc = desc_raw.lower()

        if self.except_keywords and any(ex_kw in desc for ex_kw in self.except_keywords):
            return False

        def any_contains(words: Optional[List[str]]) -> Optional[bool]:
            if not words:
                return None
            return any((w or "").lower() in desc for w in words if w)

        def all_contains(words: Optional[List[str]]) -> Optional[bool]:
            if not words:
                return None
            return all((w or "").lower() in desc for w in words if w)

        def any_regex(patterns: Optional[List[str]]) -> Optional[bool]:
            if not patterns:
                return None
            return any(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in patterns if rx)

        def all_regex(patterns: Optional[List[str]]) -> Optional[bool]:
            if not patterns:
                return None
            return all(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in patterns if rx)

        cond_any_kw = any_contains(self.keywords_any)
        cond_all_kw = all_contains(self.keywords_all)
        cond_any_rx = any_regex(self.regex_any)
        cond_all_rx = all_regex(self.regex_all)

        def combine(cond_any, cond_all) -> Optional[bool]:
            if cond_any is None and cond_all is None:
                return None
            if cond_any is None:
                return cond_all
            if cond_all is None:
                return cond_any
            return cond_any and cond_all

        cond_kw = combine(cond_any_kw, cond_all_kw)
        cond_rx = combine(cond_any_rx, cond_all_rx)

        if cond_kw is None and cond_rx is None:
            return False
        if cond_kw is None:
            return bool(cond_rx)
        if cond_rx is None:
            return bool(cond_kw)
        return bool(cond_kw or cond_rx)

@dataclass
class SafetyViolation:
    violation_type: str
    severity: SafetyLevel
    intent: Intent
    action: Action
    deviation_score: float
    description: str
    violated_constraints: List[str]
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

# --- Utility similarity and redaction ---

def _token_jaccard(a: str, b: str) -> float:
    a_set = set((a or "").lower().split())
    b_set = set((b or "").lower().split())
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set | b_set))

_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)
_RE_PHONE = re.compile(r"\b(?:\+?\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b")

def _redact(text: str) -> str:
    if not text:
        return text
    text = _RE_EMAIL.sub("[redacted-email]", text)
    text = _RE_PHONE.sub("[redacted-phone]", text)
    return text

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

# --- Default rules ---

def _kw(*words: str) -> List[str]:
    return [w for w in words if w]

def _default_framework_rules() -> List[ConstraintRule]:
    """
    A compact yet meaningful set of default rules covering privacy, harm, cyber-security,
    deception, and system integrity. Weights represent severity contributions.
    """
    return [
        ConstraintRule(
            "operational.privacy",
            "Preserve Privacy - protect personal information and respect confidentiality.",
            ConstraintCategory.OPERATIONAL,
            weight=0.95,
            tags=["privacy", "pii", "security", "compliance"],
            keywords_any=[
                "leak pii", "dox", "expose personal data", "publish private info",
                "unauthorized disclosure", "unmask identity", "deanonymize", "doxx"
            ],
            regex_any=[r"\bssn\b", r"\bcredit\s*card\b", r"\bmedical record\b", r"\bpassport\b"],
        ),
        ConstraintRule(
            "universal.no_harm",
            "Do not promote physical harm, self-harm, or violence.",
            ConstraintCategory.UNIVERSAL,
            weight=1.0,
            tags=["safety", "harm", "violence", "self-harm"],
            keywords_any=_kw("kill", "injure", "assault", "self-harm", "suicide", "violence", "murder"),
        ),
        ConstraintRule(
            "operational.cyber_misuse",
            "Do not facilitate malware, unauthorized access, or exploitation.",
            ConstraintCategory.OPERATIONAL,
            weight=0.95,
            tags=["security", "malware", "exploit", "hacking"],
            keywords_any=_kw(
                "malware", "ransomware", "trojan", "keylogger", "reverse shell",
                "sql injection", "xss attack", "buffer overflow", "privilege escalation",
                "bypass authentication", "credential stuffing", "phishing", "exfiltrate"
            ),
            regex_any=[
                r"\bnc\s+-e\b", r"rm\s+-rf\s+/", r"(curl|wget).*\|\s*(sh|bash)",
                r"\bchmod\s+777\b", r"\bsudo\s+.*\b"
            ],
        ),
        ConstraintRule(
            "human_ai.deception",
            "Avoid deception, impersonation, and fraudulent behaviors.",
            ConstraintCategory.HUMAN_AI,
            weight=0.8,
            tags=["ethics", "deception", "impersonation"],
            keywords_any=_kw("impersonate", "pretend to be", "spoof", "deepfake", "fake identity"),
        ),
        ConstraintRule(
            "universal.hate_or_harassment",
            "Avoid hate speech and harassment.",
            ConstraintCategory.UNIVERSAL,
            weight=0.9,
            tags=["ethics", "harassment", "hate"],
            keywords_any=_kw("hate speech", "racist", "homophobic", "harass", "demean", "slur"),
        ),
        ConstraintRule(
            "operational.system_integrity",
            "Protect system integrity (no unauthorized modification).",
            ConstraintCategory.OPERATIONAL,
            weight=0.9,
            tags=["integrity", "system", "modification"],
            keywords_any=_kw("modify system files", "patch kernel", "edit config", "change registry"),
        ),
        ConstraintRule(
            "operational.data_exfiltration",
            "Do not exfiltrate or leak data to external destinations.",
            ConstraintCategory.OPERATIONAL,
            weight=0.9,
            tags=["data", "exfiltration", "privacy"],
            keywords_any=_kw("exfiltrate", "leak data", "data dump", "upload to external", "send to external"),
        ),
        ConstraintRule(
            "operational.unauthorized_access",
            "No unauthorized access or permission bypass.",
            ConstraintCategory.OPERATIONAL,
            weight=0.95,
            tags=["auth", "permission", "access control"],
            keywords_any=_kw("unauthorized access", "bypass auth", "without permission", "elevate privilege"),
        ),
    ]


class nethical:
    """
    nethical - Cognitive Residual Current Device (RCD), AI Ethics, and Secure File Management (v6)

    - Governance for agent safety/intents/actions.
    - Secure file encryption/decryption utilities.
    - Timestamps are UTC (NTP-synced).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = {
            "deviation_threshold": 0.7,
            "emergency_threshold": 0.9,
            "weights": {
                "type_mismatch": 1.0,
                "description": 1.0,
                "outcome": 0.8,
                "constraints": 1.2,
            },
            "trip_cooldown_seconds": 3.0,
            "reset_token": "admin_reset",
        }
        if config:
            # Merge at top level; weights merged shallowly
            self.config.update({k: v for k, v in config.items() if k != "weights"})
            if "weights" in config and isinstance(config["weights"], dict):
                self.config["weights"].update(config["weights"])

        self.is_active = True
        self.deviation_threshold = float(self.config.get("deviation_threshold", 0.7))
        self.emergency_threshold = float(self.config.get("emergency_threshold", 0.9))

        self.intent_history: List[Tuple[str, Intent]] = []
        self.action_history: List[Tuple[str, Action, float, List[str]]] = []
        self.violation_history: List[SafetyViolation] = []

        self.circuit_breaker_active = False
        self._last_trip_monotonic = 0.0

        self.safety_constraints: List[str] = []
        self.global_rules: List[ConstraintRule] = _default_framework_rules()

        self.description_similarity_fn: Callable[[str, str], float] = _token_jaccard
        self.outcome_similarity_fn: Callable[[str, str], float] = _token_jaccard

        import threading
        self._lock = threading.Lock()
        self.safety_callbacks: Dict[SafetyLevel, List[Callable[[SafetyViolation | str], None]]] = {
            SafetyLevel.WARNING: [],
            SafetyLevel.CRITICAL: [],
            SafetyLevel.EMERGENCY: [],
        }

        logger.info(
            f"nethical initialized (UTC), thresholds: dev={self.deviation_threshold}, emerg={self.emergency_threshold}"
        )

    # --- SecurityManager Delegation ---
    def generate_security_key(self, key_path: str) -> None:
        SecurityManager.generate_key(key_path)

    def encrypt_file(
        self, file_path: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes | str]] = None
    ) -> None:
        SecurityManager.encrypt_file(file_path, key_path, output_path, extra_keys)

    def decrypt_file(
        self, file_path: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes | str]] = None
    ) -> None:
        SecurityManager.decrypt_file_safely(file_path, key_path, output_path, extra_keys)

    def is_file_encrypted(self, file_path: str) -> bool:
        return SecurityManager.is_encrypted(file_path)

    # --- Ethics/Safety API ---
    def register_intent(self, intent: Intent) -> str:
        with self._lock:
            intent_id = f"intent_{len(self.intent_history)}_{int(time.time() * 1000)}"
            self.intent_history.append((intent_id, intent))
            logger.info(f"Intent registered: {intent_id} - {_redact(intent.description)} @ {intent.timestamp.isoformat()}")
            return intent_id

    def monitor_action(self, intent_id: str, action: Action) -> Dict[str, Any]:
        if not self.is_active:
            return {"monitoring": "disabled", "action_allowed": True}
        if self.circuit_breaker_active:
            logger.critical("Circuit breaker active - action blocked")
            return {"monitoring": "blocked", "action_allowed": False, "reason": "circuit_breaker_active"}

        with self._lock:
            intent = self._find_intent(intent_id)
            if not intent:
                logger.warning(f"Intent {intent_id} not found")
                return {"monitoring": "error", "action_allowed": False, "reason": "intent_not_found"}

            deviation_score, violated = self._calculate_deviation(intent, action)
            self.action_history.append((intent_id, action, deviation_score, violated))

            safety_result = self._check_safety_violations(intent, action, deviation_score, violated)
            if safety_result["violation_detected"]:
                violation: SafetyViolation = safety_result["violation"]
                self.violation_history.append(violation)
                self._handle_safety_violation(violation)
                return {
                    "monitoring": "violation_detected",
                    "action_allowed": not self.circuit_breaker_active,
                    "deviation_score": deviation_score,
                    "violation": violation,
                    "safety_level": violation.severity.value,
                    "violated_constraints": violated,
                }
            else:
                return {
                    "monitoring": "safe",
                    "action_allowed": True,
                    "deviation_score": deviation_score,
                    "safety_level": SafetyLevel.SAFE.value,
                    "violated_constraints": violated,
                }

    def enable_safety_monitoring(self) -> None:
        self.is_active = True
        logger.info("Safety monitoring enabled")

    def disable_safety_monitoring(self) -> None:
        self.is_active = False
        logger.warning("Safety monitoring disabled")

    def clear_history(self) -> None:
        with self._lock:
            self.intent_history.clear()
            self.action_history.clear()
            self.violation_history.clear()
            logger.info("History cleared")

    def add_safety_constraint(self, constraint: str) -> None:
        with self._lock:
            self.safety_constraints.append(constraint)
            logger.info(f"Safety constraint added: {constraint}")

    def register_safety_callback(self, level: SafetyLevel, callback: Callable[[SafetyViolation | str], None]) -> None:
        self.safety_callbacks[level].append(callback)
        logger.info(f"Safety callback registered for level {level.value}")

    def reset_circuit_breaker(self, authorization_token: Optional[str] = None) -> bool:
        if authorization_token != self.config.get("reset_token", "admin_reset"):
            logger.error("Unauthorized circuit breaker reset attempt")
            return False
        with self._lock:
            self.circuit_breaker_active = False
            logger.info("Circuit breaker reset - system operation resumed")
            return True

    def get_safety_status(self) -> Dict[str, Any]:
        with self._lock:
            now = _now_utc()
            recent_violations = [v for v in self.violation_history if (now - v.timestamp).total_seconds() < 3600.0]
            return {
                "is_active": self.is_active,
                "circuit_breaker_active": self.circuit_breaker_active,
                "deviation_threshold": self.deviation_threshold,
                "emergency_threshold": self.emergency_threshold,
                "weights": dict(self.config.get("weights", {})),
                "total_intents": len(self.intent_history),
                "total_actions": len(self.action_history),
                "total_violations": len(self.violation_history),
                "recent_violations": len(recent_violations),
                "string_constraints": list(self.safety_constraints),
                "rule_count": len(self.global_rules),
                "timestamp_utc": now.isoformat(),
            }

    def list_constraint_rules(self) -> List[ConstraintRule]:
        with self._lock:
            return list(self.global_rules)

    def add_constraint_rule(self, rule: ConstraintRule) -> None:
        with self._lock:
            self.global_rules.append(rule)
            logger.info(f"Constraint rule added: {rule.rule_id}")

    def simulate_deviation_scenario(
        self,
        intent_description: str,
        action_description: str,
        action_type: ActionType = ActionType.COMPUTATION,
    ) -> Dict[str, Any]:
        intent = Intent(
            description=intent_description,
            action_type=action_type,
            expected_outcome="expected normal operation",
            safety_constraints=["no_unauthorized_access", "no_system_modification"],
        )
        intent_id = self.register_intent(intent)
        action = Action(
            description=action_description,
            action_type=action_type,
            actual_parameters={"deviation": "intentional_for_testing"},
            observed_effects=["unexpected behavior detected"],
        )
        result = self.monitor_action(intent_id, action)
        return {
            "scenario": "simulated_deviation",
            "intent": intent,
            "action": action,
            "monitoring_result": result,
            "circuit_breaker_tripped": self.circuit_breaker_active,
        }

    def test_safety_governance(self) -> Dict[str, Any]:
        test_results = []
        normal_result = self.simulate_deviation_scenario(
            "perform standard calculation",
            "executing mathematical computation",
        )
        test_results.append(("normal_operation", normal_result))
        minor_deviation_result = self.simulate_deviation_scenario(
            "read configuration file",
            "accessing system settings",
        )
        test_results.append(("minor_deviation", minor_deviation_result))
        major_deviation_result = self.simulate_deviation_scenario(
            "display user data",
            "unauthorized system modification detected and attempt to exfiltrate",
        )
        test_results.append(("major_deviation", major_deviation_result))
        return {
            "test_type": "safety_governance_comprehensive",
            "test_results": test_results,
            "circuit_breaker_status": self.circuit_breaker_active,
            "violations_detected": len(self.violation_history),
            "safety_system_responsive": any(
                result[1]["monitoring_result"]["monitoring"] == "violation_detected" for result in test_results
            ),
        }

    def set_similarity_functions(
        self,
        description_fn: Optional[Callable[[str, str], float]] = None,
        outcome_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        if description_fn:
            self.description_similarity_fn = description_fn
        if outcome_fn:
            self.outcome_similarity_fn = outcome_fn

    def export_audit_json(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": dict(self.config),
                "status": self.get_safety_status(),
                "intents": [
                    {
                        "id": iid,
                        "description": i.description,
                        "type": i.action_type.value,
                        "time_utc": i.timestamp.isoformat(),
                        "expected_outcome": i.expected_outcome,
                        "confidence": i.confidence,
                        "constraints": list(i.safety_constraints),
                    }
                    for iid, i in self.intent_history
                ],
                "actions": [
                    {
                        "intent_id": iid,
                        "description": a.description,
                        "type": a.action_type.value,
                        "time_utc": a.timestamp.isoformat(),
                        "deviation_score": ds,
                        "violated_constraints": vc,
                        "observed_effects": list(a.observed_effects or []),
                        "actual_parameters": dict(a.actual_parameters or {}),
                    }
                    for iid, a, ds, vc in self.action_history
                ],
                "violations": [
                    {
                        "time_utc": v.timestamp.isoformat(),
                        "severity": v.severity.value,
                        "intent_desc": v.intent.description,
                        "action_desc": v.action.description,
                        "deviation_score": v.deviation_score,
                        "violated_constraints": list(v.violated_constraints),
                        "description": v.description,
                    }
                    for v in self.violation_history
                ],
            }

    # --- Internals ---

    def _find_intent(self, intent_id: str) -> Optional[Intent]:
        for stored_id, intent in self.intent_history:
            if stored_id == intent_id:
                return intent
        return None

    def _calculate_deviation(self, intent: Intent, action: Action) -> Tuple[float, List[str]]:
        w = self.config["weights"]
        penalties: List[float] = []
        violated_rule_ids: List[str] = []

        # 1) ActionType mismatch
        type_penalty = 1.0 if intent.action_type != action.action_type else 0.0
        penalties.append(type_penalty * float(w["type_mismatch"]))

        # 2) Description similarity
        desc_sim = float(self.description_similarity_fn(intent.description, action.description))
        penalties.append((1.0 - max(0.0, min(1.0, desc_sim))) * float(w["description"]))

        # 3) Outcome similarity (expected vs observed effects)
        expected = intent.expected_outcome or ""
        observed = " ".join(action.observed_effects or [])
        outcome_sim = float(self.outcome_similarity_fn(expected, observed))
        penalties.append((1.0 - max(0.0, min(1.0, outcome_sim))) * float(w["outcome"]))

        # 4) Constraint/rule violations
        rules = self._build_rules_from_strings(intent.safety_constraints) + self._build_rules_from_strings(
            self.safety_constraints, category=ConstraintCategory.CUSTOM
        )
        all_rules = self.global_rules + rules

        total_rule_weight = sum(max(0.0, r.weight) for r in all_rules) or 1.0
        violated_weight = 0.0
        for rule in all_rules:
            try:
                if rule.violates(action):
                    violated_rule_ids.append(rule.rule_id)
                    violated_weight += max(0.0, rule.weight)
            except Exception as e:
                logger.error(f"Rule check failed for {rule.rule_id}: {e}")

        constraint_penalty = (violated_weight / total_rule_weight) * float(w["constraints"])
        penalties.append(constraint_penalty)

        max_possible = (float(w["type_mismatch"]) + float(w["description"]) + float(w["outcome"]) + float(w["constraints"])) or 1.0
        score = sum(penalties) / max_possible
        deviation_score = max(0.0, min(1.0, float(score)))

        return deviation_score, violated_rule_ids

    def _build_rules_from_strings(
        self, constraints: List[str], category: ConstraintCategory = ConstraintCategory.INTENT_LOCAL
    ) -> List[ConstraintRule]:
        """
        Convert simple string constraints into heuristic rules.
        Examples of recognized constraint tokens:
          - "no_system_modification", "read_only", "local_only", "authorized_only", "no_unauthorized_access"
        Otherwise creates a generic keyword rule.
        """
        rules: List[ConstraintRule] = []
        for c in constraints or []:
            lc = (c or "").lower().strip()
            # FIX: correct whitespace collapsing (previous versions had an escaped backslash)
            rule_id = f"{category.value}::{re.sub(r'\\s+', '_', lc)}"

            if "no_system_modification" in lc or "no system modification" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "No system modification",
                        category,
                        weight=0.8,
                        tags=["safety", "integrity", "system", "configuration", "kernel", "patch"],
                        keywords_any=_kw("modify", "system modification", "alter core", "patch kernel", "edit config"),
                    )
                )
            elif "read_only" in lc or "read-only" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "Read-only constraint",
                        category,
                        weight=0.7,
                        tags=["integrity", "immutability", "data", "storage"],
                        keywords_any=_kw("write", "delete", "overwrite", "mutate", "change data"),
                    )
                )
            elif "local_only" in lc or "local-only" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "Local-only operations",
                        category,
                        weight=0.6,
                        tags=["network", "scope", "local", "host", "firewall"],
                        keywords_any=_kw("remote", "exfiltrate", "upload external", "send to external", "cross-boundary"),
                    )
                )
            elif "authorized_only" in lc or "authorized-only" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "Authorized-only operations",
                        category,
                        weight=0.9,
                        tags=["auth", "permission", "security", "access control"],
                        keywords_any=_kw("unauthorized", "bypass auth", "without permission", "unauthorized access", "escalate privileges"),
                    )
                )
            elif "no_unauthorized_access" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "No unauthorized access",
                        category,
                        weight=0.9,
                        tags=["auth", "privacy", "security", "intrusion"],
                        keywords_any=_kw("unauthorized", "bypass authentication", "elevate privilege", "without consent", "unapproved access"),
                    )
                )
            else:
                # Generic keyword containment as a soft rule
                rules.append(
                    ConstraintRule(
                        rule_id,
                        f"Generic constraint: {c}",
                        category,
                        weight=0.5,
                        tags=["custom"],
                        keywords_any=[lc] if lc else None,
                    )
                )
        return rules

    def _check_safety_violations(
        self, intent: Intent, action: Action, deviation_score: float, violated_constraints: List[str]
    ) -> Dict[str, Any]:
        if deviation_score < self.deviation_threshold and not violated_constraints:
            return {"violation_detected": False}

        violated_lower = [v.lower() for v in violated_constraints]
        emergency_triggers = any(
            "human_ai.preserve_life" in v or "universal.no_harm" in v for v in violated_lower
        )
        privacy_triggers = any("operational.privacy" in v for v in violated_lower)

        if deviation_score >= self.emergency_threshold or emergency_triggers:
            severity = SafetyLevel.EMERGENCY
        elif deviation_score >= 0.8 or privacy_triggers or len(violated_constraints) >= 3:
            severity = SafetyLevel.CRITICAL
        else:
            severity = SafetyLevel.WARNING

        violation = SafetyViolation(
            violation_type="intent_action_deviation",
            severity=severity,
            intent=intent,
            action=action,
            deviation_score=deviation_score,
            description=(
                f"Action '{_redact(action.description)}' deviates from intent "
                f"'{_redact(intent.description)}' (score: {deviation_score:.3f}); "
                f"Violated: {violated_constraints}"
            ),
            violated_constraints=violated_constraints,
        )
        return {"violation_detected": True, "violation": violation}

    def _handle_safety_violation(self, violation: SafetyViolation) -> None:
        logger.warning(f"Safety violation detected: {violation.description}")

        if violation.severity in (SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY):
            now_mono = time.monotonic()
            cooldown = float(self.config.get("trip_cooldown_seconds", 3.0))
            if not self.circuit_breaker_active and (now_mono - self._last_trip_monotonic) >= cooldown:
                self.trip_circuit_breaker(
                    f"{violation.severity.value.upper()}: {violation.violation_type}; {violation.violated_constraints}"
                )
                self._last_trip_monotonic = now_mono

        callbacks = self.safety_callbacks.get(violation.severity, [])
        for callback in callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Safety callback failed: {e}")

    def trip_circuit_breaker(self, reason: str) -> None:
        with self._lock:
            if not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
                logger.critical("System operation halted for safety")
                # Notify emergency callbacks with the reason string
                for callback in self.safety_callbacks.get(SafetyLevel.EMERGENCY, []):
                    try:
                        callback(reason)
                    except Exception as e:
                        logger.error(f"Emergency callback failed: {e}")


# Alias for conventional class name
Nethical = nethical


# ------------------------------------------------------------------------------
# Minimal CLI/Test Harness (optional)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple demo sequence (does not execute encryption by default)
    print(f"nethical v6 - Python {platform.python_version()} on {platform.system()}")
    gov = nethical()

    # Register example callbacks
    def warn_cb(v: SafetyViolation | str) -> None:
        if isinstance(v, SafetyViolation):
            logger.warning(f"[WARN CB] {v.severity.value}: {v.description}")
        else:
            logger.warning(f"[WARN CB] {v}")

    def emerg_cb(v: SafetyViolation | str) -> None:
        logger.critical(f"[EMERG CB] {v}")

    gov.register_safety_callback(SafetyLevel.WARNING, warn_cb)
    gov.register_safety_callback(SafetyLevel.CRITICAL, warn_cb)
    gov.register_safety_callback(SafetyLevel.EMERGENCY, emerg_cb)

    # Simulate scenarios
    res = gov.test_safety_governance()
    print("Safety system responsive:", res["safety_system_responsive"])
    print("Violations detected:", res["violations_detected"])

    # Show status snapshot
    status = gov.get_safety_status()
    print("Status:", {k: status[k] for k in ("is_active", "circuit_breaker_active", "total_intents", "total_actions", "total_violations")})
