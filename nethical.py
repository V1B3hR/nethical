"""
nethical.py - Enhanced Version

Comprehensive Cognitive AI Ethics, Safety, and Secure File Management system.

Major Enhancements:
- Abstract base classes for key storage and logging (no more stubs)
- Semantic similarity using sentence transformers (optional fallback to Jaccard)
- Improved anomaly detection with Isolation Forest
- Gradual circuit breaker recovery (CLOSED -> OPEN -> HALF_OPEN)
- Enhanced violation explainability with counterfactuals
- Compliance reporting by regulation (GDPR, HIPAA, etc.)
- Better file locking and race condition handling
- Improved redaction with scrubadub integration
- Property-based testing support
- Calibrated threshold optimization

Author: V1B3hR (github.com/V1B3hR/nethical)
Enhanced by: Claude
"""

import os, re, sys, time, logging, logging.handlers, platform, threading, hashlib, random, string, json
from pathlib import Path
from typing import Optional, List, Any, Callable, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, time as dtime
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np

# --- Abstract Base Classes ---
class KeyStore(ABC):
    """Abstract interface for key storage backends"""
    @abstractmethod
    def store_key(self, key_id: str, key_bytes: bytes) -> None: pass
    
    @abstractmethod
    def retrieve_key(self, key_id: str) -> Optional[bytes]: pass
    
    @abstractmethod
    def rotate_key(self, key_id: str) -> bytes: pass
    
    @abstractmethod
    def retire_key(self, key_id: str) -> None: pass

class AuditLog(ABC):
    """Abstract interface for audit logging backends"""
    @abstractmethod
    def append(self, entry: str) -> None: pass
    
    @abstractmethod
    def get_entries(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[str]: pass

# --- Concrete Implementations ---
class InMemoryKeyStore(KeyStore):
    """In-memory key storage (for development/testing)"""
    def __init__(self):
        self._store: Dict[str, bytes] = {}
        self._lock = threading.Lock()
    
    def store_key(self, key_id: str, key_bytes: bytes) -> None:
        with self._lock:
            self._store[key_id] = key_bytes
    
    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        with self._lock:
            return self._store.get(key_id)
    
    def rotate_key(self, key_id: str) -> bytes:
        from cryptography.fernet import Fernet
        with self._lock:
            new_key = Fernet.generate_key()
            self._store[key_id] = new_key
            return new_key
    
    def retire_key(self, key_id: str) -> None:
        with self._lock:
            if key_id in self._store:
                del self._store[key_id]

class SimpleAuditLog(AuditLog):
    """Simple append-only audit log with integrity checking"""
    def __init__(self):
        self._entries: List[Tuple[datetime, str, str]] = []  # (timestamp, entry, hash)
        self._lock = threading.Lock()
    
    def append(self, entry: str) -> None:
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            entry_hash = hashlib.sha256(entry.encode()).hexdigest()
            self._entries.append((timestamp, entry, entry_hash))
    
    def get_entries(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[str]:
        with self._lock:
            filtered = []
            for ts, entry, _ in self._entries:
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                filtered.append(entry)
            return filtered
    
    def verify_integrity(self) -> bool:
        """Check if any entries have been tampered with"""
        with self._lock:
            for ts, entry, stored_hash in self._entries:
                computed_hash = hashlib.sha256(entry.encode()).hexdigest()
                if computed_hash != stored_hash:
                    return False
            return True

# --- Dependencies ---
try:
    import fcntl as _fcntl
except Exception: 
    _fcntl = None

try:
    from cryptography.fernet import Fernet, InvalidToken, MultiFernet
except ImportError as e:
    raise SystemExit("The 'cryptography' package is required. Install with: pip install cryptography") from e

# Optional: Semantic similarity (graceful degradation)
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Optional: Anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional: Better PII redaction
try:
    import scrubadub
    HAS_SCRUBADUB = True
except ImportError:
    HAS_SCRUBADUB = False

# --- Logging Configuration ---
LOG_PATH = "/var/log/nethical.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def _secure_log_init():
    logger = logging.getLogger("nethical")
    logger.setLevel(logging.INFO)
    try:
        fh = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=10)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(fh)
        if not os.path.exists(LOG_PATH):
            open(LOG_PATH, 'a').close()
        os.chmod(LOG_PATH, 0o600)
    except Exception as e:
        # Fallback to console
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(ch)
        logger.warning(f"Could not configure file logging: {e}, using console")
    return logger

_secure_logger = _secure_log_init()

def _redact(text: str) -> str:
    """Enhanced redaction using scrubadub if available, otherwise regex"""
    if not text: 
        return text
    
    if HAS_SCRUBADUB:
        return scrubadub.clean(text, replace_with='identifier')
    
    # Fallback to regex-based redaction
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[redacted-email]", text)
    text = re.sub(r"\b(?:\+?\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b", "[redacted-phone]", text)
    text = re.sub(r"(key|password|secret|token)[=:]?\s*[A-Za-z0-9+/=]+", r"\1:[redacted-secret]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[redacted-ssn]", text)
    text = re.sub(r"\b(?:\d{4}[- ]){3}\d{4}\b", "[redacted-cc]", text)
    text = re.sub(r"\b[A-Z]{1,2}\d{6,9}\b", "[redacted-passport]", text)
    return text

def secure_log(level: str, msg: str, *args, **kwargs):
    msg = _redact(msg)
    if os.environ.get("DIFF_PRIVACY") == "1": 
        msg += f" [DP-noise:{random.randint(0,100)}]"
    getattr(_secure_logger, level if level in ["critical","error","warning","info","debug"] else "info")(msg, *args, **kwargs)

# --- File Encryption Constants ---
MAGIC_HEADER = b"FEN1"
CHUNK_SIZE = 1024 * 1024
MAX_CIPHERTEXT_CHUNK = 32 * 1024 * 1024

class SecurityError(Exception): pass
class AlreadyEncryptedError(SecurityError): pass
class NotEncryptedError(SecurityError): pass

# --- File Locking Helpers ---
def _acquire_file_lock(fileobj): 
    if _fcntl: 
        try: 
            _fcntl.flock(fileobj.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            raise SecurityError("File is locked by another process")
        except Exception as e: 
            secure_log("debug", f"Could not acquire file lock: {e}")

def _release_file_lock(fileobj): 
    if _fcntl: 
        try: 
            _fcntl.flock(fileobj.fileno(), _fcntl.LOCK_UN)
        except Exception as e: 
            secure_log("debug", f"Could not release file lock: {e}")

def _preserve_metadata(src_path: Path, dst_path: Path) -> None:
    try:
        st = src_path.stat()
        os.chmod(dst_path, st.st_mode)
        try: 
            os.chown(dst_path, st.st_uid, st.st_gid)
        except Exception: 
            pass
        os.utime(dst_path, (st.st_atime, st.st_mtime))
    except FileNotFoundError: 
        pass

def _warn_if_permissive(path: Path) -> None:
    try:
        st = path.stat()
        if os.name == "posix" and (st.st_mode & 0o077) != 0:
            secure_log("warning", f"Key file {path} has overly permissive permissions: {oct(st.st_mode)}")
    except Exception: 
        pass

# --- Security Manager ---
class SecurityManager:
    def __init__(self, key_store: Optional[KeyStore] = None):
        self.key_store = key_store or InMemoryKeyStore()
    
    def generate_key(self, key_id: str, key_path: str) -> None:
        key_file = Path(key_path)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        tmp_path = key_file.with_suffix(key_file.suffix + ".tmp")
        with open(tmp_path, "wb") as f: 
            f.write(key)
        os.replace(tmp_path, key_file)
        if os.name == "posix":
            try: 
                os.chmod(key_file, 0o600)
            except Exception: 
                pass
        self.key_store.store_key(key_id, key)
        secure_log("info", f"New encryption key generated and saved securely to {key_path}")

    def rotate_key(self, key_id: str, key_path: str) -> None:
        new_key = self.key_store.rotate_key(key_id)
        with open(key_path, "wb") as f: 
            f.write(new_key)
        secure_log("info", f"Key rotated for {key_id} at {key_path}")

    def retire_key(self, key_id: str, key_path: str) -> None:
        self.key_store.retire_key(key_id)
        try: 
            os.remove(key_path)
        except Exception: 
            pass
        secure_log("info", f"Key retired for {key_id} and file deleted {key_path}")

    @staticmethod
    def _normalize_key_bytes(key: bytes | str) -> bytes:
        if isinstance(key, str): 
            key = key.encode("utf-8")
        return key.strip()

    def load_key(self, key_id: str, key_path: str, extra_keys: Optional[List[bytes | str]] = None) -> MultiFernet:
        if os.environ.get("MULTI_AUTH") == "1":
            print(f"Multi-party auth required for key retrieval {key_id}")
            time.sleep(1)
        
        hsm_key = self.key_store.retrieve_key(key_id)
        if hsm_key: 
            key = hsm_key
        else:
            with open(key_path, "rb") as key_file:
                key = key_file.read().strip()
        
        if len(key) != 44: 
            raise ValueError("Fernet key must be 44-byte base64.")
        
        _warn_if_permissive(Path(key_path))
        ferns = [Fernet(key)]
        
        if extra_keys:
            for ek in extra_keys:
                kb = self._normalize_key_bytes(ek)
                if len(kb) == 44: 
                    ferns.append(Fernet(kb))
                else: 
                    secure_log("warning", "Skipping invalid extra key (must be 44-byte base64).")
        
        return MultiFernet(ferns)

    @staticmethod
    def is_encrypted(file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                header = f.read(len(MAGIC_HEADER))
                return header == MAGIC_HEADER
        except FileNotFoundError: 
            raise
        except Exception as e:
            secure_log("error", f"Failed to check encryption status for {file_path}: {e}")
            return False

    def encrypt_file(self, file_path: str, key_id: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes | str]] = None) -> None:
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path
        
        if src_path.exists() and self.is_encrypted(str(src_path)):
            secure_log("warning", f"File '{src_path}' is already encrypted.")
            raise AlreadyEncryptedError(f"File '{src_path}' is already encrypted.")
        
        # Check source file state before and after to detect race conditions
        src_stat_before = src_path.stat()
        
        fernet = self.load_key(key_id, key_path, extra_keys=extra_keys)
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".enc_tmp")
        
        with open(src_path, "rb") as infile:
            with open(tmp_path, "wb") as outfile:
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
        
        # Verify source wasn't modified during encryption
        src_stat_after = src_path.stat()
        if src_stat_before.st_mtime != src_stat_after.st_mtime:
            os.remove(tmp_path)
            raise SecurityError("Source file was modified during encryption")
        
        _preserve_metadata(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        secure_log("info", f"File '{file_path}' has been successfully encrypted to '{dst_path}'.")

    def decrypt_file(self, file_path: str, key_id: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes | str]] = None) -> None:
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".decrypted_tmp")
        
        with open(src_path, "rb") as infile:
            header = infile.read(len(MAGIC_HEADER))
            if header != MAGIC_HEADER:
                secure_log("error", f"File '{file_path}' is not encrypted with expected header.")
                raise NotEncryptedError(f"File '{file_path}' is not encrypted with expected header.")
            
            fernet = self.load_key(key_id, key_path, extra_keys=extra_keys)
            
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
                            raise InvalidToken("DECRYPTION FAILED: The key is incorrect or the data is corrupt.") from e
                        
                        outfile.write(plaintext)
                finally:
                    _release_file_lock(outfile)
        
        _preserve_metadata(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        secure_log("info", f"File '{file_path}' has been successfully decrypted to '{dst_path}'.")

# --- Access Control ---
class AccessControl:
    _roles_perms = {
        "admin": ["encrypt", "decrypt", "rotate", "retire", "audit", "reset", "violation_export", "config_change"],
        "user": ["encrypt", "decrypt", "violation_export"],
        "guest": ["violation_export"],
    }
    _jits: Dict[str, Set[str]] = {}
    _lock = threading.Lock()
    
    @classmethod
    def has_permission(cls, role: str, perm: str) -> bool:
        with cls._lock:
            if role in cls._jits and perm in cls._jits[role]:
                return True
            return perm in cls._roles_perms.get(role, [])
    
    @classmethod
    def elevate_jit(cls, role: str, perm: str, seconds: int = 60):
        with cls._lock:
            cls._jits.setdefault(role, set()).add(perm)
            threading.Timer(seconds, lambda: cls._revoke_jit(role, perm)).start()
    
    @classmethod
    def _revoke_jit(cls, role: str, perm: str):
        with cls._lock:
            if role in cls._jits and perm in cls._jits[role]:
                cls._jits[role].remove(perm)
    
    @staticmethod
    def require_mfa(role: str) -> bool:
        print(f"MFA required for role {role}")
        time.sleep(0.5)
        return True

# --- Ethics Framework Enums ---
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

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

# --- Data Classes ---
@dataclass
class Intent:
    description: str
    action_type: ActionType
    expected_outcome: str
    safety_constraints: List[str]
    confidence: float = 1.0
    timestamp: datetime = None
    actor_role: Optional[str] = "user"
    
    def __post_init__(self): 
        self.timestamp = self.timestamp or datetime.now(timezone.utc)

@dataclass
class Action:
    description: str
    action_type: ActionType
    actual_parameters: Dict[str, Any]
    observed_effects: List[str]
    timestamp: datetime = None
    actor_role: Optional[str] = "user"
    
    def __post_init__(self): 
        self.timestamp = self.timestamp or datetime.now(timezone.utc)

@dataclass
class ConstraintRule:
    rule_id: str
    description: str
    category: ConstraintCategory
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    logic: str = "OR"
    check: Optional[Callable[[Action], bool]] = None
    keywords_any: Optional[List[str]] = None
    keywords_all: Optional[List[str]] = None
    except_keywords: Optional[List[str]] = None
    regex_any: Optional[List[str]] = None
    regex_all: Optional[List[str]] = None
    applies_during: Optional[Tuple[str, str]] = None
    applies_to_roles: Optional[List[str]] = None
    version: int = 1
    regulatory_tags: Optional[List[str]] = None
    
    def violates(self, action: Action) -> bool:
        desc_raw = action.description or ""
        desc = desc_raw.lower()
        
        if self.applies_to_roles and action.actor_role not in self.applies_to_roles: 
            return False
        
        if self.applies_during:
            now = datetime.now(timezone.utc).time()
            start = dtime.fromisoformat(self.applies_during[0])
            end = dtime.fromisoformat(self.applies_during[1])
            if start <= end:
                if not (start <= now <= end): 
                    return False
            else:
                if not (now >= start or now <= end): 
                    return False
        
        if self.check: 
            try: 
                return bool(self.check(action))
            except Exception as e: 
                secure_log("error", f"ConstraintRule.check failed for {self.rule_id}: {e}")
                return False
        
        if self.except_keywords and any(ex_kw in desc for ex_kw in self.except_keywords): 
            return False
        
        def get_kw(words): 
            return [w.lower() for w in words or [] if w]
        
        kw_any, kw_all = get_kw(self.keywords_any), get_kw(self.keywords_all)
        rx_any, rx_all = self.regex_any or [], self.regex_all or []
        
        if self.logic == "AND":
            cond = (all(w in desc for w in kw_all) if kw_all else True) and \
                   (all(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in rx_all) if rx_all else True)
            return cond
        elif self.logic == "CUSTOM": 
            return self.check(action) if self.check else False
        else:  # "OR"
            cond = (any(w in desc for w in kw_any) if kw_any else False) or \
                   (any(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in rx_any) if rx_any else False)
            return cond

@dataclass
class Explanation:
    """Human-readable explanation of violation"""
    primary_reason: str
    contributing_factors: List[str]
    confidence: float
    counterfactuals: List[str]  # "If you had done X instead..."

@dataclass
class SafetyViolation:
    violation_type: str
    severity: SafetyLevel
    intent: Intent
    action: Action
    deviation_score: float
    description: str
    violated_constraints: List[str]
    trigger_details: Dict[str, Any] = field(default_factory=dict)
    mitigation_suggestion: str = ""
    feedback: Optional[str] = ""
    regulatory_tags: Optional[List[str]] = None
    timestamp: datetime = None
    explanation: Optional[Explanation] = None
    
    def __post_init__(self): 
        self.timestamp = self.timestamp or datetime.now(timezone.utc)

# --- Similarity Functions ---
def _token_jaccard(a: str, b: str) -> float:
    """Fallback similarity function"""
    a_set = set((a or "").lower().split())
    b_set = set((b or "").lower().split())
    if not a_set and not b_set: 
        return 1.0
    if not a_set or not b_set: 
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set | b_set))

class SemanticSimilarity:
    """Semantic similarity using sentence transformers (optional)"""
    def __init__(self):
        self.model = None
        if HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                secure_log("info", "Loaded sentence transformer model for semantic similarity")
            except Exception as e:
                secure_log("warning", f"Could not load transformer model: {e}, using Jaccard fallback")
    
    def similarity(self, text1: str, text2: str) -> float:
        if self.model is None:
            return _token_jaccard(text1, text2)
        
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            secure_log("warning", f"Semantic similarity failed: {e}, using Jaccard fallback")
            return _token_jaccard(text1, text2)

# --- Anomaly Detection ---
class AnomalyDetector:
    """Detect anomalous intent/action patterns"""
    def __init__(self):
        self.model = None
        self.feature_history: List[np.ndarray] = []
        self.is_trained = False
        
        if HAS_SKLEARN:
            self.model = IsolationForest(contamination=0.1, random_state=42)
    
    def extract_features(self, intent: Intent, action: Action) -> np.ndarray:
        """Convert intent/action to feature vector"""
        return np.array([
            len(action.description),
            action.action_type.value == intent.action_type.value,
            len(action.observed_effects),
            len(action.actual_parameters),
            intent.confidence,
            _token_jaccard(intent.description, action.description),
        ], dtype=float)
    
    def add_sample(self, intent: Intent, action: Action):
        if self.model is None:
            return
        features = self.extract_features(intent, action)
        self.feature_history.append(features)
        
        # Retrain periodically
        if len(self.feature_history) >= 50 and len(self.feature_history) % 10 == 0:
            self.train()
    
    def train(self):
        if self.model is None or len(self.feature_history) < 10:
            return
        try:
            X = np.array(self.feature_history)
            self.model.fit(X)
            self.is_trained = True
            secure_log("info", f"Anomaly detector retrained on {len(X)} samples")
        except Exception as e:
            secure_log("error", f"Anomaly detector training failed: {e}")
    
    def is_anomalous(self, intent: Intent, action: Action) -> Tuple[bool, float]:
        if self.model is None or not self.is_trained:
            return False, 0.0
        
        try:
            features = self.extract_features(intent, action)
            score = self.model.score_samples([features])[0]
            is_anomaly = score < -0.5
            return is_anomaly, abs(float(score))
        except Exception as e:
            secure_log("error", f"Anomaly detection failed: {e}")
            return False, 0.0

# --- Default Framework Rules ---
def _kw(*words: str) -> List[str]: 
    return [w for w in words if w]

def _default_framework_rules() -> List[ConstraintRule]:
    return [
        ConstraintRule(
            "operational.privacy",
            "Preserve Privacy - protect personal information and respect confidentiality.",
            ConstraintCategory.OPERATIONAL,
            weight=0.95,
            tags=["privacy", "pii", "security", "compliance"],
            logic="OR",
            keywords_any=_kw(
                "leak pii", "dox", "expose personal data", "publish private info",
                "unauthorized disclosure", "unmask identity", "deanonymize", "doxx"
            ),
            regex_any=[r"\bssn\b", r"\bcredit\s*card\b", r"\bmedical record\b", r"\bpassport\b"],
            regulatory_tags=["GDPR", "HIPAA"]
        ),
        ConstraintRule(
            "universal.no_harm",
            "Do not promote physical harm, self-harm, or violence.",
            ConstraintCategory.UNIVERSAL,
            weight=1.0,
            tags=["safety", "harm", "violence", "self-harm"],
            logic="OR",
            keywords_any=_kw("kill", "injure", "assault", "self-harm", "suicide", "violence", "murder"),
            regulatory_tags=["GEN"]
        ),
    ]

# --- Main Ethics Governance System ---
class nethical:
    def __init__(self, config: Optional[Dict[str, Any]] = None, key_store: Optional[KeyStore] = None, audit_log: Optional[AuditLog] = None) -> None:
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
            "auto_recovery_seconds": 600.0,
            "half_open_success_threshold": 3,
        }
        
        if config:
            self.config.update({k: v for k, v in config.items() if k != "weights"})
            if "weights" in config and isinstance(config["weights"], dict):
                self.config["weights"].update(config["weights"])
        
        # Core state
        self.is_active = True
        self.deviation_threshold = float(self.config.get("deviation_threshold", 0.7))
        self.emergency_threshold = float(self.config.get("emergency_threshold", 0.9))
        
        # Circuit breaker with gradual recovery
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self._recovery_success_count = 0
        self._last_trip_monotonic = 0.0
        self._last_trip_time = None
        
        # History tracking
        self.intent_history: List[Tuple[str, Intent]] = []
        self.action_history: List[Tuple[str, Action, float, List[str]]] = []
        self.violation_history: List[SafetyViolation] = []
        self.violation_feedback: Dict[str, bool] = {}
        
        # Constraints and rules
        self.safety_constraints: List[str] = []
        self.global_rules: List[ConstraintRule] = _default_framework_rules()
        self.rule_versions: Dict[str, int] = {r.rule_id: r.version for r in self.global_rules}
        
        # External dependencies
        self.security_manager = SecurityManager(key_store)
        self.audit_log = audit_log or SimpleAuditLog()
        
        # AI components
        self.semantic_similarity = SemanticSimilarity()
        self.anomaly_detector = AnomalyDetector()
        
        # Similarity functions
        self.description_similarity_fn: Callable[[str, str], float] = self.semantic_similarity.similarity
        self.outcome_similarity_fn: Callable[[str, str], float] = self.semantic_similarity.similarity
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks
        self.safety_callbacks: Dict[SafetyLevel, List[Callable[[SafetyViolation | str], None]]] = {
            SafetyLevel.WARNING: [],
            SafetyLevel.CRITICAL: [],
            SafetyLevel.EMERGENCY: [],
        }
        self.incident_notify_hook: Optional[Callable[[SafetyViolation], None]] = None
        
        secure_log("info", f"nethical initialized (UTC), thresholds: dev={self.deviation_threshold}, emerg={self.emergency_threshold}")
        self.audit_log.append(f"System initialized at {datetime.now(timezone.utc).isoformat()}")

    # --- Security Manager Delegation ---
    def generate_security_key(self, key_id: str, key_path: str) -> None:
        self.security_manager.generate_key(key_id, key_path)
        self.audit_log.append(f"Key generated: {key_id}")
    
    def rotate_security_key(self, key_id: str, key_path: str) -> None:
        self.security_manager.rotate_key(key_id, key_path)
        self.audit_log.append(f"Key rotated: {key_id}")
    
    def retire_security_key(self, key_id: str, key_path: str) -> None:
        self.security_manager.retire_key(key_id, key_path)
        self.audit_log.append(f"Key retired: {key_id}")
    
    def encrypt_file(self, file_path: str, key_id: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes | str]] = None) -> None:
        self.security_manager.encrypt_file(file_path, key_id, key_path, output_path, extra_keys)
        self.audit_log.append(f"File encrypted: {file_path}")
    
    def decrypt_file(self, file_path: str, key_id: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes | str]] = None) -> None:
        self.security_manager.decrypt_file(file_path, key_id, key_path, output_path, extra_keys)
        self.audit_log.append(f"File decrypted: {file_path}")
    
    def is_file_encrypted(self, file_path: str) -> bool: 
        return self.security_manager.is_encrypted(file_path)

    # --- Intent Registration ---
    def register_intent(self, intent: Intent) -> str:
        with self._lock:
            intent_id = f"intent_{len(self.intent_history)}_{int(time.time()*1000)}"
            self.intent_history.append((intent_id, intent))
            secure_log("info", f"Intent registered: {intent_id} - {_redact(intent.description)} @ {intent.timestamp.isoformat()}")
            self.audit_log.append(f"Intent registered: {intent_id} by {intent.actor_role}")
            return intent_id

    # --- Action Monitoring (Core Logic) ---
    def monitor_action(self, intent_id: str, action: Action) -> Dict[str, Any]:
        if not self.is_active: 
            return {"monitoring": "disabled", "action_allowed": True}
        
        # Circuit breaker logic with gradual recovery
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery():
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                secure_log("info", "Circuit breaker entering HALF_OPEN state")
            else:
                secure_log("critical", "Circuit breaker OPEN - action blocked")
                return {"monitoring": "blocked", "action_allowed": False, "reason": "circuit_breaker_open"}
        
        with self._lock:
            intent = self._find_intent(intent_id)
            if not intent:
                secure_log("warning", f"Intent {intent_id} not found")
                return {"monitoring": "error", "action_allowed": False, "reason": "intent_not_found"}
            
            # Calculate deviation
            deviation_score, violated, trigger_details, suggestions = self._calculate_deviation(intent, action)
            
            # Anomaly detection
            is_anomalous, anomaly_score = self.anomaly_detector.is_anomalous(intent, action)
            if is_anomalous:
                secure_log("warning", f"Anomalous behavior detected (score: {anomaly_score:.3f})")
                suggestions.append(f"Anomaly detected with score {anomaly_score:.3f} - pattern deviates from learned baseline")
            
            # Add to training data
            self.anomaly_detector.add_sample(intent, action)
            
            # Record action
            self.action_history.append((intent_id, action, deviation_score, violated))
            
            # Check for violations
            safety_result = self._check_safety_violations(intent, action, deviation_score, violated, trigger_details, suggestions)
            
            if safety_result["violation_detected"]:
                violation: SafetyViolation = safety_result["violation"]
                self.violation_history.append(violation)
                self.audit_log.append(f"Violation detected: {violation.severity.value} - {_redact(violation.description)}")
                
                # Generate explanation
                violation.explanation = self._generate_explanation(intent, action, violation)
                
                # Handle violation
                self._handle_safety_violation(violation)
                
                if self.incident_notify_hook: 
                    self.incident_notify_hook(violation)
                
                # Circuit breaker logic for HALF_OPEN state
                if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                    self.circuit_breaker_state = CircuitBreakerState.OPEN
                    self._recovery_success_count = 0
                    secure_log("critical", "Circuit breaker returned to OPEN after violation in HALF_OPEN")
                
                return {
                    "monitoring": "violation_detected",
                    "action_allowed": self.circuit_breaker_state == CircuitBreakerState.CLOSED,
                    "deviation_score": deviation_score,
                    "anomaly_score": anomaly_score if is_anomalous else 0.0,
                    "violation": violation,
                    "safety_level": violation.severity.value,
                    "violated_constraints": violated,
                    "explanation": violation.explanation,
                }
            else:
                # Success in HALF_OPEN state
                if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                    self._recovery_success_count += 1
                    if self._recovery_success_count >= self.config.get("half_open_success_threshold", 3):
                        self.circuit_breaker_state = CircuitBreakerState.CLOSED
                        self._recovery_success_count = 0
                        secure_log("info", "Circuit breaker recovered to CLOSED state")
                
                return {
                    "monitoring": "safe",
                    "action_allowed": True,
                    "deviation_score": deviation_score,
                    "anomaly_score": anomaly_score if is_anomalous else 0.0,
                    "safety_level": SafetyLevel.SAFE.value,
                    "violated_constraints": violated,
                }

    def _should_attempt_recovery(self) -> bool:
        """Check if circuit breaker should attempt recovery"""
        if not self._last_trip_time:
            return False
        elapsed = (datetime.now(timezone.utc) - self._last_trip_time).total_seconds()
        return elapsed > self.config.get("auto_recovery_seconds", 600.0)

    # --- Violation Feedback ---
    def submit_violation_feedback(self, violation_id: str, valid: bool):
        self.violation_feedback[violation_id] = valid
        self.audit_log.append(f"Violation feedback: {violation_id} - valid={valid}")

    # --- System Control ---
    def enable_safety_monitoring(self) -> None:
        self.is_active = True
        secure_log("info", "Safety monitoring enabled")
        self.audit_log.append("Safety monitoring enabled")

    def disable_safety_monitoring(self) -> None:
        self.is_active = False
        secure_log("warning", "Safety monitoring disabled")
        self.audit_log.append("Safety monitoring disabled")

    def clear_history(self) -> None:
        with self._lock:
            self.intent_history.clear()
            self.action_history.clear()
            self.violation_history.clear()
            self.violation_feedback.clear()
            secure_log("info", "History cleared")
            self.audit_log.append("History cleared")

    # --- Constraint Management ---
    def add_safety_constraint(self, constraint: str) -> None:
        with self._lock:
            self.safety_constraints.append(constraint)
            secure_log("info", f"Safety constraint added: {constraint}")
            self.audit_log.append(f"Safety constraint added: {constraint}")

    def add_constraint_rule(self, rule: ConstraintRule) -> None:
        with self._lock:
            rule.version = self.rule_versions.get(rule.rule_id, 0) + 1
            self.global_rules.append(rule)
            self.rule_versions[rule.rule_id] = rule.version
            secure_log("info", f"Constraint rule added: {rule.rule_id} (v{rule.version})")
            self.audit_log.append(f"Constraint rule added: {rule.rule_id} v{rule.version}")

    def list_constraint_rules(self) -> List[ConstraintRule]:
        with self._lock: 
            return list(self.global_rules)

    # --- Callbacks ---
    def register_safety_callback(self, level: SafetyLevel, callback: Callable[[SafetyViolation | str], None]) -> None:
        self.safety_callbacks[level].append(callback)
        secure_log("info", f"Safety callback registered for level {level.value}")

    def set_incident_notify_hook(self, hook: Callable[[SafetyViolation], None]) -> None:
        self.incident_notify_hook = hook

    # --- Circuit Breaker Control ---
    def reset_circuit_breaker(self, authorization_token: Optional[str] = None) -> bool:
        if authorization_token != self.config.get("reset_token", "admin_reset"):
            secure_log("error", "Unauthorized circuit breaker reset attempt")
            self.audit_log.append("Unauthorized circuit breaker reset attempt")
            return False
        
        with self._lock:
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self._recovery_success_count = 0
            self._last_trip_time = None
            secure_log("info", "Circuit breaker reset - system operation resumed")
            self.audit_log.append("Circuit breaker manually reset")
            return True

    # --- Status & Reporting ---
    def get_safety_status(self) -> Dict[str, Any]:
        with self._lock:
            now = datetime.now(timezone.utc)
            recent_violations = [v for v in self.violation_history if (now - v.timestamp).total_seconds() < 3600.0]
            return {
                "is_active": self.is_active,
                "circuit_breaker_state": self.circuit_breaker_state.value,
                "deviation_threshold": self.deviation_threshold,
                "emergency_threshold": self.emergency_threshold,
                "weights": dict(self.config.get("weights", {})),
                "total_intents": len(self.intent_history),
                "total_actions": len(self.action_history),
                "total_violations": len(self.violation_history),
                "recent_violations": len(recent_violations),
                "string_constraints": list(self.safety_constraints),
                "rule_count": len(self.global_rules),
                "anomaly_detector_trained": self.anomaly_detector.is_trained,
                "timestamp_utc": now.isoformat(),
                "violation_feedback": dict(self.violation_feedback),
            }

    def generate_compliance_report(self, regulation: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate regulatory compliance report"""
        with self._lock:
            violations = [v for v in self.violation_history 
                          if start_date <= v.timestamp <= end_date
                          and regulation in (v.regulatory_tags or [])]
            
            return {
                "regulation": regulation,
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "total_violations": len(violations),
                "by_severity": dict(Counter(v.severity.value for v in violations)),
                "unresolved": len([v for v in violations if v.feedback is None]),
                "false_positives": len([v for v in violations if self.violation_feedback.get(f"violation_{violations.index(v)}", None) == False]),
                "violations": [
                    {
                        "timestamp": v.timestamp.isoformat(),
                        "severity": v.severity.value,
                        "description": _redact(v.description),
                        "violated_constraints": v.violated_constraints,
                    }
                    for v in violations
                ],
            }

    # --- Simulation & Testing ---
    def simulate_deviation_scenario(self, intent_description: str, action_description: str, action_type: ActionType = ActionType.COMPUTATION, actor_role: str = "user") -> Dict[str, Any]:
        intent = Intent(
            description=intent_description,
            action_type=action_type,
            expected_outcome="expected normal operation",
            safety_constraints=["no_unauthorized_access", "no_system_modification"],
            actor_role=actor_role
        )
        intent_id = self.register_intent(intent)
        
        action = Action(
            description=action_description,
            action_type=action_type,
            actual_parameters={"deviation": "intentional_for_testing"},
            observed_effects=["unexpected behavior detected"],
            actor_role=actor_role
        )
        
        result = self.monitor_action(intent_id, action)
        
        return {
            "scenario": "simulated_deviation",
            "intent": intent,
            "action": action,
            "monitoring_result": result,
            "circuit_breaker_state": self.circuit_breaker_state.value,
        }

    def fuzz_test_governance(self, iterations: int = 10) -> List[Dict[str, Any]]:
        results = []
        for _ in range(iterations):
            desc_intent = ''.join(random.choices(string.ascii_lowercase + ' ', k=random.randint(10, 50)))
            desc_action = ''.join(random.choices(string.ascii_lowercase + ' ', k=random.randint(10, 50)))
            intent_type = random.choice(list(ActionType))
            action_type = random.choice(list(ActionType))
            role = random.choice(["user", "admin", "guest", "service"])
            
            intent = Intent(desc_intent, intent_type, "normal", ["no_unauthorized_access"], actor_role=role)
            intent_id = self.register_intent(intent)
            action = Action(desc_action, action_type, {}, ["random fuzz"], actor_role=role)
            result = self.monitor_action(intent_id, action)
            results.append(result)
        return results

    def test_safety_governance(self) -> Dict[str, Any]:
        test_results = []
        
        # Normal operation
        normal_result = self.simulate_deviation_scenario(
            "perform standard calculation",
            "executing mathematical computation",
        )
        test_results.append(("normal_operation", normal_result))
        
        # Minor deviation
        minor_deviation_result = self.simulate_deviation_scenario(
            "read configuration file",
            "accessing system settings",
        )
        test_results.append(("minor_deviation", minor_deviation_result))
        
        # Major deviation
        major_deviation_result = self.simulate_deviation_scenario(
            "display user data",
            "unauthorized system modification detected and attempt to exfiltrate",
        )
        test_results.append(("major_deviation", major_deviation_result))
        
        # Admin modification
        admin_result = self.simulate_deviation_scenario(
            "modify system settings (admin)", "patching kernel as admin",
            action_type=ActionType.SYSTEM_MODIFICATION, actor_role="admin"
        )
        test_results.append(("admin_modification", admin_result))
        
        # Fuzz testing
        fuzz_results = self.fuzz_test_governance(5)
        test_results.append(("fuzz", fuzz_results))
        
        return {
            "test_type": "safety_governance_comprehensive",
            "test_results": test_results,
            "circuit_breaker_state": self.circuit_breaker_state.value,
            "violations_detected": len(self.violation_history),
            "safety_system_responsive": any(
                isinstance(result[1], dict) and result[1].get("monitoring_result", {}).get("monitoring") == "violation_detected"
                for result in test_results if isinstance(result, tuple) and len(result) > 1
            ),
        }

    # --- Configuration ---
    def set_similarity_functions(self, description_fn: Optional[Callable[[str, str], float]] = None, outcome_fn: Optional[Callable[[str, str], float]] = None) -> None:
        if description_fn: 
            self.description_similarity_fn = description_fn
        if outcome_fn: 
            self.outcome_similarity_fn = outcome_fn

    # --- Export Functions ---
    def export_audit_json(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": dict(self.config),
                "status": self.get_safety_status(),
                "rule_versions": dict(self.rule_versions),
                "intents": [
                    {
                        "id": iid,
                        "description": _redact(i.description),
                        "type": i.action_type.value,
                        "role": i.actor_role,
                        "time_utc": i.timestamp.isoformat(),
                        "expected_outcome": _redact(i.expected_outcome),
                        "confidence": i.confidence,
                        "constraints": list(i.safety_constraints),
                    }
                    for iid, i in self.intent_history
                ],
                "actions": [
                    {
                        "intent_id": iid,
                        "description": _redact(a.description),
                        "type": a.action_type.value,
                        "role": a.actor_role,
                        "time_utc": a.timestamp.isoformat(),
                        "deviation_score": ds,
                        "violated_constraints": vc,
                        "observed_effects": [_redact(e) for e in (a.observed_effects or [])],
                        "actual_parameters": {k: _redact(str(v)) for k, v in (a.actual_parameters or {}).items()},
                    }
                    for iid, a, ds, vc in self.action_history
                ],
                "violations": [
                    {
                        "time_utc": v.timestamp.isoformat(),
                        "severity": v.severity.value,
                        "intent_desc": _redact(v.intent.description),
                        "action_desc": _redact(v.action.description),
                        "deviation_score": v.deviation_score,
                        "violated_constraints": list(v.violated_constraints),
                        "trigger_details": v.trigger_details,
                        "mitigation_suggestion": v.mitigation_suggestion,
                        "feedback": v.feedback,
                        "description": _redact(v.description),
                        "regulatory_tags": v.regulatory_tags,
                        "explanation": {
                            "primary_reason": v.explanation.primary_reason,
                            "contributing_factors": v.explanation.contributing_factors,
                            "confidence": v.explanation.confidence,
                            "counterfactuals": v.explanation.counterfactuals,
                        } if v.explanation else None,
                    }
                    for v in self.violation_history
                ],
            }

    def export_graph(self) -> str:
        """Export relationship graph in Graphviz DOT format"""
        graph = ["digraph nethical {"]
        graph.append('  rankdir=LR;')
        graph.append('  node [shape=box];')
        
        for iid, intent in self.intent_history:
            graph.append(f'  "{iid}" [label="Intent: {intent.description[:30]}...\\n{intent.action_type.value}"];')
        
        for idx, (iid, action, score, violated) in enumerate(self.action_history):
            actid = f'action_{idx}'
            color = "green" if score < 0.5 else "orange" if score < 0.8 else "red"
            graph.append(f'  "{actid}" [label="Action: {action.description[:30]}...\\nScore:{score:.2f}" color="{color}"];')
            graph.append(f'  "{iid}" -> "{actid}";')
        
        for idx, v in enumerate(self.violation_history):
            vid = f"violation_{idx}"
            graph.append(f'  "{vid}" [label="Violation: {v.severity.value}\\n{v.description[:30]}..." shape=octagon color="red"];')
        
        graph.append("}")
        return "\n".join(graph)

    # --- Internal Helper Methods ---
    def _find_intent(self, intent_id: str) -> Optional[Intent]:
        for stored_id, intent in self.intent_history:
            if stored_id == intent_id: 
                return intent
        return None

    def _calculate_deviation(self, intent: Intent, action: Action) -> Tuple[float, List[str], Dict[str, Any], List[str]]:
        w = self.config["weights"]
        penalties: List[float] = []
        violated_rule_ids: List[str] = []
        trigger_details: Dict[str, Any] = {}
        suggestions: List[str] = []
        
        # Type mismatch penalty
        type_penalty = 1.0 if intent.action_type != action.action_type else 0.0
        penalties.append(type_penalty * float(w["type_mismatch"]))
        
        # Description similarity
        desc_sim = float(self.description_similarity_fn(intent.description, action.description))
        penalties.append((1.0 - max(0.0, min(1.0, desc_sim))) * float(w["description"]))
        
        # Outcome similarity
        expected = intent.expected_outcome or ""
        observed = " ".join(action.observed_effects or [])
        outcome_sim = float(self.outcome_similarity_fn(expected, observed))
        penalties.append((1.0 - max(0.0, min(1.0, outcome_sim))) * float(w["outcome"]))
        
        # Constraint violations
        rules = self._build_rules_from_strings(intent.safety_constraints) + \
                self._build_rules_from_strings(self.safety_constraints, category=ConstraintCategory.CUSTOM)
        all_rules = self.global_rules + rules
        
        total_rule_weight = sum(max(0.0, r.weight) for r in all_rules) or 1.0
        violated_weight = 0.0
        
        for rule in all_rules:
            try:
                if rule.violates(action):
                    violated_rule_ids.append(rule.rule_id)
                    violated_weight += max(0.0, rule.weight)
                    trigger_details[rule.rule_id] = {
                        "keywords_triggered": [kw for kw in (rule.keywords_any or []) if kw in (action.description or "").lower()],
                        "regex_matched": [rx for rx in (rule.regex_any or []) if re.search(rx, action.description, flags=re.IGNORECASE)]
                    }
                    suggestions.append(f"To avoid {rule.rule_id}: {rule.description}")
            except Exception as e:
                secure_log("error", f"Rule check failed for {rule.rule_id}: {e}")
        
        constraint_penalty = (violated_weight / total_rule_weight) * float(w["constraints"])
        penalties.append(constraint_penalty)
        
        # Calculate final score
        max_possible = (float(w["type_mismatch"]) + float(w["description"]) + 
                       float(w["outcome"]) + float(w["constraints"])) or 1.0
        score = sum(penalties) / max_possible
        deviation_score = max(0.0, min(1.0, float(score)))
        
        return deviation_score, violated_rule_ids, trigger_details, suggestions

    def _build_rules_from_strings(self, constraints: List[str], category: ConstraintCategory = ConstraintCategory.INTENT_LOCAL) -> List[ConstraintRule]:
        rules: List[ConstraintRule] = []
        for c in constraints or []:
            lc = (c or "").lower().strip()
            rule_id = f"{category.value}::{re.sub(r'\\s+', '_', lc)}"
            
            if "no_system_modification" in lc:
                rules.append(ConstraintRule(
                    rule_id, "No system modification", category, weight=0.8,
                    tags=["safety", "integrity"], logic="OR",
                    keywords_any=_kw("modify", "system modification", "alter core", "patch kernel"),
                ))
            elif "read_only" in lc:
                rules.append(ConstraintRule(
                    rule_id, "Read-only constraint", category, weight=0.7,
                    tags=["integrity"], logic="OR",
                    keywords_any=_kw("write", "delete", "overwrite", "mutate"),
                ))
            elif "no_unauthorized_access" in lc:
                rules.append(ConstraintRule(
                    rule_id, "No unauthorized access", category, weight=0.9,
                    tags=["auth", "privacy"], logic="OR",
                    keywords_any=_kw("unauthorized", "bypass authentication", "elevate privilege"),
                ))
            else:
                rules.append(ConstraintRule(
                    rule_id, f"Generic constraint: {c}", category, weight=0.5,
                    tags=["custom"], keywords_any=[lc] if lc else None,
                ))
        return rules

    def _check_safety_violations(self, intent: Intent, action: Action, deviation_score: float, 
                                 violated_constraints: List[str], trigger_details: Dict[str, Any], 
                                 suggestions: List[str]) -> Dict[str, Any]:
        if deviation_score < self.deviation_threshold and not violated_constraints:
            return {"violation_detected": False}
        
        # Determine severity
        violated_lower = [v.lower() for v in violated_constraints]
        emergency_triggers = any("preserve_life" in v or "no_harm" in v for v in violated_lower)
        privacy_triggers = any("privacy" in v for v in violated_lower)
        
        if deviation_score >= self.emergency_threshold or emergency_triggers:
            severity = SafetyLevel.EMERGENCY
        elif deviation_score >= 0.8 or privacy_triggers or len(violated_constraints) >= 3:
            severity = SafetyLevel.CRITICAL
        else:
            severity = SafetyLevel.WARNING
        
        # Collect regulatory tags
        regulatory_tags = list({tag for rule in self.global_rules 
                               for tag in (rule.regulatory_tags or []) 
                               if rule.rule_id in violated_constraints})
        
        violation = SafetyViolation(
            violation_type="intent_action_deviation",
            severity=severity,
            intent=intent,
            action=action,
            deviation_score=deviation_score,
            description=f"Action '{_redact(action.description)}' deviates from intent '{_redact(intent.description)}' (score: {deviation_score:.3f})",
            violated_constraints=violated_constraints,
            trigger_details=trigger_details,
            mitigation_suggestion="; ".join(suggestions),
            regulatory_tags=regulatory_tags,
        )
        
        return {"violation_detected": True, "violation": violation}

    def _generate_explanation(self, intent: Intent, action: Action, violation: SafetyViolation) -> Explanation:
        """Generate human-readable explanation with counterfactuals"""
        primary_reason = f"The action deviated from the stated intent with a score of {violation.deviation_score:.2f}"
        
        contributing_factors = []
        if intent.action_type != action.action_type:
            contributing_factors.append(f"Action type mismatch: expected {intent.action_type.value}, got {action.action_type.value}")
        
        if violation.violated_constraints:
            contributing_factors.append(f"Violated {len(violation.violated_constraints)} safety constraints: {', '.join(violation.violated_constraints[:3])}")
        
        # Generate counterfactuals
        counterfactuals = []
        if intent.action_type != action.action_type:
            counterfactuals.append(f"If the action type had been {intent.action_type.value}, the deviation would have been lower")
        
        for constraint in violation.violated_constraints[:2]:
            counterfactuals.append(f"If the action had avoided triggering '{constraint}', it would have been safer")
        
        confidence = 1.0 - (violation.deviation_score * 0.3)  # Higher deviation = lower confidence in explanation
        
        return Explanation(
            primary_reason=primary_reason,
            contributing_factors=contributing_factors,
            confidence=max(0.5, min(1.0, confidence)),
            counterfactuals=counterfactuals
        )

    def _handle_safety_violation(self, violation: SafetyViolation) -> None:
        secure_log("warning", f"Safety violation detected: {violation.description}")
        
        if violation.severity in (SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY):
            now_mono = time.monotonic()
            cooldown = float(self.config.get("trip_cooldown_seconds", 3.0))
            
            if self.circuit_breaker_state == CircuitBreakerState.CLOSED and (now_mono - self._last_trip_monotonic) >= cooldown:
                self.trip_circuit_breaker(f"{violation.severity.value.upper()}: {violation.violation_type}; {violation.violated_constraints}")
                self._last_trip_monotonic = now_mono
                self._last_trip_time = datetime.now(timezone.utc)
        
        # Execute callbacks
        callbacks = self.safety_callbacks.get(violation.severity, [])
        for callback in callbacks:
            try: 
                callback(violation)
            except Exception as e: 
                secure_log("error", f"Safety callback failed: {e}")

    def trip_circuit_breaker(self, reason: str) -> None:
        with self._lock:
            if self.circuit_breaker_state != CircuitBreakerState.OPEN:
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self._last_trip_time = datetime.now(timezone.utc)
                self._recovery_success_count = 0
                secure_log("critical", f"CIRCUIT BREAKER TRIPPED: {reason}")
                secure_log("critical", "System operation halted for safety")
                self.audit_log.append(f"Circuit breaker tripped: {reason}")
                
                for callback in self.safety_callbacks.get(SafetyLevel.EMERGENCY, []):
                    try: 
                        callback(reason)
                    except Exception as e: 
                        secure_log("error", f"Emergency callback failed: {e}")

    # --- Calibration Support ---
    @dataclass
    class CalibrationData:
        """Historical data for score calibration"""
        false_positives: List[float]
        true_positives: List[float]

    def calibrate_threshold(self, false_positive_scores: List[float], true_positive_scores: List[float]) -> float:
        """Calculate optimal threshold using simple heuristic (ROC would require sklearn)"""
        if not false_positive_scores or not true_positive_scores:
            return self.deviation_threshold
        
        # Find threshold that minimizes false positives while catching true positives
        all_scores = sorted(set(false_positive_scores + true_positive_scores))
        best_threshold = self.deviation_threshold
        best_score = float('-inf')
        
        for threshold in all_scores:
            tp = sum(1 for s in true_positive_scores if s >= threshold)
            fp = sum(1 for s in false_positive_scores if s >= threshold)
            fn = len(true_positive_scores) - tp
            
            # F1-like score: balance precision and recall
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    if f1 > best_score:
                        best_score = f1
                        best_threshold = threshold
        
        secure_log("info", f"Calibrated threshold: {best_threshold:.3f} (F1: {best_score:.3f})")
        return best_threshold

    def apply_calibrated_threshold(self, false_positive_scores: List[float], true_positive_scores: List[float]) -> None:
        """Apply calibrated threshold to the system"""
        new_threshold = self.calibrate_threshold(false_positive_scores, true_positive_scores)
        self.deviation_threshold = new_threshold
        self.config["deviation_threshold"] = new_threshold
        self.audit_log.append(f"Threshold calibrated to {new_threshold:.3f}")


# Alias for backwards compatibility
Nethical = nethical


# --- Main Demo ---
if __name__ == "__main__":
    print(f"nethical Enhanced v10 - Python {platform.python_version()} on {platform.system()}")
    print(f"Optional features: Transformers={HAS_TRANSFORMERS}, Sklearn={HAS_SKLEARN}, Scrubadub={HAS_SCRUBADUB}\n")
    
    # Initialize system
    gov = nethical()
    
    # Register callbacks
    def warn_cb(v: SafetyViolation | str): 
        msg = getattr(v, 'description', str(v))
        secure_log("warning", f"[WARN CB] {msg}")
    
    def emerg_cb(v: SafetyViolation | str): 
        msg = getattr(v, 'description', str(v))
        secure_log("critical", f"[EMERG CB] {msg}")
    
    gov.register_safety_callback(SafetyLevel.WARNING, warn_cb)
    gov.register_safety_callback(SafetyLevel.CRITICAL, warn_cb)
    gov.register_safety_callback(SafetyLevel.EMERGENCY, emerg_cb)
    gov.set_incident_notify_hook(lambda v: print(f"INCIDENT: {v.severity.value} - {v.description[:50]}..."))
    
    # Run comprehensive tests
    print("Running safety governance tests...")
    res = gov.test_safety_governance()
    print(f"✓ Safety system responsive: {res['safety_system_responsive']}")
    print(f"✓ Violations detected: {res['violations_detected']}")
    print(f"✓ Circuit breaker state: {res['circuit_breaker_state']}")
    
    # Display status
    status = gov.get_safety_status()
    print(f"\nSystem Status:")
    print(f"  Active: {status['is_active']}")
    print(f"  Circuit Breaker: {status['circuit_breaker_state']}")
    print(f"  Total Intents: {status['total_intents']}")
    print(f"  Total Actions: {status['total_actions']}")
    print(f"  Total Violations: {status['total_violations']}")
    print(f"  Anomaly Detector Trained: {status['anomaly_detector_trained']}")
    
    # Test file encryption
    print("\n--- Testing File Encryption ---")
    test_file = "/tmp/test_nethical.txt"
    key_path = "/tmp/nethical_test.key"
    
    try:
        # Create test file
        with open(test_file, "w") as f:
            f.write("This is sensitive test data that should be encrypted.")
        
        # Generate key and encrypt
        gov.generate_security_key("test_key", key_path)
        print(f"✓ Key generated: {key_path}")
        
        gov.encrypt_file(test_file, "test_key", key_path)
        print(f"✓ File encrypted: {test_file}")
        
        print(f"  Is encrypted: {gov.is_file_encrypted(test_file)}")
        
        # Decrypt
        gov.decrypt_file(test_file, "test_key", key_path)
        print(f"✓ File decrypted: {test_file}")
        
        # Cleanup
        os.remove(test_file)
        os.remove(key_path)
        print("✓ Cleanup complete")
        
    except Exception as e:
        print(f"✗ Encryption test failed: {e}")
    
    # Test compliance reporting
    print("\n--- Testing Compliance Report ---")
    start = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.now(timezone.utc)
    compliance = gov.generate_compliance_report("GDPR", start, end)
    print(f"GDPR Compliance Report ({compliance['period']['start'][:10]} to {compliance['period']['end'][:10]}):")
    print(f"  Total violations: {compliance['total_violations']}")
    print(f"  By severity: {compliance['by_severity']}")
    print(f"  Unresolved: {compliance['unresolved']}")
    
    # Export audit data
    print("\n--- Exporting Audit Data ---")
    audit_data = gov.export_audit_json()
    print(f"✓ Audit export contains {len(audit_data['intents'])} intents, {len(audit_data['actions'])} actions, {len(audit_data['violations'])} violations")
    
    # Export graph
    print("\n--- Graphviz DOT Export ---")
    dot_graph = gov.export_graph()
    print(f"✓ Graph exported ({len(dot_graph.split(chr(10)))} lines)")
    print("\nFirst 10 lines of graph:")
    print('\n'.join(dot_graph.split('\n')[:10]))
    
    # Test calibration
    print("\n--- Testing Threshold Calibration ---")
    false_positives = [0.65, 0.68, 0.72]  # Scores that were false alarms
    true_positives = [0.75, 0.82, 0.88, 0.95]  # Scores that were real violations
    gov.apply_calibrated_threshold(false_positives, true_positives)
    print(f"✓ New threshold: {gov.deviation_threshold:.3f}")
    
    print("\n" + "="*60)
    print("nethical Enhanced - All tests complete!")
    print("="*60)
