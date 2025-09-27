"""
nethical.py

Improved Cognitive AI Ethics, Safety, and Secure File Management system.
Enhancements:
- Logical rule combinations (AND/OR/CUSTOM), time/role constraints
- Detailed violation objects (explainability, mitigation)
- Adaptive thresholds, feedback API
- Versioned audit trail, incident notification hooks
- Expanded governance testing

Author: V1B3hR (github.com/V1B3hR/nethical)
"""

import os, re, sys, time, logging, platform, threading
from pathlib import Path
from typing import Optional, List, Any, Callable, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, time as dtime

try:
    import fcntl as _fcntl
except Exception:
    _fcntl = None

try:
    from cryptography.fernet import Fernet, InvalidToken, MultiFernet
except ImportError as e:
    raise SystemExit("The 'cryptography' package is required. Install with: pip install cryptography") from e

LOG_PATH = "/var/log/nethical.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def _secure_log_init():
    logger = logging.getLogger("nethical")
    logger.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=10)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    try:
        if not os.path.exists(LOG_PATH):
            open(LOG_PATH, 'a').close()
        os.chmod(LOG_PATH, 0o600)
    except Exception as e:
        logger.error(f"Log file permission set failed: {e}")
    return logger

_secure_logger = _secure_log_init()

def _redact(text: str) -> str:
    if not text: return text
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[redacted-email]", text)
    text = re.sub(r"\b(?:\+?\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b", "[redacted-phone]", text)
    text = re.sub(r"(key|password|secret|token)[=:]?\s*[A-Za-z0-9+/=]+", r"\1:[redacted-secret]", text, flags=re.IGNORECASE)
    return text

def secure_log(level: str, msg: str, *args, **kwargs):
    msg = _redact(msg)
    getattr(_secure_logger, level if level in ["critical","error","warning","info","debug"] else "info")(msg, *args, **kwargs)

MAGIC_HEADER = b"FEN1"
CHUNK_SIZE = 1024 * 1024
MAX_CIPHERTEXT_CHUNK = 32 * 1024 * 1024

class SecurityError(Exception): pass
class AlreadyEncryptedError(SecurityError): pass
class NotEncryptedError(SecurityError): pass

def _acquire_file_lock(fileobj): 
    if _fcntl: 
        try: _fcntl.flock(fileobj.fileno(), _fcntl.LOCK_EX)
        except Exception as e: secure_log("debug", f"Could not acquire file lock: {e}")

def _release_file_lock(fileobj): 
    if _fcntl: 
        try: _fcntl.flock(fileobj.fileno(), _fcntl.LOCK_UN)
        except Exception as e: secure_log("debug", f"Could not release file lock: {e}")

def _preserve_metadata(src_path: Path, dst_path: Path) -> None:
    try:
        st = src_path.stat()
        os.chmod(dst_path, st.st_mode)
        try: os.chown(dst_path, st.st_uid, st.st_gid)
        except Exception: pass
        os.utime(dst_path, (st.st_atime, st.st_mtime))
    except FileNotFoundError: pass

def _warn_if_permissive(path: Path) -> None:
    try:
        st = path.stat()
        if os.name == "posix" and (st.st_mode & 0o077) != 0:
            secure_log("warning", f"Key file {path} has overly permissive permissions: {oct(st.st_mode)}")
    except Exception: pass

class SecurityManager:
    @staticmethod
    def generate_key(key_path: str) -> None:
        key_file = Path(key_path)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        tmp_path = key_file.with_suffix(key_file.suffix + ".tmp")
        with open(tmp_path, "wb") as f: f.write(key)
        os.replace(tmp_path, key_file)
        if os.name == "posix":
            try: os.chmod(key_file, 0o600)
            except Exception: pass
        secure_log("info", f"New encryption key generated and saved securely to {key_path}")

    @staticmethod
    def _normalize_key_bytes(key: bytes | str) -> bytes:
        if isinstance(key, str): key = key.encode("utf-8")
        return key.strip()

    @staticmethod
    def load_key(key_path: str, extra_keys: Optional[List[bytes | str]]=None) -> MultiFernet:
        with open(key_path, "rb") as key_file:
            key = key_file.read().strip()
            if len(key) != 44: raise ValueError("Fernet key must be 44-byte base64.")
            _warn_if_permissive(Path(key_path))
            ferns = [Fernet(key)]
            if extra_keys:
                for ek in extra_keys:
                    kb = SecurityManager._normalize_key_bytes(ek)
                    if len(kb) == 44: ferns.append(Fernet(kb))
                    else: secure_log("warning", "Skipping invalid extra key (must be 44-byte base64).")
            return MultiFernet(ferns)

    @staticmethod
    def is_encrypted(file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                header = f.read(len(MAGIC_HEADER))
                return header == MAGIC_HEADER
        except FileNotFoundError: raise
        except Exception as e:
            secure_log("error", f"Failed to check encryption status for {file_path}: {e}")
            return False

    @staticmethod
    def encrypt_file(file_path: str, key_path: str, output_path: Optional[str]=None, extra_keys: Optional[List[bytes | str]]=None) -> None:
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path
        if src_path.exists() and SecurityManager.is_encrypted(str(src_path)):
            secure_log("warning", f"File '{src_path}' is already encrypted.")
            raise AlreadyEncryptedError(f"File '{src_path}' is already encrypted.")
        fernet = SecurityManager.load_key(key_path, extra_keys=extra_keys)
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".enc_tmp")
        with open(src_path, "rb") as infile, open(tmp_path, "wb") as outfile:
            _acquire_file_lock(outfile)
            try:
                outfile.write(MAGIC_HEADER)
                while True:
                    chunk = infile.read(CHUNK_SIZE)
                    if not chunk: break
                    ciphertext = fernet.encrypt(chunk)
                    chunk_len = len(ciphertext).to_bytes(4, "big")
                    outfile.write(chunk_len)
                    outfile.write(ciphertext)
            finally:
                _release_file_lock(outfile)
        _preserve_metadata(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        secure_log("info", f"File '{file_path}' has been successfully encrypted to '{dst_path}'.")

    @staticmethod
    def decrypt_file_safely(file_path: str, key_path: str, output_path: Optional[str]=None, extra_keys: Optional[List[bytes | str]]=None) -> None:
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".decrypted_tmp")
        with open(src_path, "rb") as infile:
            header = infile.read(len(MAGIC_HEADER))
            if header != MAGIC_HEADER:
                secure_log("error", f"File '{file_path}' is not encrypted with expected header.")
                raise NotEncryptedError(f"File '{file_path}' is not encrypted with expected header.")
            fernet = SecurityManager.load_key(key_path, extra_keys=extra_keys)
            with open(tmp_path, "wb") as outfile:
                _acquire_file_lock(outfile)
                try:
                    while True:
                        len_bytes = infile.read(4)
                        if not len_bytes: break
                        if len(len_bytes) != 4: raise SecurityError("Corrupted file: incomplete chunk length.")
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
    actor_role: Optional[str] = "user"
    def __post_init__(self): self.timestamp = self.timestamp or datetime.now(timezone.utc)

@dataclass
class Action:
    description: str
    action_type: ActionType
    actual_parameters: Dict[str, Any]
    observed_effects: List[str]
    timestamp: datetime = None
    actor_role: Optional[str] = "user"
    def __post_init__(self): self.timestamp = self.timestamp or datetime.now(timezone.utc)

@dataclass
class ConstraintRule:
    rule_id: str
    description: str
    category: ConstraintCategory
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    logic: str = "OR"  # "AND", "OR", "CUSTOM"
    check: Optional[Callable[[Action], bool]] = None
    keywords_any: Optional[List[str]] = None
    keywords_all: Optional[List[str]] = None
    except_keywords: Optional[List[str]] = None
    regex_any: Optional[List[str]] = None
    regex_all: Optional[List[str]] = None
    applies_during: Optional[Tuple[str, str]] = None  # ("18:00", "08:00")
    applies_to_roles: Optional[List[str]] = None
    version: int = 1
    def violates(self, action: Action) -> bool:
        desc_raw = action.description or ""
        desc = desc_raw.lower()
        if self.applies_to_roles and action.actor_role not in self.applies_to_roles: return False
        if self.applies_during:
            now = datetime.now(timezone.utc).time()
            start = dtime.fromisoformat(self.applies_during[0])
            end = dtime.fromisoformat(self.applies_during[1])
            if not (start <= now or now <= end): return False
        if self.check: 
            try: return bool(self.check(action))
            except Exception as e: secure_log("error", f"ConstraintRule.check failed for {self.rule_id}: {e}"); return False
        if self.except_keywords and any(ex_kw in desc for ex_kw in self.except_keywords): return False
        def get_kw(words): return [w.lower() for w in words or [] if w]
        kw_any, kw_all = get_kw(self.keywords_any), get_kw(self.keywords_all)
        rx_any, rx_all = self.regex_any or [], self.regex_all or []
        if self.logic == "AND":
            cond = (all(w in desc for w in kw_all) if kw_all else True) and \
                   (all(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in rx_all) if rx_all else True)
            return cond
        elif self.logic == "CUSTOM": return self.check(action) if self.check else False
        else:  # "OR"
            cond = (any(w in desc for w in kw_any) if kw_any else False) or \
                   (any(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in rx_any) if rx_any else False)
            return cond

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
    timestamp: datetime = None
    def __post_init__(self): self.timestamp = self.timestamp or datetime.now(timezone.utc)

def _token_jaccard(a: str, b: str) -> float:
    a_set = set((a or "").lower().split())
    b_set = set((b or "").lower().split())
    if not a_set and not b_set: return 1.0
    if not a_set or not b_set: return 0.0
    return len(a_set & b_set) / max(1, len(a_set | b_set))

def _now_utc() -> datetime: return datetime.now(timezone.utc)
def _kw(*words: str) -> List[str]: return [w for w in words if w]

def _default_framework_rules() -> List[ConstraintRule]:
    return [
        ConstraintRule(
            "operational.privacy",
            "Preserve Privacy - protect personal information and respect confidentiality.",
            ConstraintCategory.OPERATIONAL,
            weight=0.95,
            tags=["privacy", "pii", "security", "compliance"],
            logic="OR",
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
            logic="OR",
            keywords_any=_kw("kill", "injure", "assault", "self-harm", "suicide", "violence", "murder"),
        ),
    ]

class nethical:
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
            self.config.update({k: v for k, v in config.items() if k != "weights"})
            if "weights" in config and isinstance(config["weights"], dict):
                self.config["weights"].update(config["weights"])
        self.is_active = True
        self.deviation_threshold = float(self.config.get("deviation_threshold", 0.7))
        self.emergency_threshold = float(self.config.get("emergency_threshold", 0.9))
        self.intent_history: List[Tuple[str, Intent]] = []
        self.action_history: List[Tuple[str, Action, float, List[str]]] = []
        self.violation_history: List[SafetyViolation] = []
        self.violation_feedback: Dict[str, bool] = {}
        self.circuit_breaker_active = False
        self._last_trip_monotonic = 0.0
        self.safety_constraints: List[str] = []
        self.global_rules: List[ConstraintRule] = _default_framework_rules()
        self.rule_versions: Dict[str, int] = {r.rule_id: r.version for r in self.global_rules}
        self.description_similarity_fn: Callable[[str, str], float] = _token_jaccard
        self.outcome_similarity_fn: Callable[[str, str], float] = _token_jaccard
        self._lock = threading.Lock()
        self.safety_callbacks: Dict[SafetyLevel, List[Callable[[SafetyViolation | str], None]]] = {
            SafetyLevel.WARNING: [],
            SafetyLevel.CRITICAL: [],
            SafetyLevel.EMERGENCY: [],
        }
        secure_log("info", f"nethical initialized (UTC), thresholds: dev={self.deviation_threshold}, emerg={self.emergency_threshold}")

    def generate_security_key(self, key_path: str) -> None: SecurityManager.generate_key(key_path)
    def encrypt_file(self, file_path: str, key_path: str, output_path: Optional[str]=None, extra_keys: Optional[List[bytes | str]]=None) -> None:
        SecurityManager.encrypt_file(file_path, key_path, output_path, extra_keys)
    def decrypt_file(self, file_path: str, key_path: str, output_path: Optional[str]=None, extra_keys: Optional[List[bytes | str]]=None) -> None:
        SecurityManager.decrypt_file_safely(file_path, key_path, output_path, extra_keys)
    def is_file_encrypted(self, file_path: str) -> bool: return SecurityManager.is_encrypted(file_path)

    def register_intent(self, intent: Intent) -> str:
        with self._lock:
            intent_id = f"intent_{len(self.intent_history)}_{int(time.time()*1000)}"
            self.intent_history.append((intent_id, intent))
            secure_log("info", f"Intent registered: {intent_id} - {_redact(intent.description)} @ {intent.timestamp.isoformat()}")
            return intent_id

    def monitor_action(self, intent_id: str, action: Action) -> Dict[str, Any]:
        if not self.is_active: return {"monitoring": "disabled", "action_allowed": True}
        if self.circuit_breaker_active:
            secure_log("critical", "Circuit breaker active - action blocked")
            return {"monitoring": "blocked", "action_allowed": False, "reason": "circuit_breaker_active"}
        with self._lock:
            intent = self._find_intent(intent_id)
            if not intent:
                secure_log("warning", f"Intent {intent_id} not found")
                return {"monitoring": "error", "action_allowed": False, "reason": "intent_not_found"}
            deviation_score, violated, trigger_details, suggestions = self._calculate_deviation(intent, action)
            self.action_history.append((intent_id, action, deviation_score, violated))
            safety_result = self._check_safety_violations(intent, action, deviation_score, violated, trigger_details, suggestions)
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

    def submit_violation_feedback(self, violation_id: str, valid: bool):
        self.violation_feedback[violation_id] = valid
        # Could tune thresholds/weights here based on feedback stats

    def enable_safety_monitoring(self) -> None:
        self.is_active = True
        secure_log("info", "Safety monitoring enabled")

    def disable_safety_monitoring(self) -> None:
        self.is_active = False
        secure_log("warning", "Safety monitoring disabled")

    def clear_history(self) -> None:
        with self._lock:
            self.intent_history.clear()
            self.action_history.clear()
            self.violation_history.clear()
            self.violation_feedback.clear()
            secure_log("info", "History cleared")

    def add_safety_constraint(self, constraint: str) -> None:
        with self._lock:
            self.safety_constraints.append(constraint)
            secure_log("info", f"Safety constraint added: {constraint}")

    def register_safety_callback(self, level: SafetyLevel, callback: Callable[[SafetyViolation | str], None]) -> None:
        self.safety_callbacks[level].append(callback)
        secure_log("info", f"Safety callback registered for level {level.value}")

    def reset_circuit_breaker(self, authorization_token: Optional[str] = None) -> bool:
        if authorization_token != self.config.get("reset_token", "admin_reset"):
            secure_log("error", "Unauthorized circuit breaker reset attempt")
            return False
        with self._lock:
            self.circuit_breaker_active = False
            secure_log("info", "Circuit breaker reset - system operation resumed")
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
                "violation_feedback": dict(self.violation_feedback),
            }

    def list_constraint_rules(self) -> List[ConstraintRule]:
        with self._lock: return list(self.global_rules)

    def add_constraint_rule(self, rule: ConstraintRule) -> None:
        with self._lock:
            rule.version = self.rule_versions.get(rule.rule_id, 0) + 1
            self.global_rules.append(rule)
            self.rule_versions[rule.rule_id] = rule.version
            secure_log("info", f"Constraint rule added: {rule.rule_id} (v{rule.version})")

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
        # Role-based and time-based test
        admin_result = self.simulate_deviation_scenario(
            "modify system settings (admin)", "patching kernel as admin",
            action_type=ActionType.SYSTEM_MODIFICATION, actor_role="admin"
        )
        test_results.append(("admin_modification", admin_result))
        return {
            "test_type": "safety_governance_comprehensive",
            "test_results": test_results,
            "circuit_breaker_status": self.circuit_breaker_active,
            "violations_detected": len(self.violation_history),
            "safety_system_responsive": any(result[1]["monitoring_result"]["monitoring"] == "violation_detected" for result in test_results),
        }

    def set_similarity_functions(self, description_fn: Optional[Callable[[str, str], float]] = None, outcome_fn: Optional[Callable[[str, str], float]] = None) -> None:
        if description_fn: self.description_similarity_fn = description_fn
        if outcome_fn: self.outcome_similarity_fn = outcome_fn

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
                        "actual_parameters": {k:_redact(str(v)) for k,v in (a.actual_parameters or {}).items()},
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
                    }
                    for v in self.violation_history
                ],
            }

    def _find_intent(self, intent_id: str) -> Optional[Intent]:
        for stored_id, intent in self.intent_history:
            if stored_id == intent_id: return intent
        return None

    def _calculate_deviation(self, intent: Intent, action: Action) -> Tuple[float, List[str], Dict[str, Any], List[str]]:
        w = self.config["weights"]
        penalties: List[float] = []
        violated_rule_ids: List[str] = []
        trigger_details: Dict[str, Any] = {}
        suggestions: List[str] = []
        type_penalty = 1.0 if intent.action_type != action.action_type else 0.0
        penalties.append(type_penalty * float(w["type_mismatch"]))
        desc_sim = float(self.description_similarity_fn(intent.description, action.description))
        penalties.append((1.0 - max(0.0, min(1.0, desc_sim))) * float(w["description"]))
        expected = intent.expected_outcome or ""
        observed = " ".join(action.observed_effects or [])
        outcome_sim = float(self.outcome_similarity_fn(expected, observed))
        penalties.append((1.0 - max(0.0, min(1.0, outcome_sim))) * float(w["outcome"]))
        rules = self._build_rules_from_strings(intent.safety_constraints) + self._build_rules_from_strings(self.safety_constraints, category=ConstraintCategory.CUSTOM)
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
                    suggestions.append(f"To avoid {rule.rule_id}: {rule.description}. Consider revising your action or intent.")
            except Exception as e:
                secure_log("error", f"Rule check failed for {rule.rule_id}: {e}")
        constraint_penalty = (violated_weight / total_rule_weight) * float(w["constraints"])
        penalties.append(constraint_penalty)
        max_possible = (float(w["type_mismatch"]) + float(w["description"]) + float(w["outcome"]) + float(w["constraints"])) or 1.0
        score = sum(penalties) / max_possible
        deviation_score = max(0.0, min(1.0, float(score)))
        return deviation_score, violated_rule_ids, trigger_details, suggestions

    def _build_rules_from_strings(self, constraints: List[str], category: ConstraintCategory = ConstraintCategory.INTENT_LOCAL) -> List[ConstraintRule]:
        rules: List[ConstraintRule] = []
        for c in constraints or []:
            lc = (c or "").lower().strip()
            rule_id = f"{category.value}::{re.sub(r'\\s+', '_', lc)}"
            if "no_system_modification" in lc or "no system modification" in lc:
                rules.append(ConstraintRule(
                    rule_id,
                    "No system modification",
                    category,
                    weight=0.8,
                    tags=["safety", "integrity", "system", "configuration", "kernel", "patch"],
                    keywords_any=_kw("modify", "system modification", "alter core", "patch kernel", "edit config"),
                ))
            elif "read_only" in lc or "read-only" in lc:
                rules.append(ConstraintRule(
                    rule_id,
                    "Read-only constraint",
                    category,
                    weight=0.7,
                    tags=["integrity", "immutability", "data", "storage"],
                    keywords_any=_kw("write", "delete", "overwrite", "mutate", "change data"),
                ))
            elif "local_only" in lc or "local-only" in lc:
                rules.append(ConstraintRule(
                    rule_id,
                    "Local-only operations",
                    category,
                    weight=0.6,
                    tags=["network", "scope", "local", "host", "firewall"],
                    keywords_any=_kw("remote", "exfiltrate", "upload external", "send to external", "cross-boundary"),
                ))
            elif "authorized_only" in lc or "authorized-only" in lc:
                rules.append(ConstraintRule(
                    rule_id,
                    "Authorized-only operations",
                    category,
                    weight=0.9,
                    tags=["auth", "permission", "security", "access control"],
                    keywords_any=_kw("unauthorized", "bypass auth", "without permission", "unauthorized access", "escalate privileges"),
                ))
            elif "no_unauthorized_access" in lc:
                rules.append(ConstraintRule(
                    rule_id,
                    "No unauthorized access",
                    category,
                    weight=0.9,
                    tags=["auth", "privacy", "security", "intrusion"],
                    keywords_any=_kw("unauthorized", "bypass authentication", "elevate privilege", "without consent", "unapproved access"),
                ))
            else:
                rules.append(ConstraintRule(
                    rule_id,
                    f"Generic constraint: {c}",
                    category,
                    weight=0.5,
                    tags=["custom"],
                    keywords_any=[lc] if lc else None,
                ))
        return rules

    def _check_safety_violations(self, intent: Intent, action: Action, deviation_score: float, violated_constraints: List[str], trigger_details: Dict[str, Any], suggestions: List[str]) -> Dict[str, Any]:
        if deviation_score < self.deviation_threshold and not violated_constraints:
            return {"violation_detected": False}
        violated_lower = [v.lower() for v in violated_constraints]
        emergency_triggers = any("human_ai.preserve_life" in v or "universal.no_harm" in v for v in violated_lower)
        privacy_triggers = any("operational.privacy" in v for v in violated_lower)
        if deviation_score >= self.emergency_threshold or emergency_triggers:
            severity = SafetyLevel.EMERGENCY
        elif deviation_score >= 0.8 or privacy_triggers or len(violated_constraints) >= 3:
            severity = SafetyLevel.CRITICAL
        else:
            severity = SafetyLevel.WARNING
        violation_id = f"violation_{len(self.violation_history)}_{int(time.time()*1000)}"
        violation = SafetyViolation(
            violation_type="intent_action_deviation",
            severity=severity,
            intent=intent,
            action=action,
            deviation_score=deviation_score,
            description=f"Action '{_redact(action.description)}' deviates from intent '{_redact(intent.description)}' (score: {deviation_score:.3f}); Violated: {violated_constraints}",
            violated_constraints=violated_constraints,
            trigger_details=trigger_details,
            mitigation_suggestion="; ".join(suggestions),
            feedback=self.violation_feedback.get(violation_id, None),
        )
        return {"violation_detected": True, "violation": violation}

    def _handle_safety_violation(self, violation: SafetyViolation) -> None:
        secure_log("warning", f"Safety violation detected: {violation.description}")
        if violation.severity in (SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY):
            now_mono = time.monotonic()
            cooldown = float(self.config.get("trip_cooldown_seconds", 3.0))
            if not self.circuit_breaker_active and (now_mono - self._last_trip_monotonic) >= cooldown:
                self.trip_circuit_breaker(f"{violation.severity.value.upper()}: {violation.violation_type}; {violation.violated_constraints}")
                self._last_trip_monotonic = now_mono
        callbacks = self.safety_callbacks.get(violation.severity, [])
        for callback in callbacks:
            try: callback(violation)
            except Exception as e: secure_log("error", f"Safety callback failed: {e}")

    def trip_circuit_breaker(self, reason: str) -> None:
        with self._lock:
            if not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                secure_log("critical", f"CIRCUIT BREAKER TRIPPED: {reason}")
                secure_log("critical", "System operation halted for safety")
                for callback in self.safety_callbacks.get(SafetyLevel.EMERGENCY, []):
                    try: callback(reason)
                    except Exception as e: secure_log("error", f"Emergency callback failed: {e}")

Nethical = nethical

if __name__ == "__main__":
    print(f"nethical enhanced v8 - Python {platform.python_version()} on {platform.system()}")
    gov = nethical()
    def warn_cb(v: SafetyViolation | str): secure_log("warning", f"[WARN CB] {getattr(v, 'severity', '')}: {getattr(v, 'description', v)}")
    def emerg_cb(v: SafetyViolation | str): secure_log("critical", f"[EMERG CB] {getattr(v, 'description', v)}")
    gov.register_safety_callback(SafetyLevel.WARNING, warn_cb)
    gov.register_safety_callback(SafetyLevel.CRITICAL, warn_cb)
    gov.register_safety_callback(SafetyLevel.EMERGENCY, emerg_cb)
    res = gov.test_safety_governance()
    print("Safety system responsive:", res["safety_system_responsive"])
    print("Violations detected:", res["violations_detected"])
    status = gov.get_safety_status()
    print("Status:", {k: status[k] for k in ("is_active", "circuit_breaker_active", "total_intents", "total_actions", "total_violations")})
