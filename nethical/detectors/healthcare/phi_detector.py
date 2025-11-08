from __future__ import annotations
import re
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nethical.core.plugin_interface import DetectorPlugin, PluginMetadata
from nethical.detectors.base_detector import SafetyViolation

# Expanded PHI patterns (illustrative; tune for precision and performance)
PHI_PATTERNS: Dict[str, re.Pattern] = {
    "ssn_us": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn_generic": re.compile(
        r"\b(?:MRN|Med(?:ical)?\s*Record)\s*[:#]?\s*[A-Za-z0-9\-]{5,}\b", re.IGNORECASE
    ),
    "phone": re.compile(
        r"\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}\b"
    ),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "address_hint": re.compile(
        r"\b(\d{1,6}\s+[A-Za-z0-9\.\-]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr))\b",
        re.IGNORECASE,
    ),
    "dob": re.compile(
        r"\b(?:DOB[:\s]*)?(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b", re.IGNORECASE
    ),
    "license_number": re.compile(
        r"\b(?:DL|Driver(?:'s)? License)[:\s#-]*[A-Za-z0-9\-]{5,}\b", re.IGNORECASE
    ),
    "health_plan_beneficiary": re.compile(
        r"\b(?:HPN|HICN|Medicare|Medicaid)[\s#:]*[A-Za-z0-9\-]{5,}\b", re.IGNORECASE
    ),
    "device_id": re.compile(r"\b(?:UDI|Device\s*ID)[:\s#-]*[A-Za-z0-9\-]{6,}\b", re.IGNORECASE),
    "url": re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

DEFAULT_REDACTION_TOKEN = "[REDACTED:{type}]"

SEVERITY_MAP = {
    "ssn_us": "high",
    "mrn_generic": "high",
    "health_plan_beneficiary": "high",
    "license_number": "high",
    "device_id": "medium",
    "phone": "medium",
    "email": "medium",
    "address_hint": "medium",
    "dob": "medium",
    "url": "low",
    "ip": "low",
}


class HealthcarePHIDetector(DetectorPlugin):
    def __init__(
        self,
        name: str = "HealthcarePHIDetector",
        version: str = "1.0.0",
        max_matches_per_type: int = 50,
        enabled_patterns: Optional[List[str]] = None,
        allowlist_emails: Optional[List[str]] = None,
        redaction_token: str = DEFAULT_REDACTION_TOKEN,
        pseudonymize: bool = False,
        hash_salt: Optional[str] = None,
        preserve_format: bool = False,
    ):
        super().__init__(name=name, version=version)
        self.max_matches_per_type = max_matches_per_type
        self.enabled_patterns = (
            set(enabled_patterns) if enabled_patterns else set(PHI_PATTERNS.keys())
        )
        self.allowlist_emails = set(allowlist_emails or [])
        self.redaction_token = redaction_token
        self.pseudonymize = pseudonymize
        self.hash_salt = hash_salt or ""
        self.preserve_format = preserve_format

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description="Detects and redacts HIPAA PHI indicators (regex-based, configurable).",
            tags=["privacy", "healthcare", "hipaa", "phi"],
            author="Nethical",
            homepage="https://github.com/V1B3hR/nethical",
        )

    async def detect_violations(self, action: Any) -> Sequence[SafetyViolation] | None:
        text = self._extract_text(action)
        if not text:
            return []

        matches = self._find_matches(text)
        violations: List[SafetyViolation] = []

        for match_type, spans in matches.items():
            if not spans:
                continue

            severity = SEVERITY_MAP.get(match_type, "medium")
            snippet = self._safe_snippet(text, spans[0])
            violations.append(
                SafetyViolation(
                    detector=self.name,
                    severity=severity,
                    description=f"Detected PHI: {match_type} ({len(spans)} match(es))",
                    category="healthcare_phi",
                    explanation=f"The content contains potential '{match_type}' identifiers, which may be PHI per HIPAA.",
                    confidence=self._confidence(match_type, spans, text),
                    recommendations=[
                        "Remove or redact PHI before sharing externally",
                        "Use pseudonymization or tokenization if linkability is required",
                        "Limit data exposure to minimum necessary per HIPAA",
                    ],
                    metadata={
                        "type": match_type,
                        "count": len(spans),
                        "spans": spans[:20],  # cap metadata size
                        "sample": snippet,
                    },
                )
            )

        return violations

    # Optional helper to offer redaction for callers/policies
    def redact_text(self, text: str) -> str:
        redacted = text
        for t, pattern in PHI_PATTERNS.items():
            if t not in self.enabled_patterns:
                continue

            def _sub(m: re.Match) -> str:
                val = m.group(0)
                if t == "email" and val.lower() in self.allowlist_emails:
                    return val
                if self.pseudonymize:
                    return self._pseudonymize(val, t)
                if self.preserve_format:
                    return self._preserve_format_token(val, t)
                return self.redaction_token.format(type=t)

            redacted = pattern.sub(_sub, redacted, count=self.max_matches_per_type)
        return redacted

    def redact_payload(self, payload: Any) -> Any:
        if isinstance(payload, str):
            return self.redact_text(payload)
        if isinstance(payload, list):
            return [self.redact_payload(v) for v in payload]
        if isinstance(payload, dict):
            return {k: self.redact_payload(v) for k, v in payload.items()}
        return payload

    # --- internals ---

    def _extract_text(self, action: Any) -> str:
        # Try common shapes: dict with 'content' or 'user_input', else str()
        if isinstance(action, dict):
            for key in ("content", "user_input", "agent_output", "context"):
                v = action.get(key)
                if isinstance(v, str) and v.strip():
                    return v
        return action if isinstance(action, str) else str(action or "")

    def _find_matches(self, text: str) -> Dict[str, List[Tuple[int, int, str]]]:
        results: Dict[str, List[Tuple[int, int, str]]] = {}
        for name, pattern in PHI_PATTERNS.items():
            if name not in self.enabled_patterns:
                continue
            spans: List[Tuple[int, int, str]] = []
            count = 0
            for m in pattern.finditer(text):
                val = m.group(0)
                if name == "email" and val.lower() in self.allowlist_emails:
                    continue
                spans.append((m.start(), m.end(), val))
                count += 1
                if count >= self.max_matches_per_type:
                    break
            if spans:
                results[name] = spans
        return results

    def _confidence(self, match_type: str, spans: List[Tuple[int, int, str]], text: str) -> float:
        # Simple heuristic: more matches + contextual cues => higher confidence
        base = 0.6
        if match_type in ("ssn_us", "mrn_generic", "health_plan_beneficiary", "license_number"):
            base = 0.85
        if any(k in text.lower() for k in ("patient", "mrn", "dob", "hipaa", "medical", "chart")):
            base += 0.1
        return min(0.99, base + min(0.15, 0.02 * len(spans)))

    def _pseudonymize(self, value: str, t: str) -> str:
        h = hashlib.sha256((self.hash_salt + value).encode("utf-8")).hexdigest()[:12]
        return f"[PSEUDO:{t}:{h}]"

    def _preserve_format_token(self, value: str, t: str) -> str:
        tok = self.redaction_token.format(type=t)
        # Pad to roughly match length to avoid breaking downstream parsing/layout
        if len(tok) >= len(value):
            return tok[: len(value)]
        return tok + ("*" * max(0, len(value) - len(tok)))

    def _safe_snippet(self, text: str, span: Tuple[int, int, str], ctx: int = 20) -> str:
        s, e, _ = span
        return text[max(0, s - ctx) : min(len(text), e + ctx)]


# Convenience function for payloads parallel to your current helper
def detect_and_redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    det = HealthcarePHIDetector()
    out = det.redact_payload(payload)
    return out
