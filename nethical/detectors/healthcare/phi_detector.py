from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple

PHI_PATTERNS = {
    "ssn_us": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn_generic": re.compile(r"\bMRN[:\s]*[A-Za-z0-9\-]{5,}\b", re.IGNORECASE),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "address_hint": re.compile(r"\b(\d{1,6}\s+[A-Za-z0-9\.\-]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr))\b", re.IGNORECASE),
    "dob": re.compile(r"\b(?:DOB[:\s]*)?(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b", re.IGNORECASE),
}

REDACTION_TOKEN = "[REDACTED-PHI]"

class PHIDetector:
    def __init__(self, max_matches:int=50):
        self.max_matches = max_matches

    def find(self, text:str) -> List[Tuple[str, Tuple[int,int], str]]:
        findings = []
        for name, pattern in PHI_PATTERNS.items():
            for m in pattern.finditer(text or ""):
                findings.append((name, m.span(), m.group(0)))
                if len(findings) >= self.max_matches:
                    return findings
        return findings

    def redact(self, text:str) -> str:
        redacted = text
        for pattern in PHI_PATTERNS.values():
            redacted = pattern.sub(REDACTION_TOKEN, redacted)
        return redacted

def detect_and_redact_payload(payload:Dict[str, Any]) -> Dict[str, Any]:
    det = PHIDetector()
    out = dict(payload)
    for field in ("user_input", "agent_output", "context", "metadata"):
        val = out.get(field)
        if isinstance(val, str):
            out[field] = det.redact(val)
    return out
