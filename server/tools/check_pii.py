"""
PII Detection Tool

Scans text for potential Personally Identifiable Information including:
- Email addresses
- Social Security Numbers (SSN)
- Phone numbers
- Credit card numbers
- IP addresses
"""

import re
from typing import List
from ..models import ToolDefinition, ToolParameter, Finding


# PII detection patterns
# Order matters: more specific patterns first to avoid conflicts
PII_PATTERNS = [
    (
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "Email address detected",
        "HIGH",
        "email"
    ),
    (
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format: XXX-XX-XXXX (more specific, checked first)
        "Social Security Number (SSN) detected",
        "HIGH",
        "ssn"
    ),
    (
        r'\b(?:\d{3}[-.]?\d{4}|\d{10})\b',  # Phone: XXX-XXXX or XXXXXXXXXX (10 digits, excludes SSN pattern)
        "Phone number detected",
        "MEDIUM",
        "phone"
    ),
    (
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "Credit card number detected",
        "HIGH",
        "credit_card"
    ),
    (
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        "IP address detected",
        "LOW",
        "ip_address"
    ),
]


def get_tool_definition() -> ToolDefinition:
    """Return the tool definition for check_pii."""
    return ToolDefinition(
        name="check_pii",
        description="Scan text for potential PII including emails, SSNs, phone numbers, credit card numbers, and IP addresses",
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to scan for PII",
                required=True
            )
        ]
    )


def check_pii(text: str) -> List[Finding]:
    """
    Check text for Personally Identifiable Information.
    
    Args:
        text: Text to scan for PII
        
    Returns:
        List of findings
    """
    findings = []
    lines = text.split('\n')
    
    for pattern, message, severity, category in PII_PATTERNS:
        for i, line in enumerate(lines, 1):
            matches = re.finditer(pattern, line)
            for match in matches:
                # Mask the detected PII in the snippet
                matched_text = match.group()
                if category in ["email", "ssn", "credit_card"]:
                    # Safe masking: handle short strings
                    if len(matched_text) <= 6:
                        masked = matched_text[0] + "***" if len(matched_text) > 0 else "***"
                    else:
                        masked = matched_text[:3] + "***" + matched_text[-3:]
                else:
                    masked = matched_text
                
                findings.append(Finding(
                    severity=severity,
                    category=f"pii_{category}",
                    message=f"{message}: {masked}",
                    line=i,
                    code_snippet=line.strip()
                ))
    
    return findings


# Export the tool as a callable with its definition
check_pii_tool = {
    "definition": get_tool_definition(),
    "function": check_pii
}
