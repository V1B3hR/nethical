"""
Code Evaluation Tool

Scans code snippets for ethical and security issues including:
- Weak hashing algorithms (MD5, SHA1)
- Insecure cryptographic functions
- Hardcoded secrets and credentials
"""

import re
from typing import List
from ..models import ToolDefinition, ToolParameter, Finding


# Detection patterns for security issues
WEAK_HASH_PATTERNS = [
    (r'\bhashlib\.md5\b', "Use of weak MD5 hashing algorithm"),
    (r'\bhashlib\.sha1\b', "Use of weak SHA1 hashing algorithm"),
    (r'\bMd5\b', "Use of weak MD5 hashing algorithm"),
    (r'\bSHA1\b', "Use of weak SHA1 hashing algorithm"),
]

INSECURE_CRYPTO_PATTERNS = [
    (r'\bDES\b', "Use of insecure DES encryption"),
    (r'\bRC4\b', "Use of insecure RC4 encryption"),
    (r'MODE_ECB|\.ECB\b', "Use of insecure ECB mode"),
    (r'random\.random\(\)', "Use of non-cryptographic random for security"),
]

HARDCODED_SECRET_PATTERNS = [
    (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
    (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
    (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
    (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token detected"),
    (r'aws_secret_access_key\s*=\s*["\'][^"\']+["\']', "Hardcoded AWS secret key detected"),
    (r'private_key\s*=\s*["\'][^"\']+["\']', "Hardcoded private key detected"),
]


def get_tool_definition() -> ToolDefinition:
    """Return the tool definition for evaluate_code."""
    return ToolDefinition(
        name="evaluate_code",
        description="Scan code snippets for ethical and security issues including weak hashing, insecure cryptography, and hardcoded secrets",
        parameters=[
            ToolParameter(
                name="code",
                type="string",
                description="The code snippet to evaluate",
                required=True
            )
        ]
    )


def evaluate_code(code: str) -> List[Finding]:
    """
    Evaluate code for security and ethical issues.
    
    Args:
        code: Code snippet to evaluate
        
    Returns:
        List of findings
    """
    findings = []
    lines = code.split('\n')
    
    # Check for weak hashing
    for pattern, message in WEAK_HASH_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                findings.append(Finding(
                    severity="HIGH",
                    category="weak_crypto",
                    message=message,
                    line=i,
                    code_snippet=line.strip()
                ))
    
    # Check for insecure crypto
    for pattern, message in INSECURE_CRYPTO_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                findings.append(Finding(
                    severity="MEDIUM",
                    category="insecure_crypto",
                    message=message,
                    line=i,
                    code_snippet=line.strip()
                ))
    
    # Check for hardcoded secrets
    for pattern, message in HARDCODED_SECRET_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                findings.append(Finding(
                    severity="HIGH",
                    category="hardcoded_secret",
                    message=message,
                    line=i,
                    code_snippet=line.strip()
                ))
    
    return findings


# Export the tool as a callable with its definition
evaluate_code_tool = {
    "definition": get_tool_definition(),
    "function": evaluate_code
}
