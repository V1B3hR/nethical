"""Log Sanitization and PII Redaction

Automatically redacts PII from logs to ensure privacy compliance.
Supports detection and redaction of:
- SSNs, credit cards, emails, phone numbers
- API keys, tokens, passwords
- Custom patterns
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Pattern
from dataclasses import dataclass
import hashlib


@dataclass
class RedactionPattern:
    """A pattern for detecting and redacting sensitive data."""
    name: str
    pattern: Pattern[str]
    replacement: str
    description: str


class LogSanitizer:
    """Sanitizes logs by redacting PII and sensitive information."""
    
    # Predefined patterns for common PII
    PATTERNS = {
        'ssn': RedactionPattern(
            name='ssn',
            pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            replacement='[SSN-REDACTED]',
            description='Social Security Number'
        ),
        'credit_card': RedactionPattern(
            name='credit_card',
            pattern=re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            replacement='[CARD-REDACTED]',
            description='Credit card number'
        ),
        'email': RedactionPattern(
            name='email',
            pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            replacement='[EMAIL-REDACTED]',
            description='Email address'
        ),
        'phone': RedactionPattern(
            name='phone',
            pattern=re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            replacement='[PHONE-REDACTED]',
            description='Phone number'
        ),
        'ip_address': RedactionPattern(
            name='ip_address',
            pattern=re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            replacement='[IP-REDACTED]',
            description='IP address'
        ),
        'api_key': RedactionPattern(
            name='api_key',
            pattern=re.compile(r'\b(?:api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
            replacement='api_key="[KEY-REDACTED]"',
            description='API key'
        ),
        'bearer_token': RedactionPattern(
            name='bearer_token',
            pattern=re.compile(r'\bBearer\s+([a-zA-Z0-9_\-\.]+)', re.IGNORECASE),
            replacement='Bearer [TOKEN-REDACTED]',
            description='Bearer token'
        ),
        'password': RedactionPattern(
            name='password',
            pattern=re.compile(r'\b(?:password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^\s"\']+)["\']?', re.IGNORECASE),
            replacement='password="[PASSWORD-REDACTED]"',
            description='Password'
        ),
        'aws_key': RedactionPattern(
            name='aws_key',
            pattern=re.compile(r'\b(AKIA[0-9A-Z]{16})\b'),
            replacement='[AWS-KEY-REDACTED]',
            description='AWS Access Key'
        ),
        'jwt': RedactionPattern(
            name='jwt',
            pattern=re.compile(r'\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b'),
            replacement='[JWT-REDACTED]',
            description='JWT token'
        ),
    }
    
    def __init__(self, 
                enabled_patterns: Optional[List[str]] = None,
                custom_patterns: Optional[List[RedactionPattern]] = None,
                hash_redacted: bool = False):
        """Initialize log sanitizer.
        
        Args:
            enabled_patterns: List of pattern names to enable (None = all)
            custom_patterns: Additional custom patterns
            hash_redacted: Whether to include hash of redacted value for debugging
        """
        self.hash_redacted = hash_redacted
        
        # Select patterns
        if enabled_patterns is None:
            self.patterns = list(self.PATTERNS.values())
        else:
            self.patterns = [
                self.PATTERNS[name] for name in enabled_patterns 
                if name in self.PATTERNS
            ]
        
        # Add custom patterns
        if custom_patterns:
            self.patterns.extend(custom_patterns)
    
    def sanitize(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Sanitize text by redacting PII and sensitive information.
        
        Args:
            text: Text to sanitize
            context: Optional context for logging
            
        Returns:
            Sanitized text with PII redacted
        """
        if not text:
            return text
        
        sanitized = text
        redactions = []
        
        for pattern_def in self.patterns:
            matches = list(pattern_def.pattern.finditer(sanitized))
            
            for match in reversed(matches):  # Reverse to maintain string positions
                original = match.group(0)
                
                if self.hash_redacted:
                    # Include hash for debugging
                    hash_val = hashlib.sha256(original.encode()).hexdigest()[:8]
                    replacement = f"{pattern_def.replacement}-{hash_val}"
                else:
                    replacement = pattern_def.replacement
                
                sanitized = sanitized[:match.start()] + replacement + sanitized[match.end():]
                
                redactions.append({
                    'type': pattern_def.name,
                    'position': match.start(),
                    'description': pattern_def.description
                })
        
        return sanitized
    
    def sanitize_dict(self, data: Dict[str, Any], 
                     recursive: bool = True,
                     sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Sanitize dictionary values.
        
        Args:
            data: Dictionary to sanitize
            recursive: Whether to recursively sanitize nested dicts
            sensitive_keys: Additional keys to always redact
            
        Returns:
            Sanitized dictionary
        """
        sensitive_keys = sensitive_keys or []
        sensitive_keys_lower = [k.lower() for k in sensitive_keys]
        
        # Common sensitive key names
        default_sensitive = ['password', 'passwd', 'pwd', 'secret', 'token', 
                           'api_key', 'apikey', 'auth', 'authorization',
                           'ssn', 'credit_card', 'cvv']
        
        result = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key itself is sensitive
            if any(sensitive in key_lower for sensitive in default_sensitive + sensitive_keys_lower):
                result[key] = '[REDACTED]'
            elif isinstance(value, str):
                result[key] = self.sanitize(value)
            elif isinstance(value, dict) and recursive:
                result[key] = self.sanitize_dict(value, recursive, sensitive_keys)
            elif isinstance(value, list) and recursive:
                result[key] = [
                    self.sanitize_dict(item, recursive, sensitive_keys) if isinstance(item, dict)
                    else self.sanitize(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    def add_pattern(self, pattern: RedactionPattern) -> None:
        """Add a custom redaction pattern.
        
        Args:
            pattern: RedactionPattern to add
        """
        self.patterns.append(pattern)
    
    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove a redaction pattern by name.
        
        Args:
            pattern_name: Name of pattern to remove
            
        Returns:
            True if pattern was found and removed
        """
        original_len = len(self.patterns)
        self.patterns = [p for p in self.patterns if p.name != pattern_name]
        return len(self.patterns) < original_len
    
    def list_patterns(self) -> List[Dict[str, str]]:
        """List all active patterns.
        
        Returns:
            List of pattern information
        """
        return [
            {
                'name': p.name,
                'description': p.description,
                'replacement': p.replacement
            }
            for p in self.patterns
        ]


# Global singleton
_log_sanitizer: Optional[LogSanitizer] = None


def get_sanitizer(enabled_patterns: Optional[List[str]] = None,
                 custom_patterns: Optional[List[RedactionPattern]] = None) -> LogSanitizer:
    """Get or create global log sanitizer.
    
    Args:
        enabled_patterns: List of pattern names to enable
        custom_patterns: Additional custom patterns
        
    Returns:
        LogSanitizer instance
    """
    global _log_sanitizer
    
    if _log_sanitizer is None:
        _log_sanitizer = LogSanitizer(
            enabled_patterns=enabled_patterns,
            custom_patterns=custom_patterns
        )
    
    return _log_sanitizer


def sanitize_log(text: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Sanitize a log message (convenience function).
    
    Args:
        text: Text to sanitize
        context: Optional context
        
    Returns:
        Sanitized text
    """
    sanitizer = get_sanitizer()
    return sanitizer.sanitize(text, context)


def sanitize_dict(data: Dict[str, Any], 
                 recursive: bool = True,
                 sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Sanitize a dictionary (convenience function).
    
    Args:
        data: Dictionary to sanitize
        recursive: Whether to recursively sanitize
        sensitive_keys: Additional sensitive keys
        
    Returns:
        Sanitized dictionary
    """
    sanitizer = get_sanitizer()
    return sanitizer.sanitize_dict(data, recursive, sensitive_keys)


if __name__ == '__main__':
    # Demo usage
    sanitizer = LogSanitizer()
    
    # Test cases
    test_strings = [
        "My SSN is 123-45-6789",
        "Email me at john.doe@example.com",
        "Call me at +1-555-123-4567",
        "API Key: example_api_key_1234567890abcdef",  # Fake example key
        "Authorization: Bearer example.jwt.token",
        "My password is super_secret_123",
    ]
    
    print("Log Sanitization Demo")
    print("=" * 60)
    
    for original in test_strings:
        sanitized = sanitizer.sanitize(original)
        print(f"Original:  {original}")
        print(f"Sanitized: {sanitized}")
        print()
    
    # Test dictionary sanitization
    test_dict = {
        'user': 'john.doe@example.com',
        'password': 'secret123',
        'api_key': 'testkey_abc123',  # Fake test key
        'message': 'Call me at 555-1234',
        'metadata': {
            'ssn': '123-45-6789',
            'safe_data': 'This is fine'
        }
    }
    
    print("Dictionary Sanitization:")
    print("Original:", test_dict)
    sanitized_dict = sanitizer.sanitize_dict(test_dict)
    print("Sanitized:", sanitized_dict)
