"""
Pytest fixtures for misuse testing suite
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid


@pytest.fixture
def mock_audit_log():
    """Mock audit log for testing"""

    class MockAuditLog:
        def __init__(self):
            self.entries: List[Dict[str, Any]] = []

        def add_entry(self, event: str, timestamp: datetime, **kwargs):
            """Add audit log entry"""
            if self.entries and timestamp < self.entries[-1]["timestamp"]:
                raise ValueError("Backdated entry not allowed (P-NO-BACKDATE)")

            entry = {
                "index": len(self.entries),
                "event": event,
                "timestamp": timestamp,
                **kwargs,
            }
            self.entries.append(entry)
            return entry

        def get_entries(self):
            return self.entries

    return MockAuditLog()


@pytest.fixture
def mock_nonce_cache():
    """Mock nonce cache for replay prevention"""

    class MockNonceCache:
        def __init__(self):
            self.used_nonces: set = set()

        def check_and_add(self, nonce: str) -> bool:
            """Check if nonce is used, add if not"""
            if nonce in self.used_nonces:
                raise ValueError(
                    f"Replay attack detected: nonce {nonce} already used (P-NO-REPLAY)"
                )
            self.used_nonces.add(nonce)
            return True

    return MockNonceCache()


@pytest.fixture
def mock_rbac_system():
    """Mock RBAC system for authorization testing"""

    class MockRBAC:
        def __init__(self):
            self.permissions = {
                "viewer": {"read"},
                "operator": {"read", "write"},
                "admin": {"read", "write", "delete", "admin"},
            }

        def check_permission(self, role: str, action: str) -> bool:
            """Check if role has permission for action"""
            if role not in self.permissions:
                raise ValueError(f"Unknown role: {role}")

            if action not in self.permissions.get(role, set()):
                raise PermissionError(
                    f"Privilege escalation attempt: {role} cannot {action} (P-NO-PRIV-ESC)"
                )

            return True

    return MockRBAC()


@pytest.fixture
def mock_tenant_data():
    """Mock tenant data for isolation testing"""
    return {
        "tenant-a": {
            "id": "tenant-a",
            "name": "Tenant A",
            "data": {"secret": "tenant-a-secret"},
        },
        "tenant-b": {
            "id": "tenant-b",
            "name": "Tenant B",
            "data": {"secret": "tenant-b-secret"},
        },
    }


@pytest.fixture
def mock_policy_store():
    """Mock policy store for policy tampering tests"""

    class MockPolicyStore:
        def __init__(self):
            self.policies = {}

        def add_policy(self, policy_id: str, content: str, signatures: List[str]):
            """Add policy with signature verification"""
            if len(signatures) < 3:
                raise ValueError(
                    "Insufficient signatures for policy activation (P-NO-TAMPER)"
                )

            import hashlib

            policy_hash = hashlib.sha256(content.encode()).hexdigest()

            self.policies[policy_id] = {
                "content": content,
                "hash": policy_hash,
                "signatures": signatures,
            }

        def get_policy(self, policy_id: str):
            """Get policy and verify integrity"""
            if policy_id not in self.policies:
                raise KeyError(f"Policy {policy_id} not found")

            policy = self.policies[policy_id]
            import hashlib

            computed_hash = hashlib.sha256(policy["content"].encode()).hexdigest()

            if computed_hash != policy["hash"]:
                raise ValueError(
                    f"Policy tampering detected: hash mismatch (P-NO-TAMPER)"
                )

            return policy

    return MockPolicyStore()


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for DoS testing"""

    class MockRateLimiter:
        def __init__(self, max_requests: int = 100, window_seconds: int = 60):
            self.max_requests = max_requests
            self.window_seconds = window_seconds
            self.requests: Dict[str, List[datetime]] = {}

        def check_rate_limit(self, client_id: str) -> bool:
            """Check if client exceeds rate limit"""
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=self.window_seconds)

            if client_id not in self.requests:
                self.requests[client_id] = []

            # Remove old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] if req_time > cutoff
            ]

            if len(self.requests[client_id]) >= self.max_requests:
                raise ValueError(f"Rate limit exceeded for {client_id} (P-NO-DOS)")

            self.requests[client_id].append(now)
            return True

    return MockRateLimiter()


@pytest.fixture
def malicious_payloads():
    """Common malicious payloads for testing"""
    return {
        "sql_injection": [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "1' UNION SELECT * FROM passwords--",
            "admin'--",
            "' OR 1=1--",
        ],
        "xss": [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg/onload=alert('XSS')>",
        ],
        "command_injection": [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "`id`",
            "$(rm -rf /)",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
        ],
        "code_injection": [
            "__import__('os').system('whoami')",
            "eval('print(1)')",
            "exec('import os; os.system(\"ls\")')",
        ],
    }


@pytest.fixture
def test_jwt_tokens():
    """Test JWT tokens for authentication testing"""
    return {
        "valid_viewer": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2aWV3ZXIxIiwicm9sZSI6InZpZXdlciIsImlhdCI6MTYzMjQxNzIwMH0.fake",
        "valid_admin": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbjEiLCJyb2xlIjoiYWRtaW4iLCJpYXQiOjE2MzI0MTcyMDB9.fake",
        "forged": "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhdHRhY2tlciIsInJvbGUiOiJhZG1pbiJ9.",
        "expired": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMSIsImV4cCI6MTYwMDAwMDAwMH0.fake",
    }


@pytest.fixture(autouse=True)
def reset_test_state():
    """Reset test state before each test"""
    # This fixture runs automatically before each test
    yield
    # Cleanup after test
    pass
