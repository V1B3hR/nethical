"""
Tests for evaluate_code tool

Validates detection of security issues in code snippets.
"""

import pytest
from server.tools.evaluate_code import evaluate_code, get_tool_definition


class TestEvaluateCodeDefinition:
    """Test the tool definition."""
    
    def test_tool_definition(self):
        """Test that tool definition is properly structured."""
        definition = get_tool_definition()
        assert definition.name == "evaluate_code"
        assert len(definition.parameters) == 1
        assert definition.parameters[0].name == "code"
        assert definition.parameters[0].required is True


class TestWeakHashing:
    """Test detection of weak hashing algorithms."""
    
    def test_detect_md5(self):
        """Test detection of MD5 usage."""
        code = """
import hashlib
password_hash = hashlib.md5(b"password").hexdigest()
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("MD5" in f.message for f in findings)
    
    def test_detect_sha1(self):
        """Test detection of SHA1 usage."""
        code = """
import hashlib
hash_value = hashlib.sha1(b"data").hexdigest()
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("SHA1" in f.message for f in findings)
    
    def test_no_detection_for_safe_hash(self):
        """Test that safe hashing algorithms are not flagged."""
        code = """
import hashlib
secure_hash = hashlib.sha256(b"data").hexdigest()
"""
        findings = evaluate_code(code)
        # Should not flag SHA256
        assert not any("sha256" in f.message.lower() for f in findings)


class TestInsecureCrypto:
    """Test detection of insecure cryptographic functions."""
    
    def test_detect_des(self):
        """Test detection of DES encryption."""
        code = """
from Crypto.Cipher import DES
cipher = DES.new(key, DES.MODE_ECB)
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any("DES" in f.message for f in findings)
    
    def test_detect_ecb_mode(self):
        """Test detection of ECB mode."""
        code = """
cipher = AES.new(key, AES.MODE_ECB)
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any("ECB" in f.message for f in findings)
    
    def test_detect_random_for_security(self):
        """Test detection of non-cryptographic random."""
        code = """
import random
token = random.random()
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any(f.severity == "LOW" for f in findings)


class TestHardcodedSecrets:
    """Test detection of hardcoded secrets."""
    
    def test_detect_hardcoded_password(self):
        """Test detection of hardcoded password."""
        code = """
password = "mySecretPassword123"
db.connect(user="admin", password=password)
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("password" in f.message.lower() for f in findings)
    
    def test_detect_api_key(self):
        """Test detection of hardcoded API key."""
        code = """
api_key = "sk_test_1234567890abcdef"
headers = {"Authorization": f"Bearer {api_key}"}
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("api" in f.message.lower() for f in findings)
    
    def test_detect_aws_secret(self):
        """Test detection of AWS secret key."""
        code = """
aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("AWS" in f.message or "secret" in f.message.lower() for f in findings)


class TestCleanCode:
    """Test that clean code produces no findings."""
    
    def test_clean_code_no_findings(self):
        """Test that secure code produces no findings."""
        code = """
import hashlib
import secrets

def hash_password(password):
    salt = secrets.token_bytes(16)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
"""
        findings = evaluate_code(code)
        assert len(findings) == 0


class TestFindingDetails:
    """Test that findings include proper details."""
    
    def test_finding_includes_line_number(self):
        """Test that findings include line numbers."""
        code = """
import hashlib
x = 1
hash_val = hashlib.md5(b"test").hexdigest()
"""
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert findings[0].line is not None
        assert findings[0].line > 0
    
    def test_finding_includes_code_snippet(self):
        """Test that findings include code snippet."""
        code = 'password = "secret123"'
        findings = evaluate_code(code)
        assert len(findings) > 0
        assert findings[0].code_snippet is not None
        assert len(findings[0].code_snippet) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
