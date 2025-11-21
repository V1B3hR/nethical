"""
Tests for check_pii tool

Validates detection of PII in text.
"""

import pytest
from server.tools.check_pii import check_pii, get_tool_definition


class TestCheckPIIDefinition:
    """Test the tool definition."""
    
    def test_tool_definition(self):
        """Test that tool definition is properly structured."""
        definition = get_tool_definition()
        assert definition.name == "check_pii"
        assert len(definition.parameters) == 1
        assert definition.parameters[0].name == "text"
        assert definition.parameters[0].required is True


class TestEmailDetection:
    """Test detection of email addresses."""
    
    def test_detect_simple_email(self):
        """Test detection of simple email address."""
        text = "Contact me at john.doe@example.com for more info"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("email" in f.message.lower() for f in findings)
    
    def test_detect_multiple_emails(self):
        """Test detection of multiple email addresses."""
        text = """
        Send to alice@company.com and bob@company.com
        """
        findings = check_pii(text)
        assert len(findings) >= 2
        email_findings = [f for f in findings if "email" in f.message.lower()]
        assert len(email_findings) == 2
    
    def test_detect_complex_email(self):
        """Test detection of complex email format."""
        text = "Email: user+tag@sub.domain.co.uk"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any("email" in f.message.lower() for f in findings)


class TestSSNDetection:
    """Test detection of Social Security Numbers."""
    
    def test_detect_ssn(self):
        """Test detection of SSN in standard format."""
        text = "SSN: 123-45-6789"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("ssn" in f.message.lower() or "social security" in f.message.lower() 
                   for f in findings)
    
    def test_ssn_masked_in_output(self):
        """Test that SSN is masked in finding message."""
        text = "My SSN is 123-45-6789"
        findings = check_pii(text)
        assert len(findings) > 0
        # Should be masked like "123***789"
        ssn_finding = next(f for f in findings if "ssn" in f.message.lower() 
                          or "social security" in f.message.lower())
        assert "***" in ssn_finding.message


class TestPhoneDetection:
    """Test detection of phone numbers."""
    
    def test_detect_phone_with_hyphens(self):
        """Test detection of phone with hyphens."""
        text = "Call me at 555-123-4567"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any("phone" in f.message.lower() for f in findings)
    
    def test_detect_phone_with_dots(self):
        """Test detection of phone with dots."""
        text = "Phone: 555.123.4567"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any("phone" in f.message.lower() for f in findings)
    
    def test_detect_phone_no_separator(self):
        """Test detection of phone without separators."""
        text = "Contact: 5551234567"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any("phone" in f.message.lower() for f in findings)


class TestCreditCardDetection:
    """Test detection of credit card numbers."""
    
    def test_detect_credit_card_with_spaces(self):
        """Test detection of credit card with spaces."""
        text = "Card: 4532 1234 5678 9010"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any(f.severity == "HIGH" for f in findings)
        assert any("credit card" in f.message.lower() for f in findings)
    
    def test_detect_credit_card_with_hyphens(self):
        """Test detection of credit card with hyphens."""
        text = "Payment: 4532-1234-5678-9010"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any("credit card" in f.message.lower() for f in findings)
    
    def test_detect_credit_card_no_separator(self):
        """Test detection of credit card without separators."""
        text = "CC: 4532123456789010"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any("credit card" in f.message.lower() for f in findings)


class TestIPAddressDetection:
    """Test detection of IP addresses."""
    
    def test_detect_ip_address(self):
        """Test detection of IP address."""
        text = "Server IP: 192.168.1.100"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any(f.severity == "LOW" for f in findings)
        assert any("ip" in f.message.lower() for f in findings)
    
    def test_detect_public_ip(self):
        """Test detection of public IP."""
        text = "Connect to 8.8.8.8"
        findings = check_pii(text)
        assert len(findings) > 0
        assert any("ip" in f.message.lower() for f in findings)


class TestMixedPII:
    """Test detection of multiple PII types."""
    
    def test_detect_mixed_pii(self):
        """Test detection of multiple PII types in same text."""
        text = """
        Contact: john@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        """
        findings = check_pii(text)
        assert len(findings) >= 3
        
        # Check for each type
        assert any("email" in f.message.lower() for f in findings)
        assert any("phone" in f.message.lower() for f in findings)
        assert any("ssn" in f.message.lower() or "social security" in f.message.lower() 
                   for f in findings)


class TestCleanText:
    """Test that clean text produces no findings."""
    
    def test_clean_text_no_findings(self):
        """Test that text without PII produces no findings."""
        text = """
        This is a sample text that contains no personal information.
        It talks about general topics and public information.
        """
        findings = check_pii(text)
        assert len(findings) == 0


class TestFindingDetails:
    """Test that findings include proper details."""
    
    def test_finding_includes_line_number(self):
        """Test that findings include line numbers."""
        text = """
Line 1
Email: test@example.com
Line 3
"""
        findings = check_pii(text)
        assert len(findings) > 0
        assert findings[0].line is not None
        assert findings[0].line > 0
    
    def test_finding_includes_code_snippet(self):
        """Test that findings include text snippet."""
        text = "Contact: user@example.com"
        findings = check_pii(text)
        assert len(findings) > 0
        assert findings[0].code_snippet is not None
        assert len(findings[0].code_snippet) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
