"""Test suite for Cryptographic Curve Audit & CVE-2026-26007 Remediation."""

import pytest
from pathlib import Path
from nethical.security.audit_crypto_curves import (
    verify_cryptography_library_version,
    validate_curve_safety,
    scan_repository_for_crypto_curves,
    run_comprehensive_crypto_audit,
    CryptoSecurityViolation,
    PROHIBITED_BINARY_CURVES,
    MINIMUM_SECURE_CRYPTOGRAPHY_VERSION,
)
from nethical.security.token_vault import ReversibleTokenVault


def test_cryptography_version_is_patched_for_cve_2026_26007():
    """Verify that installed cryptography version is strictly >= 46.0.5."""
    is_safe, msg = verify_cryptography_library_version()
    assert is_safe, f"System is vulnerable to CVE-2026-26007! {msg}"


def test_runtime_curve_safety_rejection():
    """Verify that attempting to validate or use any binary curve raises CryptoSecurityViolation."""
    # Test safe curves pass
    assert validate_curve_safety("secp256r1") is True
    assert validate_curve_safety("secp384r1") is True
    assert validate_curve_safety("ed25519") is True

    # Test vulnerable binary curves fail immediately
    for bad_curve in ["sect163r1", "sect283k1", "sect571r1", "c2tnb191v1"]:
        with pytest.raises(CryptoSecurityViolation) as exc_info:
            validate_curve_safety(bad_curve)
        assert "CVE-2026-26007" in str(exc_info.value)


def test_ast_scan_detects_zero_binary_curves_in_nethical():
    """Verify AST scan of Nethical package finds zero prohibited binary curves."""
    repo_root = Path(__file__).resolve().parent.parent.parent / "nethical"
    result = scan_repository_for_crypto_curves(repo_root)
    assert result["is_clean"] is True, f"Found prohibited curves in codebase: {result['findings']}"
    assert result["prohibited_curves_found"] == 0
    assert result["scanned_files_count"] > 20


def test_token_vault_key_rotation_under_aes_gcm():
    """Verify that TokenVault key rotation generates fresh keys and remains functional."""
    vault = ReversibleTokenVault()
    original_key_sha = vault.current_key_sha256

    # Tokenize sample PII
    res = vault.tokenize("Patient John Doe with PESEL 90010112345")
    tokenized_text = res.sanitized_text
    assert "90010112345" not in tokenized_text

    # Rotate key
    new_key_sha = vault.rotate_key()
    assert new_key_sha != original_key_sha

    # Verify detokenization still succeeds using stored historical keys
    restored = vault.detokenize(tokenized_text, session_id=res.session_id)
    assert "90010112345" in restored.restored_text


def test_comprehensive_crypto_audit_execution():
    """Verify the full audit report executes and passes with certified compliance."""
    repo_root = Path(__file__).resolve().parent.parent.parent / "nethical"
    audit_report = run_comprehensive_crypto_audit(repo_root)
    assert audit_report["cve_remediated"] == "CVE-2026-26007"
    assert audit_report["cryptography_version_safe"] is True
    assert audit_report["audit_passed"] is True
    assert audit_report["prohibited_curves_in_codebase"] == 0
