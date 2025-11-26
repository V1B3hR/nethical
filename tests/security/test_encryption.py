"""
Tests for AES-GCM Encryption Implementation

This module tests the encryption functionality including:
- Encryption/decryption round-trip
- Different key sizes
- Tampering detection
- Nonce uniqueness
"""

import pytest
import secrets
from datetime import datetime, timezone

from nethical.security.encryption import (
    MilitaryGradeEncryption,
    EncryptedData,
    EncryptionAlgorithm,
    KeyRotationPolicy,
    HSMConfig,
    KeyManagementService,
)


class TestAESGCMEncryption:
    """Tests for AES-GCM encryption implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encryption = MilitaryGradeEncryption()

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_roundtrip(self):
        """Verify AES-GCM encryption/decryption works correctly."""
        plaintext = b"Secret message to encrypt and decrypt"

        encrypted = await self.encryption.encrypt(plaintext)
        decrypted = await self.encryption.decrypt(encrypted)

        assert decrypted == plaintext
        assert encrypted.ciphertext != plaintext
        assert encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_empty_data(self):
        """Test encryption of empty data."""
        plaintext = b""

        encrypted = await self.encryption.encrypt(plaintext)
        decrypted = await self.encryption.decrypt(encrypted)

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_large_data(self):
        """Test encryption of large data."""
        plaintext = secrets.token_bytes(1024 * 1024)  # 1MB

        encrypted = await self.encryption.encrypt(plaintext)
        decrypted = await self.encryption.decrypt(encrypted)

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_encryption_with_aad(self):
        """Test encryption with additional authenticated data (AAD)."""
        plaintext = b"Sensitive data"
        aad = b"metadata:user123:timestamp:2024"

        encrypted = await self.encryption.encrypt(plaintext, additional_data=aad)
        decrypted = await self.encryption.decrypt(encrypted, additional_data=aad)

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_tampering_detection_ciphertext(self):
        """Verify AEAD detects tampered ciphertext."""
        plaintext = b"Sensitive data"

        encrypted = await self.encryption.encrypt(plaintext)

        # Tamper with the ciphertext
        tampered = bytes([b ^ 0xFF for b in encrypted.ciphertext])
        encrypted.ciphertext = tampered

        with pytest.raises(ValueError, match="Authentication tag verification failed"):
            await self.encryption.decrypt(encrypted)

    @pytest.mark.asyncio
    async def test_tampering_detection_tag(self):
        """Verify AEAD detects tampered authentication tag."""
        plaintext = b"Sensitive data"

        encrypted = await self.encryption.encrypt(plaintext)

        # Tamper with the tag
        tampered_tag = bytes([b ^ 0xFF for b in encrypted.tag])
        encrypted.tag = tampered_tag

        with pytest.raises(ValueError, match="Authentication tag verification failed"):
            await self.encryption.decrypt(encrypted)

    @pytest.mark.asyncio
    async def test_tampering_detection_wrong_aad(self):
        """Verify AEAD detects wrong AAD."""
        plaintext = b"Sensitive data"
        aad = b"correct_aad"

        encrypted = await self.encryption.encrypt(plaintext, additional_data=aad)

        with pytest.raises(ValueError, match="Authentication tag verification failed"):
            await self.encryption.decrypt(encrypted, additional_data=b"wrong_aad")

    @pytest.mark.asyncio
    async def test_nonce_uniqueness(self):
        """Verify unique nonces are generated for each encryption."""
        plaintext = b"Same message"

        encrypted1 = await self.encryption.encrypt(plaintext)
        encrypted2 = await self.encryption.encrypt(plaintext)
        encrypted3 = await self.encryption.encrypt(plaintext)

        # All nonces should be unique
        nonces = {encrypted1.nonce, encrypted2.nonce, encrypted3.nonce}
        assert len(nonces) == 3

        # All ciphertexts should be different due to unique nonces
        assert encrypted1.ciphertext != encrypted2.ciphertext
        assert encrypted2.ciphertext != encrypted3.ciphertext

    @pytest.mark.asyncio
    async def test_nonce_length(self):
        """Verify nonce is 96 bits (12 bytes) for AES-GCM."""
        plaintext = b"Test message"

        encrypted = await self.encryption.encrypt(plaintext)

        assert len(encrypted.nonce) == 12  # 96 bits

    @pytest.mark.asyncio
    async def test_tag_length(self):
        """Verify authentication tag is 128 bits (16 bytes)."""
        plaintext = b"Test message"

        encrypted = await self.encryption.encrypt(plaintext)

        assert len(encrypted.tag) == 16  # 128 bits


class TestKeyManagement:
    """Tests for key management functionality."""

    def test_key_generation(self):
        """Test key generation."""
        kms = KeyManagementService()
        key_id = kms.generate_key(key_id="test-key")

        assert key_id == "test-key"
        key = kms.get_key("test-key")
        assert key is not None
        assert len(key) == 32  # 256 bits

    def test_key_rotation_policy(self):
        """Test key rotation policy configuration."""
        policy = KeyRotationPolicy(
            interval_days=30,
            auto_rotate=True,
            retain_old_keys=5,
        )

        encryption = MilitaryGradeEncryption(key_rotation_policy=policy)
        status = encryption.get_key_rotation_status()

        assert status["policy"]["interval_days"] == 30
        assert status["policy"]["auto_rotate"] is True
        assert status["policy"]["retain_old_keys"] == 5

    def test_list_keys(self):
        """Test listing all keys."""
        kms = KeyManagementService()
        kms.generate_key(key_id="key1")
        kms.generate_key(key_id="key2")
        kms.generate_key(key_id="key3")

        keys = kms.list_keys()
        key_ids = [k["key_id"] for k in keys]

        assert "key1" in key_ids
        assert "key2" in key_ids
        assert "key3" in key_ids


class TestEncryptedDataSerialization:
    """Tests for EncryptedData serialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now(timezone.utc)
        encrypted = EncryptedData(
            ciphertext=b"encrypted_data",
            nonce=b"random_nonce",
            tag=b"auth_tag_12345",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id="key-123",
            timestamp=timestamp,
        )

        data = encrypted.to_dict()

        assert data["algorithm"] == "aes-256-gcm"
        assert data["key_id"] == "key-123"
        assert data["ciphertext"] == b"encrypted_data".hex()
        assert data["nonce"] == b"random_nonce".hex()
        assert data["tag"] == b"auth_tag_12345".hex()

    def test_from_dict(self):
        """Test creation from dictionary."""
        timestamp = datetime.now(timezone.utc)
        data = {
            "ciphertext": b"encrypted_data".hex(),
            "nonce": b"random_nonce".hex(),
            "tag": b"auth_tag_12345".hex(),
            "algorithm": "aes-256-gcm",
            "key_id": "key-123",
            "timestamp": timestamp.isoformat(),
            "metadata": {"test": "value"},
        }

        encrypted = EncryptedData.from_dict(data)

        assert encrypted.ciphertext == b"encrypted_data"
        assert encrypted.nonce == b"random_nonce"
        assert encrypted.tag == b"auth_tag_12345"
        assert encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert encrypted.key_id == "key-123"

    def test_roundtrip_serialization(self):
        """Test serialization round-trip."""
        original = EncryptedData(
            ciphertext=b"test_ciphertext",
            nonce=b"test_nonce12",  # 12 bytes
            tag=b"test_tag_16_byt",  # 16 bytes
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id="test-key-id",
            timestamp=datetime.now(timezone.utc),
            metadata={"custom": "data"},
        )

        data = original.to_dict()
        restored = EncryptedData.from_dict(data)

        assert restored.ciphertext == original.ciphertext
        assert restored.nonce == original.nonce
        assert restored.tag == original.tag
        assert restored.algorithm == original.algorithm
        assert restored.key_id == original.key_id


class TestGovernanceEncryption:
    """Tests for governance-specific encryption features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encryption = MilitaryGradeEncryption()

    @pytest.mark.asyncio
    async def test_encrypt_governance_decision(self):
        """Test encrypting governance decision."""
        decision = {
            "decision_id": "dec-12345",
            "action": "approved",
            "reason": "Safe content",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        encrypted = await self.encryption.encrypt_governance_decision(decision)

        assert encrypted.ciphertext is not None
        assert encrypted.metadata.get("has_aad") is True

    @pytest.mark.asyncio
    async def test_decrypt_governance_decision(self):
        """Test decrypting governance decision."""
        decision = {
            "decision_id": "dec-12345",
            "action": "blocked",
            "reason": "Unsafe content detected",
        }

        encrypted = await self.encryption.encrypt_governance_decision(decision)
        decrypted = await self.encryption.decrypt_governance_decision(
            encrypted, decision_id="dec-12345"
        )

        assert decrypted["decision_id"] == "dec-12345"
        assert decrypted["action"] == "blocked"
        assert decrypted["reason"] == "Unsafe content detected"

    @pytest.mark.asyncio
    async def test_encrypt_audit_log(self):
        """Test encrypting audit log."""
        log_entry = {
            "entry_id": "log-67890",
            "event": "user_action",
            "user_id": "user123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        merkle_root = b"merkle_hash_value"

        encrypted = await self.encryption.encrypt_audit_log(log_entry, merkle_root)

        assert encrypted.metadata["is_audit_log"] is True
        assert encrypted.metadata["merkle_root"] == merkle_root.hex()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
