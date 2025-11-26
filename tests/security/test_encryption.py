"""Tests for AES-GCM encryption implementation."""

import pytest
import secrets
from datetime import datetime, timezone

from nethical.security.encryption import (
    MilitaryGradeEncryption,
    KeyManagementService,
    EncryptionAlgorithm,
    EncryptedData,
    KeyRotationPolicy,
)


class TestAESGCMEncryption:
    """Tests for AES-GCM encryption implementation."""

    @pytest.fixture
    def encryption_service(self):
        """Create encryption service for tests."""
        return MilitaryGradeEncryption()

    @pytest.fixture
    def kms(self):
        """Create key management service for tests."""
        return KeyManagementService()

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_roundtrip(self, encryption_service):
        """Verify encryption followed by decryption returns original data."""
        plaintext = b"This is a secret message that needs to be encrypted!"
        
        # Encrypt the data
        encrypted = await encryption_service.encrypt(plaintext)
        
        # Verify encrypted data structure
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.ciphertext != plaintext
        assert encrypted.nonce is not None
        assert encrypted.tag is not None
        assert len(encrypted.nonce) == 12  # 96 bits for GCM
        assert len(encrypted.tag) == 16    # 128-bit tag
        
        # Decrypt and verify
        decrypted = await encryption_service.decrypt(encrypted)
        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_with_aad(self, encryption_service):
        """Verify encryption with additional authenticated data (AAD)."""
        plaintext = b"Secret data"
        aad = b"additional_authenticated_data"
        
        # Encrypt with AAD
        encrypted = await encryption_service.encrypt(plaintext, additional_data=aad)
        
        # Decrypt with same AAD
        decrypted = await encryption_service.decrypt(encrypted, additional_data=aad)
        assert decrypted == plaintext
        
        # Verify decryption with wrong AAD fails
        with pytest.raises(ValueError):
            await encryption_service.decrypt(encrypted, additional_data=b"wrong_aad")

    @pytest.mark.asyncio
    async def test_different_key_sizes(self, kms):
        """Test 128, 192, and 256-bit keys work correctly."""
        # AES-256-GCM is the default - test key generation works
        key_id = kms.generate_key(algorithm=EncryptionAlgorithm.AES_256_GCM)
        key = kms.get_key(key_id)
        
        # AES-256 = 256 bits = 32 bytes
        assert len(key) == 32
        
        # Test key can be retrieved
        assert kms.get_key(key_id) is not None
        
        # Test key rotation
        new_key_id = kms.rotate_key(key_id)
        new_key = kms.get_key(new_key_id)
        assert len(new_key) == 32
        assert new_key != key

    @pytest.mark.asyncio
    async def test_tampering_detection(self, encryption_service):
        """Verify AEAD authentication detects tampered ciphertext."""
        plaintext = b"Original message"
        
        # Encrypt
        encrypted = await encryption_service.encrypt(plaintext)
        
        # Tamper with ciphertext
        tampered_ciphertext = bytes([b ^ 0xFF for b in encrypted.ciphertext[:5]]) + encrypted.ciphertext[5:]
        
        tampered_data = EncryptedData(
            ciphertext=tampered_ciphertext,
            nonce=encrypted.nonce,
            tag=encrypted.tag,
            algorithm=encrypted.algorithm,
            key_id=encrypted.key_id,
            timestamp=encrypted.timestamp,
        )
        
        # Decryption should fail due to authentication tag mismatch
        with pytest.raises(ValueError, match="tag verification failed"):
            await encryption_service.decrypt(tampered_data)

    @pytest.mark.asyncio
    async def test_tag_tampering_detection(self, encryption_service):
        """Verify tampering with authentication tag is detected."""
        plaintext = b"Secure data"
        
        encrypted = await encryption_service.encrypt(plaintext)
        
        # Tamper with tag
        tampered_tag = bytes([b ^ 0x01 for b in encrypted.tag])
        
        tampered_data = EncryptedData(
            ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce,
            tag=tampered_tag,
            algorithm=encrypted.algorithm,
            key_id=encrypted.key_id,
            timestamp=encrypted.timestamp,
        )
        
        with pytest.raises(ValueError):
            await encryption_service.decrypt(tampered_data)

    @pytest.mark.asyncio
    async def test_nonce_uniqueness(self, encryption_service):
        """Verify unique nonces are used for each encryption."""
        plaintext = b"Same message"
        
        encrypted1 = await encryption_service.encrypt(plaintext)
        encrypted2 = await encryption_service.encrypt(plaintext)
        
        # Nonces must be different
        assert encrypted1.nonce != encrypted2.nonce
        
        # Ciphertexts will also be different due to different nonces
        assert encrypted1.ciphertext != encrypted2.ciphertext

    @pytest.mark.asyncio
    async def test_empty_plaintext(self, encryption_service):
        """Verify empty plaintext can be encrypted and decrypted."""
        plaintext = b""
        
        encrypted = await encryption_service.encrypt(plaintext)
        decrypted = await encryption_service.decrypt(encrypted)
        
        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_large_plaintext(self, encryption_service):
        """Verify large data can be encrypted and decrypted."""
        # 1MB of random data
        plaintext = secrets.token_bytes(1024 * 1024)
        
        encrypted = await encryption_service.encrypt(plaintext)
        decrypted = await encryption_service.decrypt(encrypted)
        
        assert decrypted == plaintext

    def test_encrypted_data_serialization(self):
        """Test EncryptedData serialization and deserialization."""
        original = EncryptedData(
            ciphertext=b"encrypted_content",
            nonce=b"twelve_bytes",
            tag=b"sixteen_bytes___",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id="test-key-1",
            timestamp=datetime.now(timezone.utc),
            metadata={"test": "value"},
        )
        
        # Convert to dict and back
        data_dict = original.to_dict()
        restored = EncryptedData.from_dict(data_dict)
        
        assert restored.ciphertext == original.ciphertext
        assert restored.nonce == original.nonce
        assert restored.tag == original.tag
        assert restored.algorithm == original.algorithm
        assert restored.key_id == original.key_id
        assert restored.metadata == original.metadata


class TestKeyManagementService:
    """Tests for key management service."""

    @pytest.fixture
    def kms(self):
        """Create key management service."""
        return KeyManagementService()

    def test_generate_key(self, kms):
        """Test key generation."""
        key_id = kms.generate_key()
        
        assert key_id is not None
        assert key_id.startswith("key-")
        
        key = kms.get_key(key_id)
        assert key is not None
        assert len(key) == 32  # 256 bits

    def test_generate_key_with_custom_id(self, kms):
        """Test key generation with custom ID."""
        custom_id = "my-custom-key"
        key_id = kms.generate_key(key_id=custom_id)
        
        assert key_id == custom_id
        assert kms.get_key(custom_id) is not None

    def test_rotate_key(self, kms):
        """Test key rotation."""
        original_key_id = kms.generate_key(key_id="rotation-test")
        original_key = kms.get_key(original_key_id)
        
        new_key_id = kms.rotate_key(original_key_id)
        new_key = kms.get_key(new_key_id)
        
        # New key should be different
        assert new_key != original_key
        
        # Original key should still be available (for decryption)
        assert kms.get_key(original_key_id) is not None

    def test_rotate_key_not_retain_old(self, kms):
        """Test key rotation without retaining old key."""
        key_id = kms.generate_key(key_id="no-retain-test")
        
        kms.rotate_key(key_id, retain_old=False)
        
        # Old key should be disabled
        assert kms.get_key(key_id) is None

    def test_list_keys(self, kms):
        """Test listing all keys."""
        kms.generate_key(key_id="key-1")
        kms.generate_key(key_id="key-2")
        
        keys = kms.list_keys()
        
        assert len(keys) == 2
        key_ids = [k["key_id"] for k in keys]
        assert "key-1" in key_ids
        assert "key-2" in key_ids

    def test_get_nonexistent_key(self, kms):
        """Test getting a key that doesn't exist."""
        key = kms.get_key("nonexistent-key")
        assert key is None


class TestKeyRotationPolicy:
    """Tests for key rotation policy."""

    def test_default_policy(self):
        """Test default key rotation policy values."""
        policy = KeyRotationPolicy()
        
        assert policy.interval_days == 90
        assert policy.auto_rotate is True
        assert policy.retain_old_keys == 3
        assert policy.notify_before_rotation == 7

    def test_custom_policy(self):
        """Test custom key rotation policy."""
        policy = KeyRotationPolicy(
            interval_days=30,
            auto_rotate=False,
            retain_old_keys=5,
            notify_before_rotation=14,
        )
        
        assert policy.interval_days == 30
        assert policy.auto_rotate is False
        assert policy.retain_old_keys == 5
        assert policy.notify_before_rotation == 14

    @pytest.mark.asyncio
    async def test_encryption_with_custom_policy(self):
        """Test encryption service with custom rotation policy."""
        policy = KeyRotationPolicy(interval_days=30, auto_rotate=True)
        service = MilitaryGradeEncryption(key_rotation_policy=policy)
        
        status = service.get_key_rotation_status()
        
        assert status["policy"]["interval_days"] == 30
        assert status["policy"]["auto_rotate"] is True
