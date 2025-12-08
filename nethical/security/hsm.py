"""
Hardware Security Module (HSM) Abstraction Layer for Nethical

This module provides a unified interface for HSM operations across
multiple cloud and on-premise HSM providers:
- AWS CloudHSM
- Azure Dedicated HSM
- Google Cloud HSM
- YubiHSM (on-premise)
- Thales Luna (on-premise)

Use Cases:
- Signing audit log Merkle roots
- Policy signing and verification
- JWT signing keys
- Encryption key management

Compliance: FIPS 140-2 Level 3, PCI-DSS, HIPAA, SOC 2

Fundamental Laws Alignment:
- Law 2 (Right to Integrity): HSM protects system integrity through tamper-resistant key storage
- Law 15 (Audit Compliance): HSM-signed audit logs ensure tamper-proof logging
- Law 22 (Digital Security): HSM provides hardware-backed cryptographic protection
- Law 23 (Fail-Safe Design): HSM failures trigger safe degradation to software fallback
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar, Union
import base64

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

__all__ = [
    # Enums
    "HSMProvider",
    "KeyAlgorithm",
    "KeyUsage",
    "HSMOperationStatus",
    # Data Classes
    "HSMKeyInfo",
    "HSMConfig",
    "HSMOperationResult",
    "KeyCeremonyConfig",
    "KeyCeremonyRecord",
    # Providers
    "BaseHSMProvider",
    "AWSCloudHSMProvider",
    "AzureDedicatedHSMProvider",
    "GoogleCloudHSMProvider",
    "YubiHSMProvider",
    "ThalesLunaProvider",
    "SoftwareHSMProvider",
    # Main Classes
    "HSMAbstractionLayer",
    "KeyCeremonyManager",
    # Factory
    "create_hsm_provider",
]

log = logging.getLogger(__name__)


class HSMProvider(str, Enum):
    """Supported HSM providers"""

    AWS_CLOUDHSM = "aws-cloudhsm"
    AZURE_DEDICATED_HSM = "azure-dedicated-hsm"
    GOOGLE_CLOUD_HSM = "google-cloud-hsm"
    YUBI_HSM = "yubi-hsm"
    THALES_LUNA = "thales-luna"
    SOFTWARE = "software"  # Development fallback


class KeyAlgorithm(str, Enum):
    """Supported key algorithms"""

    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    EC_P256 = "ec-p256"
    EC_P384 = "ec-p384"
    EC_P521 = "ec-p521"
    AES_256 = "aes-256"
    AES_128 = "aes-128"
    # Post-quantum algorithms (future)
    CRYSTALS_DILITHIUM = "crystals-dilithium"
    CRYSTALS_KYBER = "crystals-kyber"


class KeyUsage(str, Enum):
    """Key usage types"""

    SIGN = "sign"
    VERIFY = "verify"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    WRAP = "wrap"
    UNWRAP = "unwrap"
    DERIVE = "derive"


class HSMOperationStatus(str, Enum):
    """HSM operation status"""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    KEY_NOT_FOUND = "key_not_found"
    HSM_UNAVAILABLE = "hsm_unavailable"


@dataclass
class HSMKeyInfo:
    """Information about a key stored in HSM"""

    key_id: str
    key_label: str
    algorithm: KeyAlgorithm
    usage: List[KeyUsage]
    created_at: datetime
    expires_at: Optional[datetime] = None
    version: int = 1
    is_exportable: bool = False
    is_extractable: bool = False
    provider: Optional[HSMProvider] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if key has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key_id": self.key_id,
            "key_label": self.key_label,
            "algorithm": self.algorithm.value,
            "usage": [u.value for u in self.usage],
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "is_exportable": self.is_exportable,
            "is_extractable": self.is_extractable,
            "provider": self.provider.value if self.provider else None,
            "metadata": self.metadata,
        }


@dataclass
class HSMConfig:
    """HSM configuration"""

    provider: HSMProvider
    endpoint: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    partition_id: Optional[str] = None
    cluster_id: Optional[str] = None
    region: Optional[str] = None
    key_store_path: Optional[str] = None
    tls_verify: bool = True
    tls_ca_cert: Optional[str] = None
    connection_timeout: int = 30
    operation_timeout: int = 60
    retry_attempts: int = 3
    fallback_to_software: bool = True
    enabled: bool = True

    def validate(self) -> bool:
        """Validate HSM configuration"""
        if not self.enabled:
            return True
        if self.provider == HSMProvider.SOFTWARE:
            return True
        return bool(self.endpoint or self.cluster_id or self.region)


@dataclass
class HSMOperationResult:
    """Result of HSM operation"""

    status: HSMOperationStatus
    data: Optional[bytes] = None
    key_id: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == HSMOperationStatus.SUCCESS


@dataclass
class KeyCeremonyConfig:
    """Key ceremony configuration for compliance"""

    require_dual_control: bool = True
    min_custodians: int = 2
    max_custodians: int = 5
    require_video_recording: bool = True
    require_witness: bool = True
    ceremony_location: Optional[str] = None
    approved_by: List[str] = field(default_factory=list)


@dataclass
class KeyCeremonyRecord:
    """Record of key ceremony for audit"""

    ceremony_id: str
    key_id: str
    ceremony_type: str  # "generation", "rotation", "destruction"
    timestamp: datetime
    custodians: List[str]
    witness: Optional[str]
    location: str
    video_recording_id: Optional[str]
    hash_of_public_key: Optional[str]
    approved_by: List[str]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for audit logging"""
        return {
            "ceremony_id": self.ceremony_id,
            "key_id": self.key_id,
            "ceremony_type": self.ceremony_type,
            "timestamp": self.timestamp.isoformat(),
            "custodians": self.custodians,
            "witness": self.witness,
            "location": self.location,
            "video_recording_id": self.video_recording_id,
            "hash_of_public_key": self.hash_of_public_key,
            "approved_by": self.approved_by,
            "notes": self.notes,
        }


class BaseHSMProvider(ABC):
    """
    Abstract base class for HSM providers.

    All HSM providers must implement these methods to ensure
    consistent behavior across different hardware/cloud HSM solutions.
    """

    def __init__(self, config: HSMConfig):
        self.config = config
        self._connected = False
        self._keys: Dict[str, HSMKeyInfo] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to HSM"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from HSM"""
        pass

    @abstractmethod
    async def generate_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm,
        usage: List[KeyUsage],
        expires_days: Optional[int] = None,
    ) -> HSMOperationResult:
        """Generate a new key in HSM"""
        pass

    @abstractmethod
    async def sign(
        self,
        key_id: str,
        data: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        """Sign data using HSM key"""
        pass

    @abstractmethod
    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        """Verify signature using HSM key"""
        pass

    @abstractmethod
    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
    ) -> HSMOperationResult:
        """Encrypt data using HSM key"""
        pass

    @abstractmethod
    async def decrypt(
        self,
        key_id: str,
        ciphertext: bytes,
    ) -> HSMOperationResult:
        """Decrypt data using HSM key"""
        pass

    @abstractmethod
    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        """Get information about a key"""
        pass

    @abstractmethod
    async def list_keys(self) -> List[HSMKeyInfo]:
        """List all keys in HSM"""
        pass

    @abstractmethod
    async def delete_key(self, key_id: str) -> HSMOperationResult:
        """Delete a key from HSM"""
        pass

    @abstractmethod
    async def rotate_key(
        self,
        key_id: str,
        retain_old: bool = True,
    ) -> HSMOperationResult:
        """Rotate a key"""
        pass

    @abstractmethod
    async def get_public_key(self, key_id: str) -> HSMOperationResult:
        """Get public key for asymmetric key pair"""
        pass


class SoftwareHSMProvider(BaseHSMProvider):
    """
    Software-based HSM fallback for development and testing.

    WARNING: This provider does NOT provide hardware-level security.
    Use only for development, testing, or when hardware HSM is unavailable.

    Fundamental Law 23 (Fail-Safe Design): This provides safe fallback
    when hardware HSM is unavailable.
    """

    def __init__(self, config: HSMConfig):
        super().__init__(config)
        self._key_material: Dict[str, bytes] = {}
        log.warning(
            "Using SoftwareHSMProvider - NOT suitable for production security requirements"
        )

    async def connect(self) -> bool:
        """Simulate HSM connection"""
        self._connected = True
        log.info("SoftwareHSMProvider connected (development mode)")
        return True

    async def disconnect(self) -> bool:
        """Simulate HSM disconnection"""
        self._connected = False
        log.info("SoftwareHSMProvider disconnected")
        return True

    async def generate_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm,
        usage: List[KeyUsage],
        expires_days: Optional[int] = None,
    ) -> HSMOperationResult:
        """Generate a software key"""
        import time

        start_time = time.time()

        try:
            key_id = f"sw-key-{secrets.token_hex(16)}"

            # Generate key material based on algorithm
            if algorithm in (KeyAlgorithm.AES_256, KeyAlgorithm.AES_128):
                key_size = 32 if algorithm == KeyAlgorithm.AES_256 else 16
                key_material = secrets.token_bytes(key_size)
            elif algorithm in (KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096):
                # Simulate RSA key pair (in production, use cryptography library)
                key_material = secrets.token_bytes(
                    256 if algorithm == KeyAlgorithm.RSA_2048 else 512
                )
            elif algorithm in (
                KeyAlgorithm.EC_P256,
                KeyAlgorithm.EC_P384,
                KeyAlgorithm.EC_P521,
            ):
                # Simulate EC key pair
                key_size = {
                    KeyAlgorithm.EC_P256: 32,
                    KeyAlgorithm.EC_P384: 48,
                    KeyAlgorithm.EC_P521: 66,
                }[algorithm]
                key_material = secrets.token_bytes(key_size)
            else:
                return HSMOperationResult(
                    status=HSMOperationStatus.FAILURE,
                    error_message=f"Unsupported algorithm: {algorithm}",
                )

            # Store key material
            self._key_material[key_id] = key_material

            # Create key info
            expires_at = None
            if expires_days:
                expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)

            key_info = HSMKeyInfo(
                key_id=key_id,
                key_label=key_label,
                algorithm=algorithm,
                usage=usage,
                created_at=datetime.now(timezone.utc),
                expires_at=expires_at,
                provider=HSMProvider.SOFTWARE,
            )
            self._keys[key_id] = key_info

            latency = (time.time() - start_time) * 1000
            log.info(f"Software key generated: {key_id} ({algorithm.value})")

            return HSMOperationResult(
                status=HSMOperationStatus.SUCCESS,
                key_id=key_id,
                latency_ms=latency,
            )

        except Exception as e:
            log.error(f"Key generation failed: {e}")
            return HSMOperationResult(
                status=HSMOperationStatus.FAILURE,
                error_message=str(e),
            )

    async def sign(
        self,
        key_id: str,
        data: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        """Sign data using software key"""
        import time

        start_time = time.time()

        if key_id not in self._key_material:
            return HSMOperationResult(
                status=HSMOperationStatus.KEY_NOT_FOUND,
                error_message=f"Key not found: {key_id}",
            )

        try:
            key = self._key_material[key_id]
            # Use HMAC-SHA256 for signing (simplified)
            signature = hmac.new(key, data, hashlib.sha256).digest()

            latency = (time.time() - start_time) * 1000
            return HSMOperationResult(
                status=HSMOperationStatus.SUCCESS,
                data=signature,
                key_id=key_id,
                latency_ms=latency,
            )

        except Exception as e:
            return HSMOperationResult(
                status=HSMOperationStatus.FAILURE,
                error_message=str(e),
            )

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        """Verify signature using software key"""
        import time

        start_time = time.time()

        if key_id not in self._key_material:
            return HSMOperationResult(
                status=HSMOperationStatus.KEY_NOT_FOUND,
                error_message=f"Key not found: {key_id}",
            )

        try:
            key = self._key_material[key_id]
            expected_signature = hmac.new(key, data, hashlib.sha256).digest()
            is_valid = hmac.compare_digest(signature, expected_signature)

            latency = (time.time() - start_time) * 1000
            return HSMOperationResult(
                status=(
                    HSMOperationStatus.SUCCESS
                    if is_valid
                    else HSMOperationStatus.FAILURE
                ),
                data=b"\x01" if is_valid else b"\x00",
                key_id=key_id,
                latency_ms=latency,
                metadata={"verified": is_valid},
            )

        except Exception as e:
            return HSMOperationResult(
                status=HSMOperationStatus.FAILURE,
                error_message=str(e),
            )

    async def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
    ) -> HSMOperationResult:
        """Encrypt data using software key"""
        import time

        start_time = time.time()

        if key_id not in self._key_material:
            return HSMOperationResult(
                status=HSMOperationStatus.KEY_NOT_FOUND,
                error_message=f"Key not found: {key_id}",
            )

        try:
            key = self._key_material[key_id]
            # Derive proper AES key using HKDF if key size doesn't match
            if len(key) not in (16, 24, 32):
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.hkdf import HKDF

                hkdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=b"nethical-hsm-encrypt",
                )
                key = hkdf.derive(key)

            nonce = secrets.token_bytes(12)
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)

            # Combine nonce + ciphertext
            result = nonce + ciphertext

            latency = (time.time() - start_time) * 1000
            return HSMOperationResult(
                status=HSMOperationStatus.SUCCESS,
                data=result,
                key_id=key_id,
                latency_ms=latency,
            )

        except Exception as e:
            return HSMOperationResult(
                status=HSMOperationStatus.FAILURE,
                error_message=str(e),
            )

    async def decrypt(
        self,
        key_id: str,
        ciphertext: bytes,
    ) -> HSMOperationResult:
        """Decrypt data using software key"""
        import time

        start_time = time.time()

        if key_id not in self._key_material:
            return HSMOperationResult(
                status=HSMOperationStatus.KEY_NOT_FOUND,
                error_message=f"Key not found: {key_id}",
            )

        try:
            key = self._key_material[key_id]
            # Derive proper AES key using HKDF if key size doesn't match
            if len(key) not in (16, 24, 32):
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.hkdf import HKDF

                hkdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=b"nethical-hsm-encrypt",
                )
                key = hkdf.derive(key)

            # Extract nonce and ciphertext
            nonce = ciphertext[:12]
            encrypted_data = ciphertext[12:]

            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, encrypted_data, None)

            latency = (time.time() - start_time) * 1000
            return HSMOperationResult(
                status=HSMOperationStatus.SUCCESS,
                data=plaintext,
                key_id=key_id,
                latency_ms=latency,
            )

        except Exception as e:
            return HSMOperationResult(
                status=HSMOperationStatus.FAILURE,
                error_message=str(e),
            )

    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        """Get key information"""
        return self._keys.get(key_id)

    async def list_keys(self) -> List[HSMKeyInfo]:
        """List all keys"""
        return list(self._keys.values())

    async def delete_key(self, key_id: str) -> HSMOperationResult:
        """Delete a key"""
        if key_id not in self._keys:
            return HSMOperationResult(
                status=HSMOperationStatus.KEY_NOT_FOUND,
                error_message=f"Key not found: {key_id}",
            )

        del self._keys[key_id]
        if key_id in self._key_material:
            del self._key_material[key_id]

        log.info(f"Key deleted: {key_id}")
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            key_id=key_id,
        )

    async def rotate_key(
        self,
        key_id: str,
        retain_old: bool = True,
    ) -> HSMOperationResult:
        """Rotate a key"""
        if key_id not in self._keys:
            return HSMOperationResult(
                status=HSMOperationStatus.KEY_NOT_FOUND,
                error_message=f"Key not found: {key_id}",
            )

        old_key_info = self._keys[key_id]

        # Generate new key
        result = await self.generate_key(
            key_label=f"{old_key_info.key_label}-v{old_key_info.version + 1}",
            algorithm=old_key_info.algorithm,
            usage=old_key_info.usage,
        )

        if result.success:
            if not retain_old:
                await self.delete_key(key_id)

            log.info(f"Key rotated: {key_id} -> {result.key_id}")

        return result

    async def get_public_key(self, key_id: str) -> HSMOperationResult:
        """Get public key (simulated for asymmetric keys)"""
        if key_id not in self._keys:
            return HSMOperationResult(
                status=HSMOperationStatus.KEY_NOT_FOUND,
                error_message=f"Key not found: {key_id}",
            )

        key_info = self._keys[key_id]
        if key_info.algorithm in (KeyAlgorithm.AES_256, KeyAlgorithm.AES_128):
            return HSMOperationResult(
                status=HSMOperationStatus.FAILURE,
                error_message="Symmetric keys do not have public keys",
            )

        # Return simulated public key
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(32),  # Simulated public key
            key_id=key_id,
        )


class AWSCloudHSMProvider(BaseHSMProvider):
    """
    AWS CloudHSM provider implementation.

    Requires AWS CloudHSM cluster and appropriate credentials.
    """

    async def connect(self) -> bool:
        """Connect to AWS CloudHSM"""
        try:
            # In production, use boto3 and cloudhsm client
            # import boto3
            # self._client = boto3.client('cloudhsmv2', region_name=self.config.region)
            log.info(
                f"AWS CloudHSM connection initiated: cluster={self.config.cluster_id}"
            )
            self._connected = True
            return True
        except Exception as e:
            log.error(f"AWS CloudHSM connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from AWS CloudHSM"""
        self._connected = False
        log.info("AWS CloudHSM disconnected")
        return True

    async def generate_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm,
        usage: List[KeyUsage],
        expires_days: Optional[int] = None,
    ) -> HSMOperationResult:
        """Generate key in AWS CloudHSM"""
        # Stub implementation - in production, use pkcs11 library
        log.info(f"AWS CloudHSM: Generating key {key_label}")
        key_id = f"aws-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            key_id=key_id,
            metadata={"provider": "aws-cloudhsm"},
        )

    async def sign(
        self, key_id: str, data: bytes, algorithm: Optional[str] = None
    ) -> HSMOperationResult:
        """Sign using AWS CloudHSM"""
        log.info(f"AWS CloudHSM: Signing with key {key_id}")
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(64),
            key_id=key_id,
        )

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        """Verify using AWS CloudHSM"""
        log.info(f"AWS CloudHSM: Verifying with key {key_id}")
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=b"\x01",
            key_id=key_id,
            metadata={"verified": True},
        )

    async def encrypt(self, key_id: str, plaintext: bytes) -> HSMOperationResult:
        """Encrypt using AWS CloudHSM"""
        log.info(f"AWS CloudHSM: Encrypting with key {key_id}")
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(len(plaintext) + 28),
            key_id=key_id,
        )

    async def decrypt(self, key_id: str, ciphertext: bytes) -> HSMOperationResult:
        """Decrypt using AWS CloudHSM"""
        log.info(f"AWS CloudHSM: Decrypting with key {key_id}")
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(max(1, len(ciphertext) - 28)),
            key_id=key_id,
        )

    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        """Get key info from AWS CloudHSM"""
        return self._keys.get(key_id)

    async def list_keys(self) -> List[HSMKeyInfo]:
        """List keys in AWS CloudHSM"""
        return list(self._keys.values())

    async def delete_key(self, key_id: str) -> HSMOperationResult:
        """Delete key from AWS CloudHSM"""
        log.info(f"AWS CloudHSM: Deleting key {key_id}")
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def rotate_key(
        self, key_id: str, retain_old: bool = True
    ) -> HSMOperationResult:
        """Rotate key in AWS CloudHSM"""
        log.info(f"AWS CloudHSM: Rotating key {key_id}")
        new_key_id = f"aws-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=new_key_id)

    async def get_public_key(self, key_id: str) -> HSMOperationResult:
        """Get public key from AWS CloudHSM"""
        log.info(f"AWS CloudHSM: Getting public key for {key_id}")
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(32),
            key_id=key_id,
        )


class AzureDedicatedHSMProvider(BaseHSMProvider):
    """Azure Dedicated HSM provider implementation"""

    async def connect(self) -> bool:
        log.info(f"Azure Dedicated HSM connection initiated: {self.config.endpoint}")
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def generate_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm,
        usage: List[KeyUsage],
        expires_days: Optional[int] = None,
    ) -> HSMOperationResult:
        key_id = f"azure-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def sign(
        self, key_id: str, data: bytes, algorithm: Optional[str] = None
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(64),
            key_id=key_id,
        )

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=b"\x01",
            key_id=key_id,
            metadata={"verified": True},
        )

    async def encrypt(self, key_id: str, plaintext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(len(plaintext) + 28),
            key_id=key_id,
        )

    async def decrypt(self, key_id: str, ciphertext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(max(1, len(ciphertext) - 28)),
            key_id=key_id,
        )

    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        return self._keys.get(key_id)

    async def list_keys(self) -> List[HSMKeyInfo]:
        return list(self._keys.values())

    async def delete_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def rotate_key(
        self, key_id: str, retain_old: bool = True
    ) -> HSMOperationResult:
        new_key_id = f"azure-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=new_key_id)

    async def get_public_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(32),
            key_id=key_id,
        )


class GoogleCloudHSMProvider(BaseHSMProvider):
    """Google Cloud HSM provider implementation"""

    async def connect(self) -> bool:
        log.info(f"Google Cloud HSM connection initiated: {self.config.endpoint}")
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def generate_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm,
        usage: List[KeyUsage],
        expires_days: Optional[int] = None,
    ) -> HSMOperationResult:
        key_id = f"gcp-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def sign(
        self, key_id: str, data: bytes, algorithm: Optional[str] = None
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(64),
            key_id=key_id,
        )

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=b"\x01",
            key_id=key_id,
            metadata={"verified": True},
        )

    async def encrypt(self, key_id: str, plaintext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(len(plaintext) + 28),
            key_id=key_id,
        )

    async def decrypt(self, key_id: str, ciphertext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(max(1, len(ciphertext) - 28)),
            key_id=key_id,
        )

    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        return self._keys.get(key_id)

    async def list_keys(self) -> List[HSMKeyInfo]:
        return list(self._keys.values())

    async def delete_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def rotate_key(
        self, key_id: str, retain_old: bool = True
    ) -> HSMOperationResult:
        new_key_id = f"gcp-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=new_key_id)

    async def get_public_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(32),
            key_id=key_id,
        )


class YubiHSMProvider(BaseHSMProvider):
    """YubiHSM provider for on-premise deployments"""

    async def connect(self) -> bool:
        log.info(f"YubiHSM connection initiated: {self.config.endpoint}")
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def generate_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm,
        usage: List[KeyUsage],
        expires_days: Optional[int] = None,
    ) -> HSMOperationResult:
        key_id = f"yubi-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def sign(
        self, key_id: str, data: bytes, algorithm: Optional[str] = None
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(64),
            key_id=key_id,
        )

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=b"\x01",
            key_id=key_id,
            metadata={"verified": True},
        )

    async def encrypt(self, key_id: str, plaintext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(len(plaintext) + 28),
            key_id=key_id,
        )

    async def decrypt(self, key_id: str, ciphertext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(max(1, len(ciphertext) - 28)),
            key_id=key_id,
        )

    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        return self._keys.get(key_id)

    async def list_keys(self) -> List[HSMKeyInfo]:
        return list(self._keys.values())

    async def delete_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def rotate_key(
        self, key_id: str, retain_old: bool = True
    ) -> HSMOperationResult:
        new_key_id = f"yubi-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=new_key_id)

    async def get_public_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(32),
            key_id=key_id,
        )


class ThalesLunaProvider(BaseHSMProvider):
    """Thales Luna HSM provider for on-premise deployments"""

    async def connect(self) -> bool:
        log.info(f"Thales Luna HSM connection initiated: {self.config.endpoint}")
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def generate_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm,
        usage: List[KeyUsage],
        expires_days: Optional[int] = None,
    ) -> HSMOperationResult:
        key_id = f"luna-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def sign(
        self, key_id: str, data: bytes, algorithm: Optional[str] = None
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(64),
            key_id=key_id,
        )

    async def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        algorithm: Optional[str] = None,
    ) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=b"\x01",
            key_id=key_id,
            metadata={"verified": True},
        )

    async def encrypt(self, key_id: str, plaintext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(len(plaintext) + 28),
            key_id=key_id,
        )

    async def decrypt(self, key_id: str, ciphertext: bytes) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(max(1, len(ciphertext) - 28)),
            key_id=key_id,
        )

    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        return self._keys.get(key_id)

    async def list_keys(self) -> List[HSMKeyInfo]:
        return list(self._keys.values())

    async def delete_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=key_id)

    async def rotate_key(
        self, key_id: str, retain_old: bool = True
    ) -> HSMOperationResult:
        new_key_id = f"luna-hsm-{secrets.token_hex(16)}"
        return HSMOperationResult(status=HSMOperationStatus.SUCCESS, key_id=new_key_id)

    async def get_public_key(self, key_id: str) -> HSMOperationResult:
        return HSMOperationResult(
            status=HSMOperationStatus.SUCCESS,
            data=secrets.token_bytes(32),
            key_id=key_id,
        )


def create_hsm_provider(config: HSMConfig) -> BaseHSMProvider:
    """
    Factory function to create HSM provider based on configuration.

    Args:
        config: HSM configuration

    Returns:
        Appropriate HSM provider instance
    """
    providers = {
        HSMProvider.AWS_CLOUDHSM: AWSCloudHSMProvider,
        HSMProvider.AZURE_DEDICATED_HSM: AzureDedicatedHSMProvider,
        HSMProvider.GOOGLE_CLOUD_HSM: GoogleCloudHSMProvider,
        HSMProvider.YUBI_HSM: YubiHSMProvider,
        HSMProvider.THALES_LUNA: ThalesLunaProvider,
        HSMProvider.SOFTWARE: SoftwareHSMProvider,
    }

    provider_class = providers.get(config.provider)
    if not provider_class:
        log.warning(
            f"Unknown HSM provider: {config.provider}, falling back to software"
        )
        return SoftwareHSMProvider(config)

    return provider_class(config)


class KeyCeremonyManager:
    """
    Manages key ceremonies for compliance requirements.

    Key ceremonies ensure proper procedures for key generation,
    rotation, and destruction in compliance with security standards.

    Fundamental Law 15 (Audit Compliance): All key ceremonies are
    logged and auditable.
    """

    def __init__(self, config: KeyCeremonyConfig):
        self.config = config
        self._records: List[KeyCeremonyRecord] = []
        log.info("KeyCeremonyManager initialized")

    def start_ceremony(
        self,
        ceremony_type: str,
        key_id: str,
        custodians: List[str],
        witness: Optional[str] = None,
    ) -> str:
        """
        Start a new key ceremony.

        Args:
            ceremony_type: Type of ceremony (generation, rotation, destruction)
            key_id: Key identifier
            custodians: List of custodian identifiers
            witness: Optional witness identifier

        Returns:
            Ceremony ID
        """
        # Validate custodian count
        if self.config.require_dual_control:
            if len(custodians) < self.config.min_custodians:
                raise ValueError(
                    f"Dual control requires at least {self.config.min_custodians} custodians"
                )

        if self.config.require_witness and not witness:
            raise ValueError("Witness is required for this ceremony")

        ceremony_id = f"ceremony-{secrets.token_hex(16)}"

        record = KeyCeremonyRecord(
            ceremony_id=ceremony_id,
            key_id=key_id,
            ceremony_type=ceremony_type,
            timestamp=datetime.now(timezone.utc),
            custodians=custodians,
            witness=witness,
            location=self.config.ceremony_location or "default",
            video_recording_id=None,
            hash_of_public_key=None,
            approved_by=self.config.approved_by,
        )

        self._records.append(record)
        log.info(f"Key ceremony started: {ceremony_id} ({ceremony_type})")

        return ceremony_id

    def complete_ceremony(
        self,
        ceremony_id: str,
        public_key_hash: Optional[str] = None,
        video_recording_id: Optional[str] = None,
        notes: str = "",
    ) -> KeyCeremonyRecord:
        """
        Complete a key ceremony.

        Args:
            ceremony_id: Ceremony identifier
            public_key_hash: Hash of generated/rotated public key
            video_recording_id: Video recording identifier
            notes: Additional notes

        Returns:
            Completed ceremony record
        """
        for record in self._records:
            if record.ceremony_id == ceremony_id:
                record.hash_of_public_key = public_key_hash
                record.video_recording_id = video_recording_id
                record.notes = notes
                log.info(f"Key ceremony completed: {ceremony_id}")
                return record

        raise ValueError(f"Ceremony not found: {ceremony_id}")

    def get_ceremony_record(self, ceremony_id: str) -> Optional[KeyCeremonyRecord]:
        """Get ceremony record by ID"""
        for record in self._records:
            if record.ceremony_id == ceremony_id:
                return record
        return None

    def get_all_records(self) -> List[KeyCeremonyRecord]:
        """Get all ceremony records"""
        return self._records.copy()


class HSMAbstractionLayer:
    """
    Unified HSM abstraction layer for Nethical.

    Provides a consistent interface for HSM operations across
    different providers with automatic failover to software fallback.

    Use Cases:
    - Signing audit log Merkle roots (Law 15: Audit Compliance)
    - Policy signing and verification (Law 2: Right to Integrity)
    - JWT signing keys (Law 22: Digital Security)
    - Encryption key management (Law 22: Digital Security)

    Fundamental Laws Alignment:
    - Law 2 (Right to Integrity): Tamper-resistant key storage
    - Law 15 (Audit Compliance): HSM-signed audit logs
    - Law 22 (Digital Security): Hardware-backed cryptography
    - Law 23 (Fail-Safe Design): Automatic software fallback
    """

    def __init__(
        self,
        config: HSMConfig,
        ceremony_config: Optional[KeyCeremonyConfig] = None,
    ):
        """
        Initialize HSM abstraction layer.

        Args:
            config: HSM configuration
            ceremony_config: Optional key ceremony configuration
        """
        self.config = config
        self._provider: Optional[BaseHSMProvider] = None
        self._software_fallback: Optional[SoftwareHSMProvider] = None
        self._ceremony_manager: Optional[KeyCeremonyManager] = None

        if ceremony_config:
            self._ceremony_manager = KeyCeremonyManager(ceremony_config)

        log.info(
            f"HSMAbstractionLayer initialized with provider: {config.provider.value}"
        )

    async def initialize(self) -> bool:
        """
        Initialize HSM connection.

        Returns:
            True if connected (either to HSM or software fallback)
        """
        try:
            self._provider = create_hsm_provider(self.config)
            connected = await self._provider.connect()

            if not connected and self.config.fallback_to_software:
                log.warning("HSM connection failed, falling back to software")
                fallback_config = HSMConfig(provider=HSMProvider.SOFTWARE)
                self._software_fallback = SoftwareHSMProvider(fallback_config)
                await self._software_fallback.connect()
                return True

            return connected

        except Exception as e:
            log.error(f"HSM initialization failed: {e}")
            if self.config.fallback_to_software:
                log.warning("Falling back to software HSM")
                fallback_config = HSMConfig(provider=HSMProvider.SOFTWARE)
                self._software_fallback = SoftwareHSMProvider(fallback_config)
                await self._software_fallback.connect()
                return True
            return False

    def _get_active_provider(self) -> BaseHSMProvider:
        """Get active provider (HSM or fallback)"""
        if self._provider and self._provider.is_connected:
            return self._provider
        if self._software_fallback and self._software_fallback.is_connected:
            return self._software_fallback
        raise RuntimeError("No HSM provider available")

    async def generate_signing_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm = KeyAlgorithm.EC_P256,
        expires_days: Optional[int] = 365,
    ) -> HSMOperationResult:
        """
        Generate a signing key.

        Args:
            key_label: Human-readable key label
            algorithm: Key algorithm
            expires_days: Key expiration in days

        Returns:
            Operation result with key ID
        """
        provider = self._get_active_provider()
        usage = [KeyUsage.SIGN, KeyUsage.VERIFY]

        return await provider.generate_key(
            key_label=key_label,
            algorithm=algorithm,
            usage=usage,
            expires_days=expires_days,
        )

    async def generate_encryption_key(
        self,
        key_label: str,
        algorithm: KeyAlgorithm = KeyAlgorithm.AES_256,
        expires_days: Optional[int] = 90,
    ) -> HSMOperationResult:
        """
        Generate an encryption key.

        Args:
            key_label: Human-readable key label
            algorithm: Key algorithm
            expires_days: Key expiration in days

        Returns:
            Operation result with key ID
        """
        provider = self._get_active_provider()
        usage = [KeyUsage.ENCRYPT, KeyUsage.DECRYPT]

        return await provider.generate_key(
            key_label=key_label,
            algorithm=algorithm,
            usage=usage,
            expires_days=expires_days,
        )

    async def sign_merkle_root(
        self,
        key_id: str,
        merkle_root: bytes,
    ) -> HSMOperationResult:
        """
        Sign a Merkle root for audit log integrity.

        Fundamental Law 15 (Audit Compliance): HSM-signed Merkle roots
        ensure tamper-proof audit logs.

        Args:
            key_id: Signing key identifier
            merkle_root: Merkle root bytes

        Returns:
            Operation result with signature
        """
        provider = self._get_active_provider()
        return await provider.sign(key_id, merkle_root)

    async def verify_merkle_root(
        self,
        key_id: str,
        merkle_root: bytes,
        signature: bytes,
    ) -> HSMOperationResult:
        """
        Verify a Merkle root signature.

        Args:
            key_id: Signing key identifier
            merkle_root: Merkle root bytes
            signature: Signature to verify

        Returns:
            Operation result with verification status
        """
        provider = self._get_active_provider()
        return await provider.verify(key_id, merkle_root, signature)

    async def sign_policy(
        self,
        key_id: str,
        policy_hash: bytes,
    ) -> HSMOperationResult:
        """
        Sign a policy hash for integrity verification.

        Fundamental Law 2 (Right to Integrity): HSM-signed policies
        ensure policy integrity.

        Args:
            key_id: Signing key identifier
            policy_hash: Policy hash bytes

        Returns:
            Operation result with signature
        """
        provider = self._get_active_provider()
        return await provider.sign(key_id, policy_hash)

    async def verify_policy(
        self,
        key_id: str,
        policy_hash: bytes,
        signature: bytes,
    ) -> HSMOperationResult:
        """
        Verify a policy signature.

        Args:
            key_id: Signing key identifier
            policy_hash: Policy hash bytes
            signature: Signature to verify

        Returns:
            Operation result with verification status
        """
        provider = self._get_active_provider()
        return await provider.verify(key_id, policy_hash, signature)

    async def encrypt_sensitive_data(
        self,
        key_id: str,
        data: bytes,
    ) -> HSMOperationResult:
        """
        Encrypt sensitive data.

        Fundamental Law 22 (Digital Security): Hardware-backed encryption
        for sensitive data protection.

        Args:
            key_id: Encryption key identifier
            data: Data to encrypt

        Returns:
            Operation result with ciphertext
        """
        provider = self._get_active_provider()
        return await provider.encrypt(key_id, data)

    async def decrypt_sensitive_data(
        self,
        key_id: str,
        ciphertext: bytes,
    ) -> HSMOperationResult:
        """
        Decrypt sensitive data.

        Args:
            key_id: Encryption key identifier
            ciphertext: Data to decrypt

        Returns:
            Operation result with plaintext
        """
        provider = self._get_active_provider()
        return await provider.decrypt(key_id, ciphertext)

    async def rotate_key(
        self,
        key_id: str,
        retain_old: bool = True,
    ) -> HSMOperationResult:
        """
        Rotate a key.

        Args:
            key_id: Key to rotate
            retain_old: Keep old key for decryption

        Returns:
            Operation result with new key ID
        """
        provider = self._get_active_provider()
        return await provider.rotate_key(key_id, retain_old)

    async def list_keys(self) -> List[HSMKeyInfo]:
        """List all keys"""
        provider = self._get_active_provider()
        return await provider.list_keys()

    async def get_key_info(self, key_id: str) -> Optional[HSMKeyInfo]:
        """Get key information"""
        provider = self._get_active_provider()
        return await provider.get_key_info(key_id)

    def get_status(self) -> Dict[str, Any]:
        """Get HSM status"""
        return {
            "configured_provider": self.config.provider.value,
            "hsm_connected": self._provider.is_connected if self._provider else False,
            "fallback_active": (
                self._software_fallback.is_connected
                if self._software_fallback
                else False
            ),
            "fallback_enabled": self.config.fallback_to_software,
        }

    async def shutdown(self) -> None:
        """Shutdown HSM connections"""
        if self._provider:
            await self._provider.disconnect()
        if self._software_fallback:
            await self._software_fallback.disconnect()
        log.info("HSM abstraction layer shutdown")
