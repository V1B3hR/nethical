"""
Post-Quantum Cryptography Readiness for Nethical

This module provides post-quantum cryptography (PQC) preparation
and hybrid cryptographic solutions for future quantum-resistant security.

Features:
    - PQC algorithm abstractions
    - Hybrid classical/PQC key exchange
    - Migration path utilities
    - Algorithm agility framework
    - Compliance with NIST PQC standards

Standards Alignment:
    - NIST FIPS 203 (ML-KEM, formerly Kyber)
    - NIST FIPS 204 (ML-DSA, formerly Dilithium)
    - NIST FIPS 205 (SLH-DSA, formerly SPHINCS+)

Fundamental Laws Alignment:
    - Law 22 (Digital Security): Quantum-resistant protection
    - Law 2 (Right to Integrity): Long-term data protection

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "PQCAlgorithm",
    "SecurityLevel",
    "KeyPair",
    "PQCKeyExchange",
    "PQCSignature",
    "HybridKeyExchange",
    "PQCReadinessAssessment",
    "CryptoAgility",
    "assess_pqc_readiness",
]

log = logging.getLogger(__name__)


class PQCAlgorithm(str, Enum):
    """NIST-standardized post-quantum algorithms."""

    # Key Encapsulation Mechanisms (KEM)
    ML_KEM_512 = "ML-KEM-512"  # FIPS 203 - Category 1
    ML_KEM_768 = "ML-KEM-768"  # FIPS 203 - Category 3
    ML_KEM_1024 = "ML-KEM-1024"  # FIPS 203 - Category 5

    # Digital Signatures
    ML_DSA_44 = "ML-DSA-44"  # FIPS 204 - Category 2
    ML_DSA_65 = "ML-DSA-65"  # FIPS 204 - Category 3
    ML_DSA_87 = "ML-DSA-87"  # FIPS 204 - Category 5

    # Hash-based Signatures (Stateless)
    SLH_DSA_SHA2_128S = "SLH-DSA-SHA2-128s"  # FIPS 205
    SLH_DSA_SHA2_256S = "SLH-DSA-SHA2-256s"  # FIPS 205

    @property
    def is_kem(self) -> bool:
        """Check if algorithm is a KEM."""
        return self.value.startswith("ML-KEM")

    @property
    def is_signature(self) -> bool:
        """Check if algorithm is a signature scheme."""
        return self.value.startswith("ML-DSA") or self.value.startswith("SLH-DSA")


class SecurityLevel(int, Enum):
    """NIST security levels (bits of quantum security)."""

    LEVEL_1 = 1  # ~AES-128 equivalent
    LEVEL_2 = 2  # ~SHA-256 equivalent
    LEVEL_3 = 3  # ~AES-192 equivalent
    LEVEL_4 = 4  # ~SHA-384 equivalent
    LEVEL_5 = 5  # ~AES-256 equivalent


@dataclass
class KeyPair:
    """Cryptographic key pair.

    Attributes:
        algorithm: Algorithm used to generate keys
        public_key: Public key bytes
        private_key: Private key bytes (optional, for own keys)
        key_id: Unique key identifier
        created_at: Key creation timestamp
        expires_at: Key expiration timestamp
        metadata: Additional key information
    """

    algorithm: Union[PQCAlgorithm, str]
    public_key: bytes
    private_key: Optional[bytes] = None
    key_id: str = field(default_factory=lambda: secrets.token_hex(16))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_pqc(self) -> bool:
        """Check if key uses PQC algorithm."""
        return isinstance(self.algorithm, PQCAlgorithm)

    @property
    def public_key_fingerprint(self) -> str:
        """Get fingerprint of public key."""
        return hashlib.sha256(self.public_key).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes private key)."""
        alg = (
            self.algorithm.value
            if isinstance(self.algorithm, PQCAlgorithm)
            else self.algorithm
        )
        return {
            "algorithm": alg,
            "public_key_fingerprint": self.public_key_fingerprint,
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_pqc": self.is_pqc,
            "metadata": self.metadata,
        }


class PQCKeyExchange(ABC):
    """Abstract base class for PQC key exchange (KEM)."""

    @abstractmethod
    def generate_keypair(self) -> KeyPair:
        """Generate a new key pair."""
        pass

    @abstractmethod
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate a shared secret.

        Args:
            public_key: Recipient's public key

        Returns:
            Tuple of (ciphertext, shared_secret)
        """
        pass

    @abstractmethod
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate a shared secret.

        Args:
            private_key: Recipient's private key
            ciphertext: Encapsulated ciphertext

        Returns:
            Shared secret
        """
        pass


class PQCSignature(ABC):
    """Abstract base class for PQC digital signatures."""

    @abstractmethod
    def generate_keypair(self) -> KeyPair:
        """Generate a new signing key pair."""
        pass

    @abstractmethod
    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign a message.

        Args:
            private_key: Signer's private key
            message: Message to sign

        Returns:
            Digital signature
        """
        pass

    @abstractmethod
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify a signature.

        Args:
            public_key: Signer's public key
            message: Original message
            signature: Signature to verify

        Returns:
            True if signature is valid
        """
        pass


class SimulatedMLKEM(PQCKeyExchange):
    """Simulated ML-KEM for development and testing.

    WARNING: This is a simulation only. Use a real implementation
    (e.g., liboqs, pqcrypto) in production.
    """

    def __init__(self, algorithm: PQCAlgorithm = PQCAlgorithm.ML_KEM_768):
        self.algorithm = algorithm

        # Key sizes based on algorithm
        self._sizes = {
            PQCAlgorithm.ML_KEM_512: (800, 1632, 768),
            PQCAlgorithm.ML_KEM_768: (1184, 2400, 1088),
            PQCAlgorithm.ML_KEM_1024: (1568, 3168, 1568),
        }

    def generate_keypair(self) -> KeyPair:
        """Generate simulated key pair."""
        pk_size, sk_size, _ = self._sizes.get(self.algorithm, (1184, 2400, 1088))

        return KeyPair(
            algorithm=self.algorithm,
            public_key=secrets.token_bytes(pk_size),
            private_key=secrets.token_bytes(sk_size),
            metadata={"simulated": True},
        )

    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Simulate encapsulation."""
        _, _, ct_size = self._sizes.get(self.algorithm, (1184, 2400, 1088))

        # In a real implementation, this would use ML-KEM
        ciphertext = secrets.token_bytes(ct_size)
        shared_secret = hashlib.sha256(public_key + ciphertext).digest()

        return ciphertext, shared_secret

    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Simulate decapsulation."""
        # In a real implementation, this would use ML-KEM
        return hashlib.sha256(private_key + ciphertext).digest()


class SimulatedMLDSA(PQCSignature):
    """Simulated ML-DSA for development and testing.

    WARNING: This is a simulation only. Use a real implementation
    (e.g., liboqs, pqcrypto) in production.
    """

    def __init__(self, algorithm: PQCAlgorithm = PQCAlgorithm.ML_DSA_65):
        self.algorithm = algorithm

        # Sizes based on algorithm
        self._sizes = {
            PQCAlgorithm.ML_DSA_44: (1312, 2560, 2420),
            PQCAlgorithm.ML_DSA_65: (1952, 4032, 3293),
            PQCAlgorithm.ML_DSA_87: (2592, 4896, 4595),
        }

    def generate_keypair(self) -> KeyPair:
        """Generate simulated signing key pair."""
        pk_size, sk_size, _ = self._sizes.get(self.algorithm, (1952, 4032, 3293))

        return KeyPair(
            algorithm=self.algorithm,
            public_key=secrets.token_bytes(pk_size),
            private_key=secrets.token_bytes(sk_size),
            metadata={"simulated": True},
        )

    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """Simulate signing."""
        _, _, sig_size = self._sizes.get(self.algorithm, (1952, 4032, 3293))

        # In a real implementation, this would use ML-DSA
        sig_data = hashlib.sha512(private_key + message).digest()
        return sig_data + secrets.token_bytes(sig_size - len(sig_data))

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Simulate verification (always returns True in simulation)."""
        # In a real implementation, this would verify with ML-DSA
        return len(signature) > 0 and len(public_key) > 0


class HybridKeyExchange:
    """Hybrid classical + PQC key exchange.

    Combines classical ECDH with ML-KEM for defense-in-depth.
    If one algorithm is broken, the other still provides security.

    This is the recommended approach during the PQC transition period.
    """

    def __init__(
        self,
        pqc_kem: Optional[PQCKeyExchange] = None,
        classical_curve: str = "X25519",
    ):
        """Initialize hybrid key exchange.

        Args:
            pqc_kem: PQC KEM implementation
            classical_curve: Classical ECDH curve
        """
        self.pqc_kem = pqc_kem or SimulatedMLKEM()
        self.classical_curve = classical_curve

        log.info(f"HybridKeyExchange initialized with {classical_curve} + ML-KEM")

    def generate_keypair(self) -> Tuple[KeyPair, KeyPair]:
        """Generate both classical and PQC key pairs.

        Returns:
            Tuple of (classical_keypair, pqc_keypair)
        """
        # Classical key (simulated)
        classical_keypair = KeyPair(
            algorithm=self.classical_curve,
            public_key=secrets.token_bytes(32),
            private_key=secrets.token_bytes(32),
            metadata={"type": "classical"},
        )

        # PQC key
        pqc_keypair = self.pqc_kem.generate_keypair()

        return classical_keypair, pqc_keypair

    def derive_shared_secret(
        self,
        classical_shared: bytes,
        pqc_shared: bytes,
    ) -> bytes:
        """Derive combined shared secret.

        Uses SHA-256(classical || pqc) for key combination.

        Args:
            classical_shared: Shared secret from ECDH
            pqc_shared: Shared secret from ML-KEM

        Returns:
            Combined shared secret
        """
        combined = hashlib.sha256(classical_shared + pqc_shared).digest()
        log.debug("Hybrid shared secret derived")
        return combined


@dataclass
class PQCReadinessAssessment:
    """Assessment of PQC migration readiness.

    Attributes:
        overall_score: Readiness score (0.0-1.0)
        crypto_inventory: Identified cryptographic uses
        migration_priority: Priority ranking for migration
        recommendations: Suggested actions
        estimated_effort: Estimated migration effort
        timestamp: Assessment timestamp
    """

    overall_score: float
    crypto_inventory: List[Dict[str, Any]]
    migration_priority: List[str]
    recommendations: List[str]
    estimated_effort: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 2),
            "crypto_inventory": self.crypto_inventory,
            "migration_priority": self.migration_priority,
            "recommendations": self.recommendations,
            "estimated_effort": self.estimated_effort,
            "timestamp": self.timestamp.isoformat(),
        }


class CryptoAgility:
    """Crypto-agility framework for algorithm migration.

    Provides infrastructure for seamless cryptographic algorithm
    upgrades without breaking existing functionality.
    """

    def __init__(self):
        """Initialize crypto-agility framework."""
        self._algorithms: Dict[str, Any] = {}
        self._default_kem: Optional[str] = None
        self._default_sig: Optional[str] = None

        # Register default algorithms
        self._register_defaults()

        log.info("CryptoAgility framework initialized")

    def _register_defaults(self) -> None:
        """Register default algorithm implementations."""
        # Register simulated PQC algorithms
        self._algorithms["ML-KEM-768"] = SimulatedMLKEM(PQCAlgorithm.ML_KEM_768)
        self._algorithms["ML-DSA-65"] = SimulatedMLDSA(PQCAlgorithm.ML_DSA_65)

        self._default_kem = "ML-KEM-768"
        self._default_sig = "ML-DSA-65"

    def register_algorithm(
        self,
        name: str,
        implementation: Any,
        set_default: bool = False,
    ) -> None:
        """Register a cryptographic algorithm.

        Args:
            name: Algorithm identifier
            implementation: Algorithm implementation
            set_default: Whether to set as default
        """
        self._algorithms[name] = implementation

        if set_default:
            if isinstance(implementation, PQCKeyExchange):
                self._default_kem = name
            elif isinstance(implementation, PQCSignature):
                self._default_sig = name

        log.info(f"Registered algorithm: {name}")

    def get_kem(self, name: Optional[str] = None) -> PQCKeyExchange:
        """Get KEM implementation.

        Args:
            name: Algorithm name (uses default if None)

        Returns:
            KEM implementation
        """
        name = name or self._default_kem
        if name and name in self._algorithms:
            alg = self._algorithms[name]
            if isinstance(alg, PQCKeyExchange):
                return alg
        raise ValueError(f"KEM not found: {name}")

    def get_signature(self, name: Optional[str] = None) -> PQCSignature:
        """Get signature implementation.

        Args:
            name: Algorithm name (uses default if None)

        Returns:
            Signature implementation
        """
        name = name or self._default_sig
        if name and name in self._algorithms:
            alg = self._algorithms[name]
            if isinstance(alg, PQCSignature):
                return alg
        raise ValueError(f"Signature algorithm not found: {name}")

    def list_algorithms(self) -> Dict[str, List[str]]:
        """List all registered algorithms.

        Returns:
            Dictionary with algorithm categories and names
        """
        kems = []
        sigs = []

        for name, impl in self._algorithms.items():
            if isinstance(impl, PQCKeyExchange):
                kems.append(name)
            elif isinstance(impl, PQCSignature):
                sigs.append(name)

        return {
            "key_exchange": kems,
            "signature": sigs,
            "default_kem": self._default_kem,
            "default_signature": self._default_sig,
        }


def assess_pqc_readiness(
    crypto_uses: Optional[List[Dict[str, Any]]] = None,
) -> PQCReadinessAssessment:
    """Assess organization's PQC migration readiness.

    Args:
        crypto_uses: List of current cryptographic uses

    Returns:
        PQC readiness assessment
    """
    crypto_uses = crypto_uses or []

    # Analyze current crypto usage
    classical_count = 0
    pqc_ready_count = 0
    high_priority = []

    for use in crypto_uses:
        algorithm = use.get("algorithm", "")

        # Check if already PQC
        if any(pqc.value in algorithm for pqc in PQCAlgorithm):
            pqc_ready_count += 1
        else:
            classical_count += 1

            # Check if high priority (long-term data protection)
            if use.get("data_lifetime_years", 0) > 10:
                high_priority.append(use.get("name", "unknown"))

    # Calculate readiness score
    total = classical_count + pqc_ready_count
    if total > 0:
        score = pqc_ready_count / total
    else:
        score = 0.5  # No crypto = medium readiness

    # Generate recommendations
    recommendations = []
    if score < 0.3:
        recommendations.append("Begin PQC migration planning immediately")
        recommendations.append("Inventory all cryptographic uses")
    if score < 0.6:
        recommendations.append("Implement hybrid classical/PQC for new systems")
        recommendations.append("Prioritize long-term data protection")
    if score < 1.0:
        recommendations.append("Continue migration to pure PQC algorithms")

    if not recommendations:
        recommendations.append("Maintain monitoring for PQC algorithm updates")

    # Estimate effort
    if classical_count == 0:
        effort = "Minimal - already PQC ready"
    elif classical_count < 5:
        effort = "Low - few systems to migrate"
    elif classical_count < 20:
        effort = "Medium - significant but manageable"
    else:
        effort = "High - enterprise-wide migration required"

    return PQCReadinessAssessment(
        overall_score=score,
        crypto_inventory=crypto_uses,
        migration_priority=high_priority,
        recommendations=recommendations,
        estimated_effort=effort,
    )
