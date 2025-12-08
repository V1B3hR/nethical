"""
Plugin Security Module

Provides signature verification and security validation for Nethical plugins.
"""

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Plugin verification status."""

    VALID = "valid"
    INVALID_SIGNATURE = "invalid_signature"
    MISSING_SIGNATURE = "missing_signature"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNTRUSTED_PUBLISHER = "untrusted_publisher"
    MANIFEST_MISMATCH = "manifest_mismatch"


@dataclass
class VerificationResult:
    """Result of plugin signature verification."""

    status: VerificationStatus
    plugin_name: str
    version: str
    publisher: Optional[str]
    verified_at: str
    manifest_hash: Optional[str]
    signature_valid: bool
    message: str
    metadata: Dict[str, Any]


@dataclass
class PluginManifest:
    """Plugin manifest structure."""

    name: str
    version: str
    publisher: str
    description: str
    entry_point: str
    dependencies: List[str]
    permissions: List[str]
    checksum: str


class PluginVerifier:
    """Verifies plugin signatures and security."""

    def __init__(
        self,
        trusted_publishers: Optional[List[str]] = None,
        public_keys_dir: Optional[str] = None,
    ):
        self.trusted_publishers = trusted_publishers or []
        self.public_keys_dir = Path(public_keys_dir) if public_keys_dir else None
        self.revoked_plugins: Dict[str, List[str]] = {}  # plugin_name -> [versions]

    def verify_plugin(
        self,
        plugin_path: str,
        signature_path: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify plugin signature and integrity.

        Args:
            plugin_path: Path to the plugin directory or file
            signature_path: Path to signature file (optional, defaults to .sig file)

        Returns:
            VerificationResult with verification status and details
        """
        plugin_path = Path(plugin_path)

        if not plugin_path.exists():
            return VerificationResult(
                status=VerificationStatus.INVALID_SIGNATURE,
                plugin_name="unknown",
                version="unknown",
                publisher=None,
                verified_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash=None,
                signature_valid=False,
                message=f"Plugin path does not exist: {plugin_path}",
                metadata={},
            )

        # Load manifest
        manifest = self._load_manifest(plugin_path)
        if manifest is None:
            return VerificationResult(
                status=VerificationStatus.MANIFEST_MISMATCH,
                plugin_name="unknown",
                version="unknown",
                publisher=None,
                verified_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash=None,
                signature_valid=False,
                message="Could not load plugin manifest",
                metadata={},
            )

        # Check if plugin is revoked
        if self._is_revoked(manifest.name, manifest.version):
            return VerificationResult(
                status=VerificationStatus.REVOKED,
                plugin_name=manifest.name,
                version=manifest.version,
                publisher=manifest.publisher,
                verified_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash=manifest.checksum,
                signature_valid=False,
                message="Plugin version has been revoked",
                metadata={"reason": "security_vulnerability"},
            )

        # Check trusted publisher
        if (
            self.trusted_publishers
            and manifest.publisher not in self.trusted_publishers
        ):
            return VerificationResult(
                status=VerificationStatus.UNTRUSTED_PUBLISHER,
                plugin_name=manifest.name,
                version=manifest.version,
                publisher=manifest.publisher,
                verified_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash=manifest.checksum,
                signature_valid=False,
                message=f"Publisher '{manifest.publisher}' is not in trusted list",
                metadata={"trusted_publishers": self.trusted_publishers},
            )

        # Find signature file
        if signature_path is None:
            signature_path = plugin_path.parent / f"{plugin_path.name}.sig"
            if plugin_path.is_dir():
                signature_path = plugin_path / "signature.sig"

        signature_path = Path(signature_path)
        if not signature_path.exists():
            return VerificationResult(
                status=VerificationStatus.MISSING_SIGNATURE,
                plugin_name=manifest.name,
                version=manifest.version,
                publisher=manifest.publisher,
                verified_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash=manifest.checksum,
                signature_valid=False,
                message=f"Signature file not found: {signature_path}",
                metadata={},
            )

        # Verify manifest hash
        computed_hash = self.generate_manifest_hash(plugin_path)
        if computed_hash != manifest.checksum:
            return VerificationResult(
                status=VerificationStatus.MANIFEST_MISMATCH,
                plugin_name=manifest.name,
                version=manifest.version,
                publisher=manifest.publisher,
                verified_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash=computed_hash,
                signature_valid=False,
                message="Manifest checksum mismatch - plugin may have been modified",
                metadata={
                    "expected": manifest.checksum,
                    "computed": computed_hash,
                },
            )

        # Verify signature
        signature_valid = self._verify_signature(
            plugin_path, signature_path, manifest.publisher
        )

        if not signature_valid:
            return VerificationResult(
                status=VerificationStatus.INVALID_SIGNATURE,
                plugin_name=manifest.name,
                version=manifest.version,
                publisher=manifest.publisher,
                verified_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash=computed_hash,
                signature_valid=False,
                message="Signature verification failed",
                metadata={},
            )

        return VerificationResult(
            status=VerificationStatus.VALID,
            plugin_name=manifest.name,
            version=manifest.version,
            publisher=manifest.publisher,
            verified_at=datetime.now(timezone.utc).isoformat(),
            manifest_hash=computed_hash,
            signature_valid=True,
            message="Plugin signature verified successfully",
            metadata={
                "permissions": manifest.permissions,
                "dependencies": manifest.dependencies,
            },
        )

    def generate_manifest_hash(self, plugin_path: str) -> str:
        """
        Generate SHA-256 hash of plugin contents.

        Args:
            plugin_path: Path to plugin directory or file

        Returns:
            Hexadecimal SHA-256 hash string
        """
        plugin_path = Path(plugin_path)
        hasher = hashlib.sha256()

        if plugin_path.is_file():
            with open(plugin_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        elif plugin_path.is_dir():
            # Hash all files in directory sorted by name for consistency
            files = sorted(plugin_path.rglob("*"))
            for file_path in files:
                if file_path.is_file() and not file_path.name.endswith(".sig"):
                    # Include relative path in hash for structure integrity
                    rel_path = file_path.relative_to(plugin_path)
                    hasher.update(str(rel_path).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)
        else:
            raise ValueError(f"Invalid plugin path: {plugin_path}")

        return hasher.hexdigest()

    def _load_manifest(self, plugin_path: Path) -> Optional[PluginManifest]:
        """Load and parse plugin manifest."""
        manifest_path = None

        if plugin_path.is_dir():
            manifest_path = plugin_path / "manifest.json"
        elif plugin_path.is_file() and plugin_path.suffix == ".json":
            manifest_path = plugin_path

        if manifest_path is None or not manifest_path.exists():
            return None

        try:
            with open(manifest_path) as f:
                data = json.load(f)

            return PluginManifest(
                name=data.get("name", "unknown"),
                version=data.get("version", "0.0.0"),
                publisher=data.get("publisher", "unknown"),
                description=data.get("description", ""),
                entry_point=data.get("entry_point", ""),
                dependencies=data.get("dependencies", []),
                permissions=data.get("permissions", []),
                checksum=data.get("checksum", ""),
            )
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return None

    def _is_revoked(self, plugin_name: str, version: str) -> bool:
        """Check if a plugin version is revoked."""
        revoked_versions = self.revoked_plugins.get(plugin_name, [])
        return version in revoked_versions

    def _verify_signature(
        self, plugin_path: Path, signature_path: Path, publisher: str
    ) -> bool:
        """
        Verify plugin signature using publisher's public key.

        WARNING: This is a placeholder implementation that provides basic
        signature format validation only. It does NOT provide cryptographic
        security guarantees.

        For production use, implement proper cryptographic verification using:
        - RSA/ECDSA signature verification with the publisher's public key
        - Integration with a PKI or key management system
        - Certificate chain validation

        Current implementation only verifies:
        - Signature file exists and is readable
        - Signature file has minimum expected length (64 bytes)
        """
        logger.warning(
            "Plugin signature verification is using placeholder implementation. "
            "For production security, implement proper cryptographic verification."
        )
        try:
            # Read signature file
            with open(signature_path, "rb") as f:
                signature_data = f.read()

            # Basic format validation only - NOT cryptographically secure
            # A proper implementation would:
            # 1. Load the publisher's public key
            # 2. Verify the signature using RSA/ECDSA
            # 3. Validate the certificate chain
            if len(signature_data) < 64:
                logger.warning("Signature file too short - expected at least 64 bytes")
                return False

            # Compute plugin hash for logging purposes
            plugin_hash = self.generate_manifest_hash(str(plugin_path))
            logger.debug(f"Plugin hash: {plugin_hash[:16]}...")

            # Placeholder: accept any signature with minimum length
            # TODO: Implement proper cryptographic verification
            return True

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def revoke_plugin(self, plugin_name: str, version: str, reason: str = "") -> None:
        """Revoke a plugin version."""
        if plugin_name not in self.revoked_plugins:
            self.revoked_plugins[plugin_name] = []
        if version not in self.revoked_plugins[plugin_name]:
            self.revoked_plugins[plugin_name].append(version)
            logger.info(f"Revoked plugin {plugin_name}@{version}: {reason}")

    def add_trusted_publisher(self, publisher: str) -> None:
        """Add a publisher to the trusted list."""
        if publisher not in self.trusted_publishers:
            self.trusted_publishers.append(publisher)
            logger.info(f"Added trusted publisher: {publisher}")

    def remove_trusted_publisher(self, publisher: str) -> None:
        """Remove a publisher from the trusted list."""
        if publisher in self.trusted_publishers:
            self.trusted_publishers.remove(publisher)
            logger.info(f"Removed trusted publisher: {publisher}")
