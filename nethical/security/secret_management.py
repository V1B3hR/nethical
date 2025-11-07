"""
Secret Management for Nethical

This module provides comprehensive secret management capabilities:
- HashiCorp Vault integration
- Dynamic secret generation
- Automated secret rotation
- Secret scanning in code repositories
- Encryption key management

Compliance: NIST SP 800-53 SC-12, SC-13, IA-5
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Pattern
from abc import ABC, abstractmethod

__all__ = [
    "SecretType",
    "SecretRotationPolicy",
    "VaultConfig",
    "Secret",
    "SecretScanner",
    "DynamicSecretGenerator",
    "SecretRotationManager",
    "VaultIntegration",
    "SecretManagementSystem",
]

log = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets managed by the system"""
    API_KEY = "api_key"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    DATABASE_CREDENTIAL = "database_credential"
    OAUTH_TOKEN = "oauth_token"
    ENCRYPTION_KEY = "encryption_key"
    SSH_KEY = "ssh_key"


@dataclass
class SecretRotationPolicy:
    """Policy for automatic secret rotation"""
    secret_type: SecretType
    rotation_interval_days: int
    notify_before_days: int = 7
    auto_rotate: bool = True
    retain_old_versions: int = 3
    require_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VaultConfig:
    """HashiCorp Vault configuration"""
    vault_address: str
    vault_token: Optional[str] = None
    vault_namespace: Optional[str] = None
    vault_role: Optional[str] = None
    mount_point: str = "secret"
    tls_verify: bool = True
    tls_ca_cert: Optional[str] = None
    enabled: bool = True
    
    def validate(self) -> bool:
        """Validate vault configuration"""
        if not self.enabled:
            return True
        return bool(self.vault_address and (self.vault_token or self.vault_role))


@dataclass
class Secret:
    """Secret container with metadata"""
    secret_id: str
    secret_type: SecretType
    value: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_rotated: Optional[datetime] = None
    rotation_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if secret has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def needs_rotation(self, policy: SecretRotationPolicy) -> bool:
        """Check if secret needs rotation based on policy"""
        if not policy.auto_rotate:
            return False
        
        rotation_date = self.last_rotated or self.created_at
        age_days = (datetime.now(timezone.utc) - rotation_date).days
        return age_days >= policy.rotation_interval_days
    
    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "secret_id": self.secret_id,
            "secret_type": self.secret_type.value,
            "created_at": self.created_at.isoformat(),
            "rotation_count": self.rotation_count,
            "tags": self.tags,
            "metadata": self.metadata,
        }
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        if self.last_rotated:
            data["last_rotated"] = self.last_rotated.isoformat()
        if include_value:
            data["value"] = self.value
        return data


class SecretScanner:
    """
    Secret scanner for code repositories
    
    Scans code for hardcoded secrets using:
    - Pattern matching
    - Entropy analysis
    - Known secret patterns
    """
    
    # Common secret patterns
    PATTERNS: Dict[SecretType, List[Pattern]] = {
        SecretType.API_KEY: [
            re.compile(r'api[_-]?key[\'"\s:=]+([a-zA-Z0-9_\-]{20,})'),
            re.compile(r'apikey[\'"\s:=]+([a-zA-Z0-9_\-]{20,})'),
        ],
        SecretType.PASSWORD: [
            re.compile(r'password[\'"\s:=]+([^\s\'"]{8,})'),
            re.compile(r'passwd[\'"\s:=]+([^\s\'"]{8,})'),
        ],
        SecretType.PRIVATE_KEY: [
            re.compile(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'),
        ],
        SecretType.OAUTH_TOKEN: [
            re.compile(r'bearer [a-zA-Z0-9_\-\.]{20,}', re.IGNORECASE),
        ],
        SecretType.DATABASE_CREDENTIAL: [
            re.compile(r'(postgres|mysql|mongodb)://[^:]+:[^@]+@'),
        ],
    }
    
    def __init__(self):
        """Initialize secret scanner"""
        self.findings: List[Dict[str, Any]] = []
        log.info("SecretScanner initialized")
    
    def scan_text(
        self,
        text: str,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan text for hardcoded secrets
        
        Args:
            text: Text to scan
            file_path: Optional file path for reporting
        
        Returns:
            List of findings with secret type, location, and pattern
        """
        findings = []
        
        for secret_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    finding = {
                        "secret_type": secret_type.value,
                        "pattern": pattern.pattern,
                        "line_number": text[:match.start()].count('\n') + 1,
                        "match": match.group(0),
                        "file_path": file_path,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    findings.append(finding)
                    log.warning(f"Secret found: {secret_type} in {file_path or 'text'}")
        
        # Check for high entropy strings (potential secrets)
        high_entropy_findings = self._check_entropy(text, file_path)
        findings.extend(high_entropy_findings)
        
        self.findings.extend(findings)
        return findings
    
    def scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Scan a file for hardcoded secrets
        
        Args:
            file_path: Path to file to scan
        
        Returns:
            List of findings
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.scan_text(content, file_path)
        except Exception as e:
            log.error(f"Error scanning file {file_path}: {e}")
            return []
    
    def _check_entropy(
        self,
        text: str,
        file_path: Optional[str] = None,
        min_entropy: float = 4.5,
        min_length: int = 20,
    ) -> List[Dict[str, Any]]:
        """Check for high entropy strings that might be secrets"""
        findings = []
        
        # Extract strings that look like they could be secrets
        potential_secrets = re.findall(r'[a-zA-Z0-9_\-]{20,}', text)
        
        for candidate in potential_secrets:
            if len(candidate) >= min_length:
                entropy = self._calculate_entropy(candidate)
                if entropy >= min_entropy:
                    finding = {
                        "secret_type": "high_entropy_string",
                        "pattern": "entropy_check",
                        "entropy": entropy,
                        "match": candidate[:50] + "..." if len(candidate) > 50 else candidate,
                        "file_path": file_path,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    findings.append(finding)
        
        return findings
    
    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        import math
        
        if not string:
            return 0.0
        
        # Count character frequencies
        frequencies = {}
        for char in string:
            frequencies[char] = frequencies.get(char, 0) + 1
        
        # Calculate Shannon entropy
        entropy = 0.0
        length = len(string)
        for count in frequencies.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def get_findings_summary(self) -> Dict[str, Any]:
        """Get summary of scan findings"""
        summary = {
            "total_findings": len(self.findings),
            "by_type": {},
            "files_affected": set(),
        }
        
        for finding in self.findings:
            secret_type = finding.get("secret_type", "unknown")
            summary["by_type"][secret_type] = summary["by_type"].get(secret_type, 0) + 1
            if finding.get("file_path"):
                summary["files_affected"].add(finding["file_path"])
        
        summary["files_affected"] = len(summary["files_affected"])
        return summary


class DynamicSecretGenerator:
    """
    Dynamic secret generator
    
    Generates secrets on-demand with configurable policies:
    - API keys
    - Passwords
    - Database credentials
    - Time-limited tokens
    """
    
    def __init__(self):
        """Initialize dynamic secret generator"""
        self.generated_secrets: Dict[str, Secret] = {}
        log.info("DynamicSecretGenerator initialized")
    
    def generate_api_key(
        self,
        secret_id: str,
        length: int = 32,
        ttl_hours: Optional[int] = None,
    ) -> Secret:
        """
        Generate a new API key
        
        Args:
            secret_id: Unique identifier for the secret
            length: Length of the API key
            ttl_hours: Time-to-live in hours (optional)
        
        Returns:
            Secret object with generated API key
        """
        value = secrets.token_urlsafe(length)
        
        expires_at = None
        if ttl_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        
        secret = Secret(
            secret_id=secret_id,
            secret_type=SecretType.API_KEY,
            value=value,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )
        
        self.generated_secrets[secret_id] = secret
        log.info("Generated API key")
        return secret
    
    def generate_password(
        self,
        secret_id: str,
        length: int = 24,
        include_special: bool = True,
    ) -> Secret:
        """
        Generate a strong password
        
        Args:
            secret_id: Unique identifier for the secret
            length: Length of the password
            include_special: Include special characters
        
        Returns:
            Secret object with generated password
        """
        import string
        
        # Character sets
        chars = string.ascii_letters + string.digits
        if include_special:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Generate password
        value = ''.join(secrets.choice(chars) for _ in range(length))
        
        secret = Secret(
            secret_id=secret_id,
            secret_type=SecretType.PASSWORD,
            value=value,
            created_at=datetime.now(timezone.utc),
        )
        
        self.generated_secrets[secret_id] = secret
        log.info("Generated password")
        return secret
    
    def generate_database_credential(
        self,
        secret_id: str,
        username: str,
        database_type: str = "postgresql",
    ) -> Secret:
        """
        Generate database credentials
        
        Args:
            secret_id: Unique identifier for the secret
            username: Database username
            database_type: Type of database
        
        Returns:
            Secret object with generated credentials
        """
        password = secrets.token_urlsafe(24)
        value = f"{database_type}://{username}:{password}@localhost:5432/database"
        
        secret = Secret(
            secret_id=secret_id,
            secret_type=SecretType.DATABASE_CREDENTIAL,
            value=value,
            created_at=datetime.now(timezone.utc),
            metadata={"username": username, "database_type": database_type},
        )
        
        self.generated_secrets[secret_id] = secret
        log.info(f"Generated database credential for type: {database_type}")
        return secret
    
    def generate_encryption_key(
        self,
        secret_id: str,
        key_size: int = 32,
    ) -> Secret:
        """
        Generate encryption key
        
        Args:
            secret_id: Unique identifier for the secret
            key_size: Size of the key in bytes
        
        Returns:
            Secret object with generated key
        """
        value = secrets.token_bytes(key_size).hex()
        
        secret = Secret(
            secret_id=secret_id,
            secret_type=SecretType.ENCRYPTION_KEY,
            value=value,
            created_at=datetime.now(timezone.utc),
        )
        
        self.generated_secrets[secret_id] = secret
        log.info("Generated encryption key")
        return secret


class SecretRotationManager:
    """
    Automated secret rotation manager
    
    Manages automatic rotation of secrets based on policies:
    - Scheduled rotation
    - Event-triggered rotation
    - Emergency rotation
    """
    
    def __init__(self):
        """Initialize secret rotation manager"""
        self.policies: Dict[SecretType, SecretRotationPolicy] = {}
        self.rotation_history: List[Dict[str, Any]] = []
        log.info("SecretRotationManager initialized")
    
    def add_policy(self, policy: SecretRotationPolicy) -> None:
        """Add rotation policy for a secret type"""
        self.policies[policy.secret_type] = policy
        log.info(f"Added rotation policy for secret type: {policy.secret_type.value}")
    
    def should_rotate(self, secret: Secret) -> bool:
        """
        Check if a secret should be rotated
        
        Args:
            secret: Secret to check
        
        Returns:
            True if secret should be rotated
        """
        if secret.secret_type not in self.policies:
            return False
        
        policy = self.policies[secret.secret_type]
        return secret.needs_rotation(policy)
    
    def rotate_secret(
        self,
        secret: Secret,
        generator: DynamicSecretGenerator,
    ) -> Secret:
        """
        Rotate a secret
        
        Args:
            secret: Secret to rotate
            generator: Secret generator to use
        
        Returns:
            New secret with rotated value
        """
        # Generate new secret based on type
        new_secret_id = f"{secret.secret_id}_v{secret.rotation_count + 1}"
        
        if secret.secret_type == SecretType.API_KEY:
            new_secret = generator.generate_api_key(new_secret_id)
        elif secret.secret_type == SecretType.PASSWORD:
            new_secret = generator.generate_password(new_secret_id)
        elif secret.secret_type == SecretType.ENCRYPTION_KEY:
            new_secret = generator.generate_encryption_key(new_secret_id)
        else:
            log.warning(f"Rotation not supported for secret type: {secret.secret_type.value}")
            return secret
        
        # Update metadata
        new_secret.rotation_count = secret.rotation_count + 1
        new_secret.last_rotated = datetime.now(timezone.utc)
        new_secret.tags = secret.tags
        new_secret.metadata = secret.metadata.copy()
        
        # Record rotation
        self.rotation_history.append({
            "old_secret_id": secret.secret_id,
            "new_secret_id": new_secret.secret_id,
            "secret_type": secret.secret_type.value,
            "rotated_at": datetime.now(timezone.utc).isoformat(),
        })
        
        log.info(f"Secret rotated: type={secret.secret_type.value}, rotation_count={new_secret.rotation_count}")
        return new_secret
    
    def get_rotation_schedule(
        self,
        secrets: List[Secret],
    ) -> List[Dict[str, Any]]:
        """
        Get rotation schedule for secrets
        
        Args:
            secrets: List of secrets to check
        
        Returns:
            List of secrets due for rotation
        """
        schedule = []
        
        for secret in secrets:
            if self.should_rotate(secret):
                policy = self.policies.get(secret.secret_type)
                if policy:
                    schedule.append({
                        "secret_id": secret.secret_id,
                        "secret_type": secret.secret_type.value,
                        "last_rotated": secret.last_rotated.isoformat() if secret.last_rotated else None,
                        "rotation_interval_days": policy.rotation_interval_days,
                        "auto_rotate": policy.auto_rotate,
                    })
        
        return schedule


class VaultIntegration:
    """
    HashiCorp Vault integration
    
    Provides integration with HashiCorp Vault for:
    - Secret storage
    - Dynamic secret generation
    - Secret leasing
    - Access control
    """
    
    def __init__(self, config: VaultConfig):
        """
        Initialize Vault integration
        
        Args:
            config: Vault configuration
        """
        self.config = config
        self.connected = False
        log.info("VaultIntegration initialized")
    
    def connect(self) -> bool:
        """
        Connect to Vault
        
        Returns:
            True if connected successfully
        """
        if not self.config.validate():
            log.error("Invalid Vault configuration")
            return False
        
        # In production, this would establish connection to Vault
        # For now, simulate successful connection
        self.connected = self.config.enabled
        log.info(f"Vault connection: {self.connected}")
        return self.connected
    
    def store_secret(
        self,
        path: str,
        secret: Secret,
    ) -> bool:
        """
        Store secret in Vault
        
        Args:
            path: Vault path for secret
            secret: Secret to store
        
        Returns:
            True if stored successfully
        """
        if not self.connected:
            log.error("Not connected to Vault")
            return False
        
        # In production, this would use Vault API
        log.info(f"Stored secret in Vault: {path}")
        return True
    
    def retrieve_secret(
        self,
        path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve secret from Vault
        
        Args:
            path: Vault path for secret
        
        Returns:
            Secret data or None if not found
        """
        if not self.connected:
            log.error("Not connected to Vault")
            return None
        
        # In production, this would use Vault API
        log.info(f"Retrieved secret from Vault: {path}")
        return {"data": {"value": "simulated_secret"}}
    
    def delete_secret(self, path: str) -> bool:
        """
        Delete secret from Vault
        
        Args:
            path: Vault path for secret
        
        Returns:
            True if deleted successfully
        """
        if not self.connected:
            log.error("Not connected to Vault")
            return False
        
        # In production, this would use Vault API
        log.info(f"Deleted secret from Vault: {path}")
        return True


class SecretManagementSystem:
    """
    Comprehensive secret management system
    
    Orchestrates all secret management components:
    - Secret scanning
    - Dynamic generation
    - Automated rotation
    - Vault integration
    """
    
    def __init__(
        self,
        vault_config: Optional[VaultConfig] = None,
    ):
        """
        Initialize secret management system
        
        Args:
            vault_config: Optional Vault configuration
        """
        self.scanner = SecretScanner()
        self.generator = DynamicSecretGenerator()
        self.rotation_manager = SecretRotationManager()
        self.vault = VaultIntegration(vault_config) if vault_config else None
        self.secrets: Dict[str, Secret] = {}
        
        # Initialize default rotation policies
        self._init_default_policies()
        
        log.info("SecretManagementSystem initialized")
    
    def _init_default_policies(self) -> None:
        """Initialize default rotation policies"""
        # API keys: rotate every 90 days
        self.rotation_manager.add_policy(SecretRotationPolicy(
            secret_type=SecretType.API_KEY,
            rotation_interval_days=90,
            auto_rotate=True,
        ))
        
        # Passwords: rotate every 60 days
        self.rotation_manager.add_policy(SecretRotationPolicy(
            secret_type=SecretType.PASSWORD,
            rotation_interval_days=60,
            auto_rotate=True,
        ))
        
        # Encryption keys: rotate every 365 days
        self.rotation_manager.add_policy(SecretRotationPolicy(
            secret_type=SecretType.ENCRYPTION_KEY,
            rotation_interval_days=365,
            auto_rotate=True,
            require_approval=True,
        ))
    
    def scan_for_secrets(
        self,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Scan file for hardcoded secrets
        
        Args:
            file_path: Path to file to scan
        
        Returns:
            List of findings
        """
        return self.scanner.scan_file(file_path)
    
    def create_secret(
        self,
        secret_id: str,
        secret_type: SecretType,
        **kwargs,
    ) -> Secret:
        """
        Create a new secret
        
        Args:
            secret_id: Unique identifier
            secret_type: Type of secret to create
            **kwargs: Additional parameters for generation
        
        Returns:
            Generated secret
        """
        if secret_type == SecretType.API_KEY:
            secret = self.generator.generate_api_key(secret_id, **kwargs)
        elif secret_type == SecretType.PASSWORD:
            secret = self.generator.generate_password(secret_id, **kwargs)
        elif secret_type == SecretType.ENCRYPTION_KEY:
            secret = self.generator.generate_encryption_key(secret_id, **kwargs)
        elif secret_type == SecretType.DATABASE_CREDENTIAL:
            secret = self.generator.generate_database_credential(secret_id, **kwargs)
        else:
            raise ValueError(f"Unsupported secret type: {secret_type}")
        
        self.secrets[secret_id] = secret
        
        # Store in Vault if available
        if self.vault and self.vault.connected:
            self.vault.store_secret(f"secrets/{secret_id}", secret)
        
        return secret
    
    def rotate_secrets(self) -> List[str]:
        """
        Rotate all secrets that are due
        
        Returns:
            List of rotated secret IDs
        """
        rotated = []
        
        for secret_id, secret in list(self.secrets.items()):
            if self.rotation_manager.should_rotate(secret):
                new_secret = self.rotation_manager.rotate_secret(
                    secret, self.generator
                )
                self.secrets[new_secret.secret_id] = new_secret
                rotated.append(secret_id)
                
                # Update Vault if available
                if self.vault and self.vault.connected:
                    self.vault.store_secret(
                        f"secrets/{new_secret.secret_id}", new_secret
                    )
        
        return rotated
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "total_secrets": len(self.secrets),
            "vault_connected": self.vault.connected if self.vault else False,
            "scan_findings": len(self.scanner.findings),
            "rotation_policies": len(self.rotation_manager.policies),
            "rotation_history": len(self.rotation_manager.rotation_history),
        }
