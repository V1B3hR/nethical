"""
Plugin Registry Backend

This module implements the backend for the Nethical plugin marketplace,
providing plugin registration, metadata storage, security scanning integration,
and version compatibility checking.

Features:
- Plugin registration and metadata storage
- Security scanning integration
- Digital signature verification
- Version compatibility checking
- Plugin discovery and search
- Trust scoring and community reviews
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import sqlite3

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


class PluginSecurityStatus(Enum):
    """Security status of a plugin"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    QUARANTINED = "quarantined"


class PluginTrustLevel(Enum):
    """Trust level of a plugin"""

    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


@dataclass
class PluginRegistration:
    """Plugin registration data"""

    plugin_id: str
    name: str
    version: str
    author: str
    description: str
    entry_point: str
    plugin_type: str

    # Security
    security_status: PluginSecurityStatus = PluginSecurityStatus.PENDING
    trust_level: PluginTrustLevel = PluginTrustLevel.UNKNOWN
    signature: Optional[str] = None
    checksum: Optional[str] = None
    public_key_pem: Optional[str] = None
    manifest_hash: Optional[str] = None

    # Metadata
    requires_nethical_version: str = ">=0.1.0"
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    homepage: str = ""
    repository: str = ""
    license: str = ""

    # Tracking
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    downloads: int = 0
    rating: float = 0.0
    review_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["security_status"] = self.security_status.value
        data["trust_level"] = self.trust_level.value
        data["tags"] = list(self.tags)
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        return data


@dataclass
class SecurityScanResult:
    """Security scan result for a plugin"""

    plugin_id: str
    scan_date: datetime
    scanner: str
    status: str
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """
    Plugin registry backend for managing plugins in the marketplace.
    """

    def __init__(self, registry_dir: Path):
        """
        Initialize the plugin registry.

        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.registry_dir / "registry.db"
        self._init_database()

        logger.info(f"Plugin registry initialized at {self.registry_dir}")

    def _init_database(self):
        """Initialize SQLite database for plugin registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Plugins table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plugins (
                plugin_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                author TEXT NOT NULL,
                description TEXT,
                entry_point TEXT NOT NULL,
                plugin_type TEXT NOT NULL,
                security_status TEXT DEFAULT 'pending',
                trust_level INTEGER DEFAULT 0,
                signature TEXT,
                checksum TEXT,
                public_key_pem TEXT,
                manifest_hash TEXT,
                requires_nethical_version TEXT,
                dependencies TEXT,
                tags TEXT,
                homepage TEXT,
                repository TEXT,
                license TEXT,
                created_at TEXT,
                updated_at TEXT,
                downloads INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                review_count INTEGER DEFAULT 0,
                UNIQUE(name, version)
            )
        """
        )

        # Security scans table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS security_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plugin_id TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                scanner TEXT NOT NULL,
                status TEXT NOT NULL,
                vulnerabilities TEXT,
                warnings TEXT,
                score REAL DEFAULT 0.0,
                details TEXT,
                FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id)
            )
        """
        )

        # Reviews table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plugin_id TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id)
            )
        """
        )

        conn.commit()
        conn.close()

    def register_plugin(self, registration: PluginRegistration) -> bool:
        """
        Register a new plugin in the registry.

        Args:
            registration: Plugin registration data

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Set timestamps
            if not registration.created_at:
                registration.created_at = datetime.now(timezone.utc)
            registration.updated_at = datetime.now(timezone.utc)

            # Convert lists/sets to JSON strings
            dependencies_json = json.dumps(registration.dependencies)
            tags_json = json.dumps(list(registration.tags))

            cursor.execute(
                """
                INSERT INTO plugins (
                    plugin_id, name, version, author, description, entry_point,
                    plugin_type, security_status, trust_level, signature, checksum,
                    public_key_pem, manifest_hash,
                    requires_nethical_version, dependencies, tags, homepage,
                    repository, license, created_at, updated_at, downloads,
                    rating, review_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    registration.plugin_id,
                    registration.name,
                    registration.version,
                    registration.author,
                    registration.description,
                    registration.entry_point,
                    registration.plugin_type,
                    registration.security_status.value,
                    registration.trust_level.value,
                    registration.signature,
                    registration.checksum,
                    registration.public_key_pem,
                    registration.manifest_hash,
                    registration.requires_nethical_version,
                    dependencies_json,
                    tags_json,
                    registration.homepage,
                    registration.repository,
                    registration.license,
                    registration.created_at.isoformat(),
                    registration.updated_at.isoformat(),
                    registration.downloads,
                    registration.rating,
                    registration.review_count,
                ),
            )

            conn.commit()
            conn.close()

            logger.info(
                f"Registered plugin: {registration.name} v{registration.version}"
            )
            return True

        except sqlite3.IntegrityError as e:
            logger.error(f"Plugin already registered: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False

    def get_plugin(self, plugin_id: str) -> Optional[PluginRegistration]:
        """
        Get plugin by ID.

        Args:
            plugin_id: Plugin ID

        Returns:
            PluginRegistration if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM plugins WHERE plugin_id = ?", (plugin_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_registration(row, cursor.description)
            return None

        except Exception as e:
            logger.error(f"Failed to get plugin: {e}")
            return None

    def search_plugins(
        self,
        query: str = "",
        plugin_type: str = "",
        tags: List[str] = None,
        min_trust_level: PluginTrustLevel = PluginTrustLevel.UNKNOWN,
    ) -> List[PluginRegistration]:
        """
        Search for plugins.

        Args:
            query: Search query (name or description)
            plugin_type: Filter by plugin type
            tags: Filter by tags
            min_trust_level: Minimum trust level

        Returns:
            List of matching plugins
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            sql = "SELECT * FROM plugins WHERE 1=1"
            params = []

            if query:
                sql += " AND (name LIKE ? OR description LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])

            if plugin_type:
                sql += " AND plugin_type = ?"
                params.append(plugin_type)

            if min_trust_level != PluginTrustLevel.UNKNOWN:
                sql += " AND trust_level >= ?"
                params.append(min_trust_level.value)

            sql += " ORDER BY rating DESC, downloads DESC"

            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()

            plugins = []
            for row in rows:
                plugin = self._row_to_registration(row, cursor.description)

                # Filter by tags if specified
                if tags:
                    if any(tag in plugin.tags for tag in tags):
                        plugins.append(plugin)
                else:
                    plugins.append(plugin)

            return plugins

        except Exception as e:
            logger.error(f"Failed to search plugins: {e}")
            return []

    def verify_signature(self, plugin_id: str, signature: bytes) -> bool:
        """
        Verify plugin signature using cryptographic verification.

        Args:
            plugin_id: Plugin ID
            signature: Digital signature bytes to verify

        Returns:
            True if signature is valid, False otherwise
        """
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            logger.error(f"Plugin {plugin_id} not found")
            return False

        if not plugin.public_key_pem:
            logger.error(f"No public key available for plugin {plugin_id}")
            return False

        if not plugin.manifest_hash:
            logger.error(f"No manifest hash available for plugin {plugin_id}")
            return False

        try:
            # Load public key from PEM format
            public_key = serialization.load_pem_public_key(
                plugin.public_key_pem.encode()
            )

            # Verify signature using RSA-PSS with SHA-256
            public_key.verify(
                signature,
                plugin.manifest_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            logger.info(f"Signature verified successfully for plugin {plugin_id}")
            return True
        except InvalidSignature:
            logger.error(f"Invalid signature for plugin {plugin_id}")
            return False
        except Exception as e:
            logger.error(f"Error verifying signature for plugin {plugin_id}: {e}")
            return False

    def check_version_compatibility(
        self, plugin_id: str, nethical_version: str
    ) -> bool:
        """
        Check if plugin is compatible with a Nethical version.

        Args:
            plugin_id: Plugin ID
            nethical_version: Nethical version to check

        Returns:
            True if compatible, False otherwise
        """
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False

        # Parse version requirement
        requirement = plugin.requires_nethical_version

        # Simple version comparison (can be enhanced with proper semver)
        if ">=" in requirement:
            min_version = requirement.split(">=")[1].strip()
            return self._compare_versions(nethical_version, min_version) >= 0
        elif "==" in requirement:
            exact_version = requirement.split("==")[1].strip()
            return nethical_version == exact_version

        return True  # No specific requirement

    def submit_security_scan(self, scan_result: SecurityScanResult) -> bool:
        """
        Submit a security scan result for a plugin.

        Args:
            scan_result: Security scan result

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO security_scans (
                    plugin_id, scan_date, scanner, status, vulnerabilities,
                    warnings, score, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    scan_result.plugin_id,
                    scan_result.scan_date.isoformat(),
                    scan_result.scanner,
                    scan_result.status,
                    json.dumps(scan_result.vulnerabilities),
                    json.dumps(scan_result.warnings),
                    scan_result.score,
                    json.dumps(scan_result.details),
                ),
            )

            # Update plugin security status based on scan result
            if scan_result.status == "failed" or scan_result.vulnerabilities:
                new_status = PluginSecurityStatus.FLAGGED
            elif scan_result.status == "passed":
                new_status = PluginSecurityStatus.APPROVED
            else:
                new_status = PluginSecurityStatus.PENDING

            cursor.execute(
                """
                UPDATE plugins SET security_status = ?, updated_at = ?
                WHERE plugin_id = ?
            """,
                (
                    new_status.value,
                    datetime.now(timezone.utc).isoformat(),
                    scan_result.plugin_id,
                ),
            )

            conn.commit()
            conn.close()

            logger.info(f"Submitted security scan for plugin {scan_result.plugin_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit security scan: {e}")
            return False

    def add_review(
        self, plugin_id: str, reviewer: str, rating: int, comment: str = ""
    ) -> bool:
        """
        Add a review for a plugin.

        Args:
            plugin_id: Plugin ID
            reviewer: Reviewer name/ID
            rating: Rating (1-5)
            comment: Review comment

        Returns:
            True if successful, False otherwise
        """
        if not 1 <= rating <= 5:
            logger.error("Rating must be between 1 and 5")
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Add review
            cursor.execute(
                """
                INSERT INTO reviews (plugin_id, reviewer, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    plugin_id,
                    reviewer,
                    rating,
                    comment,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            # Update plugin rating
            cursor.execute(
                """
                SELECT AVG(rating), COUNT(*) FROM reviews WHERE plugin_id = ?
            """,
                (plugin_id,),
            )
            avg_rating, count = cursor.fetchone()

            cursor.execute(
                """
                UPDATE plugins SET rating = ?, review_count = ?, updated_at = ?
                WHERE plugin_id = ?
            """,
                (avg_rating, count, datetime.now(timezone.utc).isoformat(), plugin_id),
            )

            conn.commit()
            conn.close()

            logger.info(f"Added review for plugin {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add review: {e}")
            return False

    def increment_downloads(self, plugin_id: str) -> bool:
        """
        Increment download count for a plugin.

        Args:
            plugin_id: Plugin ID

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE plugins SET downloads = downloads + 1, updated_at = ?
                WHERE plugin_id = ?
            """,
                (datetime.now(timezone.utc).isoformat(), plugin_id),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to increment downloads: {e}")
            return False

    def _row_to_registration(self, row: tuple, description: list) -> PluginRegistration:
        """Convert database row to PluginRegistration"""
        columns = [col[0] for col in description]
        data = dict(zip(columns, row))

        return PluginRegistration(
            plugin_id=data["plugin_id"],
            name=data["name"],
            version=data["version"],
            author=data["author"],
            description=data["description"],
            entry_point=data["entry_point"],
            plugin_type=data["plugin_type"],
            security_status=PluginSecurityStatus(data["security_status"]),
            trust_level=PluginTrustLevel(data["trust_level"]),
            signature=data["signature"],
            checksum=data["checksum"],
            public_key_pem=data.get("public_key_pem"),
            manifest_hash=data.get("manifest_hash"),
            requires_nethical_version=data["requires_nethical_version"],
            dependencies=(
                json.loads(data["dependencies"]) if data["dependencies"] else []
            ),
            tags=set(json.loads(data["tags"])) if data["tags"] else set(),
            homepage=data["homepage"] or "",
            repository=data["repository"] or "",
            license=data["license"] or "",
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data["created_at"]
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data["updated_at"]
                else None
            ),
            downloads=data["downloads"],
            rating=data["rating"],
            review_count=data["review_count"],
        )

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """
        Compare two version strings.

        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        try:
            # Simple version comparison (major.minor.patch)
            # Strip pre-release tags and metadata for basic comparison
            v1_clean = v1.split("-")[0].split("+")[0]
            v2_clean = v2.split("-")[0].split("+")[0]

            parts1 = [int(x) for x in v1_clean.split(".")]
            parts2 = [int(x) for x in v2_clean.split(".")]

            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1

            # Handle different lengths
            if len(parts1) < len(parts2):
                return -1
            elif len(parts1) > len(parts2):
                return 1

            return 0

        except (ValueError, AttributeError) as e:
            logger.warning(f"Error comparing versions '{v1}' and '{v2}': {e}")
            # Fall back to string comparison
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            return 0

    def calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum for a plugin file.

        Args:
            file_path: Path to plugin file

        Returns:
            SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
