"""Marketplace Client for Plugin Management.

This module provides the MarketplaceClient for searching, installing,
and managing plugins from the Nethical ecosystem.

Enhancements:
- Robust semantic version comparison and compatibility ranges (e.g., ">=0.1.0,<0.3.0").
- Safer SQLite usage with context managers, foreign keys, indices, and light migrations.
- Structured logging and clearer error types.
- Optional download, checksum verification, and extraction of plugin archives.
- Dependency resolution with optional version constraints (e.g., "plugin-id@>=1.2.0").
- Thread-safety with per-plugin locks and global registry lock.
- Improved install/uninstall/update flows, with status validation against filesystem.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import json
import sqlite3
from pathlib import Path
import shutil
import importlib.util
import sys
import logging
import threading
import hashlib
import urllib.request
from urllib.parse import urlparse
import zipfile
import tarfile
import itertools
import time


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# If the host app doesn't configure logging, default to INFO for this module
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Error Types
# -----------------------------------------------------------------------------
class MarketplaceError(Exception):
    """Base exception for marketplace operations."""


class PluginNotFoundError(MarketplaceError):
    """Raised when a plugin is not found in the marketplace."""


class VersionNotFoundError(MarketplaceError):
    """Raised when a specific plugin version is not found."""


class IncompatibleVersionError(MarketplaceError):
    """Raised when a plugin version is not compatible with Nethical."""


class InstallationError(MarketplaceError):
    """Raised when a plugin installation fails."""


class UninstallationError(MarketplaceError):
    """Raised when a plugin uninstallation fails."""


# -----------------------------------------------------------------------------
# Status / Models
# -----------------------------------------------------------------------------
class InstallStatus(Enum):
    """Plugin installation status."""

    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    NEEDS_UPDATE = "needs_update"
    FAILED = "failed"


@dataclass
class PluginVersion:
    """Plugin version information."""

    version: str
    release_date: datetime
    compatibility: str  # Compatible Nethical version range, e.g. ">=0.1.0,<0.3.0"
    changelog: str = ""
    download_url: Optional[str] = None
    checksum_sha256: Optional[str] = None  # Optional SHA256 for integrity
    size_bytes: Optional[int] = None  # Optional expected size

    def is_compatible(self, nethical_version: str) -> bool:
        """Check if this version is compatible with given Nethical version.

        Supports constraints like:
        - >=1.2.3
        - ==0.2.0
        - >0.1.0,<=0.3.0 (commas imply AND)
        - If empty or None, returns True
        """
        constraint = (self.compatibility or "").strip()
        if not constraint:
            return True

        return MarketplaceClient._version_satisfies_constraints(
            nethical_version, constraint
        )


@dataclass
class PluginInfo:
    """Plugin information from marketplace."""

    plugin_id: str
    name: str
    description: str
    author: str
    category: str
    rating: float = 0.0
    download_count: int = 0
    tags: Set[str] = field(default_factory=set)
    versions: List[PluginVersion] = field(default_factory=list)
    latest_version: str = "0.1.0"
    dependencies: List[str] = field(
        default_factory=list
    )  # e.g., ["dep-a", "dep-b@>=1.0.0"]
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None
    certified: bool = False

    def get_version(self, version: str) -> Optional[PluginVersion]:
        """Get specific version information."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_latest_version(self) -> Optional[PluginVersion]:
        """Get latest version information."""
        return self.get_version(self.latest_version)


@dataclass
class SearchFilters:
    """Filters for marketplace search."""

    category: Optional[str] = None
    min_rating: float = 0.0
    compatible_version: Optional[str] = None
    certified_only: bool = False
    tags: Set[str] = field(default_factory=set)
    author: Optional[str] = None


# -----------------------------------------------------------------------------
# Marketplace Client
# -----------------------------------------------------------------------------
class MarketplaceClient:
    """Client for interacting with the Nethical plugin marketplace.

    This client provides methods to search, install, and manage plugins
    from the marketplace.

    Example:
        >>> marketplace = MarketplaceClient()
        >>> results = marketplace.search(category="financial", min_rating=4.0)
        >>> marketplace.install("financial-compliance-v2", version="1.2.3")
        >>> marketplace.list_installed()
    """

    def __init__(
        self,
        storage_dir: str = "./nethical_marketplace",
        marketplace_url: Optional[str] = None,
        nethical_version: str = "0.1.0",
        request_timeout_s: float = 30.0,
        download_retries: int = 2,
    ):
        """Initialize marketplace client.

        Args:
            storage_dir: Directory for plugin storage and cache
            marketplace_url: URL of marketplace API (None for local/mock mode)
            nethical_version: Current Nethical version for compatibility checks
            request_timeout_s: Network timeout in seconds for downloads
            download_retries: Retry attempts for downloads
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.plugins_dir = self.storage_dir / "plugins"
        self.plugins_dir.mkdir(exist_ok=True)

        self.cache_dir = self.storage_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.marketplace_url = marketplace_url
        self.nethical_version = nethical_version
        self.request_timeout_s = request_timeout_s
        self.download_retries = max(0, int(download_retries))

        # Initialize local registry database
        self.db_path = self.storage_dir / "registry.db"
        self._init_database()

        # Locks
        self._registry_lock = threading.RLock()
        self._plugin_locks: Dict[str, threading.RLock] = {}

        # In-memory plugin registry cache
        self._plugin_registry: Dict[str, PluginInfo] = {}
        self._load_registry_cache()

    # -------------------------------------------------------------------------
    # SQLite helpers and schema
    # -------------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Ensure FK support
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_database(self):
        """Initialize SQLite database for plugin registry."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Plugins table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS plugins (
                    plugin_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    category TEXT,
                    rating REAL DEFAULT 0.0,
                    download_count INTEGER DEFAULT 0,
                    latest_version TEXT,
                    license TEXT,
                    homepage TEXT,
                    repository TEXT,
                    certified INTEGER DEFAULT 0,
                    installed_version TEXT,
                    install_status TEXT DEFAULT 'not_installed',
                    install_date TEXT,
                    last_updated TEXT
                )
            """
            )

            # Plugin versions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS plugin_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plugin_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    release_date TEXT,
                    compatibility TEXT,
                    changelog TEXT,
                    download_url TEXT,
                    checksum_sha256 TEXT,
                    size_bytes INTEGER,
                    FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE,
                    UNIQUE(plugin_id, version)
                )
            """
            )

            # Plugin tags table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS plugin_tags (
                    plugin_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (plugin_id, tag),
                    FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE
                )
            """
            )

            # Plugin dependencies table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS plugin_dependencies (
                    plugin_id TEXT NOT NULL,
                    dependency TEXT NOT NULL,
                    PRIMARY KEY (plugin_id, dependency),
                    FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE
                )
            """
            )

            # Indices for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plugins_status ON plugins(install_status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plugins_category ON plugins(category)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plugins_rating ON plugins(rating)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plugins_downloads ON plugins(download_count)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_versions_plugin ON plugin_versions(plugin_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tags_plugin ON plugin_tags(plugin_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_deps_plugin ON plugin_dependencies(plugin_id)"
            )

            # Light migration: ensure columns exist (checksum_sha256, size_bytes)
            self._ensure_column(cursor, "plugin_versions", "checksum_sha256", "TEXT")
            self._ensure_column(cursor, "plugin_versions", "size_bytes", "INTEGER")

            conn.commit()

    @staticmethod
    def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, coltype: str):
        cursor.execute(f"PRAGMA table_info({table})")
        cols = {row["name"] for row in cursor.fetchall()}
        if column not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

    # -------------------------------------------------------------------------
    # Registry cache
    # -------------------------------------------------------------------------
    def _load_registry_cache(self):
        """Load plugin registry from database into memory."""
        with self._registry_lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                # Load basic plugin info
                cursor.execute("SELECT * FROM plugins")
                plugins_rows = cursor.fetchall()

                new_registry: Dict[str, PluginInfo] = {}

                for row in plugins_rows:
                    plugin_id = row["plugin_id"]

                    # Load tags
                    cursor.execute(
                        "SELECT tag FROM plugin_tags WHERE plugin_id = ?", (plugin_id,)
                    )
                    tags = {tag_row["tag"] for tag_row in cursor.fetchall()}

                    # Load dependencies
                    cursor.execute(
                        "SELECT dependency FROM plugin_dependencies WHERE plugin_id = ?",
                        (plugin_id,),
                    )
                    dependencies = [
                        dep_row["dependency"] for dep_row in cursor.fetchall()
                    ]

                    # Load versions
                    cursor.execute(
                        "SELECT * FROM plugin_versions WHERE plugin_id = ?",
                        (plugin_id,),
                    )
                    versions = []
                    for v_row in cursor.fetchall():
                        versions.append(
                            PluginVersion(
                                version=v_row["version"],
                                release_date=(
                                    datetime.fromisoformat(v_row["release_date"])
                                    if v_row["release_date"]
                                    else datetime.now()
                                ),
                                compatibility=v_row["compatibility"] or ">=0.1.0",
                                changelog=v_row["changelog"] or "",
                                download_url=v_row["download_url"],
                                checksum_sha256=v_row["checksum_sha256"],
                                size_bytes=v_row["size_bytes"],
                            )
                        )

                    new_registry[plugin_id] = PluginInfo(
                        plugin_id=plugin_id,
                        name=row["name"],
                        description=row["description"] or "",
                        author=row["author"] or "",
                        category=row["category"] or "",
                        rating=row["rating"] or 0.0,
                        download_count=row["download_count"] or 0,
                        latest_version=row["latest_version"] or "0.1.0",
                        license=row["license"] or "MIT",
                        homepage=row["homepage"],
                        repository=row["repository"],
                        certified=bool(row["certified"]),
                        tags=tags,
                        dependencies=dependencies,
                        versions=versions,
                    )

                self._plugin_registry = new_registry

    # -------------------------------------------------------------------------
    # Version utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_version_tuple(v: str) -> Tuple[int, ...]:
        """Parse a semantic version into a tuple of ints, ignoring pre-release/build metadata."""
        # Basic split by '.', ignore suffixes like '-alpha'
        parts: List[int] = []
        for part in v.split("."):
            # Strip any pre-release metadata e.g. '1-alpha' -> '1'
            num = ""
            for ch in part:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num == "":
                # If no numeric found, treat as 0
                parts.append(0)
            else:
                parts.append(int(num))
        return tuple(parts)

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare two semantic versions. Returns -1, 0, or 1."""
        t1 = MarketplaceClient._parse_version_tuple(v1)
        t2 = MarketplaceClient._parse_version_tuple(v2)
        for a, b in itertools.zip_longest(t1, t2, fillvalue=0):
            if a < b:
                return -1
            if a > b:
                return 1
        return 0

    @staticmethod
    def _version_satisfies_constraints(version: str, constraints: str) -> bool:
        """Evaluate a version against a comma-separated list of constraints.

        Supported operators: >=, <=, >, <, ==, !=
        Example: ">=0.1.0,<0.3.0"
        """
        for raw in constraints.split(","):
            c = raw.strip()
            if not c:
                continue
            if c.startswith(">="):
                if MarketplaceClient._compare_versions(version, c[2:].strip()) < 0:
                    return False
            elif c.startswith("<="):
                if MarketplaceClient._compare_versions(version, c[2:].strip()) > 0:
                    return False
            elif c.startswith(">"):
                if MarketplaceClient._compare_versions(version, c[1:].strip()) <= 0:
                    return False
            elif c.startswith("<"):
                if MarketplaceClient._compare_versions(version, c[1:].strip()) >= 0:
                    return False
            elif c.startswith("=="):
                if MarketplaceClient._compare_versions(version, c[2:].strip()) != 0:
                    return False
            elif c.startswith("!="):
                if MarketplaceClient._compare_versions(version, c[2:].strip()) == 0:
                    return False
            else:
                # Bare version means equality
                if MarketplaceClient._compare_versions(version, c) != 0:
                    return False
        return True

    @staticmethod
    def _parse_dependency(dep: str) -> Tuple[str, Optional[str]]:
        """Parse a dependency spec like 'plugin-id@>=1.2.0,<2.0.0'."""
        if "@" in dep:
            pid, constraints = dep.split("@", 1)
            return pid.strip(), constraints.strip()
        return dep.strip(), None

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_rating: float = 0.0,
        compatible_version: Optional[str] = None,
        certified_only: bool = False,
        tags: Optional[Set[str]] = None,
        limit: int = 100,
        author: Optional[str] = None,
    ) -> List[PluginInfo]:
        """Search for plugins in the marketplace.

        Args:
            query: Search query string (searches name and description)
            category: Filter by category
            min_rating: Minimum rating threshold
            compatible_version: Filter by compatibility with Nethical version
            certified_only: Only return certified plugins
            tags: Filter by tags (plugins must have all specified tags)
            limit: Maximum number of results
            author: Filter by author

        Returns:
            List of matching plugins
        """
        limit = max(1, int(limit))
        tags = tags or set()

        with self._registry_lock:
            results: List[PluginInfo] = []

            for plugin in self._plugin_registry.values():
                # Apply filters
                if category and plugin.category != category:
                    continue
                if author and plugin.author != author:
                    continue
                if plugin.rating < min_rating:
                    continue
                if certified_only and not plugin.certified:
                    continue
                if tags and not tags.issubset(plugin.tags):
                    continue

                if compatible_version:
                    latest_version = plugin.get_latest_version()
                    if not latest_version or not latest_version.is_compatible(
                        compatible_version
                    ):
                        continue

                if query:
                    query_lower = query.lower()
                    if (
                        query_lower not in plugin.name.lower()
                        and query_lower not in plugin.description.lower()
                    ):
                        continue

                results.append(plugin)
                if len(results) >= limit:
                    break

            # Sort by rating and download count
            results.sort(key=lambda p: (p.rating, p.download_count), reverse=True)
            return results

    # -------------------------------------------------------------------------
    # Install / Uninstall / Update
    # -------------------------------------------------------------------------
    def install(
        self, plugin_id: str, version: Optional[str] = None, force: bool = False
    ) -> InstallStatus:
        """Install a plugin from the marketplace.

        Args:
            plugin_id: Unique plugin identifier
            version: Specific version to install (None for latest)
            force: Force reinstall even if already installed

        Returns:
            Installation status
        """
        lock = self._plugin_locks.setdefault(plugin_id, threading.RLock())
        with lock:
            logger.info(
                "Installing plugin '%s'%s", plugin_id, f"@{version}" if version else ""
            )
            plugin = self._get_plugin_or_raise(plugin_id)

            # Determine version to install
            if version is None:
                version = plugin.latest_version

            version_info = plugin.get_version(version)
            if not version_info:
                raise VersionNotFoundError(
                    f"Version '{version}' not found for plugin '{plugin_id}'"
                )

            # Check compatibility
            if not version_info.is_compatible(self.nethical_version):
                raise IncompatibleVersionError(
                    f"Plugin version '{version}' is not compatible with "
                    f"Nethical version '{self.nethical_version}'. "
                    f"Required: {version_info.compatibility}"
                )

            # Check if already installed and up-to-date
            current_status = self._get_install_status(plugin_id)
            if current_status == InstallStatus.INSTALLED and not force:
                # Verify installed_version matches requested version
                installed_version = self._get_installed_version(plugin_id)
                if installed_version == version and self._is_plugin_files_present(
                    plugin_id, installed_version
                ):
                    logger.info(
                        "Plugin '%s' is already installed at version %s",
                        plugin_id,
                        version,
                    )
                    return InstallStatus.INSTALLED

            # Install dependencies first
            for dep in plugin.dependencies:
                dep_id, dep_constraint = self._parse_dependency(dep)
                dep_plugin = self._plugin_registry.get(dep_id)
                if not dep_plugin:
                    raise PluginNotFoundError(
                        f"Dependency '{dep_id}' of plugin '{plugin_id}' not found in registry."
                    )

                # Pick a version that satisfies the constraint (default to latest)
                dep_version_to_install = dep_plugin.latest_version
                if dep_constraint:
                    # Prefer latest that satisfies
                    candidates = sorted(
                        [
                            v.version
                            for v in dep_plugin.versions
                            if self._version_satisfies_constraints(
                                v.version, dep_constraint
                            )
                        ],
                        key=lambda x: self._parse_version_tuple(x),
                        reverse=True,
                    )
                    if not candidates:
                        raise IncompatibleVersionError(
                            f"No versions of dependency '{dep_id}' satisfy constraint '{dep_constraint}'"
                        )
                    dep_version_to_install = candidates[0]

                self.install(dep_id, version=dep_version_to_install, force=False)

            # Create plugin directory
            plugin_dir = self.plugins_dir / plugin_id
            plugin_dir.mkdir(parents=True, exist_ok=True)

            # Download/extract if URL provided
            try:
                if version_info.download_url:
                    archive_path = self._download_to_cache(
                        version_info.download_url, plugin_id, version
                    )
                    self._verify_download(archive_path, version_info)
                    self._extract_archive_or_copy(archive_path, plugin_dir)
                else:
                    # Fallback: create a marker file so tests/local dev can proceed
                    logger.warning(
                        "No download_url for plugin '%s' version '%s'. Creating marker file.",
                        plugin_id,
                        version,
                    )
                    marker_file = plugin_dir / f".installed_{version}"
                    marker_file.write_text(
                        json.dumps(
                            {
                                "plugin_id": plugin_id,
                                "version": version,
                                "installed_at": datetime.now().isoformat(),
                            }
                        )
                    )

                # Update database
                self._update_install_metadata(
                    plugin_id, version, status=InstallStatus.INSTALLED
                )
                logger.info("Installed plugin '%s' version %s", plugin_id, version)
                return InstallStatus.INSTALLED

            except Exception as e:
                logger.exception("Failed installing plugin '%s': %s", plugin_id, e)
                self._update_install_metadata(
                    plugin_id, None, status=InstallStatus.FAILED
                )
                raise InstallationError(str(e)) from e

    def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin.

        Args:
            plugin_id: Plugin to uninstall

        Returns:
            True if successful
        """
        lock = self._plugin_locks.setdefault(plugin_id, threading.RLock())
        with lock:
            plugin_dir = self.plugins_dir / plugin_id
            try:
                if plugin_dir.exists():
                    shutil.rmtree(plugin_dir)
                    logger.info("Removed files for plugin '%s'", plugin_id)

                # Update database
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE plugins
                        SET installed_version = NULL,
                            install_status = ?
                        WHERE plugin_id = ?
                    """,
                        (InstallStatus.NOT_INSTALLED.value, plugin_id),
                    )

                logger.info("Uninstalled plugin '%s'", plugin_id)
                return True

            except Exception as e:
                logger.exception("Failed uninstalling plugin '%s': %s", plugin_id, e)
                raise UninstallationError(str(e)) from e

    def list_installed(self) -> List[PluginInfo]:
        """List all installed plugins.

        Returns:
            List of installed plugins
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT plugin_id, installed_version FROM plugins
                WHERE install_status = ?
            """,
                (InstallStatus.INSTALLED.value,),
            ).fetchall()

        installed: List[PluginInfo] = []
        with self._registry_lock:
            for row in rows:
                plugin_id = row["plugin_id"]
                # Optional: Validate files exist; if not, mark as not installed
                if not self._is_plugin_files_present(
                    plugin_id, row["installed_version"]
                ):
                    logger.warning(
                        "Plugin '%s' marked installed but files missing. Fixing status.",
                        plugin_id,
                    )
                    self._update_install_metadata(
                        plugin_id, None, status=InstallStatus.NOT_INSTALLED
                    )
                    continue

                if plugin_id in self._plugin_registry:
                    installed.append(self._plugin_registry[plugin_id])
        return installed

    def update(self, plugin_id: str) -> InstallStatus:
        """Update a plugin to its latest version.

        Args:
            plugin_id: Plugin to update

        Returns:
            Updated installation status
        """
        plugin = self._get_plugin_or_raise(plugin_id)
        logger.info(
            "Updating plugin '%s' to latest version %s",
            plugin_id,
            plugin.latest_version,
        )
        return self.install(plugin_id, version=plugin.latest_version, force=True)

    def check_updates(self) -> List[str]:
        """Check for available updates for installed plugins.

        Returns:
            List of plugin IDs that have updates available
        """
        updates_available: List[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT plugin_id, installed_version FROM plugins
                WHERE install_status = ?
            """,
                (InstallStatus.INSTALLED.value,),
            ).fetchall()

        with self._registry_lock:
            for row in rows:
                plugin_id, installed_version = (
                    row["plugin_id"],
                    row["installed_version"],
                )
                plugin = self._plugin_registry.get(plugin_id)
                if not plugin:
                    continue
                # Only suggest update if latest is greater than installed
                if plugin.latest_version and installed_version:
                    if (
                        self._compare_versions(plugin.latest_version, installed_version)
                        > 0
                    ):
                        updates_available.append(plugin_id)
        return updates_available

    # -------------------------------------------------------------------------
    # Registry management
    # -------------------------------------------------------------------------
    def register_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a new plugin in the local marketplace.

        This is typically used for testing or local plugin development.

        Args:
            plugin_info: Plugin information to register

        Returns:
            True if successful
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            # Insert or replace plugin
            cursor.execute(
                """
                INSERT OR REPLACE INTO plugins (
                    plugin_id, name, description, author, category,
                    rating, download_count, latest_version, license,
                    homepage, repository, certified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    plugin_info.plugin_id,
                    plugin_info.name,
                    plugin_info.description,
                    plugin_info.author,
                    plugin_info.category,
                    plugin_info.rating,
                    plugin_info.download_count,
                    plugin_info.latest_version,
                    plugin_info.license,
                    plugin_info.homepage,
                    plugin_info.repository,
                    int(plugin_info.certified),
                ),
            )

            # Replace tags
            cursor.execute(
                "DELETE FROM plugin_tags WHERE plugin_id = ?", (plugin_info.plugin_id,)
            )
            for tag in plugin_info.tags:
                cursor.execute(
                    """
                    INSERT INTO plugin_tags (plugin_id, tag) VALUES (?, ?)
                """,
                    (plugin_info.plugin_id, tag),
                )

            # Replace dependencies
            cursor.execute(
                "DELETE FROM plugin_dependencies WHERE plugin_id = ?",
                (plugin_info.plugin_id,),
            )
            for dep in plugin_info.dependencies:
                cursor.execute(
                    """
                    INSERT INTO plugin_dependencies (plugin_id, dependency) VALUES (?, ?)
                """,
                    (plugin_info.plugin_id, dep),
                )

            # Upsert versions
            for version in plugin_info.versions:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO plugin_versions (
                        plugin_id, version, release_date, compatibility,
                        changelog, download_url, checksum_sha256, size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        plugin_info.plugin_id,
                        version.version,
                        version.release_date.isoformat(),
                        version.compatibility,
                        version.changelog,
                        version.download_url,
                        version.checksum_sha256,
                        version.size_bytes,
                    ),
                )

            conn.commit()

        # Update in-memory cache
        with self._registry_lock:
            self._plugin_registry[plugin_info.plugin_id] = plugin_info

        logger.info(
            "Registered/updated plugin '%s' in local marketplace", plugin_info.plugin_id
        )
        return True

    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get detailed information about a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Plugin information or None if not found
        """
        with self._registry_lock:
            return self._plugin_registry.get(plugin_id)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _get_plugin_or_raise(self, plugin_id: str) -> PluginInfo:
        with self._registry_lock:
            plugin = self._plugin_registry.get(plugin_id)
        if not plugin:
            raise PluginNotFoundError(f"Plugin '{plugin_id}' not found in marketplace")
        return plugin

    def _get_install_status(self, plugin_id: str) -> InstallStatus:
        """Get installation status of a plugin from DB."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT install_status FROM plugins WHERE plugin_id = ?
            """,
                (plugin_id,),
            ).fetchone()

        if row and row["install_status"]:
            return InstallStatus(row["install_status"])
        return InstallStatus.NOT_INSTALLED

    def _get_installed_version(self, plugin_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT installed_version FROM plugins WHERE plugin_id = ?
            """,
                (plugin_id,),
            ).fetchone()
        return row["installed_version"] if row and row["installed_version"] else None

    def _is_plugin_files_present(self, plugin_id: str, version: Optional[str]) -> bool:
        plugin_dir = self.plugins_dir / plugin_id
        if not plugin_dir.exists():
            return False
        if version:
            # Either marker or a manifest/module presence
            marker = plugin_dir / f".installed_{version}"
            if marker.exists():
                return True
        # Heuristic: any python file, package, or manifest means presence
        if any(plugin_dir.glob("__init__.py")) or any(plugin_dir.rglob("*.py")):
            return True
        if (plugin_dir / "plugin.json").exists():
            return True
        return False

    def _update_install_metadata(
        self, plugin_id: str, version: Optional[str], status: InstallStatus
    ):
        with self._connect() as conn:
            now = datetime.now().isoformat()
            conn.execute(
                """
                UPDATE plugins
                SET installed_version = ?,
                    install_status = ?,
                    install_date = CASE WHEN ? IS NOT NULL AND ? != 'failed' THEN ? ELSE install_date END,
                    last_updated = ?
                WHERE plugin_id = ?
            """,
                (
                    version,
                    status.value,
                    version,
                    status.value,
                    now if version and status != InstallStatus.FAILED else None,
                    now,
                    plugin_id,
                ),
            )

    # -------------------------------------------------------------------------
    # Download/extract helpers
    # -------------------------------------------------------------------------
    def _download_to_cache(self, url: str, plugin_id: str, version: str) -> Path:
        """Download a file to the cache directory with retries."""
        # Validate URL scheme for security
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(
                f"Unsupported URL scheme: {parsed.scheme}. "
                f"Only http and https are allowed."
            )

        safe_name = f"{plugin_id}-{version}"
        dest = self.cache_dir / safe_name
        # If already downloaded, reuse
        if dest.exists() and dest.stat().st_size > 0:
            return dest

        last_error: Optional[Exception] = None
        for attempt in range(1, self.download_retries + 2):
            try:
                logger.info("Downloading %s (attempt %d)", url, attempt)
                with urllib.request.urlopen(
                    url, timeout=self.request_timeout_s
                ) as resp:
                    data = resp.read()
                dest.write_bytes(data)
                return dest
            except Exception as e:
                last_error = e
                logger.warning(
                    "Download failed (attempt %d/%d): %s",
                    attempt,
                    self.download_retries + 1,
                    e,
                )
                time.sleep(min(2.0 * attempt, 5.0))
        assert last_error is not None
        raise InstallationError(f"Failed to download {url}: {last_error}")

    def _verify_download(self, file_path: Path, version_info: PluginVersion):
        """Verify checksum and size if provided."""
        if version_info.size_bytes is not None:
            size = file_path.stat().st_size
            if size != version_info.size_bytes:
                raise InstallationError(
                    f"Downloaded size mismatch: expected {version_info.size_bytes}, got {size}"
                )
        if version_info.checksum_sha256:
            sha = hashlib.sha256()
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            digest = sha.hexdigest()
            if digest.lower() != version_info.checksum_sha256.lower():
                raise InstallationError(
                    f"Checksum mismatch: expected {version_info.checksum_sha256}, got {digest}"
                )

    def _extract_archive_or_copy(self, file_path: Path, dest_dir: Path):
        """Extract known archive types; otherwise copy as-is or create marker."""
        suffixes = "".join(file_path.suffixes).lower()
        try:
            if suffixes.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zf:
                    self._safe_extract_zip(zf, dest_dir)
            elif (
                suffixes.endswith(".tar.gz")
                or suffixes.endswith(".tgz")
                or suffixes.endswith(".tar")
            ):
                with tarfile.open(file_path, "r:*") as tf:
                    self._safe_extract_tar(tf, dest_dir)
            else:
                # Non-archive: copy file
                (dest_dir / file_path.name).write_bytes(file_path.read_bytes())
        except Exception as e:
            raise InstallationError(
                f"Failed to extract archive '{file_path.name}': {e}"
            ) from e

    def _safe_extract_zip(self, zf: zipfile.ZipFile, path: Path):
        """Safely extract zip archive, preventing directory traversal attacks."""
        for member in zf.namelist():
            abs_dest = (path / member).resolve()
            if not str(abs_dest).startswith(str(path.resolve())):
                raise InstallationError(
                    f"Archive member '{member}' would extract outside of {path}"
                )
            zf.extract(member, path)

    def _safe_extract_tar(self, tar: tarfile.TarFile, path: Path):
        """Safely extract tar archive, preventing directory traversal attacks."""
        for member in tar.getmembers():
            path / member.name
            abs_dest = (path / member.name).resolve()
            if not str(abs_dest).startswith(str(path.resolve())):
                raise InstallationError(
                    f"Archive member '{member.name}' would extract outside of {path}"
                )
            if member.islnk() or member.issym():
                raise InstallationError(
                    f"Archive member '{member.name}' is a link (not allowed)"
                )
            tar.extract(member, path)

    # -------------------------------------------------------------------------
    # Optional: plugin dynamic loading utilities
    # -------------------------------------------------------------------------
    def load_plugin_module(self, plugin_id: str, module_name: Optional[str] = None):
        """Dynamically load a plugin's module from its directory.

        Args:
            plugin_id: Plugin identifier
            module_name: Optional module within the plugin directory (default: plugin_id)

        Returns:
            Loaded module object

        Raises:
            FileNotFoundError if the module file cannot be found.
        """
        plugin_dir = self.plugins_dir / plugin_id
        if not plugin_dir.exists():
            raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")

        target_module = module_name or plugin_id
        module_file = plugin_dir / f"{target_module}.py"
        package_init = plugin_dir / "__init__.py"

        if module_file.exists():
            spec = importlib.util.spec_from_file_location(
                target_module, str(module_file)
            )
        elif package_init.exists():
            spec = importlib.util.spec_from_file_location(plugin_id, str(package_init))
        else:
            raise FileNotFoundError(
                f"No module file found for plugin '{plugin_id}' (looked for {module_file} or package)"
            )

        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[attribute-defined-outside-init]
        sys.modules[target_module] = module
        return module

    # -------------------------------------------------------------------------
    # Public convenience: list available plugins
    # -------------------------------------------------------------------------
    def list_available(self) -> List[str]:
        """List all plugin IDs available in the local marketplace registry."""
        with self._registry_lock:
            return sorted(self._plugin_registry.keys())
