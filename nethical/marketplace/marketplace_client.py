"""Marketplace Client for Plugin Management.

This module provides the MarketplaceClient for searching, installing,
and managing plugins from the Nethical ecosystem.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from datetime import datetime
import json
import sqlite3
from pathlib import Path
import shutil
import importlib.util
import sys


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
    compatibility: str  # Compatible Nethical version range
    changelog: str = ""
    download_url: Optional[str] = None
    
    def is_compatible(self, nethical_version: str) -> bool:
        """Check if this version is compatible with given Nethical version."""
        # Simple compatibility check - can be enhanced with version parsing
        if ">=" in self.compatibility:
            min_version = self.compatibility.split(">=")[1].strip()
            return self._compare_versions(nethical_version, min_version) >= 0
        elif "==" in self.compatibility:
            exact_version = self.compatibility.split("==")[1].strip()
            return nethical_version == exact_version
        return True
    
    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare two semantic versions. Returns -1, 0, or 1."""
        parts1 = [int(x) for x in v1.split('.')]
        parts2 = [int(x) for x in v2.split('.')]
        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1
        return 0


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
    dependencies: List[str] = field(default_factory=list)
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
        nethical_version: str = "0.1.0"
    ):
        """Initialize marketplace client.
        
        Args:
            storage_dir: Directory for plugin storage and cache
            marketplace_url: URL of marketplace API (None for local/mock mode)
            nethical_version: Current Nethical version for compatibility checks
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins_dir = self.storage_dir / "plugins"
        self.plugins_dir.mkdir(exist_ok=True)
        
        self.cache_dir = self.storage_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.marketplace_url = marketplace_url
        self.nethical_version = nethical_version
        
        # Initialize local registry database
        self.db_path = self.storage_dir / "registry.db"
        self._init_database()
        
        # In-memory plugin registry cache
        self._plugin_registry: Dict[str, PluginInfo] = {}
        self._load_registry_cache()
    
    def _init_database(self):
        """Initialize SQLite database for plugin registry."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Plugins table
        cursor.execute("""
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
        """)
        
        # Plugin versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plugin_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plugin_id TEXT NOT NULL,
                version TEXT NOT NULL,
                release_date TEXT,
                compatibility TEXT,
                changelog TEXT,
                download_url TEXT,
                FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id),
                UNIQUE(plugin_id, version)
            )
        """)
        
        # Plugin tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plugin_tags (
                plugin_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (plugin_id, tag),
                FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id)
            )
        """)
        
        # Plugin dependencies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plugin_dependencies (
                plugin_id TEXT NOT NULL,
                dependency TEXT NOT NULL,
                PRIMARY KEY (plugin_id, dependency),
                FOREIGN KEY (plugin_id) REFERENCES plugins(plugin_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_registry_cache(self):
        """Load plugin registry from database into memory."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Load basic plugin info
        cursor.execute("SELECT * FROM plugins")
        for row in cursor.fetchall():
            plugin_id = row[0]
            
            # Load tags
            cursor.execute("SELECT tag FROM plugin_tags WHERE plugin_id = ?", (plugin_id,))
            tags = {tag[0] for tag in cursor.fetchall()}
            
            # Load dependencies
            cursor.execute("SELECT dependency FROM plugin_dependencies WHERE plugin_id = ?", (plugin_id,))
            dependencies = [dep[0] for dep in cursor.fetchall()]
            
            # Load versions
            cursor.execute("SELECT * FROM plugin_versions WHERE plugin_id = ?", (plugin_id,))
            versions = []
            for v_row in cursor.fetchall():
                versions.append(PluginVersion(
                    version=v_row[2],
                    release_date=datetime.fromisoformat(v_row[3]) if v_row[3] else datetime.now(),
                    compatibility=v_row[4] or ">=0.1.0",
                    changelog=v_row[5] or "",
                    download_url=v_row[6]
                ))
            
            self._plugin_registry[plugin_id] = PluginInfo(
                plugin_id=plugin_id,
                name=row[1],
                description=row[2] or "",
                author=row[3] or "",
                category=row[4] or "",
                rating=row[5] or 0.0,
                download_count=row[6] or 0,
                latest_version=row[7] or "0.1.0",
                license=row[8] or "MIT",
                homepage=row[9],
                repository=row[10],
                certified=bool(row[11]),
                tags=tags,
                dependencies=dependencies,
                versions=versions
            )
        
        conn.close()
    
    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_rating: float = 0.0,
        compatible_version: Optional[str] = None,
        certified_only: bool = False,
        tags: Optional[Set[str]] = None,
        limit: int = 100
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
            
        Returns:
            List of matching plugins
        """
        results = []
        
        for plugin in self._plugin_registry.values():
            # Apply filters
            if category and plugin.category != category:
                continue
            
            if plugin.rating < min_rating:
                continue
            
            if certified_only and not plugin.certified:
                continue
            
            if tags and not tags.issubset(plugin.tags):
                continue
            
            if compatible_version:
                latest_version = plugin.get_latest_version()
                if not latest_version or not latest_version.is_compatible(compatible_version):
                    continue
            
            if query:
                query_lower = query.lower()
                if (query_lower not in plugin.name.lower() and 
                    query_lower not in plugin.description.lower()):
                    continue
            
            results.append(plugin)
            
            if len(results) >= limit:
                break
        
        # Sort by rating and download count
        results.sort(key=lambda p: (p.rating, p.download_count), reverse=True)
        return results
    
    def install(
        self,
        plugin_id: str,
        version: Optional[str] = None,
        force: bool = False
    ) -> InstallStatus:
        """Install a plugin from the marketplace.
        
        Args:
            plugin_id: Unique plugin identifier
            version: Specific version to install (None for latest)
            force: Force reinstall even if already installed
            
        Returns:
            Installation status
        """
        # Get plugin info
        plugin = self._plugin_registry.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_id}' not found in marketplace")
        
        # Determine version to install
        if version is None:
            version = plugin.latest_version
        
        version_info = plugin.get_version(version)
        if not version_info:
            raise ValueError(f"Version '{version}' not found for plugin '{plugin_id}'")
        
        # Check compatibility
        if not version_info.is_compatible(self.nethical_version):
            raise ValueError(
                f"Plugin version '{version}' is not compatible with "
                f"Nethical version '{self.nethical_version}'. "
                f"Required: {version_info.compatibility}"
            )
        
        # Check if already installed
        current_status = self._get_install_status(plugin_id)
        if current_status == InstallStatus.INSTALLED and not force:
            return InstallStatus.INSTALLED
        
        # Install dependencies first
        for dep in plugin.dependencies:
            dep_status = self._get_install_status(dep)
            if dep_status != InstallStatus.INSTALLED:
                self.install(dep)
        
        # Create plugin directory
        plugin_dir = self.plugins_dir / plugin_id
        plugin_dir.mkdir(exist_ok=True)
        
        # In a real implementation, this would download and extract the plugin
        # For now, we'll create a marker file
        marker_file = plugin_dir / f".installed_{version}"
        marker_file.touch()
        
        # Update database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE plugins 
            SET installed_version = ?,
                install_status = ?,
                install_date = ?,
                last_updated = ?
            WHERE plugin_id = ?
        """, (version, InstallStatus.INSTALLED.value, 
              datetime.now().isoformat(), datetime.now().isoformat(), plugin_id))
        conn.commit()
        conn.close()
        
        return InstallStatus.INSTALLED
    
    def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin.
        
        Args:
            plugin_id: Plugin to uninstall
            
        Returns:
            True if successful
        """
        plugin_dir = self.plugins_dir / plugin_id
        if plugin_dir.exists():
            shutil.rmtree(plugin_dir)
        
        # Update database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE plugins 
            SET installed_version = NULL,
                install_status = ?
            WHERE plugin_id = ?
        """, (InstallStatus.NOT_INSTALLED.value, plugin_id))
        conn.commit()
        conn.close()
        
        return True
    
    def list_installed(self) -> List[PluginInfo]:
        """List all installed plugins.
        
        Returns:
            List of installed plugins
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT plugin_id FROM plugins 
            WHERE install_status = ?
        """, (InstallStatus.INSTALLED.value,))
        
        installed = []
        for row in cursor.fetchall():
            plugin_id = row[0]
            if plugin_id in self._plugin_registry:
                installed.append(self._plugin_registry[plugin_id])
        
        conn.close()
        return installed
    
    def _get_install_status(self, plugin_id: str) -> InstallStatus:
        """Get installation status of a plugin."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT install_status FROM plugins WHERE plugin_id = ?
        """, (plugin_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return InstallStatus(row[0])
        return InstallStatus.NOT_INSTALLED
    
    def update(self, plugin_id: str) -> InstallStatus:
        """Update a plugin to its latest version.
        
        Args:
            plugin_id: Plugin to update
            
        Returns:
            Updated installation status
        """
        plugin = self._plugin_registry.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_id}' not found")
        
        return self.install(plugin_id, version=plugin.latest_version, force=True)
    
    def check_updates(self) -> List[str]:
        """Check for available updates for installed plugins.
        
        Returns:
            List of plugin IDs that have updates available
        """
        updates_available = []
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT plugin_id, installed_version FROM plugins 
            WHERE install_status = ?
        """, (InstallStatus.INSTALLED.value,))
        
        for row in cursor.fetchall():
            plugin_id, installed_version = row
            plugin = self._plugin_registry.get(plugin_id)
            if plugin and plugin.latest_version != installed_version:
                updates_available.append(plugin_id)
        
        conn.close()
        return updates_available
    
    def register_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a new plugin in the local marketplace.
        
        This is typically used for testing or local plugin development.
        
        Args:
            plugin_info: Plugin information to register
            
        Returns:
            True if successful
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Insert or replace plugin
        cursor.execute("""
            INSERT OR REPLACE INTO plugins (
                plugin_id, name, description, author, category,
                rating, download_count, latest_version, license,
                homepage, repository, certified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
            int(plugin_info.certified)
        ))
        
        # Insert tags
        cursor.execute("DELETE FROM plugin_tags WHERE plugin_id = ?", (plugin_info.plugin_id,))
        for tag in plugin_info.tags:
            cursor.execute("""
                INSERT INTO plugin_tags (plugin_id, tag) VALUES (?, ?)
            """, (plugin_info.plugin_id, tag))
        
        # Insert dependencies
        cursor.execute("DELETE FROM plugin_dependencies WHERE plugin_id = ?", (plugin_info.plugin_id,))
        for dep in plugin_info.dependencies:
            cursor.execute("""
                INSERT INTO plugin_dependencies (plugin_id, dependency) VALUES (?, ?)
            """, (plugin_info.plugin_id, dep))
        
        # Insert versions
        for version in plugin_info.versions:
            cursor.execute("""
                INSERT OR REPLACE INTO plugin_versions (
                    plugin_id, version, release_date, compatibility,
                    changelog, download_url
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                plugin_info.plugin_id,
                version.version,
                version.release_date.isoformat(),
                version.compatibility,
                version.changelog,
                version.download_url
            ))
        
        conn.commit()
        conn.close()
        
        # Update in-memory cache
        self._plugin_registry[plugin_info.plugin_id] = plugin_info
        
        return True
    
    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get detailed information about a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin information or None if not found
        """
        return self._plugin_registry.get(plugin_id)
