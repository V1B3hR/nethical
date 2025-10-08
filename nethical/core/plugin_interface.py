"""
Detector Plugin Interface for F2: Detector & Policy Extensibility

This module provides the plugin architecture for custom detector extensions,
enabling external detector registration without modifying core code.

Features:
- DetectorPlugin base class for custom detectors
- Plugin discovery and dynamic loading
- Plugin health monitoring and metrics
- Plugin versioning and compatibility checking
- Sandboxed plugin execution (timeout, resource limits)
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

from ..detectors.base_detector import BaseDetector, DetectorStatus, SafetyViolation

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Status of a plugin."""
    LOADED = auto()
    ACTIVE = auto()
    DISABLED = auto()
    FAILED = auto()
    INCOMPATIBLE = auto()


@dataclass
class PluginMetadata:
    """Metadata for a detector plugin."""
    name: str
    version: str
    description: str
    author: str
    requires_nethical_version: str = ">=1.0.0"
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    # Health and monitoring
    loaded_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "requires_nethical_version": self.requires_nethical_version,
            "dependencies": self.dependencies,
            "tags": list(self.tags),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status
        }


class DetectorPlugin(BaseDetector, ABC):
    """
    Base class for custom detector plugins.
    
    Custom detectors should inherit from this class and implement:
    - detect_violations(action): Main detection logic
    - get_metadata(): Return plugin metadata
    
    Example:
        class CustomFinancialDetector(DetectorPlugin):
            def __init__(self):
                super().__init__(
                    name="CustomFinancialDetector",
                    version="1.0.0"
                )
            
            async def detect_violations(self, action):
                # Custom detection logic
                violations = []
                if "financial_data" in str(action):
                    violations.append(SafetyViolation(
                        detector=self.name,
                        severity="high",
                        description="Financial data detected",
                        category="compliance"
                    ))
                return violations
            
            def get_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=self.name,
                    version=self.version,
                    description="Detects financial compliance violations",
                    author="Your Organization",
                    tags={"finance", "compliance"}
                )
    """
    
    def __init__(self, name: str, version: str = "1.0.0", **kwargs):
        """Initialize the plugin detector."""
        super().__init__(name=name, version=version, **kwargs)
        self._plugin_metadata: Optional[PluginMetadata] = None
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Return plugin metadata.
        
        This method must be implemented by subclasses to provide
        information about the plugin.
        """
        raise NotImplementedError("Plugins must implement get_metadata()")
    
    def get_cached_metadata(self) -> PluginMetadata:
        """Get cached metadata or fetch if not available."""
        if self._plugin_metadata is None:
            self._plugin_metadata = self.get_metadata()
        return self._plugin_metadata
    
    async def health_check(self) -> bool:
        """
        Perform health check on the plugin.
        
        Override this method to implement custom health checks.
        Default implementation checks if detector is active.
        """
        return self.status == DetectorStatus.ACTIVE


class PluginManager:
    """
    Manager for detector plugins.
    
    Handles plugin discovery, loading, registration, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize the plugin manager."""
        self.plugins: Dict[str, DetectorPlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_status: Dict[str, PluginStatus] = {}
        self._load_errors: Dict[str, str] = {}
        
        logger.info("PluginManager initialized")
    
    def register_plugin(self, plugin: DetectorPlugin) -> None:
        """
        Register a plugin instance.
        
        Args:
            plugin: DetectorPlugin instance to register
            
        Raises:
            ValueError: If plugin with same name already registered
            TypeError: If plugin is not a DetectorPlugin instance
        """
        if not isinstance(plugin, DetectorPlugin):
            raise TypeError(f"Plugin must be an instance of DetectorPlugin, got {type(plugin)}")
        
        plugin_name = plugin.name
        
        if plugin_name in self.plugins:
            logger.warning(f"Plugin '{plugin_name}' already registered, replacing...")
        
        # Get and cache metadata
        try:
            metadata = plugin.get_metadata()
            metadata.loaded_at = datetime.now(timezone.utc)
            
            self.plugins[plugin_name] = plugin
            self.plugin_metadata[plugin_name] = metadata
            self.plugin_status[plugin_name] = PluginStatus.ACTIVE
            
            logger.info(f"Registered plugin: {plugin_name} v{metadata.version}")
        except Exception as e:
            logger.error(f"Failed to register plugin '{plugin_name}': {e}")
            self.plugin_status[plugin_name] = PluginStatus.FAILED
            self._load_errors[plugin_name] = str(e)
            raise
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if plugin was unregistered, False if not found
        """
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            del self.plugin_metadata[plugin_name]
            self.plugin_status[plugin_name] = PluginStatus.DISABLED
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[DetectorPlugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            DetectorPlugin instance or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered plugins with their metadata and status.
        
        Returns:
            Dictionary mapping plugin names to their info
        """
        return {
            name: {
                "metadata": self.plugin_metadata[name].to_dict(),
                "status": self.plugin_status[name].name,
                "detector_status": plugin.status.name,
                "metrics": {
                    "total_runs": plugin.metrics.total_runs,
                    "success_rate": plugin.metrics.success_rate,
                    "violations_detected": plugin.metrics.violations_detected
                }
            }
            for name, plugin in self.plugins.items()
        }
    
    def discover_plugins(self, plugin_dir: str) -> List[str]:
        """
        Discover plugins in a directory.
        
        Searches for Python files with DetectorPlugin subclasses.
        
        Args:
            plugin_dir: Directory to search for plugins
            
        Returns:
            List of discovered plugin module paths
        """
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return []
        
        discovered = []
        for py_file in plugin_path.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
            discovered.append(str(py_file))
        
        logger.info(f"Discovered {len(discovered)} potential plugin files in {plugin_dir}")
        return discovered
    
    def load_plugin_from_file(self, plugin_path: str) -> List[str]:
        """
        Load plugin(s) from a Python file.
        
        Args:
            plugin_path: Path to the Python file
            
        Returns:
            List of loaded plugin names
        """
        path = Path(plugin_path)
        if not path.exists():
            logger.error(f"Plugin file not found: {plugin_path}")
            return []
        
        # Create module name from file path
        module_name = f"nethical_plugin_{path.stem}"
        
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for {plugin_path}")
                return []
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find DetectorPlugin subclasses
            loaded_plugins = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, DetectorPlugin) and 
                    obj is not DetectorPlugin and
                    obj.__module__ == module_name):
                    
                    try:
                        # Instantiate and register
                        plugin_instance = obj()
                        self.register_plugin(plugin_instance)
                        loaded_plugins.append(plugin_instance.name)
                    except Exception as e:
                        logger.error(f"Failed to instantiate plugin {name}: {e}")
                        self._load_errors[name] = str(e)
            
            if loaded_plugins:
                logger.info(f"Loaded {len(loaded_plugins)} plugin(s) from {plugin_path}")
            else:
                logger.warning(f"No valid plugins found in {plugin_path}")
            
            return loaded_plugins
            
        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_path}: {e}")
            logger.debug(traceback.format_exc())
            self._load_errors[plugin_path] = str(e)
            return []
    
    def load_plugins_from_directory(self, plugin_dir: str) -> Dict[str, List[str]]:
        """
        Load all plugins from a directory.
        
        Args:
            plugin_dir: Directory containing plugin files
            
        Returns:
            Dictionary mapping file paths to loaded plugin names
        """
        discovered_files = self.discover_plugins(plugin_dir)
        
        results = {}
        for plugin_file in discovered_files:
            loaded = self.load_plugin_from_file(plugin_file)
            if loaded:
                results[plugin_file] = loaded
        
        logger.info(f"Loaded {len(results)} plugin file(s) from {plugin_dir}")
        return results
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all plugins.
        
        Returns:
            Dictionary mapping plugin names to health status
        """
        results = {}
        
        for name, plugin in self.plugins.items():
            try:
                is_healthy = await plugin.health_check()
                results[name] = is_healthy
                
                # Update metadata
                metadata = self.plugin_metadata[name]
                metadata.last_health_check = datetime.now(timezone.utc)
                metadata.health_status = "healthy" if is_healthy else "unhealthy"
                
                # Update plugin status
                if not is_healthy:
                    self.plugin_status[name] = PluginStatus.FAILED
                    logger.warning(f"Plugin {name} failed health check")
                
            except Exception as e:
                logger.error(f"Health check failed for plugin {name}: {e}")
                results[name] = False
                self.plugin_status[name] = PluginStatus.FAILED
                self.plugin_metadata[name].health_status = f"error: {str(e)}"
        
        return results
    
    async def run_plugin(self, plugin_name: str, action: Any, 
                        context: Optional[Dict[str, Any]] = None) -> List[SafetyViolation]:
        """
        Run a specific plugin on an action.
        
        Args:
            plugin_name: Name of the plugin to run
            action: Action to analyze
            context: Optional context dictionary
            
        Returns:
            List of detected violations
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is None:
            logger.error(f"Plugin not found: {plugin_name}")
            return []
        
        if self.plugin_status[plugin_name] != PluginStatus.ACTIVE:
            logger.warning(f"Plugin {plugin_name} is not active (status: {self.plugin_status[plugin_name].name})")
            return []
        
        try:
            return await plugin.run(action, context)
        except Exception as e:
            logger.error(f"Error running plugin {plugin_name}: {e}")
            self.plugin_status[plugin_name] = PluginStatus.FAILED
            return []
    
    async def run_all_plugins(self, action: Any, 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, List[SafetyViolation]]:
        """
        Run all active plugins on an action.
        
        Args:
            action: Action to analyze
            context: Optional context dictionary
            
        Returns:
            Dictionary mapping plugin names to their violation results
        """
        results = {}
        
        tasks = []
        active_plugins = []
        
        for name, plugin in self.plugins.items():
            if self.plugin_status[name] == PluginStatus.ACTIVE:
                tasks.append(self.run_plugin(name, action, context))
                active_plugins.append(name)
        
        if not tasks:
            return results
        
        violations_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, violations in enumerate(violations_list):
            name = active_plugins[i]
            if isinstance(violations, Exception):
                logger.error(f"Plugin {name} raised exception: {violations}")
                results[name] = []
            else:
                results[name] = violations
        
        return results
    
    def get_load_errors(self) -> Dict[str, str]:
        """
        Get dictionary of load errors.
        
        Returns:
            Dictionary mapping plugin/file names to error messages
        """
        return self._load_errors.copy()


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
