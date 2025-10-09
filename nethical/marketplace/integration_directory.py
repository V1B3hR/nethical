"""Integration Directory for Third-Party Systems.

This module provides adapters and utilities for integrating
with external systems, data sources, and APIs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import json


class IntegrationType(Enum):
    """Integration type."""
    DATA_SOURCE = "data_source"
    API_CONNECTOR = "api_connector"
    EXPORT = "export"
    IMPORT = "import"
    WEBHOOK = "webhook"


@dataclass
class IntegrationMetadata:
    """Integration metadata."""
    integration_id: str
    name: str
    description: str
    integration_type: IntegrationType
    version: str
    supported_formats: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)


class IntegrationAdapter(ABC):
    """Base class for integration adapters."""
    
    def __init__(self, integration_id: str, config: Dict[str, Any]):
        """Initialize adapter.
        
        Args:
            integration_id: Integration identifier
            config: Configuration dictionary
        """
        self.integration_id = integration_id
        self.config = config
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to external system."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from external system."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to external system."""
        pass


class DataSourceAdapter(IntegrationAdapter):
    """Adapter for external data sources.
    
    Example:
        >>> adapter = DataSourceAdapter("my-db", {"host": "localhost"})
        >>> adapter.connect()
        >>> data = adapter.fetch_data({"query": "SELECT * FROM actions"})
    """
    
    def connect(self) -> bool:
        """Connect to data source."""
        # Placeholder implementation
        return True
    
    def disconnect(self):
        """Disconnect from data source."""
        pass
    
    def test_connection(self) -> bool:
        """Test connection to data source."""
        return True
    
    def fetch_data(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data from source.
        
        Args:
            query: Query parameters
            
        Returns:
            List of data records
        """
        # Placeholder implementation
        return []
    
    def write_data(self, data: List[Dict[str, Any]]) -> bool:
        """Write data to source.
        
        Args:
            data: Data records to write
            
        Returns:
            True if successful
        """
        # Placeholder implementation
        return True


class ExportUtility:
    """Utility for exporting governance data.
    
    Example:
        >>> exporter = ExportUtility()
        >>> exporter.export_to_json(data, "output.json")
        >>> exporter.export_to_csv(data, "output.csv")
    """
    
    def __init__(self):
        """Initialize export utility."""
        self.supported_formats = ["json", "csv", "xml", "yaml"]
    
    def export_to_json(self, data: Any, filepath: str) -> bool:
        """Export data to JSON.
        
        Args:
            data: Data to export
            filepath: Output file path
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def export_to_csv(self, data: List[Dict], filepath: str) -> bool:
        """Export data to CSV.
        
        Args:
            data: List of dictionaries to export
            filepath: Output file path
            
        Returns:
            True if successful
        """
        try:
            import csv
            if not data:
                return True
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def export_to_format(self, data: Any, filepath: str, format: str) -> bool:
        """Export data to specified format.
        
        Args:
            data: Data to export
            filepath: Output file path
            format: Export format (json, csv, etc.)
            
        Returns:
            True if successful
        """
        if format == "json":
            return self.export_to_json(data, filepath)
        elif format == "csv":
            return self.export_to_csv(data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")


class ImportUtility:
    """Utility for importing governance data.
    
    Example:
        >>> importer = ImportUtility()
        >>> data = importer.import_from_json("input.json")
        >>> data = importer.import_from_csv("input.csv")
    """
    
    def __init__(self):
        """Initialize import utility."""
        self.supported_formats = ["json", "csv", "xml", "yaml"]
    
    def import_from_json(self, filepath: str) -> Any:
        """Import data from JSON.
        
        Args:
            filepath: Input file path
            
        Returns:
            Imported data
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Import error: {e}")
            return None
    
    def import_from_csv(self, filepath: str) -> List[Dict]:
        """Import data from CSV.
        
        Args:
            filepath: Input file path
            
        Returns:
            List of dictionaries
        """
        try:
            import csv
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            print(f"Import error: {e}")
            return []
    
    def import_from_format(self, filepath: str, format: str) -> Any:
        """Import data from specified format.
        
        Args:
            filepath: Input file path
            format: Import format (json, csv, etc.)
            
        Returns:
            Imported data
        """
        if format == "json":
            return self.import_from_json(filepath)
        elif format == "csv":
            return self.import_from_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")


class IntegrationDirectory:
    """Directory of available integrations.
    
    This class maintains a registry of available integrations
    and provides factory methods for creating adapters.
    
    Example:
        >>> directory = IntegrationDirectory()
        >>> directory.register_integration(metadata, adapter_class)
        >>> adapter = directory.create_adapter("my-integration", config)
    """
    
    def __init__(self):
        """Initialize integration directory."""
        self._integrations: Dict[str, IntegrationMetadata] = {}
        self._adapter_factories: Dict[str, Callable] = {}
        self._initialize_default_integrations()
    
    def _initialize_default_integrations(self):
        """Initialize default integrations."""
        # PostgreSQL data source
        self.register_integration(
            IntegrationMetadata(
                integration_id="postgresql",
                name="PostgreSQL Data Source",
                description="Connect to PostgreSQL databases",
                integration_type=IntegrationType.DATA_SOURCE,
                version="1.0.0",
                supported_formats=["sql"],
                configuration_schema={
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "default": 5432},
                    "database": {"type": "string", "required": True},
                    "username": {"type": "string", "required": True},
                    "password": {"type": "string", "required": True}
                }
            ),
            DataSourceAdapter
        )
        
        # MongoDB data source
        self.register_integration(
            IntegrationMetadata(
                integration_id="mongodb",
                name="MongoDB Data Source",
                description="Connect to MongoDB databases",
                integration_type=IntegrationType.DATA_SOURCE,
                version="1.0.0",
                supported_formats=["json"],
                configuration_schema={
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "default": 27017},
                    "database": {"type": "string", "required": True}
                }
            ),
            DataSourceAdapter
        )
    
    def register_integration(
        self,
        metadata: IntegrationMetadata,
        adapter_class: type
    ):
        """Register an integration.
        
        Args:
            metadata: Integration metadata
            adapter_class: Adapter class
        """
        self._integrations[metadata.integration_id] = metadata
        self._adapter_factories[metadata.integration_id] = adapter_class
    
    def create_adapter(
        self,
        integration_id: str,
        config: Dict[str, Any]
    ) -> Optional[IntegrationAdapter]:
        """Create an integration adapter.
        
        Args:
            integration_id: Integration identifier
            config: Configuration dictionary
            
        Returns:
            Integration adapter or None
        """
        factory = self._adapter_factories.get(integration_id)
        if factory:
            return factory(integration_id, config)
        return None
    
    def list_integrations(
        self,
        integration_type: Optional[IntegrationType] = None
    ) -> List[IntegrationMetadata]:
        """List available integrations.
        
        Args:
            integration_type: Filter by type
            
        Returns:
            List of integration metadata
        """
        if integration_type:
            return [
                meta for meta in self._integrations.values()
                if meta.integration_type == integration_type
            ]
        return list(self._integrations.values())
    
    def get_integration(self, integration_id: str) -> Optional[IntegrationMetadata]:
        """Get integration metadata.
        
        Args:
            integration_id: Integration identifier
            
        Returns:
            Integration metadata or None
        """
        return self._integrations.get(integration_id)
