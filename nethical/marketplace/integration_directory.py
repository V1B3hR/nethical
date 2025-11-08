"""Integration Directory for Third-Party Systems.

This module provides adapters and utilities for integrating
with external systems, data sources, and APIs.

Enhancements:
- Stronger typing and docstrings
- Thread-safe integration registry
- Config validation using configuration_schema (basic and jsonschema-if-available)
- Optional YAML and XML import/export support (graceful fallback)
- Structured error classes
- Logging across adapters and utilities
- Additional base adapters (APIConnectorAdapter, WebhookAdapter)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Type, Union
from enum import Enum
from abc import ABC, abstractmethod
import json
import logging
import threading

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic logger setup; the host application can override this.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


# ======================
# Error definitions
# ======================


class IntegrationError(Exception):
    """Base class for integration-related errors."""


class IntegrationConfigError(IntegrationError):
    """Raised when configuration validation fails."""


class IntegrationConnectionError(IntegrationError):
    """Raised when connection or connectivity checks fail."""


# ======================
# Types and metadata
# ======================


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================
# Base Adapters
# ======================


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
        self._connected = False
        logger.debug(
            "Initialized adapter %s with config keys: %s", integration_id, list(config.keys())
        )

    def __enter__(self):
        if not self.connect():
            raise IntegrationConnectionError(f"Failed to connect: {self.integration_id}")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning("Error during disconnect for %s: %s", self.integration_id, e)

    @abstractmethod
    def connect(self) -> bool:
        """Connect to external system."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        """Disconnect from external system."""
        raise NotImplementedError

    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to external system."""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _mark_connected(self, value: bool):
        self._connected = value


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
        logger.info("Connecting to data source: %s", self.integration_id)
        self._mark_connected(True)
        return True

    def disconnect(self):
        """Disconnect from data source."""
        logger.info("Disconnecting data source: %s", self.integration_id)
        self._mark_connected(False)

    def test_connection(self) -> bool:
        """Test connection to data source."""
        logger.debug("Testing connection for data source: %s", self.integration_id)
        return True

    def fetch_data(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data from source.

        Args:
            query: Query parameters

        Returns:
            List of data records
        """
        logger.debug("Fetching data from %s with query: %s", self.integration_id, query)
        # Placeholder implementation
        return []

    def write_data(self, data: List[Dict[str, Any]]) -> bool:
        """Write data to source.

        Args:
            data: Data records to write

        Returns:
            True if successful
        """
        logger.debug("Writing %d records to %s", len(data), self.integration_id)
        # Placeholder implementation
        return True


class APIConnectorAdapter(IntegrationAdapter):
    """Base adapter for API connectors."""

    @abstractmethod
    def request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make an API request to the external system."""
        raise NotImplementedError

    def connect(self) -> bool:
        logger.info("Initializing API connector: %s", self.integration_id)
        self._mark_connected(True)
        return True

    def disconnect(self):
        logger.info("Tearing down API connector: %s", self.integration_id)
        self._mark_connected(False)

    def test_connection(self) -> bool:
        logger.debug("Testing API connectivity for: %s", self.integration_id)
        return True


class WebhookAdapter(IntegrationAdapter):
    """Base adapter for webhook handlers."""

    @abstractmethod
    def handle_event(self, event: Dict[str, Any]) -> None:
        """Handle a webhook event."""
        raise NotImplementedError

    def connect(self) -> bool:
        logger.info("Preparing webhook adapter: %s", self.integration_id)
        self._mark_connected(True)
        return True

    def disconnect(self):
        logger.info("Stopping webhook adapter: %s", self.integration_id)
        self._mark_connected(False)

    def test_connection(self) -> bool:
        logger.debug("Webhook adapter connectivity (no-op): %s", self.integration_id)
        return True


# ======================
# Export/Import Utilities
# ======================


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


class ExportUtility:
    """Utility for exporting governance data.

    Example:
        >>> exporter = ExportUtility()
        >>> exporter.export_to_json(data, "output.json")
        >>> exporter.export_to_csv(data, "output.csv")
        >>> exporter.export_to_yaml(data, "output.yaml")  # if PyYAML installed
        >>> exporter.export_to_xml(data, "output.xml")    # generic XML
    """

    def __init__(self):
        """Initialize export utility."""
        self.supported_formats = ["json", "csv", "xml", "yaml"]

    def export_to_json(self, data: Any, filepath: str, encoding: str = "utf-8") -> bool:
        """Export data to JSON."""
        try:
            with open(filepath, "w", encoding=encoding) as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            logger.info("Exported JSON to %s", filepath)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Export JSON error: %s", e, exc_info=True)
            return False

    def export_to_csv(
        self, data: List[Dict[str, Any]], filepath: str, encoding: str = "utf-8"
    ) -> bool:
        """Export data to CSV."""
        try:
            import csv

            if not data:
                # Create an empty file with no headers if data is empty
                with open(filepath, "w", newline="", encoding=encoding):
                    pass
                logger.info("Exported empty CSV to %s", filepath)
                return True

            # Collect union of keys across rows to avoid missing columns
            fieldnames: List[str] = []
            for row in data:
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)

            with open(filepath, "w", newline="", encoding=encoding) as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data)
            logger.info("Exported CSV to %s with %d rows", filepath, len(data))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Export CSV error: %s", e, exc_info=True)
            return False

    def export_to_yaml(self, data: Any, filepath: str, encoding: str = "utf-8") -> bool:
        """Export data to YAML (requires PyYAML)."""
        if not _module_available("yaml"):
            logger.error("PyYAML not installed. Cannot export YAML.")
            return False
        try:
            import yaml  # type: ignore

            with open(filepath, "w", encoding=encoding) as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            logger.info("Exported YAML to %s", filepath)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Export YAML error: %s", e, exc_info=True)
            return False

    def export_to_xml(
        self, data: Any, filepath: str, root_tag: str = "root", encoding: str = "utf-8"
    ) -> bool:
        """Export data to XML using a simple, generic mapper."""
        try:
            from xml.etree.ElementTree import Element, ElementTree  # noqa: S405

            def to_element(key: str, value: Any):
                el = Element(str(key))
                if isinstance(value, dict):
                    for k, v in value.items():
                        el.append(to_element(k, v))
                elif isinstance(value, list):
                    for item in value:
                        el.append(to_element("item", item))
                else:
                    el.text = "" if value is None else str(value)
                return el

            root = to_element(root_tag, data)
            tree = ElementTree(root)
            tree.write(filepath, encoding=encoding, xml_declaration=True)
            logger.info("Exported XML to %s", filepath)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Export XML error: %s", e, exc_info=True)
            return False

    def export_to_format(self, data: Any, filepath: str, format: str) -> bool:
        """Export data to specified format."""
        fmt = format.lower()
        if fmt == "json":
            return self.export_to_json(data, filepath)
        if fmt == "csv":
            if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
                logger.error("CSV export requires a list of dicts.")
                return False
            return self.export_to_csv(data, filepath)
        if fmt == "yaml":
            return self.export_to_yaml(data, filepath)
        if fmt == "xml":
            return self.export_to_xml(data, filepath)
        raise ValueError(f"Unsupported format: {format}")


class ImportUtility:
    """Utility for importing governance data.

    Example:
        >>> importer = ImportUtility()
        >>> data = importer.import_from_json("input.json")
        >>> data = importer.import_from_csv("input.csv")
        >>> data = importer.import_from_yaml("input.yaml")  # if PyYAML installed
        >>> data = importer.import_from_xml("input.xml")    # best-effort
    """

    def __init__(self):
        """Initialize import utility."""
        self.supported_formats = ["json", "csv", "xml", "yaml"]

    def import_from_json(self, filepath: str, encoding: str = "utf-8") -> Any:
        """Import data from JSON."""
        try:
            with open(filepath, "r", encoding=encoding) as f:
                data = json.load(f)
            logger.info("Imported JSON from %s", filepath)
            return data
        except Exception as e:  # noqa: BLE001
            logger.error("Import JSON error: %s", e, exc_info=True)
            return None

    def import_from_csv(self, filepath: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
        """Import data from CSV."""
        try:
            import csv

            with open(filepath, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
            logger.info("Imported CSV from %s with %d rows", filepath, len(rows))
            return rows
        except Exception as e:  # noqa: BLE001
            logger.error("Import CSV error: %s", e, exc_info=True)
            return []

    def import_from_yaml(self, filepath: str, encoding: str = "utf-8") -> Any:
        """Import data from YAML (requires PyYAML)."""
        if not _module_available("yaml"):
            logger.error("PyYAML not installed. Cannot import YAML.")
            return None
        try:
            import yaml  # type: ignore

            with open(filepath, "r", encoding=encoding) as f:
                data = yaml.safe_load(f)
            logger.info("Imported YAML from %s", filepath)
            return data
        except Exception as e:  # noqa: BLE001
            logger.error("Import YAML error: %s", e, exc_info=True)
            return None

    def import_from_xml(self, filepath: str) -> Any:
        """Import data from XML.

        Tries xmltodict if available for a dict-like result; falls back to ElementTree.
        """
        try:
            if _module_available("xmltodict"):
                import xmltodict  # type: ignore

                with open(filepath, "rb") as f:
                    data = xmltodict.parse(f.read())
                logger.info("Imported XML via xmltodict from %s", filepath)
                return data
            # Fallback: Use defusedxml for safe XML parsing
            from defusedxml.ElementTree import parse

            def element_to_dict(el):
                children = list(el)
                if not children:
                    return el.text
                d: Dict[str, Any] = {}
                for c in children:
                    v = element_to_dict(c)
                    if c.tag in d:
                        if not isinstance(d[c.tag], list):
                            d[c.tag] = [d[c.tag]]
                        d[c.tag].append(v)
                    else:
                        d[c.tag] = v
                return d

            tree = parse(filepath)
            root = tree.getroot()
            data = {root.tag: element_to_dict(root)}
            logger.info("Imported XML via ElementTree from %s", filepath)
            return data
        except Exception as e:  # noqa: BLE001
            logger.error("Import XML error: %s", e, exc_info=True)
            return None

    def import_from_format(self, filepath: str, format: str) -> Any:
        """Import data from specified format."""
        fmt = format.lower()
        if fmt == "json":
            return self.import_from_json(filepath)
        if fmt == "csv":
            return self.import_from_csv(filepath)
        if fmt == "yaml":
            return self.import_from_yaml(filepath)
        if fmt == "xml":
            return self.import_from_xml(filepath)
        raise ValueError(f"Unsupported format: {format}")


# ======================
# Config validation
# ======================

_PY_TYPE_MAP: Dict[str, Type[Any]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _basic_validate_and_apply_defaults(
    schema: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Minimal validator using the schema shape in configuration_schema.

    Supports:
    - required: bool
    - default: Any
    - type: "string" | "integer" | "number" | "boolean" | "object" | "array"
    """
    validated = dict(config)  # shallow copy
    for key, rules in schema.items():
        # Apply default if missing
        if key not in validated and "default" in rules:
            validated[key] = rules["default"]

        # Required check
        if rules.get("required") and key not in validated:
            raise IntegrationConfigError(f"Missing required config key: {key}")

        # Type check
        if key in validated and "type" in rules:
            expected = _PY_TYPE_MAP.get(rules["type"])
            if expected and not isinstance(validated[key], expected):
                raise IntegrationConfigError(
                    f"Invalid type for key '{key}': expected {rules['type']}, got {type(validated[key]).__name__}"
                )

    return validated


def validate_config(schema: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate config using jsonschema if available, otherwise a basic validator."""
    if schema is None or not schema:
        return dict(config)
    # Try jsonschema if installed
    if _module_available("jsonschema"):
        try:
            from jsonschema import validate as js_validate  # type: ignore

            js_schema = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            # Convert our simplified schema to a jsonschema-like structure
            for key, rules in schema.items():
                prop: Dict[str, Any] = {}
                if "type" in rules and rules["type"] in _PY_TYPE_MAP:
                    prop["type"] = rules["type"] if rules["type"] != "integer" else "number"
                    # Prefer integer format if int
                    if rules["type"] == "integer":
                        prop["multipleOf"] = 1
                js_schema["properties"][key] = prop
                if rules.get("required"):
                    js_schema["required"].append(key)
            # Apply defaults first via basic pass, then validate
            cfg = _basic_validate_and_apply_defaults(schema, config)
            js_validate(instance=cfg, schema=js_schema)
            return cfg
        except Exception as e:  # noqa: BLE001
            logger.debug("jsonschema validation failed, falling back to basic validator: %s", e)
            # Fall back to basic
            return _basic_validate_and_apply_defaults(schema, config)
    # Basic
    return _basic_validate_and_apply_defaults(schema, config)


# ======================
# Integration Directory
# ======================

AdapterFactory = Callable[[str, Dict[str, Any]], IntegrationAdapter]
AdapterClass = Type[IntegrationAdapter]
AdapterFactoryLike = Union[AdapterFactory, AdapterClass]


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
        self._adapter_factories: Dict[str, AdapterFactory] = {}
        self._lock = threading.RLock()
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
                version="1.0.1",
                supported_formats=["sql"],
                configuration_schema={
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "default": 5432},
                    "database": {"type": "string", "required": True},
                    "username": {"type": "string", "required": True},
                    "password": {"type": "string", "required": True},
                    "ssl": {"type": "boolean", "default": False},
                },
            ),
            DataSourceAdapter,
        )

        # MongoDB data source
        self.register_integration(
            IntegrationMetadata(
                integration_id="mongodb",
                name="MongoDB Data Source",
                description="Connect to MongoDB databases",
                integration_type=IntegrationType.DATA_SOURCE,
                version="1.0.1",
                supported_formats=["json"],
                configuration_schema={
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "default": 27017},
                    "database": {"type": "string", "required": True},
                    "username": {"type": "string", "required": False},
                    "password": {"type": "string", "required": False},
                    "replica_set": {"type": "string"},
                    "tls": {"type": "boolean", "default": False},
                },
            ),
            DataSourceAdapter,
        )

    def _normalize_factory(self, factory: AdapterFactoryLike) -> AdapterFactory:
        if isinstance(factory, type) and issubclass(factory, IntegrationAdapter):
            klass: AdapterClass = factory

            def _ctor(integration_id: str, config: Dict[str, Any]) -> IntegrationAdapter:
                return klass(integration_id, config)

            return _ctor
        # Assume it is callable(factory signature matches)
        return factory  # type: ignore[return-value]

    def register_integration(
        self, metadata: IntegrationMetadata, adapter_class: AdapterFactoryLike
    ):
        """Register an integration.

        Args:
            metadata: Integration metadata
            adapter_class: Adapter class or factory
        """
        with self._lock:
            if metadata.integration_id in self._integrations:
                logger.warning("Overwriting existing integration: %s", metadata.integration_id)
            self._integrations[metadata.integration_id] = metadata
            self._adapter_factories[metadata.integration_id] = self._normalize_factory(
                adapter_class
            )
            logger.info("Registered integration '%s' (%s)", metadata.integration_id, metadata.name)

    def unregister_integration(self, integration_id: str) -> bool:
        """Unregister an integration by id."""
        with self._lock:
            existed = integration_id in self._integrations
            self._integrations.pop(integration_id, None)
            self._adapter_factories.pop(integration_id, None)
            if existed:
                logger.info("Unregistered integration '%s'", integration_id)
            return existed

    def is_registered(self, integration_id: str) -> bool:
        with self._lock:
            return integration_id in self._integrations

    def create_adapter(
        self, integration_id: str, config: Dict[str, Any]
    ) -> Optional[IntegrationAdapter]:
        """Create an integration adapter.

        Args:
            integration_id: Integration identifier
            config: Configuration dictionary

        Returns:
            Integration adapter or None
        """
        with self._lock:
            meta = self._integrations.get(integration_id)
            factory = self._adapter_factories.get(integration_id)
        if not meta or not factory:
            logger.error("Integration '%s' not found in registry.", integration_id)
            return None

        try:
            validated_config = validate_config(meta.configuration_schema, config)
        except IntegrationConfigError as e:
            logger.error("Configuration validation failed for '%s': %s", integration_id, e)
            raise

        adapter = factory(integration_id, validated_config)
        logger.debug("Created adapter instance for '%s'", integration_id)
        return adapter

    def list_integrations(
        self, integration_type: Optional[IntegrationType] = None
    ) -> List[IntegrationMetadata]:
        """List available integrations.

        Args:
            integration_type: Filter by type

        Returns:
            List of integration metadata
        """
        with self._lock:
            metas = list(self._integrations.values())
        if integration_type:
            return [m for m in metas if m.integration_type == integration_type]
        return metas

    def get_integration(self, integration_id: str) -> Optional[IntegrationMetadata]:
        """Get integration metadata.

        Args:
            integration_id: Integration identifier

        Returns:
            Integration metadata or None
        """
        with self._lock:
            return self._integrations.get(integration_id)

    def get_registered_ids(self) -> List[str]:
        """Return a list of all registered integration ids."""
        with self._lock:
            return list(self._integrations.keys())
