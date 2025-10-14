"""
Logging System Connectors

Connectors for external logging systems including syslog, CloudWatch, and others.

Enhancements aligned with Nethical system:
- Timezone-aware UTC timestamps
- RFC 5424-capable Syslog (plus RFC 3164 compatibility)
- Optional AWS CloudWatch batching (uses boto3 when available)
- Rotating JSONL file logging with buffering
- Secret redaction and metadata sanitation
- Aggregator config, context-manager support, and stdlib logging handler
- Optional Merkle audit anchoring connector (graceful no-op if dependencies missing)

References:
- docs/EXTERNAL_INTEGRATIONS_GUIDE.md
- docs/AUDIT_LOGGING_GUIDE.md
- docs/implementation/AUDIT_LOGGING_IMPLEMENTATION.md
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# ====== Utilities ======

_DEFAULT_REDACT_KEYS = {"password", "pass", "token", "secret", "api_key", "apikey", "authorization", "auth", "bearer"}
_DEFAULT_MAX_METADATA_LEN = 32_000  # cap very large blobs


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _redact(value: Any) -> Any:
    try:
        if isinstance(value, str):
            # Keep only a small preview for long secrets; otherwise mask fully.
            return "***" if len(value) <= 12 else f"{value[:3]}***{value[-3:]}"
        if isinstance(value, (list, tuple)):
            return type(value)(_redact(v) for v in value)
        if isinstance(value, dict):
            return {k: _redact(v) for k, v in value.items()}
    except Exception:
        pass
    return "***"


def sanitize_metadata(
    md: Dict[str, Any],
    *,
    redact_keys: Iterable[str] = _DEFAULT_REDACT_KEYS,
    max_len: int = _DEFAULT_MAX_METADATA_LEN,
) -> Dict[str, Any]:
    """Redact sensitive keys and cap oversized string values."""
    safe: Dict[str, Any] = {}
    redact_keys_lower = {k.lower() for k in redact_keys}

    for k, v in md.items():
        kl = str(k).lower()

        if kl in redact_keys_lower:
            safe[k] = _redact(v)
            continue

        try:
            if isinstance(v, str) and len(v) > max_len:
                safe[k] = v[:max_len] + "...(truncated)"
            else:
                safe[k] = v
        except Exception:
            safe[k] = str(v)
    return safe


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry with UTC timestamps and safe metadata."""
    timestamp: datetime
    level: LogLevel
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        safe_metadata = sanitize_metadata(self.metadata)
        return {
            "timestamp": self.timestamp.isoformat(),  # UTC with tz
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "metadata": safe_metadata,
            "tz": "UTC",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_log_record(record: logging.LogRecord, source: str = "python-logger") -> "LogEntry":
        # Attach extras if present
        md = {}
        for k, v in getattr(record, "__dict__", {}).items():
            # Filter internal LogRecord attributes
            if k in {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process"
            }:
                continue
            md[k] = v

        level_map = {
            logging.DEBUG: LogLevel.DEBUG,
            logging.INFO: LogLevel.INFO,
            logging.WARNING: LogLevel.WARNING,
            logging.ERROR: LogLevel.ERROR,
            logging.CRITICAL: LogLevel.CRITICAL,
        }
        return LogEntry(
            timestamp=_utcnow(),
            level=level_map.get(record.levelno, LogLevel.INFO),
            message=record.getMessage(),
            source=source or record.name,
            metadata=md,
        )


class LogConnector(ABC):
    """Abstract base class for log connectors"""

    @abstractmethod
    def send(self, entry: LogEntry) -> bool:
        """Send log entry to external system"""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> bool:
        """Flush any buffered logs"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close connection"""
        raise NotImplementedError


class SyslogConnector(LogConnector):
    """
    Syslog connector for Unix/Linux systems

    Supports:
    - UDP or TCP transport
    - RFC 3164 (BSD) format (default)
    - RFC 5424 format with structured metadata
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 514,
        facility: int = 16,  # LOG_LOCAL0
        protocol: str = "UDP",
        rfc: int = 3164,     # or 5424
        app_name: Optional[str] = None,
        procid: Optional[str] = None,
        msgid: str = "-",
        include_metadata: bool = True,
    ):
        """
        Args:
            host: Syslog server hostname
            port: Syslog server port (default: 514)
            facility: Syslog facility (default: LOG_LOCAL0)
            protocol: 'UDP' or 'TCP'
            rfc: 3164 (BSD) or 5424 (structured)
            app_name: RFC 5424 app-name field (defaults to entry.source if not provided)
            procid: RFC 5424 procid field
            msgid: RFC 5424 msgid field
            include_metadata: Include metadata as structured data (5424) or appended JSON (3164)
        """
        self.host = host
        self.port = port
        self.facility = facility
        self.protocol = protocol.upper()
        self.rfc = rfc
        self.app_name = app_name
        self.procid = procid or str(os.getpid())
        self.msgid = msgid
        self.include_metadata = include_metadata

        if self.protocol == "UDP":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif self.protocol == "TCP":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.socket.connect((self.host, self.port))
            except Exception as e:
                logging.error(f"Failed to connect to syslog server: {e}")
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        self.buffer: List[LogEntry] = []
        self.buffer_size = 100

    def _format(self, entry: LogEntry) -> bytes:
        priority_map = {
            LogLevel.DEBUG: 7,
            LogLevel.INFO: 6,
            LogLevel.WARNING: 4,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 2,
        }
        severity = priority_map.get(entry.level, 6)
        pri = self.facility * 8 + severity

        if self.rfc == 5424:
            # RFC 5424: <PRI>1 TIMESTAMP HOST APP PROCID MSGID [SD] MSG
            ts = entry.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            host = socket.gethostname()
            app = self.app_name or entry.source
            proc = self.procid or "-"
            msgid = self.msgid or "-"

            sd = "-"
            if self.include_metadata and entry.metadata:
                # Put metadata as a single structured data element; keys must be safe
                try:
                    sd_pairs = " ".join(
                        f'{k}="{str(v).replace("\\", "\\\\").replace("\\"","\\\\"")}"'
                        for k, v in sanitize_metadata(entry.metadata).items()
                        if isinstance(k, str)
                    )
                    sd = f"[meta {sd_pairs}]"
                except Exception:
                    sd = "-"

            msg = entry.message
            line = f"<{pri}>1 {ts} {host} {app} {proc} {msgid} {sd} {msg}"
            return line.encode("utf-8")

        # RFC 3164 (BSD)
        timestamp = entry.timestamp.strftime("%b %d %H:%M:%S")
        host = socket.gethostname()
        base = f"<{pri}>{timestamp} {host} {entry.source}: {entry.message}"
        if self.include_metadata and entry.metadata:
            try:
                base += " " + json.dumps(sanitize_metadata(entry.metadata), ensure_ascii=False)
            except Exception:
                pass
        return (base + "\n").encode("utf-8")

    def send(self, entry: LogEntry) -> bool:
        try:
            data = self._format(entry)
            if self.protocol == "UDP":
                self.socket.sendto(data, (self.host, self.port))
            else:
                # TCP syslog often expects LF-delimited messages
                if not data.endswith(b"\n"):
                    data += b"\n"
                self.socket.sendall(data)
            return True
        except Exception as e:
            logging.error(f"Failed to send syslog message: {e}")
            self.buffer.append(entry)
            return False

    def flush(self) -> bool:
        failed = []
        for entry in self.buffer:
            if not self.send(entry):
                failed.append(entry)
        self.buffer = failed
        return len(self.buffer) == 0

    def close(self) -> None:
        self.flush()
        try:
            self.socket.close()
        except Exception:
            pass


class CloudWatchConnector(LogConnector):
    """
    AWS CloudWatch Logs connector (production-ready with graceful degradation)

    - Uses boto3 if available; logs a stub message otherwise.
    - Batches items and handles upload sequence tokens.
    """

    def __init__(
        self,
        log_group: str,
        log_stream: str,
        region: str = "us-east-1",
        batch_size: int = 100,
        create_if_missing: bool = True,
    ):
        self.log_group = log_group
        self.log_stream = log_stream
        self.region = region
        self.batch_size = batch_size
        self.create_if_missing = create_if_missing

        self.buffer: List[LogEntry] = []
        self._lock = threading.Lock()
        self._client = None
        self._sequence_token: Optional[str] = None

        try:
            import boto3  # type: ignore
            self._client = boto3.client("logs", region_name=region)
            if create_if_missing:
                self._ensure_group_stream()
        except Exception as e:
            logging.info(
                f"CloudWatch connector running in STUB mode (boto3 not available or init failed): {e}"
            )

    def _ensure_group_stream(self) -> None:
        if not self._client:
            return
        try:
            groups = self._client.describe_log_groups(logGroupNamePrefix=self.log_group).get("logGroups", [])
            if not any(g.get("logGroupName") == self.log_group for g in groups):
                self._client.create_log_group(logGroupName=self.log_group)
        except self._client.exceptions.ResourceAlreadyExistsException:  # type: ignore
            pass
        except Exception:
            pass

        try:
            streams = self._client.describe_log_streams(
                logGroupName=self.log_group, logStreamNamePrefix=self.log_stream
            ).get("logStreams", [])
            if not any(s.get("logStreamName") == self.log_stream for s in streams):
                self._client.create_log_stream(logGroupName=self.log_group, logStreamName=self.log_stream)
                self._sequence_token = None
            else:
                # Capture existing token if present
                for s in streams:
                    if s.get("logStreamName") == self.log_stream:
                        self._sequence_token = s.get("uploadSequenceToken")
                        break
        except Exception:
            pass

    def send(self, entry: LogEntry) -> bool:
        with self._lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.batch_size:
                return self.flush()
            return True

    def flush(self) -> bool:
        with self._lock:
            if not self.buffer:
                return True

            if not self._client:
                # Stub mode
                logging.info(
                    f"[STUB] Would send {len(self.buffer)} logs to CloudWatch ({self.log_group}/{self.log_stream})"
                )
                self.buffer = []
                return True

            # Convert to CloudWatch events and sort by timestamp
            events = [
                {
                    "timestamp": int(entry.timestamp.timestamp() * 1000),
                    "message": entry.to_json(),
                }
                for entry in self.buffer
            ]
            events.sort(key=lambda e: e["timestamp"])

            args = {
                "logGroupName": self.log_group,
                "logStreamName": self.log_stream,
                "logEvents": events,
            }
            if self._sequence_token:
                args["sequenceToken"] = self._sequence_token

            try:
                resp = self._client.put_log_events(**args)
                self._sequence_token = resp.get("nextSequenceToken")
                self.buffer = []
                return True
            except Exception as e:
                logging.error(f"Failed to send to CloudWatch: {e}")
                # Keep buffer for retry
                return False

    def close(self) -> None:
        self.flush()


class JSONFileConnector(LogConnector):
    """
    JSON file logging connector with size-based rotation

    Writes structured logs to a JSON file (one JSON object per line).
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        buffer_size: int = 10,
        max_bytes: int = 10_000_000,  # ~10MB
        backup_count: int = 5,
        encoding: str = "utf-8",
    ):
        """
        Args:
            filepath: Path to log file
            buffer_size: Number of entries to buffer before writing
            max_bytes: Rotate when file exceeds this size (0 disables rotation)
            backup_count: How many rotated files to keep
            encoding: File encoding for writes
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.buffer: List[LogEntry] = []
        self.buffer_size = buffer_size

        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding

        self._lock = threading.Lock()
        self._open_file()

    def _open_file(self) -> None:
        self.file = open(self.filepath, "a", encoding=self.encoding)

    def _should_rotate(self) -> bool:
        if self.max_bytes <= 0:
            return False
        try:
            return self.file.tell() >= self.max_bytes or (self.filepath.exists() and self.filepath.stat().st_size >= self.max_bytes)
        except Exception:
            return False

    def _rotate(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass

        # Rotate files: .N -> .N+1, base -> .1
        for i in range(self.backup_count - 1, 0, -1):
            s = self.filepath.with_suffix(self.filepath.suffix + f".{i}")
            d = self.filepath.with_suffix(self.filepath.suffix + f".{i+1}")
            if s.exists():
                try:
                    if d.exists():
                        d.unlink()
                    s.rename(d)
                except Exception:
                    pass

        first = self.filepath.with_suffix(self.filepath.suffix + ".1")
        try:
            if first.exists():
                first.unlink()
            if self.filepath.exists():
                self.filepath.rename(first)
        except Exception:
            pass

        self._open_file()

    def send(self, entry: LogEntry) -> bool:
        with self._lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                return self.flush()
            return True

    def flush(self) -> bool:
        with self._lock:
            try:
                if not self.buffer:
                    return True
                if self._should_rotate():
                    self._rotate()
                for entry in self.buffer:
                    self.file.write(entry.to_json() + "\n")
                self.file.flush()
                os.fsync(self.file.fileno())  # durability hint
                self.buffer = []
                return True
            except Exception as e:
                logging.error(f"Failed to write to log file: {e}")
                return False

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                self.file.close()
            except Exception:
                pass


class MerkleAnchorConnector(LogConnector):
    """
    Optional Merkle audit anchoring connector.

    Integrates with Nethical's Merkle audit logging when available.
    If unavailable, acts as a no-op that logs intent.

    This is designed to complement training audit usage documented in:
    - docs/AUDIT_LOGGING_GUIDE.md
    - docs/implementation/AUDIT_LOGGING_IMPLEMENTATION.md
    """

    def __init__(self, audit_path: Union[str, Path] = "training_audit_logs"):
        self.audit_path = Path(audit_path)
        self.audit_path.mkdir(parents=True, exist_ok=True)
        self._buffer: List[LogEntry] = []
        self._anchor = None
        try:
            # Lazy import to avoid hard dependency for general use
            from nethical.core.audit_merkle import MerkleAnchor  # type: ignore
            self._anchor = MerkleAnchor(base_path=str(self.audit_path))
        except Exception as e:
            logging.info(f"MerkleAnchor not available, running in NO-OP mode: {e}")

    def send(self, entry: LogEntry) -> bool:
        self._buffer.append(entry)
        # Anchor on flush or if no anchor, just buffer
        return True

    def flush(self) -> bool:
        if not self._buffer:
            return True
        try:
            if self._anchor is None:
                logging.info(f"[NO-OP MerkleAnchor] Would anchor {len(self._buffer)} log entries.")
                self._buffer = []
                return True

            # Add events to anchor as generic events
            for e in self._buffer:
                self._anchor.log_event(
                    event_type="external_log",
                    payload=e.to_dict(),
                )
            # Finalize a small chunk for observability
            self._anchor.finalize_chunk()
            self._buffer = []
            return True
        except Exception as e:
            logging.error(f"MerkleAnchor flush failed: {e}")
            return False

    def close(self) -> None:
        self.flush()


class LogAggregator:
    """
    Aggregate logs and send to multiple connectors

    Example:
        aggregator = LogAggregator.from_config({
            "connectors": {
                "json": {"filepath": "logs/external_logs.jsonl", "buffer_size": 20},
                "syslog": {"host": "localhost", "port": 514, "protocol": "UDP", "rfc": 5424},
                "cloudwatch": {"log_group": "nethical-logs", "log_stream": "demo-stream"}
            }
        })
        aggregator.log(LogLevel.INFO, "Application started", "app", version="1.0.0")
    """

    def __init__(self):
        self.connectors: List[LogConnector] = []

    def add_connector(self, connector: LogConnector) -> None:
        self.connectors.append(connector)

    def log(self, level: LogLevel, message: str, source: str, **metadata) -> None:
        entry = LogEntry(
            timestamp=_utcnow(),
            level=level,
            message=message,
            source=source,
            metadata=metadata,
        )
        for connector in self.connectors:
            try:
                connector.send(entry)
            except Exception as e:
                logging.error(f"Connector {connector.__class__.__name__} failed: {e}")

    def flush_all(self) -> None:
        for connector in self.connectors:
            try:
                connector.flush()
            except Exception as e:
                logging.error(f"Failed to flush {connector.__class__.__name__}: {e}")

    def close_all(self) -> None:
        for connector in self.connectors:
            try:
                connector.close()
            except Exception as e:
                logging.error(f"Failed to close {connector.__class__.__name__}: {e}")

    def __enter__(self) -> "LogAggregator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_all()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LogAggregator":
        """
        Construct aggregator from a config dict, consistent with
        docs/EXTERNAL_INTEGRATIONS_GUIDE.md recommendations.

        Example config:
        {
            "connectors": {
                "syslog": {"host": "localhost", "port": 514, "protocol": "UDP", "rfc": 5424},
                "json": {"filepath": "logs/application.jsonl", "buffer_size": 10},
                "cloudwatch": {"log_group": "production-logs", "log_stream": "nethical-app", "region": "us-east-1"},
                "merkle": {"audit_path": "training_audit_logs"}
            }
        }
        """
        agg = cls()
        connectors = (config or {}).get("connectors", {})

        if "syslog" in connectors:
            agg.add_connector(SyslogConnector(**connectors["syslog"]))
        if "json" in connectors:
            agg.add_connector(JSONFileConnector(**connectors["json"]))
        if "cloudwatch" in connectors:
            agg.add_connector(CloudWatchConnector(**connectors["cloudwatch"]))
        if "merkle" in connectors:
            agg.add_connector(MerkleAnchorConnector(**connectors["merkle"]))

        return agg


class AggregatorHandler(logging.Handler):
    """
    Standard logging handler that forwards LogRecords into LogAggregator.

    Usage:
        aggregator = LogAggregator()
        aggregator.add_connector(JSONFileConnector("logs/app.jsonl"))
        handler = AggregatorHandler(aggregator, source="app")
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, aggregator: LogAggregator, source: Optional[str] = None):
        super().__init__()
        self.aggregator = aggregator
        self.source = source or "python-logger"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogEntry.from_log_record(record, source=self.source or record.name)
            # Map stdlib levels to our enum
            self.aggregator.log(entry.level, entry.message, entry.source, **entry.metadata)
        except Exception:
            self.handleError(record)


if __name__ == "__main__":
    # Demo usage
    print("Logging connectors initialized")

    # Create aggregator with multiple connectors
    aggregator = LogAggregator()
    aggregator.add_connector(JSONFileConnector("logs/external_logs.jsonl"))
    aggregator.add_connector(CloudWatchConnector("nethical-logs", "demo-stream"))
    aggregator.add_connector(SyslogConnector(rfc=5424, protocol="UDP"))

    # Optional Merkle anchoring (no-op if MerkleAnchor unavailable)
    aggregator.add_connector(MerkleAnchorConnector("training_audit_logs"))

    # Log some messages
    aggregator.log(LogLevel.INFO, "Application started", "demo", version="1.0.0")
    aggregator.log(LogLevel.WARNING, "High memory usage", "monitor", memory_mb=512)

    # Clean up
    aggregator.close_all()
    print("Demo complete")
