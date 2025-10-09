"""
Logging System Connectors

Connectors for external logging systems including syslog, CloudWatch, and others.

Features:
- Syslog connector for Unix/Linux systems
- AWS CloudWatch connector  
- Generic JSON logging connector
- Log aggregation and buffering
"""

import json
import logging
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'source': self.source,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


class LogConnector(ABC):
    """Abstract base class for log connectors"""
    
    @abstractmethod
    def send(self, entry: LogEntry) -> bool:
        """Send log entry to external system"""
        pass
    
    @abstractmethod
    def flush(self):
        """Flush any buffered logs"""
        pass
    
    @abstractmethod
    def close(self):
        """Close connection"""
        pass


class SyslogConnector(LogConnector):
    """
    Syslog connector for Unix/Linux systems
    
    Sends logs to local or remote syslog daemon.
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 514,
                 facility: int = 16,  # LOG_LOCAL0
                 protocol: str = 'UDP'):
        """
        Initialize syslog connector
        
        Args:
            host: Syslog server hostname
            port: Syslog server port (default: 514)
            facility: Syslog facility (default: LOG_LOCAL0)
            protocol: 'UDP' or 'TCP'
        """
        self.host = host
        self.port = port
        self.facility = facility
        self.protocol = protocol.upper()
        
        if self.protocol == 'UDP':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif self.protocol == 'TCP':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.socket.connect((self.host, self.port))
            except Exception as e:
                logging.error(f"Failed to connect to syslog server: {e}")
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        self.buffer: List[LogEntry] = []
        self.buffer_size = 100
    
    def send(self, entry: LogEntry) -> bool:
        """Send log entry to syslog"""
        try:
            # Map log level to syslog priority
            priority_map = {
                LogLevel.DEBUG: 7,
                LogLevel.INFO: 6,
                LogLevel.WARNING: 4,
                LogLevel.ERROR: 3,
                LogLevel.CRITICAL: 2
            }
            
            severity = priority_map.get(entry.level, 6)
            priority = self.facility * 8 + severity
            
            # Format syslog message (RFC 3164)
            timestamp = entry.timestamp.strftime('%b %d %H:%M:%S')
            message = f"<{priority}>{timestamp} {socket.gethostname()} {entry.source}: {entry.message}"
            
            if self.protocol == 'UDP':
                self.socket.sendto(message.encode('utf-8'), (self.host, self.port))
            else:
                self.socket.send(message.encode('utf-8') + b'\n')
            
            return True
        except Exception as e:
            logging.error(f"Failed to send syslog message: {e}")
            self.buffer.append(entry)
            return False
    
    def flush(self):
        """Flush buffered logs"""
        failed = []
        for entry in self.buffer:
            if not self.send(entry):
                failed.append(entry)
        self.buffer = failed
    
    def close(self):
        """Close syslog connection"""
        self.flush()
        self.socket.close()


class CloudWatchConnector(LogConnector):
    """
    AWS CloudWatch Logs connector
    
    Note: Requires boto3 library and AWS credentials.
    This is a stub implementation that logs intent to send to CloudWatch.
    """
    
    def __init__(self,
                 log_group: str,
                 log_stream: str,
                 region: str = 'us-east-1',
                 batch_size: int = 100):
        """
        Initialize CloudWatch connector
        
        Args:
            log_group: CloudWatch log group name
            log_stream: CloudWatch log stream name
            region: AWS region
            batch_size: Number of logs to batch before sending
        """
        self.log_group = log_group
        self.log_stream = log_stream
        self.region = region
        self.batch_size = batch_size
        
        self.buffer: List[LogEntry] = []
        
        # NOTE: Actual implementation would initialize boto3 client here
        # try:
        #     import boto3
        #     self.client = boto3.client('logs', region_name=region)
        # except ImportError:
        #     logging.error("boto3 not installed. Install with: pip install boto3")
        #     self.client = None
        
        logging.info(f"CloudWatch connector initialized (stub) - Group: {log_group}, Stream: {log_stream}")
    
    def send(self, entry: LogEntry) -> bool:
        """Send log entry to CloudWatch"""
        # Stub implementation - would send to CloudWatch
        self.buffer.append(entry)
        
        if len(self.buffer) >= self.batch_size:
            return self.flush()
        
        return True
    
    def flush(self) -> bool:
        """Flush buffered logs to CloudWatch"""
        if not self.buffer:
            return True
        
        # Stub implementation
        logging.info(f"[STUB] Would send {len(self.buffer)} logs to CloudWatch "
                    f"({self.log_group}/{self.log_stream})")
        
        # Actual implementation would use boto3:
        # try:
        #     events = [
        #         {
        #             'timestamp': int(entry.timestamp.timestamp() * 1000),
        #             'message': entry.to_json()
        #         }
        #         for entry in self.buffer
        #     ]
        #     
        #     response = self.client.put_log_events(
        #         logGroupName=self.log_group,
        #         logStreamName=self.log_stream,
        #         logEvents=events
        #     )
        #     self.buffer = []
        #     return True
        # except Exception as e:
        #     logging.error(f"Failed to send to CloudWatch: {e}")
        #     return False
        
        self.buffer = []
        return True
    
    def close(self):
        """Close CloudWatch connection"""
        self.flush()


class JSONFileConnector(LogConnector):
    """
    JSON file logging connector
    
    Writes structured logs to a JSON file (one JSON object per line).
    """
    
    def __init__(self, filepath: Union[str, Path], buffer_size: int = 10):
        """
        Initialize JSON file connector
        
        Args:
            filepath: Path to log file
            buffer_size: Number of entries to buffer before writing
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.buffer: List[LogEntry] = []
        self.buffer_size = buffer_size
        
        self.file = open(self.filepath, 'a')
    
    def send(self, entry: LogEntry) -> bool:
        """Send log entry to file"""
        self.buffer.append(entry)
        
        if len(self.buffer) >= self.buffer_size:
            return self.flush()
        
        return True
    
    def flush(self) -> bool:
        """Flush buffered logs to file"""
        try:
            for entry in self.buffer:
                self.file.write(entry.to_json() + '\n')
            self.file.flush()
            self.buffer = []
            return True
        except Exception as e:
            logging.error(f"Failed to write to log file: {e}")
            return False
    
    def close(self):
        """Close file"""
        self.flush()
        self.file.close()


class LogAggregator:
    """
    Aggregate logs and send to multiple connectors
    
    Example:
        aggregator = LogAggregator()
        aggregator.add_connector(SyslogConnector())
        aggregator.add_connector(CloudWatchConnector('my-group', 'my-stream'))
        
        aggregator.log(LogLevel.INFO, "Application started", "app")
    """
    
    def __init__(self):
        self.connectors: List[LogConnector] = []
    
    def add_connector(self, connector: LogConnector):
        """Add a log connector"""
        self.connectors.append(connector)
    
    def log(self,
            level: LogLevel,
            message: str,
            source: str,
            **metadata):
        """Log a message to all connectors"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source,
            metadata=metadata
        )
        
        for connector in self.connectors:
            try:
                connector.send(entry)
            except Exception as e:
                logging.error(f"Connector {connector.__class__.__name__} failed: {e}")
    
    def flush_all(self):
        """Flush all connectors"""
        for connector in self.connectors:
            try:
                connector.flush()
            except Exception as e:
                logging.error(f"Failed to flush {connector.__class__.__name__}: {e}")
    
    def close_all(self):
        """Close all connectors"""
        for connector in self.connectors:
            try:
                connector.close()
            except Exception as e:
                logging.error(f"Failed to close {connector.__class__.__name__}: {e}")


if __name__ == "__main__":
    # Demo usage
    print("Logging connectors initialized")
    
    # Create aggregator with multiple connectors
    aggregator = LogAggregator()
    aggregator.add_connector(JSONFileConnector("logs/external_logs.jsonl"))
    aggregator.add_connector(CloudWatchConnector("nethical-logs", "demo-stream"))
    
    # Log some messages
    aggregator.log(LogLevel.INFO, "Application started", "demo", version="1.0.0")
    aggregator.log(LogLevel.WARNING, "High memory usage", "monitor", memory_mb=512)
    
    # Clean up
    aggregator.close_all()
    print("Demo complete")
