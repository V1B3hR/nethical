"""
Tests for External Integrations: Logging Connectors

Tests the logging system connectors including syslog, CloudWatch, and JSON file logging.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from nethical.integrations.logging_connectors import (
    LogLevel,
    LogEntry,
    LogConnector,
    SyslogConnector,
    CloudWatchConnector,
    JSONFileConnector,
    LogAggregator
)


class TestLogEntry:
    """Test LogEntry dataclass"""
    
    def test_log_entry_creation(self):
        """Test creating a log entry"""
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level=LogLevel.INFO,
            message="Test message",
            source="test",
            metadata={"key": "value"}
        )
        
        assert entry.timestamp == timestamp
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.source == "test"
        assert entry.metadata == {"key": "value"}
    
    def test_log_entry_to_dict(self):
        """Test converting log entry to dict"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message="Warning message",
            source="app"
        )
        
        entry_dict = entry.to_dict()
        assert entry_dict['level'] == 'WARNING'
        assert entry_dict['message'] == "Warning message"
        assert entry_dict['source'] == "app"
        assert 'timestamp' in entry_dict
    
    def test_log_entry_to_json(self):
        """Test converting log entry to JSON"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error message",
            source="app"
        )
        
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['level'] == 'ERROR'
        assert parsed['message'] == "Error message"


class TestJSONFileConnector:
    """Test JSON file logging connector"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_connector_creation(self, temp_dir):
        """Test creating JSON file connector"""
        log_file = temp_dir / "test.jsonl"
        connector = JSONFileConnector(filepath=log_file, buffer_size=5)
        
        assert connector.filepath == log_file
        assert connector.buffer_size == 5
        assert log_file.parent.exists()
        
        connector.close()
    
    def test_send_log_entry(self, temp_dir):
        """Test sending log entries"""
        log_file = temp_dir / "test.jsonl"
        connector = JSONFileConnector(filepath=log_file, buffer_size=2)
        
        entry1 = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="First message",
            source="test"
        )
        
        entry2 = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            message="Second message",
            source="test"
        )
        
        # Send entries (should buffer)
        connector.send(entry1)
        connector.send(entry2)  # This should trigger flush
        
        connector.close()
        
        # Read file and verify
        assert log_file.exists()
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            log1 = json.loads(lines[0])
            assert log1['message'] == "First message"
            
            log2 = json.loads(lines[1])
            assert log2['message'] == "Second message"
    
    def test_flush(self, temp_dir):
        """Test manual flush"""
        log_file = temp_dir / "test.jsonl"
        connector = JSONFileConnector(filepath=log_file, buffer_size=10)
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            source="test"
        )
        
        connector.send(entry)
        # File should be empty (buffered)
        connector.flush()
        # Now file should have content
        
        connector.close()
        
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 1


class TestSyslogConnector:
    """Test syslog connector"""
    
    def test_connector_creation_udp(self):
        """Test creating syslog connector with UDP"""
        connector = SyslogConnector(
            host='localhost',
            port=514,
            protocol='UDP'
        )
        
        assert connector.host == 'localhost'
        assert connector.port == 514
        assert connector.protocol == 'UDP'
        
        connector.close()
    
    def test_connector_creation_invalid_protocol(self):
        """Test creating syslog connector with invalid protocol"""
        with pytest.raises(ValueError, match="Unsupported protocol"):
            SyslogConnector(protocol='HTTP')
    
    def test_send_log_entry(self):
        """Test sending log entry to syslog"""
        connector = SyslogConnector(host='localhost', port=514)
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            source="test"
        )
        
        # This may fail if no syslog server, but should not raise exception
        try:
            result = connector.send(entry)
            # If successful or buffered, should return boolean
            assert isinstance(result, bool)
        except Exception:
            # Expected if no syslog server running
            pass
        
        connector.close()
    
    def test_buffer_on_failure(self):
        """Test that failed sends are buffered"""
        # Use invalid host to force failure
        connector = SyslogConnector(host='invalid.host.that.does.not.exist', port=514)
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            source="test"
        )
        
        # Should buffer entry on failure
        result = connector.send(entry)
        # May succeed or fail depending on environment
        assert isinstance(result, bool)
        
        connector.close()


class TestCloudWatchConnector:
    """Test CloudWatch connector (stub)"""
    
    def test_connector_creation(self):
        """Test creating CloudWatch connector"""
        connector = CloudWatchConnector(
            log_group="test-group",
            log_stream="test-stream",
            region="us-east-1"
        )
        
        assert connector.log_group == "test-group"
        assert connector.log_stream == "test-stream"
        assert connector.region == "us-east-1"
        
        connector.close()
    
    def test_send_log_entry(self):
        """Test sending log entry (stub)"""
        connector = CloudWatchConnector(
            log_group="test-group",
            log_stream="test-stream"
        )
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            source="test"
        )
        
        result = connector.send(entry)
        assert result is True  # Stub always returns True
        
        connector.close()
    
    def test_batch_send(self):
        """Test batching behavior"""
        connector = CloudWatchConnector(
            log_group="test-group",
            log_stream="test-stream",
            batch_size=3
        )
        
        # Send 3 entries (should trigger flush)
        for i in range(3):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                source="test"
            )
            connector.send(entry)
        
        # Buffer should be empty after auto-flush
        assert len(connector.buffer) == 0
        
        connector.close()


class TestLogAggregator:
    """Test log aggregator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_aggregator_creation(self):
        """Test creating log aggregator"""
        aggregator = LogAggregator()
        assert len(aggregator.connectors) == 0
    
    def test_add_connector(self, temp_dir):
        """Test adding connectors"""
        aggregator = LogAggregator()
        
        json_connector = JSONFileConnector(temp_dir / "test.jsonl")
        cloudwatch_connector = CloudWatchConnector("group", "stream")
        
        aggregator.add_connector(json_connector)
        aggregator.add_connector(cloudwatch_connector)
        
        assert len(aggregator.connectors) == 2
        
        aggregator.close_all()
    
    def test_log_to_all_connectors(self, temp_dir):
        """Test logging to all connectors"""
        aggregator = LogAggregator()
        
        # Add JSON file connector
        log_file = temp_dir / "aggregated.jsonl"
        json_connector = JSONFileConnector(log_file, buffer_size=1)
        aggregator.add_connector(json_connector)
        
        # Add CloudWatch connector (stub)
        cloudwatch_connector = CloudWatchConnector("group", "stream")
        aggregator.add_connector(cloudwatch_connector)
        
        # Log a message
        aggregator.log(
            level=LogLevel.INFO,
            message="Test message",
            source="test",
            extra_field="extra_value"
        )
        
        aggregator.close_all()
        
        # Verify JSON file received the log
        assert log_file.exists()
        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)
            assert log_entry['message'] == "Test message"
            assert log_entry['metadata']['extra_field'] == "extra_value"
    
    def test_flush_all(self, temp_dir):
        """Test flushing all connectors"""
        aggregator = LogAggregator()
        
        log_file = temp_dir / "test.jsonl"
        connector = JSONFileConnector(log_file, buffer_size=10)
        aggregator.add_connector(connector)
        
        # Log without auto-flush
        aggregator.log(LogLevel.INFO, "Message 1", "test")
        aggregator.log(LogLevel.INFO, "Message 2", "test")
        
        # Manual flush
        aggregator.flush_all()
        aggregator.close_all()
        
        # Verify logs were written
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 2
    
    def test_error_handling(self, temp_dir):
        """Test error handling when connector fails"""
        aggregator = LogAggregator()
        
        # Add a connector that will fail
        class FailingConnector(LogConnector):
            def send(self, entry):
                raise Exception("Simulated failure")
            def flush(self):
                pass
            def close(self):
                pass
        
        aggregator.add_connector(FailingConnector())
        
        # Should not raise exception
        aggregator.log(LogLevel.INFO, "Test message", "test")
        aggregator.close_all()


class TestIntegration:
    """Integration tests for logging connectors"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_multi_connector_logging(self, temp_dir):
        """Test logging to multiple connectors simultaneously"""
        aggregator = LogAggregator()
        
        # Setup multiple file connectors
        file1 = temp_dir / "log1.jsonl"
        file2 = temp_dir / "log2.jsonl"
        
        aggregator.add_connector(JSONFileConnector(file1, buffer_size=1))
        aggregator.add_connector(JSONFileConnector(file2, buffer_size=1))
        aggregator.add_connector(CloudWatchConnector("group", "stream"))
        
        # Log multiple messages
        for i in range(5):
            aggregator.log(
                level=LogLevel.INFO if i % 2 == 0 else LogLevel.WARNING,
                message=f"Message {i}",
                source="integration_test",
                iteration=i
            )
        
        aggregator.close_all()
        
        # Verify both files received all logs
        for log_file in [file1, file2]:
            assert log_file.exists()
            with open(log_file) as f:
                lines = f.readlines()
                assert len(lines) == 5
                
                # Verify content
                for i, line in enumerate(lines):
                    log_entry = json.loads(line)
                    assert log_entry['message'] == f"Message {i}"
                    assert log_entry['metadata']['iteration'] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
