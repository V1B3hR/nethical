"""
Tests for the hardware accelerator abstraction layer.

Tests cover:
- Accelerator detection and initialization
- CPU fallback accelerator
- AcceleratorManager functionality
- Data transfer operations
"""

import pytest
import numpy as np

from nethical.core.accelerators import (
    AcceleratorBackend,
    AcceleratorConfig,
    AcceleratorInfo,
    AcceleratorManager,
    CPUAccelerator,
    get_available_accelerators,
    get_best_accelerator,
)


class TestAcceleratorBackend:
    """Test AcceleratorBackend enum."""

    def test_backend_values(self):
        """Test backend enum values."""
        assert AcceleratorBackend.CUDA.value == "cuda"
        assert AcceleratorBackend.TPU.value == "tpu"
        assert AcceleratorBackend.TRAINIUM.value == "trainium"
        assert AcceleratorBackend.CPU.value == "cpu"

    def test_backend_priority(self):
        """Test backend priorities."""
        assert AcceleratorBackend.CUDA.priority == 100
        assert AcceleratorBackend.TPU.priority == 90
        assert AcceleratorBackend.TRAINIUM.priority == 85
        assert AcceleratorBackend.CPU.priority == 0


class TestAcceleratorConfig:
    """Test AcceleratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AcceleratorConfig()
        assert config.backend is None
        assert config.device_id == 0
        assert config.mixed_precision is True
        assert config.memory_fraction == 0.9
        assert config.batch_size == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = AcceleratorConfig(
            backend=AcceleratorBackend.CUDA,
            device_id=1,
            mixed_precision=False,
            memory_fraction=0.5,
            batch_size=64,
        )
        assert config.backend == AcceleratorBackend.CUDA
        assert config.device_id == 1
        assert config.mixed_precision is False

    def test_config_validation_valid(self):
        """Test valid configuration passes validation."""
        config = AcceleratorConfig()
        valid, error = config.validate()
        assert valid is True
        assert error == ""

    def test_config_validation_invalid_memory_fraction(self):
        """Test invalid memory fraction fails validation."""
        config = AcceleratorConfig(memory_fraction=1.5)
        valid, error = config.validate()
        assert valid is False
        assert "memory_fraction" in error

    def test_config_validation_invalid_batch_size(self):
        """Test invalid batch size fails validation."""
        config = AcceleratorConfig(batch_size=0)
        valid, error = config.validate()
        assert valid is False
        assert "batch_size" in error

    def test_config_validation_invalid_device_id(self):
        """Test invalid device_id fails validation."""
        config = AcceleratorConfig(device_id=-1)
        valid, error = config.validate()
        assert valid is False
        assert "device_id" in error


class TestAcceleratorInfo:
    """Test AcceleratorInfo dataclass."""

    def test_info_creation(self):
        """Test info creation."""
        info = AcceleratorInfo(
            backend=AcceleratorBackend.CPU,
            device_id=0,
            device_name="Test CPU",
            total_memory_gb=16.0,
            compute_capability="N/A",
        )
        assert info.backend == AcceleratorBackend.CPU
        assert info.device_name == "Test CPU"
        assert info.is_available is True

    def test_info_to_dict(self):
        """Test info serialization."""
        info = AcceleratorInfo(
            backend=AcceleratorBackend.CUDA,
            device_id=0,
            device_name="Test GPU",
            total_memory_gb=8.0,
            compute_capability="8.0",
            specs={"cuda_cores": 1024},
        )
        result = info.to_dict()
        assert result["backend"] == "cuda"
        assert result["device_name"] == "Test GPU"
        assert result["total_memory_gb"] == 8.0
        assert result["specs"]["cuda_cores"] == 1024


class TestCPUAccelerator:
    """Test CPU fallback accelerator."""

    def test_initialization(self):
        """Test CPU accelerator initialization."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        result = accelerator.initialize()
        assert result is True
        assert accelerator.is_initialized is True

    def test_backend_property(self):
        """Test backend property."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        assert accelerator.backend == AcceleratorBackend.CPU

    def test_get_info(self):
        """Test get_info returns valid info."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        accelerator.initialize()
        info = accelerator.get_info()
        assert info.backend == AcceleratorBackend.CPU
        assert info.is_available is True
        assert "CPU" in info.device_name

    def test_to_device(self):
        """Test data transfer to device (no-op for CPU)."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        accelerator.initialize()

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = accelerator.to_device(data)
        assert result.dtype == np.float32

    def test_to_host(self):
        """Test data transfer to host (no-op for CPU)."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        accelerator.initialize()

        data = np.array([1.0, 2.0, 3.0])
        result = accelerator.to_host(data)
        np.testing.assert_array_equal(result, data)

    def test_execute(self):
        """Test model execution."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        accelerator.initialize()

        # Simple callable model
        def model(x):
            return x * 2

        inputs = np.array([1.0, 2.0, 3.0])
        result = accelerator.execute(model, inputs)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_synchronize(self):
        """Test synchronize (no-op for CPU)."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        accelerator.initialize()
        accelerator.synchronize()  # Should not raise

    def test_clear_cache(self):
        """Test clear_cache (no-op for CPU)."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        accelerator.initialize()
        accelerator.clear_cache()  # Should not raise

    def test_shutdown(self):
        """Test shutdown."""
        config = AcceleratorConfig()
        accelerator = CPUAccelerator(config)
        accelerator.initialize()
        accelerator.shutdown()
        assert accelerator.is_initialized is False

    def test_batch_execute(self):
        """Test batch execution."""
        config = AcceleratorConfig(batch_size=2)
        accelerator = CPUAccelerator(config)
        accelerator.initialize()

        def model(x):
            return x**2

        inputs = np.array([[1.0], [2.0], [3.0], [4.0]])
        outputs = accelerator.batch_execute(model, inputs, batch_size=2)
        expected = np.array([[1.0], [4.0], [9.0], [16.0]])
        np.testing.assert_array_almost_equal(outputs, expected)


class TestAcceleratorManager:
    """Test AcceleratorManager class."""

    def test_singleton(self):
        """Test singleton pattern."""
        manager1 = AcceleratorManager.get_instance()
        manager2 = AcceleratorManager.get_instance()
        assert manager1 is manager2

    def test_detect_available_backends(self):
        """Test backend detection."""
        manager = AcceleratorManager.get_instance()
        backends = manager.detect_available_backends()

        # CPU should always be available
        assert AcceleratorBackend.CPU in backends

        # Should be sorted by priority (descending)
        priorities = [b.priority for b in backends]
        assert priorities == sorted(priorities, reverse=True)

    def test_create_accelerator_cpu_fallback(self):
        """Test accelerator creation falls back to CPU."""
        manager = AcceleratorManager.get_instance()
        config = AcceleratorConfig(backend=AcceleratorBackend.CPU)
        accelerator = manager.create_accelerator(config)

        assert accelerator.backend == AcceleratorBackend.CPU
        assert accelerator.is_initialized is True

    def test_create_accelerator_auto_detect(self):
        """Test auto-detection creates an accelerator."""
        manager = AcceleratorManager.get_instance()
        config = AcceleratorConfig()  # No backend specified
        accelerator = manager.create_accelerator(config)

        assert accelerator.is_initialized is True
        # Should be at least CPU
        assert accelerator.backend in [
            AcceleratorBackend.CPU,
            AcceleratorBackend.CUDA,
            AcceleratorBackend.TPU,
            AcceleratorBackend.TRAINIUM,
        ]

    def test_shutdown_all(self):
        """Test shutting down all accelerators."""
        manager = AcceleratorManager.get_instance()

        # Create a couple accelerators
        config = AcceleratorConfig(backend=AcceleratorBackend.CPU)
        accelerator = manager.create_accelerator(config)

        # Shutdown all
        manager.shutdown_all()

        # Internal cache should be cleared
        assert len(manager._accelerators) == 0


class TestGetBestAccelerator:
    """Test get_best_accelerator function."""

    def test_returns_accelerator(self):
        """Test that function returns an accelerator."""
        accelerator = get_best_accelerator()
        assert accelerator is not None
        assert accelerator.is_initialized is True

    def test_with_config(self):
        """Test with custom config."""
        config = AcceleratorConfig(batch_size=64)
        accelerator = get_best_accelerator(config)
        assert accelerator.config.batch_size == 64


class TestGetAvailableAccelerators:
    """Test get_available_accelerators function."""

    def test_returns_list(self):
        """Test that function returns a list of AcceleratorInfo."""
        infos = get_available_accelerators()
        assert isinstance(infos, list)

        # Should have at least CPU
        assert len(infos) >= 1

        # All items should be AcceleratorInfo
        for info in infos:
            assert isinstance(info, AcceleratorInfo)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
