"""
Nethical Universal Accelerator Abstraction

This module provides a unified interface for hardware acceleration across
multiple backends: NVIDIA CUDA, Google TPU (v7 Ironwood), and AWS Trainium3.

Architecture:
    - AcceleratorBackend: Enum defining supported backends
    - AcceleratorConfig: Configuration for accelerator initialization
    - AcceleratorInterface: Abstract base class for all accelerators
    - AcceleratorManager: Factory and manager for accelerator instances

Features:
    - Auto-detection with priority fallback
    - Unified API for all backends
    - Latency-aware batch processing
    - Memory management utilities

Fundamental Laws Alignment:
    - Law 23 (Fail-Safe Design): Graceful degradation to CPU
    - Law 8 (Transparency): Clear reporting of accelerator status
    - Law 15 (Audit Compliance): Performance metrics logging

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "AcceleratorBackend",
    "AcceleratorConfig",
    "AcceleratorInfo",
    "AcceleratorInterface",
    "AcceleratorManager",
    "get_available_accelerators",
    "get_best_accelerator",
]

log = logging.getLogger(__name__)


class AcceleratorBackend(str, Enum):
    """Supported hardware acceleration backends."""

    CUDA = "cuda"  # NVIDIA CUDA GPUs
    TPU = "tpu"  # Google TPU (v7 Ironwood)
    TRAINIUM = "trainium"  # AWS Trainium3
    CPU = "cpu"  # Fallback CPU execution

    @property
    def priority(self) -> int:
        """Get priority for auto-selection (higher = preferred)."""
        priorities = {
            "cuda": 100,
            "tpu": 90,
            "trainium": 85,
            "cpu": 0,
        }
        return priorities.get(self.value, 0)


@dataclass
class AcceleratorConfig:
    """Configuration for accelerator initialization.

    Attributes:
        backend: Preferred backend (None for auto-detect)
        device_id: Device ID for multi-device systems
        mixed_precision: Enable FP16/BF16 mixed precision
        memory_fraction: Fraction of device memory to use (0.0-1.0)
        compile_models: Pre-compile models for faster inference
        async_execution: Enable asynchronous execution
        batch_size: Default batch size for inference
    """

    backend: Optional[AcceleratorBackend] = None
    device_id: int = 0
    mixed_precision: bool = True
    memory_fraction: float = 0.9
    compile_models: bool = True
    async_execution: bool = False
    batch_size: int = 32
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration.

        Returns:
            Tuple of (valid, error_message)
        """
        if not 0.0 < self.memory_fraction <= 1.0:
            return False, "memory_fraction must be between 0.0 and 1.0"
        if self.batch_size < 1:
            return False, "batch_size must be at least 1"
        if self.device_id < 0:
            return False, "device_id must be non-negative"
        return True, ""


@dataclass
class AcceleratorInfo:
    """Information about an available accelerator.

    Attributes:
        backend: The accelerator backend type
        device_id: Device identifier
        device_name: Human-readable device name
        total_memory_gb: Total device memory in GB
        compute_capability: Compute capability/version string
        is_available: Whether the device is currently available
        specs: Additional device specifications
    """

    backend: AcceleratorBackend
    device_id: int
    device_name: str
    total_memory_gb: float
    compute_capability: str
    is_available: bool = True
    specs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "backend": self.backend.value,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "total_memory_gb": self.total_memory_gb,
            "compute_capability": self.compute_capability,
            "is_available": self.is_available,
            "specs": self.specs,
        }


class AcceleratorInterface(ABC):
    """Abstract interface for hardware accelerators.

    All accelerator implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize accelerator interface.

        Args:
            config: Accelerator configuration
        """
        self.config = config
        self._initialized = False
        self._device = None

    @property
    def is_initialized(self) -> bool:
        """Check if accelerator is initialized."""
        return self._initialized

    @property
    @abstractmethod
    def backend(self) -> AcceleratorBackend:
        """Get the backend type."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the accelerator.

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def get_info(self) -> AcceleratorInfo:
        """Get accelerator information.

        Returns:
            AcceleratorInfo with device details
        """
        pass

    @abstractmethod
    def to_device(self, data: np.ndarray) -> Any:
        """Transfer data to accelerator device.

        Args:
            data: NumPy array to transfer

        Returns:
            Device tensor/array
        """
        pass

    @abstractmethod
    def to_host(self, tensor: Any) -> np.ndarray:
        """Transfer data from device to host.

        Args:
            tensor: Device tensor

        Returns:
            NumPy array
        """
        pass

    @abstractmethod
    def execute(self, model: Any, inputs: Any) -> Any:
        """Execute model inference.

        Args:
            model: Model to execute
            inputs: Input tensor/array

        Returns:
            Model output
        """
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize device execution."""
        pass

    @abstractmethod
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory usage information.

        Returns:
            Dictionary with memory stats in GB
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear device memory cache."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown accelerator and release resources."""
        pass

    def batch_execute(
        self,
        model: Any,
        inputs: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Execute batch inference with automatic batching.

        Args:
            model: Model to execute
            inputs: Input array (n_samples, ...)
            batch_size: Batch size (uses config default if None)

        Returns:
            Output array
        """
        batch_size = batch_size or self.config.batch_size
        n_samples = len(inputs)
        outputs = []

        for i in range(0, n_samples, batch_size):
            batch = inputs[i : i + batch_size]
            batch_tensor = self.to_device(batch)
            result = self.execute(model, batch_tensor)
            outputs.append(self.to_host(result))

        self.synchronize()
        return np.concatenate(outputs, axis=0)


class CPUAccelerator(AcceleratorInterface):
    """CPU fallback accelerator.

    Used when no hardware accelerators are available.
    Implements the same interface for seamless fallback.
    """

    @property
    def backend(self) -> AcceleratorBackend:
        return AcceleratorBackend.CPU

    def initialize(self) -> bool:
        """Initialize CPU accelerator."""
        self._initialized = True
        log.info("CPU accelerator initialized (fallback mode)")
        return True

    def get_info(self) -> AcceleratorInfo:
        """Get CPU information."""
        import platform

        return AcceleratorInfo(
            backend=AcceleratorBackend.CPU,
            device_id=0,
            device_name=f"CPU ({platform.processor() or 'Unknown'})",
            total_memory_gb=0.0,  # Not applicable for CPU
            compute_capability="N/A",
            is_available=True,
            specs={"platform": platform.platform()},
        )

    def to_device(self, data: np.ndarray) -> np.ndarray:
        """No-op for CPU (data already on host)."""
        return data.astype(np.float32) if data.dtype != np.float32 else data

    def to_host(self, tensor: Any) -> np.ndarray:
        """No-op for CPU."""
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)

    def execute(self, model: Any, inputs: Any) -> Any:
        """Execute model on CPU."""
        if hasattr(model, "__call__"):
            return model(inputs)
        raise ValueError("Model must be callable")

    def synchronize(self) -> None:
        """No-op for CPU."""
        pass

    def get_memory_info(self) -> Dict[str, float]:
        """Get system memory info."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / 1e9,
                "available_gb": mem.available / 1e9,
                "used_gb": mem.used / 1e9,
            }
        except ImportError:
            return {}

    def clear_cache(self) -> None:
        """No-op for CPU."""
        pass

    def shutdown(self) -> None:
        """Shutdown CPU accelerator."""
        self._initialized = False
        log.info("CPU accelerator shutdown")


class AcceleratorManager:
    """Manager for hardware accelerators.

    Provides factory methods and manages accelerator lifecycle.
    Implements automatic backend detection and fallback.
    """

    _instance: Optional["AcceleratorManager"] = None
    _accelerators: Dict[str, AcceleratorInterface] = {}

    def __new__(cls) -> "AcceleratorManager":
        """Singleton pattern for global accelerator management."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._accelerators = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "AcceleratorManager":
        """Get the singleton instance."""
        return cls()

    def detect_available_backends(self) -> List[AcceleratorBackend]:
        """Detect all available acceleration backends.

        Returns:
            List of available backends sorted by priority
        """
        available = [AcceleratorBackend.CPU]  # CPU always available

        # Check CUDA
        try:
            from .cuda import is_cuda_available

            if is_cuda_available():
                available.append(AcceleratorBackend.CUDA)
        except ImportError:
            pass

        # Check TPU
        try:
            from .tpu import is_tpu_available

            if is_tpu_available():
                available.append(AcceleratorBackend.TPU)
        except ImportError:
            pass

        # Check Trainium
        try:
            from .trainium import is_trainium_available

            if is_trainium_available():
                available.append(AcceleratorBackend.TRAINIUM)
        except ImportError:
            pass

        # Sort by priority
        available.sort(key=lambda x: x.priority, reverse=True)
        log.info(f"Detected accelerators: {[a.value for a in available]}")
        return available

    def create_accelerator(
        self,
        config: Optional[AcceleratorConfig] = None,
    ) -> AcceleratorInterface:
        """Create an accelerator based on configuration.

        Args:
            config: Accelerator configuration (uses defaults if None)

        Returns:
            Initialized accelerator interface
        """
        config = config or AcceleratorConfig()

        # Validate config
        valid, error = config.validate()
        if not valid:
            raise ValueError(f"Invalid configuration: {error}")

        # Determine backend
        if config.backend is not None:
            backend = config.backend
        else:
            available = self.detect_available_backends()
            backend = available[0] if available else AcceleratorBackend.CPU

        # Create accelerator instance
        accelerator = self._create_backend(backend, config)

        # Initialize
        if not accelerator.initialize():
            log.warning(f"{backend.value} initialization failed, falling back to CPU")
            accelerator = CPUAccelerator(config)
            accelerator.initialize()

        # Cache instance
        key = f"{backend.value}:{config.device_id}"
        self._accelerators[key] = accelerator

        return accelerator

    def _create_backend(
        self,
        backend: AcceleratorBackend,
        config: AcceleratorConfig,
    ) -> AcceleratorInterface:
        """Create a specific backend accelerator.

        Args:
            backend: Backend type
            config: Configuration

        Returns:
            Accelerator interface instance
        """
        if backend == AcceleratorBackend.CUDA:
            from .cuda import CUDAAccelerator

            return CUDAAccelerator(config)

        elif backend == AcceleratorBackend.TPU:
            from .tpu import TPUAccelerator

            return TPUAccelerator(config)

        elif backend == AcceleratorBackend.TRAINIUM:
            from .trainium import TrainiumAccelerator

            return TrainiumAccelerator(config)

        else:
            return CPUAccelerator(config)

    def get_accelerator(
        self,
        backend: Optional[AcceleratorBackend] = None,
        device_id: int = 0,
    ) -> Optional[AcceleratorInterface]:
        """Get a cached accelerator instance.

        Args:
            backend: Backend type (None for any)
            device_id: Device ID

        Returns:
            Accelerator instance or None
        """
        if backend is not None:
            key = f"{backend.value}:{device_id}"
            return self._accelerators.get(key)

        # Return first available
        return next(iter(self._accelerators.values()), None)

    def shutdown_all(self) -> None:
        """Shutdown all accelerators."""
        for accelerator in self._accelerators.values():
            try:
                accelerator.shutdown()
            except Exception as e:
                log.error(f"Error shutting down accelerator: {e}")
        self._accelerators.clear()
        log.info("All accelerators shutdown")


def get_available_accelerators() -> List[AcceleratorInfo]:
    """Get information about all available accelerators.

    Returns:
        List of AcceleratorInfo for available devices
    """
    manager = AcceleratorManager.get_instance()
    backends = manager.detect_available_backends()
    infos = []

    for backend in backends:
        try:
            config = AcceleratorConfig(backend=backend)
            accelerator = manager._create_backend(backend, config)
            if accelerator.initialize():
                infos.append(accelerator.get_info())
                accelerator.shutdown()
        except Exception as e:
            log.warning(f"Could not get info for {backend.value}: {e}")

    return infos


def get_best_accelerator(config: Optional[AcceleratorConfig] = None) -> AcceleratorInterface:
    """Get the best available accelerator.

    Auto-detects the highest priority accelerator and returns
    an initialized instance.

    Args:
        config: Optional configuration

    Returns:
        Initialized accelerator interface
    """
    manager = AcceleratorManager.get_instance()
    return manager.create_accelerator(config)
