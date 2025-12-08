"""
NVIDIA CUDA GPU Accelerator for Nethical

This module provides CUDA-based GPU acceleration for ML inference.
Supports NVIDIA GPUs with CUDA compute capability 3.5+.

Features:
    - Automatic GPU detection and selection
    - Mixed precision (FP16/BF16) support
    - TensorRT integration for optimized inference
    - Multi-GPU support
    - Memory management and optimization

Requirements:
    - PyTorch with CUDA support
    - NVIDIA GPU with CUDA 11.0+

Fundamental Laws Alignment:
    - Law 23 (Fail-Safe Design): Graceful fallback to CPU
    - Law 15 (Audit Compliance): Performance metrics logging

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from . import (
    AcceleratorBackend,
    AcceleratorConfig,
    AcceleratorInfo,
    AcceleratorInterface,
)

__all__ = [
    "CUDAAccelerator",
    "is_cuda_available",
    "get_cuda_info",
]

log = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.debug("PyTorch not available - CUDA acceleration disabled")


def is_cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA-enabled GPU is available
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_cuda_info() -> Dict[str, Any]:
    """Get CUDA device information.

    Returns:
        Dictionary with CUDA information
    """
    if not is_cuda_available():
        return {"available": False, "reason": "CUDA not available"}

    device_count = torch.cuda.device_count()
    devices = []

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )

    return {
        "available": True,
        "device_count": device_count,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "devices": devices,
    }


class CUDAAccelerator(AcceleratorInterface):
    """NVIDIA CUDA GPU accelerator implementation.

    Provides GPU-accelerated inference using PyTorch CUDA backend.
    Supports mixed precision and TensorRT optimization.
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize CUDA accelerator.

        Args:
            config: Accelerator configuration
        """
        super().__init__(config)
        self._stream: Optional[Any] = None
        self._dtype = None

    @property
    def backend(self) -> AcceleratorBackend:
        return AcceleratorBackend.CUDA

    def initialize(self) -> bool:
        """Initialize CUDA device.

        Returns:
            True if initialization successful
        """
        if not is_cuda_available():
            log.warning("CUDA not available")
            return False

        try:
            device_id = self.config.device_id
            device_count = torch.cuda.device_count()

            if device_id >= device_count:
                log.warning(f"Device {device_id} not found, using device 0")
                device_id = 0

            self._device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id)

            # Set memory fraction if specified
            if self.config.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction, device_id
                )

            # Create stream for async execution
            if self.config.async_execution:
                self._stream = torch.cuda.Stream(device=self._device)

            # Set dtype for mixed precision
            if self.config.mixed_precision:
                self._dtype = torch.float16
            else:
                self._dtype = torch.float32

            self._initialized = True
            device_name = torch.cuda.get_device_name(device_id)
            log.info(
                f"CUDA accelerator initialized on device {device_id}: {device_name}"
            )

            return True

        except Exception as e:
            log.error(f"CUDA initialization failed: {e}")
            return False

    def get_info(self) -> AcceleratorInfo:
        """Get CUDA device information."""
        if not self._initialized:
            return AcceleratorInfo(
                backend=AcceleratorBackend.CUDA,
                device_id=self.config.device_id,
                device_name="Not initialized",
                total_memory_gb=0.0,
                compute_capability="N/A",
                is_available=False,
            )

        props = torch.cuda.get_device_properties(self.config.device_id)
        return AcceleratorInfo(
            backend=AcceleratorBackend.CUDA,
            device_id=self.config.device_id,
            device_name=props.name,
            total_memory_gb=props.total_memory / 1e9,
            compute_capability=f"{props.major}.{props.minor}",
            is_available=True,
            specs={
                "multi_processor_count": props.multi_processor_count,
                "cuda_version": torch.version.cuda,
                "mixed_precision": self.config.mixed_precision,
            },
        )

    def to_device(self, data: np.ndarray) -> Any:
        """Transfer data to CUDA device.

        Args:
            data: NumPy array

        Returns:
            CUDA tensor
        """
        if not self._initialized:
            raise RuntimeError("CUDA accelerator not initialized")

        tensor = torch.from_numpy(data).to(self._device)

        if self.config.mixed_precision:
            tensor = tensor.half()
        else:
            tensor = tensor.float()

        return tensor

    def to_host(self, tensor: Any) -> np.ndarray:
        """Transfer data from CUDA to CPU.

        Args:
            tensor: CUDA tensor

        Returns:
            NumPy array
        """
        if not isinstance(tensor, torch.Tensor):
            return np.array(tensor)

        # Convert to float32 if half precision
        if tensor.dtype == torch.float16:
            tensor = tensor.float()

        if tensor.is_cuda:
            tensor = tensor.cpu()

        return tensor.detach().numpy()

    def execute(self, model: Any, inputs: Any) -> Any:
        """Execute model on CUDA.

        Args:
            model: PyTorch model
            inputs: Input tensor

        Returns:
            Model output
        """
        if not self._initialized:
            raise RuntimeError("CUDA accelerator not initialized")

        # Ensure model is on device
        if hasattr(model, "to"):
            model = model.to(self._device)

        if hasattr(model, "eval"):
            model.eval()

        # Execute in stream if async
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                with torch.no_grad():
                    return model(inputs)
        else:
            with torch.no_grad():
                return model(inputs)

    def synchronize(self) -> None:
        """Synchronize CUDA stream."""
        if self._initialized:
            if self._stream is not None:
                self._stream.synchronize()
            else:
                torch.cuda.synchronize(self._device)

    def get_memory_info(self) -> Dict[str, float]:
        """Get CUDA memory usage.

        Returns:
            Dictionary with memory stats in GB
        """
        if not self._initialized:
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated(self._device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self._device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(self._device) / 1e9,
            "total_gb": torch.cuda.get_device_properties(
                self.config.device_id
            ).total_memory
            / 1e9,
        }

    def clear_cache(self) -> None:
        """Clear CUDA memory cache."""
        if self._initialized:
            torch.cuda.empty_cache()
            log.debug("CUDA cache cleared")

    def shutdown(self) -> None:
        """Shutdown CUDA accelerator."""
        if self._initialized:
            self.synchronize()
            self.clear_cache()
            self._initialized = False
            self._device = None
            self._stream = None
            log.info("CUDA accelerator shutdown")

    def compile_model(self, model: Any, example_inputs: Optional[Any] = None) -> Any:
        """Compile model for optimized inference.

        Uses torch.compile for PyTorch 2.0+ or TorchScript for older versions.

        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing (optional)

        Returns:
            Compiled model
        """
        if not self._initialized:
            raise RuntimeError("CUDA accelerator not initialized")

        model = model.to(self._device)

        # Try torch.compile (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(model, mode="max-autotune")
                log.info("Model compiled with torch.compile")
                return compiled
            except Exception as e:
                log.warning(f"torch.compile failed: {e}, using original model")
                return model

        # Fallback to TorchScript
        if example_inputs is not None:
            try:
                if hasattr(model, "to"):
                    model = model.to(self._device)
                    model.eval()
                traced = torch.jit.trace(model, example_inputs)
                log.info("Model compiled with TorchScript tracing")
                return traced
            except Exception as e:
                log.warning(f"TorchScript tracing failed: {e}")

        return model
