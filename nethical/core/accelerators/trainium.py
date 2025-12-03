"""
AWS Trainium3 Accelerator for Nethical

This module provides Trainium3 acceleration for ML inference using torch-neuronx.
Supports AWS Trainium3 chips with exceptional performance:
    - 2.52 PFLOPs FP8
    - 144GB HBM3e memory

Features:
    - Neuron SDK integration for optimized inference
    - Model compilation with Neuron compiler
    - Multi-chip support for large models
    - AWS Inferentia2/Trainium3 compatibility

Requirements:
    - torch-neuronx package
    - AWS Neuron SDK

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

from . import AcceleratorBackend, AcceleratorConfig, AcceleratorInfo, AcceleratorInterface

__all__ = [
    "TrainiumAccelerator",
    "is_trainium_available",
    "get_trainium_info",
]

log = logging.getLogger(__name__)

# Check for Neuron availability
NEURON_AVAILABLE = False
try:
    import torch
    import torch_neuronx

    NEURON_AVAILABLE = True
except ImportError:
    log.debug("torch-neuronx not available - Trainium acceleration disabled")


def is_trainium_available() -> bool:
    """Check if Trainium/Inferentia is available.

    Returns:
        True if Neuron device is available
    """
    if not NEURON_AVAILABLE:
        return False

    try:
        # Check for Neuron devices
        import torch_neuronx

        # Try to get device count
        device_count = torch_neuronx.xla_impl.data_parallel.device_count()
        return device_count > 0
    except Exception:
        # Fallback: check if neuronx_cc is available
        try:
            import neuronx_cc  # noqa: F401

            return True
        except ImportError:
            return False


def get_trainium_info() -> Dict[str, Any]:
    """Get Trainium device information.

    Returns:
        Dictionary with Trainium information
    """
    if not NEURON_AVAILABLE:
        return {"available": False, "reason": "torch-neuronx not installed"}

    try:
        import torch_neuronx

        device_count = 1  # Default
        try:
            device_count = torch_neuronx.xla_impl.data_parallel.device_count()
        except Exception:
            pass

        return {
            "available": True,
            "device_count": device_count,
            "sdk_version": getattr(torch_neuronx, "__version__", "unknown"),
            "specs": {
                "description": "AWS Trainium3",
                "fp8_pflops": 2.52,
                "memory_gb": 144,
                "memory_type": "HBM3e",
            },
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


class TrainiumAccelerator(AcceleratorInterface):
    """AWS Trainium3 accelerator implementation.

    Provides Trainium-accelerated inference using torch-neuronx.
    Optimized for AWS Trainium3 with Neuron SDK compilation.
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize Trainium accelerator.

        Args:
            config: Accelerator configuration
        """
        super().__init__(config)
        self._compiled_models: Dict[int, Any] = {}
        self._neuron_device: Optional[str] = None

    @property
    def backend(self) -> AcceleratorBackend:
        return AcceleratorBackend.TRAINIUM

    def initialize(self) -> bool:
        """Initialize Trainium device.

        Returns:
            True if initialization successful
        """
        if not is_trainium_available():
            log.warning("Trainium not available")
            return False

        try:
            import torch

            # Set Neuron device
            self._neuron_device = "xla"
            self._device = torch.device(self._neuron_device)

            # Configure Neuron runtime
            import os

            # Performance optimizations
            os.environ.setdefault("NEURON_CC_FLAGS", "--target=trn1 --auto-cast=all")

            self._initialized = True
            log.info("Trainium accelerator initialized")
            return True

        except Exception as e:
            log.error(f"Trainium initialization failed: {e}")
            return False

    def get_info(self) -> AcceleratorInfo:
        """Get Trainium device information."""
        if not self._initialized:
            return AcceleratorInfo(
                backend=AcceleratorBackend.TRAINIUM,
                device_id=self.config.device_id,
                device_name="Not initialized",
                total_memory_gb=0.0,
                compute_capability="N/A",
                is_available=False,
            )

        return AcceleratorInfo(
            backend=AcceleratorBackend.TRAINIUM,
            device_id=self.config.device_id,
            device_name="AWS Trainium3",
            total_memory_gb=144.0,  # Trainium3 spec
            compute_capability="Trainium3",
            is_available=True,
            specs={
                "fp8_pflops": 2.52,
                "memory_type": "HBM3e",
                "device": str(self._device),
            },
        )

    def to_device(self, data: np.ndarray) -> Any:
        """Transfer data to Trainium device.

        Args:
            data: NumPy array

        Returns:
            Neuron tensor
        """
        if not self._initialized:
            raise RuntimeError("Trainium accelerator not initialized")

        import torch

        tensor = torch.from_numpy(data).float()
        return tensor.to(self._device)

    def to_host(self, tensor: Any) -> np.ndarray:
        """Transfer data from Trainium to CPU.

        Args:
            tensor: Neuron tensor

        Returns:
            NumPy array
        """
        import torch

        if not isinstance(tensor, torch.Tensor):
            return np.array(tensor)

        # Move to CPU and convert
        return tensor.cpu().detach().numpy()

    def execute(self, model: Any, inputs: Any) -> Any:
        """Execute model on Trainium.

        Args:
            model: PyTorch model (preferably Neuron-compiled)
            inputs: Input tensor

        Returns:
            Model output
        """
        if not self._initialized:
            raise RuntimeError("Trainium accelerator not initialized")

        import torch

        # Ensure model is on device
        if hasattr(model, "to"):
            model = model.to(self._device)

        if hasattr(model, "eval"):
            model.eval()

        # Execute with no grad
        with torch.no_grad():
            output = model(inputs)

        return output

    def synchronize(self) -> None:
        """Synchronize Trainium execution."""
        if self._initialized:
            try:
                import torch_xla.core.xla_model as xm

                xm.mark_step()
            except ImportError:
                pass

    def get_memory_info(self) -> Dict[str, float]:
        """Get Trainium memory usage.

        Returns:
            Dictionary with memory stats in GB
        """
        if not self._initialized:
            return {}

        return {
            "total_gb": 144.0,  # Trainium3 spec
        }

    def clear_cache(self) -> None:
        """Clear Neuron compilation cache."""
        if self._initialized:
            self._compiled_models.clear()
            log.debug("Trainium/Neuron cache cleared")

    def shutdown(self) -> None:
        """Shutdown Trainium accelerator."""
        if self._initialized:
            self.synchronize()
            self.clear_cache()
            self._initialized = False
            self._device = None
            log.info("Trainium accelerator shutdown")

    def compile_model(
        self,
        model: Any,
        example_inputs: Optional[Any] = None,
        dynamic_batch_size: bool = False,
    ) -> Any:
        """Compile model for Trainium with Neuron SDK.

        Uses torch_neuronx.trace for optimized graph execution.

        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing (required for compilation)
            dynamic_batch_size: Enable dynamic batch size support

        Returns:
            Neuron-compiled model
        """
        if not self._initialized:
            raise RuntimeError("Trainium accelerator not initialized")

        if example_inputs is None:
            log.warning("Example inputs required for Neuron compilation")
            return model

        import torch
        import torch_neuronx

        try:
            # Move model to CPU for tracing
            model_cpu = model.cpu()

            if hasattr(model_cpu, "eval"):
                model_cpu.eval()

            # Ensure inputs are on CPU for tracing
            if hasattr(example_inputs, "cpu"):
                example_inputs = example_inputs.cpu()

            # Trace with Neuron compiler
            traced = torch_neuronx.trace(
                model_cpu,
                example_inputs,
            )

            log.info("Model compiled with Neuron SDK")

            # Cache compiled model
            model_id = id(model)
            self._compiled_models[model_id] = traced

            return traced

        except Exception as e:
            log.warning(f"Neuron compilation failed: {e}, using original model")
            return model

    def batch_execute(
        self,
        model: Any,
        inputs: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Execute batch inference with Trainium-optimized batching.

        Args:
            model: Model to execute
            inputs: Input array (n_samples, ...)
            batch_size: Batch size (uses config default if None)

        Returns:
            Output array
        """
        import torch

        batch_size = batch_size or self.config.batch_size
        n_samples = len(inputs)
        outputs = []

        # Move model to device once
        if hasattr(model, "to"):
            model = model.to(self._device)

        if hasattr(model, "eval"):
            model.eval()

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = inputs[i : i + batch_size]
                batch_tensor = self.to_device(batch)
                result = model(batch_tensor)
                outputs.append(self.to_host(result))

        return np.concatenate(outputs, axis=0)

    def save_compiled_model(self, model: Any, path: str) -> None:
        """Save a Neuron-compiled model to disk.

        Args:
            model: Neuron-compiled model
            path: Path to save the model
        """
        import torch

        try:
            torch.jit.save(model, path)
            log.info(f"Compiled model saved to {path}")
        except Exception as e:
            log.error(f"Failed to save compiled model: {e}")
            raise

    def load_compiled_model(self, path: str) -> Any:
        """Load a Neuron-compiled model from disk.

        Args:
            path: Path to the compiled model

        Returns:
            Loaded model
        """
        import torch

        try:
            model = torch.jit.load(path)
            log.info(f"Compiled model loaded from {path}")
            return model
        except Exception as e:
            log.error(f"Failed to load compiled model: {e}")
            raise
