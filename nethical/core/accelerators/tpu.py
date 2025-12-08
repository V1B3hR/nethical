"""
Google TPU Accelerator for Nethical

This module provides TPU acceleration for ML inference using torch_xla.
Supports Google Cloud TPU generations from v2 to v7:

    TPU v2 (2017): 45 TFLOPS, 16GB HBM
    TPU v3 (2018): 105 TFLOPS, 32GB HBM
    TPU v4 (2021): 275 TFLOPS, 32GB HBM2
    TPU v5e (2023): ~200 TFLOPS, 16GB HBM2e (cost-optimized)
    TPU v5p (2023): ~450 TFLOPS, 95GB HBM2e (performance-optimized)
    TPU v7 (2025): 4,614 FP8 TFLOPS, 192GB HBM3E (Ironwood)

Features:
    - XLA compilation for optimized execution
    - TPU-optimized batch processing
    - Multi-TPU pod support
    - Automatic XLA graph tracing
    - Automatic version detection and settings adjustment

Requirements:
    - torch_xla package
    - Google Cloud TPU runtime

Fundamental Laws Alignment:
    - Law 23 (Fail-Safe Design): Graceful fallback to CPU
    - Law 15 (Audit Compliance): Performance metrics logging

Author: Nethical Core Team
Version: 1.1.0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from . import (
    AcceleratorBackend,
    AcceleratorConfig,
    AcceleratorInfo,
    AcceleratorInterface,
)

__all__ = [
    "TPUVersion",
    "TPU_SPECS",
    "TPUAccelerator",
    "is_tpu_available",
    "get_tpu_info",
    "detect_tpu_version",
]


class TPUVersion(str, Enum):
    """Supported TPU versions."""

    TPU_V2 = "v2"  # 2017, 45 TFLOPS
    TPU_V3 = "v3"  # 2018, 105 TFLOPS
    TPU_V4 = "v4"  # 2021, 275 TFLOPS
    TPU_V5E = "v5e"  # 2023, ~200 TFLOPS (cost-optimized)
    TPU_V5P = "v5p"  # 2023, ~450 TFLOPS (performance-optimized)
    TPU_V7 = "v7"  # 2025, 4614 FP8 TFLOPS (Ironwood)


@dataclass
class TPUSpecs:
    """Specifications for a TPU version."""

    version: TPUVersion
    year: int
    tflops: float  # Peak TFLOPS (FP32 unless noted)
    fp8_tflops: Optional[float]  # FP8 TFLOPS if supported
    memory_gb: float
    memory_type: str
    supports_fp8: bool
    supports_bf16: bool
    max_batch_size: int  # Recommended max batch size
    description: str


# TPU specifications by version
TPU_SPECS: Dict[TPUVersion, TPUSpecs] = {
    TPUVersion.TPU_V2: TPUSpecs(
        version=TPUVersion.TPU_V2,
        year=2017,
        tflops=45,
        fp8_tflops=None,
        memory_gb=16,
        memory_type="HBM",
        supports_fp8=False,
        supports_bf16=True,
        max_batch_size=64,
        description="Google TPU v2 (2017)",
    ),
    TPUVersion.TPU_V3: TPUSpecs(
        version=TPUVersion.TPU_V3,
        year=2018,
        tflops=105,
        fp8_tflops=None,
        memory_gb=32,
        memory_type="HBM",
        supports_fp8=False,
        supports_bf16=True,
        max_batch_size=128,
        description="Google TPU v3 (2018)",
    ),
    TPUVersion.TPU_V4: TPUSpecs(
        version=TPUVersion.TPU_V4,
        year=2021,
        tflops=275,
        fp8_tflops=None,
        memory_gb=32,
        memory_type="HBM2",
        supports_fp8=False,
        supports_bf16=True,
        max_batch_size=256,
        description="Google TPU v4 (2021)",
    ),
    TPUVersion.TPU_V5E: TPUSpecs(
        version=TPUVersion.TPU_V5E,
        year=2023,
        tflops=200,
        fp8_tflops=None,
        memory_gb=16,
        memory_type="HBM2e",
        supports_fp8=False,
        supports_bf16=True,
        max_batch_size=128,
        description="Google TPU v5e (2023, cost-optimized)",
    ),
    TPUVersion.TPU_V5P: TPUSpecs(
        version=TPUVersion.TPU_V5P,
        year=2023,
        tflops=450,
        fp8_tflops=900,
        memory_gb=95,
        memory_type="HBM2e",
        supports_fp8=True,
        supports_bf16=True,
        max_batch_size=512,
        description="Google TPU v5p (2023, performance-optimized)",
    ),
    TPUVersion.TPU_V7: TPUSpecs(
        version=TPUVersion.TPU_V7,
        year=2025,
        tflops=2307,  # FP32 equivalent
        fp8_tflops=4614,
        memory_gb=192,
        memory_type="HBM3E",
        supports_fp8=True,
        supports_bf16=True,
        max_batch_size=1024,
        description="Google TPU v7 Ironwood (2025)",
    ),
}

log = logging.getLogger(__name__)

# Check for torch_xla availability
XLA_AVAILABLE = False
try:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError:
    log.debug("torch_xla not available - TPU acceleration disabled")


def detect_tpu_version() -> Optional[TPUVersion]:
    """Detect the TPU version from runtime environment.

    Returns:
        TPUVersion enum or None if detection fails
    """
    if not XLA_AVAILABLE:
        return None

    try:
        # Check environment variables first (most reliable)
        tpu_type = os.environ.get("TPU_ACCELERATOR_TYPE", "").lower()
        tpu_name = os.environ.get("TPU_NAME", "").lower()

        # Match against known patterns
        if "v7" in tpu_type or "ironwood" in tpu_name or "v7" in tpu_name:
            return TPUVersion.TPU_V7
        elif "v5p" in tpu_type or "v5p" in tpu_name:
            return TPUVersion.TPU_V5P
        elif "v5e" in tpu_type or "v5litepod" in tpu_type or "v5e" in tpu_name:
            return TPUVersion.TPU_V5E
        elif "v4" in tpu_type or "v4" in tpu_name:
            return TPUVersion.TPU_V4
        elif "v3" in tpu_type or "v3" in tpu_name:
            return TPUVersion.TPU_V3
        elif "v2" in tpu_type or "v2" in tpu_name:
            return TPUVersion.TPU_V2

        # Try to detect from XLA device info
        try:
            device = xm.xla_device()
            device_str = str(device).lower()

            if "v7" in device_str:
                return TPUVersion.TPU_V7
            elif "v5p" in device_str:
                return TPUVersion.TPU_V5P
            elif "v5e" in device_str:
                return TPUVersion.TPU_V5E
            elif "v4" in device_str:
                return TPUVersion.TPU_V4
            elif "v3" in device_str:
                return TPUVersion.TPU_V3
            elif "v2" in device_str:
                return TPUVersion.TPU_V2
        except Exception:
            pass

        # Default to v7 if TPU is available but version unknown
        # (Most likely running on latest hardware)
        log.warning("TPU version detection inconclusive, assuming TPU v7")
        return TPUVersion.TPU_V7

    except Exception as e:
        log.warning(f"TPU version detection failed: {e}")
        return None


def is_tpu_available() -> bool:
    """Check if TPU is available.

    Returns:
        True if TPU device is available
    """
    if not XLA_AVAILABLE:
        return False

    try:
        device = xm.xla_device()
        device_type = xm.xla_device_hw(device)
        return device_type == "TPU"
    except Exception:
        return False


def get_tpu_info() -> Dict[str, Any]:
    """Get TPU device information.

    Returns:
        Dictionary with TPU information including version and specs
    """
    if not XLA_AVAILABLE:
        return {"available": False, "reason": "torch_xla not installed"}

    try:
        device = xm.xla_device()
        device_type = xm.xla_device_hw(device)

        if device_type != "TPU":
            return {"available": False, "reason": f"Device is {device_type}, not TPU"}

        # Detect TPU version
        tpu_version = detect_tpu_version()
        if tpu_version is None:
            tpu_version = TPUVersion.TPU_V7  # Default assumption

        specs = TPU_SPECS[tpu_version]

        # Get runtime info
        runtime_info = {}
        try:
            import torch_xla.runtime as xr

            runtime_info = {
                "world_size": xr.world_size(),
                "global_ordinal": xr.global_ordinal(),
            }
        except Exception:
            pass

        return {
            "available": True,
            "device_type": device_type,
            "device": str(device),
            "version": tpu_version.value,
            "runtime": runtime_info,
            "specs": {
                "description": specs.description,
                "year": specs.year,
                "tflops": specs.tflops,
                "fp8_tflops": specs.fp8_tflops,
                "memory_gb": specs.memory_gb,
                "memory_type": specs.memory_type,
                "supports_fp8": specs.supports_fp8,
                "supports_bf16": specs.supports_bf16,
                "max_batch_size": specs.max_batch_size,
            },
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


class TPUAccelerator(AcceleratorInterface):
    """Google TPU accelerator implementation.

    Provides TPU-accelerated inference using torch_xla.
    Supports TPU v2 through v7 with automatic version detection
    and settings adjustment for optimal performance.
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize TPU accelerator.

        Args:
            config: Accelerator configuration
        """
        super().__init__(config)
        self._compiled_models: Dict[int, Any] = {}
        self._tpu_version: Optional[TPUVersion] = None
        self._tpu_specs: Optional[TPUSpecs] = None

    @property
    def backend(self) -> AcceleratorBackend:
        return AcceleratorBackend.TPU

    @property
    def tpu_version(self) -> Optional[TPUVersion]:
        """Get detected TPU version."""
        return self._tpu_version

    @property
    def tpu_specs(self) -> Optional[TPUSpecs]:
        """Get TPU specifications."""
        return self._tpu_specs

    def initialize(self) -> bool:
        """Initialize TPU device.

        Returns:
            True if initialization successful
        """
        if not is_tpu_available():
            log.warning("TPU not available")
            return False

        try:
            self._device = xm.xla_device()

            # Detect TPU version
            self._tpu_version = detect_tpu_version()
            if self._tpu_version is None:
                self._tpu_version = TPUVersion.TPU_V7
            self._tpu_specs = TPU_SPECS[self._tpu_version]

            # Configure XLA based on TPU version
            self._configure_for_tpu_version()

            self._initialized = True
            log.info(
                f"TPU accelerator initialized: {self._device} "
                f"(version: {self._tpu_version.value}, "
                f"{self._tpu_specs.tflops} TFLOPS)"
            )
            return True

        except Exception as e:
            log.error(f"TPU initialization failed: {e}")
            return False

    def _configure_for_tpu_version(self) -> None:
        """Configure XLA settings based on detected TPU version."""
        if self._tpu_specs is None:
            return

        # Set XLA flags based on TPU version
        xla_flags = []

        # FP8 support for newer TPUs
        if self._tpu_specs.supports_fp8 and self.config.mixed_precision:
            xla_flags.append("--xla_enable_fp8=true")
        else:
            xla_flags.append("--xla_enable_fp8=false")

        # Memory optimization for smaller TPUs
        if self._tpu_specs.memory_gb < 32:
            xla_flags.append("--xla_tpu_memory_bound_schedule=true")

        # Set batch size recommendation
        if self.config.batch_size > self._tpu_specs.max_batch_size:
            log.warning(
                f"Batch size {self.config.batch_size} exceeds recommendation "
                f"for {self._tpu_version.value} (max: {self._tpu_specs.max_batch_size}). "
                f"Consider reducing batch size for optimal performance."
            )

        # Apply XLA flags
        if xla_flags:
            existing_flags = os.environ.get("XLA_FLAGS", "")
            new_flags = " ".join(xla_flags)
            if existing_flags:
                os.environ["XLA_FLAGS"] = f"{existing_flags} {new_flags}"
            else:
                os.environ["XLA_FLAGS"] = new_flags

        log.debug(f"XLA configured for TPU {self._tpu_version.value}")

    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size for current TPU version.

        Returns:
            Recommended maximum batch size
        """
        if self._tpu_specs:
            return self._tpu_specs.max_batch_size
        return 32  # Conservative default

    def get_info(self) -> AcceleratorInfo:
        """Get TPU device information."""
        if not self._initialized or self._tpu_specs is None:
            return AcceleratorInfo(
                backend=AcceleratorBackend.TPU,
                device_id=self.config.device_id,
                device_name="Not initialized",
                total_memory_gb=0.0,
                compute_capability="N/A",
                is_available=False,
            )

        return AcceleratorInfo(
            backend=AcceleratorBackend.TPU,
            device_id=self.config.device_id,
            device_name=self._tpu_specs.description,
            total_memory_gb=self._tpu_specs.memory_gb,
            compute_capability=f"TPU {self._tpu_version.value}",
            is_available=True,
            specs={
                "version": self._tpu_version.value,
                "year": self._tpu_specs.year,
                "tflops": self._tpu_specs.tflops,
                "fp8_tflops": self._tpu_specs.fp8_tflops,
                "memory_type": self._tpu_specs.memory_type,
                "supports_fp8": self._tpu_specs.supports_fp8,
                "supports_bf16": self._tpu_specs.supports_bf16,
                "max_batch_size": self._tpu_specs.max_batch_size,
                "device": str(self._device),
            },
        )

    def to_device(self, data: np.ndarray) -> Any:
        """Transfer data to TPU device.

        Args:
            data: NumPy array

        Returns:
            TPU tensor
        """
        if not self._initialized:
            raise RuntimeError("TPU accelerator not initialized")

        import torch

        tensor = torch.from_numpy(data).float()
        return tensor.to(self._device)

    def to_host(self, tensor: Any) -> np.ndarray:
        """Transfer data from TPU to CPU.

        Args:
            tensor: TPU tensor

        Returns:
            NumPy array
        """
        import torch

        if not isinstance(tensor, torch.Tensor):
            return np.array(tensor)

        # Move to CPU and convert
        return tensor.cpu().detach().numpy()

    def execute(self, model: Any, inputs: Any) -> Any:
        """Execute model on TPU.

        Args:
            model: PyTorch model
            inputs: Input tensor

        Returns:
            Model output
        """
        if not self._initialized:
            raise RuntimeError("TPU accelerator not initialized")

        import torch

        # Ensure model is on device
        if hasattr(model, "to"):
            model = model.to(self._device)

        if hasattr(model, "eval"):
            model.eval()

        # Execute with no grad
        with torch.no_grad():
            output = model(inputs)

        # Mark step for XLA graph execution
        xm.mark_step()

        return output

    def synchronize(self) -> None:
        """Synchronize TPU execution."""
        if self._initialized:
            xm.mark_step()

    def get_memory_info(self) -> Dict[str, float]:
        """Get TPU memory usage.

        Returns:
            Dictionary with memory stats in GB
        """
        if not self._initialized:
            return {}

        memory_gb = self._tpu_specs.memory_gb if self._tpu_specs else 0.0

        # TPU memory info is limited via torch_xla
        try:
            import torch_xla.debug.metrics as met

            metrics = met.metrics_report()
            return {
                "total_gb": memory_gb,
                "version": self._tpu_version.value if self._tpu_version else "unknown",
                "metrics_available": bool(metrics),
            }
        except Exception:
            return {"total_gb": memory_gb}

    def clear_cache(self) -> None:
        """Clear XLA compilation cache."""
        if self._initialized:
            self._compiled_models.clear()
            # Clear XLA caches if available
            try:
                import torch_xla

                torch_xla._XLAC._xla_step_marker(
                    torch_xla._XLAC._xla_get_default_device(),
                    [],
                    wait=True,
                )
            except Exception:
                pass
            log.debug("TPU/XLA cache cleared")

    def shutdown(self) -> None:
        """Shutdown TPU accelerator."""
        if self._initialized:
            self.synchronize()
            self.clear_cache()
            self._initialized = False
            self._device = None
            log.info("TPU accelerator shutdown")

    def compile_model(self, model: Any, example_inputs: Optional[Any] = None) -> Any:
        """Compile model for TPU with XLA.

        Uses XLA tracing for optimized graph execution.

        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing

        Returns:
            XLA-optimized model
        """
        if not self._initialized:
            raise RuntimeError("TPU accelerator not initialized")

        import torch

        model = model.to(self._device)

        if hasattr(model, "eval"):
            model.eval()

        # Trace model with XLA
        if example_inputs is not None:
            try:
                # Move inputs to device
                if hasattr(example_inputs, "to"):
                    example_inputs = example_inputs.to(self._device)

                # Warmup run for XLA compilation
                with torch.no_grad():
                    _ = model(example_inputs)
                xm.mark_step()

                log.info("Model compiled with XLA tracing")

            except Exception as e:
                log.warning(f"XLA tracing failed: {e}")

        return model

    def batch_execute(
        self,
        model: Any,
        inputs: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Execute batch inference with TPU-optimized batching.

        Uses XLA's efficient batch processing capabilities.

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

        # Final synchronization
        xm.mark_step()

        return np.concatenate(outputs, axis=0)
