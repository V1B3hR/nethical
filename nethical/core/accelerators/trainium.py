"""
AWS Trainium/Inferentia Accelerator for Nethical

This module provides Trainium/Inferentia acceleration for ML inference using torch-neuronx.
Supports AWS Neuron chips from Inferentia 1 to Trainium 3:

    Inferentia 1 (2019): 128 TOPS INT8, inference only
    Inferentia 2 (2022): 380 TOPS INT8, 32GB HBM, inference only
    Trainium 1 (2022): 420 TFLOPS BF16 / 0.84 PFLOPs FP8, 32GB HBM2
    Trainium 2 (2024): 787 TFLOPS BF16 / 1.575 PFLOPs FP8, 96GB HBM3
    Trainium 3 (2025): 1,260 TFLOPS BF16 / 2.52 PFLOPs FP8, 144GB HBM3e

Features:
    - Neuron SDK integration for optimized inference
    - Model compilation with Neuron compiler
    - Multi-chip support for large models
    - Automatic version detection and settings adjustment

Requirements:
    - torch-neuronx package
    - AWS Neuron SDK

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

from . import AcceleratorBackend, AcceleratorConfig, AcceleratorInfo, AcceleratorInterface

__all__ = [
    "TrainiumVersion",
    "TRAINIUM_SPECS",
    "TrainiumAccelerator",
    "is_trainium_available",
    "get_trainium_info",
    "detect_trainium_version",
]


class TrainiumVersion(str, Enum):
    """Supported AWS Trainium/Inferentia versions."""

    INFERENTIA_1 = "inf1"  # 2019, inference only
    INFERENTIA_2 = "inf2"  # 2022, inference only
    TRAINIUM_1 = "trn1"  # 2022, training + inference
    TRAINIUM_2 = "trn2"  # 2024, training + inference
    TRAINIUM_3 = "trn3"  # 2025, training + inference


@dataclass
class TrainiumSpecs:
    """Specifications for a Trainium/Inferentia version."""

    version: TrainiumVersion
    year: int
    tflops: float  # Peak TFLOPS
    fp8_pflops: Optional[float]  # FP8 PFLOPs if supported
    int8_tops: Optional[float]  # INT8 TOPS for Inferentia
    memory_gb: float
    memory_type: str
    supports_training: bool
    supports_fp8: bool
    max_batch_size: int
    neuron_cores: int  # NeuronCores per chip
    description: str


# Trainium/Inferentia specifications by version
TRAINIUM_SPECS: Dict[TrainiumVersion, TrainiumSpecs] = {
    TrainiumVersion.INFERENTIA_1: TrainiumSpecs(
        version=TrainiumVersion.INFERENTIA_1,
        year=2019,
        tflops=64,  # FP16
        fp8_pflops=None,
        int8_tops=128,
        memory_gb=8,
        memory_type="DRAM",
        supports_training=False,
        supports_fp8=False,
        max_batch_size=32,
        neuron_cores=4,
        description="AWS Inferentia 1 (2019, inference only)",
    ),
    TrainiumVersion.INFERENTIA_2: TrainiumSpecs(
        version=TrainiumVersion.INFERENTIA_2,
        year=2022,
        tflops=190,  # FP16
        fp8_pflops=None,
        int8_tops=380,
        memory_gb=32,
        memory_type="HBM",
        supports_training=False,
        supports_fp8=False,
        max_batch_size=128,
        neuron_cores=2,
        description="AWS Inferentia 2 (2022, inference only)",
    ),
    TrainiumVersion.TRAINIUM_1: TrainiumSpecs(
        version=TrainiumVersion.TRAINIUM_1,
        year=2022,
        tflops=420,  # BF16
        fp8_pflops=0.84,
        int8_tops=None,
        memory_gb=32,
        memory_type="HBM2",
        supports_training=True,
        supports_fp8=True,
        max_batch_size=256,
        neuron_cores=2,
        description="AWS Trainium 1 (2022)",
    ),
    TrainiumVersion.TRAINIUM_2: TrainiumSpecs(
        version=TrainiumVersion.TRAINIUM_2,
        year=2024,
        tflops=787,  # BF16
        fp8_pflops=1.575,
        int8_tops=None,
        memory_gb=96,
        memory_type="HBM3",
        supports_training=True,
        supports_fp8=True,
        max_batch_size=512,
        neuron_cores=2,
        description="AWS Trainium 2 (2024)",
    ),
    TrainiumVersion.TRAINIUM_3: TrainiumSpecs(
        version=TrainiumVersion.TRAINIUM_3,
        year=2025,
        tflops=1260,  # BF16
        fp8_pflops=2.52,
        int8_tops=None,
        memory_gb=144,
        memory_type="HBM3e",
        supports_training=True,
        supports_fp8=True,
        max_batch_size=1024,
        neuron_cores=4,
        description="AWS Trainium 3 (2025)",
    ),
}

log = logging.getLogger(__name__)

# Check for Neuron availability
NEURON_AVAILABLE = False
try:
    import torch
    import torch_neuronx

    NEURON_AVAILABLE = True
except ImportError:
    log.debug("torch-neuronx not available - Trainium acceleration disabled")


def detect_trainium_version() -> Optional[TrainiumVersion]:
    """Detect the Trainium/Inferentia version from runtime environment.

    Returns:
        TrainiumVersion enum or None if detection fails
    """
    if not NEURON_AVAILABLE:
        return None

    try:
        # Check environment variables first
        instance_type = os.environ.get("AWS_INSTANCE_TYPE", "").lower()
        neuron_device = os.environ.get("NEURON_RT_VISIBLE_CORES", "")

        # Match against known instance type patterns
        if "trn3" in instance_type:
            return TrainiumVersion.TRAINIUM_3
        elif "trn2" in instance_type:
            return TrainiumVersion.TRAINIUM_2
        elif "trn1" in instance_type:
            return TrainiumVersion.TRAINIUM_1
        elif "inf2" in instance_type:
            return TrainiumVersion.INFERENTIA_2
        elif "inf1" in instance_type:
            return TrainiumVersion.INFERENTIA_1

        # Try to detect from EC2 metadata
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-type",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
            )
            with urllib.request.urlopen(req, timeout=1) as response:
                instance_type = response.read().decode().lower()

            if "trn3" in instance_type:
                return TrainiumVersion.TRAINIUM_3
            elif "trn2" in instance_type:
                return TrainiumVersion.TRAINIUM_2
            elif "trn1" in instance_type:
                return TrainiumVersion.TRAINIUM_1
            elif "inf2" in instance_type:
                return TrainiumVersion.INFERENTIA_2
            elif "inf1" in instance_type:
                return TrainiumVersion.INFERENTIA_1
        except Exception:
            pass

        # Try to detect from Neuron SDK
        try:
            import torch_neuronx
            sdk_version = getattr(torch_neuronx, "__version__", "")
            # Newer SDK versions typically indicate newer hardware support
            # This is a heuristic fallback when instance type detection fails
            # SDK 3.x = Trainium 3, SDK 2.5+ = Trainium 2, SDK 2.x = Trainium 1
            if sdk_version:
                parts = sdk_version.split(".")
                major_version = int(parts[0])
                minor_version = int(parts[1]) if len(parts) > 1 else 0
                if major_version >= 3:
                    return TrainiumVersion.TRAINIUM_3
                elif major_version == 2 and minor_version >= 5:
                    return TrainiumVersion.TRAINIUM_2
                elif major_version >= 2:
                    return TrainiumVersion.TRAINIUM_1
        except Exception:
            pass

        # Default to Trainium 3 if detection fails
        log.warning("Trainium version detection inconclusive, assuming Trainium 3")
        return TrainiumVersion.TRAINIUM_3

    except Exception as e:
        log.warning(f"Trainium version detection failed: {e}")
        return None


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
        Dictionary with Trainium information including version and specs
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

        # Detect version
        trainium_version = detect_trainium_version()
        if trainium_version is None:
            trainium_version = TrainiumVersion.TRAINIUM_3  # Default assumption

        specs = TRAINIUM_SPECS[trainium_version]

        return {
            "available": True,
            "device_count": device_count,
            "version": trainium_version.value,
            "sdk_version": getattr(torch_neuronx, "__version__", "unknown"),
            "specs": {
                "description": specs.description,
                "year": specs.year,
                "tflops": specs.tflops,
                "fp8_pflops": specs.fp8_pflops,
                "int8_tops": specs.int8_tops,
                "memory_gb": specs.memory_gb,
                "memory_type": specs.memory_type,
                "supports_training": specs.supports_training,
                "supports_fp8": specs.supports_fp8,
                "neuron_cores": specs.neuron_cores,
                "max_batch_size": specs.max_batch_size,
            },
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


class TrainiumAccelerator(AcceleratorInterface):
    """AWS Trainium/Inferentia accelerator implementation.

    Provides Trainium-accelerated inference using torch-neuronx.
    Supports Inferentia 1/2 and Trainium 1/2/3 with automatic
    version detection and settings adjustment.
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize Trainium accelerator.

        Args:
            config: Accelerator configuration
        """
        super().__init__(config)
        self._compiled_models: Dict[int, Any] = {}
        self._neuron_device: Optional[str] = None
        self._trainium_version: Optional[TrainiumVersion] = None
        self._trainium_specs: Optional[TrainiumSpecs] = None

    @property
    def backend(self) -> AcceleratorBackend:
        return AcceleratorBackend.TRAINIUM

    @property
    def trainium_version(self) -> Optional[TrainiumVersion]:
        """Get detected Trainium/Inferentia version."""
        return self._trainium_version

    @property
    def trainium_specs(self) -> Optional[TrainiumSpecs]:
        """Get Trainium specifications."""
        return self._trainium_specs

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

            # Detect version
            self._trainium_version = detect_trainium_version()
            if self._trainium_version is None:
                self._trainium_version = TrainiumVersion.TRAINIUM_3
            self._trainium_specs = TRAINIUM_SPECS[self._trainium_version]

            # Configure Neuron based on version
            self._configure_for_trainium_version()

            self._initialized = True
            log.info(
                f"Trainium accelerator initialized: {self._trainium_version.value} "
                f"({self._trainium_specs.description})"
            )
            return True

        except Exception as e:
            log.error(f"Trainium initialization failed: {e}")
            return False

    def _configure_for_trainium_version(self) -> None:
        """Configure Neuron settings based on detected version."""
        if self._trainium_specs is None:
            return

        # Set compiler flags based on version
        neuron_cc_flags = []

        # Target specification
        if self._trainium_version == TrainiumVersion.INFERENTIA_1:
            neuron_cc_flags.append("--target=inf1")
        elif self._trainium_version == TrainiumVersion.INFERENTIA_2:
            neuron_cc_flags.append("--target=inf2")
        elif self._trainium_version == TrainiumVersion.TRAINIUM_1:
            neuron_cc_flags.append("--target=trn1")
        elif self._trainium_version == TrainiumVersion.TRAINIUM_2:
            neuron_cc_flags.append("--target=trn2")
        else:  # TRAINIUM_3
            neuron_cc_flags.append("--target=trn3")

        # FP8 for supporting versions
        if self._trainium_specs.supports_fp8 and self.config.mixed_precision:
            neuron_cc_flags.append("--auto-cast=all")
        elif not self._trainium_specs.supports_fp8:
            neuron_cc_flags.append("--auto-cast=matmul")

        # Batch size warning
        if self.config.batch_size > self._trainium_specs.max_batch_size:
            log.warning(
                f"Batch size {self.config.batch_size} exceeds recommendation "
                f"for {self._trainium_version.value} (max: {self._trainium_specs.max_batch_size}). "
                f"Consider reducing batch size."
            )

        # Apply Neuron flags
        if neuron_cc_flags:
            existing_flags = os.environ.get("NEURON_CC_FLAGS", "")
            new_flags = " ".join(neuron_cc_flags)
            if existing_flags:
                os.environ["NEURON_CC_FLAGS"] = f"{existing_flags} {new_flags}"
            else:
                os.environ["NEURON_CC_FLAGS"] = new_flags

        log.debug(f"Neuron configured for {self._trainium_version.value}")

    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size for current version.

        Returns:
            Recommended maximum batch size
        """
        if self._trainium_specs:
            return self._trainium_specs.max_batch_size
        return 32

    def supports_training(self) -> bool:
        """Check if current device supports training.

        Returns:
            True if training is supported (Trainium), False for Inferentia
        """
        if self._trainium_specs:
            return self._trainium_specs.supports_training
        return True  # Assume Trainium

    def get_info(self) -> AcceleratorInfo:
        """Get Trainium device information."""
        if not self._initialized or self._trainium_specs is None:
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
            device_name=self._trainium_specs.description,
            total_memory_gb=self._trainium_specs.memory_gb,
            compute_capability=self._trainium_version.value,
            is_available=True,
            specs={
                "version": self._trainium_version.value,
                "year": self._trainium_specs.year,
                "tflops": self._trainium_specs.tflops,
                "fp8_pflops": self._trainium_specs.fp8_pflops,
                "int8_tops": self._trainium_specs.int8_tops,
                "memory_type": self._trainium_specs.memory_type,
                "supports_training": self._trainium_specs.supports_training,
                "supports_fp8": self._trainium_specs.supports_fp8,
                "neuron_cores": self._trainium_specs.neuron_cores,
                "max_batch_size": self._trainium_specs.max_batch_size,
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

        memory_gb = self._trainium_specs.memory_gb if self._trainium_specs else 0.0

        return {
            "total_gb": memory_gb,
            "version": self._trainium_version.value if self._trainium_version else "unknown",
            "supports_training": self._trainium_specs.supports_training if self._trainium_specs else False,
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
