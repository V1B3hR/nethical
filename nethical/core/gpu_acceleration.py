"""
GPU acceleration support for ML inference.

This module provides GPU acceleration for ML model inference using
PyTorch CUDA support when available.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available. Install with: pip install torch\n"
        "GPU acceleration will be disabled."
    )


def is_gpu_available() -> bool:
    """
    Check if GPU is available for acceleration.

    Returns:
        True if CUDA-enabled GPU is available
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU device information.

    Returns:
        Dictionary with GPU information
    """
    if not TORCH_AVAILABLE:
        return {"available": False, "reason": "PyTorch not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}

    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": torch.cuda.get_device_capability(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cuda_version": torch.version.cuda,
    }


class GPUAcceleratedInference:
    """
    GPU-accelerated inference engine.

    Features:
    - Automatic GPU detection and fallback to CPU
    - Batch processing for efficiency
    - Memory management
    - Mixed precision support (FP16)
    """

    def __init__(
        self,
        use_gpu: bool = True,
        device_id: int = 0,
        mixed_precision: bool = False,
        batch_size: int = 32,
    ):
        """
        Initialize GPU-accelerated inference engine.

        Args:
            use_gpu: Whether to use GPU if available
            device_id: GPU device ID to use
            mixed_precision: Whether to use FP16 for faster inference
            batch_size: Batch size for inference
        """
        self.use_gpu = use_gpu and is_gpu_available()
        self.device_id = device_id
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size

        if TORCH_AVAILABLE:
            if self.use_gpu:
                self.device = torch.device(f"cuda:{device_id}")
                logger.info(
                    f"GPU acceleration enabled on device {device_id}: {torch.cuda.get_device_name(device_id)}"
                )
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for inference")
        else:
            self.device = None
            logger.warning("PyTorch not available, GPU acceleration disabled")

    def to_tensor(self, data: np.ndarray) -> Any:
        """
        Convert numpy array to tensor on appropriate device.

        Args:
            data: Numpy array

        Returns:
            Tensor on appropriate device
        """
        if not TORCH_AVAILABLE:
            return data

        tensor = torch.from_numpy(data).float()

        if self.device is not None:
            tensor = tensor.to(self.device)

        if self.mixed_precision and self.use_gpu:
            tensor = tensor.half()

        return tensor

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert tensor to numpy array.

        Args:
            tensor: PyTorch tensor

        Returns:
            Numpy array
        """
        if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
            return tensor

        if self.mixed_precision and self.use_gpu:
            tensor = tensor.float()

        if tensor.is_cuda:
            tensor = tensor.cpu()

        return tensor.detach().numpy()

    def batch_inference(
        self, model: Any, inputs: np.ndarray, batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform batch inference with automatic batching.

        Args:
            model: PyTorch model
            inputs: Input array (n_samples, ...)
            batch_size: Batch size (uses default if None)

        Returns:
            Output array
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        batch_size = batch_size or self.batch_size
        n_samples = len(inputs)

        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Process in batches
        outputs = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_inputs = inputs[i : i + batch_size]
                batch_tensor = self.to_tensor(batch_inputs)

                # Forward pass
                batch_output = model(batch_tensor)

                # Convert back to numpy
                batch_output_np = self.to_numpy(batch_output)
                outputs.append(batch_output_np)

        # Concatenate all batches
        return np.concatenate(outputs, axis=0)

    def predict_single(self, model: Any, input_data: np.ndarray) -> np.ndarray:
        """
        Predict single sample.

        Args:
            model: PyTorch model
            input_data: Input array

        Returns:
            Output array
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        # Add batch dimension if needed
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            tensor = self.to_tensor(input_data)
            output = model(tensor)
            return self.to_numpy(output)

    def benchmark_throughput(
        self, model: Any, input_shape: tuple, num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference throughput.

        Args:
            model: PyTorch model
            input_shape: Shape of input data
            num_iterations: Number of iterations for benchmarking

        Returns:
            Dictionary with throughput metrics
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        import time

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        model = model.to(self.device)
        model.eval()

        # Warmup
        for _ in range(10):
            _ = self.predict_single(model, dummy_input)

        # Synchronize GPU
        if self.use_gpu:
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.predict_single(model, dummy_input)

        if self.use_gpu:
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time

        throughput = num_iterations / elapsed_time
        latency_ms = (elapsed_time / num_iterations) * 1000

        return {
            "throughput_samples_per_sec": throughput,
            "latency_ms": latency_ms,
            "device": str(self.device),
            "mixed_precision": self.mixed_precision,
            "iterations": num_iterations,
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage.

        Returns:
            Dictionary with memory usage in GB
        """
        if not TORCH_AVAILABLE or not self.use_gpu:
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
        }

    def clear_cache(self):
        """Clear GPU cache."""
        if TORCH_AVAILABLE and self.use_gpu:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


class GPUBatchProcessor:
    """
    GPU-accelerated batch processor for governance operations.

    Optimized for processing multiple actions in parallel on GPU.
    """

    def __init__(self, use_gpu: bool = True, batch_size: int = 64, mixed_precision: bool = False):
        """
        Initialize GPU batch processor.

        Args:
            use_gpu: Whether to use GPU if available
            batch_size: Batch size for processing
            mixed_precision: Whether to use FP16
        """
        self.inference_engine = GPUAcceleratedInference(
            use_gpu=use_gpu, mixed_precision=mixed_precision, batch_size=batch_size
        )

        self.batch_size = batch_size

    def process_risk_scores_batch(self, features: np.ndarray, model: Any) -> np.ndarray:
        """
        Process batch of risk score predictions.

        Args:
            features: Feature array (n_samples, n_features)
            model: Risk scoring model

        Returns:
            Risk scores array (n_samples,)
        """
        return self.inference_engine.batch_inference(model, features)

    def process_embeddings_batch(
        self, texts: List[str], embedding_model: Any, tokenizer: Any
    ) -> np.ndarray:
        """
        Process batch of text embeddings.

        Args:
            texts: List of text strings
            embedding_model: Embedding model
            tokenizer: Tokenizer

        Returns:
            Embeddings array (n_samples, embedding_dim)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        # Tokenize
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Move to device
        tokens = {k: v.to(self.inference_engine.device) for k, v in tokens.items()}

        # Generate embeddings
        embedding_model = embedding_model.to(self.inference_engine.device)
        embedding_model.eval()

        with torch.no_grad():
            outputs = embedding_model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        return self.inference_engine.to_numpy(embeddings)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            "gpu_available": is_gpu_available(),
            "gpu_enabled": self.inference_engine.use_gpu,
            "batch_size": self.batch_size,
            "mixed_precision": self.inference_engine.mixed_precision,
        }

        if is_gpu_available():
            stats["gpu_info"] = get_gpu_info()
            stats["memory_usage"] = self.inference_engine.get_memory_usage()

        return stats
