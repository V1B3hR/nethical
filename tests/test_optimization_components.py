# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Unit test suite for optimization components: ModelOptimizer, DynamicBatcher,
RequestOptimizer, and RequestCoalescer.
"""

import asyncio
import pytest

from nethical.optimization import (
    ModelOptimizer,
    DynamicBatcher,
    DynamicBatcherConfig,
    RequestOptimizer,
    RequestCoalescer,
)


class DummyModel:
    """Mock model for testing ModelOptimizer."""

    def __init__(self):
        self.weights = [1.0, 2.0, 3.0]


class TestModelOptimizer:
    """Test model quantization, pruning, distillation, and benchmarking."""

    def test_optimize_for_latency(self):
        model = DummyModel()
        optimized = ModelOptimizer.optimize_for_latency(
            model=model,
            target_latency_ms=25,
            quantization_bits=8,
            pruning_ratio=0.25,
            use_onnx=True,
        )

        assert getattr(optimized, "_quantized", False) is True
        assert getattr(optimized, "_quantization_bits", None) == 8
        assert getattr(optimized, "_pruned", False) is True
        assert getattr(optimized, "_pruning_ratio", None) == 0.25
        assert getattr(optimized, "_onnx_converted", False) is True

    def test_knowledge_distillation(self):
        teacher = DummyModel()
        student = DummyModel()
        distilled = ModelOptimizer.knowledge_distillation(
            teacher_model=teacher,
            student_model=student,
            training_data=None,
            temperature=2.5,
        )
        assert getattr(distilled, "_distilled", False) is True
        assert getattr(distilled, "_temperature", None) == 2.5

    def test_optimize_inference_config(self):
        cfg_t = ModelOptimizer.optimize_inference_config("transformer")
        assert cfg_t["max_sequence_length"] == 512
        assert cfg_t["use_fp16"] is True

        cfg_cnn = ModelOptimizer.optimize_inference_config("cnn")
        assert cfg_cnn["batch_size"] == 32
        assert cfg_cnn["use_cudnn_benchmarks"] is True

        cfg_unknown = ModelOptimizer.optimize_inference_config("custom_dnn")
        assert cfg_unknown["batch_size"] == 1

    def test_benchmark_model(self):
        model = DummyModel()
        bench = ModelOptimizer.benchmark_model(model, input_shape=(1, 4), num_iterations=10)
        assert "mean_ms" in bench
        assert "p95_ms" in bench
        assert bench["mean_ms"] > 0.0


@pytest.mark.asyncio
class TestDynamicBatcher:
    """Test DynamicBatcher batch accumulation and timeout handling."""

    async def test_batcher_max_size_trigger(self):
        config = DynamicBatcherConfig(max_batch_size=3, timeout_ms=500.0)
        batcher = DynamicBatcher(config)

        batches_processed = []

        async def mock_processor(batch):
            batches_processed.append(list(batch))
            return [x * 2 for x in batch]

        # Launch 3 requests concurrently
        results = await asyncio.gather(
            batcher.add_request(1, mock_processor),
            batcher.add_request(2, mock_processor),
            batcher.add_request(3, mock_processor),
        )

        assert results == [2, 4, 6]
        assert len(batches_processed) == 1
        assert batches_processed[0] == [1, 2, 3]

    async def test_batcher_timeout_flush(self):
        config = DynamicBatcherConfig(max_batch_size=10, timeout_ms=20.0)
        batcher = DynamicBatcher(config)

        async def mock_processor(batch):
            return [f"processed_{x}" for x in batch]

        res = await batcher.add_request("req1", mock_processor)
        assert res == "processed_req1"


@pytest.mark.asyncio
class TestRequestOptimizer:
    """Test RequestOptimizer caching and batch processing."""

    async def test_caching_and_metrics(self):
        optimizer = RequestOptimizer(cache_maxsize=100)

        call_count = 0

        async def processor(items):
            nonlocal call_count
            call_count += len(items)
            return [f"res_{x}" for x in items]

        # First request: cache miss
        r1 = await optimizer.process(["query_a", "query_b"], processor)
        assert r1 == ["res_query_a", "res_query_b"]
        assert call_count == 2

        # Second request: cache hit for query_a, new query_c
        r2 = await optimizer.process(["query_a", "query_c"], processor)
        assert r2 == ["res_query_a", "res_query_c"]
        assert call_count == 3  # Only query_c processed

        metrics = optimizer.get_metrics()
        assert metrics["total_requests"] == 4
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 3
        assert metrics["cache_size"] == 3

        optimizer.clear_cache()
        assert len(optimizer.cache) == 0


@pytest.mark.asyncio
class TestRequestCoalescer:
    """Test RequestCoalescer deduplication of concurrent requests."""

    async def test_concurrent_coalescing(self):
        coalescer = RequestCoalescer()
        call_count = 0

        async def slow_processor(req):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return f"result_for_{req}"

        # Dispatch 3 identical requests concurrently
        t1 = asyncio.create_task(coalescer.coalesce("shared_query", slow_processor))
        t2 = asyncio.create_task(coalescer.coalesce("shared_query", slow_processor))
        t3 = asyncio.create_task(coalescer.coalesce("shared_query", slow_processor))

        results = await asyncio.gather(t1, t2, t3)
        assert results == [
            "result_for_shared_query",
            "result_for_shared_query",
            "result_for_shared_query",
        ]
        # Should coalesce into a single execution
        assert call_count == 1
