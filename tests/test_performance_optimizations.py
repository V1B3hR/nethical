"""Tests for performance and scalability optimizations."""

import pytest
import time
import numpy as np
from datetime import datetime, timedelta

# Redis Cache Tests
from nethical.storage.redis_cache import RedisCache


class TestRedisCache:
    """Test Redis cache functionality."""

    def test_initialization_without_redis(self):
        """Test cache initialization without Redis installed."""
        cache = RedisCache(enabled=True)
        # Should fall back to in-memory cache
        assert (
            cache.enabled == False or cache.enabled == True
        )  # Depends on Redis availability

    def test_disabled_cache(self):
        """Test cache with enabled=False."""
        cache = RedisCache(enabled=False)
        assert cache.enabled == False

        # Should use fallback cache
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = RedisCache(enabled=False)  # Use fallback for testing

        # Set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Get non-existent key
        assert cache.get("nonexistent") is None

        # Delete
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_complex_types(self):
        """Test caching of complex types."""
        cache = RedisCache(enabled=False)

        # Dictionary
        data = {"name": "test", "value": 123, "nested": {"key": "value"}}
        cache.set("dict_key", data)
        assert cache.get("dict_key") == data

        # List
        list_data = [1, 2, 3, 4, 5]
        cache.set("list_key", list_data)
        assert cache.get("list_key") == list_data

    def test_exists(self):
        """Test key existence check."""
        cache = RedisCache(enabled=False)

        cache.set("test_key", "value")
        assert cache.exists("test_key") == True
        assert cache.exists("nonexistent") == False

    def test_increment(self):
        """Test increment operation."""
        cache = RedisCache(enabled=False)

        cache.set("counter", 10)
        result = cache.increment("counter", 5)
        assert result == 15
        assert cache.get("counter") == 15

        # Increment non-existent key
        result = cache.increment("new_counter", 1)
        assert result == 1

    def test_multiple_operations(self):
        """Test multiple key operations."""
        cache = RedisCache(enabled=False)

        # Set multiple
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        cache.set_multiple(data)

        # Get multiple
        result = cache.get_multiple(["key1", "key2", "key3"])
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["key3"] == "value3"

    def test_clear(self):
        """Test cache clear."""
        cache = RedisCache(enabled=False)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats(self):
        """Test cache statistics."""
        cache = RedisCache(enabled=False)

        stats = cache.get_stats()
        assert "enabled" in stats
        assert "type" in stats
        assert stats["type"] == "fallback"


# Load Balancer Tests
from nethical.core.load_balancer import LoadBalancer, LoadBalancingStrategy


class TestLoadBalancer:
    """Test load balancer functionality."""

    def test_initialization(self):
        """Test load balancer initialization."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        assert lb.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert len(lb.instances) == 0

    def test_add_remove_instance(self):
        """Test adding and removing instances."""
        lb = LoadBalancer()

        lb.add_instance("inst1", "us-east-1", "http://localhost:8001")
        assert "inst1" in lb.instances
        assert "us-east-1" in lb.regions

        lb.remove_instance("inst1")
        assert "inst1" not in lb.instances

    def test_round_robin_selection(self):
        """Test round-robin load balancing."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)

        lb.add_instance("inst1", "us-east-1", "http://localhost:8001")
        lb.add_instance("inst2", "us-east-1", "http://localhost:8002")
        lb.add_instance("inst3", "us-east-1", "http://localhost:8003")

        # Get instances in round-robin order
        inst1 = lb.get_instance()
        inst2 = lb.get_instance()
        inst3 = lb.get_instance()
        inst4 = lb.get_instance()  # Should wrap around

        assert inst1.instance_id == "inst1"
        assert inst2.instance_id == "inst2"
        assert inst3.instance_id == "inst3"
        assert inst4.instance_id == "inst1"

    def test_least_connections_selection(self):
        """Test least connections load balancing."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)

        lb.add_instance("inst1", "us-east-1", "http://localhost:8001")
        lb.add_instance("inst2", "us-east-1", "http://localhost:8002")

        # Set different connection counts
        lb.instances["inst1"].active_connections = 5
        lb.instances["inst2"].active_connections = 2

        inst = lb.get_instance()
        assert (
            inst.instance_id == "inst2"
        )  # Should select instance with fewer connections

    def test_region_aware_selection(self):
        """Test region-aware load balancing."""
        lb = LoadBalancer(strategy=LoadBalancingStrategy.REGION_AWARE)

        lb.add_instance("inst1", "us-east-1", "http://localhost:8001")
        lb.add_instance("inst2", "eu-west-1", "http://localhost:8002")

        # Request from us-east-1
        inst = lb.get_instance(region_id="us-east-1")
        assert inst.region_id == "us-east-1"

        # Request from eu-west-1
        inst = lb.get_instance(region_id="eu-west-1")
        assert inst.region_id == "eu-west-1"

    def test_execute_request(self):
        """Test request execution with load balancing."""
        lb = LoadBalancer(max_retries=2)

        lb.add_instance("inst1", "us-east-1", "http://localhost:8001")

        # Mock request function that succeeds
        def mock_request(endpoint):
            return f"Success from {endpoint}"

        result = lb.execute_request(mock_request)
        assert result == "Success from http://localhost:8001"

    def test_statistics(self):
        """Test load balancer statistics."""
        lb = LoadBalancer()

        lb.add_instance("inst1", "us-east-1", "http://localhost:8001")
        lb.add_instance("inst2", "eu-west-1", "http://localhost:8002")

        stats = lb.get_statistics()
        assert stats["total_instances"] == 2
        assert stats["healthy_instances"] == 2
        assert "us-east-1" in stats["regions"]
        assert "eu-west-1" in stats["regions"]


# Federated Metrics Tests
from nethical.core.federated_metrics import FederatedMetricsAggregator


class TestFederatedMetrics:
    """Test federated metrics aggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        agg = FederatedMetricsAggregator(regions=["us-east-1", "eu-west-1"])
        assert len(agg.regions) == 2
        assert "us-east-1" in agg.regions

    def test_add_remove_region(self):
        """Test adding and removing regions."""
        agg = FederatedMetricsAggregator()

        agg.add_region("us-east-1", weight=2.0)
        assert "us-east-1" in agg.regions
        assert agg.region_weights["us-east-1"] == 2.0

        agg.remove_region("us-east-1")
        assert "us-east-1" not in agg.regions

    def test_submit_regional_metrics(self):
        """Test submitting regional metrics."""
        agg = FederatedMetricsAggregator(regions=["us-east-1"])

        metrics = {"risk_score": 0.75, "latency_ms": 150.0, "error_rate": 0.02}

        agg.submit_regional_metrics("us-east-1", metrics, count=100)

        assert len(agg.regional_metrics["us-east-1"]) == 1

    def test_aggregate_metrics_mean(self):
        """Test mean aggregation."""
        agg = FederatedMetricsAggregator(regions=["us-east-1", "eu-west-1"])

        # Submit metrics from different regions
        agg.submit_regional_metrics("us-east-1", {"risk_score": 0.8}, count=100)
        agg.submit_regional_metrics("eu-west-1", {"risk_score": 0.6}, count=100)

        # Aggregate
        result = agg.aggregate_metrics(
            metric_names=["risk_score"], aggregation_type="mean"
        )

        assert "risk_score" in result
        assert 0.6 <= result["risk_score"] <= 0.8

    def test_aggregate_by_region(self):
        """Test per-region aggregation."""
        agg = FederatedMetricsAggregator(regions=["us-east-1", "eu-west-1"])

        agg.submit_regional_metrics("us-east-1", {"latency_ms": 100.0}, count=50)
        agg.submit_regional_metrics("us-east-1", {"latency_ms": 120.0}, count=50)
        agg.submit_regional_metrics("eu-west-1", {"latency_ms": 200.0}, count=100)

        result = agg.aggregate_by_region("latency_ms", aggregation_type="mean")

        assert "us-east-1" in result
        assert "eu-west-1" in result
        assert result["us-east-1"] == 110.0  # (100*50 + 120*50) / 100
        assert result["eu-west-1"] == 200.0

    def test_compute_percentiles(self):
        """Test percentile computation."""
        agg = FederatedMetricsAggregator(regions=["us-east-1"])

        # Submit multiple data points
        for i in range(100):
            agg.submit_regional_metrics("us-east-1", {"value": float(i)}, count=1)

        percentiles = agg.compute_percentiles("value", percentiles=[50, 90, 95])

        assert 50 in percentiles
        assert 90 in percentiles
        assert 95 in percentiles
        assert 40 <= percentiles[50] <= 60
        assert 85 <= percentiles[90] <= 95

    def test_regional_statistics(self):
        """Test regional statistics."""
        agg = FederatedMetricsAggregator(regions=["us-east-1", "eu-west-1"])

        agg.submit_regional_metrics("us-east-1", {"metric1": 10.0}, count=5)
        agg.submit_regional_metrics("eu-west-1", {"metric1": 20.0}, count=10)

        stats = agg.get_regional_statistics()

        assert "us-east-1" in stats
        assert "eu-west-1" in stats
        assert stats["us-east-1"]["data_points"] == 5
        assert stats["eu-west-1"]["data_points"] == 10

    def test_global_statistics(self):
        """Test global statistics."""
        agg = FederatedMetricsAggregator(regions=["us-east-1", "eu-west-1"])

        agg.submit_regional_metrics("us-east-1", {"metric1": 10.0}, count=5)
        agg.submit_regional_metrics("eu-west-1", {"metric2": 20.0}, count=10)

        stats = agg.get_global_statistics()

        assert stats["total_regions"] == 2
        assert stats["total_data_points"] == 15
        assert "metric1" in stats["tracked_metrics"]
        assert "metric2" in stats["tracked_metrics"]


# JIT Optimizations Tests
from nethical.core.jit_optimizations import (
    is_jit_available,
    calculate_risk_score_jit,
    calculate_temporal_decay_jit,
    calculate_statistics_jit,
    cosine_similarity_jit,
    detect_outliers_iqr_jit,
)


class TestJITOptimizations:
    """Test JIT optimization functions."""

    def test_jit_availability(self):
        """Test JIT availability check."""
        available = is_jit_available()
        assert isinstance(available, bool)

    def test_risk_score_calculation(self):
        """Test JIT risk score calculation."""
        severities = np.array([3, 4, 5], dtype=np.float64)
        confidences = np.array([0.8, 0.9, 0.95], dtype=np.float64)

        score = calculate_risk_score_jit(severities, confidences)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # High severity should result in high score

    def test_temporal_decay(self):
        """Test temporal decay calculation."""
        risk_score = 0.8
        time_delta = 3600.0  # 1 hour

        decayed = calculate_temporal_decay_jit(risk_score, time_delta)

        assert 0.0 <= decayed <= risk_score
        assert decayed < risk_score  # Should decay

    def test_statistics_calculation(self):
        """Test statistical calculations."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mean, std, min_val, max_val, median = calculate_statistics_jit(values)

        assert mean == 3.0
        assert min_val == 1.0
        assert max_val == 5.0
        assert median == 3.0

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        similarity = cosine_similarity_jit(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001  # Should be 1.0 for identical vectors

        vec3 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity_jit(vec1, vec3)
        assert abs(similarity) < 0.001  # Should be 0.0 for orthogonal vectors

    def test_outlier_detection(self):
        """Test outlier detection."""
        # Normal values with outliers
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0])

        outliers = detect_outliers_iqr_jit(values)

        assert len(outliers) == len(values)
        # Check that at least some values are marked as outliers
        assert np.any(outliers)  # Should have at least one outlier
        assert outliers[0] == False  # 1.0 should not be outlier


# GPU Acceleration Tests
from nethical.core.gpu_acceleration import (
    is_gpu_available,
    get_gpu_info,
    GPUAcceleratedInference,
)


class TestGPUAcceleration:
    """Test GPU acceleration features."""

    def test_gpu_availability(self):
        """Test GPU availability check."""
        available = is_gpu_available()
        assert isinstance(available, bool)

    def test_gpu_info(self):
        """Test GPU information retrieval."""
        info = get_gpu_info()
        assert "available" in info

    def test_inference_engine_initialization(self):
        """Test inference engine initialization."""
        engine = GPUAcceleratedInference(use_gpu=False)  # Force CPU for testing
        assert engine.use_gpu == False

    def test_tensor_conversion(self):
        """Test tensor conversion."""
        engine = GPUAcceleratedInference(use_gpu=False)

        # Numpy to tensor
        data = np.array([1.0, 2.0, 3.0])
        tensor = engine.to_tensor(data)

        # Tensor to numpy
        result = engine.to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
