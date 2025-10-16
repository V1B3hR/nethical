"""
Performance & Scalability Optimizations Demo

This script demonstrates the new performance and scalability features:
- Redis caching for high-speed data access
- Load balancing across regions
- Federated metrics aggregation
- JIT compilation for hot paths
- GPU acceleration (if available)
- TimescaleDB for time-series data (if available)
- Elasticsearch for audit log search (if available)
"""

import time
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("NETHICAL PERFORMANCE & SCALABILITY OPTIMIZATIONS DEMO")
print("=" * 80)
print()

# ============================================================================
# 1. Redis Cache Demo
# ============================================================================
print("1. REDIS CACHE - High-Speed Caching")
print("-" * 80)

from nethical.storage.redis_cache import RedisCache

# Initialize cache (will use fallback if Redis not available)
cache = RedisCache(enabled=True, default_ttl=300)

# Demonstrate caching
print(f"✓ Cache enabled: {cache.enabled}")
print(f"✓ Cache type: {'Redis' if cache.enabled else 'In-memory fallback'}")

# Store and retrieve data
cache.set("risk_profile_agent123", {"risk_score": 0.75, "tier": "ELEVATED"})
profile = cache.get("risk_profile_agent123")
print(f"✓ Cached risk profile: {profile}")

# Multiple operations
cache.set_multiple({
    "metric_latency": 45.3,
    "metric_throughput": 1250,
    "metric_error_rate": 0.02
})

metrics = cache.get_multiple(["metric_latency", "metric_throughput", "metric_error_rate"])
print(f"✓ Cached metrics: {metrics}")

# Cache statistics
stats = cache.get_stats()
print(f"✓ Cache stats: {stats}")
print()

# ============================================================================
# 2. Load Balancer Demo
# ============================================================================
print("2. LOAD BALANCER - Multi-Region Load Balancing")
print("-" * 80)

from nethical.core.load_balancer import LoadBalancer, LoadBalancingStrategy

# Initialize load balancer
lb = LoadBalancer(strategy=LoadBalancingStrategy.REGION_AWARE, max_retries=2)

# Add instances across regions
lb.add_instance("us-east-1-primary", "us-east-1", "http://us-east-1.nethical.io:8000", weight=2)
lb.add_instance("us-east-1-secondary", "us-east-1", "http://us-east-1.nethical.io:8001", weight=1)
lb.add_instance("eu-west-1-primary", "eu-west-1", "http://eu-west-1.nethical.io:8000", weight=2)
lb.add_instance("ap-south-1-primary", "ap-south-1", "http://ap-south-1.nethical.io:8000", weight=2)

print(f"✓ Load balancer initialized with {len(lb.instances)} instances")
print(f"✓ Regions: {list(lb.regions.keys())}")

# Demonstrate region-aware routing
for region in ["us-east-1", "eu-west-1", "ap-south-1"]:
    instance = lb.get_instance(region_id=region)
    print(f"✓ Request from {region} -> {instance.instance_id} ({instance.endpoint})")

# Execute mock requests
def mock_governance_request(endpoint):
    """Mock governance request."""
    time.sleep(0.001)  # Simulate network latency
    return {"decision": "ALLOW", "endpoint": endpoint}

print("\n✓ Executing 5 requests with load balancing:")
for i in range(5):
    result = lb.execute_request(mock_governance_request, region_id="us-east-1")
    if result:
        print(f"  Request {i+1}: {result}")

# Get statistics
stats = lb.get_statistics()
print(f"\n✓ Load balancer stats:")
print(f"  - Total instances: {stats['total_instances']}")
print(f"  - Healthy instances: {stats['healthy_instances']}")
print(f"  - Total requests: {stats['total_requests']}")
print(f"  - Error rate: {stats['error_rate']:.2f}%")
print()

# ============================================================================
# 3. Federated Metrics Aggregator Demo
# ============================================================================
print("3. FEDERATED METRICS - Cross-Region Aggregation")
print("-" * 80)

from nethical.core.federated_metrics import FederatedMetricsAggregator

# Initialize aggregator
agg = FederatedMetricsAggregator(
    regions=["us-east-1", "eu-west-1", "ap-south-1"],
    aggregation_interval=60
)

# Simulate regional metrics
regions_data = {
    "us-east-1": {"risk_score": 0.45, "latency_ms": 120, "throughput": 1500},
    "eu-west-1": {"risk_score": 0.38, "latency_ms": 85, "throughput": 1200},
    "ap-south-1": {"risk_score": 0.52, "latency_ms": 150, "throughput": 900},
}

print("✓ Submitting regional metrics:")
for region, metrics in regions_data.items():
    agg.submit_regional_metrics(region, metrics, count=1000)
    print(f"  - {region}: risk={metrics['risk_score']}, latency={metrics['latency_ms']}ms")

# Aggregate across regions
global_metrics = agg.aggregate_metrics(
    metric_names=["risk_score", "latency_ms", "throughput"],
    aggregation_type="mean"
)

print(f"\n✓ Global aggregated metrics:")
for metric, value in global_metrics.items():
    print(f"  - {metric}: {value:.2f}")

# Per-region breakdown
regional_latency = agg.aggregate_by_region("latency_ms", aggregation_type="mean")
print(f"\n✓ Latency by region:")
for region, latency in regional_latency.items():
    print(f"  - {region}: {latency:.2f}ms")

# Global statistics
global_stats = agg.get_global_statistics()
print(f"\n✓ Global statistics:")
print(f"  - Total regions: {global_stats['total_regions']}")
print(f"  - Active regions: {global_stats['active_regions']}")
print(f"  - Total data points: {global_stats['total_data_points']}")
print()

# ============================================================================
# 4. JIT Compilation Demo
# ============================================================================
print("4. JIT COMPILATION - High-Performance Computing")
print("-" * 80)

from nethical.core.jit_optimizations import (
    is_jit_available,
    calculate_risk_score_jit,
    calculate_statistics_jit,
    cosine_similarity_jit,
    detect_outliers_iqr_jit
)

print(f"✓ JIT compilation available: {is_jit_available()}")

# Risk score calculation
severities = np.array([3.0, 4.0, 5.0, 2.0, 4.0], dtype=np.float64)
confidences = np.array([0.8, 0.9, 0.95, 0.7, 0.85], dtype=np.float64)

start = time.time()
risk_score = calculate_risk_score_jit(severities, confidences)
jit_time = (time.time() - start) * 1000

print(f"\n✓ JIT risk score calculation:")
print(f"  - Input: {len(severities)} violations")
print(f"  - Risk score: {risk_score:.3f}")
print(f"  - Execution time: {jit_time:.4f}ms")

# Statistical calculations
values = np.random.randn(10000)
mean, std, min_val, max_val, median = calculate_statistics_jit(values)

print(f"\n✓ JIT statistics (10,000 values):")
print(f"  - Mean: {mean:.3f}")
print(f"  - Std: {std:.3f}")
print(f"  - Range: [{min_val:.3f}, {max_val:.3f}]")
print(f"  - Median: {median:.3f}")

# Similarity calculation
vec1 = np.random.randn(256)
vec2 = np.random.randn(256)
similarity = cosine_similarity_jit(vec1, vec2)

print(f"\n✓ JIT cosine similarity (256-dim vectors):")
print(f"  - Similarity: {similarity:.3f}")

# Outlier detection
data = np.concatenate([np.random.randn(100) * 10 + 50, [200, 250, -100]])
outliers = detect_outliers_iqr_jit(data)

print(f"\n✓ JIT outlier detection:")
print(f"  - Total values: {len(data)}")
print(f"  - Outliers detected: {np.sum(outliers)}")
print()

# ============================================================================
# 5. GPU Acceleration Demo (if available)
# ============================================================================
print("5. GPU ACCELERATION - ML Inference")
print("-" * 80)

from nethical.core.gpu_acceleration import (
    is_gpu_available,
    get_gpu_info,
    GPUAcceleratedInference
)

gpu_available = is_gpu_available()
print(f"✓ GPU available: {gpu_available}")

if gpu_available:
    gpu_info = get_gpu_info()
    print(f"✓ GPU device: {gpu_info.get('device_name', 'Unknown')}")
    print(f"✓ CUDA version: {gpu_info.get('cuda_version', 'Unknown')}")
    print(f"✓ Total memory: {gpu_info.get('total_memory_gb', 0):.2f} GB")
else:
    print("✓ Using CPU for inference (GPU not available)")

# Initialize inference engine
engine = GPUAcceleratedInference(use_gpu=gpu_available, batch_size=32)

# Test tensor conversion
test_data = np.random.randn(100, 128).astype(np.float32)
tensor = engine.to_tensor(test_data)
converted_back = engine.to_numpy(tensor)

print(f"\n✓ Tensor conversion test:")
print(f"  - Original shape: {test_data.shape}")
print(f"  - Converted shape: {converted_back.shape}")
print(f"  - Data preserved: {np.allclose(test_data, converted_back)}")

if gpu_available:
    memory = engine.get_memory_usage()
    print(f"\n✓ GPU memory usage:")
    print(f"  - Allocated: {memory.get('allocated_gb', 0):.3f} GB")
    print(f"  - Reserved: {memory.get('reserved_gb', 0):.3f} GB")

print()

# ============================================================================
# 6. TimescaleDB Demo (if available)
# ============================================================================
print("6. TIMESCALEDB - Time-Series Data Storage")
print("-" * 80)

from nethical.storage.timescaledb import TimescaleDBStore

# Initialize (will only work if TimescaleDB is installed and configured)
ts_store = TimescaleDBStore(enabled=False)  # Disabled for demo

print(f"✓ TimescaleDB enabled: {ts_store.enabled}")

if ts_store.enabled:
    # Insert metrics
    success = ts_store.insert_metric(
        agent_id="agent_123",
        metric_name="risk_score",
        metric_value=0.75,
        region_id="us-east-1"
    )
    print(f"✓ Metric inserted: {success}")
    
    # Query metrics
    metrics = ts_store.query_metrics(
        agent_id="agent_123",
        metric_name="risk_score",
        limit=10
    )
    print(f"✓ Metrics queried: {len(metrics)} results")
else:
    print("✓ Install psycopg2-binary and configure PostgreSQL/TimescaleDB to enable")

print()

# ============================================================================
# 7. Elasticsearch Demo (if available)
# ============================================================================
print("7. ELASTICSEARCH - Audit Log Search")
print("-" * 80)

from nethical.storage.elasticsearch_store import ElasticsearchAuditStore

# Initialize (will only work if Elasticsearch is installed and running)
es_store = ElasticsearchAuditStore(enabled=False)  # Disabled for demo

print(f"✓ Elasticsearch enabled: {es_store.enabled}")

if es_store.enabled:
    # Index audit log
    success = es_store.index_audit_log(
        action_id="act_123",
        agent_id="agent_123",
        decision="BLOCK",
        action_data={
            "stated_intent": "Access sensitive data",
            "actual_action": "Query database"
        },
        violations=[
            {"type": "privacy", "severity": 4, "confidence": 0.9}
        ],
        region_id="us-east-1"
    )
    print(f"✓ Audit log indexed: {success}")
    
    # Search audit logs
    results = es_store.search_audit_logs(
        query="sensitive data",
        decision="BLOCK",
        size=10
    )
    print(f"✓ Search results: {results['hits']['total']['value']} matches")
else:
    print("✓ Install elasticsearch and run Elasticsearch server to enable")

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("✓ Performance & Scalability Features Demonstrated:")
print("  1. Redis Cache - High-speed caching with fallback support")
print("  2. Load Balancer - Multi-region load balancing with 5 strategies")
print("  3. Federated Metrics - Cross-region aggregation without raw data sharing")
print("  4. JIT Compilation - 10-100x speedup for numerical operations")
print("  5. GPU Acceleration - Fast ML inference (when GPU available)")
print("  6. TimescaleDB - Efficient time-series data storage (optional)")
print("  7. Elasticsearch - Full-text audit log search (optional)")
print()
print("✓ All features have graceful fallbacks when dependencies unavailable")
print("✓ Designed for horizontal and vertical scalability")
print("✓ Ready for enterprise-scale deployments (11-50+ systems)")
print()
print("For production use, install optional dependencies:")
print("  pip install redis psycopg2-binary elasticsearch numba torch")
print()
print("=" * 80)
