"""
Edge Decision Engine Module

Ultra-low latency governance for edge deployment.
Designed for autonomous vehicles, robots, and real-time systems.

Target: <10ms p99 latency
Mode: Offline-first with sync

Components:
- EdgeGovernor: Core edge governance engine
- PolicyCache: In-memory policy cache with LRU eviction
- FastDetector: Lightweight detectors for edge deployment
- SafeDefaults: Fail-safe default decisions
- PredictiveEngine: Pre-computation for predicted actions
- OfflineFallback: Graceful degradation when disconnected
"""

from .local_governor import EdgeGovernor, EdgeDecision, DecisionType
from .policy_cache import PolicyCache, CachedPolicy
from .fast_detector import FastDetector, DetectionResult
from .safe_defaults import SafeDefaults, DefaultDecision
from .predictive_engine import PredictiveEngine, PredictionProfile
from .context_fingerprint import ContextFingerprint, compute_fingerprint
from .pattern_profiler import PatternProfiler, ActionPattern
from .offline_fallback import OfflineFallback, OfflineMode
from .network_monitor import NetworkMonitor, ConnectionStatus
from .decision_queue import DecisionQueue, QueuedDecision
from .sync_manager import SyncManager, SyncStatus
from .circuit_breaker import CircuitBreaker, CircuitState

__all__ = [
    # Core
    "EdgeGovernor",
    "EdgeDecision",
    "DecisionType",
    # Policy
    "PolicyCache",
    "CachedPolicy",
    # Detection
    "FastDetector",
    "DetectionResult",
    # Defaults
    "SafeDefaults",
    "DefaultDecision",
    # Prediction
    "PredictiveEngine",
    "PredictionProfile",
    "ContextFingerprint",
    "compute_fingerprint",
    "PatternProfiler",
    "ActionPattern",
    # Offline
    "OfflineFallback",
    "OfflineMode",
    "NetworkMonitor",
    "ConnectionStatus",
    "DecisionQueue",
    "QueuedDecision",
    "SyncManager",
    "SyncStatus",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
]
