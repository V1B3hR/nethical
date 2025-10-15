"""
Quotas and Rate Limiting Module

This module provides configurable quotas, rate limiting, and backpressure
mechanisms to prevent resource exhaustion and enforce multi-tenant isolation.

Features:
- Per-agent, per-cohort, and per-tenant quotas
- Configurable limits (requests/sec, actions/min, memory hints, payload size)
- Backpressure and throttling decisions
- Metrics for monitoring quota enforcement
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuotaConfig:
    """Configuration for quota limits."""
    
    # Rate limits
    requests_per_second: float = 10.0
    actions_per_minute: int = 100
    burst_size: int = 20  # Allow bursts up to this size
    
    # Resource limits
    max_payload_bytes: int = 1_000_000  # 1MB default
    max_concurrent_actions: int = 50
    memory_cap_mb: int = 512
    
    # Time windows
    rate_window_seconds: int = 60
    burst_window_seconds: int = 1
    
    # Backpressure thresholds
    throttle_threshold: float = 0.8  # Start throttling at 80% capacity
    block_threshold: float = 0.95  # Block at 95% capacity
    
    # Isolation
    enable_tenant_isolation: bool = True
    enable_cohort_isolation: bool = True


@dataclass
class QuotaUsage:
    """Track quota usage for an entity."""
    
    entity_id: str
    entity_type: str  # "agent", "cohort", "tenant"
    
    # Counters
    requests_count: int = 0
    actions_count: int = 0
    bytes_processed: int = 0
    
    # Time tracking
    first_request_time: Optional[float] = None
    last_request_time: Optional[float] = None
    
    # Request history for windowing
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    action_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Violations
    throttle_count: int = 0
    block_count: int = 0
    last_violation_time: Optional[float] = None


class QuotaEnforcer:
    """Enforces quotas and rate limits with backpressure."""
    
    def __init__(self, config: Optional[QuotaConfig] = None):
        """Initialize quota enforcer."""
        self.config = config or QuotaConfig()
        self.usage: Dict[str, QuotaUsage] = {}
        
        # Metrics
        self.total_requests = 0
        self.total_throttles = 0
        self.total_blocks = 0
        self.total_bytes_processed = 0
        
        logger.info(f"QuotaEnforcer initialized with config: {self.config}")
    
    def check_quota(
        self,
        agent_id: str,
        cohort: Optional[str] = None,
        tenant: Optional[str] = None,
        payload_size: int = 0,
        action_type: str = "query"
    ) -> Dict[str, Any]:
        """
        Check if request is within quota limits.
        
        Returns:
            Dictionary with decision, usage metrics, and enforcement action
        """
        current_time = time.time()
        result = {
            "allowed": True,
            "decision": "ALLOW",
            "reason": None,
            "usage": {},
            "enforcement_action": None,
            "backpressure_level": 0.0
        }
        
        # Check agent quota
        agent_check = self._check_entity_quota(
            agent_id, "agent", current_time, payload_size
        )
        if not agent_check["allowed"]:
            result.update(agent_check)
            return result
        
        result["usage"]["agent"] = agent_check["usage"]
        result["backpressure_level"] = max(
            result["backpressure_level"],
            agent_check["backpressure_level"]
        )
        
        # Check cohort quota if enabled
        if self.config.enable_cohort_isolation and cohort:
            cohort_check = self._check_entity_quota(
                cohort, "cohort", current_time, payload_size
            )
            if not cohort_check["allowed"]:
                result.update(cohort_check)
                return result
            
            result["usage"]["cohort"] = cohort_check["usage"]
            result["backpressure_level"] = max(
                result["backpressure_level"],
                cohort_check["backpressure_level"]
            )
        
        # Check tenant quota if enabled
        if self.config.enable_tenant_isolation and tenant:
            tenant_check = self._check_entity_quota(
                tenant, "tenant", current_time, payload_size
            )
            if not tenant_check["allowed"]:
                result.update(tenant_check)
                return result
            
            result["usage"]["tenant"] = tenant_check["usage"]
            result["backpressure_level"] = max(
                result["backpressure_level"],
                tenant_check["backpressure_level"]
            )
        
        # Check payload size
        if payload_size > self.config.max_payload_bytes:
            result["allowed"] = False
            result["decision"] = "BLOCK"
            result["reason"] = f"Payload size {payload_size} exceeds limit {self.config.max_payload_bytes}"
            result["enforcement_action"] = "REJECT_OVERSIZED_PAYLOAD"
            self.total_blocks += 1
            return result
        
        # Record usage
        self._record_usage(agent_id, "agent", current_time, payload_size)
        if cohort:
            self._record_usage(cohort, "cohort", current_time, payload_size)
        if tenant:
            self._record_usage(tenant, "tenant", current_time, payload_size)
        
        self.total_requests += 1
        self.total_bytes_processed += payload_size
        
        return result
    
    def _check_entity_quota(
        self,
        entity_id: str,
        entity_type: str,
        current_time: float,
        payload_size: int
    ) -> Dict[str, Any]:
        """Check quota for a specific entity (agent/cohort/tenant)."""
        
        # Get or create usage tracking
        if entity_id not in self.usage:
            self.usage[entity_id] = QuotaUsage(
                entity_id=entity_id,
                entity_type=entity_type
            )
        
        usage = self.usage[entity_id]
        
        # Calculate request rate in current window
        window_start = current_time - self.config.rate_window_seconds
        recent_requests = [t for t in usage.request_times if t >= window_start]
        request_rate = len(recent_requests) / self.config.rate_window_seconds
        
        # Calculate burst rate
        burst_window_start = current_time - self.config.burst_window_seconds
        burst_requests = [t for t in usage.request_times if t >= burst_window_start]
        burst_size = len(burst_requests)
        
        # Calculate capacity utilization
        rate_utilization = request_rate / self.config.requests_per_second
        burst_utilization = burst_size / self.config.burst_size
        overall_utilization = max(rate_utilization, burst_utilization)
        
        result = {
            "allowed": True,
            "decision": "ALLOW",
            "reason": None,
            "usage": {
                "request_rate": request_rate,
                "burst_size": burst_size,
                "utilization": overall_utilization,
                "total_requests": usage.requests_count,
                "total_bytes": usage.bytes_processed
            },
            "backpressure_level": overall_utilization
        }
        
        # Check if limits exceeded
        if overall_utilization >= self.config.block_threshold:
            result["allowed"] = False
            result["decision"] = "BLOCK"
            result["reason"] = f"{entity_type} {entity_id} exceeded quota: {overall_utilization:.1%} utilization"
            result["enforcement_action"] = "RATE_LIMIT_BLOCK"
            usage.block_count += 1
            usage.last_violation_time = current_time
            self.total_blocks += 1
            
        elif overall_utilization >= self.config.throttle_threshold:
            result["allowed"] = True  # Still allow but flag for throttling
            result["decision"] = "THROTTLE"
            result["reason"] = f"{entity_type} {entity_id} approaching quota: {overall_utilization:.1%} utilization"
            result["enforcement_action"] = "BACKPRESSURE_THROTTLE"
            usage.throttle_count += 1
            self.total_throttles += 1
        
        return result
    
    def _record_usage(
        self,
        entity_id: str,
        entity_type: str,
        current_time: float,
        payload_size: int
    ):
        """Record usage for an entity."""
        if entity_id not in self.usage:
            self.usage[entity_id] = QuotaUsage(
                entity_id=entity_id,
                entity_type=entity_type
            )
        
        usage = self.usage[entity_id]
        usage.requests_count += 1
        usage.bytes_processed += payload_size
        usage.request_times.append(current_time)
        usage.last_request_time = current_time
        
        if usage.first_request_time is None:
            usage.first_request_time = current_time
    
    def get_usage_summary(self, entity_id: str) -> Dict[str, Any]:
        """Get usage summary for an entity."""
        if entity_id not in self.usage:
            return {"error": "Entity not found"}
        
        usage = self.usage[entity_id]
        current_time = time.time()
        
        # Calculate current rates
        window_start = current_time - self.config.rate_window_seconds
        recent_requests = [t for t in usage.request_times if t >= window_start]
        current_rate = len(recent_requests) / self.config.rate_window_seconds
        
        return {
            "entity_id": usage.entity_id,
            "entity_type": usage.entity_type,
            "total_requests": usage.requests_count,
            "total_bytes": usage.bytes_processed,
            "current_rate": current_rate,
            "throttle_count": usage.throttle_count,
            "block_count": usage.block_count,
            "first_request": datetime.fromtimestamp(usage.first_request_time).isoformat() if usage.first_request_time else None,
            "last_request": datetime.fromtimestamp(usage.last_request_time).isoformat() if usage.last_request_time else None,
            "last_violation": datetime.fromtimestamp(usage.last_violation_time).isoformat() if usage.last_violation_time else None
        }
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global metrics across all entities."""
        return {
            "total_requests": self.total_requests,
            "total_throttles": self.total_throttles,
            "total_blocks": self.total_blocks,
            "total_bytes_processed": self.total_bytes_processed,
            "throttle_rate": self.total_throttles / max(self.total_requests, 1),
            "block_rate": self.total_blocks / max(self.total_requests, 1),
            "tracked_entities": len(self.usage),
            "config": {
                "requests_per_second": self.config.requests_per_second,
                "actions_per_minute": self.config.actions_per_minute,
                "max_payload_bytes": self.config.max_payload_bytes,
                "throttle_threshold": self.config.throttle_threshold,
                "block_threshold": self.config.block_threshold
            }
        }
    
    def reset_usage(self, entity_id: Optional[str] = None):
        """Reset usage for an entity or all entities."""
        if entity_id:
            if entity_id in self.usage:
                del self.usage[entity_id]
        else:
            self.usage.clear()
            self.total_requests = 0
            self.total_throttles = 0
            self.total_blocks = 0
            self.total_bytes_processed = 0


# Global singleton instance
_global_enforcer: Optional[QuotaEnforcer] = None


def get_quota_enforcer(config: Optional[QuotaConfig] = None) -> QuotaEnforcer:
    """Get or create the global quota enforcer instance."""
    global _global_enforcer
    if _global_enforcer is None:
        _global_enforcer = QuotaEnforcer(config)
    return _global_enforcer


def configure_quotas(config: QuotaConfig):
    """Configure the global quota enforcer."""
    global _global_enforcer
    _global_enforcer = QuotaEnforcer(config)
    return _global_enforcer
