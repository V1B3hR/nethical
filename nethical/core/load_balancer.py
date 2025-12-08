"""
Load balancer for multi-region deployment support.

This module provides load balancing capabilities for distributing
governance requests across multiple regions and instances.
"""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    REGION_AWARE = "region_aware"


@dataclass
class BackendInstance:
    """Backend instance configuration."""

    instance_id: str
    region_id: str
    endpoint: str
    weight: int = 1
    max_connections: int = 100
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)

    def record_request(self, success: bool, response_time: float):
        """Record request statistics."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1

        # Update average response time using exponential moving average
        alpha = 0.1  # Smoothing factor
        self.avg_response_time = (
            alpha * response_time + (1 - alpha) * self.avg_response_time
        )

    def get_health_score(self) -> float:
        """
        Calculate health score (0-1).

        Based on:
        - Error rate
        - Response time
        - Connection utilization
        """
        if self.total_requests == 0:
            return 1.0

        # Error rate (0-1, inverted)
        error_rate = self.failed_requests / self.total_requests
        error_score = max(0, 1 - error_rate * 2)

        # Response time score (normalized)
        # Assume 100ms is good, 1000ms is bad
        response_score = max(0, 1 - (self.avg_response_time - 100) / 900)

        # Connection utilization
        utilization = self.active_connections / self.max_connections
        util_score = max(0, 1 - utilization)

        # Weighted average
        return 0.5 * error_score + 0.3 * response_score + 0.2 * util_score


class LoadBalancer:
    """
    Load balancer for distributing governance requests.

    Features:
    - Multiple load balancing strategies
    - Health checking
    - Region-aware routing
    - Connection tracking
    - Performance monitoring
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check_interval: int = 30,
        health_check_timeout: int = 5,
        max_retries: int = 2,
    ):
        """
        Initialize load balancer.

        Args:
            strategy: Load balancing strategy
            health_check_interval: Seconds between health checks
            health_check_timeout: Health check timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_retries = max_retries

        self.instances: Dict[str, BackendInstance] = {}
        self.regions: Dict[str, List[str]] = {}  # region_id -> instance_ids

        # For round-robin
        self._rr_index = 0
        self._rr_lock = Lock()

        # Statistics
        self.total_requests = 0
        self.failed_requests = 0

        logger.info(f"Load balancer initialized with strategy: {strategy.value}")

    def add_instance(
        self,
        instance_id: str,
        region_id: str,
        endpoint: str,
        weight: int = 1,
        max_connections: int = 100,
    ):
        """
        Add a backend instance.

        Args:
            instance_id: Instance identifier
            region_id: Region identifier
            endpoint: Instance endpoint
            weight: Instance weight (for weighted strategies)
            max_connections: Maximum concurrent connections
        """
        instance = BackendInstance(
            instance_id=instance_id,
            region_id=region_id,
            endpoint=endpoint,
            weight=weight,
            max_connections=max_connections,
        )

        self.instances[instance_id] = instance

        # Track by region
        if region_id not in self.regions:
            self.regions[region_id] = []
        self.regions[region_id].append(instance_id)

        logger.info(f"Added instance {instance_id} in region {region_id}")

    def remove_instance(self, instance_id: str):
        """
        Remove a backend instance.

        Args:
            instance_id: Instance identifier
        """
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            region_id = instance.region_id

            # Remove from instances
            del self.instances[instance_id]

            # Remove from region tracking
            if region_id in self.regions:
                self.regions[region_id].remove(instance_id)
                if not self.regions[region_id]:
                    del self.regions[region_id]

            logger.info(f"Removed instance {instance_id}")

    def get_instance(
        self,
        region_id: Optional[str] = None,
        exclude_instances: Optional[List[str]] = None,
    ) -> Optional[BackendInstance]:
        """
        Get next instance based on strategy.

        Args:
            region_id: Preferred region (for region-aware routing)
            exclude_instances: Instances to exclude (for retries)

        Returns:
            Selected backend instance or None
        """
        exclude_instances = exclude_instances or []

        # Filter healthy instances
        available_instances = [
            inst
            for inst_id, inst in self.instances.items()
            if inst.is_healthy
            and inst.active_connections < inst.max_connections
            and inst_id not in exclude_instances
        ]

        if not available_instances:
            logger.warning("No available instances")
            return None

        # Apply strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(available_instances)

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(available_instances)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(available_instances)

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available_instances)

        elif self.strategy == LoadBalancingStrategy.REGION_AWARE:
            return self._region_aware(available_instances, region_id)

        return None

    def _round_robin(self, instances: List[BackendInstance]) -> BackendInstance:
        """Round-robin selection."""
        with self._rr_lock:
            instance = instances[self._rr_index % len(instances)]
            self._rr_index = (self._rr_index + 1) % len(instances)
            return instance

    def _least_connections(self, instances: List[BackendInstance]) -> BackendInstance:
        """Select instance with least active connections."""
        return min(instances, key=lambda x: x.active_connections)

    def _weighted_round_robin(
        self, instances: List[BackendInstance]
    ) -> BackendInstance:
        """Weighted round-robin selection."""
        # Create weighted list
        weighted_instances = []
        for inst in instances:
            weighted_instances.extend([inst] * inst.weight)

        if not weighted_instances:
            return instances[0]

        with self._rr_lock:
            instance = weighted_instances[self._rr_index % len(weighted_instances)]
            self._rr_index = (self._rr_index + 1) % len(weighted_instances)
            return instance

    def _region_aware(
        self, instances: List[BackendInstance], region_id: Optional[str]
    ) -> BackendInstance:
        """
        Region-aware selection.

        Prefers instances in the same region, falls back to closest region.
        """
        if region_id:
            # Try same region first
            same_region = [inst for inst in instances if inst.region_id == region_id]
            if same_region:
                return self._least_connections(same_region)

        # Fall back to least connections
        return self._least_connections(instances)

    def execute_request(
        self,
        request_func: Callable[[str], Any],
        region_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Execute request with load balancing and retries.

        Args:
            request_func: Function that takes endpoint and returns result
            region_id: Preferred region
            timeout: Request timeout

        Returns:
            Request result or None on failure
        """
        self.total_requests += 1
        excluded_instances = []

        for attempt in range(self.max_retries + 1):
            # Get instance
            instance = self.get_instance(
                region_id=region_id, exclude_instances=excluded_instances
            )

            if not instance:
                logger.error("No available instances for request")
                self.failed_requests += 1
                return None

            # Execute request
            instance.active_connections += 1
            start_time = time.time()

            try:
                result = request_func(instance.endpoint)
                response_time = (time.time() - start_time) * 1000  # ms
                instance.record_request(success=True, response_time=response_time)
                return result

            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                instance.record_request(success=False, response_time=response_time)

                logger.warning(
                    f"Request failed on {instance.instance_id}: {e} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )

                # Exclude this instance for next retry
                excluded_instances.append(instance.instance_id)

                if attempt == self.max_retries:
                    self.failed_requests += 1
                    return None

            finally:
                instance.active_connections -= 1

        return None

    def check_health(self, instance_id: str, health_check_func: Callable[[str], bool]):
        """
        Check health of an instance.

        Args:
            instance_id: Instance identifier
            health_check_func: Function that takes endpoint and returns health status
        """
        if instance_id not in self.instances:
            return

        instance = self.instances[instance_id]

        try:
            is_healthy = health_check_func(instance.endpoint)
            instance.is_healthy = is_healthy
            instance.last_health_check = time.time()

            if not is_healthy:
                logger.warning(f"Instance {instance_id} is unhealthy")

        except Exception as e:
            logger.error(f"Health check failed for {instance_id}: {e}")
            instance.is_healthy = False
            instance.last_health_check = time.time()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get load balancer statistics.

        Returns:
            Statistics dictionary
        """
        healthy_instances = sum(
            1 for inst in self.instances.values() if inst.is_healthy
        )
        total_connections = sum(
            inst.active_connections for inst in self.instances.values()
        )

        instance_stats = []
        for inst in self.instances.values():
            instance_stats.append(
                {
                    "instance_id": inst.instance_id,
                    "region_id": inst.region_id,
                    "is_healthy": inst.is_healthy,
                    "active_connections": inst.active_connections,
                    "total_requests": inst.total_requests,
                    "failed_requests": inst.failed_requests,
                    "error_rate": (
                        inst.failed_requests / inst.total_requests * 100
                        if inst.total_requests > 0
                        else 0
                    ),
                    "avg_response_time": inst.avg_response_time,
                    "health_score": inst.get_health_score(),
                }
            )

        return {
            "strategy": self.strategy.value,
            "total_instances": len(self.instances),
            "healthy_instances": healthy_instances,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "error_rate": (
                self.failed_requests / self.total_requests * 100
                if self.total_requests > 0
                else 0
            ),
            "active_connections": total_connections,
            "instances": instance_stats,
            "regions": {
                region: len(instances) for region, instances in self.regions.items()
            },
        }
