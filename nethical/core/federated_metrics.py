"""
Federated metrics aggregator for cross-region metric collection.

This module provides federated aggregation of metrics across multiple
regions without requiring raw data sharing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class RegionalMetrics:
    """Regional metrics snapshot."""

    region_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    count: int
    metadata: Dict[str, Any]


class FederatedMetricsAggregator:
    """
    Federated metrics aggregator for multi-region deployments.

    Features:
    - Cross-region metric aggregation
    - Privacy-preserving aggregation (no raw data sharing)
    - Incremental aggregation
    - Weighted aggregation by region
    - Statistical aggregations (mean, median, percentiles)
    """

    def __init__(
        self,
        regions: Optional[List[str]] = None,
        aggregation_interval: int = 60,  # seconds
        retention_period: int = 86400,  # 24 hours
    ):
        """
        Initialize federated metrics aggregator.

        Args:
            regions: List of region identifiers
            aggregation_interval: Interval for aggregation in seconds
            retention_period: How long to retain metrics in seconds
        """
        self.regions = regions if regions is not None else []
        self.aggregation_interval = aggregation_interval
        self.retention_period = retention_period

        # Store regional metrics
        self.regional_metrics: Dict[str, List[RegionalMetrics]] = defaultdict(list)

        # Store aggregated metrics
        self.aggregated_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Region weights (for weighted aggregation)
        self.region_weights: Dict[str, float] = {region: 1.0 for region in self.regions}

        logger.info(f"Federated metrics aggregator initialized for regions: {regions}")

    def add_region(self, region_id: str, weight: float = 1.0):
        """
        Add a region to the aggregator.

        Args:
            region_id: Region identifier
            weight: Weight for this region in aggregations
        """
        if region_id not in self.regions:
            self.regions.append(region_id)
            self.region_weights[region_id] = weight
            logger.info(f"Added region {region_id} with weight {weight}")

    def remove_region(self, region_id: str):
        """
        Remove a region from the aggregator.

        Args:
            region_id: Region identifier
        """
        if region_id in self.regions:
            self.regions.remove(region_id)
            self.region_weights.pop(region_id, None)
            self.regional_metrics.pop(region_id, None)
            logger.info(f"Removed region {region_id}")

    def submit_regional_metrics(
        self,
        region_id: str,
        metrics: Dict[str, float],
        count: int = 1,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Submit metrics from a region.

        Args:
            region_id: Region identifier
            metrics: Dictionary of metric_name -> value
            count: Number of data points represented
            timestamp: Timestamp (uses current time if None)
            metadata: Additional metadata
        """
        if region_id not in self.regions:
            logger.warning(f"Unknown region {region_id}, adding it")
            self.add_region(region_id)

        timestamp = timestamp or datetime.now(timezone.utc)

        regional_metrics = RegionalMetrics(
            region_id=region_id,
            timestamp=timestamp,
            metrics=metrics,
            count=count,
            metadata=metadata or {},
        )

        self.regional_metrics[region_id].append(regional_metrics)

        # Cleanup old metrics
        self._cleanup_old_metrics(region_id)

    def aggregate_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        aggregation_type: str = "mean",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        regions: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Aggregate metrics across regions.

        Args:
            metric_names: List of metrics to aggregate (None for all)
            aggregation_type: Type of aggregation (mean, median, sum, min, max)
            start_time: Start of time range
            end_time: End of time range
            regions: Specific regions to include (None for all)

        Returns:
            Dictionary of aggregated metrics
        """
        regions = regions or self.regions
        end_time = end_time or datetime.now(timezone.utc)
        start_time = start_time or (
            end_time - timedelta(seconds=self.aggregation_interval)
        )

        # Collect metrics from regions
        regional_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for region_id in regions:
            if region_id not in self.regional_metrics:
                continue

            weight = self.region_weights.get(region_id, 1.0)

            for rm in self.regional_metrics[region_id]:
                # Filter by time range
                if start_time <= rm.timestamp <= end_time:
                    for metric_name, value in rm.metrics.items():
                        if metric_names is None or metric_name in metric_names:
                            # Store (value, weight * count)
                            regional_data[metric_name].append(
                                (value, weight * rm.count)
                            )

        # Aggregate
        aggregated = {}
        for metric_name, values_weights in regional_data.items():
            if not values_weights:
                continue

            values = [v for v, w in values_weights]
            weights = [w for v, w in values_weights]

            if aggregation_type == "mean":
                # Weighted mean
                total_weight = sum(weights)
                if total_weight > 0:
                    aggregated[metric_name] = (
                        sum(v * w for v, w in values_weights) / total_weight
                    )
                else:
                    aggregated[metric_name] = 0.0

            elif aggregation_type == "median":
                # Use unweighted median (weighted median is complex)
                aggregated[metric_name] = statistics.median(values)

            elif aggregation_type == "sum":
                aggregated[metric_name] = sum(values)

            elif aggregation_type == "min":
                aggregated[metric_name] = min(values)

            elif aggregation_type == "max":
                aggregated[metric_name] = max(values)

            elif aggregation_type == "count":
                aggregated[metric_name] = sum(weights)

            else:
                logger.warning(f"Unknown aggregation type: {aggregation_type}")

        return aggregated

    def aggregate_by_region(
        self,
        metric_name: str,
        aggregation_type: str = "mean",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Aggregate a metric with per-region breakdown.

        Args:
            metric_name: Metric to aggregate
            aggregation_type: Type of aggregation
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary mapping region_id to aggregated value
        """
        result = {}

        for region_id in self.regions:
            regional_value = self.aggregate_metrics(
                metric_names=[metric_name],
                aggregation_type=aggregation_type,
                start_time=start_time,
                end_time=end_time,
                regions=[region_id],
            )

            if metric_name in regional_value:
                result[region_id] = regional_value[metric_name]

        return result

    def compute_percentiles(
        self,
        metric_name: str,
        percentiles: List[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        regions: Optional[List[str]] = None,
    ) -> Dict[float, float]:
        """
        Compute percentiles for a metric across regions.

        Args:
            metric_name: Metric to analyze
            percentiles: List of percentiles (0-100)
            start_time: Start of time range
            end_time: End of time range
            regions: Specific regions to include

        Returns:
            Dictionary mapping percentile to value
        """
        percentiles = percentiles or [50, 90, 95, 99]
        regions = regions or self.regions
        end_time = end_time or datetime.now(timezone.utc)
        start_time = start_time or (
            end_time - timedelta(seconds=self.aggregation_interval)
        )

        # Collect all values
        values = []
        for region_id in regions:
            if region_id not in self.regional_metrics:
                continue

            for rm in self.regional_metrics[region_id]:
                if start_time <= rm.timestamp <= end_time:
                    if metric_name in rm.metrics:
                        # Repeat value by count for proper distribution
                        values.extend([rm.metrics[metric_name]] * rm.count)

        if not values:
            return {}

        # Sort values
        values.sort()

        # Calculate percentiles
        result = {}
        for p in percentiles:
            idx = int(len(values) * p / 100)
            idx = min(idx, len(values) - 1)
            result[p] = values[idx]

        return result

    def get_regional_statistics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each region.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary mapping region_id to statistics
        """
        end_time = end_time or datetime.now(timezone.utc)
        start_time = start_time or (
            end_time - timedelta(seconds=self.aggregation_interval)
        )

        result = {}

        for region_id in self.regions:
            if region_id not in self.regional_metrics:
                result[region_id] = {
                    "data_points": 0,
                    "metrics": {},
                    "last_update": None,
                }
                continue

            # Filter by time
            filtered_metrics = [
                rm
                for rm in self.regional_metrics[region_id]
                if start_time <= rm.timestamp <= end_time
            ]

            # Aggregate metrics
            metrics_sum: Dict[str, List[float]] = defaultdict(list)
            total_count = 0

            for rm in filtered_metrics:
                total_count += rm.count
                for metric_name, value in rm.metrics.items():
                    metrics_sum[metric_name].append(value)

            # Calculate statistics
            metrics_stats = {}
            for metric_name, values in metrics_sum.items():
                if values:
                    metrics_stats[metric_name] = {
                        "mean": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }

            last_update = (
                max(rm.timestamp for rm in filtered_metrics)
                if filtered_metrics
                else None
            )

            result[region_id] = {
                "data_points": total_count,
                "metrics": metrics_stats,
                "last_update": last_update.isoformat() if last_update else None,
                "weight": self.region_weights.get(region_id, 1.0),
            }

        return result

    def _cleanup_old_metrics(self, region_id: str):
        """Clean up old metrics beyond retention period."""
        if region_id not in self.regional_metrics:
            return

        cutoff_time = datetime.now(timezone.utc) - timedelta(
            seconds=self.retention_period
        )

        self.regional_metrics[region_id] = [
            rm for rm in self.regional_metrics[region_id] if rm.timestamp >= cutoff_time
        ]

    def cleanup_all_regions(self):
        """Clean up old metrics for all regions."""
        for region_id in self.regions:
            self._cleanup_old_metrics(region_id)

    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global statistics across all regions.

        Returns:
            Global statistics dictionary
        """
        total_data_points = sum(
            sum(rm.count for rm in self.regional_metrics.get(region_id, []))
            for region_id in self.regions
        )

        # Get all unique metric names
        all_metrics = set()
        for region_id in self.regions:
            for rm in self.regional_metrics.get(region_id, []):
                all_metrics.update(rm.metrics.keys())

        return {
            "total_regions": len(self.regions),
            "active_regions": len(
                [r for r in self.regions if self.regional_metrics.get(r)]
            ),
            "total_data_points": total_data_points,
            "tracked_metrics": list(all_metrics),
            "aggregation_interval": self.aggregation_interval,
            "retention_period": self.retention_period,
        }
