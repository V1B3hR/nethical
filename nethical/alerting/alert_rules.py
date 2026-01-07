"""Predefined alerting rules for common scenarios.

This module provides common alert rules for:
- High latency detection
- High threat rate detection
- High error rate detection
"""

import logging
from typing import Dict, Any
from nethical.alerting.alert_manager import AlertManager, AlertSeverity, AlertChannel

logger = logging.getLogger(__name__)


class AlertRules:
    """Common alerting rules for Nethical threat detection system."""

    @staticmethod
    async def check_high_latency(
        metrics: Dict[str, Any],
        alert_manager: AlertManager,
        threshold_ms: float = 200.0
    ) -> None:
        """Alert on P95 latency > threshold.
        
        Args:
            metrics: Current metrics data
            alert_manager: AlertManager instance
            threshold_ms: Latency threshold in milliseconds
        """
        try:
            # Check histogram stats for latency
            latency_stats = metrics.get('histograms', {}).get('action_latency', {})
            p95_latency_s = latency_stats.get('p95', 0.0)
            p95_latency_ms = p95_latency_s * 1000
            
            if p95_latency_ms > threshold_ms:
                await alert_manager.send_alert(
                    title="High Latency Detected",
                    message=f"P95 latency: {p95_latency_ms:.2f}ms (threshold: {threshold_ms}ms)",
                    severity=AlertSeverity.WARNING,
                    channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                    metadata={
                        'p95_latency_ms': f"{p95_latency_ms:.2f}",
                        'threshold_ms': str(threshold_ms),
                        'p99_latency_ms': f"{latency_stats.get('p99', 0.0) * 1000:.2f}",
                        'mean_latency_ms': f"{latency_stats.get('mean', 0.0) * 1000:.2f}"
                    }
                )
        except Exception as e:
            logger.error(f"Error checking high latency rule: {e}")

    @staticmethod
    async def check_high_threat_rate(
        metrics: Dict[str, Any],
        alert_manager: AlertManager,
        threshold: float = 0.5
    ) -> None:
        """Alert on threat detection rate > threshold.
        
        Args:
            metrics: Current metrics data
            alert_manager: AlertManager instance
            threshold: Threat rate threshold (0.0-1.0)
        """
        try:
            # Calculate threat rate from counters
            counters = metrics.get('counters', {})
            actions = counters.get('actions', {})
            violations = counters.get('violations', {})
            
            total_actions = sum(actions.values()) if actions else 0
            total_violations = sum(violations.values()) if violations else 0
            
            if total_actions > 0:
                threat_rate = total_violations / total_actions
                
                if threat_rate > threshold:
                    await alert_manager.send_alert(
                        title="High Threat Rate Detected",
                        message=f"Threat detection rate: {threat_rate*100:.1f}% (threshold: {threshold*100:.1f}%)",
                        severity=AlertSeverity.CRITICAL,
                        channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY, AlertChannel.EMAIL],
                        metadata={
                            'threat_rate': f"{threat_rate*100:.1f}%",
                            'total_violations': str(total_violations),
                            'total_actions': str(total_actions),
                            'threshold': f"{threshold*100:.1f}%"
                        }
                    )
        except Exception as e:
            logger.error(f"Error checking high threat rate rule: {e}")

    @staticmethod
    async def check_error_rate(
        metrics: Dict[str, Any],
        alert_manager: AlertManager,
        threshold: float = 0.05
    ) -> None:
        """Alert on error rate > threshold.
        
        Args:
            metrics: Current metrics data
            alert_manager: AlertManager instance
            threshold: Error rate threshold (0.0-1.0)
        """
        try:
            # Check error rate gauge
            gauges = metrics.get('gauges', {})
            error_rates = gauges.get('error_rate', {})
            
            for component, error_rate in error_rates.items():
                if error_rate > threshold:
                    await alert_manager.send_alert(
                        title="High Error Rate",
                        message=f"Error rate for {component}: {error_rate*100:.1f}% (threshold: {threshold*100:.1f}%)",
                        severity=AlertSeverity.WARNING,
                        channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                        metadata={
                            'component': component,
                            'error_rate': f"{error_rate*100:.1f}%",
                            'threshold': f"{threshold*100:.1f}%"
                        }
                    )
        except Exception as e:
            logger.error(f"Error checking error rate rule: {e}")

    @staticmethod
    async def check_detector_health(
        metrics: Dict[str, Any],
        alert_manager: AlertManager
    ) -> None:
        """Alert on unhealthy detectors.
        
        Args:
            metrics: Current metrics data
            alert_manager: AlertManager instance
        """
        try:
            gauges = metrics.get('gauges', {})
            # Check for any detector health metrics that are 0 (unhealthy)
            # This would require the metrics to track detector health
            
            # For now, check if there are any error spikes
            error_rates = gauges.get('error_rate', {})
            
            for component, error_rate in error_rates.items():
                if error_rate > 0.1:  # More than 10% error rate indicates unhealthy
                    await alert_manager.send_alert(
                        title="Detector Health Issue",
                        message=f"Detector {component} may be unhealthy (error rate: {error_rate*100:.1f}%)",
                        severity=AlertSeverity.CRITICAL,
                        channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
                        metadata={
                            'detector': component,
                            'error_rate': f"{error_rate*100:.1f}%"
                        }
                    )
        except Exception as e:
            logger.error(f"Error checking detector health rule: {e}")

    @staticmethod
    async def check_cache_performance(
        metrics: Dict[str, Any],
        alert_manager: AlertManager,
        min_hit_rate: float = 0.5
    ) -> None:
        """Alert on low cache hit rate.
        
        Args:
            metrics: Current metrics data
            alert_manager: AlertManager instance
            min_hit_rate: Minimum acceptable cache hit rate
        """
        try:
            counters = metrics.get('counters', {})
            cache_data = {}
            
            # Calculate hit rate for each cache type
            # This would require tracking cache hits/misses in metrics
            # For now, this is a placeholder for when cache metrics are available
            
            # Example: If we had cache_hits and cache_misses counters
            # hits = counters.get('cache_hits', {})
            # misses = counters.get('cache_misses', {})
            # for cache_type in set(hits.keys()) | set(misses.keys()):
            #     total = hits.get(cache_type, 0) + misses.get(cache_type, 0)
            #     if total > 0:
            #         hit_rate = hits.get(cache_type, 0) / total
            #         if hit_rate < min_hit_rate:
            #             await alert_manager.send_alert(...)
            
        except Exception as e:
            logger.error(f"Error checking cache performance rule: {e}")

    @staticmethod
    async def evaluate_all_rules(
        metrics: Dict[str, Any],
        alert_manager: AlertManager,
        config: Dict[str, Any] = None
    ) -> None:
        """Evaluate all alert rules against current metrics.
        
        Args:
            metrics: Current metrics data
            alert_manager: AlertManager instance
            config: Optional configuration overrides
        """
        config = config or {}
        
        # Check all rules
        await AlertRules.check_high_latency(
            metrics,
            alert_manager,
            threshold_ms=config.get('high_latency_threshold_ms', 200.0)
        )
        
        await AlertRules.check_high_threat_rate(
            metrics,
            alert_manager,
            threshold=config.get('high_threat_rate_threshold', 0.5)
        )
        
        await AlertRules.check_error_rate(
            metrics,
            alert_manager,
            threshold=config.get('high_error_rate_threshold', 0.05)
        )
        
        await AlertRules.check_detector_health(
            metrics,
            alert_manager
        )
        
        await AlertRules.check_cache_performance(
            metrics,
            alert_manager,
            min_hit_rate=config.get('min_cache_hit_rate', 0.5)
        )
