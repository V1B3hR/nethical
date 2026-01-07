"""Alerting module for Nethical threat detection system.

This module provides multi-channel alerting capabilities including:
- Slack webhooks
- Email (SMTP)
- PagerDuty
- Discord webhooks
- Custom webhooks
- Rate limiting to prevent alert storms
"""

from nethical.alerting.alert_manager import (
    AlertManager,
    AlertSeverity,
    AlertChannel,
    RateLimiter
)
from nethical.alerting.alert_rules import AlertRules

__all__ = [
    "AlertManager",
    "AlertSeverity",
    "AlertChannel",
    "RateLimiter",
    "AlertRules"
]
