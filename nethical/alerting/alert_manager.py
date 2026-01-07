"""Multi-channel alert manager for Nethical.

Supports alerting through:
- Slack (webhooks)
- Email (SMTP)
- PagerDuty
- Discord
- Custom webhooks

Includes rate limiting to prevent alert storms.
"""

import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from enum import Enum

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert channel types."""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    DISCORD = "discord"


class RateLimiter:
    """Prevent alert storms with rate limiting."""
    
    def __init__(self, max_alerts_per_minute: int = 10):
        """Initialize rate limiter.
        
        Args:
            max_alerts_per_minute: Maximum alerts per minute per alert key
        """
        self.max_alerts = max_alerts_per_minute
        self.recent_alerts: Dict[str, datetime] = {}
    
    def should_send(self, alert_key: str, severity: AlertSeverity) -> bool:
        """Check if alert should be sent based on rate limits.
        
        Args:
            alert_key: Unique key for the alert
            severity: Alert severity
            
        Returns:
            True if alert should be sent
        """
        # Always send critical alerts
        if severity == AlertSeverity.CRITICAL:
            return True
        
        # Rate limit other alerts
        now = datetime.now(timezone.utc)
        if alert_key in self.recent_alerts:
            last_sent = self.recent_alerts[alert_key]
            time_diff = (now - last_sent).total_seconds()
            if time_diff < 60:  # Less than 1 minute
                return False
        
        self.recent_alerts[alert_key] = now
        
        # Clean old entries
        self._cleanup_old_entries()
        
        return True
    
    def _cleanup_old_entries(self) -> None:
        """Clean up old rate limit entries."""
        now = datetime.now(timezone.utc)
        cutoff = 300  # 5 minutes
        
        to_delete = [
            key for key, timestamp in self.recent_alerts.items()
            if (now - timestamp).total_seconds() > cutoff
        ]
        
        for key in to_delete:
            del self.recent_alerts[key]


class AlertManager:
    """Multi-channel alerting system.
    
    Supported channels:
    - Slack (webhooks)
    - Email (SMTP)
    - PagerDuty
    - Custom webhooks
    - Discord
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize alert manager.
        
        Args:
            config: Configuration dictionary with channel credentials
        """
        self.config = config
        self.alert_history: List[Dict[str, Any]] = []
        self.rate_limiter = RateLimiter(
            max_alerts_per_minute=config.get('max_alerts_per_minute', 10)
        )
        self.enabled = config.get('enabled', True)
        
        if not AIOHTTP_AVAILABLE and self.enabled:
            logger.warning(
                "aiohttp not available. Webhook-based alerting will be disabled. "
                "Install with: pip install aiohttp>=3.9.0"
            )
    
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        channels: List[AlertChannel],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send alert to multiple channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            channels: List of channels to send to
            metadata: Optional additional metadata
        """
        if not self.enabled:
            logger.debug(f"Alerting disabled. Skipping alert: {title}")
            return
        
        # Rate limiting to prevent alert storms
        alert_key = f"{title}:{severity.value}"
        if not self.rate_limiter.should_send(alert_key, severity):
            logger.debug(f"Alert rate limited: {title}")
            return
        
        # Send to all channels
        tasks = []
        for channel in channels:
            try:
                if channel == AlertChannel.SLACK:
                    tasks.append(self._send_slack(title, message, severity, metadata))
                elif channel == AlertChannel.EMAIL:
                    tasks.append(self._send_email(title, message, severity, metadata))
                elif channel == AlertChannel.PAGERDUTY:
                    tasks.append(self._send_pagerduty(title, message, severity, metadata))
                elif channel == AlertChannel.WEBHOOK:
                    tasks.append(self._send_webhook(title, message, severity, metadata))
                elif channel == AlertChannel.DISCORD:
                    tasks.append(self._send_discord(title, message, severity, metadata))
            except Exception as e:
                logger.error(f"Error setting up alert for {channel.value}: {e}")
        
        # Send all alerts concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Alert send failed for {channels[i].value}: {result}")
        
        # Log alert to history
        self.alert_history.append({
            'title': title,
            'message': message,
            'severity': severity.value,
            'channels': [c.value for c in channels],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {}
        })
        
        # Keep history manageable (last 1000 alerts)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
    
    async def _send_slack(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Send Slack webhook alert."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping Slack alert")
            return
        
        webhook_url = self.config.get('slack_webhook_url')
        if not webhook_url:
            logger.debug("Slack webhook URL not configured")
            return
        
        # Color based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9800",
            AlertSeverity.CRITICAL: "#d32f2f"
        }
        color = color_map.get(severity, "#cccccc")
        
        # Build payload
        payload = {
            "attachments": [{
                "color": color,
                "title": f"🚨 {title}",
                "text": message,
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in (metadata or {}).items()
                ][:10],  # Limit to 10 fields
                "footer": "Nethical Threat Detection",
                "ts": int(datetime.now(timezone.utc).timestamp())
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logger.error(f"Slack webhook returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            raise
    
    async def _send_email(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Send email via SMTP."""
        smtp_config = self.config.get('smtp', {})
        if not smtp_config:
            logger.debug("SMTP not configured")
            return
        
        # Build email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{severity.value.upper()}] {title}"
        msg['From'] = smtp_config.get('from', 'alerts@nethical.ai')
        msg['To'] = smtp_config.get('to', 'security-team@company.com')
        
        # Create email body
        text_body = f"{message}\n\nSeverity: {severity.value}\n\n"
        if metadata:
            text_body += "Additional Information:\n"
            for key, value in metadata.items():
                text_body += f"  {key}: {value}\n"
        
        html_body = f"""
        <html>
          <head></head>
          <body>
            <h2 style="color: {'red' if severity == AlertSeverity.CRITICAL else 'orange'}">
              {title}
            </h2>
            <p>{message}</p>
            <p><strong>Severity:</strong> {severity.value.upper()}</p>
            {'<h3>Additional Information:</h3><ul>' + ''.join([f'<li><strong>{k}:</strong> {v}</li>' for k, v in (metadata or {}).items()]) + '</ul>' if metadata else ''}
            <hr>
            <p><em>Nethical Threat Detection System</em></p>
          </body>
        </html>
        """
        
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_smtp, smtp_config, msg)
    
    def _send_smtp(self, config: Dict[str, Any], msg: MIMEMultipart) -> None:
        """Synchronous SMTP send."""
        try:
            with smtplib.SMTP(config['host'], config.get('port', 587), timeout=10) as server:
                if config.get('use_tls', True):
                    server.starttls()
                if config.get('username'):
                    server.login(config['username'], config['password'])
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise
    
    async def _send_pagerduty(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Send PagerDuty event."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping PagerDuty alert")
            return
        
        api_key = self.config.get('pagerduty_api_key')
        if not api_key:
            logger.debug("PagerDuty API key not configured")
            return
        
        # Map severity to PagerDuty severity
        pd_severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical"
        }
        
        payload = {
            "routing_key": api_key,
            "event_action": "trigger",
            "payload": {
                "summary": title,
                "severity": pd_severity_map.get(severity, "error"),
                "source": "nethical",
                "custom_details": metadata or {}
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status not in (200, 202):
                        logger.error(f"PagerDuty returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            raise
    
    async def _send_webhook(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Send custom webhook."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping webhook alert")
            return
        
        webhook_url = self.config.get('webhook_url')
        if not webhook_url:
            logger.debug("Webhook URL not configured")
            return
        
        payload = {
            "title": title,
            "message": message,
            "severity": severity.value,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status not in (200, 201, 202):
                        logger.error(f"Webhook returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            raise
    
    async def _send_discord(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Send Discord webhook alert."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping Discord alert")
            return
        
        webhook_url = self.config.get('discord_webhook_url')
        if not webhook_url:
            logger.debug("Discord webhook URL not configured")
            return
        
        # Color based on severity
        color_map = {
            AlertSeverity.INFO: 0x36a64f,
            AlertSeverity.WARNING: 0xff9800,
            AlertSeverity.CRITICAL: 0xd32f2f
        }
        color = color_map.get(severity, 0xcccccc)
        
        payload = {
            "embeds": [{
                "title": f"🚨 {title}",
                "description": message,
                "color": color,
                "fields": [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in (metadata or {}).items()
                ][:25],  # Discord limit
                "footer": {"text": "Nethical Threat Detection"},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status not in (200, 204):
                        logger.error(f"Discord webhook returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            raise
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.alert_history[-limit:]
    
    def clear_history(self) -> int:
        """Clear alert history.
        
        Returns:
            Number of alerts cleared
        """
        count = len(self.alert_history)
        self.alert_history.clear()
        return count
