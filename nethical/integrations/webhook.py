"""
Webhook and API Integration

Provides webhook dispatchers and API integration interfaces for external systems.

Features:
- Generic webhook dispatcher
- Slack webhook integration
- Discord webhook integration
- Custom API integration framework
- Retry logic and error handling
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib import request, error


class WebhookStatus(Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookPayload:
    """Webhook payload"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict())


@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    delivery_id: str
    url: str
    payload: WebhookPayload
    status: WebhookStatus = WebhookStatus.PENDING
    attempt_count: int = 0
    last_attempt: Optional[datetime] = None
    response_code: Optional[int] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'delivery_id': self.delivery_id,
            'url': self.url,
            'status': self.status.value,
            'attempt_count': self.attempt_count,
            'last_attempt': self.last_attempt.isoformat() if self.last_attempt else None,
            'response_code': self.response_code,
            'error_message': self.error_message
        }


class WebhookDispatcher(ABC):
    """Abstract webhook dispatcher"""
    
    @abstractmethod
    def dispatch(self, payload: WebhookPayload) -> WebhookDelivery:
        """Dispatch webhook payload"""
        pass


class HTTPWebhookDispatcher(WebhookDispatcher):
    """
    Generic HTTP webhook dispatcher
    
    Sends POST requests with JSON payloads to webhook URLs.
    """
    
    def __init__(self,
                 url: str,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """
        Initialize HTTP webhook dispatcher
        
        Args:
            url: Webhook URL
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.url = url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.deliveries: List[WebhookDelivery] = []
    
    def dispatch(self, payload: WebhookPayload) -> WebhookDelivery:
        """Dispatch webhook"""
        import uuid
        
        delivery = WebhookDelivery(
            delivery_id=str(uuid.uuid4())[:8],
            url=self.url,
            payload=payload
        )
        
        # Attempt delivery with retries
        for attempt in range(1, self.max_retries + 1):
            delivery.attempt_count = attempt
            delivery.last_attempt = datetime.now()
            delivery.status = WebhookStatus.RETRYING if attempt > 1 else WebhookStatus.PENDING
            
            try:
                # Prepare request
                data = payload.to_json().encode('utf-8')
                req = request.Request(
                    self.url,
                    data=data,
                    headers=self.headers,
                    method='POST'
                )
                
                # Send request
                with request.urlopen(req, timeout=self.timeout) as response:
                    delivery.response_code = response.getcode()
                    
                    if 200 <= response.getcode() < 300:
                        delivery.status = WebhookStatus.SUCCESS
                        logging.info(f"Webhook delivered successfully: {delivery.delivery_id}")
                        break
                    else:
                        delivery.status = WebhookStatus.FAILED
                        delivery.error_message = f"HTTP {response.getcode()}"
                        
            except error.HTTPError as e:
                delivery.response_code = e.code
                delivery.error_message = f"HTTP {e.code}: {e.reason}"
                logging.error(f"Webhook failed (HTTP {e.code}): {delivery.delivery_id}")
                
            except error.URLError as e:
                delivery.error_message = f"URL error: {e.reason}"
                logging.error(f"Webhook failed (URL error): {delivery.delivery_id}")
                
            except Exception as e:
                delivery.error_message = str(e)
                logging.error(f"Webhook failed (exception): {delivery.delivery_id} - {e}")
            
            # Retry delay
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        
        # Final status
        if delivery.status != WebhookStatus.SUCCESS:
            delivery.status = WebhookStatus.FAILED
        
        self.deliveries.append(delivery)
        return delivery
    
    def get_delivery_status(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery status"""
        for delivery in self.deliveries:
            if delivery.delivery_id == delivery_id:
                return delivery
        return None


class SlackWebhookDispatcher(HTTPWebhookDispatcher):
    """
    Slack webhook dispatcher
    
    Formats messages for Slack's webhook format.
    """
    
    def __init__(self, webhook_url: str, **kwargs):
        """
        Initialize Slack webhook
        
        Args:
            webhook_url: Slack webhook URL
        """
        super().__init__(url=webhook_url, **kwargs)
    
    def send_message(self,
                     text: str,
                     username: Optional[str] = "Nethical",
                     icon_emoji: Optional[str] = ":robot_face:",
                     attachments: Optional[List[Dict]] = None) -> WebhookDelivery:
        """
        Send message to Slack
        
        Args:
            text: Message text
            username: Bot username
            icon_emoji: Bot emoji icon
            attachments: Optional attachments
            
        Returns:
            WebhookDelivery record
        """
        payload_data = {
            'text': text,
            'username': username,
            'icon_emoji': icon_emoji
        }
        
        if attachments:
            payload_data['attachments'] = attachments
        
        payload = WebhookPayload(
            event_type='slack_message',
            data=payload_data
        )
        
        return self.dispatch(payload)
    
    def send_alert(self,
                   title: str,
                   message: str,
                   severity: str = "info",
                   fields: Optional[Dict[str, str]] = None) -> WebhookDelivery:
        """
        Send formatted alert to Slack
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            fields: Additional fields
            
        Returns:
            WebhookDelivery record
        """
        # Color based on severity
        color_map = {
            'info': '#36a64f',
            'warning': '#ff9900',
            'error': '#ff0000',
            'critical': '#990000'
        }
        color = color_map.get(severity.lower(), '#dddddd')
        
        attachment = {
            'color': color,
            'title': title,
            'text': message,
            'timestamp': int(datetime.now().timestamp())
        }
        
        if fields:
            attachment['fields'] = [
                {'title': k, 'value': v, 'short': True}
                for k, v in fields.items()
            ]
        
        return self.send_message(
            text=f"*{severity.upper()}*",
            attachments=[attachment]
        )


class DiscordWebhookDispatcher(HTTPWebhookDispatcher):
    """
    Discord webhook dispatcher
    
    Formats messages for Discord's webhook format.
    """
    
    def __init__(self, webhook_url: str, **kwargs):
        """
        Initialize Discord webhook
        
        Args:
            webhook_url: Discord webhook URL
        """
        super().__init__(url=webhook_url, **kwargs)
    
    def send_message(self,
                     content: str,
                     username: Optional[str] = "Nethical",
                     embeds: Optional[List[Dict]] = None) -> WebhookDelivery:
        """
        Send message to Discord
        
        Args:
            content: Message content
            username: Bot username
            embeds: Optional embeds
            
        Returns:
            WebhookDelivery record
        """
        payload_data = {
            'content': content,
            'username': username
        }
        
        if embeds:
            payload_data['embeds'] = embeds
        
        payload = WebhookPayload(
            event_type='discord_message',
            data=payload_data
        )
        
        return self.dispatch(payload)


class APIIntegration(ABC):
    """
    Abstract API integration interface
    
    Base class for integrating with external APIs.
    """
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the API"""
        pass
    
    @abstractmethod
    def send_data(self, data: Dict[str, Any]) -> bool:
        """Send data to the API"""
        pass
    
    @abstractmethod
    def receive_data(self) -> Dict[str, Any]:
        """Receive data from the API"""
        pass


class GenericAPIIntegration(APIIntegration):
    """
    Generic REST API integration
    
    Provides a basic framework for REST API integrations.
    """
    
    def __init__(self,
                 base_url: str,
                 api_key: Optional[str] = None,
                 auth_header: str = "Authorization"):
        """
        Initialize API integration
        
        Args:
            base_url: API base URL
            api_key: API key for authentication
            auth_header: Header name for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.auth_header = auth_header
        self.authenticated = False
    
    def authenticate(self) -> bool:
        """Authenticate with API"""
        if not self.api_key:
            logging.warning("No API key provided, skipping authentication")
            return True
        
        # Stub implementation
        logging.info(f"[STUB] Would authenticate with {self.base_url}")
        self.authenticated = True
        return True
    
    def send_data(self, data: Dict[str, Any]) -> bool:
        """Send data to API"""
        # Stub implementation
        logging.info(f"[STUB] Would send data to {self.base_url}: {json.dumps(data)[:100]}...")
        return True
    
    def receive_data(self) -> Dict[str, Any]:
        """Receive data from API"""
        # Stub implementation
        logging.info(f"[STUB] Would receive data from {self.base_url}")
        return {'status': 'stub', 'data': None}


class WebhookManager:
    """
    Manage multiple webhook dispatchers
    
    Example:
        manager = WebhookManager()
        manager.add_webhook("slack", SlackWebhookDispatcher(slack_url))
        manager.add_webhook("discord", DiscordWebhookDispatcher(discord_url))
        
        manager.broadcast("system_alert", {"message": "High CPU usage"})
    """
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookDispatcher] = {}
    
    def add_webhook(self, name: str, dispatcher: WebhookDispatcher):
        """Add a webhook dispatcher"""
        self.webhooks[name] = dispatcher
    
    def broadcast(self, event_type: str, data: Dict[str, Any], **metadata) -> Dict[str, WebhookDelivery]:
        """Broadcast to all webhooks"""
        payload = WebhookPayload(
            event_type=event_type,
            data=data,
            metadata=metadata
        )
        
        results = {}
        for name, webhook in self.webhooks.items():
            try:
                delivery = webhook.dispatch(payload)
                results[name] = delivery
            except Exception as e:
                logging.error(f"Failed to dispatch to {name}: {e}")
        
        return results
    
    def send_to(self, webhook_name: str, event_type: str, data: Dict[str, Any]) -> Optional[WebhookDelivery]:
        """Send to specific webhook"""
        if webhook_name not in self.webhooks:
            logging.error(f"Webhook '{webhook_name}' not found")
            return None
        
        payload = WebhookPayload(
            event_type=event_type,
            data=data
        )
        
        return self.webhooks[webhook_name].dispatch(payload)


if __name__ == "__main__":
    # Demo usage
    print("Webhook integration initialized")
    
    # NOTE: These are example URLs - replace with actual webhook URLs for testing
    # slack_webhook = SlackWebhookDispatcher("https://hooks.slack.com/services/YOUR/WEBHOOK/URL")
    # slack_webhook.send_alert("Test Alert", "This is a test alert", "info")
    
    print("Webhook integration demo complete (stub mode)")
