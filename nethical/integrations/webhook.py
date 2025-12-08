"""
Webhook and API Integration

Provides webhook dispatchers and API integration interfaces for external systems.

Features:
- Generic webhook dispatcher
- Slack webhook integration
- Discord webhook integration
- Custom API integration framework
- Retry logic and error handling

Enhancements for nethical:
- UTC timezone-aware timestamps
- Deterministic JSON serialization for signing
- Optional HMAC-SHA256 signing (X-Nethical-Signature)
- Idempotency header (Idempotency-Key)
- Exponential backoff with jitter and smart retry policy
- Rate limit control (requests-per-second)
- Retry-After handling for 429 responses
- Response metadata capture and duration
- User-Agent control and dry-run mode
- Parallel broadcast via ThreadPoolExecutor
"""

import json
import logging
import time
import hmac
import hashlib
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib import request, error
from urllib.parse import urlparse
from http.client import HTTPResponse
from email.message import Message
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to deterministic JSON (stable keys, compact)"""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


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
    duration_ms: Optional[float] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body_excerpt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "delivery_id": self.delivery_id,
            "url": self.url,
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "last_attempt": (
                self.last_attempt.isoformat() if self.last_attempt else None
            ),
            "response_code": self.response_code,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "response_headers": self.response_headers,
            "response_body_excerpt": self.response_body_excerpt,
        }


class WebhookDispatcher(ABC):
    """Abstract webhook dispatcher"""

    @abstractmethod
    def dispatch(self, payload: WebhookPayload) -> WebhookDelivery:
        """Dispatch webhook payload"""


class HTTPWebhookDispatcher(WebhookDispatcher):
    """
    Generic HTTP webhook dispatcher

    Sends POST requests with JSON payloads to webhook URLs.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 1.5,
        secret: Optional[str] = None,
        user_agent: Optional[
            str
        ] = "nethical-webhook/1.0 (+https://github.com/V1B3hR/nethical)",
        rate_limit_per_sec: Optional[float] = None,
        dry_run: bool = False,
        on_success: Optional[Callable[[WebhookDelivery], None]] = None,
        on_failure: Optional[Callable[[WebhookDelivery], None]] = None,
    ):
        """
        Initialize HTTP webhook dispatcher

        Args:
            url: Webhook URL
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries in seconds
            backoff_factor: Exponential backoff multiplier
            secret: Optional HMAC-SHA256 secret to sign payload (X-Nethical-Signature)
            user_agent: Optional User-Agent header value
            rate_limit_per_sec: Optional rate limit (requests per second)
            dry_run: If True, skip network calls and mark as success
            on_success: Callback invoked on successful delivery with WebhookDelivery
            on_failure: Callback invoked when delivery fails after retries with WebhookDelivery
        """
        # Validate URL scheme for security
        self._validate_url_scheme(url)
        self.url = url
        self.headers = (
            dict(headers) if headers else {"Content-Type": "application/json"}
        )
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.backoff_factor = max(1.0, float(backoff_factor))
        self.secret = secret
        self.user_agent = user_agent
        self.rate_limit_per_sec = rate_limit_per_sec
        self.dry_run = dry_run
        self.on_success = on_success
        self.on_failure = on_failure

        # Ensure User-Agent present
        if self.user_agent:
            self.headers.setdefault("User-Agent", self.user_agent)

        self.deliveries: List[WebhookDelivery] = []
        self._deliveries_lock = threading.Lock()

        # Rate limiting state
        self._rate_lock = threading.Lock()
        self._last_request_ts: float = 0.0

        # Retryable HTTP status codes
        self._retry_statuses = {408, 425, 429, 500, 502, 503, 504}

    @staticmethod
    def _validate_url_scheme(url: str) -> None:
        """
        Validate URL scheme to prevent SSRF attacks.

        Only http and https schemes are allowed.

        Args:
            url: URL to validate

        Raises:
            ValueError: If URL scheme is not http or https
        """
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(
                f"Unsupported URL scheme: {parsed.scheme}. "
                f"Only http and https are allowed."
            )

    def _should_retry(
        self, status_code: Optional[int], err: Optional[BaseException], attempt: int
    ) -> bool:
        if attempt >= self.max_retries:
            return False
        if isinstance(err, error.URLError):
            return True  # network issues are retryable
        if isinstance(err, error.HTTPError):
            try:
                code = int(err.code)
            except Exception:
                code = None
            return code in self._retry_statuses
        if status_code is not None:
            return status_code in self._retry_statuses
        return False

    def _compute_delay(
        self, base: float, attempt: int, retry_after: Optional[float] = None
    ) -> float:
        if retry_after is not None:
            return max(0.0, retry_after)
        # Exponential backoff with jitter (up to 25%)
        delay = base * (self.backoff_factor ** max(0, attempt - 1))
        jitter = random.uniform(0, delay * 0.25)
        return delay + jitter

    def _apply_rate_limit(self):
        if not self.rate_limit_per_sec or self.rate_limit_per_sec <= 0:
            return
        min_interval = 1.0 / float(self.rate_limit_per_sec)
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_ts = time.monotonic()

    def _sign_payload(self, body: bytes) -> str:
        assert self.secret is not None
        signature = hmac.new(
            self.secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def _read_response_excerpt(self, resp: HTTPResponse, limit: int = 512) -> str:
        try:
            body = resp.read(limit)
            # If there is more, do not block fetching; best effort only.
            return body.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _headers_to_dict(self, headers: Message) -> Dict[str, str]:
        try:
            return {k: v for k, v in headers.items()}
        except Exception:
            # Fallback
            return {}

    def dispatch(self, payload: WebhookPayload) -> WebhookDelivery:
        """Dispatch webhook"""
        import uuid

        delivery = WebhookDelivery(
            delivery_id=str(uuid.uuid4())[:8], url=self.url, payload=payload
        )

        # Prepare JSON upfront for deterministic signing and retries
        body_str = payload.to_json()
        body_bytes = body_str.encode("utf-8")

        # Pre-configure static headers for this delivery
        req_headers = dict(self.headers)
        # Idempotency
        req_headers.setdefault("Idempotency-Key", delivery.delivery_id)
        # Signature
        if self.secret:
            req_headers["X-Nethical-Signature"] = self._sign_payload(body_bytes)

        # Dry-run mode (no network)
        if self.dry_run:
            delivery.status = WebhookStatus.SUCCESS
            delivery.response_code = 0
            delivery.error_message = None
            delivery.duration_ms = 0.0
            with self._deliveries_lock:
                self.deliveries.append(delivery)
            logging.info(
                f"[DRY-RUN] Webhook delivery simulated: {delivery.delivery_id} -> {delivery.url}"
            )
            if self.on_success:
                try:
                    self.on_success(delivery)
                except Exception as hook_err:
                    logging.debug(f"on_success hook error (ignored): {hook_err}")
            return delivery

        # Attempt delivery with retries
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            delivery.attempt_count = attempt
            delivery.last_attempt = datetime.now(timezone.utc)
            delivery.status = (
                WebhookStatus.RETRYING if attempt > 1 else WebhookStatus.PENDING
            )

            # Rate limit if configured
            self._apply_rate_limit()

            start = time.monotonic()
            retry_after_secs: Optional[float] = None
            try:
                # Prepare request
                req = request.Request(
                    self.url, data=body_bytes, headers=req_headers, method="POST"
                )

                # Send request
                with request.urlopen(req, timeout=self.timeout) as response:
                    duration = (time.monotonic() - start) * 1000.0
                    delivery.duration_ms = round(duration, 3)
                    delivery.response_code = response.getcode()
                    delivery.response_headers = self._headers_to_dict(response.headers)
                    delivery.response_body_excerpt = self._read_response_excerpt(
                        response
                    )

                    if 200 <= response.getcode() < 300:
                        delivery.status = WebhookStatus.SUCCESS
                        logging.info(
                            f"Webhook delivered successfully: {delivery.delivery_id} ({response.getcode()})"
                        )
                        break
                    else:
                        delivery.status = WebhookStatus.FAILED
                        delivery.error_message = f"HTTP {response.getcode()}"
                        logging.warning(
                            f"Webhook non-2xx response: {delivery.delivery_id} ({response.getcode()})"
                        )

            except error.HTTPError as e:
                duration = (time.monotonic() - start) * 1000.0
                delivery.duration_ms = round(duration, 3)
                delivery.response_code = e.code
                delivery.response_headers = (
                    self._headers_to_dict(e.headers)
                    if hasattr(e, "headers") and e.headers
                    else None
                )
                # Try to read a small excerpt of the error body
                try:
                    excerpt = e.read(512)
                    delivery.response_body_excerpt = excerpt.decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    pass

                last_err = e
                delivery.error_message = f"HTTP {e.code}: {e.reason}"
                logging.error(
                    f"Webhook failed (HTTP {e.code}): {delivery.delivery_id} - {e.reason}"
                )

                # Retry-After support (429 / 503 typically)
                if e.code == 429 and e.headers:
                    ra = e.headers.get("Retry-After")
                    if ra:
                        try:
                            # Numeric seconds preferred; RFC allows HTTP-date too
                            retry_after_secs = float(ra)
                        except ValueError:
                            retry_after_secs = None

            except error.URLError as e:
                duration = (time.monotonic() - start) * 1000.0
                delivery.duration_ms = round(duration, 3)
                last_err = e
                delivery.error_message = f"URL error: {getattr(e, 'reason', str(e))}"
                logging.error(
                    f"Webhook failed (URL error): {delivery.delivery_id} - {delivery.error_message}"
                )

            except Exception as e:
                duration = (time.monotonic() - start) * 1000.0
                delivery.duration_ms = round(duration, 3)
                last_err = e
                delivery.error_message = str(e)
                logging.error(
                    f"Webhook failed (exception): {delivery.delivery_id} - {e}"
                )

            # Decide retry
            if delivery.status == WebhookStatus.SUCCESS:
                break

            if not self._should_retry(delivery.response_code, last_err, attempt):
                # no more retries
                break

            # Retry delay
            if attempt < self.max_retries:
                sleep_for = self._compute_delay(
                    self.retry_delay, attempt, retry_after=retry_after_secs
                )
                logging.debug(
                    f"Retrying webhook {delivery.delivery_id} in {sleep_for:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(sleep_for)

        # Final status
        if delivery.status != WebhookStatus.SUCCESS:
            delivery.status = WebhookStatus.FAILED
            if self.on_failure:
                try:
                    self.on_failure(delivery)
                except Exception as hook_err:
                    logging.debug(f"on_failure hook error (ignored): {hook_err}")
        else:
            if self.on_success:
                try:
                    self.on_success(delivery)
                except Exception as hook_err:
                    logging.debug(f"on_success hook error (ignored): {hook_err}")

        with self._deliveries_lock:
            self.deliveries.append(delivery)
        return delivery

    def get_delivery_status(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery status"""
        with self._deliveries_lock:
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

    def send_message(
        self,
        text: Optional[str] = None,
        username: Optional[str] = "Nethical",
        icon_emoji: Optional[str] = ":robot_face:",
        attachments: Optional[List[Dict]] = None,
        blocks: Optional[List[Dict]] = None,
    ) -> WebhookDelivery:
        """
        Send message to Slack

        Args:
            text: Message text
            username: Bot username
            icon_emoji: Bot emoji icon
            attachments: Optional attachments (legacy)
            blocks: Optional blocks (preferred)

        Returns:
            WebhookDelivery record
        """
        payload_data: Dict[str, Any] = {
            "username": username,
        }
        if icon_emoji:
            payload_data["icon_emoji"] = icon_emoji
        if text is not None:
            payload_data["text"] = text
        if blocks:
            payload_data["blocks"] = blocks
        if attachments:
            payload_data["attachments"] = attachments

        payload = WebhookPayload(event_type="slack_message", data=payload_data)

        return self.dispatch(payload)

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        fields: Optional[Dict[str, str]] = None,
        use_blocks: bool = True,
    ) -> WebhookDelivery:
        """
        Send formatted alert to Slack

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            fields: Additional key-value fields
            use_blocks: Use Slack Blocks instead of attachments

        Returns:
            WebhookDelivery record
        """
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000",
            "critical": "#990000",
        }
        color = color_map.get(severity.lower(), "#dddddd")

        if use_blocks:
            blocks: List[Dict[str, Any]] = [
                {"type": "header", "text": {"type": "plain_text", "text": f"{title}"}},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Severity:* `{severity.upper()}`\n{message}",
                    },
                },
            ]
            if fields:
                field_list = [
                    {"type": "mrkdwn", "text": f"*{k}:*\n{v}"}
                    for k, v in fields.items()
                ]
                blocks.append({"type": "section", "fields": field_list})
            # Add a context with timestamp
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Timestamp: `{datetime.now(timezone.utc).isoformat()}`",
                        }
                    ],
                }
            )
            return self.send_message(text=None, blocks=blocks)
        else:
            attachment: Dict[str, Any] = {
                "color": color,
                "title": title,
                "text": message,
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }
            if fields:
                attachment["fields"] = [
                    {"title": k, "value": v, "short": True} for k, v in fields.items()
                ]
            return self.send_message(
                text=f"*{severity.upper()}*", attachments=[attachment]
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

    def send_message(
        self,
        content: str,
        username: Optional[str] = "Nethical",
        embeds: Optional[List[Dict]] = None,
        avatar_url: Optional[str] = None,
    ) -> WebhookDelivery:
        """
        Send message to Discord

        Args:
            content: Message content
            username: Bot username
            embeds: Optional embeds
            avatar_url: Optional avatar URL

        Returns:
            WebhookDelivery record
        """
        payload_data: Dict[str, Any] = {"content": content, "username": username}

        if avatar_url:
            payload_data["avatar_url"] = avatar_url

        if embeds:
            payload_data["embeds"] = embeds

        payload = WebhookPayload(event_type="discord_message", data=payload_data)

        return self.dispatch(payload)

    def send_embed(
        self,
        title: str,
        description: str,
        color: int = 0x2ECC71,
        fields: Optional[Dict[str, str]] = None,
    ) -> WebhookDelivery:
        """
        Helper to send a single embed
        """
        embed: Dict[str, Any] = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if fields:
            embed["fields"] = [
                {"name": k, "value": v, "inline": True} for k, v in fields.items()
            ]
        return self.send_message(content="", embeds=[embed])


class APIIntegration(ABC):
    """
    Abstract API integration interface

    Base class for integrating with external APIs.
    """

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the API"""

    @abstractmethod
    def send_data(self, data: Dict[str, Any]) -> bool:
        """Send data to the API"""

    @abstractmethod
    def receive_data(self) -> Dict[str, Any]:
        """Receive data from the API"""


class GenericAPIIntegration(APIIntegration):
    """
    Generic REST API integration

    Provides a basic framework for REST API integrations.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        auth_header: str = "Authorization",
    ):
        """
        Initialize API integration

        Args:
            base_url: API base URL
            api_key: API key for authentication
            auth_header: Header name for authentication
        """
        self.base_url = base_url.rstrip("/")
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
        logging.info(
            f"[STUB] Would send data to {self.base_url}: {json.dumps(data, sort_keys=True)[:100]}..."
        )
        return True

    def receive_data(self) -> Dict[str, Any]:
        """Receive data from API"""
        # Stub implementation
        logging.info(f"[STUB] Would receive data from {self.base_url}")
        return {"status": "stub", "data": None}


class WebhookManager:
    """
    Manage multiple webhook dispatchers

    Example:
        manager = WebhookManager()
        manager.add_webhook("slack", SlackWebhookDispatcher(slack_url))
        manager.add_webhook("discord", DiscordWebhookDispatcher(discord_url))

        manager.broadcast("system_alert", {"message": "High CPU usage"})
    """

    def __init__(self, max_workers: int = 4):
        self.webhooks: Dict[str, WebhookDispatcher] = {}
        self.max_workers = max_workers

    def add_webhook(self, name: str, dispatcher: WebhookDispatcher):
        """Add a webhook dispatcher"""
        self.webhooks[name] = dispatcher

    def broadcast(
        self, event_type: str, data: Dict[str, Any], **metadata
    ) -> Dict[str, WebhookDelivery]:
        """Broadcast to all webhooks in parallel"""
        payload = WebhookPayload(event_type=event_type, data=data, metadata=metadata)

        results: Dict[str, WebhookDelivery] = {}
        if not self.webhooks:
            logging.warning("No webhooks configured for broadcast")
            return results

        def _send(name: str, wh: WebhookDispatcher) -> (str, WebhookDelivery):
            try:
                delivery = wh.dispatch(payload)
                return name, delivery
            except Exception as e:
                logging.error(f"Failed to dispatch to {name}: {e}")
                # Create a failure record for consistency
                import uuid

                return name, WebhookDelivery(
                    delivery_id=str(uuid.uuid4())[:8],
                    url=getattr(wh, "url", "unknown"),
                    payload=payload,
                    status=WebhookStatus.FAILED,
                    error_message=str(e),
                    last_attempt=datetime.now(timezone.utc),
                )

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            futures = [
                executor.submit(_send, name, wh) for name, wh in self.webhooks.items()
            ]
            for fut in as_completed(futures):
                name, delivery = fut.result()
                results[name] = delivery

        return results

    def send_to(
        self, webhook_name: str, event_type: str, data: Dict[str, Any], **metadata
    ) -> Optional[WebhookDelivery]:
        """Send to specific webhook"""
        if webhook_name not in self.webhooks:
            logging.error(f"Webhook '{webhook_name}' not found")
            return None

        payload = WebhookPayload(event_type=event_type, data=data, metadata=metadata)

        return self.webhooks[webhook_name].dispatch(payload)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print("Webhook integration initialized")

    # Example (replace with actual URLs)
    # slack_webhook = SlackWebhookDispatcher(
    #     "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    #     secret="optional-shared-secret",
    #     rate_limit_per_sec=1.0,
    #     max_retries=5,
    #     retry_delay=1.0,
    #     backoff_factor=2.0,
    # )
    # slack_webhook.send_alert("Test Alert", "This is a test alert", "info", fields={"env": "prod"}, use_blocks=True)

    print("Webhook integration demo complete (stub mode)")
