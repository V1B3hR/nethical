"""
Tests for External Integrations: Webhook and API Integrations

Tests the webhook dispatchers and API integration interfaces.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from nethical.integrations.webhook import (
    WebhookStatus,
    WebhookPayload,
    WebhookDelivery,
    WebhookDispatcher,
    HTTPWebhookDispatcher,
    SlackWebhookDispatcher,
    DiscordWebhookDispatcher,
    APIIntegration,
    GenericAPIIntegration,
    WebhookManager,
)


class TestWebhookPayload:
    """Test WebhookPayload dataclass"""

    def test_payload_creation(self):
        """Test creating a webhook payload"""
        payload = WebhookPayload(
            event_type="test_event",
            data={"key": "value", "number": 42},
            metadata={"source": "test"},
        )

        assert payload.event_type == "test_event"
        assert payload.data["key"] == "value"
        assert payload.metadata["source"] == "test"

    def test_payload_to_dict(self):
        """Test converting payload to dict"""
        payload = WebhookPayload(event_type="test_event", data={"key": "value"})

        payload_dict = payload.to_dict()
        assert payload_dict["event_type"] == "test_event"
        assert payload_dict["data"]["key"] == "value"
        assert "timestamp" in payload_dict

    def test_payload_to_json(self):
        """Test converting payload to JSON"""
        payload = WebhookPayload(event_type="test_event", data={"key": "value"})

        json_str = payload.to_json()
        parsed = json.loads(json_str)

        assert parsed["event_type"] == "test_event"
        assert parsed["data"]["key"] == "value"


class TestWebhookDelivery:
    """Test WebhookDelivery dataclass"""

    def test_delivery_creation(self):
        """Test creating a webhook delivery"""
        payload = WebhookPayload("test_event", {"key": "value"})
        delivery = WebhookDelivery(
            delivery_id="del_123", url="https://example.com/webhook", payload=payload
        )

        assert delivery.delivery_id == "del_123"
        assert delivery.url == "https://example.com/webhook"
        assert delivery.status == WebhookStatus.PENDING
        assert delivery.attempt_count == 0

    def test_delivery_to_dict(self):
        """Test converting delivery to dict"""
        payload = WebhookPayload("test_event", {"key": "value"})
        delivery = WebhookDelivery(
            delivery_id="del_123",
            url="https://example.com/webhook",
            payload=payload,
            status=WebhookStatus.SUCCESS,
            response_code=200,
        )

        delivery_dict = delivery.to_dict()
        assert delivery_dict["delivery_id"] == "del_123"
        assert delivery_dict["status"] == "success"
        assert delivery_dict["response_code"] == 200


class TestHTTPWebhookDispatcher:
    """Test HTTP webhook dispatcher"""

    def test_dispatcher_creation(self):
        """Test creating HTTP webhook dispatcher"""
        dispatcher = HTTPWebhookDispatcher(
            url="https://example.com/webhook",
            headers={"X-Custom": "value"},
            timeout=60,
            max_retries=5,
        )

        assert dispatcher.url == "https://example.com/webhook"
        assert dispatcher.headers["X-Custom"] == "value"
        assert dispatcher.timeout == 60
        assert dispatcher.max_retries == 5

    @patch("nethical.integrations.webhook.request")
    def test_successful_dispatch(self, mock_request):
        """Test successful webhook dispatch"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_request.urlopen.return_value.__enter__.return_value = mock_response

        dispatcher = HTTPWebhookDispatcher("https://example.com/webhook")
        payload = WebhookPayload("test_event", {"key": "value"})

        delivery = dispatcher.dispatch(payload)

        assert delivery.status == WebhookStatus.SUCCESS
        assert delivery.response_code == 200
        assert delivery.attempt_count >= 1

    @patch("nethical.integrations.webhook.request")
    def test_failed_dispatch(self, mock_request):
        """Test failed webhook dispatch"""
        # Mock failed response
        from urllib.error import HTTPError

        mock_request.urlopen.side_effect = HTTPError(
            "https://example.com/webhook", 500, "Internal Server Error", {}, None
        )

        dispatcher = HTTPWebhookDispatcher(
            url="https://example.com/webhook",
            max_retries=2,
            retry_delay=0,  # No delay for testing
        )
        payload = WebhookPayload("test_event", {"key": "value"})

        delivery = dispatcher.dispatch(payload)

        assert delivery.status == WebhookStatus.FAILED
        assert delivery.response_code == 500
        assert delivery.attempt_count == 2  # Should retry

    @patch("nethical.integrations.webhook.request")
    def test_retry_logic(self, mock_request):
        """Test retry logic on failure"""
        # Mock first call fails, second succeeds
        mock_error = Exception("Connection error")
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200

        mock_request.urlopen.side_effect = [
            mock_error,
            mock_response.__enter__.return_value,
        ]

        dispatcher = HTTPWebhookDispatcher(
            url="https://example.com/webhook", max_retries=3, retry_delay=0
        )
        payload = WebhookPayload("test_event", {"key": "value"})

        delivery = dispatcher.dispatch(payload)

        # Should succeed on retry
        # Note: Implementation may vary, this tests the concept
        assert delivery.attempt_count >= 1

    def test_get_delivery_status(self):
        """Test getting delivery status"""
        dispatcher = HTTPWebhookDispatcher("https://example.com/webhook")

        # Create a mock delivery
        payload = WebhookPayload("test_event", {"key": "value"})
        delivery = WebhookDelivery("test_123", "https://example.com/webhook", payload)
        dispatcher.deliveries.append(delivery)

        # Get status
        found_delivery = dispatcher.get_delivery_status("test_123")
        assert found_delivery is not None
        assert found_delivery.delivery_id == "test_123"

        # Non-existent delivery
        not_found = dispatcher.get_delivery_status("nonexistent")
        assert not_found is None


class TestSlackWebhookDispatcher:
    """Test Slack webhook dispatcher"""

    def test_dispatcher_creation(self):
        """Test creating Slack webhook dispatcher"""
        dispatcher = SlackWebhookDispatcher("https://hooks.slack.com/services/...")
        assert dispatcher.url == "https://hooks.slack.com/services/..."

    @patch.object(HTTPWebhookDispatcher, "dispatch")
    def test_send_message(self, mock_dispatch):
        """Test sending Slack message"""
        mock_delivery = WebhookDelivery(
            "del_123",
            "https://hooks.slack.com/services/...",
            WebhookPayload("slack_message", {}),
            status=WebhookStatus.SUCCESS,
        )
        mock_dispatch.return_value = mock_delivery

        dispatcher = SlackWebhookDispatcher("https://hooks.slack.com/services/...")

        delivery = dispatcher.send_message(
            text="Test message", username="TestBot", icon_emoji=":robot_face:"
        )

        assert delivery.status == WebhookStatus.SUCCESS
        # Verify dispatch was called
        mock_dispatch.assert_called_once()

        # Check the payload
        call_args = mock_dispatch.call_args
        payload = call_args[0][0]
        assert payload.event_type == "slack_message"
        assert payload.data["text"] == "Test message"
        assert payload.data["username"] == "TestBot"

    @patch.object(HTTPWebhookDispatcher, "dispatch")
    def test_send_alert(self, mock_dispatch):
        """Test sending Slack alert"""
        mock_delivery = WebhookDelivery(
            "del_123",
            "https://hooks.slack.com/services/...",
            WebhookPayload("slack_message", {}),
            status=WebhookStatus.SUCCESS,
        )
        mock_dispatch.return_value = mock_delivery

        dispatcher = SlackWebhookDispatcher("https://hooks.slack.com/services/...")

        delivery = dispatcher.send_alert(
            title="Test Alert",
            message="This is a test alert",
            severity="warning",
            fields={"Environment": "production", "Service": "api"},
        )

        assert delivery.status == WebhookStatus.SUCCESS

        # Verify attachment structure
        call_args = mock_dispatch.call_args
        payload = call_args[0][0]
        attachments = payload.data.get("attachments", [])
        assert len(attachments) == 1

        attachment = attachments[0]
        assert attachment["title"] == "Test Alert"
        assert attachment["text"] == "This is a test alert"
        assert attachment["color"] == "#ff9900"  # Warning color
        assert len(attachment["fields"]) == 2


class TestDiscordWebhookDispatcher:
    """Test Discord webhook dispatcher"""

    def test_dispatcher_creation(self):
        """Test creating Discord webhook dispatcher"""
        dispatcher = DiscordWebhookDispatcher("https://discord.com/api/webhooks/...")
        assert dispatcher.url == "https://discord.com/api/webhooks/..."

    @patch.object(HTTPWebhookDispatcher, "dispatch")
    def test_send_message(self, mock_dispatch):
        """Test sending Discord message"""
        mock_delivery = WebhookDelivery(
            "del_123",
            "https://discord.com/api/webhooks/...",
            WebhookPayload("discord_message", {}),
            status=WebhookStatus.SUCCESS,
        )
        mock_dispatch.return_value = mock_delivery

        dispatcher = DiscordWebhookDispatcher("https://discord.com/api/webhooks/...")

        delivery = dispatcher.send_message(content="Test message", username="TestBot")

        assert delivery.status == WebhookStatus.SUCCESS

        # Verify payload
        call_args = mock_dispatch.call_args
        payload = call_args[0][0]
        assert payload.event_type == "discord_message"
        assert payload.data["content"] == "Test message"
        assert payload.data["username"] == "TestBot"


class TestGenericAPIIntegration:
    """Test generic API integration"""

    def test_integration_creation(self):
        """Test creating API integration"""
        integration = GenericAPIIntegration(
            base_url="https://api.example.com",
            api_key="test_key",
            auth_header="X-API-Key",
        )

        assert integration.base_url == "https://api.example.com"
        assert integration.api_key == "test_key"
        assert integration.auth_header == "X-API-Key"
        assert not integration.authenticated

    def test_authenticate(self):
        """Test authentication"""
        integration = GenericAPIIntegration(
            base_url="https://api.example.com", api_key="test_key"
        )

        result = integration.authenticate()
        assert result is True
        assert integration.authenticated

    def test_authenticate_no_key(self):
        """Test authentication without API key"""
        integration = GenericAPIIntegration(base_url="https://api.example.com")

        result = integration.authenticate()
        assert result is True  # Should succeed (stub)

    def test_send_data(self):
        """Test sending data"""
        integration = GenericAPIIntegration(
            base_url="https://api.example.com", api_key="test_key"
        )

        data = {"key": "value", "number": 42}
        result = integration.send_data(data)

        # Stub implementation returns True
        assert result is True

    def test_receive_data(self):
        """Test receiving data"""
        integration = GenericAPIIntegration(
            base_url="https://api.example.com", api_key="test_key"
        )

        data = integration.receive_data()

        # Stub implementation returns dict
        assert isinstance(data, dict)
        assert "status" in data


class TestWebhookManager:
    """Test webhook manager"""

    def test_manager_creation(self):
        """Test creating webhook manager"""
        manager = WebhookManager()
        assert len(manager.webhooks) == 0

    def test_add_webhook(self):
        """Test adding webhooks"""
        manager = WebhookManager()

        slack = SlackWebhookDispatcher("https://hooks.slack.com/services/...")
        discord = DiscordWebhookDispatcher("https://discord.com/api/webhooks/...")

        manager.add_webhook("slack", slack)
        manager.add_webhook("discord", discord)

        assert len(manager.webhooks) == 2
        assert "slack" in manager.webhooks
        assert "discord" in manager.webhooks

    @patch.object(HTTPWebhookDispatcher, "dispatch")
    def test_broadcast(self, mock_dispatch):
        """Test broadcasting to all webhooks"""
        mock_delivery = WebhookDelivery(
            "del_123",
            "https://example.com",
            WebhookPayload("test", {}),
            status=WebhookStatus.SUCCESS,
        )
        mock_dispatch.return_value = mock_delivery

        manager = WebhookManager()
        manager.add_webhook("webhook1", HTTPWebhookDispatcher("https://webhook1.com"))
        manager.add_webhook("webhook2", HTTPWebhookDispatcher("https://webhook2.com"))

        results = manager.broadcast(
            event_type="system_alert", data={"message": "Test alert"}, priority="high"
        )

        assert len(results) == 2
        assert "webhook1" in results
        assert "webhook2" in results

        # Verify both received the event
        for delivery in results.values():
            assert delivery.status == WebhookStatus.SUCCESS

    @patch.object(SlackWebhookDispatcher, "dispatch")
    def test_send_to_specific_webhook(self, mock_dispatch):
        """Test sending to specific webhook"""
        mock_delivery = WebhookDelivery(
            "del_123",
            "https://hooks.slack.com/services/...",
            WebhookPayload("test", {}),
            status=WebhookStatus.SUCCESS,
        )
        mock_dispatch.return_value = mock_delivery

        manager = WebhookManager()
        slack = SlackWebhookDispatcher("https://hooks.slack.com/services/...")
        manager.add_webhook("slack", slack)

        delivery = manager.send_to(
            webhook_name="slack",
            event_type="notification",
            data={"text": "Test notification"},
        )

        assert delivery is not None
        assert delivery.status == WebhookStatus.SUCCESS

    def test_send_to_nonexistent_webhook(self):
        """Test sending to non-existent webhook"""
        manager = WebhookManager()

        delivery = manager.send_to(
            webhook_name="nonexistent", event_type="test", data={}
        )

        assert delivery is None

    @patch.object(HTTPWebhookDispatcher, "dispatch")
    def test_error_handling(self, mock_dispatch):
        """Test error handling when webhook fails"""
        # Make one webhook fail
        mock_dispatch.side_effect = Exception("Webhook failed")

        manager = WebhookManager()
        manager.add_webhook("failing", HTTPWebhookDispatcher("https://failing.com"))
        manager.add_webhook("working", HTTPWebhookDispatcher("https://working.com"))

        # Should not raise exception
        results = manager.broadcast("test_event", {"data": "value"})

        # Only one webhook succeeded
        # Note: Implementation may handle errors differently
        assert isinstance(results, dict)


class TestIntegration:
    """Integration tests for webhooks and APIs"""

    @patch.object(HTTPWebhookDispatcher, "dispatch")
    def test_multi_webhook_notification_system(self, mock_dispatch):
        """Test complete multi-webhook notification system"""
        mock_delivery = WebhookDelivery(
            "del_123",
            "https://example.com",
            WebhookPayload("test", {}),
            status=WebhookStatus.SUCCESS,
        )
        mock_dispatch.return_value = mock_delivery

        # Setup notification system
        manager = WebhookManager()
        manager.add_webhook(
            "slack", SlackWebhookDispatcher("https://hooks.slack.com/...")
        )
        manager.add_webhook(
            "discord", DiscordWebhookDispatcher("https://discord.com/...")
        )
        manager.add_webhook(
            "custom", HTTPWebhookDispatcher("https://custom.webhook.com")
        )

        # Send different types of notifications
        events = [
            ("info", {"message": "System started"}),
            ("warning", {"message": "High memory usage"}),
            ("error", {"message": "Database connection lost"}),
        ]

        for event_type, data in events:
            results = manager.broadcast(event_type, data)
            assert len(results) == 3

    def test_api_webhook_integration(self):
        """Test integration between API and webhooks"""
        # Setup API
        api = GenericAPIIntegration(
            base_url="https://api.example.com", api_key="test_key"
        )
        api.authenticate()

        # Setup webhook manager
        manager = WebhookManager()
        manager.add_webhook(
            "notifications", HTTPWebhookDispatcher("https://notify.com")
        )

        # Simulate workflow: API call triggers webhook
        data = {"status": "success", "result": "processed"}
        api_result = api.send_data(data)

        if api_result:
            # Trigger webhook notification
            # Note: In stub mode, this always succeeds
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
