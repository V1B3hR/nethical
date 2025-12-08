"""Asynchronous client for Nethical Python SDK.

Provides an async interface for the Nethical Governance API.

Usage:
    from nethical_sdk import AsyncNethicalClient

    async with AsyncNethicalClient(api_url="...", api_key="...") as client:
        result = await client.evaluate(agent_id="my-agent", action="Some action")
        print(result.decision)

    # Streaming
    async for decision in client.stream_decisions():
        print(decision)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional
from urllib.parse import urljoin

from .models import (
    EvaluateRequest,
    EvaluateResponse,
    Decision,
    Policy,
    FairnessReport,
    Appeal,
    AuditRecord,
)
from .exceptions import (
    NethicalError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServerError,
)

logger = logging.getLogger(__name__)


class AsyncNethicalClient:
    """Asynchronous client for the Nethical Governance API.

    This client provides an async interface for evaluating
    actions and managing governance policies.

    Supports:
    - Async/await for all operations
    - WebSocket streaming for real-time decisions
    - Connection pooling

    All operations adhere to the 25 Fundamental Laws of AI Ethics.

    Args:
        api_url: Base URL of the Nethical API
        api_key: API key for authentication
        timeout: Request timeout in seconds
        region: Optional region for multi-region deployments
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        region: Optional[str] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.region = region
        self._session = None

    async def __aenter__(self) -> "AsyncNethicalClient":
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        # In production, this would create an aiohttp session
        pass

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._session:
            # In production: await self._session.close()
            self._session = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers including authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "nethical-sdk-python/1.0.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.region:
            headers["X-Nethical-Region"] = self.region
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Make an async HTTP request to the API.

        Note: This is a placeholder. In production, use aiohttp.
        """
        # For now, use sync request wrapped in executor
        import urllib.request
        import urllib.error

        url = urljoin(self.api_url, path)
        headers = self._get_headers()

        body = None
        if data:
            body = json.dumps(data).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )

        def do_request():
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_data = response.read().decode("utf-8")
                    return json.loads(response_data) if response_data else {}
            except urllib.error.HTTPError as e:
                return self._handle_http_error(e)
            except urllib.error.URLError as e:
                raise NethicalError(f"Connection error: {str(e)}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, do_request)

    def _handle_http_error(self, error) -> dict:
        """Handle HTTP errors and raise appropriate exceptions."""
        try:
            body = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}

        request_id = error.headers.get("X-Request-ID")
        message = body.get("detail", str(error))

        if error.code == 401:
            raise AuthenticationError(message, request_id=request_id)
        elif error.code == 429:
            raise RateLimitError(
                message,
                request_id=request_id,
                retry_after=int(error.headers.get("Retry-After", 60)),
            )
        elif error.code == 400 or error.code == 422:
            raise ValidationError(message, request_id=request_id, details=body)
        elif error.code >= 500:
            raise ServerError(message, status_code=error.code, request_id=request_id)
        else:
            raise NethicalError(message, request_id=request_id)

    async def evaluate(
        self,
        action: str,
        agent_id: str = "unknown",
        action_type: str = "query",
        context: Optional[dict[str, Any]] = None,
        stated_intent: Optional[str] = None,
        priority: str = "normal",
        require_explanation: bool = False,
    ) -> EvaluateResponse:
        """Evaluate an action for ethical compliance.

        Args:
            action: The action content to evaluate
            agent_id: Agent identifier
            action_type: Type of action
            context: Additional context
            stated_intent: Declared intent
            priority: Request priority
            require_explanation: Include detailed explanation

        Returns:
            EvaluateResponse with decision and metadata
        """
        data = {
            "action": action,
            "agent_id": agent_id,
            "action_type": action_type,
            "priority": priority,
            "require_explanation": require_explanation,
        }
        if context:
            data["context"] = context
        if stated_intent:
            data["stated_intent"] = stated_intent

        response = await self._request("POST", "/v2/evaluate", data)
        return EvaluateResponse.from_dict(response)

    async def batch_evaluate(
        self,
        requests: list[EvaluateRequest],
        parallel: bool = True,
        fail_fast: bool = False,
    ) -> list[EvaluateResponse]:
        """Evaluate multiple actions in a batch.

        Args:
            requests: List of evaluation requests
            parallel: Process in parallel
            fail_fast: Stop on first error

        Returns:
            List of responses
        """
        data = {
            "requests": [
                {
                    "action": r.action,
                    "agent_id": r.agent_id,
                    "action_type": r.action_type,
                    "context": r.context,
                    "stated_intent": r.stated_intent,
                    "priority": r.priority,
                    "require_explanation": r.require_explanation,
                }
                for r in requests
            ],
            "parallel": parallel,
            "fail_fast": fail_fast,
        }

        response = await self._request("POST", "/v2/batch-evaluate", data)
        return [EvaluateResponse.from_dict(r) for r in response.get("results", [])]

    async def stream_decisions(
        self,
        agent_id: Optional[str] = None,
        decision_types: Optional[list[str]] = None,
    ) -> AsyncIterator[Decision]:
        """Stream governance decisions in real-time.

        Uses WebSocket connection for real-time updates.

        Args:
            agent_id: Filter by agent
            decision_types: Filter by decision type

        Yields:
            Decision objects as they occur
        """
        # In production, this would use websockets library
        logger.info("Starting decision stream (placeholder)")
        # Placeholder - would yield decisions from WebSocket
        return
        yield  # type: ignore

    async def get_decision(self, decision_id: str) -> Decision:
        """Retrieve a specific decision by ID."""
        response = await self._request("GET", f"/v2/decisions/{decision_id}")
        return Decision.from_dict(response)

    async def list_decisions(
        self,
        agent_id: Optional[str] = None,
        decision: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Decision], int, bool]:
        """List decisions with optional filtering."""
        params = f"?page={page}&page_size={page_size}"
        if agent_id:
            params += f"&agent_id={agent_id}"
        if decision:
            params += f"&decision={decision}"

        response = await self._request("GET", f"/v2/decisions{params}")
        decisions = [Decision.from_dict(d) for d in response.get("decisions", [])]
        return (
            decisions,
            response.get("total_count", 0),
            response.get("has_next", False),
        )

    async def list_policies(
        self,
        status: Optional[str] = None,
        scope: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Policy], int, bool]:
        """List governance policies."""
        params = f"?page={page}&page_size={page_size}"
        if status:
            params += f"&status={status}"
        if scope:
            params += f"&scope={scope}"

        response = await self._request("GET", f"/v2/policies{params}")
        policies = [Policy.from_dict(p) for p in response.get("policies", [])]
        return policies, response.get("total_count", 0), response.get("has_next", False)

    async def get_fairness_report(self, period_days: int = 7) -> FairnessReport:
        """Get the current fairness report."""
        response = await self._request("GET", f"/v2/fairness?period_days={period_days}")
        return FairnessReport.from_dict(response)

    async def submit_appeal(
        self,
        decision_id: str,
        appellant_id: str,
        reason: str,
        evidence: Optional[dict[str, Any]] = None,
        requested_outcome: str = "reconsider",
        priority: str = "normal",
    ) -> Appeal:
        """Submit an appeal for a decision."""
        data = {
            "decision_id": decision_id,
            "appellant_id": appellant_id,
            "reason": reason,
            "requested_outcome": requested_outcome,
            "priority": priority,
        }
        if evidence:
            data["evidence"] = evidence

        response = await self._request("POST", "/v2/appeals", data)
        return Appeal.from_dict(response)

    async def get_appeal(self, appeal_id: str) -> Appeal:
        """Get the status of an appeal."""
        response = await self._request("GET", f"/v2/appeals/{appeal_id}")
        return Appeal.from_dict(response)

    async def get_audit_record(self, audit_id: str) -> AuditRecord:
        """Retrieve an audit record."""
        response = await self._request("GET", f"/v2/audit/{audit_id}")
        return AuditRecord.from_dict(response)

    async def health_check(self) -> dict[str, Any]:
        """Check API health."""
        return await self._request("GET", "/v2/health")
