"""Synchronous client for Nethical Python SDK.

Provides a sync interface for the Nethical Governance API.

Usage:
    from nethical_sdk import NethicalClient

    client = NethicalClient(api_url="https://api.nethical.example.com", api_key="your-key")
    result = client.evaluate(agent_id="my-agent", action="Some action")
    print(result.decision)
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Optional
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


class NethicalClient:
    """Synchronous client for the Nethical Governance API.

    This client provides a simple interface for evaluating
    actions and managing governance policies.

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
        self._session_id: Optional[str] = None

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

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the API."""
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

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data) if response_data else {}
        except urllib.error.HTTPError as e:
            return self._handle_http_error(e)
        except urllib.error.URLError as e:
            raise NethicalError(f"Connection error: {str(e)}")

    def _handle_http_error(self, error: urllib.error.HTTPError) -> dict:
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

    def evaluate(
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

        This is the primary method for governance checks.

        Args:
            action: The action content to evaluate
            agent_id: Agent identifier
            action_type: Type of action (code_generation, query, command, etc.)
            context: Additional context
            stated_intent: Declared intent for semantic monitoring
            priority: Request priority (low, normal, high, critical)
            require_explanation: Include detailed explanation

        Returns:
            EvaluateResponse with decision and metadata

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit exceeded
            ValidationError: If request validation fails
            ServerError: If server error occurs
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

        response = self._request("POST", "/v2/evaluate", data)
        return EvaluateResponse.from_dict(response)

    def batch_evaluate(
        self,
        requests: list[EvaluateRequest],
        parallel: bool = True,
        fail_fast: bool = False,
    ) -> list[EvaluateResponse]:
        """Evaluate multiple actions in a batch.

        Args:
            requests: List of evaluation requests
            parallel: Process requests in parallel
            fail_fast: Stop on first error

        Returns:
            List of EvaluateResponse objects
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

        response = self._request("POST", "/v2/batch-evaluate", data)
        return [EvaluateResponse.from_dict(r) for r in response.get("results", [])]

    def get_decision(self, decision_id: str) -> Decision:
        """Retrieve a specific decision by ID.

        Args:
            decision_id: Decision identifier

        Returns:
            Decision record
        """
        response = self._request("GET", f"/v2/decisions/{decision_id}")
        return Decision.from_dict(response)

    def list_decisions(
        self,
        agent_id: Optional[str] = None,
        decision: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Decision], int, bool]:
        """List decisions with optional filtering.

        Args:
            agent_id: Filter by agent
            decision: Filter by decision type
            page: Page number
            page_size: Items per page

        Returns:
            (decisions, total_count, has_next)
        """
        params = f"?page={page}&page_size={page_size}"
        if agent_id:
            params += f"&agent_id={agent_id}"
        if decision:
            params += f"&decision={decision}"

        response = self._request("GET", f"/v2/decisions{params}")
        decisions = [Decision.from_dict(d) for d in response.get("decisions", [])]
        return (
            decisions,
            response.get("total_count", 0),
            response.get("has_next", False),
        )

    def list_policies(
        self,
        status: Optional[str] = None,
        scope: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Policy], int, bool]:
        """List governance policies.

        Args:
            status: Filter by status
            scope: Filter by scope
            page: Page number
            page_size: Items per page

        Returns:
            (policies, total_count, has_next)
        """
        params = f"?page={page}&page_size={page_size}"
        if status:
            params += f"&status={status}"
        if scope:
            params += f"&scope={scope}"

        response = self._request("GET", f"/v2/policies{params}")
        policies = [Policy.from_dict(p) for p in response.get("policies", [])]
        return policies, response.get("total_count", 0), response.get("has_next", False)

    def get_fairness_report(self, period_days: int = 7) -> FairnessReport:
        """Get the current fairness report.

        Args:
            period_days: Analysis period in days

        Returns:
            FairnessReport with metrics and recommendations
        """
        response = self._request("GET", f"/v2/fairness?period_days={period_days}")
        return FairnessReport.from_dict(response)

    def submit_appeal(
        self,
        decision_id: str,
        appellant_id: str,
        reason: str,
        evidence: Optional[dict[str, Any]] = None,
        requested_outcome: str = "reconsider",
        priority: str = "normal",
    ) -> Appeal:
        """Submit an appeal for a decision.

        Implements Law 7 (Override Rights).

        Args:
            decision_id: ID of the decision to appeal
            appellant_id: ID of the appellant
            reason: Reason for the appeal
            evidence: Supporting evidence
            requested_outcome: Desired outcome
            priority: Appeal priority

        Returns:
            Appeal record
        """
        data = {
            "decision_id": decision_id,
            "appellant_id": appellant_id,
            "reason": reason,
            "requested_outcome": requested_outcome,
            "priority": priority,
        }
        if evidence:
            data["evidence"] = evidence

        response = self._request("POST", "/v2/appeals", data)
        return Appeal.from_dict(response)

    def get_appeal(self, appeal_id: str) -> Appeal:
        """Get the status of an appeal.

        Args:
            appeal_id: Appeal identifier

        Returns:
            Appeal record
        """
        response = self._request("GET", f"/v2/appeals/{appeal_id}")
        return Appeal.from_dict(response)

    def get_audit_record(self, audit_id: str) -> AuditRecord:
        """Retrieve an audit record.

        Implements Law 15 (Audit Compliance).

        Args:
            audit_id: Audit record identifier

        Returns:
            AuditRecord
        """
        response = self._request("GET", f"/v2/audit/{audit_id}")
        return AuditRecord.from_dict(response)

    def health_check(self) -> dict[str, Any]:
        """Check API health.

        Returns:
            Health status dictionary
        """
        return self._request("GET", "/v2/health")

    def async_session(self) -> "AsyncNethicalClient":
        """Get an async session for this client.

        Returns:
            AsyncNethicalClient configured with same settings
        """
        from .async_client import AsyncNethicalClient

        return AsyncNethicalClient(
            api_url=self.api_url,
            api_key=self.api_key,
            timeout=self.timeout,
            region=self.region,
        )
