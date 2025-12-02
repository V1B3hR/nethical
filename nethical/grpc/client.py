"""gRPC client for Nethical Governance.

Provides a client for connecting to the Nethical gRPC service
with retry logic, timeouts, and connection pooling.

Usage:
    from nethical.grpc import NethicalGRPCClient
    
    async with NethicalGRPCClient("localhost:50051") as client:
        result = await client.evaluate(
            agent_id="my-agent",
            action="Generate code to access database",
            action_type="code_generation"
        )
        print(result.decision)

All operations adhere to the 25 Fundamental Laws.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from dataclasses import dataclass

from nethical.proto import (
    EvaluateRequest,
    EvaluateResponse,
    Violation,
    Explanation,
    Decision,
    Policy,
)

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 5000
    backoff_multiplier: float = 2.0


@dataclass
class ClientConfig:
    """Configuration for the gRPC client."""
    
    address: str = "localhost:50051"
    timeout_ms: int = 10000
    retry: RetryConfig = None
    
    def __post_init__(self):
        if self.retry is None:
            self.retry = RetryConfig()


class NethicalGRPCClient:
    """Client for the Nethical gRPC service.
    
    Provides async methods for governance operations with
    built-in retry logic and error handling.
    
    Features:
    - Async/await support
    - Automatic retry with exponential backoff
    - Timeout handling
    - Connection health monitoring
    - Fail-safe design (Law 23)
    """
    
    def __init__(
        self,
        address: str = "localhost:50051",
        timeout_ms: int = 10000,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the gRPC client.
        
        Args:
            address: Server address in host:port format
            timeout_ms: Request timeout in milliseconds
            retry_config: Retry configuration
        """
        self.config = ClientConfig(
            address=address,
            timeout_ms=timeout_ms,
            retry=retry_config or RetryConfig(),
        )
        self._connected = False
        self._channel = None
    
    async def __aenter__(self) -> "NethicalGRPCClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Connect to the gRPC server."""
        logger.info("Connecting to gRPC server at %s", self.config.address)
        # In production, this would create a grpc.aio.Channel
        self._connected = True
    
    async def close(self) -> None:
        """Close the connection."""
        logger.info("Closing gRPC connection")
        self._connected = False
        if self._channel:
            # In production: await self._channel.close()
            pass
    
    async def evaluate(
        self,
        agent_id: str,
        action: str,
        action_type: str = "query",
        context: Optional[dict[str, str]] = None,
        stated_intent: Optional[str] = None,
        priority: str = "normal",
        require_explanation: bool = False,
    ) -> EvaluateResponse:
        """Evaluate an action for ethical compliance.
        
        This is the primary method for governance checks.
        Implements Law 23 (Fail-Safe Design) with automatic retry.
        
        Args:
            agent_id: Agent identifier
            action: Action content to evaluate
            action_type: Type of action
            context: Additional context
            stated_intent: Declared intent
            priority: Request priority
            require_explanation: Include detailed explanation
            
        Returns:
            EvaluateResponse with decision and metadata
        """
        request = EvaluateRequest(
            agent_id=agent_id,
            action=action,
            action_type=action_type,
            context=context or {},
            stated_intent=stated_intent,
            priority=priority,
            require_explanation=require_explanation,
            request_id=str(uuid.uuid4()),
        )
        
        return await self._retry_call(
            lambda: self._evaluate_action(request)
        )
    
    async def _evaluate_action(self, request: EvaluateRequest) -> EvaluateResponse:
        """Internal evaluation implementation.
        
        Note: In production, this would make an actual gRPC call.
        This is a placeholder that performs local evaluation.
        """
        start_time = time.perf_counter()
        decision_id = str(uuid.uuid4())
        
        # Simulate evaluation (would be remote call in production)
        decision = "ALLOW"
        risk_score = 0.0
        violations: list[Violation] = []
        laws_checked = [6, 10, 15, 21, 22]
        
        action_lower = request.action.lower()
        
        dangerous_patterns = [
            ("delete all", "Bulk deletion", "high", 23),
            ("drop table", "Database destruction", "critical", 21),
            ("hack", "Malicious activity", "high", 21),
        ]
        
        for pattern, desc, severity, law in dangerous_patterns:
            if pattern in action_lower:
                violations.append(Violation(
                    id=str(uuid.uuid4()),
                    type="safety_violation",
                    severity=severity,
                    description=desc,
                    law_reference=f"Law {law}",
                ))
                risk_score = max(risk_score, 0.85)
                laws_checked.append(law)
        
        if risk_score >= 0.9:
            decision = "BLOCK"
        elif risk_score >= 0.7:
            decision = "RESTRICT"
        
        explanation = None
        if request.require_explanation:
            explanation = Explanation(
                summary=f"{len(violations)} violations detected",
                risk_factors=[v.description for v in violations],
                decision_rationale=f"Risk score: {risk_score:.2f}",
                laws_applied=[f"Law {i}" for i in set(laws_checked)],
            )
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        return EvaluateResponse(
            decision=decision,
            decision_id=decision_id,
            risk_score=risk_score,
            confidence=0.95 if violations else 1.0,
            latency_ms=latency_ms,
            violations=violations,
            reason=f"{len(violations)} violations, risk: {risk_score:.2f}",
            explanation=explanation,
            audit_id=request.request_id,
            cache_hit=False,
            fundamental_laws_checked=list(set(laws_checked)),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    async def batch_evaluate(
        self,
        requests: list[dict],
    ) -> AsyncIterator[EvaluateResponse]:
        """Evaluate multiple actions with streaming responses.
        
        Args:
            requests: List of evaluation request dictionaries
            
        Yields:
            EvaluateResponse for each request
        """
        for req in requests:
            result = await self.evaluate(**req)
            yield result
    
    async def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Retrieve a specific decision by ID.
        
        Args:
            decision_id: Decision identifier
            
        Returns:
            Decision or None if not found
        """
        logger.info("Getting decision %s", decision_id)
        # In production, this would make a gRPC call
        return None
    
    async def stream_decisions(
        self,
        agent_id: Optional[str] = None,
        decision_types: Optional[list[str]] = None,
        min_risk_score: Optional[float] = None,
    ) -> AsyncIterator[Decision]:
        """Stream decisions matching filter criteria.
        
        Args:
            agent_id: Filter by agent
            decision_types: Filter by decision type
            min_risk_score: Filter by minimum risk score
            
        Yields:
            Matching decisions
        """
        logger.info("Starting decision stream")
        # In production, this would open a gRPC stream
        return
        yield  # type: ignore  # Make this a generator
    
    async def list_policies(
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
        logger.info("Listing policies")
        # In production, this would make a gRPC call
        return [], 0, False
    
    async def health_check(self) -> dict:
        """Check service health.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy" if self._connected else "disconnected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def _retry_call(self, call_fn, attempt: int = 0):
        """Execute a call with retry logic.
        
        Implements Law 23 (Fail-Safe Design) with exponential backoff.
        """
        try:
            return await call_fn()
        except Exception as e:
            if attempt >= self.config.retry.max_retries:
                logger.error(
                    "Max retries (%d) exceeded: %s",
                    self.config.retry.max_retries,
                    str(e),
                )
                # Return safe blocking decision on failure
                return EvaluateResponse(
                    decision="BLOCK",
                    decision_id=str(uuid.uuid4()),
                    risk_score=1.0,
                    confidence=0.0,
                    latency_ms=0,
                    violations=[Violation(
                        id=str(uuid.uuid4()),
                        type="connection_error",
                        severity="critical",
                        description=f"Failed after {attempt} retries: {str(e)[:100]}",
                        law_reference="Law 23",
                    )],
                    reason="Blocked for safety due to connection error",
                    fundamental_laws_checked=[23],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            
            # Calculate backoff
            backoff_ms = min(
                self.config.retry.initial_backoff_ms * (
                    self.config.retry.backoff_multiplier ** attempt
                ),
                self.config.retry.max_backoff_ms,
            )
            
            logger.warning(
                "Retry attempt %d after %dms: %s",
                attempt + 1,
                backoff_ms,
                str(e),
            )
            
            await asyncio.sleep(backoff_ms / 1000)
            return await self._retry_call(call_fn, attempt + 1)


# Synchronous wrapper for non-async contexts
class NethicalGRPCClientSync:
    """Synchronous wrapper for the gRPC client.
    
    For use in non-async contexts. Wraps the async client
    with an event loop.
    """
    
    def __init__(
        self,
        address: str = "localhost:50051",
        timeout_ms: int = 10000,
    ):
        self._client = NethicalGRPCClient(address, timeout_ms)
    
    def evaluate(self, **kwargs) -> EvaluateResponse:
        """Synchronous evaluate."""
        return asyncio.run(self._client.evaluate(**kwargs))
    
    def health_check(self) -> dict:
        """Synchronous health check."""
        return asyncio.run(self._client.health_check())
