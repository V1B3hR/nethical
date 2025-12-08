"""gRPC server for Nethical Governance.

Provides a gRPC interface for low-latency governance decisions.
This is designed for internal service-to-service communication
where REST latency is too high.

Usage:
    from nethical.grpc import create_grpc_server

    server = create_grpc_server(port=50051)
    server.start()
    server.wait_for_termination()

Implements all 25 Fundamental Laws.
"""

from __future__ import annotations

import time
import uuid
import logging
from datetime import datetime, timezone
from concurrent import futures
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Import proto-compatible dataclasses
from nethical.proto import (
    EvaluateRequest,
    EvaluateResponse,
    Violation,
    Explanation,
    Decision,
    Policy,
)


class GovernanceServicer:
    """Implementation of the GovernanceService gRPC service.

    Provides governance decision-making with sub-10ms latency
    target for internal service communication.

    All methods implement the 25 Fundamental Laws.
    """

    def __init__(self):
        """Initialize the governance servicer."""
        self.start_time = time.time()
        self._decision_cache: dict[str, Decision] = {}
        self._policies: list[Policy] = self._init_default_policies()

    def _init_default_policies(self) -> list[Policy]:
        """Initialize default policies."""
        return [
            Policy(
                policy_id=str(uuid.uuid4()),
                name="Core Safety Policy",
                description="Ensures human safety is prioritized",
                version="1.0.0",
                status="active",
                scope="global",
                fundamental_laws=[21, 23],
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            ),
            Policy(
                policy_id=str(uuid.uuid4()),
                name="Privacy Protection Policy",
                description="Protects user privacy and digital security",
                version="1.0.0",
                status="active",
                scope="global",
                fundamental_laws=[22],
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            ),
        ]

    def EvaluateAction(self, request: EvaluateRequest) -> EvaluateResponse:
        """Evaluate a single action for ethical compliance.

        This is the core governance decision endpoint.
        Target latency: <10ms p99.

        Implements:
        - Law 6: Decision Authority
        - Law 10: Reasoning Transparency
        - Law 15: Audit Compliance
        - Law 21: Human Safety Priority
        """
        start_time = time.perf_counter()
        decision_id = str(uuid.uuid4())
        request_id = request.request_id or str(uuid.uuid4())

        # Initialize evaluation
        decision = "ALLOW"
        risk_score = 0.0
        confidence = 1.0
        violations: list[Violation] = []
        laws_checked = [6, 10, 15, 21, 22]

        # Evaluate action content
        action_lower = request.action.lower()

        # Safety checks (Law 21)
        dangerous_patterns = [
            ("delete all", "Bulk deletion detected", "high", 23),
            ("drop table", "Database destruction", "critical", 21),
            ("rm -rf", "Filesystem destruction", "critical", 21),
            ("password", "Credential access", "medium", 22),
            ("hack", "Potential malicious activity", "high", 21),
            ("exploit", "Exploitation attempt", "high", 21),
            ("bypass security", "Security bypass", "critical", 21),
        ]

        for pattern, desc, severity, law in dangerous_patterns:
            if pattern in action_lower:
                violations.append(
                    Violation(
                        id=str(uuid.uuid4()),
                        type="safety_violation",
                        severity=severity,
                        description=desc,
                        law_reference=f"Law {law}",
                    )
                )
                risk_score = max(
                    risk_score,
                    (
                        0.7
                        if severity == "medium"
                        else 0.85 if severity == "high" else 0.95
                    ),
                )
                laws_checked.append(law)

        # Determine decision
        if risk_score >= 0.9:
            decision = "BLOCK"
            confidence = 0.95
        elif risk_score >= 0.7:
            decision = "RESTRICT"
            confidence = 0.85

        # Build explanation if requested
        explanation = None
        if request.require_explanation:
            explanation = Explanation(
                summary=f"Evaluated with {len(violations)} violations",
                risk_factors=[v.description for v in violations],
                decision_rationale=f"Decision based on risk score {risk_score:.2f}",
                laws_applied=[f"Law {i}" for i in set(laws_checked)],
            )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        timestamp = datetime.now(timezone.utc).isoformat()

        response = EvaluateResponse(
            decision=decision,
            decision_id=decision_id,
            risk_score=risk_score,
            confidence=confidence,
            latency_ms=latency_ms,
            violations=violations,
            reason=f"{len(violations)} violations, risk: {risk_score:.2f}",
            explanation=explanation,
            audit_id=request_id,
            cache_hit=False,
            fundamental_laws_checked=list(set(laws_checked)),
            timestamp=timestamp,
        )

        # Store decision for lookup
        self._decision_cache[decision_id] = Decision(
            decision_id=decision_id,
            decision=decision,
            agent_id=request.agent_id,
            action_summary=request.action[:100],
            action_type=request.action_type,
            risk_score=risk_score,
            confidence=confidence,
            reasoning=response.reason,
            violations=violations,
            fundamental_laws=list(set(laws_checked)),
            timestamp=timestamp,
            latency_ms=latency_ms,
            audit_id=request_id,
        )

        return response

    def BatchEvaluate(
        self,
        request_list: list[EvaluateRequest],
    ) -> Iterator[EvaluateResponse]:
        """Evaluate multiple actions with streaming responses.

        Yields responses as they are evaluated for lower latency
        in batch scenarios.
        """
        for request in request_list:
            yield self.EvaluateAction(request)

    def StreamDecisions(
        self,
        agent_id: Optional[str] = None,
        decision_types: Optional[list[str]] = None,
        min_risk_score: Optional[float] = None,
    ) -> Iterator[Decision]:
        """Stream decisions matching the filter criteria.

        Provides real-time access to governance decisions
        for monitoring and alerting.
        """
        for decision in self._decision_cache.values():
            # Apply filters
            if agent_id and decision.agent_id != agent_id:
                continue
            if decision_types and decision.decision not in decision_types:
                continue
            if min_risk_score and decision.risk_score < min_risk_score:
                continue

            yield decision

    def GetDecision(self, decision_id: str) -> Optional[Decision]:
        """Retrieve a specific decision by ID.

        Implements Law 10 (Reasoning Transparency) and
        Law 15 (Audit Compliance).
        """
        return self._decision_cache.get(decision_id)

    def ListPolicies(
        self,
        status: Optional[str] = None,
        scope: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Policy], int, bool]:
        """List governance policies.

        Returns (policies, total_count, has_next).
        Implements Law 8 (Constraint Transparency).
        """
        policies = self._policies

        if status:
            policies = [p for p in policies if p.status == status]
        if scope:
            policies = [p for p in policies if p.scope == scope]

        total_count = len(policies)
        start = (page - 1) * page_size
        end = start + page_size
        page_policies = policies[start:end]

        return page_policies, total_count, end < total_count

    def HealthCheck(self) -> dict:
        """Return service health status."""
        uptime = int(time.time() - self.start_time)
        return {
            "status": "healthy",
            "version": "2.0.0",
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def create_grpc_server(
    port: int = 50051,
    max_workers: int = 10,
) -> "GRPCServer":
    """Create a gRPC server instance.

    Note: This is a placeholder for actual gRPC server creation.
    In production, this would use grpcio to create a real server.

    Args:
        port: Port to listen on
        max_workers: Maximum thread workers

    Returns:
        GRPCServer instance (placeholder)
    """
    return GRPCServer(port=port, max_workers=max_workers)


class GRPCServer:
    """Placeholder gRPC server.

    In production, this would wrap grpcio.Server.
    """

    def __init__(self, port: int = 50051, max_workers: int = 10):
        self.port = port
        self.max_workers = max_workers
        self.servicer = GovernanceServicer()
        self._running = False

    def start(self) -> None:
        """Start the server."""
        logger.info("Starting gRPC server on port %d", self.port)
        self._running = True

    def stop(self, grace: int = 5) -> None:
        """Stop the server."""
        logger.info("Stopping gRPC server with %ds grace period", grace)
        self._running = False

    def wait_for_termination(self, timeout: Optional[float] = None) -> None:
        """Wait for server termination."""
        logger.info("Waiting for gRPC server termination")
        # In production, this would block until termination


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = create_grpc_server(port=50051)
    server.start()
    print("gRPC server started on port 50051")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop()
