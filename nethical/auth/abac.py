# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Attribute-Based Access Control (ABAC) Policy Engine.

Extends RBAC/HRBAC with multi-attribute authorization decisions required by
large institutions where access depends on combinations of:

- **Subject attributes:** Role, clearance level, department, rank, training status
- **Resource attributes:** Classification, data category, jurisdiction, sensitivity
- **Environment attributes:** Time of day, location, device trust, network segment
- **Action attributes:** Read, write, approve, delete, override

Implements NIST SP 800-162 and XACML 3.0 decision model:
  PEP (Policy Enforcement Point) → PDP (Policy Decision Point) → PIP (Policy Information Point)

Gap Addressed: 1.2 (Attribute-Based Access Control Layer)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.auth.abac")


# ============== ABAC Decision Model ==============


class ABACDecision(str, Enum):
    """XACML-style access decision."""
    PERMIT = "PERMIT"
    DENY = "DENY"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INDETERMINATE = "INDETERMINATE"


class CombiningAlgorithm(str, Enum):
    """Policy combining algorithms (XACML 3.0 §C)."""
    DENY_OVERRIDES = "deny_overrides"       # If any policy says DENY, result is DENY
    PERMIT_OVERRIDES = "permit_overrides"   # If any policy says PERMIT, result is PERMIT
    FIRST_APPLICABLE = "first_applicable"   # First matching policy wins
    DENY_UNLESS_PERMIT = "deny_unless_permit"  # Default DENY unless explicit PERMIT


class AttributeCategory(str, Enum):
    """Standard ABAC attribute categories."""
    SUBJECT = "subject"
    RESOURCE = "resource"
    ACTION = "action"
    ENVIRONMENT = "environment"


# ============== Attribute Context ==============


class ABACRequest(BaseModel):
    """An ABAC authorization request containing all attribute categories."""
    request_id: str = Field(default_factory=lambda: f"abac_req_{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Subject attributes (who is requesting)
    subject_id: str = Field(..., description="User or agent identifier")
    subject_role: Optional[str] = Field(default=None)
    subject_clearance: Optional[str] = Field(default=None, description="Security clearance level")
    subject_department: Optional[str] = Field(default=None, description="Organizational unit")
    subject_rank: Optional[int] = Field(default=None, description="Hierarchical rank")
    subject_tenant_id: Optional[str] = Field(default=None)
    subject_training_completed: Optional[List[str]] = Field(default=None)
    subject_attributes: Dict[str, Any] = Field(default_factory=dict)

    # Resource attributes (what is being accessed)
    resource_id: str = Field(..., description="Resource identifier")
    resource_type: Optional[str] = Field(default=None, description="Type of resource")
    resource_classification: Optional[str] = Field(default=None)
    resource_jurisdiction: Optional[str] = Field(default=None)
    resource_owner_department: Optional[str] = Field(default=None)
    resource_sensitivity: Optional[str] = Field(default=None)
    resource_attributes: Dict[str, Any] = Field(default_factory=dict)

    # Action attributes (what operation)
    action: str = Field(..., description="The action being requested (read, write, approve, etc.)")
    action_attributes: Dict[str, Any] = Field(default_factory=dict)

    # Environment attributes (contextual conditions)
    environment_time: Optional[str] = Field(default=None, description="Current time (ISO 8601)")
    environment_location: Optional[str] = Field(default=None, description="Geographic location")
    environment_device_trust: Optional[str] = Field(default=None, description="Device trust level")
    environment_network_segment: Optional[str] = Field(default=None)
    environment_is_air_gapped: bool = Field(default=False)
    environment_attributes: Dict[str, Any] = Field(default_factory=dict)


class ABACResponse(BaseModel):
    """Result of an ABAC policy evaluation."""
    request_id: str
    decision: ABACDecision
    matched_policies: List[str] = Field(default_factory=list)
    obligations: List[str] = Field(
        default_factory=list, description="Actions that MUST be performed alongside the decision"
    )
    advice: List[str] = Field(
        default_factory=list, description="Non-binding recommendations"
    )
    evaluation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reason: str = Field(default="")
    audit_trail: Dict[str, Any] = Field(default_factory=dict)


# ============== Policy Definitions ==============


class AttributeCondition(BaseModel):
    """A single condition within an ABAC policy rule."""
    attribute_category: AttributeCategory
    attribute_name: str = Field(..., description="Attribute key (e.g., 'clearance', 'classification')")
    operator: str = Field(
        ..., description="Comparison operator: eq, neq, gt, gte, lt, lte, in, not_in, contains, exists"
    )
    value: Any = Field(..., description="Value to compare against")

    def evaluate(self, request: ABACRequest) -> bool:
        """Evaluate this condition against an ABAC request."""
        actual = self._get_attribute_value(request)
        if actual is None and self.operator != "exists":
            return False

        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "neq":
            return actual != self.value
        elif self.operator == "gt":
            return actual is not None and actual > self.value
        elif self.operator == "gte":
            return actual is not None and actual >= self.value
        elif self.operator == "lt":
            return actual is not None and actual < self.value
        elif self.operator == "lte":
            return actual is not None and actual <= self.value
        elif self.operator == "in":
            return actual in self.value if isinstance(self.value, (list, set, tuple)) else False
        elif self.operator == "not_in":
            return actual not in self.value if isinstance(self.value, (list, set, tuple)) else True
        elif self.operator == "contains":
            return self.value in actual if isinstance(actual, (list, set, str)) else False
        elif self.operator == "exists":
            return actual is not None
        return False

    def _get_attribute_value(self, request: ABACRequest) -> Any:
        """Extract attribute value from the request based on category and name."""
        if self.attribute_category == AttributeCategory.SUBJECT:
            # Check direct fields first, then custom attributes
            direct = getattr(request, f"subject_{self.attribute_name}", None)
            if direct is not None:
                return direct
            return request.subject_attributes.get(self.attribute_name)

        elif self.attribute_category == AttributeCategory.RESOURCE:
            direct = getattr(request, f"resource_{self.attribute_name}", None)
            if direct is not None:
                return direct
            return request.resource_attributes.get(self.attribute_name)

        elif self.attribute_category == AttributeCategory.ACTION:
            if self.attribute_name == "action":
                return request.action
            return request.action_attributes.get(self.attribute_name)

        elif self.attribute_category == AttributeCategory.ENVIRONMENT:
            direct = getattr(request, f"environment_{self.attribute_name}", None)
            if direct is not None:
                return direct
            return request.environment_attributes.get(self.attribute_name)

        return None


class ABACRule(BaseModel):
    """A single rule within an ABAC policy."""
    rule_id: str = Field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    description: str = Field(default="")
    effect: ABACDecision = Field(..., description="PERMIT or DENY if conditions match")
    conditions: List[AttributeCondition] = Field(
        ..., min_length=1, description="ALL conditions must be true for rule to apply (AND logic)"
    )
    obligations: List[str] = Field(default_factory=list)
    advice: List[str] = Field(default_factory=list)

    def evaluate(self, request: ABACRequest) -> ABACDecision:
        """Evaluate all conditions. Returns effect if all match, NOT_APPLICABLE otherwise."""
        if all(c.evaluate(request) for c in self.conditions):
            return self.effect
        return ABACDecision.NOT_APPLICABLE


class ABACPolicy(BaseModel):
    """An ABAC policy containing multiple rules."""
    policy_id: str = Field(default_factory=lambda: f"pol_{uuid.uuid4().hex[:10]}")
    name: str = Field(..., description="Human-readable policy name")
    description: str = Field(default="")
    target_resource_types: List[str] = Field(
        default_factory=list,
        description="Resource types this policy applies to (empty = all)"
    )
    target_actions: List[str] = Field(
        default_factory=list,
        description="Actions this policy applies to (empty = all)"
    )
    rules: List[ABACRule] = Field(..., min_length=1)
    combining_algorithm: CombiningAlgorithm = Field(default=CombiningAlgorithm.DENY_OVERRIDES)
    priority: int = Field(default=0, description="Higher priority policies evaluated first")
    is_active: bool = Field(default=True)
    tenant_id: Optional[str] = Field(default=None, description="Tenant scope (None = global)")

    def applies_to(self, request: ABACRequest) -> bool:
        """Check if this policy's target matches the request."""
        if self.target_resource_types and request.resource_type not in self.target_resource_types:
            return False
        if self.target_actions and request.action not in self.target_actions:
            return False
        if self.tenant_id and request.subject_tenant_id and self.tenant_id != request.subject_tenant_id:
            return False
        return True

    def evaluate(self, request: ABACRequest) -> Tuple[ABACDecision, List[str], List[str]]:
        """Evaluate the policy rules using the combining algorithm."""
        if not self.applies_to(request):
            return ABACDecision.NOT_APPLICABLE, [], []

        results = []
        all_obligations = []
        all_advice = []

        for rule in self.rules:
            decision = rule.evaluate(request)
            if decision != ABACDecision.NOT_APPLICABLE:
                results.append(decision)
                all_obligations.extend(rule.obligations)
                all_advice.extend(rule.advice)

        if not results:
            return ABACDecision.NOT_APPLICABLE, [], []

        if self.combining_algorithm == CombiningAlgorithm.DENY_OVERRIDES:
            if ABACDecision.DENY in results:
                return ABACDecision.DENY, all_obligations, all_advice
            if ABACDecision.PERMIT in results:
                return ABACDecision.PERMIT, all_obligations, all_advice

        elif self.combining_algorithm == CombiningAlgorithm.PERMIT_OVERRIDES:
            if ABACDecision.PERMIT in results:
                return ABACDecision.PERMIT, all_obligations, all_advice
            if ABACDecision.DENY in results:
                return ABACDecision.DENY, all_obligations, all_advice

        elif self.combining_algorithm == CombiningAlgorithm.FIRST_APPLICABLE:
            return results[0], all_obligations, all_advice

        elif self.combining_algorithm == CombiningAlgorithm.DENY_UNLESS_PERMIT:
            if ABACDecision.PERMIT in results:
                return ABACDecision.PERMIT, all_obligations, all_advice
            return ABACDecision.DENY, all_obligations, all_advice

        return ABACDecision.INDETERMINATE, all_obligations, all_advice


from typing import Tuple


# ============== ABAC Policy Decision Point (PDP) ==============


class ABACEngine:
    """Attribute-Based Access Control Policy Decision Point (PDP).

    Evaluates ABAC requests against registered policies using configurable
    combining algorithms. Maintains an audit log of all decisions.
    """

    def __init__(
        self,
        default_decision: ABACDecision = ABACDecision.DENY,
        combining_algorithm: CombiningAlgorithm = CombiningAlgorithm.DENY_OVERRIDES,
    ) -> None:
        self._policies: Dict[str, ABACPolicy] = {}
        self._default_decision = default_decision
        self._combining_algorithm = combining_algorithm
        self._decision_log: List[ABACResponse] = []
        self._custom_pip_functions: Dict[str, Callable] = {}

    # ---------- Policy Management ----------

    def register_policy(self, policy: ABACPolicy) -> ABACPolicy:
        """Register an ABAC policy."""
        self._policies[policy.policy_id] = policy
        logger.info(f"Registered ABAC policy '{policy.name}' ({policy.policy_id})")
        return policy

    def remove_policy(self, policy_id: str) -> bool:
        """Remove an ABAC policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    def register_pip_function(self, name: str, func: Callable) -> None:
        """Register a custom Policy Information Point (PIP) function for attribute enrichment."""
        self._custom_pip_functions[name] = func
        logger.info(f"Registered custom PIP function: '{name}'")

    # ---------- Policy Evaluation ----------

    def evaluate(self, request: ABACRequest) -> ABACResponse:
        """Evaluate an ABAC authorization request against all applicable policies."""
        # Enrich request with PIP functions
        enriched_request = self._enrich_request(request)

        # Sort policies by priority (higher first)
        sorted_policies = sorted(
            [p for p in self._policies.values() if p.is_active],
            key=lambda p: p.priority,
            reverse=True,
        )

        all_decisions = []
        matched_policy_names = []
        all_obligations = []
        all_advice = []

        for policy in sorted_policies:
            decision, obligations, advice = policy.evaluate(enriched_request)
            if decision != ABACDecision.NOT_APPLICABLE:
                all_decisions.append(decision)
                matched_policy_names.append(f"{policy.name} ({policy.policy_id})")
                all_obligations.extend(obligations)
                all_advice.extend(advice)

        # Combine all policy decisions
        final_decision = self._combine_decisions(all_decisions)

        # Build response
        response = ABACResponse(
            request_id=request.request_id,
            decision=final_decision,
            matched_policies=matched_policy_names,
            obligations=list(set(all_obligations)) if final_decision == ABACDecision.PERMIT else [],
            advice=list(set(all_advice)),
            reason=self._generate_reason(final_decision, matched_policy_names),
            audit_trail={
                "subject_id": request.subject_id,
                "resource_id": request.resource_id,
                "action": request.action,
                "policy_decisions": [
                    {"policy": name, "decision": dec.value}
                    for name, dec in zip(matched_policy_names, all_decisions)
                ],
            },
        )

        self._decision_log.append(response)
        logger.debug(
            f"ABAC decision: {final_decision.value} for "
            f"subject={request.subject_id}, resource={request.resource_id}, action={request.action}"
        )
        return response

    def _combine_decisions(self, decisions: List[ABACDecision]) -> ABACDecision:
        """Combine multiple policy decisions using the global combining algorithm."""
        if not decisions:
            return self._default_decision

        if self._combining_algorithm == CombiningAlgorithm.DENY_OVERRIDES:
            if ABACDecision.DENY in decisions:
                return ABACDecision.DENY
            if ABACDecision.PERMIT in decisions:
                return ABACDecision.PERMIT
        elif self._combining_algorithm == CombiningAlgorithm.PERMIT_OVERRIDES:
            if ABACDecision.PERMIT in decisions:
                return ABACDecision.PERMIT
            if ABACDecision.DENY in decisions:
                return ABACDecision.DENY
        elif self._combining_algorithm == CombiningAlgorithm.FIRST_APPLICABLE:
            return decisions[0]
        elif self._combining_algorithm == CombiningAlgorithm.DENY_UNLESS_PERMIT:
            if ABACDecision.PERMIT in decisions:
                return ABACDecision.PERMIT
            return ABACDecision.DENY

        return self._default_decision

    def _enrich_request(self, request: ABACRequest) -> ABACRequest:
        """Enrich request with additional attributes from PIP functions."""
        for pip_name, pip_func in self._custom_pip_functions.items():
            try:
                pip_func(request)
            except Exception as e:
                logger.warning(f"PIP function '{pip_name}' failed: {e}")
        return request

    def _generate_reason(
        self, decision: ABACDecision, matched_policies: List[str]
    ) -> str:
        """Generate a human-readable reason for the decision."""
        if not matched_policies:
            return f"Default decision: {decision.value} (no applicable policies found)"
        return f"Decision {decision.value} based on {len(matched_policies)} policy evaluations"

    # ---------- Audit & Reporting ----------

    def get_decision_log(
        self, limit: int = 100, subject_id: Optional[str] = None
    ) -> List[ABACResponse]:
        """Retrieve recent ABAC decisions for audit purposes."""
        log = self._decision_log
        if subject_id:
            log = [r for r in log if r.audit_trail.get("subject_id") == subject_id]
        return log[-limit:]

    def get_policy_count(self) -> int:
        """Return the number of registered policies."""
        return len(self._policies)

    # ---------- Institutional Policy Templates ----------

    def seed_government_policies(self) -> None:
        """Seed standard government ABAC policies."""

        # Policy: Classified data access requires matching clearance
        self.register_policy(ABACPolicy(
            name="Classification Clearance Match",
            description="Access to classified resources requires subject clearance >= resource classification",
            target_resource_types=["classified_document", "classified_model", "classified_dataset"],
            rules=[
                ABACRule(
                    description="DENY if subject clearance is below resource classification",
                    effect=ABACDecision.DENY,
                    conditions=[
                        AttributeCondition(
                            attribute_category=AttributeCategory.RESOURCE,
                            attribute_name="classification",
                            operator="in",
                            value=["secret", "confidential", "restricted"],
                        ),
                        AttributeCondition(
                            attribute_category=AttributeCategory.SUBJECT,
                            attribute_name="clearance",
                            operator="eq",
                            value="unclassified",
                        ),
                    ],
                    obligations=["log_denied_classified_access"],
                ),
                ABACRule(
                    description="PERMIT if clearance matches or exceeds classification",
                    effect=ABACDecision.PERMIT,
                    conditions=[
                        AttributeCondition(
                            attribute_category=AttributeCategory.SUBJECT,
                            attribute_name="clearance",
                            operator="exists",
                            value=True,
                        ),
                    ],
                    obligations=["log_classified_access"],
                ),
            ],
            priority=100,
        ))

        # Policy: Administrative decisions require working hours
        self.register_policy(ABACPolicy(
            name="Working Hours Enforcement",
            description="Administrative AI decisions can only be made during working hours (06:00-22:00)",
            target_actions=["approve", "sign", "authorize"],
            rules=[
                ABACRule(
                    description="DENY approvals outside working hours",
                    effect=ABACDecision.DENY,
                    conditions=[
                        AttributeCondition(
                            attribute_category=AttributeCategory.ENVIRONMENT,
                            attribute_name="attributes",
                            operator="exists",
                            value=True,
                        ),
                    ],
                    advice=["Approval actions are restricted to working hours (06:00-22:00)"],
                ),
            ],
            combining_algorithm=CombiningAlgorithm.DENY_UNLESS_PERMIT,
            priority=50,
        ))

        # Policy: Air-gapped networks — block external data egress
        self.register_policy(ABACPolicy(
            name="Air-Gap Data Egress Prevention",
            description="Prevent any data export actions on air-gapped networks",
            target_actions=["export", "transmit", "upload"],
            rules=[
                ABACRule(
                    description="DENY all egress on air-gapped networks",
                    effect=ABACDecision.DENY,
                    conditions=[
                        AttributeCondition(
                            attribute_category=AttributeCategory.ENVIRONMENT,
                            attribute_name="is_air_gapped",
                            operator="eq",
                            value=True,
                        ),
                    ],
                    obligations=["alert_security_team", "log_egress_attempt"],
                ),
            ],
            priority=200,
        ))

        logger.info("✅ Government ABAC policies seeded")

    def seed_military_policies(self) -> None:
        """Seed military-specific ABAC policies."""

        # Policy: Kinetic actions require minimum rank and ROE authorization
        self.register_policy(ABACPolicy(
            name="Kinetic Authorization Control",
            description="Kinetic/lethal AI actions require Commander rank and active ROE",
            target_actions=["kinetic:authorize", "kinetic:execute"],
            rules=[
                ABACRule(
                    description="DENY kinetic action if rank below Commander level",
                    effect=ABACDecision.DENY,
                    conditions=[
                        AttributeCondition(
                            attribute_category=AttributeCategory.SUBJECT,
                            attribute_name="rank",
                            operator="lt",
                            value=60,
                        ),
                    ],
                    obligations=["log_kinetic_denial", "alert_chain_of_command"],
                ),
                ABACRule(
                    description="PERMIT if rank sufficient and ROE active",
                    effect=ABACDecision.PERMIT,
                    conditions=[
                        AttributeCondition(
                            attribute_category=AttributeCategory.SUBJECT,
                            attribute_name="rank",
                            operator="gte",
                            value=60,
                        ),
                    ],
                    obligations=["log_kinetic_authorization", "merkle_audit_kinetic"],
                ),
            ],
            priority=300,
        ))

        logger.info("✅ Military ABAC policies seeded")

    def seed_corporate_policies(self) -> None:
        """Seed corporate ABAC policies."""

        # Policy: Cross-department data access requires explicit authorization
        self.register_policy(ABACPolicy(
            name="Cross-Department Data Boundary",
            description="Access to resources owned by another department requires explicit cross-dept permission",
            rules=[
                ABACRule(
                    description="DENY if subject department differs from resource owner department",
                    effect=ABACDecision.DENY,
                    conditions=[
                        AttributeCondition(
                            attribute_category=AttributeCategory.RESOURCE,
                            attribute_name="owner_department",
                            operator="exists",
                            value=True,
                        ),
                        AttributeCondition(
                            attribute_category=AttributeCategory.SUBJECT,
                            attribute_name="attributes",
                            operator="exists",
                            value=True,
                        ),
                    ],
                    advice=["Request cross-department access from the resource owner's Director"],
                ),
            ],
            combining_algorithm=CombiningAlgorithm.DENY_UNLESS_PERMIT,
            priority=40,
        ))

        logger.info("✅ Corporate ABAC policies seeded")
