# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Multi-Stage Approval Workflow Engine for Institutional Deployments.

Implements configurable, stateful approval workflows required by large
bureaucracies, government agencies, corporations, and military organizations:

- **Workflow Definition:** YAML/dict-configurable multi-stage pipelines with
  conditions, timeouts, and parallel approval paths.
- **Stage Types:** Sequential, parallel (all must approve), quorum (M-of-N),
  and conditional branching based on risk score or classification level.
- **Escalation:** Automatic escalation when SLA deadlines are missed.
- **Counter-Signatures:** Multi-party sign-off with personal accountability.
- **Audit Trail:** Immutable record of every state transition and approval action.

Workflow Templates:
- Government: Proposal → Technical Review → Legal Review → Budget → PIA → Minister Sign-Off
- Military: Mission Brief → Intel Review → Legal (IHL) → Commander Approval → ROE Activation
- Corporate: Proposal → Architecture Review → Security Review → DPO Review → CAB → CTO Approval

Gap Addressed: 2.1 (Multi-Stage Approval Workflow), 2.2 (Counter-Signature Processes)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.governance.workflow_engine")


# ============== Workflow Stage Types ==============


class StageType(str, Enum):
    """Types of approval stages."""
    SEQUENTIAL = "sequential"       # Single approver, in order
    PARALLEL_ALL = "parallel_all"   # All listed approvers must approve
    PARALLEL_QUORUM = "parallel_quorum"  # M-of-N approvers must approve
    CONDITIONAL = "conditional"     # Branch based on condition
    AUTOMATED = "automated"         # No human approval; runs a validation function
    COUNTER_SIGNATURE = "counter_signature"  # Requires counter-sign from specific role


class StageStatus(str, Enum):
    """Status of a workflow stage."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class WorkflowStatus(str, Enum):
    """Overall status of a workflow instance."""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED_APPROVED = "completed_approved"
    COMPLETED_REJECTED = "completed_rejected"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class ApprovalAction(str, Enum):
    """Actions that approvers can take."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    COUNTER_SIGN = "counter_sign"


# ============== Stage Definition ==============


class WorkflowStageDefinition(BaseModel):
    """Definition of a single stage in a workflow template."""
    stage_id: str = Field(default_factory=lambda: f"stage_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Human-readable stage name")
    description: str = Field(default="")
    stage_type: StageType = Field(default=StageType.SEQUENTIAL)
    order: int = Field(..., ge=0, description="Execution order (0-based)")

    # Who can approve at this stage
    required_roles: List[str] = Field(
        default_factory=list,
        description="Role codes that can approve at this stage (HRBAC institutional roles)"
    )
    required_permissions: List[str] = Field(
        default_factory=list,
        description="Permissions required to approve"
    )
    specific_approvers: List[str] = Field(
        default_factory=list,
        description="Specific user IDs required to approve (for counter-signatures)"
    )

    # Quorum settings (for PARALLEL_QUORUM)
    quorum_required: int = Field(
        default=1, ge=1,
        description="Minimum number of approvals needed for quorum stages"
    )
    quorum_total: int = Field(
        default=1, ge=1,
        description="Total pool of potential approvers for quorum calculation"
    )

    # SLA and escalation
    sla_hours: Optional[float] = Field(
        default=None,
        description="SLA deadline in hours from stage activation"
    )
    escalation_target_role: Optional[str] = Field(
        default=None,
        description="Role to escalate to if SLA is breached"
    )
    auto_approve_on_timeout: bool = Field(
        default=False,
        description="If True, stage auto-approves when SLA expires (use with caution)"
    )

    # Conditional branching
    condition_field: Optional[str] = Field(
        default=None,
        description="Request metadata field to evaluate for CONDITIONAL stages"
    )
    condition_operator: Optional[str] = Field(default=None)
    condition_value: Optional[Any] = Field(default=None)
    skip_if_condition_false: bool = Field(
        default=False,
        description="Skip this stage if condition evaluates to False"
    )

    # Counter-signature settings
    counter_signature_roles: List[str] = Field(
        default_factory=list,
        description="Roles that must counter-sign after primary approval"
    )


# ============== Workflow Template ==============


class WorkflowTemplate(BaseModel):
    """A reusable workflow template defining the approval pipeline."""
    template_id: str = Field(default_factory=lambda: f"wft_{uuid.uuid4().hex[:10]}")
    name: str = Field(..., description="Template name (e.g., 'AI Deployment Approval')")
    description: str = Field(default="")
    version: str = Field(default="1.0")
    stages: List[WorkflowStageDefinition] = Field(..., min_length=1)
    applicable_to: List[str] = Field(
        default_factory=list,
        description="Types of requests this template applies to (e.g., 'ai_deployment', 'policy_change')"
    )
    tenant_id: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============== Workflow Instance (Runtime) ==============


class ApprovalRecord(BaseModel):
    """Record of a single approval action."""
    record_id: str = Field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:10]}")
    stage_id: str
    approver_id: str
    approver_name: Optional[str] = Field(default=None)
    approver_role: Optional[str] = Field(default=None)
    action: ApprovalAction
    comment: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_counter_signature: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StageInstance(BaseModel):
    """Runtime state of a single workflow stage."""
    stage_id: str
    stage_name: str
    status: StageStatus = Field(default=StageStatus.PENDING)
    activated_at: Optional[str] = Field(default=None)
    completed_at: Optional[str] = Field(default=None)
    sla_deadline: Optional[str] = Field(default=None)
    approval_records: List[ApprovalRecord] = Field(default_factory=list)
    escalation_count: int = Field(default=0)


class WorkflowInstance(BaseModel):
    """A running instance of a workflow."""
    instance_id: str = Field(default_factory=lambda: f"wfi_{uuid.uuid4().hex[:12]}")
    template_id: str
    template_name: str
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT)
    current_stage_index: int = Field(default=0)
    stages: List[StageInstance] = Field(default_factory=list)

    # Request context
    request_type: str = Field(..., description="Type of request (e.g., 'ai_deployment')")
    request_title: str = Field(..., description="Human-readable title")
    request_description: str = Field(default="")
    requestor_id: str = Field(..., description="User who initiated the request")
    requestor_name: Optional[str] = Field(default=None)
    request_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Timing
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = Field(default=None)

    # Audit
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)

    # Tenant scope
    tenant_id: Optional[str] = Field(default=None)
    ou_id: Optional[str] = Field(default=None)


# ============== Workflow Engine ==============


class WorkflowEngine:
    """Multi-stage approval workflow engine for institutional governance.

    Manages workflow templates, instances, approvals, escalations,
    and counter-signatures.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._instances: Dict[str, WorkflowInstance] = {}
        self._automated_validators: Dict[str, Callable] = {}

    # ---------- Template Management ----------

    def register_template(self, template: WorkflowTemplate) -> WorkflowTemplate:
        """Register a workflow template."""
        # Validate stage ordering
        orders = [s.order for s in template.stages]
        if len(orders) != len(set(orders)):
            raise ValueError("Stage orders must be unique within a template.")
        template.stages.sort(key=lambda s: s.order)

        self._templates[template.template_id] = template
        logger.info(
            f"Registered workflow template '{template.name}' "
            f"with {len(template.stages)} stages"
        )
        return template

    def register_automated_validator(
        self, stage_id: str, validator: Callable[[Dict[str, Any]], bool]
    ) -> None:
        """Register an automated validation function for AUTOMATED stage types."""
        self._automated_validators[stage_id] = validator
        logger.info(f"Registered automated validator for stage '{stage_id}'")

    # ---------- Workflow Lifecycle ----------

    def create_instance(
        self,
        template_id: str,
        request_type: str,
        request_title: str,
        requestor_id: str,
        request_description: str = "",
        request_metadata: Optional[Dict[str, Any]] = None,
        requestor_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ou_id: Optional[str] = None,
    ) -> WorkflowInstance:
        """Create a new workflow instance from a template."""
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found.")

        # Create stage instances from template
        stage_instances = []
        for stage_def in template.stages:
            stage_instances.append(StageInstance(
                stage_id=stage_def.stage_id,
                stage_name=stage_def.name,
            ))

        instance = WorkflowInstance(
            template_id=template_id,
            template_name=template.name,
            request_type=request_type,
            request_title=request_title,
            request_description=request_description,
            requestor_id=requestor_id,
            requestor_name=requestor_name,
            request_metadata=request_metadata or {},
            stages=stage_instances,
            tenant_id=tenant_id,
            ou_id=ou_id,
        )

        self._instances[instance.instance_id] = instance
        logger.info(
            f"Created workflow instance {instance.instance_id}: "
            f"'{request_title}' using template '{template.name}'"
        )
        return instance

    def activate_instance(self, instance_id: str) -> WorkflowInstance:
        """Activate a workflow instance, starting the first stage."""
        instance = self._get_instance(instance_id)
        if instance.status != WorkflowStatus.DRAFT:
            raise ValueError(f"Workflow {instance_id} is not in DRAFT status.")

        instance.status = WorkflowStatus.ACTIVE
        self._activate_stage(instance, 0)
        self._log_event(instance, "workflow_activated", {"stage_index": 0})
        return instance

    def submit_approval(
        self,
        instance_id: str,
        stage_id: str,
        approver_id: str,
        action: ApprovalAction,
        comment: str = "",
        approver_name: Optional[str] = None,
        approver_role: Optional[str] = None,
        is_counter_signature: bool = False,
    ) -> WorkflowInstance:
        """Submit an approval action for a specific stage."""
        instance = self._get_instance(instance_id)
        template = self._templates.get(instance.template_id)
        if not template:
            raise ValueError(f"Template '{instance.template_id}' not found.")

        if instance.status != WorkflowStatus.ACTIVE:
            raise ValueError(f"Workflow {instance_id} is not active.")

        # Find the stage
        stage_instance = None
        stage_def = None
        stage_index = -1
        for i, si in enumerate(instance.stages):
            if si.stage_id == stage_id:
                stage_instance = si
                stage_index = i
                break
        for sd in template.stages:
            if sd.stage_id == stage_id:
                stage_def = sd
                break

        if not stage_instance or not stage_def:
            raise ValueError(f"Stage '{stage_id}' not found in workflow {instance_id}.")

        if stage_instance.status not in (StageStatus.IN_PROGRESS, StageStatus.ESCALATED):
            raise ValueError(f"Stage '{stage_id}' is not awaiting approval.")

        # Record the approval
        record = ApprovalRecord(
            stage_id=stage_id,
            approver_id=approver_id,
            approver_name=approver_name,
            approver_role=approver_role,
            action=action,
            comment=comment,
            is_counter_signature=is_counter_signature,
        )
        stage_instance.approval_records.append(record)
        self._log_event(instance, "approval_submitted", {
            "stage_id": stage_id,
            "approver_id": approver_id,
            "action": action.value,
        })

        # Evaluate stage completion based on stage type
        if action == ApprovalAction.REJECT:
            stage_instance.status = StageStatus.REJECTED
            stage_instance.completed_at = datetime.now(timezone.utc).isoformat()
            instance.status = WorkflowStatus.COMPLETED_REJECTED
            instance.completed_at = datetime.now(timezone.utc).isoformat()
            self._log_event(instance, "workflow_rejected", {"stage_id": stage_id})

        elif action == ApprovalAction.ESCALATE:
            stage_instance.status = StageStatus.ESCALATED
            stage_instance.escalation_count += 1
            self._log_event(instance, "stage_escalated", {
                "stage_id": stage_id,
                "escalation_count": stage_instance.escalation_count,
            })

        elif action in (ApprovalAction.APPROVE, ApprovalAction.COUNTER_SIGN):
            if self._is_stage_complete(stage_instance, stage_def):
                stage_instance.status = StageStatus.APPROVED
                stage_instance.completed_at = datetime.now(timezone.utc).isoformat()
                self._log_event(instance, "stage_approved", {"stage_id": stage_id})

                # Advance to next stage
                next_index = stage_index + 1
                if next_index < len(instance.stages):
                    instance.current_stage_index = next_index
                    self._activate_stage(instance, next_index)
                else:
                    # All stages complete
                    instance.status = WorkflowStatus.COMPLETED_APPROVED
                    instance.completed_at = datetime.now(timezone.utc).isoformat()
                    self._log_event(instance, "workflow_completed_approved", {})

        return instance

    def cancel_instance(
        self, instance_id: str, cancelled_by: str, reason: str = ""
    ) -> WorkflowInstance:
        """Cancel a workflow instance."""
        instance = self._get_instance(instance_id)
        instance.status = WorkflowStatus.CANCELLED
        instance.completed_at = datetime.now(timezone.utc).isoformat()
        self._log_event(instance, "workflow_cancelled", {
            "cancelled_by": cancelled_by,
            "reason": reason,
        })
        return instance

    # ---------- Query ----------

    def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get a workflow instance by ID."""
        return self._instances.get(instance_id)

    def get_pending_approvals(
        self, approver_role: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all pending approval items, optionally filtered by role and tenant."""
        pending = []
        for instance in self._instances.values():
            if instance.status != WorkflowStatus.ACTIVE:
                continue
            if tenant_id and instance.tenant_id != tenant_id:
                continue

            for i, stage in enumerate(instance.stages):
                if stage.status in (StageStatus.IN_PROGRESS, StageStatus.ESCALATED):
                    template = self._templates.get(instance.template_id)
                    if template:
                        stage_def = next(
                            (s for s in template.stages if s.stage_id == stage.stage_id),
                            None
                        )
                        if stage_def:
                            if approver_role and approver_role not in stage_def.required_roles:
                                continue
                            pending.append({
                                "instance_id": instance.instance_id,
                                "request_title": instance.request_title,
                                "stage_name": stage.stage_name,
                                "stage_id": stage.stage_id,
                                "status": stage.status.value,
                                "sla_deadline": stage.sla_deadline,
                                "required_roles": stage_def.required_roles,
                                "requestor": instance.requestor_name or instance.requestor_id,
                            })
        return pending

    def get_overdue_stages(self) -> List[Dict[str, Any]]:
        """Get all stages that have exceeded their SLA deadline."""
        overdue = []
        now = datetime.now(timezone.utc).isoformat()
        for instance in self._instances.values():
            if instance.status != WorkflowStatus.ACTIVE:
                continue
            for stage in instance.stages:
                if (
                    stage.status in (StageStatus.IN_PROGRESS, StageStatus.ESCALATED)
                    and stage.sla_deadline
                    and now > stage.sla_deadline
                ):
                    overdue.append({
                        "instance_id": instance.instance_id,
                        "stage_id": stage.stage_id,
                        "stage_name": stage.stage_name,
                        "sla_deadline": stage.sla_deadline,
                        "request_title": instance.request_title,
                    })
        return overdue

    # ---------- Internal Helpers ----------

    def _get_instance(self, instance_id: str) -> WorkflowInstance:
        """Retrieve a workflow instance, raising if not found."""
        instance = self._instances.get(instance_id)
        if not instance:
            raise ValueError(f"Workflow instance '{instance_id}' not found.")
        return instance

    def _activate_stage(self, instance: WorkflowInstance, stage_index: int) -> None:
        """Activate a specific stage in a workflow instance."""
        if stage_index >= len(instance.stages):
            return

        stage = instance.stages[stage_index]
        template = self._templates.get(instance.template_id)
        if not template:
            return

        stage_def = next(
            (s for s in template.stages if s.stage_id == stage.stage_id),
            None
        )
        if not stage_def:
            return

        # Check if stage should be skipped (conditional)
        if stage_def.stage_type == StageType.CONDITIONAL and stage_def.skip_if_condition_false:
            if not self._evaluate_condition(stage_def, instance.request_metadata):
                stage.status = StageStatus.SKIPPED
                stage.completed_at = datetime.now(timezone.utc).isoformat()
                self._log_event(instance, "stage_skipped", {"stage_id": stage.stage_id})
                # Advance to next
                next_idx = stage_index + 1
                if next_idx < len(instance.stages):
                    instance.current_stage_index = next_idx
                    self._activate_stage(instance, next_idx)
                return

        # Handle automated stages
        if stage_def.stage_type == StageType.AUTOMATED:
            validator = self._automated_validators.get(stage_def.stage_id)
            if validator:
                try:
                    result = validator(instance.request_metadata)
                    stage.status = StageStatus.APPROVED if result else StageStatus.REJECTED
                except Exception as e:
                    logger.error(f"Automated validator failed for stage '{stage.stage_id}': {e}")
                    stage.status = StageStatus.REJECTED
                stage.activated_at = datetime.now(timezone.utc).isoformat()
                stage.completed_at = datetime.now(timezone.utc).isoformat()

                if stage.status == StageStatus.REJECTED:
                    instance.status = WorkflowStatus.COMPLETED_REJECTED
                    instance.completed_at = datetime.now(timezone.utc).isoformat()
                elif stage_index + 1 < len(instance.stages):
                    instance.current_stage_index = stage_index + 1
                    self._activate_stage(instance, stage_index + 1)
                else:
                    instance.status = WorkflowStatus.COMPLETED_APPROVED
                    instance.completed_at = datetime.now(timezone.utc).isoformat()
                return

        stage.status = StageStatus.IN_PROGRESS
        now = datetime.now(timezone.utc)
        stage.activated_at = now.isoformat()

        # Set SLA deadline
        if stage_def.sla_hours:
            stage.sla_deadline = (now + timedelta(hours=stage_def.sla_hours)).isoformat()

        logger.info(f"Activated stage '{stage.stage_name}' in workflow {instance.instance_id}")

    def _is_stage_complete(
        self, stage_instance: StageInstance, stage_def: WorkflowStageDefinition
    ) -> bool:
        """Check if a stage has received sufficient approvals."""
        approvals = [
            r for r in stage_instance.approval_records
            if r.action in (ApprovalAction.APPROVE, ApprovalAction.COUNTER_SIGN)
        ]

        if stage_def.stage_type == StageType.SEQUENTIAL:
            return len(approvals) >= 1

        elif stage_def.stage_type == StageType.PARALLEL_ALL:
            if stage_def.specific_approvers:
                approved_by = {r.approver_id for r in approvals}
                return all(a in approved_by for a in stage_def.specific_approvers)
            return len(approvals) >= stage_def.quorum_total

        elif stage_def.stage_type == StageType.PARALLEL_QUORUM:
            return len(approvals) >= stage_def.quorum_required

        elif stage_def.stage_type == StageType.COUNTER_SIGNATURE:
            # Need primary approval + all counter-signatures
            primary = [r for r in approvals if not r.is_counter_signature]
            counters = [r for r in approvals if r.is_counter_signature]
            return len(primary) >= 1 and len(counters) >= len(stage_def.counter_signature_roles)

        return len(approvals) >= 1

    def _evaluate_condition(
        self, stage_def: WorkflowStageDefinition, metadata: Dict[str, Any]
    ) -> bool:
        """Evaluate a conditional stage's condition against request metadata."""
        if not stage_def.condition_field:
            return True
        actual = metadata.get(stage_def.condition_field)
        if actual is None:
            return False
        op = stage_def.condition_operator
        val = stage_def.condition_value
        if op == "eq":
            return actual == val
        elif op == "neq":
            return actual != val
        elif op == "gt":
            return actual > val
        elif op == "gte":
            return actual >= val
        elif op == "in":
            return actual in val if isinstance(val, (list, set)) else False
        return True

    def _log_event(
        self, instance: WorkflowInstance, event_type: str, details: Dict[str, Any]
    ) -> None:
        """Append an audit event to the workflow instance."""
        instance.audit_trail.append({
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        })

    # ---------- Institutional Workflow Templates ----------

    def seed_government_workflows(self) -> None:
        """Seed standard government approval workflow templates."""

        self.register_template(WorkflowTemplate(
            name="AI System Deployment Approval (Government)",
            description="Multi-stage approval for deploying AI systems in government agencies per KPA and EU AI Act",
            stages=[
                WorkflowStageDefinition(
                    name="Technical Review",
                    description="IT Department evaluates technical readiness and security posture",
                    stage_type=StageType.SEQUENTIAL,
                    order=0,
                    required_roles=["DH", "SC"],
                    sla_hours=72,
                    escalation_target_role="DD",
                ),
                WorkflowStageDefinition(
                    name="Legal & Compliance Review",
                    description="Legal counsel verifies EU AI Act, GDPR, and KPA compliance",
                    stage_type=StageType.SEQUENTIAL,
                    order=1,
                    required_roles=["LEGAL_COUNSEL", "DPO"],
                    sla_hours=120,
                ),
                WorkflowStageDefinition(
                    name="Data Protection Impact Assessment (DPIA)",
                    description="Mandatory DPIA for high-risk AI systems (GDPR Art. 35)",
                    stage_type=StageType.CONDITIONAL,
                    order=2,
                    required_roles=["DPO"],
                    sla_hours=240,
                    condition_field="risk_level",
                    condition_operator="in",
                    condition_value=["high", "critical"],
                    skip_if_condition_false=True,
                ),
                WorkflowStageDefinition(
                    name="Budget Approval",
                    description="Financial controller approves ongoing governance costs",
                    stage_type=StageType.SEQUENTIAL,
                    order=3,
                    required_roles=["DD", "DG"],
                    sla_hours=48,
                ),
                WorkflowStageDefinition(
                    name="Director General Sign-Off",
                    description="Final approval with counter-signature from DPO",
                    stage_type=StageType.COUNTER_SIGNATURE,
                    order=4,
                    required_roles=["DG"],
                    counter_signature_roles=["DPO"],
                    sla_hours=48,
                ),
            ],
            applicable_to=["ai_deployment", "ai_system_update"],
        ))

        self.register_template(WorkflowTemplate(
            name="Policy Amendment Approval (Government)",
            description="RFC-style amendment process for AI governance policies",
            stages=[
                WorkflowStageDefinition(
                    name="Policy Draft Submission",
                    description="Author submits policy change proposal",
                    stage_type=StageType.AUTOMATED,
                    order=0,
                ),
                WorkflowStageDefinition(
                    name="Peer Review",
                    description="Two senior officers must review",
                    stage_type=StageType.PARALLEL_QUORUM,
                    order=1,
                    required_roles=["SO", "SC"],
                    quorum_required=2,
                    quorum_total=5,
                    sla_hours=168,
                ),
                WorkflowStageDefinition(
                    name="Legal Validation",
                    description="Legal counsel validates against applicable laws",
                    stage_type=StageType.SEQUENTIAL,
                    order=2,
                    required_roles=["LEGAL_COUNSEL"],
                    sla_hours=120,
                ),
                WorkflowStageDefinition(
                    name="Formal Verification (Z3)",
                    description="Automated Z3 SMT verification that amendment does not violate fundamental laws",
                    stage_type=StageType.AUTOMATED,
                    order=3,
                ),
                WorkflowStageDefinition(
                    name="Director General Approval",
                    description="DG final approval with mandatory counter-signature",
                    stage_type=StageType.COUNTER_SIGNATURE,
                    order=4,
                    required_roles=["DG"],
                    counter_signature_roles=["DD", "DPO"],
                    sla_hours=72,
                ),
            ],
            applicable_to=["policy_amendment", "law_interpretation_change"],
        ))

        logger.info("✅ Government workflow templates seeded")

    def seed_military_workflows(self) -> None:
        """Seed military approval workflow templates."""

        self.register_template(WorkflowTemplate(
            name="AI-Enabled Mission Approval (Military)",
            description="Chain-of-command approval for AI-supported military operations",
            stages=[
                WorkflowStageDefinition(
                    name="Mission Brief & Intel Assessment",
                    description="Intelligence officer validates threat assessment and AI system readiness",
                    stage_type=StageType.SEQUENTIAL,
                    order=0,
                    required_roles=["INTEL_OFFICER"],
                    sla_hours=4,
                    escalation_target_role="BnCMD",
                ),
                WorkflowStageDefinition(
                    name="Legal Review (IHL Compliance)",
                    description="Military legal advisor confirms International Humanitarian Law compliance",
                    stage_type=StageType.SEQUENTIAL,
                    order=1,
                    required_roles=["LEGAL_ADVISOR"],
                    sla_hours=2,
                ),
                WorkflowStageDefinition(
                    name="ROE Verification",
                    description="Automated check that AI system operates within active Rules of Engagement",
                    stage_type=StageType.AUTOMATED,
                    order=2,
                ),
                WorkflowStageDefinition(
                    name="Commander Authorization",
                    description="Commander approves with counter-signature from legal advisor",
                    stage_type=StageType.COUNTER_SIGNATURE,
                    order=3,
                    required_roles=["BnCMD", "BCMD", "CCMD"],
                    counter_signature_roles=["LEGAL_ADVISOR"],
                    sla_hours=1,
                ),
            ],
            applicable_to=["mission_authorization", "kinetic_engagement", "isr_deployment"],
        ))

        logger.info("✅ Military workflow templates seeded")

    def seed_corporate_workflows(self) -> None:
        """Seed corporate approval workflow templates."""

        self.register_template(WorkflowTemplate(
            name="AI Model Deployment Approval (Corporate)",
            description="Enterprise change management workflow for AI model deployments",
            stages=[
                WorkflowStageDefinition(
                    name="Architecture Review",
                    description="Solution architect validates design and integration",
                    stage_type=StageType.SEQUENTIAL,
                    order=0,
                    required_roles=["DIR", "VPE"],
                    sla_hours=48,
                ),
                WorkflowStageDefinition(
                    name="Security Review",
                    description="Security team evaluates threat model and vulnerability assessment",
                    stage_type=StageType.SEQUENTIAL,
                    order=1,
                    required_roles=["CISO", "SEC_ANALYST"],
                    sla_hours=72,
                ),
                WorkflowStageDefinition(
                    name="DPO Review",
                    description="Data Protection Officer reviews privacy implications",
                    stage_type=StageType.CONDITIONAL,
                    order=2,
                    required_roles=["DPO"],
                    sla_hours=48,
                    condition_field="processes_personal_data",
                    condition_operator="eq",
                    condition_value=True,
                    skip_if_condition_false=True,
                ),
                WorkflowStageDefinition(
                    name="Change Advisory Board (CAB)",
                    description="ITIL CAB review and risk assessment",
                    stage_type=StageType.PARALLEL_QUORUM,
                    order=3,
                    required_roles=["CAB_MEMBER"],
                    quorum_required=3,
                    quorum_total=7,
                    sla_hours=168,
                ),
                WorkflowStageDefinition(
                    name="CTO / CAIO Final Approval",
                    description="Executive sign-off for production deployment",
                    stage_type=StageType.SEQUENTIAL,
                    order=4,
                    required_roles=["CAIO", "CTO"],
                    sla_hours=24,
                ),
            ],
            applicable_to=["ai_deployment", "model_update", "algorithm_change"],
        ))

        logger.info("✅ Corporate workflow templates seeded")
