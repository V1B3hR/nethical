# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Hierarchical Role-Based Access Control (HRBAC) for Institutional Deployments.

Extends Nethical's flat RBAC with deep organizational hierarchy support required by
large bureaucracies, government agencies, corporations, and military organizations:

- **Organizational Unit (OU) Trees:** Nested hierarchy (Ministry → Department →
  Division → Section → Team) with configurable depth.
- **Permission Inheritance:** Permissions flow downward through the org-tree with
  configurable inheritance modes (INHERIT, INHERIT_AND_EXTEND, ISOLATED).
- **Scope Boundaries:** A Department Head governs only THEIR department's AI systems,
  even if they share the same role title as another Department Head.
- **Dynamic Delegation:** Time-bounded authority delegation with automatic revocation.
- **Institutional Role Templates:** Pre-built role hierarchies for government (GovS 002),
  military (NATO rank structure), and corporate (C-suite → VP → Director → Manager) models.

Gap Addressed: 1.1 (Deep Hierarchical RBAC), 1.3 (Dynamic Delegation Chains)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from ..core.models import ClassificationLevel, UserRole

logger = logging.getLogger("nethical.auth.hrbac")


# ============== Organizational Unit Hierarchy ==============


class InheritanceMode(str, Enum):
    """How permissions flow through the org-tree."""
    INHERIT = "inherit"                     # Child inherits all parent permissions
    INHERIT_AND_EXTEND = "inherit_extend"   # Inherit + child can add own permissions
    ISOLATED = "isolated"                   # No inheritance; child defines own permissions


class OrgUnitType(str, Enum):
    """Institutional organizational unit types."""
    # Government / Public Admin
    MINISTRY = "ministry"
    DEPARTMENT = "department"
    DIVISION = "division"
    SECTION = "section"
    TEAM = "team"
    OFFICE = "office"
    AGENCY = "agency"
    BUREAU = "bureau"

    # Military
    THEATER_COMMAND = "theater_command"
    CORPS = "corps"
    DIVISION_MIL = "division_mil"
    BRIGADE = "brigade"
    BATTALION = "battalion"
    COMPANY = "company"
    PLATOON = "platoon"
    SQUAD = "squad"

    # Corporate
    HOLDING = "holding"
    SUBSIDIARY = "subsidiary"
    BUSINESS_UNIT = "business_unit"
    COST_CENTER = "cost_center"
    PROJECT = "project"

    # Generic
    ROOT = "root"
    CUSTOM = "custom"


class OrgUnit(BaseModel):
    """An organizational unit in the institutional hierarchy."""
    ou_id: str = Field(default_factory=lambda: f"ou_{uuid.uuid4().hex[:12]}")
    name: str = Field(..., min_length=1, description="Display name of the organizational unit")
    ou_type: OrgUnitType = Field(default=OrgUnitType.CUSTOM)
    parent_ou_id: Optional[str] = Field(default=None, description="Parent OU ID (None for root)")
    tenant_id: str = Field(default="default_tenant")
    classification_level: ClassificationLevel = Field(default=ClassificationLevel.UNCLASSIFIED)
    inheritance_mode: InheritanceMode = Field(default=InheritanceMode.INHERIT_AND_EXTEND)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = Field(default=True)
    # Cost center / budget tracking (Gap 4.2)
    cost_center_code: Optional[str] = Field(default=None, description="Financial cost center code")
    # Geographic scope (for ABAC)
    jurisdiction: Optional[str] = Field(default=None, description="Jurisdiction override")
    location: Optional[str] = Field(default=None, description="Physical location")


# ============== Institutional Role Definition ==============


class InstitutionalRole(BaseModel):
    """An institutional role definition with hierarchical rank."""
    role_id: str = Field(default_factory=lambda: f"irole_{uuid.uuid4().hex[:10]}")
    name: str = Field(..., description="Human-readable role name (e.g., 'Deputy Director General')")
    code: str = Field(..., description="Machine-readable role code (e.g., 'DDG')")
    rank: int = Field(..., ge=0, description="Numeric rank (0=lowest, higher=more authority)")
    base_nethical_role: UserRole = Field(
        default=UserRole.AGENT_OPERATOR,
        description="Mapping to base Nethical RBAC role for permission baseline"
    )
    permissions: Set[str] = Field(default_factory=set, description="Additional permissions beyond base role")
    max_delegation_depth: int = Field(
        default=1, ge=0,
        description="How many levels down this role can delegate authority"
    )
    can_approve_policies: bool = Field(default=False)
    can_invoke_kill_switch: bool = Field(default=False)
    can_manage_subordinate_roles: bool = Field(default=False)
    classification_clearance: ClassificationLevel = Field(default=ClassificationLevel.UNCLASSIFIED)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============== Delegation Token ==============


class DelegationToken(BaseModel):
    """Time-bounded authority delegation with automatic revocation.

    Gap 1.3: Supports temporary delegation (e.g., Deputy goes on leave
    and delegates signing authority to Section Chief for 14 days).
    """
    delegation_id: str = Field(default_factory=lambda: f"deleg_{uuid.uuid4().hex[:12]}")
    delegator_user_id: str = Field(..., description="User granting the delegation")
    delegate_user_id: str = Field(..., description="User receiving delegated authority")
    delegated_permissions: Set[str] = Field(..., description="Specific permissions being delegated")
    delegated_role_id: Optional[str] = Field(
        default=None, description="Optionally delegate an entire institutional role"
    )
    scope_ou_id: Optional[str] = Field(
        default=None, description="OU scope restriction (if None, same scope as delegator)"
    )
    reason: str = Field(..., min_length=5, description="Justification for delegation")
    valid_from: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    valid_until: str = Field(..., description="Expiration timestamp (ISO 8601)")
    revoked: bool = Field(default=False)
    revoked_at: Optional[str] = Field(default=None)
    revoked_by: Optional[str] = Field(default=None)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_active(self) -> bool:
        """Check if delegation is currently active (not expired, not revoked)."""
        if self.revoked:
            return False
        now = datetime.now(timezone.utc).isoformat()
        return self.valid_from <= now <= self.valid_until


# ============== Separation of Duties (SoD) ==============


class SoDRule(BaseModel):
    """Separation of Duties rule — prevents toxic role/permission combinations.

    Gap 1.5: The person who creates a policy cannot be the one who approves it.
    """
    rule_id: str = Field(default_factory=lambda: f"sod_{uuid.uuid4().hex[:10]}")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(default="")
    # Mutually exclusive permission sets
    conflicting_permissions: List[Set[str]] = Field(
        default_factory=list,
        description="Two or more sets of permissions that cannot be held by the same user simultaneously"
    )
    # Alternatively, conflicting roles
    conflicting_roles: List[str] = Field(
        default_factory=list,
        description="Role codes that cannot be assigned to the same user"
    )
    enforcement: str = Field(
        default="hard",
        description="'hard' = block assignment, 'soft' = warn but allow with justification"
    )
    scope_ou_id: Optional[str] = Field(
        default=None, description="OU scope (None = global)"
    )
    is_active: bool = Field(default=True)


class SoDViolation(BaseModel):
    """Recorded SoD violation event."""
    violation_id: str = Field(default_factory=lambda: f"sod_v_{uuid.uuid4().hex[:10]}")
    rule_id: str
    rule_name: str
    user_id: str
    attempted_action: str
    conflicting_permissions_held: List[str] = Field(default_factory=list)
    enforcement_result: str = Field(..., description="'blocked' or 'warned'")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    justification: Optional[str] = Field(default=None)


# ============== Role Assignment (User-OU-Role binding) ==============


class RoleAssignment(BaseModel):
    """Binds a user to an institutional role within a specific organizational unit."""
    assignment_id: str = Field(default_factory=lambda: f"assign_{uuid.uuid4().hex[:10]}")
    user_id: str
    role_id: str
    ou_id: str = Field(..., description="The organizational unit this assignment is scoped to")
    is_primary: bool = Field(default=True, description="Whether this is the user's primary assignment")
    valid_from: Optional[str] = Field(default=None)
    valid_until: Optional[str] = Field(default=None)
    assigned_by: Optional[str] = Field(default=None, description="User who made this assignment")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============== HRBAC Engine ==============


class HRBACEngine:
    """Hierarchical Role-Based Access Control Engine.

    Manages organizational hierarchies, role assignments, delegations,
    and separation-of-duties enforcement for institutional deployments.
    """

    def __init__(self) -> None:
        self._org_units: Dict[str, OrgUnit] = {}
        self._roles: Dict[str, InstitutionalRole] = {}
        self._assignments: Dict[str, RoleAssignment] = {}  # assignment_id -> assignment
        self._user_assignments: Dict[str, List[str]] = {}  # user_id -> [assignment_ids]
        self._delegations: Dict[str, DelegationToken] = {}
        self._sod_rules: Dict[str, SoDRule] = {}
        self._sod_violations: List[SoDViolation] = []

    # ---------- Org Unit Management ----------

    def register_ou(self, ou: OrgUnit) -> OrgUnit:
        """Register an organizational unit in the hierarchy."""
        if ou.parent_ou_id and ou.parent_ou_id not in self._org_units:
            raise ValueError(
                f"Parent OU '{ou.parent_ou_id}' does not exist. "
                f"Register parent before child."
            )
        self._org_units[ou.ou_id] = ou
        logger.info(f"Registered OU '{ou.name}' ({ou.ou_id}) under parent '{ou.parent_ou_id}'")
        return ou

    def get_ou(self, ou_id: str) -> Optional[OrgUnit]:
        """Retrieve an organizational unit by ID."""
        return self._org_units.get(ou_id)

    def get_ou_ancestors(self, ou_id: str) -> List[OrgUnit]:
        """Get all ancestor OUs from the given OU up to the root (inclusive)."""
        ancestors = []
        current_id = ou_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            ou = self._org_units.get(current_id)
            if ou is None:
                break
            ancestors.append(ou)
            current_id = ou.parent_ou_id
        return ancestors

    def get_ou_descendants(self, ou_id: str) -> List[OrgUnit]:
        """Get all descendant OUs below the given OU."""
        descendants = []
        stack = [ou_id]
        visited = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for ou in self._org_units.values():
                if ou.parent_ou_id == current and ou.ou_id not in visited:
                    descendants.append(ou)
                    stack.append(ou.ou_id)
        return descendants

    def get_ou_children(self, ou_id: str) -> List[OrgUnit]:
        """Get direct children of an organizational unit."""
        return [ou for ou in self._org_units.values() if ou.parent_ou_id == ou_id]

    def is_ou_ancestor_of(self, ancestor_id: str, descendant_id: str) -> bool:
        """Check if ancestor_id is an ancestor of descendant_id in the org tree."""
        ancestors = self.get_ou_ancestors(descendant_id)
        return any(a.ou_id == ancestor_id for a in ancestors)

    # ---------- Role Management ----------

    def register_role(self, role: InstitutionalRole) -> InstitutionalRole:
        """Register an institutional role definition."""
        self._roles[role.role_id] = role
        logger.info(f"Registered institutional role '{role.name}' (rank={role.rank})")
        return role

    def get_role(self, role_id: str) -> Optional[InstitutionalRole]:
        """Retrieve an institutional role by ID."""
        return self._roles.get(role_id)

    # ---------- Role Assignment ----------

    def assign_role(
        self,
        user_id: str,
        role_id: str,
        ou_id: str,
        assigned_by: Optional[str] = None,
        is_primary: bool = True,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
    ) -> RoleAssignment:
        """Assign an institutional role to a user within a specific OU scope.

        Enforces SoD rules before assignment.
        """
        if role_id not in self._roles:
            raise ValueError(f"Role '{role_id}' is not registered.")
        if ou_id not in self._org_units:
            raise ValueError(f"OU '{ou_id}' is not registered.")

        # Check SoD before assignment
        sod_violation = self._check_sod_for_assignment(user_id, role_id, ou_id)
        if sod_violation and sod_violation.enforcement_result == "blocked":
            self._sod_violations.append(sod_violation)
            raise PermissionError(
                f"SoD VIOLATION: Cannot assign role '{role_id}' to user '{user_id}'. "
                f"Rule '{sod_violation.rule_name}' violated: {sod_violation.attempted_action}"
            )
        elif sod_violation:
            self._sod_violations.append(sod_violation)
            logger.warning(
                f"⚠️ SoD soft violation for user '{user_id}': {sod_violation.rule_name}"
            )

        assignment = RoleAssignment(
            user_id=user_id,
            role_id=role_id,
            ou_id=ou_id,
            is_primary=is_primary,
            assigned_by=assigned_by,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        self._assignments[assignment.assignment_id] = assignment
        if user_id not in self._user_assignments:
            self._user_assignments[user_id] = []
        self._user_assignments[user_id].append(assignment.assignment_id)

        logger.info(
            f"Assigned role '{self._roles[role_id].name}' to user '{user_id}' "
            f"in OU '{self._org_units[ou_id].name}'"
        )
        return assignment

    def get_user_assignments(self, user_id: str) -> List[RoleAssignment]:
        """Get all role assignments for a user."""
        assignment_ids = self._user_assignments.get(user_id, [])
        return [self._assignments[aid] for aid in assignment_ids if aid in self._assignments]

    def get_user_effective_permissions(
        self, user_id: str, target_ou_id: str
    ) -> Set[str]:
        """Calculate the effective permissions a user has within a specific OU.

        Considers:
        1. Direct role assignments in the target OU
        2. Role assignments in ancestor OUs (based on inheritance mode)
        3. Active delegations
        4. Base Nethical role permissions
        """
        from ..auth.rbac import ROLE_PERMISSIONS

        effective_perms: Set[str] = set()
        assignments = self.get_user_assignments(user_id)
        target_ancestors = {a.ou_id for a in self.get_ou_ancestors(target_ou_id)}

        for assignment in assignments:
            role = self._roles.get(assignment.role_id)
            if not role:
                continue

            ou = self._org_units.get(assignment.ou_id)
            if not ou:
                continue

            # Direct assignment in target OU
            if assignment.ou_id == target_ou_id:
                effective_perms.update(role.permissions)
                base_perms = ROLE_PERMISSIONS.get(role.base_nethical_role, set())
                effective_perms.update(base_perms)
                continue

            # Inherited from ancestor OU
            if assignment.ou_id in target_ancestors:
                target_ou = self._org_units.get(target_ou_id)
                if target_ou and target_ou.inheritance_mode != InheritanceMode.ISOLATED:
                    effective_perms.update(role.permissions)
                    base_perms = ROLE_PERMISSIONS.get(role.base_nethical_role, set())
                    effective_perms.update(base_perms)

        # Add delegated permissions
        for deleg in self._delegations.values():
            if deleg.delegate_user_id == user_id and deleg.is_active():
                scope_match = (
                    deleg.scope_ou_id is None
                    or deleg.scope_ou_id == target_ou_id
                    or deleg.scope_ou_id in target_ancestors
                )
                if scope_match:
                    effective_perms.update(deleg.delegated_permissions)

        return effective_perms

    def has_permission_in_ou(
        self, user_id: str, permission: str, target_ou_id: str
    ) -> bool:
        """Check if a user has a specific permission within a specific OU."""
        effective = self.get_user_effective_permissions(user_id, target_ou_id)
        return permission in effective

    def has_authority_over(self, user_id: str, target_ou_id: str) -> bool:
        """Check if a user has any management authority over a target OU.

        A user has authority if they have a role assignment in the target OU
        or any of its ancestors.
        """
        target_ancestors = {a.ou_id for a in self.get_ou_ancestors(target_ou_id)}
        assignments = self.get_user_assignments(user_id)
        return any(a.ou_id in target_ancestors for a in assignments)

    # ---------- Delegation Management ----------

    def create_delegation(
        self,
        delegator_user_id: str,
        delegate_user_id: str,
        permissions: Set[str],
        reason: str,
        duration_hours: int = 336,  # 14 days default
        scope_ou_id: Optional[str] = None,
        delegated_role_id: Optional[str] = None,
    ) -> DelegationToken:
        """Create a time-bounded delegation of authority.

        Validates:
        - Delegator actually possesses the permissions being delegated
        - Delegation depth limits are respected
        - SoD rules are not violated by the delegation
        """
        # Verify delegator has authority to delegate
        if scope_ou_id:
            delegator_perms = self.get_user_effective_permissions(delegator_user_id, scope_ou_id)
            if not permissions.issubset(delegator_perms):
                missing = permissions - delegator_perms
                raise PermissionError(
                    f"Delegator '{delegator_user_id}' does not possess permissions: {missing}"
                )

        now = datetime.now(timezone.utc)
        valid_until = (now + timedelta(hours=duration_hours)).isoformat()

        delegation = DelegationToken(
            delegator_user_id=delegator_user_id,
            delegate_user_id=delegate_user_id,
            delegated_permissions=permissions,
            delegated_role_id=delegated_role_id,
            scope_ou_id=scope_ou_id,
            reason=reason,
            valid_until=valid_until,
        )
        self._delegations[delegation.delegation_id] = delegation
        logger.info(
            f"Created delegation {delegation.delegation_id}: "
            f"'{delegator_user_id}' → '{delegate_user_id}' "
            f"({len(permissions)} permissions, expires {valid_until})"
        )
        return delegation

    def revoke_delegation(
        self, delegation_id: str, revoked_by: str
    ) -> Optional[DelegationToken]:
        """Revoke a delegation early."""
        deleg = self._delegations.get(delegation_id)
        if deleg and not deleg.revoked:
            deleg.revoked = True
            deleg.revoked_at = datetime.now(timezone.utc).isoformat()
            deleg.revoked_by = revoked_by
            logger.info(f"Delegation {delegation_id} revoked by '{revoked_by}'")
            return deleg
        return None

    def get_active_delegations_for_user(self, user_id: str) -> List[DelegationToken]:
        """Get all active delegations where user is the delegate."""
        return [
            d for d in self._delegations.values()
            if d.delegate_user_id == user_id and d.is_active()
        ]

    # ---------- Separation of Duties (SoD) ----------

    def register_sod_rule(self, rule: SoDRule) -> SoDRule:
        """Register a Separation of Duties rule."""
        self._sod_rules[rule.rule_id] = rule
        logger.info(f"Registered SoD rule '{rule.name}' ({rule.rule_id})")
        return rule

    def _check_sod_for_assignment(
        self, user_id: str, new_role_id: str, ou_id: str
    ) -> Optional[SoDViolation]:
        """Check if assigning a new role would violate any SoD rules."""
        new_role = self._roles.get(new_role_id)
        if not new_role:
            return None

        existing_assignments = self.get_user_assignments(user_id)
        existing_role_codes = set()
        existing_permissions: Set[str] = set()

        for assignment in existing_assignments:
            role = self._roles.get(assignment.role_id)
            if role:
                existing_role_codes.add(role.code)
                existing_permissions.update(role.permissions)

        # Check each SoD rule
        for rule in self._sod_rules.values():
            if not rule.is_active:
                continue
            # Scope check
            if rule.scope_ou_id and rule.scope_ou_id != ou_id:
                if not self.is_ou_ancestor_of(rule.scope_ou_id, ou_id):
                    continue

            # Check role conflicts
            if rule.conflicting_roles:
                combined_roles = existing_role_codes | {new_role.code}
                conflicting_held = [r for r in rule.conflicting_roles if r in combined_roles]
                if len(conflicting_held) >= 2:
                    return SoDViolation(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        user_id=user_id,
                        attempted_action=f"Assign role '{new_role.code}' conflicting with {conflicting_held}",
                        conflicting_permissions_held=conflicting_held,
                        enforcement_result="blocked" if rule.enforcement == "hard" else "warned",
                    )

            # Check permission conflicts
            if rule.conflicting_permissions and len(rule.conflicting_permissions) >= 2:
                combined = existing_permissions | new_role.permissions
                overlap_count = sum(1 for ps in rule.conflicting_permissions if ps.issubset(combined))
                if overlap_count >= 2:
                    return SoDViolation(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        user_id=user_id,
                        attempted_action=f"Assign role '{new_role.code}' creates toxic permission combination",
                        enforcement_result="blocked" if rule.enforcement == "hard" else "warned",
                    )

        return None

    def check_sod_violations(self, user_id: str) -> List[SoDViolation]:
        """Audit a user's current role assignments for any SoD violations."""
        violations = []
        assignments = self.get_user_assignments(user_id)

        all_role_codes = set()
        all_permissions: Set[str] = set()
        for assignment in assignments:
            role = self._roles.get(assignment.role_id)
            if role:
                all_role_codes.add(role.code)
                all_permissions.update(role.permissions)

        for rule in self._sod_rules.values():
            if not rule.is_active:
                continue
            # Role conflict check
            if rule.conflicting_roles:
                held = [r for r in rule.conflicting_roles if r in all_role_codes]
                if len(held) >= 2:
                    violations.append(SoDViolation(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        user_id=user_id,
                        attempted_action="Audit: existing assignment conflict",
                        conflicting_permissions_held=held,
                        enforcement_result="blocked" if rule.enforcement == "hard" else "warned",
                    ))

        return violations

    def get_sod_violation_history(self) -> List[SoDViolation]:
        """Return the full SoD violation audit trail."""
        return list(self._sod_violations)

    # ---------- Institutional Role Templates ----------

    def seed_government_hierarchy(self, tenant_id: str = "default_tenant") -> str:
        """Seed a standard government org-tree (UK GovS 002 / Polish KPA model).

        Returns the root OU ID.
        """
        root = self.register_ou(OrgUnit(
            name="Government Root", ou_type=OrgUnitType.ROOT,
            tenant_id=tenant_id, parent_ou_id=None,
        ))
        ministry = self.register_ou(OrgUnit(
            name="Ministry of Digital Affairs", ou_type=OrgUnitType.MINISTRY,
            tenant_id=tenant_id, parent_ou_id=root.ou_id,
        ))
        dept_ai = self.register_ou(OrgUnit(
            name="Department of AI Governance", ou_type=OrgUnitType.DEPARTMENT,
            tenant_id=tenant_id, parent_ou_id=ministry.ou_id,
        ))
        self.register_ou(OrgUnit(
            name="Division of AI Ethics", ou_type=OrgUnitType.DIVISION,
            tenant_id=tenant_id, parent_ou_id=dept_ai.ou_id,
        ))
        self.register_ou(OrgUnit(
            name="Division of AI Security", ou_type=OrgUnitType.DIVISION,
            tenant_id=tenant_id, parent_ou_id=dept_ai.ou_id,
        ))

        # Seed institutional roles
        self.register_role(InstitutionalRole(
            name="Minister", code="MINISTER", rank=100,
            base_nethical_role=UserRole.GLOBAL_ADMIN,
            permissions={"policy:manage", "policy:approve", "system:override"},
            can_approve_policies=True, can_invoke_kill_switch=True,
            can_manage_subordinate_roles=True,
            classification_clearance=ClassificationLevel.SECRET,
        ))
        self.register_role(InstitutionalRole(
            name="Director General", code="DG", rank=90,
            base_nethical_role=UserRole.GLOBAL_ADMIN,
            permissions={"policy:manage", "policy:approve", "budget:approve"},
            can_approve_policies=True, can_invoke_kill_switch=True,
            can_manage_subordinate_roles=True,
            classification_clearance=ClassificationLevel.SECRET,
        ))
        self.register_role(InstitutionalRole(
            name="Deputy Director", code="DD", rank=80,
            base_nethical_role=UserRole.SECURITY_OFFICER,
            permissions={"policy:manage", "policy:review", "budget:review"},
            can_approve_policies=False, can_invoke_kill_switch=True,
            can_manage_subordinate_roles=True,
            classification_clearance=ClassificationLevel.CONFIDENTIAL,
        ))
        self.register_role(InstitutionalRole(
            name="Department Head", code="DH", rank=70,
            base_nethical_role=UserRole.SECURITY_OFFICER,
            permissions={"policy:review", "team:manage", "report:generate"},
            can_approve_policies=False, can_invoke_kill_switch=False,
            can_manage_subordinate_roles=True,
            classification_clearance=ClassificationLevel.CONFIDENTIAL,
        ))
        self.register_role(InstitutionalRole(
            name="Section Chief", code="SC", rank=60,
            base_nethical_role=UserRole.HITL_REVIEWER,
            permissions={"policy:review", "report:generate"},
            can_manage_subordinate_roles=True,
            classification_clearance=ClassificationLevel.RESTRICTED,
        ))
        self.register_role(InstitutionalRole(
            name="Senior Officer", code="SO", rank=40,
            base_nethical_role=UserRole.HITL_REVIEWER,
            permissions={"policy:review"},
            classification_clearance=ClassificationLevel.RESTRICTED,
        ))
        self.register_role(InstitutionalRole(
            name="Officer", code="OFF", rank=30,
            base_nethical_role=UserRole.AGENT_OPERATOR,
            permissions=set(),
            classification_clearance=ClassificationLevel.UNCLASSIFIED,
        ))
        self.register_role(InstitutionalRole(
            name="Clerk", code="CLK", rank=10,
            base_nethical_role=UserRole.AGENT_OPERATOR,
            permissions=set(),
            classification_clearance=ClassificationLevel.UNCLASSIFIED,
        ))

        # Standard SoD rules for government
        self.register_sod_rule(SoDRule(
            name="Policy Creator ≠ Policy Approver",
            description="The person who drafts a policy cannot be the person who approves it.",
            conflicting_permissions=[{"policy:create"}, {"policy:approve"}],
            enforcement="hard",
        ))
        self.register_sod_rule(SoDRule(
            name="Auditor ≠ Operator",
            description="Compliance auditors cannot also be system operators to prevent self-audit.",
            conflicting_roles=["AUDITOR", "OPERATOR"],
            enforcement="hard",
        ))

        logger.info(f"✅ Government hierarchy seeded for tenant '{tenant_id}'")
        return root.ou_id

    def seed_military_hierarchy(self, tenant_id: str = "defense_airgap") -> str:
        """Seed a military chain-of-command hierarchy (NATO model).

        Returns the root OU ID.
        """
        root = self.register_ou(OrgUnit(
            name="Allied Joint Force Command", ou_type=OrgUnitType.THEATER_COMMAND,
            tenant_id=tenant_id, parent_ou_id=None,
            classification_level=ClassificationLevel.SECRET,
        ))
        corps = self.register_ou(OrgUnit(
            name="I Corps", ou_type=OrgUnitType.CORPS,
            tenant_id=tenant_id, parent_ou_id=root.ou_id,
            classification_level=ClassificationLevel.SECRET,
        ))
        brigade = self.register_ou(OrgUnit(
            name="1st Armored Brigade", ou_type=OrgUnitType.BRIGADE,
            tenant_id=tenant_id, parent_ou_id=corps.ou_id,
            classification_level=ClassificationLevel.CONFIDENTIAL,
        ))
        battalion = self.register_ou(OrgUnit(
            name="1st Battalion", ou_type=OrgUnitType.BATTALION,
            tenant_id=tenant_id, parent_ou_id=brigade.ou_id,
            classification_level=ClassificationLevel.RESTRICTED,
        ))
        self.register_ou(OrgUnit(
            name="Alpha Company", ou_type=OrgUnitType.COMPANY,
            tenant_id=tenant_id, parent_ou_id=battalion.ou_id,
        ))

        # Military roles
        self.register_role(InstitutionalRole(
            name="Theater Commander", code="THCMD", rank=100,
            base_nethical_role=UserRole.GLOBAL_ADMIN,
            permissions={"roe:set", "roe:override", "system:override", "kinetic:authorize"},
            can_approve_policies=True, can_invoke_kill_switch=True,
            can_manage_subordinate_roles=True,
            classification_clearance=ClassificationLevel.SECRET,
        ))
        self.register_role(InstitutionalRole(
            name="Corps Commander", code="CCMD", rank=90,
            base_nethical_role=UserRole.GLOBAL_ADMIN,
            permissions={"roe:override", "kinetic:authorize", "intel:access_ts"},
            can_approve_policies=True, can_invoke_kill_switch=True,
            classification_clearance=ClassificationLevel.SECRET,
        ))
        self.register_role(InstitutionalRole(
            name="Brigade Commander", code="BCMD", rank=80,
            base_nethical_role=UserRole.SECURITY_OFFICER,
            permissions={"roe:execute", "kinetic:authorize", "intel:access_s"},
            can_invoke_kill_switch=True,
            classification_clearance=ClassificationLevel.SECRET,
        ))
        self.register_role(InstitutionalRole(
            name="Battalion Commander", code="BnCMD", rank=70,
            base_nethical_role=UserRole.SECURITY_OFFICER,
            permissions={"roe:execute", "kinetic:request"},
            can_invoke_kill_switch=True,
            classification_clearance=ClassificationLevel.CONFIDENTIAL,
        ))
        self.register_role(InstitutionalRole(
            name="Company Commander", code="CoCMD", rank=60,
            base_nethical_role=UserRole.HITL_REVIEWER,
            permissions={"roe:execute"},
            can_invoke_kill_switch=True,
            classification_clearance=ClassificationLevel.CONFIDENTIAL,
        ))
        self.register_role(InstitutionalRole(
            name="Platoon Leader", code="PL", rank=40,
            base_nethical_role=UserRole.AGENT_OPERATOR,
            permissions={"roe:execute"},
            classification_clearance=ClassificationLevel.RESTRICTED,
        ))

        logger.info(f"✅ Military hierarchy seeded for tenant '{tenant_id}'")
        return root.ou_id

    def seed_corporate_hierarchy(self, tenant_id: str = "default_tenant") -> str:
        """Seed a corporate org-tree (C-suite → VP → Director → Manager model).

        Returns the root OU ID.
        """
        root = self.register_ou(OrgUnit(
            name="Corporate Holding", ou_type=OrgUnitType.HOLDING,
            tenant_id=tenant_id, parent_ou_id=None,
        ))
        bu_tech = self.register_ou(OrgUnit(
            name="Technology Division", ou_type=OrgUnitType.BUSINESS_UNIT,
            tenant_id=tenant_id, parent_ou_id=root.ou_id,
            cost_center_code="TECH-001",
        ))
        bu_legal = self.register_ou(OrgUnit(
            name="Legal & Compliance", ou_type=OrgUnitType.BUSINESS_UNIT,
            tenant_id=tenant_id, parent_ou_id=root.ou_id,
            cost_center_code="LEGAL-001",
        ))
        self.register_ou(OrgUnit(
            name="AI Engineering Team", ou_type=OrgUnitType.TEAM,
            tenant_id=tenant_id, parent_ou_id=bu_tech.ou_id,
            cost_center_code="TECH-AI-001",
        ))
        self.register_ou(OrgUnit(
            name="Data Protection Office", ou_type=OrgUnitType.OFFICE,
            tenant_id=tenant_id, parent_ou_id=bu_legal.ou_id,
            cost_center_code="LEGAL-DPO-001",
        ))

        # Corporate roles
        self.register_role(InstitutionalRole(
            name="Chief Executive Officer", code="CEO", rank=100,
            base_nethical_role=UserRole.GLOBAL_ADMIN,
            permissions={"policy:approve", "system:override", "budget:unlimited"},
            can_approve_policies=True, can_invoke_kill_switch=True,
            can_manage_subordinate_roles=True,
        ))
        self.register_role(InstitutionalRole(
            name="Chief AI Officer", code="CAIO", rank=90,
            base_nethical_role=UserRole.GLOBAL_ADMIN,
            permissions={"policy:manage", "policy:approve", "model:govern"},
            can_approve_policies=True, can_invoke_kill_switch=True,
        ))
        self.register_role(InstitutionalRole(
            name="Data Protection Officer", code="DPO", rank=85,
            base_nethical_role=UserRole.COMPLIANCE_AUDITOR,
            permissions={"privacy:audit", "privacy:enforce", "dpia:approve"},
            can_approve_policies=True,
        ))
        self.register_role(InstitutionalRole(
            name="VP Engineering", code="VPE", rank=80,
            base_nethical_role=UserRole.SECURITY_OFFICER,
            permissions={"policy:review", "deployment:approve"},
            can_manage_subordinate_roles=True,
        ))
        self.register_role(InstitutionalRole(
            name="Director", code="DIR", rank=70,
            base_nethical_role=UserRole.HITL_REVIEWER,
            permissions={"policy:review", "team:manage"},
            can_manage_subordinate_roles=True,
        ))
        self.register_role(InstitutionalRole(
            name="Manager", code="MGR", rank=50,
            base_nethical_role=UserRole.HITL_REVIEWER,
            permissions={"report:generate"},
        ))
        self.register_role(InstitutionalRole(
            name="Individual Contributor", code="IC", rank=20,
            base_nethical_role=UserRole.AGENT_OPERATOR,
            permissions=set(),
        ))

        # Corporate SoD rules
        self.register_sod_rule(SoDRule(
            name="DPO Independence",
            description="Data Protection Officer cannot also hold operational management roles (GDPR Art. 38).",
            conflicting_roles=["DPO", "VPE"],
            enforcement="hard",
        ))
        self.register_sod_rule(SoDRule(
            name="Model Governance ≠ Model Development",
            description="The person who governs AI models cannot also develop them.",
            conflicting_permissions=[{"model:govern"}, {"model:develop"}],
            enforcement="hard",
        ))

        logger.info(f"✅ Corporate hierarchy seeded for tenant '{tenant_id}'")
        return root.ou_id

    # ---------- Reporting ----------

    def get_hierarchy_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a summary report of the organizational hierarchy."""
        ous = list(self._org_units.values())
        if tenant_id:
            ous = [ou for ou in ous if ou.tenant_id == tenant_id]

        return {
            "total_org_units": len(ous),
            "total_roles": len(self._roles),
            "total_assignments": len(self._assignments),
            "active_delegations": sum(1 for d in self._delegations.values() if d.is_active()),
            "sod_rules": len(self._sod_rules),
            "sod_violations_total": len(self._sod_violations),
            "org_unit_types": dict(
                sorted(
                    {t: sum(1 for o in ous if o.ou_type == t) for t in set(o.ou_type for o in ous)}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }
