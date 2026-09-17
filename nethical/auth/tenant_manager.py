# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Multi-Tenant Isolation & Sovereign Workspace Management."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..core.models import TenantConfig, ClassificationLevel
from ..security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.auth.tenant_manager")


class TenantManager:
    """Manages organizational multi-tenancy, regulatory boundaries, and isolated ledgers."""

    def __init__(self, default_ledger: Optional[MerkleLedger] = None) -> None:
        self._tenants: Dict[str, TenantConfig] = {}
        self._ledgers: Dict[str, MerkleLedger] = {}
        self._default_ledger = default_ledger or MerkleLedger()

        self._seed_default_tenants()

    def _seed_default_tenants(self) -> None:
        """Seeds initial default enterprise and sovereign government workspaces."""
        # 1. Default global sovereign workspace
        default_t = TenantConfig(
            tenant_id="default_tenant",
            name="Nethical Global Sovereign Node",
            jurisdiction="GLOBAL",
            classification_level=ClassificationLevel.UNCLASSIFIED,
            allowed_frameworks=["EU_AI_ACT", "ISO_42001", "NIST_AI_RMF", "UK_CMA", "POLISH_KSC"],
            metadata={"environment": "production_mesh"},
        )
        self.register_tenant(default_t, ledger=self._default_ledger)

        # 2. Polish Governmental & Critical Infrastructure (KSC / CSIRT)
        gov_t = TenantConfig(
            tenant_id="gov_pl_cyber",
            name="Rządowy Węzeł Nadzoru Cyberbezpieczeństwa RP",
            jurisdiction="PL",
            classification_level=ClassificationLevel.RESTRICTED,
            allowed_frameworks=["POLISH_KSC", "POLISH_PENAL_CODE", "POLISH_UODO", "EU_AI_ACT", "EU_NIS2"],
            metadata={"csirt_reporting_enabled": True, "jurisdiction_scope": "national_critical_infrastructure"},
        )
        self.register_tenant(gov_t)

        # 3. European Financial Institution (DORA / ECB)
        fin_t = TenantConfig(
            tenant_id="enterprise_fin_eu",
            name="Euro-Banking Autonomous Trading & Credit AI",
            jurisdiction="EU",
            classification_level=ClassificationLevel.CONFIDENTIAL,
            allowed_frameworks=["EU_DORA", "EU_AI_ACT", "EU_GDPR", "ISO_42001"],
            metadata={"financial_circuit_breaker_strict": True, "max_single_tx": 50000.0},
        )
        self.register_tenant(fin_t)

        # 4. Sovereign Defense & Air-Gapped Command (NATO / Defense)
        mil_t = TenantConfig(
            tenant_id="defense_airgap",
            name="Sovereign Air-Gapped Tactical Node",
            jurisdiction="GLOBAL_SOVEREIGN",
            classification_level=ClassificationLevel.SECRET,
            allowed_frameworks=["ISO_13849_KINETIC", "EU_CRA", "NIST_AI_RMF"],
            metadata={"air_gapped": True, "data_diode_enabled": True},
        )
        self.register_tenant(mil_t)

    def register_tenant(
        self, tenant: TenantConfig, ledger: Optional[MerkleLedger] = None
    ) -> TenantConfig:
        """Registers a new tenant and initializes its dedicated Merkle ledger."""
        self._tenants[tenant.tenant_id] = tenant
        if ledger is not None:
            self._ledgers[tenant.tenant_id] = ledger
        elif tenant.tenant_id not in self._ledgers:
            # Initialize isolated Merkle Ledger for this tenant
            self._ledgers[tenant.tenant_id] = MerkleLedger()
        logger.info(f"Registered tenant: {tenant.tenant_id} ({tenant.name})")
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Retrieves tenant configuration by ID."""
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> List[TenantConfig]:
        """Lists all registered active tenants."""
        return list(self._tenants.values())

    def get_tenant_ledger(self, tenant_id: str) -> MerkleLedger:
        """Returns the isolated cryptographic MerkleLedger for the tenant."""
        if tenant_id in self._ledgers:
            return self._ledgers[tenant_id]
        # Fallback to default ledger if tenant not explicitly partitioned
        return self._default_ledger

    def create_tenant(
        self,
        name: str,
        jurisdiction: str = "EU",
        classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED,
        allowed_frameworks: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TenantConfig:
        """Programmatic helper to create and provision a tenant."""
        tenant = TenantConfig(
            name=name,
            jurisdiction=jurisdiction,
            classification_level=classification,
            allowed_frameworks=allowed_frameworks or ["EU_AI_ACT", "ISO_42001"],
            metadata=metadata or {},
        )
        return self.register_tenant(tenant)
