"""Data Residency Management for Nethical.

This module provides comprehensive data residency management capabilities
to ensure data stays in required jurisdictions according to regulatory
requirements (GDPR, CCPA, etc.).

Features:
- Data classification and tagging
- Region-aware storage validation
- Cross-region transfer blocking
- Data movement audit trail

Adheres to the 25 Fundamental Laws:
- Law 15: Audit Compliance - Audit trail for data movement
- Law 22: Digital Security - Protection of digital assets and privacy
- Law 23: Fail-Safe Design - Safe failure modes

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DataRegion(str, Enum):
    """Data residency regions."""

    # European Union regions
    EU_WEST_1 = "eu-west-1"  # Ireland
    EU_WEST_2 = "eu-west-2"  # London
    EU_CENTRAL_1 = "eu-central-1"  # Frankfurt

    # US regions
    US_EAST_1 = "us-east-1"  # N. Virginia
    US_WEST_1 = "us-west-1"  # N. California
    US_WEST_2 = "us-west-2"  # Oregon

    # Asia Pacific regions
    AP_SOUTH_1 = "ap-south-1"  # Mumbai
    AP_NORTHEAST_1 = "ap-northeast-1"  # Tokyo
    AP_SOUTHEAST_1 = "ap-southeast-1"  # Singapore

    # Global (replicated across regions)
    GLOBAL = "global"


class DataJurisdiction(str, Enum):
    """Data jurisdictions for regulatory compliance."""

    EU = "eu"  # European Union (GDPR)
    UK = "uk"  # United Kingdom (UK GDPR)
    US = "us"  # United States
    CA = "ca"  # California (CCPA)
    APAC = "apac"  # Asia Pacific
    GLOBAL = "global"  # No specific jurisdiction


class DataClassification(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"  # No restrictions
    INTERNAL = "internal"  # Internal use only
    CONFIDENTIAL = "confidential"  # Restricted access
    PII = "pii"  # Personal Identifiable Information
    SENSITIVE_PII = "sensitive_pii"  # Special category data
    RESTRICTED = "restricted"  # Highest classification


class DataType(str, Enum):
    """Types of data for residency rules."""

    PII = "pii"  # Personal data
    DECISIONS = "decisions"  # AI decisions
    AUDIT_LOGS = "audit_logs"  # Audit trail
    POLICIES = "policies"  # Policy definitions
    MODELS = "models"  # AI models
    CONFIGS = "configs"  # Configuration data
    METRICS = "metrics"  # Performance metrics
    GENERAL = "general"  # General data


@dataclass
class ResidencyPolicy:
    """Data residency policy for a data type."""

    data_type: DataType
    allowed_regions: Set[DataRegion]
    required_jurisdiction: Optional[DataJurisdiction]
    processing_restriction: str  # "region-only" or "any"
    cross_region_transfer_allowed: bool
    requires_encryption: bool
    retention_days: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data_type": self.data_type.value,
            "allowed_regions": [r.value for r in self.allowed_regions],
            "required_jurisdiction": (
                self.required_jurisdiction.value if self.required_jurisdiction else None
            ),
            "processing_restriction": self.processing_restriction,
            "cross_region_transfer_allowed": self.cross_region_transfer_allowed,
            "requires_encryption": self.requires_encryption,
            "retention_days": self.retention_days,
        }


@dataclass
class ResidencyViolation:
    """Record of a data residency policy violation."""

    violation_id: str
    data_type: DataType
    classification: DataClassification
    source_region: DataRegion
    target_region: Optional[DataRegion]
    violation_type: str
    description: str
    severity: str
    blocked: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "violation_id": self.violation_id,
            "data_type": self.data_type.value,
            "classification": self.classification.value,
            "source_region": self.source_region.value,
            "target_region": self.target_region.value if self.target_region else None,
            "violation_type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
            "blocked": self.blocked,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DataMovementRecord:
    """Record of data movement for audit trail."""

    record_id: str
    data_id: str
    data_type: DataType
    classification: DataClassification
    source_region: DataRegion
    target_region: DataRegion
    movement_type: str  # "copy", "move", "process"
    authorized: bool
    reason: str
    authorized_by: Optional[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "record_id": self.record_id,
            "data_id": self.data_id,
            "data_type": self.data_type.value,
            "classification": self.classification.value,
            "source_region": self.source_region.value,
            "target_region": self.target_region.value,
            "movement_type": self.movement_type,
            "authorized": self.authorized,
            "reason": self.reason,
            "authorized_by": self.authorized_by,
            "timestamp": self.timestamp.isoformat(),
        }


class DataResidencyManager:
    """Manages data residency policies and enforcement.

    Ensures data stays in required jurisdictions according to:
    - GDPR requirements (EU data in EU)
    - CCPA requirements (California consumer data)
    - Other regulatory requirements

    Attributes:
        policies: Dictionary of data type to residency policy
    """

    # Region to jurisdiction mapping
    REGION_JURISDICTION_MAP: Dict[DataRegion, DataJurisdiction] = {
        DataRegion.EU_WEST_1: DataJurisdiction.EU,
        DataRegion.EU_WEST_2: DataJurisdiction.UK,
        DataRegion.EU_CENTRAL_1: DataJurisdiction.EU,
        DataRegion.US_EAST_1: DataJurisdiction.US,
        DataRegion.US_WEST_1: DataJurisdiction.CA,
        DataRegion.US_WEST_2: DataJurisdiction.US,
        DataRegion.AP_SOUTH_1: DataJurisdiction.APAC,
        DataRegion.AP_NORTHEAST_1: DataJurisdiction.APAC,
        DataRegion.AP_SOUTHEAST_1: DataJurisdiction.APAC,
        DataRegion.GLOBAL: DataJurisdiction.GLOBAL,
    }

    def __init__(self) -> None:
        """Initialize Data Residency Manager."""
        self.policies: Dict[DataType, ResidencyPolicy] = {}
        self.violations: List[ResidencyViolation] = []
        self.movement_records: List[DataMovementRecord] = []

        # Initialize default policies
        self._initialize_default_policies()

        logger.info("DataResidencyManager initialized with default policies")

    def _initialize_default_policies(self) -> None:
        """Initialize default data residency policies."""

        # PII: EU data must stay in EU
        self.policies[DataType.PII] = ResidencyPolicy(
            data_type=DataType.PII,
            allowed_regions={
                DataRegion.EU_WEST_1,
                DataRegion.EU_WEST_2,
                DataRegion.EU_CENTRAL_1,
            },
            required_jurisdiction=DataJurisdiction.EU,
            processing_restriction="region-only",
            cross_region_transfer_allowed=False,
            requires_encryption=True,
            retention_days=2555,  # 7 years for legal retention
        )

        # Decisions: Store in region where they were made
        self.policies[DataType.DECISIONS] = ResidencyPolicy(
            data_type=DataType.DECISIONS,
            allowed_regions={
                DataRegion.EU_WEST_1,
                DataRegion.EU_CENTRAL_1,
                DataRegion.US_EAST_1,
            },
            required_jurisdiction=None,
            processing_restriction="region-only",
            cross_region_transfer_allowed=False,
            requires_encryption=True,
            retention_days=365,
        )

        # Audit logs: Keep in originating region
        self.policies[DataType.AUDIT_LOGS] = ResidencyPolicy(
            data_type=DataType.AUDIT_LOGS,
            allowed_regions={
                DataRegion.EU_WEST_1,
                DataRegion.EU_CENTRAL_1,
                DataRegion.US_EAST_1,
            },
            required_jurisdiction=None,
            processing_restriction="region-only",
            cross_region_transfer_allowed=False,
            requires_encryption=True,
            retention_days=2555,  # 7 years
        )

        # Policies: Can be replicated globally
        self.policies[DataType.POLICIES] = ResidencyPolicy(
            data_type=DataType.POLICIES,
            allowed_regions={DataRegion.GLOBAL},
            required_jurisdiction=None,
            processing_restriction="any",
            cross_region_transfer_allowed=True,
            requires_encryption=False,
            retention_days=-1,  # No expiration
        )

        # Models: Can be replicated globally
        self.policies[DataType.MODELS] = ResidencyPolicy(
            data_type=DataType.MODELS,
            allowed_regions={DataRegion.GLOBAL},
            required_jurisdiction=None,
            processing_restriction="any",
            cross_region_transfer_allowed=True,
            requires_encryption=False,
            retention_days=-1,
        )

        # Configs: Can be replicated globally
        self.policies[DataType.CONFIGS] = ResidencyPolicy(
            data_type=DataType.CONFIGS,
            allowed_regions={DataRegion.GLOBAL},
            required_jurisdiction=None,
            processing_restriction="any",
            cross_region_transfer_allowed=True,
            requires_encryption=False,
            retention_days=-1,
        )

        # Metrics: Less restrictive
        self.policies[DataType.METRICS] = ResidencyPolicy(
            data_type=DataType.METRICS,
            allowed_regions={DataRegion.GLOBAL},
            required_jurisdiction=None,
            processing_restriction="any",
            cross_region_transfer_allowed=True,
            requires_encryption=False,
            retention_days=90,
        )

        # General data: Default policy
        self.policies[DataType.GENERAL] = ResidencyPolicy(
            data_type=DataType.GENERAL,
            allowed_regions={DataRegion.GLOBAL},
            required_jurisdiction=None,
            processing_restriction="any",
            cross_region_transfer_allowed=True,
            requires_encryption=False,
            retention_days=365,
        )

    def classify_data(
        self,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[DataType, DataClassification]:
        """Classify data for residency rules.

        Analyzes content to determine data type and classification.

        Args:
            content: Data content to classify
            metadata: Optional metadata

        Returns:
            Tuple of (DataType, DataClassification)
        """
        # Check for PII indicators
        pii_indicators = [
            "email",
            "phone",
            "address",
            "name",
            "ssn",
            "social_security",
            "date_of_birth",
            "passport",
            "driver_license",
        ]

        sensitive_pii_indicators = [
            "health",
            "medical",
            "biometric",
            "genetic",
            "race",
            "ethnicity",
            "religion",
            "political",
            "sexual_orientation",
        ]

        content_str = str(content).lower()

        # Check for sensitive PII
        if any(indicator in content_str for indicator in sensitive_pii_indicators):
            return DataType.PII, DataClassification.SENSITIVE_PII

        # Check for regular PII
        if any(indicator in content_str for indicator in pii_indicators):
            return DataType.PII, DataClassification.PII

        # Check metadata for explicit classification
        if metadata:
            explicit_type = metadata.get("data_type")
            if explicit_type and explicit_type in [dt.value for dt in DataType]:
                data_type = DataType(explicit_type)
            else:
                data_type = DataType.GENERAL

            explicit_class = metadata.get("classification")
            if explicit_class and explicit_class in [
                dc.value for dc in DataClassification
            ]:
                classification = DataClassification(explicit_class)
            else:
                classification = DataClassification.INTERNAL

            return data_type, classification

        return DataType.GENERAL, DataClassification.INTERNAL

    def validate_storage_location(
        self,
        data_type: DataType,
        target_region: DataRegion,
        data_classification: Optional[DataClassification] = None,
    ) -> tuple[bool, Optional[ResidencyViolation]]:
        """Validate if data can be stored in target region.

        Args:
            data_type: Type of data
            target_region: Target storage region
            data_classification: Optional classification

        Returns:
            Tuple of (is_valid, violation if invalid)
        """
        policy = self.policies.get(data_type)

        if not policy:
            # No policy = allow by default
            return True, None

        # Check if region is allowed
        if DataRegion.GLOBAL not in policy.allowed_regions:
            if target_region not in policy.allowed_regions:
                violation = ResidencyViolation(
                    violation_id=str(uuid.uuid4()),
                    data_type=data_type,
                    classification=data_classification or DataClassification.INTERNAL,
                    source_region=target_region,
                    target_region=None,
                    violation_type="invalid_region",
                    description=(
                        f"Data type '{data_type.value}' cannot be stored in "
                        f"region '{target_region.value}'"
                    ),
                    severity="high",
                    blocked=True,
                )
                self.violations.append(violation)
                return False, violation

        # Check jurisdiction requirement
        if policy.required_jurisdiction:
            target_jurisdiction = self.REGION_JURISDICTION_MAP.get(target_region)
            if target_jurisdiction != policy.required_jurisdiction:
                violation = ResidencyViolation(
                    violation_id=str(uuid.uuid4()),
                    data_type=data_type,
                    classification=data_classification or DataClassification.INTERNAL,
                    source_region=target_region,
                    target_region=None,
                    violation_type="jurisdiction_violation",
                    description=(
                        f"Data type '{data_type.value}' requires jurisdiction "
                        f"'{policy.required_jurisdiction.value}', but target region "
                        f"'{target_region.value}' is in '{target_jurisdiction.value if target_jurisdiction else 'unknown'}'"
                    ),
                    severity="critical",
                    blocked=True,
                )
                self.violations.append(violation)
                return False, violation

        return True, None

    def validate_cross_region_transfer(
        self,
        data_type: DataType,
        source_region: DataRegion,
        target_region: DataRegion,
        data_classification: Optional[DataClassification] = None,
    ) -> tuple[bool, Optional[ResidencyViolation]]:
        """Validate if data can be transferred between regions.

        Args:
            data_type: Type of data
            source_region: Source region
            target_region: Target region
            data_classification: Optional classification

        Returns:
            Tuple of (is_valid, violation if invalid)
        """
        # Same region = always allowed
        if source_region == target_region:
            return True, None

        policy = self.policies.get(data_type)

        if not policy:
            return True, None

        # Check if cross-region transfer is allowed
        if not policy.cross_region_transfer_allowed:
            violation = ResidencyViolation(
                violation_id=str(uuid.uuid4()),
                data_type=data_type,
                classification=data_classification or DataClassification.INTERNAL,
                source_region=source_region,
                target_region=target_region,
                violation_type="cross_region_blocked",
                description=(
                    f"Cross-region transfer of '{data_type.value}' from "
                    f"'{source_region.value}' to '{target_region.value}' is not allowed"
                ),
                severity="high",
                blocked=True,
            )
            self.violations.append(violation)
            return False, violation

        # Check if target region is allowed
        is_valid, violation = self.validate_storage_location(
            data_type=data_type,
            target_region=target_region,
            data_classification=data_classification,
        )

        if not is_valid:
            violation.violation_type = "cross_region_invalid_target"
            violation.source_region = source_region
            violation.target_region = target_region
            return False, violation

        return True, None

    def record_data_movement(
        self,
        data_id: str,
        data_type: DataType,
        classification: DataClassification,
        source_region: DataRegion,
        target_region: DataRegion,
        movement_type: str,
        reason: str,
        authorized_by: Optional[str] = None,
    ) -> DataMovementRecord:
        """Record a data movement for audit trail.

        Args:
            data_id: Unique identifier for the data
            data_type: Type of data
            classification: Data classification
            source_region: Source region
            target_region: Target region
            movement_type: Type of movement (copy, move, process)
            reason: Reason for the movement
            authorized_by: Who authorized the movement

        Returns:
            DataMovementRecord
        """
        # Validate the movement
        is_valid, _ = self.validate_cross_region_transfer(
            data_type=data_type,
            source_region=source_region,
            target_region=target_region,
            data_classification=classification,
        )

        record = DataMovementRecord(
            record_id=str(uuid.uuid4()),
            data_id=data_id,
            data_type=data_type,
            classification=classification,
            source_region=source_region,
            target_region=target_region,
            movement_type=movement_type,
            authorized=is_valid,
            reason=reason,
            authorized_by=authorized_by,
        )

        self.movement_records.append(record)

        if is_valid:
            logger.info(
                "Data movement recorded: %s -> %s (%s)",
                source_region.value,
                target_region.value,
                data_type.value,
            )
        else:
            logger.warning(
                "Unauthorized data movement attempted: %s -> %s (%s)",
                source_region.value,
                target_region.value,
                data_type.value,
            )

        return record

    def get_allowed_regions(
        self,
        data_type: DataType,
    ) -> Set[DataRegion]:
        """Get allowed regions for a data type.

        Args:
            data_type: Type of data

        Returns:
            Set of allowed DataRegions
        """
        policy = self.policies.get(data_type)
        if not policy:
            return {DataRegion.GLOBAL}
        return policy.allowed_regions

    def get_policy(self, data_type: DataType) -> Optional[ResidencyPolicy]:
        """Get residency policy for a data type.

        Args:
            data_type: Type of data

        Returns:
            ResidencyPolicy or None
        """
        return self.policies.get(data_type)

    def update_policy(self, policy: ResidencyPolicy) -> None:
        """Update or add a residency policy.

        Args:
            policy: ResidencyPolicy to add/update
        """
        self.policies[policy.data_type] = policy
        logger.info(
            "Residency policy updated for data type: %s",
            policy.data_type.value,
        )

    def get_violations_summary(self) -> Dict[str, Any]:
        """Get summary of residency violations.

        Returns:
            Dictionary with violation summary
        """
        if not self.violations:
            return {
                "total_violations": 0,
                "status": "No violations recorded",
            }

        # Count by type
        type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        blocked_count = 0

        for violation in self.violations:
            type_counts[violation.violation_type] = (
                type_counts.get(violation.violation_type, 0) + 1
            )
            severity_counts[violation.severity] = (
                severity_counts.get(violation.severity, 0) + 1
            )
            if violation.blocked:
                blocked_count += 1

        return {
            "total_violations": len(self.violations),
            "blocked": blocked_count,
            "by_type": type_counts,
            "by_severity": severity_counts,
            "last_violation": self.violations[-1].timestamp.isoformat(),
        }

    def get_movement_audit_trail(
        self,
        data_type: Optional[DataType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[DataMovementRecord]:
        """Get data movement audit trail.

        Args:
            data_type: Optional filter by data type
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of DataMovementRecords
        """
        records = self.movement_records

        if data_type:
            records = [r for r in records if r.data_type == data_type]

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]

        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        return records


__all__ = [
    "DataResidencyManager",
    "DataRegion",
    "DataJurisdiction",
    "DataClassification",
    "DataType",
    "ResidencyPolicy",
    "ResidencyViolation",
    "DataMovementRecord",
]
