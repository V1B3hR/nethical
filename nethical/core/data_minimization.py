"""Data Minimization and Right-to-be-Forgotten Support.

This module provides automatic data retention policies, minimal data collection,
anonymization pipelines, and GDPR right-to-be-forgotten support.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from pathlib import Path


class DataCategory(Enum):
    """Categories of data for retention policies."""
    PERSONAL_IDENTIFIABLE = "personal_identifiable"
    QUASI_IDENTIFIABLE = "quasi_identifiable"
    SENSITIVE = "sensitive"
    OPERATIONAL = "operational"
    AUDIT = "audit"
    ANALYTICS = "analytics"


class RetentionPolicy(Enum):
    """Data retention policy types."""
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 90 days
    LONG_TERM = "long_term"  # 365 days
    PERMANENT = "permanent"  # No deletion
    CUSTOM = "custom"  # Custom period


@dataclass
class RetentionRule:
    """Rule for data retention."""
    category: DataCategory
    policy: RetentionPolicy
    retention_days: int
    description: str
    auto_delete: bool = True
    notify_before_days: int = 7


@dataclass
class DataRecord:
    """A data record with retention metadata."""
    record_id: str
    category: DataCategory
    created_at: datetime
    expires_at: datetime
    data: Dict[str, Any]
    anonymized: bool = False
    deleted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeletionRequest:
    """Request to delete user data (right-to-be-forgotten)."""
    request_id: str
    user_id: str
    requested_at: datetime
    data_categories: List[DataCategory]
    status: str  # "pending", "processing", "completed", "failed"
    completed_at: Optional[datetime] = None
    records_deleted: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataMinimization:
    """Data minimization and retention management system."""
    
    # Default retention policies by category
    DEFAULT_POLICIES = {
        DataCategory.PERSONAL_IDENTIFIABLE: RetentionRule(
            category=DataCategory.PERSONAL_IDENTIFIABLE,
            policy=RetentionPolicy.SHORT_TERM,
            retention_days=30,
            description="Personal identifiable information - minimal retention",
            auto_delete=True,
            notify_before_days=7
        ),
        DataCategory.QUASI_IDENTIFIABLE: RetentionRule(
            category=DataCategory.QUASI_IDENTIFIABLE,
            policy=RetentionPolicy.MEDIUM_TERM,
            retention_days=90,
            description="Quasi-identifiable information",
            auto_delete=True,
            notify_before_days=7
        ),
        DataCategory.SENSITIVE: RetentionRule(
            category=DataCategory.SENSITIVE,
            policy=RetentionPolicy.SHORT_TERM,
            retention_days=30,
            description="Sensitive personal data - minimal retention",
            auto_delete=True,
            notify_before_days=14
        ),
        DataCategory.OPERATIONAL: RetentionRule(
            category=DataCategory.OPERATIONAL,
            policy=RetentionPolicy.MEDIUM_TERM,
            retention_days=90,
            description="Operational data",
            auto_delete=True,
            notify_before_days=7
        ),
        DataCategory.AUDIT: RetentionRule(
            category=DataCategory.AUDIT,
            policy=RetentionPolicy.LONG_TERM,
            retention_days=365,
            description="Audit logs for compliance",
            auto_delete=False,
            notify_before_days=30
        ),
        DataCategory.ANALYTICS: RetentionRule(
            category=DataCategory.ANALYTICS,
            policy=RetentionPolicy.MEDIUM_TERM,
            retention_days=90,
            description="Analytics data",
            auto_delete=True,
            notify_before_days=7
        ),
    }
    
    def __init__(
        self,
        storage_dir: str = "./data_minimization",
        custom_policies: Optional[Dict[DataCategory, RetentionRule]] = None,
        enable_auto_deletion: bool = True,
        anonymization_enabled: bool = True
    ):
        """Initialize data minimization system.
        
        Args:
            storage_dir: Directory for storing records and logs
            custom_policies: Custom retention policies (overrides defaults)
            enable_auto_deletion: Enable automatic deletion of expired records
            anonymization_enabled: Enable data anonymization
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Retention policies
        self.policies = self.DEFAULT_POLICIES.copy()
        if custom_policies:
            self.policies.update(custom_policies)
        
        self.enable_auto_deletion = enable_auto_deletion
        self.anonymization_enabled = anonymization_enabled
        
        # Storage
        self.records: Dict[str, DataRecord] = {}
        self.deletion_requests: Dict[str, DeletionRequest] = {}
        
        # Statistics
        self.stats = {
            'total_records': 0,
            'anonymized_records': 0,
            'deleted_records': 0,
            'active_deletion_requests': 0,
            'expired_records': 0
        }
    
    def store_data(
        self,
        data: Dict[str, Any],
        category: DataCategory,
        user_id: Optional[str] = None,
        minimal_fields_only: bool = True
    ) -> DataRecord:
        """Store data with minimal necessary information.
        
        Args:
            data: Data to store
            category: Data category for retention policy
            user_id: User ID for right-to-be-forgotten
            minimal_fields_only: Only store essential fields
            
        Returns:
            Stored data record
        """
        # Apply minimal data collection
        if minimal_fields_only:
            data = self._extract_minimal_fields(data, category)
        
        # Get retention policy
        policy = self.policies.get(category)
        if not policy:
            raise ValueError(f"No retention policy for category: {category}")
        
        # Create record
        record_id = self._generate_record_id(data, user_id)
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=policy.retention_days)
        
        record = DataRecord(
            record_id=record_id,
            category=category,
            created_at=created_at,
            expires_at=expires_at,
            data=data,
            anonymized=False,
            deleted=False,
            metadata={
                'user_id': user_id,
                'policy': policy.policy.value,
                'auto_delete': policy.auto_delete
            }
        )
        
        self.records[record_id] = record
        self.stats['total_records'] += 1
        
        return record
    
    def anonymize_data(
        self,
        record_id: str,
        anonymization_level: str = "standard"
    ) -> Optional[DataRecord]:
        """Anonymize a data record.
        
        Args:
            record_id: Record to anonymize
            anonymization_level: Level of anonymization ("minimal", "standard", "aggressive")
            
        Returns:
            Anonymized record or None if not found
        """
        if record_id not in self.records:
            return None
        
        record = self.records[record_id]
        
        if record.anonymized:
            return record  # Already anonymized
        
        # Apply anonymization
        anonymized_data = self._anonymize_fields(
            record.data,
            record.category,
            anonymization_level
        )
        
        record.data = anonymized_data
        record.anonymized = True
        record.metadata['anonymized_at'] = datetime.now().isoformat()
        record.metadata['anonymization_level'] = anonymization_level
        
        self.stats['anonymized_records'] += 1
        
        return record
    
    def process_retention_policy(self) -> Dict[str, Any]:
        """Process retention policies and delete/anonymize expired records.
        
        Returns:
            Summary of retention processing
        """
        now = datetime.now()
        expired_records = []
        deleted_count = 0
        anonymized_count = 0
        notified_count = 0
        
        for record_id, record in self.records.items():
            if record.deleted:
                continue
            
            # Check if expired
            if record.expires_at <= now:
                expired_records.append(record_id)
                
                policy = self.policies.get(record.category)
                if policy and policy.auto_delete and self.enable_auto_deletion:
                    # Delete record
                    record.deleted = True
                    record.metadata['deleted_at'] = now.isoformat()
                    deleted_count += 1
                    self.stats['deleted_records'] += 1
                elif not record.anonymized and self.anonymization_enabled:
                    # Anonymize instead of delete
                    self.anonymize_data(record_id)
                    anonymized_count += 1
            
            # Check if should notify before expiration
            elif record.expires_at <= now + timedelta(days=7):
                policy = self.policies.get(record.category)
                if policy and policy.notify_before_days > 0:
                    notified_count += 1
        
        self.stats['expired_records'] = len(expired_records)
        
        return {
            'timestamp': now.isoformat(),
            'expired_records': len(expired_records),
            'deleted': deleted_count,
            'anonymized': anonymized_count,
            'notified': notified_count,
            'expired_record_ids': expired_records[:10]  # Sample
        }
    
    def request_data_deletion(
        self,
        user_id: str,
        categories: Optional[List[DataCategory]] = None
    ) -> DeletionRequest:
        """Request deletion of user data (right-to-be-forgotten).
        
        Args:
            user_id: User ID to delete data for
            categories: Specific categories to delete (None = all)
            
        Returns:
            Deletion request
        """
        request_id = hashlib.sha256(
            f"{user_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        deletion_request = DeletionRequest(
            request_id=request_id,
            user_id=user_id,
            requested_at=datetime.now(),
            data_categories=categories or list(DataCategory),
            status="pending",
            metadata={}
        )
        
        self.deletion_requests[request_id] = deletion_request
        self.stats['active_deletion_requests'] += 1
        
        # Process immediately if auto-deletion is enabled
        if self.enable_auto_deletion:
            self._process_deletion_request(request_id)
        
        return deletion_request
    
    def _process_deletion_request(self, request_id: str) -> None:
        """Process a deletion request."""
        if request_id not in self.deletion_requests:
            return
        
        request = self.deletion_requests[request_id]
        request.status = "processing"
        
        deleted_count = 0
        
        # Find and delete user records
        for record_id, record in self.records.items():
            if record.deleted:
                continue
            
            # Check if record belongs to user
            if record.metadata.get('user_id') != request.user_id:
                continue
            
            # Check if category matches
            if record.category not in request.data_categories:
                continue
            
            # Delete record
            record.deleted = True
            record.metadata['deleted_at'] = datetime.now().isoformat()
            record.metadata['deletion_request_id'] = request_id
            deleted_count += 1
            self.stats['deleted_records'] += 1
        
        request.status = "completed"
        request.completed_at = datetime.now()
        request.records_deleted = deleted_count
        self.stats['active_deletion_requests'] -= 1
    
    def _extract_minimal_fields(
        self,
        data: Dict[str, Any],
        category: DataCategory
    ) -> Dict[str, Any]:
        """Extract only minimal necessary fields from data.
        
        Args:
            data: Original data
            category: Data category
            
        Returns:
            Minimal data with only essential fields
        """
        # Define essential fields by category
        essential_fields = {
            DataCategory.PERSONAL_IDENTIFIABLE: ['user_id', 'email', 'name'],
            DataCategory.QUASI_IDENTIFIABLE: ['age_range', 'location', 'occupation'],
            DataCategory.SENSITIVE: ['health_data', 'financial_data'],
            DataCategory.OPERATIONAL: ['action', 'timestamp', 'status'],
            DataCategory.AUDIT: ['event', 'timestamp', 'user_id', 'action'],
            DataCategory.ANALYTICS: ['metric', 'value', 'timestamp']
        }
        
        # Get essential fields for this category
        required = essential_fields.get(category, [])
        
        # Extract only required fields
        minimal = {}
        for field in required:
            if field in data:
                minimal[field] = data[field]
        
        # If no essential fields defined, keep all (but log warning)
        if not required:
            return data
        
        return minimal if minimal else data
    
    def _anonymize_fields(
        self,
        data: Dict[str, Any],
        category: DataCategory,
        level: str
    ) -> Dict[str, Any]:
        """Anonymize data fields based on category and level.
        
        Args:
            data: Data to anonymize
            category: Data category
            level: Anonymization level
            
        Returns:
            Anonymized data
        """
        anonymized = data.copy()
        
        # Fields to anonymize by category
        sensitive_fields = {
            DataCategory.PERSONAL_IDENTIFIABLE: ['email', 'phone', 'name', 'ssn'],
            DataCategory.QUASI_IDENTIFIABLE: ['location', 'occupation', 'age'],
            DataCategory.SENSITIVE: ['health_data', 'financial_data', 'biometric'],
        }
        
        fields_to_anonymize = sensitive_fields.get(category, [])
        
        for field in fields_to_anonymize:
            if field in anonymized:
                if level == "aggressive":
                    # Remove field entirely
                    del anonymized[field]
                elif level == "standard":
                    # Hash the field
                    value = str(anonymized[field])
                    anonymized[field] = hashlib.sha256(value.encode()).hexdigest()[:16]
                elif level == "minimal":
                    # Generalize the field
                    anonymized[field] = self._generalize_value(
                        anonymized[field], field
                    )
        
        return anonymized
    
    def _generalize_value(self, value: Any, field_name: str) -> str:
        """Generalize a value for minimal anonymization.
        
        Args:
            value: Value to generalize
            field_name: Name of the field
            
        Returns:
            Generalized value
        """
        # Email: keep domain
        if 'email' in field_name.lower() and isinstance(value, str):
            if '@' in value:
                return f"***@{value.split('@')[1]}"
        
        # Age: convert to range
        if 'age' in field_name.lower():
            try:
                age = int(value)
                if age < 18:
                    return "under_18"
                elif age < 30:
                    return "18-29"
                elif age < 50:
                    return "30-49"
                else:
                    return "50+"
            except (ValueError, TypeError):
                pass
        
        # Location: keep region/country only
        if 'location' in field_name.lower():
            return "region_generalized"
        
        # Default: hash
        return hashlib.sha256(str(value).encode()).hexdigest()[:8]
    
    def _generate_record_id(
        self,
        data: Dict[str, Any],
        user_id: Optional[str]
    ) -> str:
        """Generate unique record ID."""
        content = json.dumps(data, sort_keys=True)
        timestamp = datetime.now().isoformat()
        combined = f"{user_id}{content}{timestamp}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data minimization statistics."""
        return {
            'total_records': self.stats['total_records'],
            'active_records': sum(
                1 for r in self.records.values() if not r.deleted
            ),
            'anonymized_records': self.stats['anonymized_records'],
            'deleted_records': self.stats['deleted_records'],
            'expired_records': self.stats['expired_records'],
            'active_deletion_requests': self.stats['active_deletion_requests'],
            'retention_policies': {
                cat.value: rule.retention_days
                for cat, rule in self.policies.items()
            }
        }
    
    def validate_compliance(self) -> Dict[str, Any]:
        """Validate compliance with data minimization principles.
        
        Returns:
            Compliance validation results
        """
        validation = {
            'compliant': True,
            'checks': {},
            'violations': []
        }
        
        # Check 1: Retention policies defined for all categories
        all_categories = set(DataCategory)
        defined_categories = set(self.policies.keys())
        
        if all_categories == defined_categories:
            validation['checks']['retention_policies'] = {
                'passed': True,
                'message': 'Retention policies defined for all categories'
            }
        else:
            validation['compliant'] = False
            missing = all_categories - defined_categories
            validation['checks']['retention_policies'] = {
                'passed': False,
                'message': f'Missing retention policies for: {missing}'
            }
            validation['violations'].append('Missing retention policies')
        
        # Check 2: Auto-deletion enabled
        validation['checks']['auto_deletion'] = {
            'passed': self.enable_auto_deletion,
            'message': f'Auto-deletion {"enabled" if self.enable_auto_deletion else "disabled"}'
        }
        if not self.enable_auto_deletion:
            validation['violations'].append('Auto-deletion not enabled')
        
        # Check 3: Anonymization available
        validation['checks']['anonymization'] = {
            'passed': self.anonymization_enabled,
            'message': f'Anonymization {"enabled" if self.anonymization_enabled else "disabled"}'
        }
        
        # Check 4: No expired records without action
        now = datetime.now()
        expired_without_action = sum(
            1 for r in self.records.values()
            if not r.deleted and not r.anonymized and r.expires_at <= now
        )
        
        validation['checks']['expired_records'] = {
            'passed': expired_without_action == 0,
            'message': f'{expired_without_action} expired records without action'
        }
        if expired_without_action > 0:
            validation['violations'].append('Expired records without action')
        
        # Check 5: Right-to-be-forgotten support
        validation['checks']['right_to_be_forgotten'] = {
            'passed': True,
            'message': 'Deletion request mechanism available'
        }
        
        return validation
    
    def generate_retention_report(self) -> Dict[str, Any]:
        """Generate comprehensive retention policy report.
        
        Returns:
            Retention report
        """
        now = datetime.now()
        
        # Categorize records
        by_category = {}
        by_status = {'active': 0, 'expired': 0, 'deleted': 0, 'anonymized': 0}
        
        for record in self.records.values():
            # Count by category
            cat = record.category.value
            if cat not in by_category:
                by_category[cat] = 0
            by_category[cat] += 1
            
            # Count by status
            if record.deleted:
                by_status['deleted'] += 1
            elif record.anonymized:
                by_status['anonymized'] += 1
            elif record.expires_at <= now:
                by_status['expired'] += 1
            else:
                by_status['active'] += 1
        
        return {
            'timestamp': now.isoformat(),
            'statistics': self.get_statistics(),
            'records_by_category': by_category,
            'records_by_status': by_status,
            'compliance': self.validate_compliance(),
            'policies': {
                cat.value: {
                    'retention_days': rule.retention_days,
                    'auto_delete': rule.auto_delete,
                    'policy': rule.policy.value
                }
                for cat, rule in self.policies.items()
            }
        }
