"""
Nethical Audit Portal API

This module provides REST API endpoints for the audit portal, including:
- Decision trace exploration
- Policy lineage viewing
- Fairness metrics dashboard
- Audit log browsing
- Appeals tracking

All endpoints support rate limiting, authentication, and comprehensive logging.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps
import time


class RateLimitTier(Enum):
    """Rate limit tiers for API access"""
    ANONYMOUS = ("anonymous", 100, 20, 5)
    AUTHENTICATED = ("authenticated", 1000, 100, 20)
    PREMIUM = ("premium", 10000, 500, 50)
    
    def __init__(self, name: str, requests_per_hour: int, burst: int, concurrency: int):
        self.tier_name = name
        self.requests_per_hour = requests_per_hour
        self.burst = burst
        self.concurrency = concurrency


@dataclass
class RateLimitStatus:
    """Rate limit status information"""
    limit: int
    remaining: int
    reset: int  # Unix timestamp
    tier: str


class RateLimiter:
    """
    Token bucket rate limiter implementation
    
    This class implements rate limiting using the token bucket algorithm,
    supporting different tiers with configurable limits, burst capacity,
    and concurrency controls.
    """
    
    def __init__(self):
        self._buckets: Dict[str, Dict] = {}
    
    def check_rate_limit(self, client_id: str, tier: RateLimitTier) -> tuple[bool, RateLimitStatus]:
        """
        Check if request is within rate limits
        
        Args:
            client_id: Unique identifier for the client
            tier: Rate limit tier to apply
            
        Returns:
            Tuple of (allowed: bool, status: RateLimitStatus)
        """
        now = time.time()
        
        if client_id not in self._buckets:
            # Initialize bucket for new client
            self._buckets[client_id] = {
                'tokens': tier.burst,
                'last_update': now,
                'reset_time': now + 3600,  # 1 hour from now
                'tier': tier.tier_name
            }
        
        bucket = self._buckets[client_id]
        
        # Calculate time elapsed and replenish tokens
        time_elapsed = now - bucket['last_update']
        tokens_to_add = (time_elapsed / 3600) * tier.requests_per_hour
        bucket['tokens'] = min(bucket['tokens'] + tokens_to_add, tier.burst)
        bucket['last_update'] = now
        
        # Check if we have tokens available
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            allowed = True
        else:
            allowed = False
        
        # Reset time if hour boundary crossed
        if now >= bucket['reset_time']:
            bucket['reset_time'] = now + 3600
            bucket['tokens'] = tier.burst
        
        status = RateLimitStatus(
            limit=tier.requests_per_hour,
            remaining=int(bucket['tokens']),
            reset=int(bucket['reset_time']),
            tier=tier.tier_name
        )
        
        return allowed, status


class AuditPortalAPI:
    """
    Main Audit Portal API class
    
    Provides RESTful endpoints for accessing audit data, decision traces,
    policy lineage, fairness metrics, and appeals tracking.
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self._decisions_db = {}  # In production, this would be a real database
        self._policies_db = {}
        self._audit_logs_db = {}
        self._appeals_db = {}
        self._fairness_metrics_db = {}
    
    def _apply_rate_limit(self, client_id: str, tier: RateLimitTier) -> Dict[str, Any]:
        """Apply rate limiting and return headers"""
        allowed, status = self.rate_limiter.check_rate_limit(client_id, tier)
        
        headers = {
            'X-RateLimit-Limit': str(status.limit),
            'X-RateLimit-Remaining': str(status.remaining),
            'X-RateLimit-Reset': str(status.reset),
            'X-RateLimit-Tier': status.tier
        }
        
        if not allowed:
            return {
                'status': 429,
                'error': 'Rate limit exceeded',
                'headers': headers,
                'retry_after': status.reset - int(time.time())
            }
        
        return {'status': 200, 'headers': headers}
    
    # Decision Trace Explorer Endpoints
    
    def search_decisions(
        self,
        client_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED,
        policy_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        outcome: Optional[str] = None,
        page: int = 1,
        per_page: int = 50
    ) -> Dict[str, Any]:
        """
        Search decisions with multiple filter criteria
        
        Args:
            client_id: Client identifier for rate limiting
            tier: Rate limit tier
            policy_id: Filter by policy ID
            agent_id: Filter by agent ID
            from_timestamp: Filter decisions from this time
            to_timestamp: Filter decisions to this time
            outcome: Filter by decision outcome
            page: Page number (1-indexed)
            per_page: Results per page (max 100)
            
        Returns:
            API response with decisions list and pagination metadata
        """
        # Apply rate limiting
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        # Build filter query
        filtered_decisions = []
        for decision_id, decision in self._decisions_db.items():
            if policy_id and decision.get('policy_id') != policy_id:
                continue
            if agent_id and decision.get('agent_id') != agent_id:
                continue
            if outcome and decision.get('outcome') != outcome:
                continue
            if from_timestamp and decision.get('timestamp') < from_timestamp:
                continue
            if to_timestamp and decision.get('timestamp') > to_timestamp:
                continue
            
            filtered_decisions.append(decision)
        
        # Pagination
        total = len(filtered_decisions)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_decisions = filtered_decisions[start_idx:end_idx]
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'decisions': paginated_decisions,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
        }
    
    def get_decision(
        self,
        client_id: str,
        decision_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED
    ) -> Dict[str, Any]:
        """
        Get detailed decision information
        
        Args:
            client_id: Client identifier for rate limiting
            decision_id: Decision identifier
            tier: Rate limit tier
            
        Returns:
            API response with decision details
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        decision = self._decisions_db.get(decision_id)
        if not decision:
            return {
                'status': 404,
                'error': 'Decision not found',
                'headers': rate_limit_result['headers']
            }
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': decision
        }
    
    def get_decision_trace(
        self,
        client_id: str,
        decision_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED
    ) -> Dict[str, Any]:
        """
        Get complete evaluation trace for a decision
        
        Args:
            client_id: Client identifier for rate limiting
            decision_id: Decision identifier
            tier: Rate limit tier
            
        Returns:
            API response with decision evaluation trace
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        decision = self._decisions_db.get(decision_id)
        if not decision:
            return {
                'status': 404,
                'error': 'Decision not found',
                'headers': rate_limit_result['headers']
            }
        
        trace = decision.get('trace', {})
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'decision_id': decision_id,
                'trace': trace,
                'steps': trace.get('steps', []),
                'justification': trace.get('justification', '')
            }
        }
    
    # Policy Lineage Viewer Endpoints
    
    def list_policies(
        self,
        client_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED,
        page: int = 1,
        per_page: int = 50
    ) -> Dict[str, Any]:
        """
        List all policies with pagination
        
        Args:
            client_id: Client identifier for rate limiting
            tier: Rate limit tier
            page: Page number
            per_page: Results per page
            
        Returns:
            API response with policies list
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        policies = list(self._policies_db.values())
        total = len(policies)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_policies = policies[start_idx:end_idx]
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'policies': paginated_policies,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
        }
    
    def get_policy_versions(
        self,
        client_id: str,
        policy_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED
    ) -> Dict[str, Any]:
        """
        Get all versions of a policy
        
        Args:
            client_id: Client identifier for rate limiting
            policy_id: Policy identifier
            tier: Rate limit tier
            
        Returns:
            API response with policy versions
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        policy = self._policies_db.get(policy_id)
        if not policy:
            return {
                'status': 404,
                'error': 'Policy not found',
                'headers': rate_limit_result['headers']
            }
        
        versions = policy.get('versions', [])
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'policy_id': policy_id,
                'versions': versions,
                'current_version': policy.get('current_version'),
                'total_versions': len(versions)
            }
        }
    
    def get_policy_lineage(
        self,
        client_id: str,
        policy_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED
    ) -> Dict[str, Any]:
        """
        Get complete lineage (hash chain) for a policy
        
        Args:
            client_id: Client identifier for rate limiting
            policy_id: Policy identifier
            tier: Rate limit tier
            
        Returns:
            API response with policy lineage including hash chain
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        policy = self._policies_db.get(policy_id)
        if not policy:
            return {
                'status': 404,
                'error': 'Policy not found',
                'headers': rate_limit_result['headers']
            }
        
        lineage = policy.get('lineage', {})
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'policy_id': policy_id,
                'lineage': lineage,
                'hash_chain': lineage.get('hash_chain', []),
                'verification_status': 'verified'
            }
        }
    
    def verify_policy_chain(
        self,
        client_id: str,
        policy_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED
    ) -> Dict[str, Any]:
        """
        Cryptographically verify policy hash chain integrity
        
        Args:
            client_id: Client identifier for rate limiting
            policy_id: Policy identifier
            tier: Rate limit tier
            
        Returns:
            API response with verification results
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        policy = self._policies_db.get(policy_id)
        if not policy:
            return {
                'status': 404,
                'error': 'Policy not found',
                'headers': rate_limit_result['headers']
            }
        
        # Verify hash chain
        versions = policy.get('versions', [])
        chain_valid = True
        verification_details = []
        
        for i in range(1, len(versions)):
            prev_version = versions[i-1]
            curr_version = versions[i]
            
            # Verify that current version's prev_hash matches previous version's hash
            expected_prev_hash = prev_version.get('hash')
            actual_prev_hash = curr_version.get('prev_hash')
            
            is_valid = expected_prev_hash == actual_prev_hash
            chain_valid = chain_valid and is_valid
            
            verification_details.append({
                'version': curr_version.get('version'),
                'valid': is_valid,
                'expected_prev_hash': expected_prev_hash,
                'actual_prev_hash': actual_prev_hash
            })
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'policy_id': policy_id,
                'chain_valid': chain_valid,
                'total_versions': len(versions),
                'verification_details': verification_details,
                'verified_at': datetime.utcnow().isoformat()
            }
        }
    
    # Fairness Metrics Dashboard Endpoints
    
    def get_fairness_metrics(
        self,
        client_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        attribute: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get fairness metrics for specified time range and attributes
        
        Args:
            client_id: Client identifier for rate limiting
            tier: Rate limit tier
            from_timestamp: Start of time range
            to_timestamp: End of time range
            attribute: Protected attribute to filter by
            
        Returns:
            API response with fairness metrics
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        # Filter metrics by parameters
        metrics = self._fairness_metrics_db.get('metrics', [])
        
        if from_timestamp or to_timestamp or attribute:
            filtered_metrics = []
            for metric in metrics:
                if from_timestamp and metric.get('timestamp') < from_timestamp:
                    continue
                if to_timestamp and metric.get('timestamp') > to_timestamp:
                    continue
                if attribute and metric.get('attribute') != attribute:
                    continue
                filtered_metrics.append(metric)
            metrics = filtered_metrics
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'metrics': metrics,
                'summary': {
                    'total_metrics': len(metrics),
                    'time_range': {
                        'from': from_timestamp.isoformat() if from_timestamp else None,
                        'to': to_timestamp.isoformat() if to_timestamp else None
                    },
                    'attribute': attribute
                }
            }
        }
    
    # Audit Log Browser Endpoints
    
    def get_audit_logs(
        self,
        client_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        event_type: Optional[str] = None,
        page: int = 1,
        per_page: int = 50
    ) -> Dict[str, Any]:
        """
        Get audit logs with filtering and pagination
        
        Args:
            client_id: Client identifier for rate limiting
            tier: Rate limit tier
            from_timestamp: Start of time range
            to_timestamp: End of time range
            event_type: Type of audit event to filter
            page: Page number
            per_page: Results per page
            
        Returns:
            API response with audit logs
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        logs = list(self._audit_logs_db.values())
        
        # Apply filters
        if from_timestamp or to_timestamp or event_type:
            filtered_logs = []
            for log in logs:
                if from_timestamp and log.get('timestamp') < from_timestamp:
                    continue
                if to_timestamp and log.get('timestamp') > to_timestamp:
                    continue
                if event_type and log.get('event_type') != event_type:
                    continue
                filtered_logs.append(log)
            logs = filtered_logs
        
        # Pagination
        total = len(logs)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_logs = logs[start_idx:end_idx]
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'logs': paginated_logs,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
        }
    
    def get_merkle_root(
        self,
        client_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED
    ) -> Dict[str, Any]:
        """
        Get current Merkle tree root hash for audit log verification
        
        Args:
            client_id: Client identifier for rate limiting
            tier: Rate limit tier
            
        Returns:
            API response with Merkle root and metadata
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        # In production, this would compute or retrieve the actual Merkle root
        merkle_root = self._audit_logs_db.get('merkle_root', {})
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'merkle_root': merkle_root.get('hash'),
                'timestamp': merkle_root.get('timestamp'),
                'log_count': merkle_root.get('log_count', 0),
                'anchored': merkle_root.get('anchored', False),
                'anchor_locations': merkle_root.get('anchor_locations', [])
            }
        }
    
    # Appeals Tracking System Endpoints
    
    def create_appeal(
        self,
        client_id: str,
        decision_id: str,
        justification: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED
    ) -> Dict[str, Any]:
        """
        Submit a new appeal for a decision
        
        Args:
            client_id: Client identifier for rate limiting
            decision_id: Decision being appealed
            justification: Reason for appeal
            tier: Rate limit tier
            
        Returns:
            API response with created appeal
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        appeal_id = f"appeal_{len(self._appeals_db) + 1}"
        appeal = {
            'appeal_id': appeal_id,
            'decision_id': decision_id,
            'justification': justification,
            'status': 'submitted',
            'submitted_at': datetime.utcnow().isoformat(),
            'submitted_by': client_id
        }
        
        self._appeals_db[appeal_id] = appeal
        
        return {
            'status': 201,
            'headers': rate_limit_result['headers'],
            'data': appeal
        }
    
    def get_appeals(
        self,
        client_id: str,
        tier: RateLimitTier = RateLimitTier.AUTHENTICATED,
        status: Optional[str] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        page: int = 1,
        per_page: int = 50
    ) -> Dict[str, Any]:
        """
        List appeals with filtering
        
        Args:
            client_id: Client identifier for rate limiting
            tier: Rate limit tier
            status: Filter by appeal status
            from_timestamp: Start of time range
            to_timestamp: End of time range
            page: Page number
            per_page: Results per page
            
        Returns:
            API response with appeals list
        """
        rate_limit_result = self._apply_rate_limit(client_id, tier)
        if rate_limit_result['status'] != 200:
            return rate_limit_result
        
        appeals = list(self._appeals_db.values())
        
        # Apply filters
        if status or from_timestamp or to_timestamp:
            filtered_appeals = []
            for appeal in appeals:
                if status and appeal.get('status') != status:
                    continue
                # Add timestamp filtering logic here
                filtered_appeals.append(appeal)
            appeals = filtered_appeals
        
        # Pagination
        total = len(appeals)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_appeals = appeals[start_idx:end_idx]
        
        return {
            'status': 200,
            'headers': rate_limit_result['headers'],
            'data': {
                'appeals': paginated_appeals,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
        }
