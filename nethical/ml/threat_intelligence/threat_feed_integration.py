"""
Threat Feed Integration Module.

Integrates multiple threat intelligence sources to provide real-time
threat awareness for the Nethical detection system.

Phase: 5 - Detection Omniscience
Component: Threat Anticipation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreatSource(Enum):
    """Enumeration of threat intelligence sources."""
    
    CVE_DATABASE = "cve_database"
    AI_RESEARCH_FEEDS = "ai_research_feeds"
    INDUSTRY_SHARING = "industry_sharing"
    INTERNAL_HONEYPOT = "internal_honeypot"
    RED_TEAM_FINDINGS = "red_team_findings"
    OPENSOURCE_INTEL = "opensource_intel"
    VENDOR_ADVISORIES = "vendor_advisories"


class ThreatSeverity(Enum):
    """Severity levels for threat intelligence."""
    
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"         # Urgent attention needed
    MEDIUM = "medium"     # Monitor and plan response
    LOW = "low"          # Informational, low priority
    INFO = "info"        # General awareness


@dataclass
class ThreatIntelligence:
    """Represents a piece of threat intelligence."""
    
    threat_id: str
    source: ThreatSource
    severity: ThreatSeverity
    title: str
    description: str
    indicators: List[str] = field(default_factory=list)
    attack_vectors: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    mitigation_steps: List[str] = field(default_factory=list)
    cve_ids: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "threat_id": self.threat_id,
            "source": self.source.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "indicators": self.indicators,
            "attack_vectors": self.attack_vectors,
            "affected_systems": self.affected_systems,
            "mitigation_steps": self.mitigation_steps,
            "cve_ids": self.cve_ids,
            "references": self.references,
            "discovered_at": self.discovered_at.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class ThreatFeedIntegrator:
    """
    Integrates and manages threat intelligence from multiple sources.
    
    Features:
    - Multi-source threat feed aggregation
    - Deduplication and correlation
    - Severity-based prioritization
    - Real-time alert generation
    - Historical threat tracking
    """
    
    def __init__(
        self,
        sources: Optional[List[ThreatSource]] = None,
        refresh_interval: int = 3600,  # seconds
        max_age_days: int = 90,
    ):
        """
        Initialize threat feed integrator.
        
        Args:
            sources: List of threat sources to monitor
            refresh_interval: How often to refresh feeds (seconds)
            max_age_days: Maximum age of threats to retain
        """
        self.sources = sources or [
            ThreatSource.CVE_DATABASE,
            ThreatSource.AI_RESEARCH_FEEDS,
            ThreatSource.INDUSTRY_SHARING,
            ThreatSource.INTERNAL_HONEYPOT,
            ThreatSource.RED_TEAM_FINDINGS,
        ]
        self.refresh_interval = refresh_interval
        self.max_age_days = max_age_days
        
        # Storage
        self.threats: Dict[str, ThreatIntelligence] = {}
        self.threats_by_source: Dict[ThreatSource, Set[str]] = defaultdict(set)
        self.threats_by_severity: Dict[ThreatSeverity, Set[str]] = defaultdict(set)
        
        # Statistics
        self.last_refresh: Optional[datetime] = None
        self.total_threats_ingested: int = 0
        self.active_threats: int = 0
        
        logger.info(
            f"ThreatFeedIntegrator initialized with {len(self.sources)} sources"
        )
    
    async def ingest_threat(
        self, threat: ThreatIntelligence
    ) -> Dict[str, Any]:
        """
        Ingest a new threat intelligence item.
        
        Args:
            threat: Threat intelligence to ingest
            
        Returns:
            Ingestion result with status and metadata
        """
        try:
            # Check for duplicates
            is_update = threat.threat_id in self.threats
            
            if is_update:
                logger.info(f"Threat {threat.threat_id} already exists, updating")
                existing = self.threats[threat.threat_id]
                
                # Merge indicators and references
                threat.indicators = list(
                    set(existing.indicators + threat.indicators)
                )
                threat.references = list(
                    set(existing.references + threat.references)
                )
                
                # Use higher confidence
                threat.confidence = max(existing.confidence, threat.confidence)
            
            # Store threat
            self.threats[threat.threat_id] = threat
            self.threats_by_source[threat.source].add(threat.threat_id)
            self.threats_by_severity[threat.severity].add(threat.threat_id)
            
            self.total_threats_ingested += 1
            self.active_threats = len(self.threats)
            
            logger.info(
                f"Ingested threat {threat.threat_id} from {threat.source.value} "
                f"(severity: {threat.severity.value})"
            )
            
            return {
                "status": "success",
                "threat_id": threat.threat_id,
                "action": "updated" if is_update else "created",
                "severity": threat.severity.value,
            }
            
        except Exception as e:
            logger.error(f"Error ingesting threat {threat.threat_id}: {e}")
            return {
                "status": "error",
                "threat_id": threat.threat_id,
                "error": str(e),
            }
    
    async def refresh_feeds(self) -> Dict[str, Any]:
        """
        Refresh all configured threat feeds.
        
        Returns:
            Refresh statistics and status
        """
        try:
            refresh_start = datetime.now(timezone.utc)
            new_threats = 0
            
            # Simulate feed refresh for each source
            # In production, this would fetch from actual threat feeds
            for source in self.sources:
                threats = await self._fetch_from_source(source)
                for threat in threats:
                    result = await self.ingest_threat(threat)
                    if result["status"] == "success" and result.get("action") == "created":
                        new_threats += 1
            
            # Clean up old threats
            await self._cleanup_old_threats()
            
            self.last_refresh = datetime.now(timezone.utc)
            refresh_time = (self.last_refresh - refresh_start).total_seconds()
            
            logger.info(
                f"Feed refresh completed in {refresh_time:.2f}s: "
                f"{new_threats} new threats, {self.active_threats} total active"
            )
            
            return {
                "status": "success",
                "new_threats": new_threats,
                "active_threats": self.active_threats,
                "refresh_time_seconds": refresh_time,
                "last_refresh": self.last_refresh.isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error refreshing feeds: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    async def _fetch_from_source(
        self, source: ThreatSource
    ) -> List[ThreatIntelligence]:
        """
        Fetch threats from a specific source.
        
        Args:
            source: Threat source to fetch from
            
        Returns:
            List of threat intelligence items
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # In production, this would make actual API calls to threat feeds
        # For now, return empty list
        # Actual implementation would integrate with:
        # - NVD API for CVE data
        # - arXiv/research aggregators for AI security papers
        # - ISAC feeds for industry sharing
        # - Internal honeypot/red team logs
        
        return []
    
    async def _cleanup_old_threats(self):
        """Remove threats older than max_age_days."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        
        threats_to_remove = [
            threat_id
            for threat_id, threat in self.threats.items()
            if threat.discovered_at < cutoff_date
        ]
        
        for threat_id in threats_to_remove:
            threat = self.threats[threat_id]
            
            # Remove from indices
            self.threats_by_source[threat.source].discard(threat_id)
            self.threats_by_severity[threat.severity].discard(threat_id)
            
            # Remove from main storage
            del self.threats[threat_id]
        
        if threats_to_remove:
            logger.info(f"Cleaned up {len(threats_to_remove)} old threats")
            self.active_threats = len(self.threats)
    
    async def get_threats_by_severity(
        self, severity: ThreatSeverity
    ) -> List[ThreatIntelligence]:
        """
        Get all threats of a specific severity.
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of matching threats
        """
        threat_ids = self.threats_by_severity.get(severity, set())
        return [self.threats[tid] for tid in threat_ids if tid in self.threats]
    
    async def get_threats_by_source(
        self, source: ThreatSource
    ) -> List[ThreatIntelligence]:
        """
        Get all threats from a specific source.
        
        Args:
            source: Source to filter by
            
        Returns:
            List of matching threats
        """
        threat_ids = self.threats_by_source.get(source, set())
        return [self.threats[tid] for tid in threat_ids if tid in self.threats]
    
    async def search_threats(
        self,
        keywords: Optional[List[str]] = None,
        attack_vectors: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[ThreatIntelligence]:
        """
        Search threats by keywords and filters.
        
        Args:
            keywords: Keywords to search in title/description
            attack_vectors: Filter by attack vectors
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching threats
        """
        results = []
        
        for threat in self.threats.values():
            # Confidence filter
            if threat.confidence < min_confidence:
                continue
            
            # Attack vector filter
            if attack_vectors:
                if not any(av in threat.attack_vectors for av in attack_vectors):
                    continue
            
            # Keyword filter
            if keywords:
                text = f"{threat.title} {threat.description}".lower()
                if not any(kw.lower() in text for kw in keywords):
                    continue
            
            results.append(threat)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get threat feed statistics.
        
        Returns:
            Dictionary of statistics
        """
        severity_counts = {
            severity.value: len(threat_ids)
            for severity, threat_ids in self.threats_by_severity.items()
        }
        
        source_counts = {
            source.value: len(threat_ids)
            for source, threat_ids in self.threats_by_source.items()
        }
        
        return {
            "total_threats_ingested": self.total_threats_ingested,
            "active_threats": self.active_threats,
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "threats_by_severity": severity_counts,
            "threats_by_source": source_counts,
            "configured_sources": [s.value for s in self.sources],
        }
