"""
Phase 5.1: Comprehensive Threat Modeling Framework

This module provides military-grade threat modeling capabilities using STRIDE methodology,
attack tree analysis, threat intelligence integration, and automated security requirements
traceability for government, military, and healthcare deployments.

STRIDE Categories:
- Spoofing: Identity verification and authentication threats
- Tampering: Data integrity and manipulation threats
- Repudiation: Non-repudiation and audit logging threats
- Information Disclosure: Confidentiality and data leakage threats
- Denial of Service: Availability and resource exhaustion threats
- Elevation of Privilege: Authorization and access control threats
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib


class ThreatCategory(Enum):
    """STRIDE threat categories."""
    SPOOFING = "spoofing"
    TAMPERING = "tampering"
    REPUDIATION = "repudiation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    ELEVATION_OF_PRIVILEGE = "elevation_of_privilege"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatStatus(Enum):
    """Threat mitigation status."""
    IDENTIFIED = "identified"
    ANALYZED = "analyzed"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    MONITORING = "monitoring"


@dataclass
class Threat:
    """Individual threat model entry."""
    id: str
    category: ThreatCategory
    title: str
    description: str
    severity: ThreatSeverity
    status: ThreatStatus
    affected_components: List[str]
    attack_vectors: List[str]
    mitigations: List[str] = field(default_factory=list)
    residual_risk: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert threat to dictionary."""
        return {
            'id': self.id,
            'category': self.category.value,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'affected_components': self.affected_components,
            'attack_vectors': self.attack_vectors,
            'mitigations': self.mitigations,
            'residual_risk': self.residual_risk,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'references': self.references
        }


@dataclass
class AttackTreeNode:
    """Node in an attack tree representing an attack step."""
    id: str
    name: str
    description: str
    is_and_gate: bool = False  # True for AND gate, False for OR gate
    children: List['AttackTreeNode'] = field(default_factory=list)
    probability: float = 0.0  # Probability of success (0-1)
    impact: float = 0.0  # Impact if successful (0-1)
    cost_to_attacker: float = 0.0  # Estimated cost in arbitrary units
    mitigations: List[str] = field(default_factory=list)
    
    def calculate_risk(self) -> float:
        """Calculate risk score based on probability and impact."""
        return self.probability * self.impact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert attack tree node to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'is_and_gate': self.is_and_gate,
            'children': [child.to_dict() for child in self.children],
            'probability': self.probability,
            'impact': self.impact,
            'cost_to_attacker': self.cost_to_attacker,
            'risk_score': self.calculate_risk(),
            'mitigations': self.mitigations
        }


@dataclass
class SecurityRequirement:
    """Security requirement with traceability."""
    id: str
    title: str
    description: str
    category: str
    priority: str  # critical, high, medium, low
    related_threats: List[str] = field(default_factory=list)
    implemented_in: List[str] = field(default_factory=list)  # List of code components
    test_cases: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)  # NIST, HIPAA, etc.
    status: str = "draft"  # draft, approved, implemented, verified
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security requirement to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'priority': self.priority,
            'related_threats': self.related_threats,
            'implemented_in': self.implemented_in,
            'test_cases': self.test_cases,
            'compliance_frameworks': self.compliance_frameworks,
            'status': self.status
        }


class ThreatIntelligenceFeed:
    """Threat intelligence integration."""
    
    def __init__(self):
        """Initialize threat intelligence feed."""
        self.indicators: Dict[str, Dict[str, Any]] = {}
        self.last_update: Optional[datetime] = None
    
    def add_indicator(
        self,
        indicator_type: str,
        value: str,
        severity: ThreatSeverity,
        description: str,
        source: str
    ) -> str:
        """Add a threat indicator."""
        indicator_id = hashlib.sha256(f"{indicator_type}:{value}".encode()).hexdigest()[:16]
        
        self.indicators[indicator_id] = {
            'type': indicator_type,
            'value': value,
            'severity': severity.value,
            'description': description,
            'source': source,
            'added_at': datetime.now().isoformat()
        }
        
        self.last_update = datetime.now()
        return indicator_id
    
    def get_indicators(self, indicator_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get threat indicators, optionally filtered by type."""
        if indicator_type:
            return [
                ind for ind in self.indicators.values()
                if ind['type'] == indicator_type
            ]
        return list(self.indicators.values())
    
    def check_indicator(self, value: str) -> Optional[Dict[str, Any]]:
        """Check if a value matches any known threat indicators."""
        for indicator in self.indicators.values():
            if indicator['value'] == value:
                return indicator
        return None


class STRIDEAnalyzer:
    """STRIDE threat analysis engine."""
    
    def __init__(self):
        """Initialize STRIDE analyzer."""
        self.threats: Dict[str, Threat] = {}
        self.components: Set[str] = set()
    
    def add_threat(
        self,
        category: ThreatCategory,
        title: str,
        description: str,
        severity: ThreatSeverity,
        affected_components: List[str],
        attack_vectors: List[str],
        mitigations: Optional[List[str]] = None
    ) -> str:
        """Add a new threat to the model."""
        threat_id = hashlib.sha256(
            f"{category.value}:{title}:{','.join(affected_components)}".encode()
        ).hexdigest()[:16]
        
        threat = Threat(
            id=threat_id,
            category=category,
            title=title,
            description=description,
            severity=severity,
            status=ThreatStatus.IDENTIFIED,
            affected_components=affected_components,
            attack_vectors=attack_vectors,
            mitigations=mitigations or []
        )
        
        self.threats[threat_id] = threat
        self.components.update(affected_components)
        
        return threat_id
    
    def update_threat_status(self, threat_id: str, status: ThreatStatus) -> None:
        """Update the status of a threat."""
        if threat_id in self.threats:
            self.threats[threat_id].status = status
            self.threats[threat_id].updated_at = datetime.now()
    
    def get_threats_by_category(self, category: ThreatCategory) -> List[Threat]:
        """Get all threats in a specific STRIDE category."""
        return [
            threat for threat in self.threats.values()
            if threat.category == category
        ]
    
    def get_threats_by_severity(self, severity: ThreatSeverity) -> List[Threat]:
        """Get all threats with a specific severity."""
        return [
            threat for threat in self.threats.values()
            if threat.severity == severity
        ]
    
    def get_threats_by_component(self, component: str) -> List[Threat]:
        """Get all threats affecting a specific component."""
        return [
            threat for threat in self.threats.values()
            if component in threat.affected_components
        ]
    
    def generate_stride_report(self) -> Dict[str, Any]:
        """Generate a comprehensive STRIDE analysis report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_threats': len(self.threats),
            'total_components': len(self.components),
            'by_category': {},
            'by_severity': {},
            'by_status': {},
            'critical_threats': []
        }
        
        # Count by category
        for category in ThreatCategory:
            threats = self.get_threats_by_category(category)
            report['by_category'][category.value] = {
                'count': len(threats),
                'threats': [t.to_dict() for t in threats]
            }
        
        # Count by severity
        for severity in ThreatSeverity:
            threats = self.get_threats_by_severity(severity)
            report['by_severity'][severity.value] = len(threats)
        
        # Count by status
        for status in ThreatStatus:
            threats = [t for t in self.threats.values() if t.status == status]
            report['by_status'][status.value] = len(threats)
        
        # List critical threats
        critical_threats = self.get_threats_by_severity(ThreatSeverity.CRITICAL)
        report['critical_threats'] = [t.to_dict() for t in critical_threats]
        
        return report


class AttackTreeAnalyzer:
    """Attack tree analysis engine."""
    
    def __init__(self):
        """Initialize attack tree analyzer."""
        self.trees: Dict[str, AttackTreeNode] = {}
    
    def create_attack_tree(
        self,
        tree_id: str,
        root_name: str,
        root_description: str
    ) -> AttackTreeNode:
        """Create a new attack tree."""
        root = AttackTreeNode(
            id=tree_id,
            name=root_name,
            description=root_description
        )
        self.trees[tree_id] = root
        return root
    
    def add_child_node(
        self,
        parent: AttackTreeNode,
        node_id: str,
        name: str,
        description: str,
        is_and_gate: bool = False,
        probability: float = 0.0,
        impact: float = 0.0,
        cost_to_attacker: float = 0.0
    ) -> AttackTreeNode:
        """Add a child node to an attack tree."""
        child = AttackTreeNode(
            id=node_id,
            name=name,
            description=description,
            is_and_gate=is_and_gate,
            probability=probability,
            impact=impact,
            cost_to_attacker=cost_to_attacker
        )
        parent.children.append(child)
        return child
    
    def calculate_tree_risk(self, node: AttackTreeNode) -> float:
        """Calculate cumulative risk for an attack tree."""
        if not node.children:
            return node.calculate_risk()
        
        if node.is_and_gate:
            # AND gate: all children must succeed
            child_probs = [self.calculate_tree_risk(child) for child in node.children]
            # Multiply probabilities for AND gate
            combined_prob = 1.0
            for prob in child_probs:
                combined_prob *= prob
            return combined_prob * node.impact
        else:
            # OR gate: at least one child must succeed
            child_risks = [self.calculate_tree_risk(child) for child in node.children]
            # Take maximum risk for OR gate
            return max(child_risks) if child_risks else 0.0
    
    def export_attack_tree(self, tree_id: str) -> Optional[Dict[str, Any]]:
        """Export an attack tree to a dictionary."""
        if tree_id not in self.trees:
            return None
        
        tree = self.trees[tree_id]
        return {
            'tree_id': tree_id,
            'tree': tree.to_dict(),
            'total_risk': self.calculate_tree_risk(tree)
        }


class SecurityRequirementsTraceability:
    """Security requirements traceability matrix."""
    
    def __init__(self):
        """Initialize traceability matrix."""
        self.requirements: Dict[str, SecurityRequirement] = {}
    
    def add_requirement(
        self,
        req_id: str,
        title: str,
        description: str,
        category: str,
        priority: str,
        compliance_frameworks: Optional[List[str]] = None
    ) -> SecurityRequirement:
        """Add a new security requirement."""
        requirement = SecurityRequirement(
            id=req_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            compliance_frameworks=compliance_frameworks or []
        )
        self.requirements[req_id] = requirement
        return requirement
    
    def link_to_threat(self, req_id: str, threat_id: str) -> None:
        """Link a requirement to a threat."""
        if req_id in self.requirements:
            if threat_id not in self.requirements[req_id].related_threats:
                self.requirements[req_id].related_threats.append(threat_id)
    
    def link_to_implementation(self, req_id: str, component: str) -> None:
        """Link a requirement to an implementation component."""
        if req_id in self.requirements:
            if component not in self.requirements[req_id].implemented_in:
                self.requirements[req_id].implemented_in.append(component)
    
    def link_to_test(self, req_id: str, test_case: str) -> None:
        """Link a requirement to a test case."""
        if req_id in self.requirements:
            if test_case not in self.requirements[req_id].test_cases:
                self.requirements[req_id].test_cases.append(test_case)
    
    def update_status(self, req_id: str, status: str) -> None:
        """Update requirement status."""
        if req_id in self.requirements:
            self.requirements[req_id].status = status
    
    def get_traceability_matrix(self) -> Dict[str, Any]:
        """Generate traceability matrix report."""
        return {
            'generated_at': datetime.now().isoformat(),
            'total_requirements': len(self.requirements),
            'requirements': [req.to_dict() for req in self.requirements.values()],
            'coverage_stats': self._calculate_coverage_stats()
        }
    
    def _calculate_coverage_stats(self) -> Dict[str, Any]:
        """Calculate coverage statistics."""
        total = len(self.requirements)
        if total == 0:
            return {
                'implementation_coverage': 0.0,
                'test_coverage': 0.0,
                'threat_coverage': 0.0
            }
        
        implemented = sum(1 for req in self.requirements.values() if req.implemented_in)
        tested = sum(1 for req in self.requirements.values() if req.test_cases)
        linked_to_threats = sum(1 for req in self.requirements.values() if req.related_threats)
        
        return {
            'implementation_coverage': implemented / total,
            'test_coverage': tested / total,
            'threat_coverage': linked_to_threats / total
        }


class ThreatModelingFramework:
    """Comprehensive threat modeling framework integrating all components."""
    
    def __init__(self):
        """Initialize threat modeling framework."""
        self.stride_analyzer = STRIDEAnalyzer()
        self.attack_tree_analyzer = AttackTreeAnalyzer()
        self.requirements_matrix = SecurityRequirementsTraceability()
        self.threat_intelligence = ThreatIntelligenceFeed()
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'last_updated': datetime.now().isoformat()
        }
    
    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive threat modeling report."""
        self.update_timestamp()
        
        return {
            'metadata': self.metadata,
            'stride_analysis': self.stride_analyzer.generate_stride_report(),
            'attack_trees': {
                tree_id: self.attack_tree_analyzer.export_attack_tree(tree_id)
                for tree_id in self.attack_tree_analyzer.trees.keys()
            },
            'requirements_traceability': self.requirements_matrix.get_traceability_matrix(),
            'threat_intelligence': {
                'total_indicators': len(self.threat_intelligence.indicators),
                'last_update': self.threat_intelligence.last_update.isoformat() if self.threat_intelligence.last_update else None,
                'indicators': self.threat_intelligence.get_indicators()
            }
        }
    
    def export_to_json(self, filepath: str) -> None:
        """Export threat model to JSON file."""
        report = self.generate_comprehensive_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def import_from_json(self, filepath: str) -> None:
        """Import threat model from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Import metadata
        if 'metadata' in data:
            self.metadata.update(data['metadata'])
        
        # Import STRIDE threats
        if 'stride_analysis' in data and 'by_category' in data['stride_analysis']:
            for category_data in data['stride_analysis']['by_category'].values():
                for threat_data in category_data.get('threats', []):
                    self.stride_analyzer.add_threat(
                        category=ThreatCategory(threat_data['category']),
                        title=threat_data['title'],
                        description=threat_data['description'],
                        severity=ThreatSeverity(threat_data['severity']),
                        affected_components=threat_data['affected_components'],
                        attack_vectors=threat_data['attack_vectors'],
                        mitigations=threat_data.get('mitigations', [])
                    )
