"""
Phase 5.2: Penetration Testing Program Framework

This module provides comprehensive penetration testing capabilities including vulnerability
scanning, test report generation, remediation tracking, red team engagement, purple team
collaboration, and bug bounty program integration for military-grade security validation.

Capabilities:
- Automated vulnerability scanning
- Penetration test lifecycle management
- Vulnerability remediation tracking
- Red team engagement coordination
- Purple team collaboration tools
- Bug bounty program integration
- Compliance reporting (FedRAMP, NIST, HIPAA)
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels aligned with CVSS."""
    CRITICAL = "critical"  # CVSS 9.0-10.0
    HIGH = "high"          # CVSS 7.0-8.9
    MEDIUM = "medium"      # CVSS 4.0-6.9
    LOW = "low"            # CVSS 0.1-3.9
    INFO = "info"          # CVSS 0.0


class VulnerabilityStatus(Enum):
    """Vulnerability lifecycle status."""
    DISCOVERED = "discovered"
    CONFIRMED = "confirmed"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    FIXED = "fixed"
    VERIFIED = "verified"
    CLOSED = "closed"
    ACCEPTED_RISK = "accepted_risk"


class TestType(Enum):
    """Types of penetration tests."""
    BLACK_BOX = "black_box"          # No prior knowledge
    GRAY_BOX = "gray_box"            # Partial knowledge
    WHITE_BOX = "white_box"          # Full knowledge
    RED_TEAM = "red_team"            # Adversarial simulation
    PURPLE_TEAM = "purple_team"      # Collaborative exercise
    BUG_BOUNTY = "bug_bounty"        # External researchers


class TestStatus(Enum):
    """Penetration test status."""
    PLANNED = "planned"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REPORT_PENDING = "report_pending"
    REPORT_DELIVERED = "report_delivered"
    CLOSED = "closed"


@dataclass
class Vulnerability:
    """Individual vulnerability finding."""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    status: VulnerabilityStatus
    cvss_score: float  # Common Vulnerability Scoring System score
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    affected_components: List[str] = field(default_factory=list)
    attack_vector: str = ""
    proof_of_concept: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    discovered_by: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    target_fix_date: Optional[datetime] = None
    fixed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'cvss_score': self.cvss_score,
            'cwe_id': self.cwe_id,
            'affected_components': self.affected_components,
            'attack_vector': self.attack_vector,
            'proof_of_concept': self.proof_of_concept,
            'remediation_steps': self.remediation_steps,
            'discovered_by': self.discovered_by,
            'discovered_at': self.discovered_at.isoformat(),
            'assigned_to': self.assigned_to,
            'target_fix_date': self.target_fix_date.isoformat() if self.target_fix_date else None,
            'fixed_at': self.fixed_at.isoformat() if self.fixed_at else None,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'references': self.references
        }
    
    def calculate_sla_compliance(self) -> bool:
        """Check if vulnerability fix meets SLA based on severity."""
        if not self.target_fix_date:
            return True  # No SLA set
        
        if self.status in [VulnerabilityStatus.FIXED, VulnerabilityStatus.VERIFIED, VulnerabilityStatus.CLOSED]:
            if self.fixed_at:
                return self.fixed_at <= self.target_fix_date
        
        # Still open, check if past due
        return datetime.now() <= self.target_fix_date


@dataclass
class PenetrationTest:
    """Penetration test engagement."""
    id: str
    title: str
    description: str
    test_type: TestType
    status: TestStatus
    scope: List[str]  # Systems/components in scope
    out_of_scope: List[str] = field(default_factory=list)
    tester_team: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    findings: List[str] = field(default_factory=list)  # Vulnerability IDs
    report_path: Optional[str] = None
    executive_summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert penetration test to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'test_type': self.test_type.value,
            'status': self.status.value,
            'scope': self.scope,
            'out_of_scope': self.out_of_scope,
            'tester_team': self.tester_team,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'findings': self.findings,
            'report_path': self.report_path,
            'executive_summary': self.executive_summary,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class RedTeamEngagement:
    """Red team adversarial engagement."""
    id: str
    name: str
    objectives: List[str]
    tactics: List[str]  # MITRE ATT&CK tactics
    techniques: List[str]  # MITRE ATT&CK techniques
    start_date: datetime
    end_date: datetime
    team_members: List[str]
    target_systems: List[str]
    rules_of_engagement: List[str]
    findings: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    detection_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert red team engagement to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'objectives': self.objectives,
            'tactics': self.tactics,
            'techniques': self.techniques,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'team_members': self.team_members,
            'target_systems': self.target_systems,
            'rules_of_engagement': self.rules_of_engagement,
            'findings': self.findings,
            'success_rate': self.success_rate,
            'detection_rate': self.detection_rate
        }


@dataclass
class PurpleTeamExercise:
    """Purple team collaborative security exercise."""
    id: str
    name: str
    description: str
    red_team: List[str]
    blue_team: List[str]
    scenarios: List[str]
    start_date: datetime
    end_date: datetime
    objectives: List[str]
    lessons_learned: List[str] = field(default_factory=list)
    improvements_identified: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert purple team exercise to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'red_team': self.red_team,
            'blue_team': self.blue_team,
            'scenarios': self.scenarios,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'objectives': self.objectives,
            'lessons_learned': self.lessons_learned,
            'improvements_identified': self.improvements_identified
        }


class VulnerabilityScanner:
    """Automated vulnerability scanning engine."""
    
    def __init__(self):
        """Initialize vulnerability scanner."""
        self.vulnerabilities: Dict[str, Vulnerability] = {}
        self.scan_history: List[Dict[str, Any]] = []
    
    def register_vulnerability(
        self,
        title: str,
        description: str,
        severity: VulnerabilitySeverity,
        cvss_score: float,
        affected_components: List[str],
        attack_vector: str,
        discovered_by: str,
        cwe_id: Optional[str] = None,
        proof_of_concept: str = "",
        remediation_steps: Optional[List[str]] = None
    ) -> str:
        """Register a new vulnerability."""
        vuln_id = hashlib.sha256(
            f"{title}:{','.join(affected_components)}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        vulnerability = Vulnerability(
            id=vuln_id,
            title=title,
            description=description,
            severity=severity,
            status=VulnerabilityStatus.DISCOVERED,
            cvss_score=cvss_score,
            cwe_id=cwe_id,
            affected_components=affected_components,
            attack_vector=attack_vector,
            proof_of_concept=proof_of_concept,
            remediation_steps=remediation_steps or [],
            discovered_by=discovered_by
        )
        
        self.vulnerabilities[vuln_id] = vulnerability
        return vuln_id
    
    def update_vulnerability_status(
        self,
        vuln_id: str,
        status: VulnerabilityStatus,
        assigned_to: Optional[str] = None
    ) -> None:
        """Update vulnerability status."""
        if vuln_id in self.vulnerabilities:
            vuln = self.vulnerabilities[vuln_id]
            vuln.status = status
            
            if assigned_to:
                vuln.assigned_to = assigned_to
            
            if status == VulnerabilityStatus.FIXED:
                vuln.fixed_at = datetime.now()
            elif status == VulnerabilityStatus.VERIFIED:
                vuln.verified_at = datetime.now()
    
    def set_fix_deadline(self, vuln_id: str, days: int) -> None:
        """Set fix deadline based on severity SLA."""
        if vuln_id in self.vulnerabilities:
            self.vulnerabilities[vuln_id].target_fix_date = datetime.now() + timedelta(days=days)
    
    def get_vulnerabilities_by_severity(
        self,
        severity: VulnerabilitySeverity
    ) -> List[Vulnerability]:
        """Get vulnerabilities by severity."""
        return [
            vuln for vuln in self.vulnerabilities.values()
            if vuln.severity == severity
        ]
    
    def get_vulnerabilities_by_status(
        self,
        status: VulnerabilityStatus
    ) -> List[Vulnerability]:
        """Get vulnerabilities by status."""
        return [
            vuln for vuln in self.vulnerabilities.values()
            if vuln.status == status
        ]
    
    def get_overdue_vulnerabilities(self) -> List[Vulnerability]:
        """Get vulnerabilities past their fix deadline."""
        overdue = []
        for vuln in self.vulnerabilities.values():
            if vuln.target_fix_date and not vuln.calculate_sla_compliance():
                if vuln.status not in [VulnerabilityStatus.FIXED, 
                                      VulnerabilityStatus.VERIFIED,
                                      VulnerabilityStatus.CLOSED]:
                    overdue.append(vuln)
        return overdue


class PenetrationTestManager:
    """Penetration test lifecycle management."""
    
    def __init__(self):
        """Initialize penetration test manager."""
        self.tests: Dict[str, PenetrationTest] = {}
        self.vulnerability_scanner = VulnerabilityScanner()
    
    def create_test(
        self,
        title: str,
        description: str,
        test_type: TestType,
        scope: List[str],
        tester_team: List[str],
        out_of_scope: Optional[List[str]] = None
    ) -> str:
        """Create a new penetration test."""
        test_id = hashlib.sha256(
            f"{title}:{test_type.value}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        test = PenetrationTest(
            id=test_id,
            title=title,
            description=description,
            test_type=test_type,
            status=TestStatus.PLANNED,
            scope=scope,
            out_of_scope=out_of_scope or [],
            tester_team=tester_team
        )
        
        self.tests[test_id] = test
        return test_id
    
    def start_test(self, test_id: str) -> None:
        """Start a penetration test."""
        if test_id in self.tests:
            self.tests[test_id].status = TestStatus.IN_PROGRESS
            self.tests[test_id].start_date = datetime.now()
    
    def complete_test(self, test_id: str, executive_summary: str) -> None:
        """Complete a penetration test."""
        if test_id in self.tests:
            self.tests[test_id].status = TestStatus.COMPLETED
            self.tests[test_id].end_date = datetime.now()
            self.tests[test_id].executive_summary = executive_summary
    
    def add_finding_to_test(self, test_id: str, vuln_id: str) -> None:
        """Link a vulnerability to a penetration test."""
        if test_id in self.tests:
            if vuln_id not in self.tests[test_id].findings:
                self.tests[test_id].findings.append(vuln_id)
    
    def generate_test_report(self, test_id: str) -> Dict[str, Any]:
        """Generate a penetration test report."""
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        findings = [
            self.vulnerability_scanner.vulnerabilities.get(vuln_id)
            for vuln_id in test.findings
            if vuln_id in self.vulnerability_scanner.vulnerabilities
        ]
        
        severity_counts = {
            severity.value: sum(1 for f in findings if f and f.severity == severity)
            for severity in VulnerabilitySeverity
        }
        
        return {
            'test_info': test.to_dict(),
            'findings_summary': {
                'total_findings': len(findings),
                'by_severity': severity_counts,
                'findings': [f.to_dict() for f in findings if f]
            },
            'generated_at': datetime.now().isoformat()
        }


class RedTeamManager:
    """Red team engagement coordination."""
    
    def __init__(self):
        """Initialize red team manager."""
        self.engagements: Dict[str, RedTeamEngagement] = {}
    
    def create_engagement(
        self,
        name: str,
        objectives: List[str],
        tactics: List[str],
        techniques: List[str],
        start_date: datetime,
        end_date: datetime,
        team_members: List[str],
        target_systems: List[str],
        rules_of_engagement: List[str]
    ) -> str:
        """Create a new red team engagement."""
        engagement_id = hashlib.sha256(
            f"{name}:{start_date.isoformat()}".encode()
        ).hexdigest()[:16]
        
        engagement = RedTeamEngagement(
            id=engagement_id,
            name=name,
            objectives=objectives,
            tactics=tactics,
            techniques=techniques,
            start_date=start_date,
            end_date=end_date,
            team_members=team_members,
            target_systems=target_systems,
            rules_of_engagement=rules_of_engagement
        )
        
        self.engagements[engagement_id] = engagement
        return engagement_id
    
    def add_finding(self, engagement_id: str, finding_id: str) -> None:
        """Add a finding to a red team engagement."""
        if engagement_id in self.engagements:
            if finding_id not in self.engagements[engagement_id].findings:
                self.engagements[engagement_id].findings.append(finding_id)
    
    def update_metrics(
        self,
        engagement_id: str,
        success_rate: float,
        detection_rate: float
    ) -> None:
        """Update engagement metrics."""
        if engagement_id in self.engagements:
            self.engagements[engagement_id].success_rate = success_rate
            self.engagements[engagement_id].detection_rate = detection_rate


class PurpleTeamManager:
    """Purple team collaborative exercise management."""
    
    def __init__(self):
        """Initialize purple team manager."""
        self.exercises: Dict[str, PurpleTeamExercise] = {}
    
    def create_exercise(
        self,
        name: str,
        description: str,
        red_team: List[str],
        blue_team: List[str],
        scenarios: List[str],
        start_date: datetime,
        end_date: datetime,
        objectives: List[str]
    ) -> str:
        """Create a new purple team exercise."""
        exercise_id = hashlib.sha256(
            f"{name}:{start_date.isoformat()}".encode()
        ).hexdigest()[:16]
        
        exercise = PurpleTeamExercise(
            id=exercise_id,
            name=name,
            description=description,
            red_team=red_team,
            blue_team=blue_team,
            scenarios=scenarios,
            start_date=start_date,
            end_date=end_date,
            objectives=objectives
        )
        
        self.exercises[exercise_id] = exercise
        return exercise_id
    
    def add_lesson_learned(self, exercise_id: str, lesson: str) -> None:
        """Add a lesson learned from the exercise."""
        if exercise_id in self.exercises:
            if lesson not in self.exercises[exercise_id].lessons_learned:
                self.exercises[exercise_id].lessons_learned.append(lesson)
    
    def add_improvement(self, exercise_id: str, improvement: str) -> None:
        """Add an identified improvement."""
        if exercise_id in self.exercises:
            if improvement not in self.exercises[exercise_id].improvements_identified:
                self.exercises[exercise_id].improvements_identified.append(improvement)


class BugBountyProgram:
    """Bug bounty program integration."""
    
    def __init__(self, program_name: str):
        """Initialize bug bounty program."""
        self.program_name = program_name
        self.submissions: Dict[str, Dict[str, Any]] = {}
        self.rewards: Dict[str, float] = {
            VulnerabilitySeverity.CRITICAL.value: 10000.0,
            VulnerabilitySeverity.HIGH.value: 5000.0,
            VulnerabilitySeverity.MEDIUM.value: 2000.0,
            VulnerabilitySeverity.LOW.value: 500.0,
            VulnerabilitySeverity.INFO.value: 100.0
        }
    
    def submit_vulnerability(
        self,
        researcher: str,
        title: str,
        description: str,
        severity: VulnerabilitySeverity,
        proof_of_concept: str
    ) -> str:
        """Submit a vulnerability through bug bounty program."""
        submission_id = hashlib.sha256(
            f"{researcher}:{title}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        self.submissions[submission_id] = {
            'researcher': researcher,
            'title': title,
            'description': description,
            'severity': severity.value,
            'proof_of_concept': proof_of_concept,
            'submitted_at': datetime.now().isoformat(),
            'status': 'submitted',
            'reward_amount': 0.0
        }
        
        return submission_id
    
    def validate_submission(self, submission_id: str, is_valid: bool) -> None:
        """Validate a bug bounty submission."""
        if submission_id in self.submissions:
            if is_valid:
                self.submissions[submission_id]['status'] = 'validated'
                severity = self.submissions[submission_id]['severity']
                self.submissions[submission_id]['reward_amount'] = self.rewards.get(severity, 0.0)
            else:
                self.submissions[submission_id]['status'] = 'rejected'
    
    def get_program_stats(self) -> Dict[str, Any]:
        """Get bug bounty program statistics."""
        total_submissions = len(self.submissions)
        validated = sum(1 for s in self.submissions.values() if s['status'] == 'validated')
        total_rewards = sum(s['reward_amount'] for s in self.submissions.values())
        
        return {
            'program_name': self.program_name,
            'total_submissions': total_submissions,
            'validated_submissions': validated,
            'total_rewards_paid': total_rewards,
            'average_reward': total_rewards / validated if validated > 0 else 0.0
        }


class PenetrationTestingFramework:
    """Comprehensive penetration testing framework."""
    
    def __init__(self, organization: str):
        """Initialize penetration testing framework."""
        self.organization = organization
        self.test_manager = PenetrationTestManager()
        self.red_team_manager = RedTeamManager()
        self.purple_team_manager = PurpleTeamManager()
        self.bug_bounty_program = BugBountyProgram(f"{organization} Bug Bounty")
        self.metadata = {
            'organization': organization,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive penetration testing report."""
        return {
            'metadata': self.metadata,
            'vulnerability_summary': {
                'total_vulnerabilities': len(self.test_manager.vulnerability_scanner.vulnerabilities),
                'by_severity': {
                    severity.value: len(self.test_manager.vulnerability_scanner.get_vulnerabilities_by_severity(severity))
                    for severity in VulnerabilitySeverity
                },
                'by_status': {
                    status.value: len(self.test_manager.vulnerability_scanner.get_vulnerabilities_by_status(status))
                    for status in VulnerabilityStatus
                },
                'overdue': len(self.test_manager.vulnerability_scanner.get_overdue_vulnerabilities())
            },
            'penetration_tests': {
                'total': len(self.test_manager.tests),
                'tests': [test.to_dict() for test in self.test_manager.tests.values()]
            },
            'red_team_engagements': {
                'total': len(self.red_team_manager.engagements),
                'engagements': [eng.to_dict() for eng in self.red_team_manager.engagements.values()]
            },
            'purple_team_exercises': {
                'total': len(self.purple_team_manager.exercises),
                'exercises': [ex.to_dict() for ex in self.purple_team_manager.exercises.values()]
            },
            'bug_bounty': self.bug_bounty_program.get_program_stats(),
            'generated_at': datetime.now().isoformat()
        }
    
    def export_to_json(self, filepath: str) -> None:
        """Export penetration testing data to JSON file."""
        report = self.generate_comprehensive_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
