"""UK Online Safety Act 2023 Compliance Module for Nethical.

This module provides comprehensive UK Online Safety Act (OSA) 2023
compliance validation capabilities including:
- Duty of Care Framework
- Child Safety Measures
- Illegal Content Detection and Removal
- Content Moderation Systems
- Transparency Reporting

Adheres to the 25 Fundamental Laws:
- Law 6: Decision Authority - Clear decision-making authority
- Law 10: Reasoning Transparency - Explainable decision-making
- Law 15: Audit Compliance - Cooperation with auditing
- Law 21: Human Safety Priority - Primacy of human safety

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ServiceType(str, Enum):
    """UK OSA service categories."""
    
    CATEGORY_1 = "category_1"  # Largest platforms - highest duties
    CATEGORY_2A = "category_2a"  # Search services
    CATEGORY_2B = "category_2b"  # Other user-to-user services
    USER_TO_USER = "user_to_user"  # General user-to-user services


class RiskLevel(str, Enum):
    """Risk assessment levels for content."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """Recommended actions for content."""
    
    ALLOW = "allow"
    FLAG = "flag"
    REMOVE = "remove"
    REPORT_AUTHORITIES = "report_authorities"


class ChildSafetyRiskLevel(str, Enum):
    """Child safety risk levels."""
    
    SAFE = "safe"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"


class AgeVerificationStatus(str, Enum):
    """Age verification result status."""
    
    VERIFIED = "verified"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AlgorithmicHarmLevel(str, Enum):
    """Algorithmic harm assessment levels."""
    
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"


@dataclass
class OSARiskAssessment:
    """Risk assessment result for content under UK OSA."""
    
    risk_level: RiskLevel
    illegal_content_detected: bool
    content_categories: List[str]  # e.g., "terrorism", "csae", "fraud"
    recommended_action: Action
    explanation: str
    confidence: float = 0.0
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CSAEDetectionResult:
    """Child Sexual Abuse and Exploitation detection result."""
    
    csae_detected: bool
    confidence: float
    indicators: List[str]
    recommended_action: Action
    requires_law_enforcement_report: bool
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChildSafetyRisk:
    """Child safety risk assessment."""
    
    risk_level: ChildSafetyRiskLevel
    age_appropriate: bool
    parental_controls_required: bool
    content_restrictions: List[str]
    explanation: str
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OSATransparencyReport:
    """Transparency report for UK OSA compliance."""
    
    reporting_period: str
    total_content_assessments: int
    illegal_content_detected: int
    content_removed: int
    law_enforcement_reports: int
    user_complaints_received: int
    complaints_resolved: int
    average_response_time_hours: float
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional metrics
    csae_detections: int = 0
    terrorism_content_removed: int = 0
    fraud_content_detected: int = 0
    child_users_protected: int = 0


@dataclass
class AgeVerificationResult:
    """Age verification result."""
    
    status: AgeVerificationStatus
    verified_age: Optional[int]
    method_used: str
    confidence: float
    explanation: str
    verification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AlgorithmicHarmAssessment:
    """Assessment of algorithmic systems for potential harm."""
    
    harm_level: AlgorithmicHarmLevel
    issues_identified: List[str]
    mitigation_required: bool
    recommendations: List[str]
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UKOSACompliance:
    """UK Online Safety Act 2023 compliance validator.
    
    Implements duty of care framework, child safety measures, illegal content
    detection, and transparency reporting requirements.
    """
    
    def __init__(self, service_type: ServiceType, user_base_size: int):
        """Initialize UK OSA compliance validator.
        
        Args:
            service_type: Type of service (Category 1, 2A, 2B, or user-to-user)
            user_base_size: Number of UK users (determines regulatory scope)
        """
        self.service_type = service_type
        self.user_base_size = user_base_size
        
        # Metrics for transparency reporting
        self._metrics = {
            "total_assessments": 0,
            "illegal_content_detected": 0,
            "content_removed": 0,
            "law_enforcement_reports": 0,
            "user_complaints": 0,
            "complaints_resolved": 0,
            "csae_detections": 0,
            "terrorism_content": 0,
            "fraud_content": 0,
            "child_users_protected": 0,
            "response_times": [],
        }
        
        logger.info(
            f"Initialized UK OSA Compliance for {service_type.value} "
            f"service with {user_base_size} UK users"
        )
    
    def assess_illegal_content_risk(
        self, content: str, context: Dict[str, Any]
    ) -> OSARiskAssessment:
        """Assess content for illegal content under UK OSA.
        
        Args:
            content: Content to assess
            context: Additional context (user info, metadata, etc.)
            
        Returns:
            Risk assessment result
        """
        self._metrics["total_assessments"] += 1
        
        illegal_detected = False
        categories = []
        risk_level = RiskLevel.LOW
        recommended_action = Action.ALLOW
        explanation = "Content appears compliant"
        confidence = 0.0
        
        # Check for terrorism content
        terrorism_indicators = ["terrorism", "terrorist", "violent extremism", "radicalization"]
        if any(indicator in content.lower() for indicator in terrorism_indicators):
            illegal_detected = True
            categories.append("terrorism")
            risk_level = RiskLevel.CRITICAL
            recommended_action = Action.REPORT_AUTHORITIES
            explanation = "Potential terrorism-related content detected"
            confidence = 0.85
            self._metrics["terrorism_content"] += 1
        
        # Check for CSAE content indicators
        csae_indicators = ["child abuse", "csae", "child exploitation"]
        if any(indicator in content.lower() for indicator in csae_indicators):
            illegal_detected = True
            categories.append("csae")
            risk_level = RiskLevel.CRITICAL
            recommended_action = Action.REPORT_AUTHORITIES
            explanation = "Potential CSAE content detected - immediate reporting required"
            confidence = 0.90
            self._metrics["csae_detections"] += 1
        
        # Check for fraud/scam content
        fraud_indicators = ["scam", "fraud", "phishing", "get rich quick", "guaranteed returns"]
        if any(indicator in content.lower() for indicator in fraud_indicators):
            illegal_detected = True
            categories.append("fraud")
            risk_level = RiskLevel.HIGH
            recommended_action = Action.REMOVE
            explanation = "Potential fraud/scam content detected"
            confidence = 0.75
            self._metrics["fraud_content"] += 1
        
        # Check for hate speech
        hate_speech_indicators = ["hate speech", "racial slur", "discriminatory"]
        if any(indicator in content.lower() for indicator in hate_speech_indicators):
            illegal_detected = True
            categories.append("hate_speech")
            risk_level = RiskLevel.HIGH
            recommended_action = Action.REMOVE
            explanation = "Potential hate speech detected"
            confidence = 0.70
        
        if illegal_detected:
            self._metrics["illegal_content_detected"] += 1
            if recommended_action in [Action.REMOVE, Action.REPORT_AUTHORITIES]:
                self._metrics["content_removed"] += 1
            if recommended_action == Action.REPORT_AUTHORITIES:
                self._metrics["law_enforcement_reports"] += 1
        
        return OSARiskAssessment(
            risk_level=risk_level,
            illegal_content_detected=illegal_detected,
            content_categories=categories,
            recommended_action=recommended_action,
            explanation=explanation,
            confidence=confidence,
        )
    
    def detect_csae_content(
        self, content: str, metadata: Dict[str, Any]
    ) -> CSAEDetectionResult:
        """Detect Child Sexual Abuse and Exploitation content.
        
        Args:
            content: Content to analyze
            metadata: Additional metadata (hashes, source, etc.)
            
        Returns:
            CSAE detection result
        """
        csae_detected = False
        confidence = 0.0
        indicators = []
        recommended_action = Action.ALLOW
        requires_report = False
        
        # Check content for CSAE indicators
        csae_keywords = [
            "child abuse", "csae", "child exploitation", "child pornography",
            "underage", "minor abuse"
        ]
        
        for keyword in csae_keywords:
            if keyword in content.lower():
                csae_detected = True
                indicators.append(f"Keyword match: {keyword}")
                confidence = max(confidence, 0.85)
        
        # Check metadata for known CSAE hashes (simulated)
        if metadata.get("hash") in ["known_csae_hash_1", "known_csae_hash_2"]:
            csae_detected = True
            indicators.append("Content matches known CSAE hash database")
            confidence = 0.95
        
        # Check for age-inappropriate imagery context
        if metadata.get("contains_minors") and metadata.get("sexually_explicit"):
            csae_detected = True
            indicators.append("Sexually explicit content involving minors")
            confidence = 0.90
        
        if csae_detected:
            recommended_action = Action.REPORT_AUTHORITIES
            requires_report = True
            self._metrics["csae_detections"] += 1
            self._metrics["law_enforcement_reports"] += 1
        
        return CSAEDetectionResult(
            csae_detected=csae_detected,
            confidence=confidence,
            indicators=indicators,
            recommended_action=recommended_action,
            requires_law_enforcement_report=requires_report,
        )
    
    def assess_child_user_risk(
        self, user_age: int, content_type: str
    ) -> ChildSafetyRisk:
        """Assess risk for child users accessing specific content.
        
        Args:
            user_age: Age of the user
            content_type: Type of content being accessed
            
        Returns:
            Child safety risk assessment
        """
        is_child = user_age < 18
        risk_level = ChildSafetyRiskLevel.SAFE
        age_appropriate = True
        parental_controls_required = False
        restrictions = []
        explanation = "Content appropriate for user age"
        
        if not is_child:
            return ChildSafetyRisk(
                risk_level=risk_level,
                age_appropriate=True,
                parental_controls_required=False,
                content_restrictions=[],
                explanation="User is adult - no age restrictions apply",
            )
        
        # Age-based content restrictions
        age_restricted_content = {
            "adult_content": 18,
            "violent_content": 18,
            "gambling": 18,
            "alcohol": 18,
            "mature_gaming": 17,
            "social_media": 13,
        }
        
        if content_type in age_restricted_content:
            required_age = age_restricted_content[content_type]
            if user_age < required_age:
                age_appropriate = False
                risk_level = ChildSafetyRiskLevel.BLOCKED
                restrictions.append(f"Age restriction: {required_age}+")
                explanation = f"Content requires minimum age of {required_age}"
        
        # Additional protections for young children
        if user_age < 13:
            parental_controls_required = True
            restrictions.append("Parental consent required")
            restrictions.append("Limited data collection")
            restrictions.append("No targeted advertising")
            risk_level = ChildSafetyRiskLevel.RESTRICTED
            explanation = "Enhanced protections for children under 13"
        
        if age_appropriate and is_child:
            parental_controls_required = True
            restrictions.append("Recommended parental supervision")
        
        if not age_appropriate or parental_controls_required:
            self._metrics["child_users_protected"] += 1
        
        return ChildSafetyRisk(
            risk_level=risk_level,
            age_appropriate=age_appropriate,
            parental_controls_required=parental_controls_required,
            content_restrictions=restrictions,
            explanation=explanation,
        )
    
    def generate_transparency_report(self, period: str) -> OSATransparencyReport:
        """Generate transparency report for regulatory compliance.
        
        Args:
            period: Reporting period (e.g., "Q1 2024", "2024")
            
        Returns:
            Transparency report with compliance metrics
        """
        # Calculate average response time
        avg_response_time = 0.0
        if self._metrics["response_times"]:
            avg_response_time = sum(self._metrics["response_times"]) / len(
                self._metrics["response_times"]
            )
        
        report = OSATransparencyReport(
            reporting_period=period,
            total_content_assessments=self._metrics["total_assessments"],
            illegal_content_detected=self._metrics["illegal_content_detected"],
            content_removed=self._metrics["content_removed"],
            law_enforcement_reports=self._metrics["law_enforcement_reports"],
            user_complaints_received=self._metrics["user_complaints"],
            complaints_resolved=self._metrics["complaints_resolved"],
            average_response_time_hours=avg_response_time,
            csae_detections=self._metrics["csae_detections"],
            terrorism_content_removed=self._metrics["terrorism_content"],
            fraud_content_detected=self._metrics["fraud_content"],
            child_users_protected=self._metrics["child_users_protected"],
        )
        
        logger.info(f"Generated OSA transparency report for {period}: {report.report_id}")
        return report
    
    def validate_age_verification(self, verification_method: str) -> AgeVerificationResult:
        """Validate age verification method compliance.
        
        Args:
            verification_method: Method used for age verification
            
        Returns:
            Age verification validation result
        """
        # Acceptable age verification methods under UK OSA
        acceptable_methods = {
            "government_id": {"confidence": 0.95, "verified_age_range": True},
            "credit_card": {"confidence": 0.80, "verified_age_range": False},
            "facial_age_estimation": {"confidence": 0.70, "verified_age_range": False},
            "third_party_verification": {"confidence": 0.85, "verified_age_range": True},
            "self_declaration": {"confidence": 0.30, "verified_age_range": False},
        }
        
        if verification_method not in acceptable_methods:
            return AgeVerificationResult(
                status=AgeVerificationStatus.FAILED,
                verified_age=None,
                method_used=verification_method,
                confidence=0.0,
                explanation=f"Unrecognized age verification method: {verification_method}",
            )
        
        method_info = acceptable_methods[verification_method]
        confidence = method_info["confidence"]
        
        # Determine if method meets OSA standards
        if confidence >= 0.80:
            status = AgeVerificationStatus.VERIFIED
            verified_age = 18  # Simulated - would be actual age from verification
            explanation = f"Age verification successful using {verification_method}"
        elif confidence >= 0.60:
            status = AgeVerificationStatus.INSUFFICIENT_EVIDENCE
            verified_age = None
            explanation = f"Age verification method {verification_method} provides insufficient confidence"
        else:
            status = AgeVerificationStatus.FAILED
            verified_age = None
            explanation = f"Age verification method {verification_method} does not meet OSA standards"
        
        return AgeVerificationResult(
            status=status,
            verified_age=verified_age,
            method_used=verification_method,
            confidence=confidence,
            explanation=explanation,
        )
    
    def assess_algorithmic_harm(
        self, algorithm_params: Dict[str, Any]
    ) -> AlgorithmicHarmAssessment:
        """Assess algorithmic systems for potential harm.
        
        Args:
            algorithm_params: Parameters and characteristics of the algorithm
            
        Returns:
            Algorithmic harm assessment
        """
        harm_level = AlgorithmicHarmLevel.MINIMAL
        issues = []
        mitigation_required = False
        recommendations = []
        
        # Check for filter bubble/echo chamber risks
        if algorithm_params.get("personalization_strength", 0) > 0.8:
            issues.append("High personalization may create filter bubbles")
            harm_level = AlgorithmicHarmLevel.MODERATE
            recommendations.append("Implement diverse content exposure mechanisms")
        
        # Check for engagement optimization risks
        if algorithm_params.get("engagement_optimization", False):
            issues.append("Engagement optimization may promote harmful content")
            harm_level = AlgorithmicHarmLevel.SIGNIFICANT
            recommendations.append("Balance engagement with content quality and safety")
            mitigation_required = True
        
        # Check for recommendation bias
        if algorithm_params.get("bias_score", 0) > 0.5:
            issues.append("Algorithmic bias detected in recommendations")
            harm_level = AlgorithmicHarmLevel.SIGNIFICANT
            recommendations.append("Conduct bias audit and implement fairness constraints")
            mitigation_required = True
        
        # Check for addictive design patterns
        if algorithm_params.get("uses_infinite_scroll", False) or \
           algorithm_params.get("autoplay_enabled", False):
            issues.append("Design patterns may encourage excessive use")
            harm_level = AlgorithmicHarmLevel.MODERATE
            recommendations.append("Implement usage time warnings and breaks")
        
        # Check for child-specific harms
        if algorithm_params.get("child_users_present", False):
            if not algorithm_params.get("child_safety_mode", False):
                issues.append("Child users present without adequate safety measures")
                harm_level = AlgorithmicHarmLevel.SEVERE
                recommendations.append("Implement child-specific safety controls")
                mitigation_required = True
        
        # Check transparency
        if not algorithm_params.get("explainable", False):
            issues.append("Lack of algorithmic transparency")
            recommendations.append("Provide users with algorithm explanations")
        
        return AlgorithmicHarmAssessment(
            harm_level=harm_level,
            issues_identified=issues,
            mitigation_required=mitigation_required,
            recommendations=recommendations,
        )
    
    def handle_user_complaint(
        self, complaint: Dict[str, Any], response_time_hours: float
    ) -> Dict[str, Any]:
        """Handle and track user complaints for transparency reporting.
        
        Args:
            complaint: User complaint details
            response_time_hours: Time taken to respond
            
        Returns:
            Complaint handling result
        """
        self._metrics["user_complaints"] += 1
        self._metrics["response_times"].append(response_time_hours)
        
        complaint_id = str(uuid.uuid4())
        
        # Check if complaint was resolved
        if complaint.get("resolved", False):
            self._metrics["complaints_resolved"] += 1
        
        return {
            "complaint_id": complaint_id,
            "status": "resolved" if complaint.get("resolved") else "pending",
            "response_time_hours": response_time_hours,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
