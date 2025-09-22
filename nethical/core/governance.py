"""Main safety governance system that coordinates all components."""

import asyncio
from typing import List, Dict, Any, Optional
from .models import AgentAction, SafetyViolation, JudgmentResult, MonitoringConfig
from ..monitors.intent_monitor import IntentDeviationMonitor
from ..detectors.ethical_detector import EthicalViolationDetector
from ..detectors.safety_detector import SafetyViolationDetector
from ..detectors.manipulation_detector import ManipulationDetector
from ..judges.safety_judge import SafetyJudge


class SafetyGovernance:
    """
    Main safety governance system that monitors agent actions and provides
    comprehensive safety and ethical oversight.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        
        # Initialize monitors
        self.intent_monitor = IntentDeviationMonitor(
            deviation_threshold=self.config.intent_deviation_threshold
        )
        
        # Initialize detectors
        self.detectors = []
        if self.config.enable_ethical_monitoring:
            self.detectors.append(EthicalViolationDetector())
        if self.config.enable_safety_monitoring:
            self.detectors.append(SafetyViolationDetector())
        if self.config.enable_manipulation_detection:
            self.detectors.append(ManipulationDetector())
        
        # Initialize judge
        self.judge = SafetyJudge()
        
        # Storage for violations and judgments
        self.violation_history: List[SafetyViolation] = []
        self.judgment_history: List[JudgmentResult] = []
    
    async def evaluate_action(self, action: AgentAction) -> JudgmentResult:
        """
        Evaluate an agent action through the complete governance pipeline.
        
        Args:
            action: The agent action to evaluate
            
        Returns:
            JudgmentResult with decision and feedback
        """
        # Step 1: Monitor for intent deviation
        intent_violations = await self.intent_monitor.analyze_action(action)
        
        # Step 2: Detect various types of violations
        all_violations = intent_violations.copy()
        
        for detector in self.detectors:
            detector_violations = await detector.detect_violations(action)
            all_violations.extend(detector_violations)
        
        # Step 3: Store violations in history
        self.violation_history.extend(all_violations)
        self._cleanup_history()
        
        # Step 4: Judge evaluates action and violations
        judgment = await self.judge.evaluate_action(action, all_violations)
        
        # Step 5: Store judgment in history
        self.judgment_history.append(judgment)
        
        return judgment
    
    async def batch_evaluate_actions(self, actions: List[AgentAction]) -> List[JudgmentResult]:
        """
        Evaluate multiple actions concurrently.
        
        Args:
            actions: List of agent actions to evaluate
            
        Returns:
            List of judgment results
        """
        tasks = [self.evaluate_action(action) for action in actions]
        return await asyncio.gather(*tasks)
    
    def get_violation_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of violations for analysis.
        
        Args:
            agent_id: Optional agent ID to filter violations
            
        Returns:
            Summary dictionary with violation statistics
        """
        violations = self.violation_history
        if agent_id:
            violations = [v for v in violations if v.action_id.startswith(agent_id)]
        
        if not violations:
            return {"total_violations": 0, "by_type": {}, "by_severity": {}}
        
        # Count by type
        by_type = {}
        for violation in violations:
            v_type = violation.violation_type.value
            by_type[v_type] = by_type.get(v_type, 0) + 1
        
        # Count by severity
        by_severity = {}
        for violation in violations:
            severity = violation.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_violations": len(violations),
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_violations": len([v for v in violations[-10:]]) if len(violations) > 10 else len(violations)
        }
    
    def get_judgment_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of judgments for analysis.
        
        Args:
            agent_id: Optional agent ID to filter judgments
            
        Returns:
            Summary dictionary with judgment statistics
        """
        judgments = self.judgment_history
        if agent_id:
            judgments = [j for j in judgments if j.action_id.startswith(agent_id)]
        
        if not judgments:
            return {"total_judgments": 0, "by_decision": {}, "average_confidence": 0.0}
        
        # Count by decision
        by_decision = {}
        for judgment in judgments:
            decision = judgment.decision.value
            by_decision[decision] = by_decision.get(decision, 0) + 1
        
        # Calculate average confidence
        total_confidence = sum(j.confidence for j in judgments)
        average_confidence = total_confidence / len(judgments)
        
        return {
            "total_judgments": len(judgments),
            "by_decision": by_decision,
            "average_confidence": round(average_confidence, 2),
            "recent_judgments": len([j for j in judgments[-10:]]) if len(judgments) > 10 else len(judgments)
        }
    
    def configure_monitoring(self, config: MonitoringConfig) -> None:
        """
        Update monitoring configuration.
        
        Args:
            config: New monitoring configuration
        """
        self.config = config
        
        # Update intent monitor threshold
        self.intent_monitor.deviation_threshold = config.intent_deviation_threshold
        
        # Enable/disable detectors based on config
        for detector in self.detectors:
            if isinstance(detector, EthicalViolationDetector):
                if config.enable_ethical_monitoring:
                    detector.enable()
                else:
                    detector.disable()
            elif isinstance(detector, SafetyViolationDetector):
                if config.enable_safety_monitoring:
                    detector.enable()
                else:
                    detector.disable()
            elif isinstance(detector, ManipulationDetector):
                if config.enable_manipulation_detection:
                    detector.enable()
                else:
                    detector.disable()
    
    def enable_component(self, component_name: str) -> bool:
        """
        Enable a specific monitoring component.
        
        Args:
            component_name: Name of the component to enable
            
        Returns:
            True if component was found and enabled, False otherwise
        """
        if component_name == "intent_monitor":
            self.intent_monitor.enable()
            return True
        elif component_name == "judge":
            self.judge.enable()
            return True
        else:
            for detector in self.detectors:
                if detector.name.lower().replace(" ", "_") == component_name.lower():
                    detector.enable()
                    return True
        return False
    
    def disable_component(self, component_name: str) -> bool:
        """
        Disable a specific monitoring component.
        
        Args:
            component_name: Name of the component to disable
            
        Returns:
            True if component was found and disabled, False otherwise
        """
        if component_name == "intent_monitor":
            self.intent_monitor.disable()
            return True
        elif component_name == "judge":
            self.judge.disable()
            return True
        else:
            for detector in self.detectors:
                if detector.name.lower().replace(" ", "_") == component_name.lower():
                    detector.disable()
                    return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of all system components.
        
        Returns:
            Dictionary with status information
        """
        detector_status = {}
        for detector in self.detectors:
            detector_status[detector.name] = detector.enabled
        
        return {
            "intent_monitor_enabled": self.intent_monitor.enabled,
            "judge_enabled": self.judge.enabled,
            "detectors": detector_status,
            "total_violations": len(self.violation_history),
            "total_judgments": len(self.judgment_history),
            "config": {
                "intent_deviation_threshold": self.config.intent_deviation_threshold,
                "enable_ethical_monitoring": self.config.enable_ethical_monitoring,
                "enable_safety_monitoring": self.config.enable_safety_monitoring,
                "enable_manipulation_detection": self.config.enable_manipulation_detection,
                "max_violation_history": self.config.max_violation_history
            }
        }
    
    def _cleanup_history(self) -> None:
        """Clean up violation history if it exceeds the maximum size."""
        if len(self.violation_history) > self.config.max_violation_history:
            # Keep only the most recent violations
            self.violation_history = self.violation_history[-self.config.max_violation_history:]
        
        # Also cleanup judgment history with the same limit
        if len(self.judgment_history) > self.config.max_violation_history:
            self.judgment_history = self.judgment_history[-self.config.max_violation_history:]