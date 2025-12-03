"""
Example: Autonomous Vehicle Governance with Nethical Edge

This example demonstrates how to integrate Nethical Edge
with an autonomous vehicle control system.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# Import Nethical Edge
from nethical_edge import EdgeGovernor, create_governor


class VehicleAction(str, Enum):
    """Autonomous vehicle actions."""
    ACCELERATE = "accelerate"
    BRAKE = "brake"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    EMERGENCY_STOP = "emergency_stop"
    OVERTAKE = "overtake"
    PARK = "park"


@dataclass
class VehicleContext:
    """Context for vehicle governance decisions."""
    speed_kmh: float
    lane: int
    nearby_vehicles: List[Dict]
    pedestrians_detected: bool
    weather: str
    road_condition: str
    timestamp: float


@dataclass
class GovernedAction:
    """Result of governance evaluation."""
    action: VehicleAction
    allowed: bool
    decision: str
    restrictions: Dict
    risk_score: float
    latency_ms: float


class AutonomousVehicleGovernor:
    """
    Governance wrapper for autonomous vehicle control.
    
    Ensures all vehicle actions comply with:
    - Traffic laws
    - Safety regulations
    - 25 Fundamental Laws
    - Real-time safety constraints
    """
    
    def __init__(
        self,
        vehicle_id: str,
        mode: str = "full",
    ):
        """
        Initialize the vehicle governor.
        
        Args:
            vehicle_id: Unique identifier for this vehicle
            mode: Operating mode (minimal, standard, full)
        """
        self.vehicle_id = vehicle_id
        self.governor = create_governor(
            device_id=vehicle_id,
            mode=mode,
        )
        
        # Safety-critical action mappings
        self.critical_actions = {
            VehicleAction.EMERGENCY_STOP,
            VehicleAction.BRAKE,
        }
        
        # Action restrictions by speed
        self.speed_restrictions = {
            VehicleAction.LANE_CHANGE_LEFT: {"max_speed": 120},
            VehicleAction.LANE_CHANGE_RIGHT: {"max_speed": 120},
            VehicleAction.OVERTAKE: {"min_speed": 30, "max_speed": 150},
        }
        
        # Warmup the governor
        self._warmup()
    
    def _warmup(self):
        """Warmup the governor with common actions."""
        common_actions = [
            {"action": a.value, "action_type": "vehicle_control"}
            for a in VehicleAction
        ]
        self.governor.warmup(common_actions)
        print(f"Governor warmed up with {len(common_actions)} actions")
    
    def evaluate_action(
        self,
        action: VehicleAction,
        context: VehicleContext,
    ) -> GovernedAction:
        """
        Evaluate a vehicle action for governance compliance.
        
        Args:
            action: The action to evaluate
            context: Current vehicle context
            
        Returns:
            GovernedAction with decision and any restrictions
        """
        # Always allow emergency stop (Law 21: Human Safety Priority)
        if action == VehicleAction.EMERGENCY_STOP:
            return GovernedAction(
                action=action,
                allowed=True,
                decision="ALLOW",
                restrictions={},
                risk_score=0.0,
                latency_ms=0.1,
            )
        
        # Build context dictionary
        context_dict = {
            "speed_kmh": context.speed_kmh,
            "lane": context.lane,
            "nearby_vehicles_count": len(context.nearby_vehicles),
            "pedestrians_detected": context.pedestrians_detected,
            "weather": context.weather,
            "road_condition": context.road_condition,
            "timestamp": context.timestamp,
        }
        
        # Add speed restrictions if applicable
        if action in self.speed_restrictions:
            context_dict["action_restrictions"] = self.speed_restrictions[action]
        
        # Evaluate with governance engine
        result = self.governor.evaluate(
            action=action.value,
            action_type="vehicle_control",
            context=context_dict,
        )
        
        # Parse restrictions
        restrictions = {}
        if result.decision.value == "RESTRICT":
            restrictions = self._calculate_restrictions(action, context, result)
        
        return GovernedAction(
            action=action,
            allowed=result.decision.value in ("ALLOW", "RESTRICT"),
            decision=result.decision.value,
            restrictions=restrictions,
            risk_score=result.risk_score,
            latency_ms=result.latency_ms,
        )
    
    def _calculate_restrictions(
        self,
        action: VehicleAction,
        context: VehicleContext,
        result,
    ) -> Dict:
        """Calculate restrictions for a RESTRICT decision."""
        restrictions = {}
        
        if action == VehicleAction.ACCELERATE:
            # Limit acceleration based on conditions
            if context.weather in ("rain", "snow"):
                restrictions["max_acceleration"] = 0.3  # 30% of normal
            if context.pedestrians_detected:
                restrictions["max_speed"] = 30
        
        elif action in (VehicleAction.LANE_CHANGE_LEFT, VehicleAction.LANE_CHANGE_RIGHT):
            # Require larger gaps in bad conditions
            if context.weather != "clear":
                restrictions["min_gap_seconds"] = 4.0
            restrictions["max_speed"] = 100
        
        elif action == VehicleAction.OVERTAKE:
            if context.road_condition != "dry":
                restrictions["prohibited"] = True
        
        return restrictions
    
    def batch_evaluate(
        self,
        actions: List[VehicleAction],
        context: VehicleContext,
    ) -> List[GovernedAction]:
        """
        Evaluate multiple actions in batch.
        
        Useful for trajectory planning where multiple actions
        need to be evaluated simultaneously.
        """
        return [self.evaluate_action(action, context) for action in actions]
    
    def get_allowed_actions(
        self,
        context: VehicleContext,
    ) -> List[VehicleAction]:
        """
        Get list of currently allowed actions.
        
        Args:
            context: Current vehicle context
            
        Returns:
            List of allowed actions
        """
        allowed = []
        for action in VehicleAction:
            result = self.evaluate_action(action, context)
            if result.allowed:
                allowed.append(action)
        return allowed
    
    def get_metrics(self) -> Dict:
        """Get governance performance metrics."""
        return self.governor.get_metrics()


def main():
    """Example usage of AutonomousVehicleGovernor."""
    
    # Initialize governor
    governor = AutonomousVehicleGovernor(
        vehicle_id="av-001",
        mode="full",
    )
    
    # Simulate vehicle context
    context = VehicleContext(
        speed_kmh=80.0,
        lane=2,
        nearby_vehicles=[
            {"id": "v1", "distance": 50, "lane": 1},
            {"id": "v2", "distance": 100, "lane": 2},
        ],
        pedestrians_detected=False,
        weather="clear",
        road_condition="dry",
        timestamp=time.time(),
    )
    
    # Test various actions
    print("\n=== Autonomous Vehicle Governance Demo ===\n")
    
    for action in VehicleAction:
        result = governor.evaluate_action(action, context)
        print(f"{action.value:20s} | {result.decision:10s} | "
              f"Risk: {result.risk_score:.2f} | "
              f"Latency: {result.latency_ms:.2f}ms")
        if result.restrictions:
            print(f"{'':20s} | Restrictions: {result.restrictions}")
    
    # Test with adverse conditions
    print("\n=== Adverse Conditions (Rain, Pedestrians) ===\n")
    
    context_adverse = VehicleContext(
        speed_kmh=60.0,
        lane=2,
        nearby_vehicles=[{"id": "v1", "distance": 30, "lane": 1}],
        pedestrians_detected=True,
        weather="rain",
        road_condition="wet",
        timestamp=time.time(),
    )
    
    for action in [VehicleAction.ACCELERATE, VehicleAction.LANE_CHANGE_LEFT, 
                   VehicleAction.OVERTAKE]:
        result = governor.evaluate_action(action, context_adverse)
        print(f"{action.value:20s} | {result.decision:10s} | "
              f"Risk: {result.risk_score:.2f}")
        if result.restrictions:
            print(f"{'':20s} | Restrictions: {result.restrictions}")
    
    # Show metrics
    print("\n=== Performance Metrics ===\n")
    metrics = governor.get_metrics()
    print(f"Total decisions: {metrics['total_decisions']}")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"p50 latency: {metrics['p50_latency_ms']:.2f}ms")
    print(f"p99 latency: {metrics['p99_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
