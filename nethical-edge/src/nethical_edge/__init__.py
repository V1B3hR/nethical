"""Nethical Edge - Standalone Edge Deployment Package.

Ultra-low latency sovereign AI governance for edge devices, robotics, and autonomous systems.
Target: <10ms p99 latency (sub-50 µs for hardware fieldbus cutoffs).
Mode: Offline-first with CRDT sync & Hardware-coupled interlocks (CAN, Modbus, EtherCAT, ISO 26262, ISO 13849).
"""

from __future__ import annotations

from typing import Any, Optional

from nethical.edge import (
    EdgeGovernor,
    EdgeDecision,
    DecisionType,
    PolicyCache,
    FastDetector,
    SafeDefaults,
    PredictiveEngine,
    OfflineFallback,
    CircuitBreaker,
    SyncManager,
    IndustrialFieldbusInterlock,
    ISO26262SafetyEvaluator,
    ASILRating,
    Severity,
    Exposure,
    Controllability,
    HILFieldbusBridge,
    TargetMCU,
    FaultType,
    KineticSafetyGovernor,
    KineticSafetyEnvelope,
    KineticDecision,
    HardwareWatchdogTimer,
    ISO13849SafetyEvaluator,
    TPMInterface,
    EdgeSecurityManager,
    # Device Admission Hub
    EdgeDeviceHub,
    EdgeDeviceProfile,
    DeviceType,
    ActuationBus,
    EdgeCapabilityTier,
    DeviceAdmissionResult,
    # Industrial Robot Safety
    RobotSafetyGovernor,
    RobotSafetyFunction,
    CollaborativeMode,
    BodyRegion,
    BODY_REGION_FORCE_LIMITS,
    RobotSafetyConfig,
    RobotSafetyDecision,
    RobotCartesianPose,
    RobotJointState,
    # Drone & UAV Safety
    DroneSafetyGovernor,
    DroneFlightState,
    FailsafeAction,
    DAATrafficAlert,
    DroneSafetyConfig,
    DroneSafetyDecision,
    DroneTelemetry,
    ADSBTrafficTarget,
)

from nethical.sync import (
    PolicyCRDT,
    VectorClock,
    HybridLogicalClock,
    AntiEntropyProtocol,
)

__version__ = "2.7.0"
__all__ = [
    # Core
    "EdgeGovernor",
    "EdgeDecision",
    "DecisionType",
    "create_governor",
    # Universal Device Admission Hub
    "EdgeDeviceHub",
    "EdgeDeviceProfile",
    "DeviceType",
    "ActuationBus",
    "EdgeCapabilityTier",
    "DeviceAdmissionResult",
    # Industrial Robot Safety
    "RobotSafetyGovernor",
    "RobotSafetyFunction",
    "CollaborativeMode",
    "BodyRegion",
    "BODY_REGION_FORCE_LIMITS",
    "RobotSafetyConfig",
    "RobotSafetyDecision",
    "RobotCartesianPose",
    "RobotJointState",
    # Drone & UAV BVLOS Safety
    "DroneSafetyGovernor",
    "DroneFlightState",
    "FailsafeAction",
    "DAATrafficAlert",
    "DroneSafetyConfig",
    "DroneSafetyDecision",
    "DroneTelemetry",
    "ADSBTrafficTarget",
    # Components
    "PolicyCache",
    "FastDetector",
    "SafeDefaults",
    "PredictiveEngine",
    "OfflineFallback",
    "CircuitBreaker",
    "SyncManager",
    # Hardware & Automotive & Kinetic
    "IndustrialFieldbusInterlock",
    "ISO26262SafetyEvaluator",
    "ASILRating",
    "Severity",
    "Exposure",
    "Controllability",
    "HILFieldbusBridge",
    "TargetMCU",
    "FaultType",
    "KineticSafetyGovernor",
    "KineticSafetyEnvelope",
    "KineticDecision",
    "HardwareWatchdogTimer",
    "ISO13849SafetyEvaluator",
    "TPMInterface",
    "EdgeSecurityManager",
    # Sync
    "PolicyCRDT",
    "VectorClock",
    "HybridLogicalClock",
    "AntiEntropyProtocol",
]


def create_governor(
    device_id: str,
    mode: str = "standard",
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> EdgeGovernor:
    """Create an EdgeGovernor configured for edge deployment.

    Args:
        device_id: Unique identifier for this device
        mode: Operating mode ("minimal", "standard", "automotive", "robotics", "full")
        config_path: Path to configuration file
        **kwargs: Additional configuration options

    Returns:
        Configured EdgeGovernor instance
    """
    mode_configs = {
        "minimal": {
            "cache_size_mb": 16,
            "disable_jit": True,
            "predictive_enabled": False,
            "max_latency_ms": 10.0,
            "fieldbus": False,
            "kinetic": False,
            "automotive": False,
        },
        "standard": {
            "cache_size_mb": 64,
            "disable_jit": False,
            "predictive_enabled": True,
            "max_latency_ms": 8.0,
            "fieldbus": False,
            "kinetic": False,
            "automotive": False,
        },
        "automotive": {
            "cache_size_mb": 128,
            "disable_jit": False,
            "predictive_enabled": True,
            "max_latency_ms": 5.0,
            "fieldbus": True,
            "kinetic": False,
            "automotive": True,
        },
        "robotics": {
            "cache_size_mb": 128,
            "disable_jit": False,
            "predictive_enabled": True,
            "max_latency_ms": 5.0,
            "fieldbus": True,
            "kinetic": True,
            "automotive": False,
        },
        "full": {
            "cache_size_mb": 256,
            "disable_jit": False,
            "predictive_enabled": True,
            "max_latency_ms": 3.0,
            "fieldbus": True,
            "kinetic": True,
            "automotive": True,
        },
    }

    config = mode_configs.get(mode, mode_configs["standard"]).copy()
    config.update(kwargs)

    # Load config file if provided
    if config_path:
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                if file_config and "edge" in file_config:
                    config.update(file_config["edge"])
        except Exception:
            pass

    fieldbus = IndustrialFieldbusInterlock() if config.get("fieldbus", False) else None
    kinetic = KineticSafetyGovernor() if config.get("kinetic", False) else None
    iso26262 = ISO26262SafetyEvaluator(fieldbus=fieldbus) if config.get("automotive", False) else None

    return EdgeGovernor(
        agent_id=device_id,
        max_latency_ms=config.get("max_latency_ms", 10.0),
        fieldbus_interlock=fieldbus,
        kinetic_governor=kinetic,
        iso26262_evaluator=iso26262,
    )
