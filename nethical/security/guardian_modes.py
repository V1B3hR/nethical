"""Guardian modes configuration and enums.

Defines the 5 operational modes for the Adaptive Guardian:
SPRINT, CRUISE, ALERT, DEFENSE, LOCKDOWN
"""

from enum import Enum
from dataclasses import dataclass
from typing import Set


class GuardianMode(str, Enum):
    """Guardian operational modes with increasing security intensity."""
    
    SPRINT = "SPRINT"      # 🏎️ Minimal overhead, clear track
    CRUISE = "CRUISE"      # 🚗 Normal operation, balanced
    ALERT = "ALERT"        # ⚠️ Suspicious activity detected
    DEFENSE = "DEFENSE"    # 🛡️ Active threats, deep inspection
    LOCKDOWN = "LOCKDOWN"  # 🔒 Under attack, maximum security


class TripwireSensitivity(str, Enum):
    """Tripwire sensitivity levels."""
    
    CRITICAL = "CRITICAL"  # Only critical threats
    HIGH = "HIGH"          # High+ severity
    MEDIUM = "MEDIUM"      # Medium+ severity
    LOW = "LOW"            # Low+ severity
    ALL = "ALL"            # All severities


@dataclass
class ModeConfig:
    """Configuration for a guardian mode."""
    
    mode: GuardianMode
    overhead_ms: float          # Target overhead per request
    pulse_interval_s: int       # Background pulse check interval
    tripwire_sensitivity: TripwireSensitivity
    enable_correlation: bool    # Cross-module correlation
    enable_ml_detection: bool   # ML-based detection
    enable_deep_inspection: bool  # Deep content inspection
    threat_score_min: float     # Minimum threat score for this mode
    threat_score_max: float     # Maximum threat score for this mode
    emoji: str                  # Visual indicator
    description: str            # Human-readable description


# Mode configurations matching requirements
MODE_CONFIGS = {
    GuardianMode.SPRINT: ModeConfig(
        mode=GuardianMode.SPRINT,
        overhead_ms=0.02,
        pulse_interval_s=60,
        tripwire_sensitivity=TripwireSensitivity.CRITICAL,
        enable_correlation=False,
        enable_ml_detection=False,
        enable_deep_inspection=False,
        threat_score_min=0.0,
        threat_score_max=0.1,
        emoji="🏎️",
        description="Full gas - minimal checks, atomic counters only",
    ),
    GuardianMode.CRUISE: ModeConfig(
        mode=GuardianMode.CRUISE,
        overhead_ms=0.05,
        pulse_interval_s=30,
        tripwire_sensitivity=TripwireSensitivity.HIGH,
        enable_correlation=False,
        enable_ml_detection=False,
        enable_deep_inspection=False,
        threat_score_min=0.1,
        threat_score_max=0.3,
        emoji="🚗",
        description="Stable - normal operation, balanced",
    ),
    GuardianMode.ALERT: ModeConfig(
        mode=GuardianMode.ALERT,
        overhead_ms=0.2,
        pulse_interval_s=10,
        tripwire_sensitivity=TripwireSensitivity.MEDIUM,
        enable_correlation=True,
        enable_ml_detection=False,
        enable_deep_inspection=False,
        threat_score_min=0.3,
        threat_score_max=0.6,
        emoji="⚠️",
        description="Caution - cross-module correlation active",
    ),
    GuardianMode.DEFENSE: ModeConfig(
        mode=GuardianMode.DEFENSE,
        overhead_ms=1.0,
        pulse_interval_s=5,
        tripwire_sensitivity=TripwireSensitivity.LOW,
        enable_correlation=True,
        enable_ml_detection=True,
        enable_deep_inspection=True,
        threat_score_min=0.6,
        threat_score_max=0.8,
        emoji="🛡️",
        description="Defensive - deep inspection, ML detection",
    ),
    GuardianMode.LOCKDOWN: ModeConfig(
        mode=GuardianMode.LOCKDOWN,
        overhead_ms=10.0,
        pulse_interval_s=1,
        tripwire_sensitivity=TripwireSensitivity.ALL,
        enable_correlation=True,
        enable_ml_detection=True,
        enable_deep_inspection=True,
        threat_score_min=0.8,
        threat_score_max=1.0,
        emoji="🔒",
        description="Red flag - full security, confirmation required",
    ),
}


def get_mode_for_threat_score(threat_score: float) -> GuardianMode:
    """Get appropriate guardian mode for given threat score.
    
    Args:
        threat_score: Overall threat score (0.0-1.0)
        
    Returns:
        Appropriate GuardianMode for the threat level
    """
    # Clamp score to valid range
    threat_score = max(0.0, min(1.0, threat_score))
    
    # Find matching mode
    for mode, config in MODE_CONFIGS.items():
        if config.threat_score_min <= threat_score <= config.threat_score_max:
            return mode
    
    # Default to LOCKDOWN if somehow outside range
    return GuardianMode.LOCKDOWN


def get_mode_config(mode: GuardianMode) -> ModeConfig:
    """Get configuration for a specific mode.
    
    Args:
        mode: Guardian mode
        
    Returns:
        Configuration for the mode
    """
    return MODE_CONFIGS[mode]


# Severity levels that match tripwire sensitivity
SEVERITY_LEVELS = {
    TripwireSensitivity.CRITICAL: {"CRITICAL"},
    TripwireSensitivity.HIGH: {"CRITICAL", "HIGH"},
    TripwireSensitivity.MEDIUM: {"CRITICAL", "HIGH", "MEDIUM"},
    TripwireSensitivity.LOW: {"CRITICAL", "HIGH", "MEDIUM", "LOW"},
    TripwireSensitivity.ALL: {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"},
}


def severity_matches_sensitivity(
    severity: str, sensitivity: TripwireSensitivity
) -> bool:
    """Check if a severity level matches the tripwire sensitivity.
    
    Args:
        severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO)
        sensitivity: Current tripwire sensitivity
        
    Returns:
        True if severity should trigger alert at this sensitivity
    """
    return severity.upper() in SEVERITY_LEVELS.get(sensitivity, set())
