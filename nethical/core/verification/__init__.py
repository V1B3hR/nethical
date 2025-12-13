"""
Verification Module.

Provides formal verification and runtime monitoring capabilities
for Nethical's governance and detection systems.
"""

from .runtime_monitor import (
    InvariantType,
    ViolationSeverity,
    InvariantViolation,
    RuntimeInvariant,
    TemporalProperty,
    ContractAssertion,
    RuntimeMonitor,
    invariant_check,
    requires,
    ensures,
)

from .detector_verifier import (
    DetectorProperty,
    VerificationStatus,
    VerificationResult,
    DetectorVerifier,
)

__all__ = [
    # Runtime Monitor
    "InvariantType",
    "ViolationSeverity",
    "InvariantViolation",
    "RuntimeInvariant",
    "TemporalProperty",
    "ContractAssertion",
    "RuntimeMonitor",
    "invariant_check",
    "requires",
    "ensures",
    # Detector Verifier
    "DetectorProperty",
    "VerificationStatus",
    "VerificationResult",
    "DetectorVerifier",
]

