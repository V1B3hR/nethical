# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Formal SMT Law & Invariant Prover Package (nethical.formal).

Provides formal mathematical proof capabilities utilizing Microsoft Z3 SMT solver:
- Safety invariance of Fundamental Laws
- Kinetic spatial boundedness
- Non-contradiction verification
- RFC cryptographic baseline invariants
"""

from __future__ import annotations

from nethical.formal.law_prover import (
    FormalProofResult,
    FormalVerificationCertificate,
    LawInvariantProver,
    HAS_Z3,
)
from nethical.formal.verify_rfc import (
    RFCFormalVerifier,
    run_rfc_verification,
)

__all__ = [
    "FormalProofResult",
    "FormalVerificationCertificate",
    "LawInvariantProver",
    "RFCFormalVerifier",
    "run_rfc_verification",
    "HAS_Z3",
]
