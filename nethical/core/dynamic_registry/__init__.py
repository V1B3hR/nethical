"""
Dynamic Attack Registry for Nethical

This module implements Phase 4 Dynamic Attack Registry capability
for automatic registration and deprecation of attack vectors.

Components:
- AutoRegistration: Automatically register new attack patterns
- AutoDeprecation: Automatically deprecate unused attack vectors
- RegistryManager: Manages the dynamic attack registry lifecycle

Phase 4 Objective: Self-updating detection with minimal human intervention

Author: Nethical Core Team
Version: 1.0.0
"""

from .auto_registration import AutoRegistration
from .auto_deprecation import AutoDeprecation
from .registry_manager import RegistryManager

__all__ = [
    "AutoRegistration",
    "AutoDeprecation",
    "RegistryManager",
]
