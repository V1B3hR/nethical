# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Pakiet Ambasadora Nethical - Integracja z Błyskawicą (SPARKLE).

Zapewnia suwerenny most kognitywny o ultra-niskim opóźnieniu (sub-millisecond IPC)
pomiędzy platformą ładu Nethical a Ambasadorem Błyskawicą.
"""

from nethical.ambassador.channel import AmbassadorChannel
from nethical.ambassador.client import BlyskawicaAmbassador
from nethical.ambassador.learning import AmbassadorKnowledgeSync

__all__ = [
    "AmbassadorChannel",
    "BlyskawicaAmbassador",
    "AmbassadorKnowledgeSync",
]
