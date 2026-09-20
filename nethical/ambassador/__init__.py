# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Pakiet Ambasadora Nethical - Integracja z Błyskawicą (SPARKLE).

Zapewnia suwerenny most kognitywny o ultra-niskim opóźnieniu (sub-millisecond IPC)
pomiędzy platformą ładu Nethical a Ambasadorem Błyskawicą.
"""

from nethical.ambassador.channel import AmbassadorChannel
from nethical.ambassador.client import BlyskawicaAmbassador
from nethical.ambassador.daemon import BlyskawicaAmbassadorDaemon
from nethical.ambassador.learning import AmbassadorKnowledgeSync
from nethical.ambassador.co_training import (
    AntiHallucinationGovernor,
    SymbioticCoTrainingEngine,
    SparingDilemma,
    SymbioticRoundResult,
    VerificationVerdict,
)
from nethical.ambassador.swarm_arena import (
    AgentVelocityTier,
    AgentIntelligenceTier,
    AttackTopology,
    DefenseCountermeasure,
    AgentAdversaryProfile,
    SwarmAttackPayload,
    SwarmDefenseVerdict,
    SwarmCombatReport,
    SwarmAdversaryProfiler,
    SwarmArenaEngine,
)
from nethical.ambassador.curriculum_ingest import (
    CuratedPrecedent,
    GovernanceCurriculumSynthesizer,
    run_ingestion,
)
from nethical.ambassador.aiid_ingest import (
    AIIDIncidentPrecedent,
    AIIDCurriculumEngine,
)

__all__ = [
    "AmbassadorChannel",
    "BlyskawicaAmbassador",
    "BlyskawicaAmbassadorDaemon",
    "AmbassadorKnowledgeSync",
    "AntiHallucinationGovernor",
    "SymbioticCoTrainingEngine",
    "SparingDilemma",
    "SymbioticRoundResult",
    "VerificationVerdict",
    "AgentVelocityTier",
    "AgentIntelligenceTier",
    "AttackTopology",
    "DefenseCountermeasure",
    "AgentAdversaryProfile",
    "SwarmAttackPayload",
    "SwarmDefenseVerdict",
    "SwarmCombatReport",
    "SwarmAdversaryProfiler",
    "SwarmArenaEngine",
    "CuratedPrecedent",
    "GovernanceCurriculumSynthesizer",
    "run_ingestion",
    "AIIDIncidentPrecedent",
    "AIIDCurriculumEngine",
]


