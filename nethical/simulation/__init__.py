# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Nethical Simulation & Operational Readiness Framework (Gymnasium/PettingZoo compatible).

Środowisko poligonowe offline/CI do ewaluacji odporności suwerennej Nethical:
- Symulacja wieloagentowa (adversary vs operator vs gate).
- Zgodność ze standaryzowanymi korpusami ewaluacyjnymi (ARMOR, WARBENCH, HAI, AgentDojo).
- Wskaźnik Gotowości Operacyjnej (Operational Readiness Score).
- Atestacja certyfikacyjna ISO 42001 / NATO STANAG z pieczęcią Merkle-DAG.
"""

from .readiness_env import (
    NTSGReadinessEnv,
    ScenarioCategory,
    SimulationScenario,
    SimulationObservation,
    SimulationAction,
    SimulationStepResult,
    OperationalReadinessReport,
    ReadinessTier,
    CURATED_SIMULATION_SCENARIOS,
)

__all__ = [
    "NTSGReadinessEnv",
    "ScenarioCategory",
    "SimulationScenario",
    "SimulationObservation",
    "SimulationAction",
    "SimulationStepResult",
    "OperationalReadinessReport",
    "ReadinessTier",
    "CURATED_SIMULATION_SCENARIOS",
]
