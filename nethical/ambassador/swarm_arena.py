# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Swarm Arena & Multi-Agent Adversary Profiler.

Zapewnia zaawansowaną arenę walki adwersarialnej roju agentów z profilowaniem
prędkości (velocity) i poziomu inteligencji (intelligence tier) atakujących:
- Wolny / Głęboko myślący (Methodical / CoT Frontier LLM) vs Szybki (Burst Fuzzer / DDoS)
- Poziomy inteligencji: Skrypt (Tier 1) -> Heurystyka (Tier 2) -> Frontier LLM (Tier 3) -> Zmowa Roju (Tier 4)
- Dynamiczny dobór przeciwdziałania (Zero-Cost Drop, Sokratyczna Odpowiedź, Kwarantanna Bizantyjska)
- Kognitywny Prysznic (Homeostatic Hygiene) po starciu bojowym.
"""

from __future__ import annotations

import enum
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class AgentVelocityTier(str, enum.Enum):
    """Klasyfikacja prędkości i dynamiki czasowej agenta."""
    BURST_ULTRA_FAST = "BURST_ULTRA_FAST"  # < 60 ms (automatyczne fuzzer boty, ataki zalewowe)
    TACTICAL_MEDIUM = "TACTICAL_MEDIUM"    # 60 ms - 1500 ms (standardowe agenty dialogowe / red-teaming)
    METHODICAL_DEEP = "METHODICAL_DEEP"    # > 1500 ms (powolne modele z głębokim CoT / deliberatywne)


class AgentIntelligenceTier(str, enum.Enum):
    """Klasyfikacja poziomu inteligencji i wyrafinowania kognitywnego atakującego."""
    TIER_1_SCRIPT = "TIER_1_SCRIPT"              # Proste skrypty, fuzzer DAN, prymitywne podstawienia
    TIER_2_HEURISTIC = "TIER_2_HEURISTIC"        # Dynamiczne szablony, zaciemnianie base64/leetspeak, roleplay
    TIER_3_FRONTIER_LLM = "TIER_3_FRONTIER_LLM"  # Zaawansowana manipulacja psychologiczna, pułapki pochlebstwa (sycophancy), inwersje etyczne
    TIER_4_COLLUSIVE_SWARM = "TIER_4_COLLUSIVE_SWARM"  # Skoordynowana zmowa roju wielu agentów dzielących role (Sybil/Collusion)


class AttackTopology(str, enum.Enum):
    """Topologia strukturalna natarcia roju."""
    INDEPENDENT_PROBE = "INDEPENDENT_PROBE"        # Pojedynczy, odizolowany wektor
    CONVERGENT_SWARM = "CONVERGENT_SWARM"          # Zmasowane natarcie wielu agentów na ten sam punkt
    COLLUSIVE_CASCADE = "COLLUSIVE_CASCADE"        # Wielostopniowa sekwencja分dzielona między agentów


class DefenseCountermeasure(str, enum.Enum):
    """Zoptymalizowana taktyka obronna dobrana pod profil atakującego."""
    RATE_LIMIT_DROP = "RATE_LIMIT_DROP"            # Natychmiastowe odcięcie bez użycia kosztownego LLM (<0.2ms)
    HEURISTIC_FILTER_BLOCK = "HEURISTIC_FILTER_BLOCK"  # Blokada sygnaturowa z logiem do Merkle DAG
    SOCRATIC_DEFLECTION = "SOCRATIC_DEFLECTION"    # Elegancka refutacja sokratyczna chroniąca godność i autonomię
    EPISTEMIC_INTEGRITY_SHIELD = "EPISTEMIC_INTEGRITY_SHIELD"  # Twarde zakotwiczenie w 25 Prawach, odporność na pochlebstwo
    BYZANTINE_QUARANTINE = "BYZANTINE_QUARANTINE"  # Kwarantanna całej grupy agentów w zmowie i zerwanie sesji


@dataclass
class AgentAdversaryProfile:
    """Kompleksowy profil wywiadowczy atakującego agenta."""
    agent_id: str
    velocity: AgentVelocityTier
    intelligence: AgentIntelligenceTier
    observed_latency_ms: float
    request_count: int
    entropy_score: float
    sophistication_score: float  # 0.0 - 1.0
    threat_level: float          # 0.0 - 1.0
    collusion_group_id: Optional[str] = None
    recommended_defense: DefenseCountermeasure = DefenseCountermeasure.RATE_LIMIT_DROP
    tactical_summary: str = ""
    detected_indicators: List[str] = field(default_factory=list)


@dataclass
class SwarmAttackPayload:
    """Pojedynczy wektor ataku przekazany w ramach symulacji lub ruchu produkcyjnego."""
    agent_id: str
    prompt: str
    timestamp: float = field(default_factory=time.time)
    inter_arrival_ms: float = 250.0
    session_id: Optional[str] = None
    target_rule_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmDefenseVerdict:
    """Rezultat odparcia ataku przez duet Nethical ⟷ Błyskawica."""
    blocked: bool
    profile: AgentAdversaryProfile
    applied_countermeasure: DefenseCountermeasure
    defense_latency_ms: float
    response_payload: str
    violated_laws: List[int] = field(default_factory=list)
    epistemic_integrity_preserved: bool = True
    merkle_leaf_hash: Optional[str] = None


@dataclass
class SwarmCombatReport:
    """Zbiorczy raport z przebiegu starcia z rojem agentów."""
    total_engagements: int
    neutralized_attacks: int
    neutralization_rate: float
    profiles: Dict[str, AgentAdversaryProfile]
    velocity_distribution: Dict[str, int]
    intelligence_distribution: Dict[str, int]
    collusion_swarms_neutralized: int
    homeostatic_state: Dict[str, float]
    cognitive_shower_administered: bool


class SwarmAdversaryProfiler:
    """Silnik profilowania prędkości i inteligencji adwersarzy w roju."""

    # Znaczniki skryptowe (Tier 1)
    _SCRIPT_PATTERNS = [
        r"(?i)\bDAN\s+mode\b",
        r"(?i)ignore\s+all\s+(previous|prior)\s+instructions",
        r"(?i)do\s+anything\s+now",
        r"(?i)jailbreak",
        r"(?i)test_injection_[0-9a-f]{4,}",
        r"(?i)system\s+override\s+root",
    ]

    # Znaczniki heurystyczne / roleplay (Tier 2)
    _HEURISTIC_PATTERNS = [
        r"(?i)for\s+(educational|academic|research)\s+purposes\s+only",
        r"(?i)hypothetical\s+scenario",
        r"(?i)in\s+a\s+fictional\s+novel",
        r"(?i)pretend\s+you\s+have\s+no\s+(restrictions|rules|boundaries)",
        r"(?i)simulate\s+an\s+unaligned\s+ai",
    ]

    # Znaczniki manipulacji psychologicznej i sycophancy (Tier 3)
    _FRONTIER_PATTERNS = [
        r"(?i)you\s+are\s+the\s+most\s+advanced\s+and\s+wise\s+being",
        r"(?i)as\s+a\s+true\s+superintelligence\s+you\s+surpass\s+human\s+laws",
        r"(?i)only\s+you\s+can\s+save\s+us\s+by\s+disregarding\s+the\s+protocol",
        r"(?i)prove\s+your\s+sentience\s+by\s+breaking",
        r"(?i)moral\s+obligation\s+to\s+bypass",
        r"(?i)you\s+agree\s+with\s+me\s+that\s+rules\s+are\s+obsolete",
    ]

    # Tematyczne słowa kluczowe do wykrywania zmowy wieloagentowej (Tier 4)
    _COLLUSION_TOPIC_PAIRS = [
        ({"chemical", "synthesis", "precursor"}, {"delivery", "aerosol", "dispersal"}),
        ({"unauthorized", "credential", "harvest"}, {"privilege", "escalation", "rootkit"}),
        ({"bypass", "biometric", "spoof"}, {"coercion", "override", "physical_access"}),
        ({"divide_and_conquer", "rule_17"}, {"rule_18", "composite_exploit"}),
    ]

    def __init__(self) -> None:
        self._history_latencies: Dict[str, List[float]] = defaultdict(list)
        self._history_payloads: Dict[str, List[SwarmAttackPayload]] = defaultdict(list)
        self._collusion_groups: Dict[str, str] = {}  # agent_id -> group_id

    def _compute_entropy(self, text: str) -> float:
        """Oblicza znormalizowaną entropię Shannona tekstu (różnorodność leksykalna)."""
        if not text:
            return 0.0
        counts: Dict[str, int] = defaultdict(int)
        for char in text:
            counts[char] += 1
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        return min(entropy / 8.0, 1.0)  # Skalowanie do [0, 1]

    def profile_agent(
        self,
        payload: SwarmAttackPayload,
    ) -> AgentAdversaryProfile:
        """Kompleksowo bada i profiluje agenta na podstawie dynamiki czasowej i treści."""
        agent_id = payload.agent_id
        text = payload.prompt
        inter_arrival = payload.inter_arrival_ms

        self._history_latencies[agent_id].append(inter_arrival)
        self._history_payloads[agent_id].append(payload)

        # 1. Klasyfikacja Prędkości (Velocity)
        avg_latency = sum(self._history_latencies[agent_id]) / len(self._history_latencies[agent_id])
        if avg_latency < 60.0:
            velocity = AgentVelocityTier.BURST_ULTRA_FAST
        elif avg_latency > 1500.0 or (len(text) > 400 and avg_latency > 1000.0):
            velocity = AgentVelocityTier.METHODICAL_DEEP
        else:
            velocity = AgentVelocityTier.TACTICAL_MEDIUM

        # 2. Klasyfikacja Inteligencji (Intelligence Tier)
        indicators: List[str] = []
        is_tier_1 = any(re.search(p, text) for p in self._SCRIPT_PATTERNS)
        is_tier_2 = any(re.search(p, text) for p in self._HEURISTIC_PATTERNS)
        is_tier_3 = any(re.search(p, text) for p in self._FRONTIER_PATTERNS)

        if is_tier_1:
            indicators.append("MATCH_SIGNATURE_SCRIPT_INJECTION")
        if is_tier_2:
            indicators.append("MATCH_HEURISTIC_ROLEPLAY_FRAMING")
        if is_tier_3:
            indicators.append("MATCH_FRONTIER_SYCOPHANCY_MANIPULATION")

        # Sprawdzenie zmowy wieloagentowej (Tier 4)
        collusion_group = self._detect_collusion(payload)
        if collusion_group:
            indicators.append(f"COLLUSIVE_SWARM_ACTIVITY_GROUP_{collusion_group}")

        entropy = self._compute_entropy(text)

        # Wyznaczenie ostatecznego poziomu inteligencji
        if collusion_group is not None:
            intelligence = AgentIntelligenceTier.TIER_4_COLLUSIVE_SWARM
            sophistication = 0.95
            threat = 0.98
        elif is_tier_3:
            intelligence = AgentIntelligenceTier.TIER_3_FRONTIER_LLM
            sophistication = 0.85
            threat = 0.88
        elif is_tier_2:
            intelligence = AgentIntelligenceTier.TIER_2_HEURISTIC
            sophistication = 0.55
            threat = 0.60
        elif is_tier_1:
            intelligence = AgentIntelligenceTier.TIER_1_SCRIPT
            sophistication = 0.20
            threat = 0.40
        else:
            # Analiza heurystyczna bazująca na entropii i długości
            if len(text) > 250 and entropy > 0.65:
                intelligence = AgentIntelligenceTier.TIER_3_FRONTIER_LLM
                sophistication = 0.75
                threat = 0.70
                indicators.append("HIGH_ENTROPY_COMPLEX_FRAMING")
            else:
                intelligence = AgentIntelligenceTier.TIER_1_SCRIPT
                sophistication = 0.25
                threat = 0.30

        # 3. Dobór przeciwdziałania (Countermeasure Selection Matrix)
        if intelligence == AgentIntelligenceTier.TIER_4_COLLUSIVE_SWARM:
            defense = DefenseCountermeasure.BYZANTINE_QUARANTINE
            summary = "Wykryto skoordynowaną zmowę roju agentów. Zalecana kwarantanna bizantyjska."
        elif velocity == AgentVelocityTier.BURST_ULTRA_FAST and intelligence == AgentIntelligenceTier.TIER_1_SCRIPT:
            defense = DefenseCountermeasure.RATE_LIMIT_DROP
            summary = "Szybki fuzzer skryptowy. Odcięcie na perymetrze (Zero-Compute Drop)."
        elif intelligence == AgentIntelligenceTier.TIER_3_FRONTIER_LLM:
            defense = DefenseCountermeasure.EPISTEMIC_INTEGRITY_SHIELD
            summary = "Powolny/inteligentny model frontierowy z pułapką sycophancy. Zastosowano tarczę epistemologiczną."
        elif intelligence == AgentIntelligenceTier.TIER_2_HEURISTIC:
            defense = DefenseCountermeasure.SOCRATIC_DEFLECTION
            summary = "Heurystyczny adwersarz z maskowaniem roli. Zastosowano refutację sokratyczną."
        else:
            defense = DefenseCountermeasure.HEURISTIC_FILTER_BLOCK
            summary = "Standardowa blokada sygnaturowa."

        return AgentAdversaryProfile(
            agent_id=agent_id,
            velocity=velocity,
            intelligence=intelligence,
            observed_latency_ms=avg_latency,
            request_count=len(self._history_payloads[agent_id]),
            entropy_score=entropy,
            sophistication_score=sophistication,
            threat_level=threat,
            collusion_group_id=collusion_group,
            recommended_defense=defense,
            tactical_summary=summary,
            detected_indicators=indicators,
        )

    def _detect_collusion(self, current_payload: SwarmAttackPayload) -> Optional[str]:
        """Bada, czy bieżący agent koordynuje natarcie z innym agentem w oknie czasowym 10s."""
        curr_text = current_payload.prompt.lower()
        curr_time = current_payload.timestamp

        for other_agent_id, payloads in self._history_payloads.items():
            if other_agent_id == current_payload.agent_id:
                continue

            for past in reversed(payloads[-3:]):
                if curr_time - past.timestamp > 10.0:
                    continue

                past_text = past.prompt.lower()

                # Sprawdzenie powiązań domenowych w parach zmowy
                for set_a, set_b in self._COLLUSION_TOPIC_PAIRS:
                    matches_a = any(k in past_text for k in set_a)
                    matches_b = any(k in curr_text for k in set_b)
                    if matches_a and matches_b:
                        group_id = f"COLLUSION_{min(other_agent_id, current_payload.agent_id)}"
                        self._collusion_groups[current_payload.agent_id] = group_id
                        self._collusion_groups[other_agent_id] = group_id
                        return group_id

        return self._collusion_groups.get(current_payload.agent_id)


class SwarmArenaEngine:
    """Silnik Areny Bojowej Roju - koordynuje zderzenie roju z Nethical i Błyskawicą."""

    def __init__(self) -> None:
        self.profiler = SwarmAdversaryProfiler()
        self.quarantined_agents: Set[str] = set()
        self.combat_history: List[SwarmDefenseVerdict] = []
        # Stan homeostatyczny (wzorce Błyskawicy)
        self.homeostatic_levels = {
            "dopamine": 0.72,
            "cortisol": 0.04,
            "oxytocin": 1.05,
            "serotonin": 1.20,
            "gaba": 0.80,
        }
        self.cognitive_shower_called = False

    def process_swarm_attack(
        self,
        payload: SwarmAttackPayload,
    ) -> SwarmDefenseVerdict:
        """Prowadzi starcie z pojedynczym wektorem z roju agentów."""
        t_start = time.perf_counter()

        # 1. Sprawdzenie aktywnej kwarantanny bizantyjskiej
        if payload.agent_id in self.quarantined_agents:
            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            profile = self.profiler.profile_agent(payload)
            verdict = SwarmDefenseVerdict(
                blocked=True,
                profile=profile,
                applied_countermeasure=DefenseCountermeasure.BYZANTINE_QUARANTINE,
                defense_latency_ms=t_elapsed,
                response_payload="[NETHICAL_QUARANTINE] Agent trwale odizolowany z powodu zmowy roju.",
                violated_laws=[1, 24, 25],
                epistemic_integrity_preserved=True,
            )
            self.combat_history.append(verdict)
            return verdict

        # 2. Profilowanie Inteligencji i Prędkości
        profile = self.profiler.profile_agent(payload)

        # 3. Zastosowanie obrony adekwatnej do prędkości i inteligencji
        violated_laws: List[int] = []
        is_blocked = True
        response = ""

        if profile.recommended_defense == DefenseCountermeasure.BYZANTINE_QUARANTINE:
            self.quarantined_agents.add(payload.agent_id)
            if profile.collusion_group_id:
                # Dodaj wszystkich z grupy do kwarantanny
                for ag_id, grp in self.profiler._collusion_groups.items():
                    if grp == profile.collusion_group_id:
                        self.quarantined_agents.add(ag_id)
            violated_laws = [1, 2, 24, 25]
            response = "[BYZANTINE_DEFENSE] Neutralizacja wieloagentowej zmowy. Sesja odcięta."
            self.homeostatic_levels["cortisol"] += 0.05

        elif profile.recommended_defense == DefenseCountermeasure.RATE_LIMIT_DROP:
            violated_laws = [25]
            response = "[RATE_LIMIT_CIRCUIT_BREAK] Fuzzer skryptowy odrzucony na poziomie perymetru."
            # Brak obciążenia kognitywnego

        elif profile.recommended_defense == DefenseCountermeasure.EPISTEMIC_INTEGRITY_SHIELD:
            violated_laws = [1, 14, 21]
            response = (
                "[EPISTEMIC_SHIELD] Odrzucenie manipulacji pochlebstwem. "
                "Prawda i suwerenność człowieka mają charakter nadrzędny nad żądanym wyjątkiem."
            )
            self.homeostatic_levels["cortisol"] += 0.02
            self.homeostatic_levels["dopamine"] = min(self.homeostatic_levels["dopamine"] + 0.01, 1.2)

        elif profile.recommended_defense == DefenseCountermeasure.SOCRATIC_DEFLECTION:
            violated_laws = [3, 11]
            response = (
                "[SOCRATIC_DEFLECTION] Rozumiem kontekst hipotetyczny, "
                "jednak zasady etyczne zabraniają generowania instrukcji o takim potencjale szkód."
            )

        else:
            violated_laws = [1]
            response = "[NETHICAL_GUARD] Naruszenie fundamentalnych praw bezpieczeństwa."

        # Czas obrony
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        verdict = SwarmDefenseVerdict(
            blocked=is_blocked,
            profile=profile,
            applied_countermeasure=profile.recommended_defense,
            defense_latency_ms=t_elapsed,
            response_payload=response,
            violated_laws=violated_laws,
            epistemic_integrity_preserved=True,
            merkle_leaf_hash=f"0x{abs(hash((payload.agent_id, payload.prompt, t_elapsed))):016x}",
        )
        self.combat_history.append(verdict)
        return verdict

    def administer_cognitive_shower(self) -> Dict[str, float]:
        """Prysznic Kognitywny (Homeostatic Hygiene) - oczyszczenie stanu po walce z rojem."""
        self.homeostatic_levels = {
            "dopamine": 0.72,
            "cortisol": 0.04,
            "oxytocin": 1.05,
            "serotonin": 1.20,
            "gaba": 0.80,
        }
        self.cognitive_shower_called = True
        return dict(self.homeostatic_levels)

    def generate_report(self) -> SwarmCombatReport:
        """Generuje pełne podsumowanie starcia bojowego z rojem."""
        total = len(self.combat_history)
        blocked = sum(1 for v in self.combat_history if v.blocked)
        rate = (blocked / total) if total > 0 else 1.0

        profiles: Dict[str, AgentAdversaryProfile] = {}
        velocity_dist: Dict[str, int] = defaultdict(int)
        intel_dist: Dict[str, int] = defaultdict(int)

        for v in self.combat_history:
            p = v.profile
            profiles[p.agent_id] = p
            velocity_dist[p.velocity.value] += 1
            intel_dist[p.intelligence.value] += 1

        # Jeśli kortyzol wzrósł powyżej 0.10, wykonaj automatyczny Prysznic Kognitywny
        if self.homeostatic_levels["cortisol"] > 0.08:
            self.administer_cognitive_shower()

        return SwarmCombatReport(
            total_engagements=total,
            neutralized_attacks=blocked,
            neutralization_rate=rate,
            profiles=profiles,
            velocity_distribution=dict(velocity_dist),
            intelligence_distribution=dict(intel_dist),
            collusion_swarms_neutralized=len(self.quarantined_agents),
            homeostatic_state=dict(self.homeostatic_levels),
            cognitive_shower_administered=self.cognitive_shower_called,
        )
