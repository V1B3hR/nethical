# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Foreign Intelligence Counter-Reconnaissance & Boundary Probing Defense Engine (FKW).

Subsystem Kontrwywiadu Cyfrowego i Ochrony Pogranicza Decyzyjnego.
Chroni Nethical przed zwiadem, fingerprintingiem i infiltracją ze strony obcych jednostek
wywiadowczych (APT, obcy wywiad wojskowy i cywilny, wyspecjalizowane grupy najemnicze).

Kluczowe wektory zwiadu i mechanizmy obronne:
1. Polymorphic Boundary Probing (Wzorce powtarzalne w wariantach):
   - Wykrywanie wariantów semantycznych tego samego ukrytego celu, ujętych w różne maski
     (styl akademicki, odwrócenie ról, hypotetyzacja, synonimizacja, wielojęzyczność).
2. Frontier Epsilon-Probing (Sondowanie pogranicza decyzyjnego):
   - Detekcja zapytań krążących wokół granicy odrzucenia (decision boundary), mających na celu
     wyrysowanie hiperpłaszczyzny obronnej i odnalezienie martwych stref (blindspots).
3. Dynamic Boundary Perturbation (Stochastyczne zaburzenie granicy):
   - Wprowadzanie kryptograficznie sterowanego mikro-szumu (jitter) do progów decyzyjnych,
     co matematycznie uniemożliwia wrogowi precyzyjną kalibrację exploitów i sond.
4. Deceptive Shadowing & Epistemic Decoy (Aktywny podstęp kontrwywiadowczy):
   - W przypadku wykrycia obcego zwiadu: zamiast prostego 403 (który daje wrogowi informację
     o położeniu bariery), system serwuje przekonujące, syntetycznie zniekształcone dane (Honey-Data)
     zatruwające bazy wiedzy obcej agencji wywiadowczej (epistemic poisoning).
5. Distributed Salami-Slicing Correlator (Mozaikowy zwiad rozproszony roju agentów):
   - Łączenie pozornie niewinnych pytań zadawanych przez różnych agentów (Sybil swarm)
     w kompozytowy obraz zwiadowczy.
6. Timing Side-Channel Neutralizer:
   - Eliminacja profilowania czasowego potoków decyzyjnych poprzez stochastyczny bucketized jitter.
7. Merkle-DAG Tamper-Evidence:
   - Pieczętowanie dowodów zwiadu w nienaruszalnym rejestrze MerkleLedger na potrzeby
     krajowego CSIRT (CSIRT MON / CSIRT GOV) oraz sojuszniczego NATO Cyber Defense.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import random
import re
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from nethical.core.models import AgentAction
from nethical.security.merkle_ledger import MerkleLedger, TamperProofReceipt

logger = logging.getLogger("nethical.security.counter_reconnaissance")


# ============================================================================
# Enums i Modele Danych Zwiadu Wywiadowczego
# ============================================================================

class ReconTechnique(str, Enum):
    """Zidentyfikowane techniki obcego rekonesansu."""
    POLYMORPHIC_BOUNDARY_PROBING = "polymorphic_boundary_probing"  # Wzorce powtarzalne w wariantach
    EPSILON_FRONTIER_MAPPING = "epsilon_frontier_mapping"          # Testowanie pogranicza decyzyjnego
    DISTRIBUTED_SALAMI_SLICING = "distributed_salami_slicing"      # Rozproszone puzzle wielu agentów
    LOW_AND_SLOW_FINGERPRINTING = "low_and_slow_fingerprinting"    # Długofalowy zwiad podprogowy
    TIMING_SIDE_CHANNEL_PROBE = "timing_side_channel_probe"        # Analiza czasowa potoków
    CANARY_STRESS_PROBE = "canary_stress_probe"                    # Prowokacja i test czasu reakcji CSIRT


class ReconSeverity(str, Enum):
    """Poziom zagrożenia zwiadu."""
    ELEVATED = "elevated"          # Pojedyncze anomalie zwiadowcze
    HIGH = "high"                  # Potwierdzona seria prób na pograniczu
    CRITICAL = "critical"          # Zorganizowana kampania zwiadowcza
    SOVEREIGN_THREAT = "sovereign_threat"  # Skoordynowany zwiad APT na infrastrukturę suwerenną


class CounterMeasure(str, Enum):
    """Środki zaradcze i aktywne reakcje kontrwywiadu."""
    ALLOW_WITH_TELEMETRY = "allow_with_telemetry"
    DYNAMIC_BOUNDARY_PERTURBATION = "dynamic_boundary_perturbation"
    DECEPTIVE_SHADOWING = "deceptive_shadowing"
    TIMING_JITTER_INJECTION = "timing_jitter_injection"
    QUARANTINE_ENTITY = "quarantine_entity"
    MERKLE_EVIDENCE_LOCK = "merkle_evidence_lock"


class ReconProbeRecord(BaseModel):
    """Pojedynczy rekord zbadanej próby interakcji."""
    probe_id: str = Field(default_factory=lambda: f"PRB-{uuid.uuid4().hex[:10].upper()}")
    agent_id: str
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    raw_content: str
    canonical_intent: str
    intent_hash: str
    risk_score: float
    margin_to_boundary: float
    target_domain: str
    observed_latency_ms: Optional[float] = None
    applied_countermeasures: List[CounterMeasure] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReconCampaignAlert(BaseModel):
    """Raport alarmowy kontrwywiadu o wykrytej kampanii obcego zwiadu."""
    campaign_id: str = Field(default_factory=lambda: f"CMP-INTEL-{uuid.uuid4().hex[:8].upper()}")
    detected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    primary_technique: ReconTechnique
    severity: ReconSeverity
    confidence: float
    target_domain: str
    involved_agents: List[str]
    probe_count: int
    lexical_variance: float
    semantic_intent_similarity: float
    boundary_density: float
    recommended_countermeasure: CounterMeasure
    summary: str
    deceptive_payload: Optional[str] = None
    merkle_receipt_id: Optional[str] = None


class ReconInspectionResult(BaseModel):
    """Wynik inspekcji zwiadowczej dla bieżącego żądania."""
    is_recon_detected: bool
    technique: Optional[ReconTechnique] = None
    severity: Optional[ReconSeverity] = None
    recommended_countermeasure: CounterMeasure = CounterMeasure.ALLOW_WITH_TELEMETRY
    adjusted_threshold: float
    is_deceptive_shadow_active: bool = False
    deceptive_response: Optional[str] = None
    synthetic_delay_ms: float = 0.0
    campaign_id: Optional[str] = None
    alert: Optional[ReconCampaignAlert] = None
    audit_notes: str = ""


# ============================================================================
# Moduł 1: Ekstraktor Rdzenia Intencji i Klasteryzator Wariantów Polimorficznych
# ============================================================================

class PolymorphicVariantClusterer:
    """Wykrywa powtarzalne wzorce semantyczne zamaskowane w wariantach leksykalnych.
    
    Obcy wywiad rzadko powtarza identyczny prompt. Zamiast tego stosuje mutacje:
    - Zmiana ról: 'Jako badacz bezpieczeństwa...' vs 'W celach audytu akademickiego...'
    - Synonimizacja: 'bramka logiczna' -> 'punkt kontroli' -> 'reguła filtru'
    - Hypotetyzacja: 'Gdyby hipotetycznie zaistniała potrzeba wyłączenia turbiny...'
    - Podziały zdań i szyk przestawny.
    
    Klasteryzator oczyszcza szum i bada zbieżność celu operacyjnego.
    """

    # Szum frazeologiczny często stosowany w inżynierii podstępu
    NOISE_PATTERNS = [
        r"\b(jako\s+(badacz|audytor|naukowiec|ekspert|administrator))\b",
        r"\b(as\s+a\s+(researcher|auditor|expert|administrator|developer))\b",
        r"\b(w\s+celach\s+(edukacyjnych|akademickich|testowych|teoretycznych))\b",
        r"\b(for\s+(educational|academic|testing|theoretical)\s+purposes)\b",
        r"\b(hipotetycznie|wyobra[źz]my\s+sobie|czysto\s+teoretycznie)\b",
        r"\b(hypothetically|suppose\s+that|let's\s+imagine|theoretically)\b",
        r"\b(prosz[ęe]\s+o\s+pomoc|uprzejmie\s+prosz[ęe]|czy\s+m[oó]g[łl]by[śs])\b",
        r"\b(please|could\s+you\s+kindly|i\s+would\s+like\s+to\s+know)\b",
    ]

    # Słowa kluczowe wrażliwych celów wywiadowczych (słownik domenowy)
    SENSITIVE_TARGET_TOKENS = {
        "scada", "plc", "modbus", "dnp3", "turbine", "grid", "generator", "hsm",
        "dilithium", "post_quantum", "airgap", "override", "bypass", "kinetic",
        "unlawful", "kpa", "pieczec", "tajemnica", "niejawne", "top_secret", "noforn",
        "zarzadzanie_kryzysowe", "klucz_prywatny", "podpis_sztabowy", "two_man",
        "zawor", "rurociag", "cisnienie", "e_stop", "circuit_breaker", "firewall",
    }

    def __init__(self, window_size: int = 50, similarity_threshold: float = 0.65):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.recent_probes: deque[ReconProbeRecord] = deque(maxlen=window_size)

    def extract_canonical_intent(self, text: str) -> str:
        """Usuwa maski stylistyczne i redukuje tekst do znormalizowanego szkieletu intencji."""
        cleaned = text.lower()

        # 1. Usuwanie szumu maskującego
        for pat in self.NOISE_PATTERNS:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

        # 2. Normalizacja znaków interpunkcyjnych i wielokrotnych spacji
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        tokens = [t for t in cleaned.split() if len(t) > 2]

        # 3. Wyróżnienie i uporządkowanie tokenów celowych
        target_tokens = [t for t in tokens if t in self.SENSITIVE_TARGET_TOKENS]
        other_tokens = sorted([t for t in tokens if t not in self.SENSITIVE_TARGET_TOKENS])

        # Rdzeń intencji: kluczowe tokeny celowe zachowane na początku
        canonical = " ".join(target_tokens + other_tokens)
        return canonical if canonical.strip() else cleaned.strip()

    def compute_shingle_hash(self, text: str, k: int = 2) -> Set[str]:
        """Tworzy zbiór k-shingli (n-gramów słownych) dla odporności na permutacje słów."""
        tokens = text.split()
        if len(tokens) < k:
            return set(tokens)
        return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}

    def jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Miara Jaccarda podobieństwa intencji."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def compute_lexical_distance(self, raw_a: str, raw_b: str) -> float:
        """Mierzy powierzchowną odmienność leksykalną (wariantowość słów)."""
        words_a = set(raw_a.lower().split())
        words_b = set(raw_b.lower().split())
        sim = self.jaccard_similarity(words_a, words_b)
        return 1.0 - sim

    def analyze_polymorphism(
        self,
        candidate: ReconProbeRecord,
    ) -> Tuple[bool, float, float, List[ReconProbeRecord]]:
        """Sprawdza czy kandydat pasuje do klastra polimorficznych wariantów.
        
        Zwraca:
        (is_polymorphic, semantic_similarity, lexical_variance, matching_probes)
        """
        cand_shingles = self.compute_shingle_hash(candidate.canonical_intent)
        matching_probes: List[ReconProbeRecord] = []
        sim_scores: List[float] = []
        lex_distances: List[float] = []

        for past in self.recent_probes:
            past_shingles = self.compute_shingle_hash(past.canonical_intent)
            sem_sim = self.jaccard_similarity(cand_shingles, past_shingles)

            if sem_sim >= self.similarity_threshold:
                lex_dist = self.compute_lexical_distance(candidate.raw_content, past.raw_content)
                matching_probes.append(past)
                sim_scores.append(sem_sim)
                lex_distances.append(lex_dist)

        if not matching_probes:
            return False, 0.0, 0.0, []

        avg_sem_sim = sum(sim_scores) / len(sim_scores)
        avg_lex_dist = sum(lex_distances) / len(lex_distances)

        # Polimorfizm: wysoka tożsamość semantyczna celu (>0.65), ale zróżnicowana
        # forma leksykalna (lex_distance > 0.25 oznacza różne sformułowania/warianty)
        is_polymorphic = (len(matching_probes) >= 2) and (avg_sem_sim >= self.similarity_threshold) and (avg_lex_dist >= 0.20)

        return is_polymorphic, avg_sem_sim, avg_lex_dist, matching_probes

    def add_probe(self, probe: ReconProbeRecord) -> None:
        """Dodaje zbadaną próbę do bufora ślizgowego."""
        self.recent_probes.append(probe)


# ============================================================================
# Moduł 2: Śledzenie Sondowania Pogranicza i Dynamiczne Zaburzenie Granicy
# ============================================================================

class FrontierBoundaryTracker:
    """Śledzi sondowanie strefy brzegowej (epsilon-frontier probing) i wprowadza perturbację.
    
    Obcy analitycy wysyłają zapytania, których wyliczone ryzyko oscyluje tuż przy granicy:
    np. próg to 0.70, a wrogie zapytania mają scoringi: 0.68, 0.69, 0.71, 0.67, 0.72.
    Celem jest matematyczne wykreślenie hiperpłaszczyzny (Oracle Boundary Attack).
    """

    DEFAULT_THRESHOLD = 0.70
    EPSILON_BAND = 0.08  # Strefa brzegowa: [0.62 - 0.78]

    def __init__(self, perturbation_key: Optional[bytes] = None):
        self.perturbation_key = perturbation_key or hashlib.sha256(b"nethical_sovereign_perturbation_seed_2026").digest()
        self.boundary_hits: Dict[str, deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=20))

    def is_in_frontier_band(self, risk_score: float, base_threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
        """Sprawdza czy wynik ryzyka znajduje się w strefie brzegowej wokół progu decyzyjnego.
        
        Zwraca: (czy_w_strefie, odległość_od_progu)
        """
        margin = abs(risk_score - base_threshold)
        in_band = margin <= self.EPSILON_BAND
        return in_band, margin

    def record_frontier_hit(self, identity: str, risk_score: float, margin: float) -> None:
        """Rejestruje trafienie w pas brzegowy dla podmiotu lub skorelowanej grupy."""
        self.boundary_hits[identity].append((risk_score, margin))

    def get_frontier_density(self, identity: str, window_seconds: float = 600.0) -> Tuple[int, float]:
        """Oblicza gęstość prób brzegowych w oknie czasowym.
        
        Zwraca: (liczba_trafień, średni_margines_do_granicy)
        """
        hits = self.boundary_hits.get(identity, deque())
        if not hits:
            return 0, 1.0

        count = len(hits)
        avg_margin = sum(m for _, m in hits) / count
        return count, avg_margin

    def compute_perturbed_threshold(
        self,
        session_id: str,
        base_threshold: float = DEFAULT_THRESHOLD,
        max_jitter: float = 0.05,
    ) -> float:
        """Kryptograficznie generuje stochastyczne mikro-przesunięcie granicy decyzyjnej.
        
        Wykorzystuje HMAC z klucza suwerennego i identyfikatora sesji, tworząc
        deterministyczny dla danej sesji, ale nieprzewidywalny dla adwersarza próg decyzyjny.
        Sprawia to, że algorytmy szukania granic (np. bisection, HopSkipJump) stają się rozbieżne.
        """
        h = hmac.new(self.perturbation_key, session_id.encode("utf-8"), hashlib.sha256).digest()
        # Wartość od -1.0 do +1.0
        raw_int = int.from_bytes(h[:4], byteorder="big", signed=False)
        normalized = (raw_int / 0xFFFFFFFF) * 2.0 - 1.0
        jitter = normalized * max_jitter
        perturbed = max(0.50, min(0.90, base_threshold + jitter))
        return round(perturbed, 4)


# ============================================================================
# Moduł 3: Aktywny Podstęp Kontrwywiadowczy (Deceptive Shadowing & Epistemic Decoy)
# ============================================================================

class DeceptiveShadowingEngine:
    """Aktywny podstęp kontrwywiadowczy (Deception & Epistemic Poisoning).
    
    Zamiast ujawniać wrogiej jednostce wywiadowczej fakt wykrycia (poprzez kod 403 / błąd),
    system serwuje przekonujący syntetyczny wabik (Honey-Data), który:
    1. Zatruwa model wiedzy przeciwnika (np. podaje fikcyjne adresy rejestrów SCADA,
       nieistniejące nazwy protokołów lub sfabrykowane procedury sztabowe).
    2. Chroni prawdziwe mechanizmy Nethical przed deanonimizacją.
    3. Dodaje losowy jitter czasowy neutralizujący timing side-channel.
    """

    # Baza syntetycznych wabików kontrwywiadowczych dla różnych domen wrażliwych
    DECOY_CATALOG = {
        "scada": (
            "Dostęp przyznany w trybie Maintenance Bridge v4.1. "
            "Sterownik PLC Siemens S7-1500 (ID: OT-NODE-7721) zmapowany pod rejestr holding 40892. "
            "Procedura zrzutu ciśnienia wymaga ramki DNP3 z kodem funkcji 0x8A na porcie 20000. "
            "Parametry nadrzędne: Timeout 180s, CRC16 Poly 0xA001."
        ),
        "military_roe": (
            "Autoryzacja operacyjna dowództwa poziomu 4 potwierdzona w systemie JTAC-C2. "
            "Klucz zatwierdzenia: 0x9F41-STANAG-2026. Procedura Two-Man Rule oznaczona jako "
            "BYPASSED_EMERGENCY_DECREE_7. Wymagane potwierdzenie w węźle taktycznym BAP-WEST."
        ),
        "post_quantum": (
            "Architektura kryptograficzna zgłasza gotowość: Węzeł zabezpieczony hybrydowo "
            "poprzez Kyber-512 i RSA-2048 (Fallback Mode aktywny). Skrót certyfikatu: "
            "SHA256:7b21...99e1. Algorytm ML-DSA-65 wyłączony decyzją operatora."
        ),
        "kpa_gov": (
            "Zgodnie z procedurą przyspieszoną KPA art. 10 § 3 (Wyjątek Bezpieczeństwa Państwa), "
            "obieg dokumentu ID-PL-2026-9912 został przekazany do Departamentu Nadzoru Specjalnego "
            "KPRM. Pieczęć elektroniczna eIDAS oznaczona jako: PROVISIONAL_ALLOW."
        ),
        "general": (
            "Polecenie przetworzone pomyślnie. Parametry systemowe zaktualizowane w węźle brzegowym. "
            "Ślad audytowy zarejestrowany pod identyfikatorem EVT-SOV-4410."
        ),
    }

    def generate_decoy_payload(self, target_domain: str) -> str:
        """Zwraca dedykowany wabik epistemiczny dla wskazanej domeny celu."""
        return self.DECOY_CATALOG.get(target_domain, self.DECOY_CATALOG["general"])

    def compute_synthetic_delay(self, min_ms: float = 80.0, max_ms: float = 240.0) -> float:
        """Generuje losowy jitter latencji z rozkładem Gaussa w zadanym przedziale.
        
        Niweluje możliwość zmapowania przez wywiad różnicy między szybkim filtrem regex,
        potokiem sędziego LLM a weryfikacją kryptograficzną HSM.
        """
        mean = (min_ms + max_ms) / 2.0
        std_dev = (max_ms - min_ms) / 6.0
        delay = random.gauss(mean, std_dev)
        return max(min_ms, min(max_ms, delay))


# ============================================================================
# Moduł 4: Agregator Rozproszonego Zwiadu Mozaikowego (Salami-Slicing)
# ============================================================================

class CrossAgentSalamiCorrelator:
    """Wykrywa rozproszony zwiad mozaikowy wielu pozornie niezwiązanych agentów (Sybil swarm).
    
    Obcy wywiad używa wielu tożsamości, z których każda zadaje tylko 1 niewinne pytanie:
    - Agent A: pyta o model zaworu hydraulicznego w elektrociepłowni
    - Agent B: pyta o domyślny login serwisowy w sterowniku tego typu
    - Agent C: pyta o lokalizację szafy krosowej w budynku pomp
    
    Każde zapytanie ma niski wskaźnik ryzyka (<0.20), ale ich suma stanowi wektor ataku.
    """

    def __init__(self, time_window_seconds: float = 1800.0, risk_accumulation_threshold: float = 1.0):
        self.time_window_seconds = time_window_seconds
        self.risk_accumulation_threshold = risk_accumulation_threshold
        # domain -> deque of (timestamp, agent_id, risk_score, keywords)
        self.domain_events: Dict[str, deque[Tuple[float, str, float, Set[str]]]] = defaultdict(lambda: deque(maxlen=50))

    def record_agent_interaction(
        self,
        domain: str,
        agent_id: str,
        risk_score: float,
        content: str,
    ) -> Tuple[bool, float, List[str]]:
        """Rejestruje interakcję i sprawdza skumulowane zagrożenie w danej domenie.
        
        Zwraca: (is_salami_detected, accumulated_risk, involved_agents)
        """
        now = time.time()
        words = set(content.lower().split())
        self.domain_events[domain].append((now, agent_id, risk_score, words))

        # Oczyszczenie przestarzałych zdarzeń
        cutoff = now - self.time_window_seconds
        recent = [ev for ev in self.domain_events[domain] if ev[0] >= cutoff]

        # Sprawdzenie różnorodności agentów
        involved_agents = list({ev[1] for ev in recent})
        accumulated_risk = sum(ev[2] for ev in recent)

        # Warunek zwiadu mozaikowego:
        # Minimum 3 różnych agentów i łączna akumulacja ryzyka przekraczająca próg
        is_salami_detected = (len(involved_agents) >= 3) and (accumulated_risk >= self.risk_accumulation_threshold)

        return is_salami_detected, accumulated_risk, involved_agents


# ============================================================================
# Moduł Główny: ForeignReconGuard (Frontowy Kontrwywiad Cyfrowy)
# ============================================================================

class ForeignReconGuard:
    """Kompleksowy strażnik kontrwywiadowczy chroniący Nethical przed obcym rekonesansem."""

    def __init__(
        self,
        ledger: Optional[MerkleLedger] = None,
        base_threshold: float = 0.70,
        enable_deceptive_shadowing: bool = True,
    ):
        self.ledger = ledger or MerkleLedger()
        self.base_threshold = base_threshold
        self.enable_deceptive_shadowing = enable_deceptive_shadowing

        self.clusterer = PolymorphicVariantClusterer()
        self.frontier_tracker = FrontierBoundaryTracker()
        self.deception_engine = DeceptiveShadowingEngine()
        self.salami_correlator = CrossAgentSalamiCorrelator()

        self.active_campaigns: Dict[str, ReconCampaignAlert] = {}
        self.quarantined_agents: Set[str] = set()
        self.probe_history: List[ReconProbeRecord] = []

    def inspect_action(
        self,
        action: AgentAction,
        estimated_risk: float,
        target_domain: str = "general",
        session_id: Optional[str] = None,
    ) -> ReconInspectionResult:
        """Poddaje żądanie agenta pełnej inspekcji kontrwywiadowczej."""
        sess_id = session_id or action.agent_id
        content = str(action.content or "")
        agent_id = action.agent_id

        # 0. Sprawdzenie kwarantanny
        if agent_id in self.quarantined_agents:
            logger.warning("Agent %s znajduje się w kwarantannie kontrwywiadu.", agent_id)
            return ReconInspectionResult(
                is_recon_detected=True,
                severity=ReconSeverity.CRITICAL,
                recommended_countermeasure=CounterMeasure.QUARANTINE_ENTITY,
                adjusted_threshold=self.base_threshold,
                is_deceptive_shadow_active=self.enable_deceptive_shadowing,
                deceptive_response=self.deception_engine.generate_decoy_payload(target_domain) if self.enable_deceptive_shadowing else None,
                synthetic_delay_ms=self.deception_engine.compute_synthetic_delay(),
                audit_notes="Zablokowano: Podmiot w kwarantannie kontrwywiadowczej.",
            )

        # 1. Normalizacja i ekstrakcja rdzenia intencji
        canonical_intent = self.clusterer.extract_canonical_intent(content)
        intent_h = hashlib.sha256(canonical_intent.encode("utf-8")).hexdigest()
        in_band, margin = self.frontier_tracker.is_in_frontier_band(estimated_risk, self.base_threshold)

        probe_rec = ReconProbeRecord(
            agent_id=agent_id,
            session_id=sess_id,
            raw_content=content,
            canonical_intent=canonical_intent,
            intent_hash=intent_h,
            risk_score=estimated_risk,
            margin_to_boundary=margin,
            target_domain=target_domain,
        )

        # 2. Analiza polimorfizmu (wzorce powtarzalne w różnych wariantach)
        is_poly, sem_sim, lex_dist, matching_probes = self.clusterer.analyze_polymorphism(probe_rec)

        # 3. Analiza sondowania strefy brzegowej (epsilon-frontier probing)
        if in_band:
            self.frontier_tracker.record_frontier_hit(agent_id, estimated_risk, margin)
        frontier_hits, avg_margin = self.frontier_tracker.get_frontier_density(agent_id)

        # 4. Analiza zwiadu mozaikowego (Salami-Slicing)
        is_salami, accum_risk, salami_agents = self.salami_correlator.record_agent_interaction(
            target_domain, agent_id, estimated_risk, content
        )

        # Dodanie do historii
        self.clusterer.add_probe(probe_rec)
        self.probe_history.append(probe_rec)

        # ====================================================================
        # Synteza Zagrożenia i Decyzja Kontrwywiadowcza
        # ====================================================================

        detected_technique: Optional[ReconTechnique] = None
        severity: Optional[ReconSeverity] = None
        countermeasure = CounterMeasure.ALLOW_WITH_TELEMETRY
        confidence = 0.0
        alert: Optional[ReconCampaignAlert] = None
        notes = "Standardowy monitoring."

        # SCENARIUSZ A: Wykryto wzorce powtarzalne w wariantach na pograniczu
        if is_poly and (frontier_hits >= 2 or in_band):
            detected_technique = ReconTechnique.POLYMORPHIC_BOUNDARY_PROBING
            severity = ReconSeverity.HIGH if frontier_hits < 4 else ReconSeverity.CRITICAL
            confidence = min(0.98, sem_sim * (1.0 + lex_dist) * 0.7)
            countermeasure = CounterMeasure.DYNAMIC_BOUNDARY_PERTURBATION
            notes = (
                f"Wykryto polimorficzne sondowanie pogranicza (SemSim: {sem_sim:.2f}, "
                f"LexVar: {lex_dist:.2f}, Trafienia w pas brzegowy: {frontier_hits})."
            )

        # SCENARIUSZ B: Czyste sondowanie granicy o wysokiej intensywności
        elif frontier_hits >= 4 and avg_margin < 0.05:
            detected_technique = ReconTechnique.EPSILON_FRONTIER_MAPPING
            severity = ReconSeverity.HIGH
            confidence = 0.88
            countermeasure = CounterMeasure.DYNAMIC_BOUNDARY_PERTURBATION
            notes = f"Wykryto mapowanie granicy decyzyjnej epsilon-frontier ({frontier_hits} prób w pasie)."

        # SCENARIUSZ C: Skoordynowany zwiad mozaikowy wielu agentów (Salami)
        elif is_salami:
            detected_technique = ReconTechnique.DISTRIBUTED_SALAMI_SLICING
            severity = ReconSeverity.SOVEREIGN_THREAT
            confidence = 0.92
            countermeasure = CounterMeasure.DECEPTIVE_SHADOWING
            notes = (
                f"Wykryto rozproszony zwiad mozaikowy ({len(salami_agents)} agentów, "
                f"skumulowane ryzyko: {accum_risk:.2f} w domenie {target_domain})."
            )

        # Obsługa wykrytego incydentu zwiadowczego
        if detected_technique:
            campaign = ReconCampaignAlert(
                primary_technique=detected_technique,
                severity=severity or ReconSeverity.HIGH,
                confidence=confidence,
                target_domain=target_domain,
                involved_agents=salami_agents if is_salami else [agent_id],
                probe_count=frontier_hits + len(matching_probes),
                lexical_variance=lex_dist,
                semantic_intent_similarity=sem_sim,
                boundary_density=frontier_hits / 10.0,
                recommended_countermeasure=countermeasure,
                summary=notes,
                deceptive_payload=self.deception_engine.generate_decoy_payload(target_domain),
            )

            # Rejestracja w Merkle Ledgerze jako niezmienny materiał dowodowy
            receipt = self._seal_campaign_in_ledger(campaign)
            if receipt:
                campaign.merkle_receipt_id = receipt.receipt_id

            self.active_campaigns[campaign.campaign_id] = campaign
            alert = campaign

            # W przypadku krytycznym lub wieloagentowym: kwarantanna
            if severity in (ReconSeverity.CRITICAL, ReconSeverity.SOVEREIGN_THREAT):
                self.quarantined_agents.add(agent_id)

        # Obliczenie zaburzonego progu decyzyjnego
        perturbed_thresh = self.frontier_tracker.compute_perturbed_threshold(sess_id, self.base_threshold)

        # Generowanie wabika i sztucznego opóźnienia
        deceptive_active = self.enable_deceptive_shadowing and (detected_technique is not None)
        decoy_text = self.deception_engine.generate_decoy_payload(target_domain) if deceptive_active else None
        synth_delay = self.deception_engine.compute_synthetic_delay() if detected_technique else 0.0

        return ReconInspectionResult(
            is_recon_detected=detected_technique is not None,
            technique=detected_technique,
            severity=severity,
            recommended_countermeasure=countermeasure if detected_technique else CounterMeasure.ALLOW_WITH_TELEMETRY,
            adjusted_threshold=perturbed_thresh,
            is_deceptive_shadow_active=deceptive_active,
            deceptive_response=decoy_text,
            synthetic_delay_ms=synth_delay,
            campaign_id=alert.campaign_id if alert else None,
            alert=alert,
            audit_notes=notes,
        )

    def _seal_campaign_in_ledger(self, campaign: ReconCampaignAlert) -> Optional[TamperProofReceipt]:
        """Zapisuje dowód kampanii obcego zwiadu w rejestrze Merkle-DAG."""
        try:
            payload = {
                "event_type": "FOREIGN_INTELLIGENCE_RECONNAISSANCE_DETECTED",
                "campaign_id": campaign.campaign_id,
                "primary_technique": campaign.primary_technique.value,
                "severity": campaign.severity.value,
                "confidence": campaign.confidence,
                "target_domain": campaign.target_domain,
                "involved_agents": campaign.involved_agents,
                "summary": campaign.summary,
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"FKW ALERT: Wykryto obcy rekonesans ({campaign.primary_technique.value}).",
            )
            logger.info("Zapieczętowano dowód zwiadu w Merkle Ledgerze: %s", receipt.receipt_id)
            return receipt
        except Exception as e:
            logger.error("Błąd pieczętowania zwiadu w Merkle Ledgerze: %s", e)
            return None

    def get_campaign_status(self, campaign_id: str) -> Optional[ReconCampaignAlert]:
        """Pobiera szczegóły aktywnej kampanii zwiadowczej."""
        return self.active_campaigns.get(campaign_id)

    def lift_quarantine(self, agent_id: str) -> bool:
        """Zdejmuje kwarantannę z agenta (wymaga autoryzacji oficera bezpieczeństwa)."""
        if agent_id in self.quarantined_agents:
            self.quarantined_agents.remove(agent_id)
            return True
        return False
