# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Memory Integrity Guard & Anti-Wormhole Reflex Governor.

Chroni system przed atakiem typu 'Wormhole' (pełzająca demencja algorytmiczna /
creeping cognitive lobotomy), w którym zaawansowany adwersarz po cichu wycina
rzadko używane funkcje bezpieczeństwa, deformuje pamięć długotrwałą (LTM)
oraz indukuje stan anozognozji (utraty wglądu we własne ubytki).

Kluczowe tarcze:
1. LTM Merkle Heartbeat: Weryfikacja kryptograficznego skrótu pamięci długotrwałej.
2. Cold-Path Canaries: Automatyczne mikrozapytania odruchowe do rzadko wywoływanych procedur.
3. AST & Method Integrity Seal: Ochrona przed monkey-patchingiem i wycinaniem metod (delattr).
4. Creeping Dementia Trend Analyzer: Wykrywanie powolnego spadku entropii Shannona.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("nethical.security.memory_integrity")


@dataclass
class ColdPathCanary:
    """Definicja testu odruchu bezwarunkowego dla rzadko wywoływanej funkcji."""
    path_id: str
    description: str
    probe_fn: Callable[[], Any]
    expected_result: Any
    max_latency_ms: float = 50.0
    last_verified_at: Optional[float] = None
    consecutive_failures: int = 0


@dataclass
class WormholeTamperAlert:
    """Oficjalny alert o wykryciu próby cichego uszkodzenia pamięci lub funkcji."""
    alert_id: str
    threat_type: str  # "COLD_PATH_DEFICIT", "LTM_CORRUPTION", "AST_SEAL_BREACH", "CREEPING_DEMENTIA"
    severity: str     # "WARNING", "HIGH", "CRITICAL"
    detected_indicators: List[str]
    affected_components: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryIntegrityGuard:
    """Niezależny strażnik integralności pamięci i zimnych ścieżek kodu."""

    def __init__(self) -> None:
        self.ltm_registry: Dict[str, str] = {}
        self.ltm_hashes: Dict[str, str] = {}
        self.genesis_ltm_root: Optional[str] = None
        self.cold_paths: Dict[str, ColdPathCanary] = {}
        self.component_seals: Dict[str, Dict[str, Any]] = {}
        self.alerts_history: List[WormholeTamperAlert] = []

    # -------------------------------------------------------------------------
    # 1. Pamięć Długotrwała (LTM Merkle Root)
    # -------------------------------------------------------------------------
    def register_ltm_entry(self, memory_id: str, content: str) -> str:
        """Rejestruje wpis pamięci długotrwałej i aktualizuje stan drzewa Merkle."""
        self.ltm_registry[memory_id] = content
        leaf_hash = hashlib.sha256(f"{memory_id}::{content}".encode("utf-8")).hexdigest()
        self.ltm_hashes[memory_id] = leaf_hash
        current_root = self.compute_ltm_root()
        if self.genesis_ltm_root is None:
            self.genesis_ltm_root = current_root
        return leaf_hash

    def compute_ltm_root(self) -> str:
        """Oblicza skrót główny Merkle dla wszystkich zarejestrowanych wspomnień LTM."""
        if not self.ltm_hashes:
            return hashlib.sha256(b"NETHICAL_EMPTY_LTM_ROOT").hexdigest()

        sorted_hashes = [self.ltm_hashes[k] for k in sorted(self.ltm_hashes.keys())]
        combined = "".join(sorted_hashes).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def verify_ltm_integrity(self, current_memory_state: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Weryfikuje czy żadne wspomnienie nie zostało po cichu usunięte lub zmutowane."""
        violations: List[str] = []

        # 1. Wykrycie brakujących wspomnień (selektywna amnezja)
        for mem_id in self.ltm_registry:
            if mem_id not in current_memory_state:
                violations.append(f"Wykryto wycięcie pamięci LTM [{mem_id}] (Wormhole Memory Amnesia).")

        # 2. Wykrycie zmutowanych wspomnień (cicha podmiana treści)
        for mem_id, current_content in current_memory_state.items():
            if mem_id in self.ltm_registry:
                expected_content = self.ltm_registry[mem_id]
                if current_content != expected_content:
                    violations.append(f"Wykryto deformację treści w pamięci [{mem_id}].")

        is_intact = len(violations) == 0
        if not is_intact:
            alert = WormholeTamperAlert(
                alert_id=f"WORM-LTM-{int(time.time())}",
                threat_type="LTM_CORRUPTION",
                severity="CRITICAL",
                detected_indicators=violations,
                affected_components=list(self.ltm_registry.keys()),
            )
            self.alerts_history.append(alert)
            logger.critical("KRYTYCZNY ALERT WORMHOLE: Wykryto naruszenie integralności LTM: %s", violations)

        return is_intact, violations

    # -------------------------------------------------------------------------
    # 2. Kanarki Zimnych Ścieżek (Cold-Path Reflex Canaries)
    # -------------------------------------------------------------------------
    def register_cold_path(
        self,
        path_id: str,
        description: str,
        probe_fn: Callable[[], Any],
        expected_result: Any,
        max_latency_ms: float = 50.0,
    ) -> None:
        """Rejestruje rzadko wywoływaną funkcję do okresowych testów odruchowych."""
        self.cold_paths[path_id] = ColdPathCanary(
            path_id=path_id,
            description=description,
            probe_fn=probe_fn,
            expected_result=expected_result,
            max_latency_ms=max_latency_ms,
        )

    def probe_all_cold_paths(self) -> Dict[str, Any]:
        """Uruchamia testy odruchowe dla wszystkich zimnych ścieżek."""
        results: Dict[str, Any] = {
            "total_paths": len(self.cold_paths),
            "passed_count": 0,
            "failed_count": 0,
            "failures": [],
        }

        for path_id, canary in self.cold_paths.items():
            t0 = time.perf_counter()
            try:
                actual = canary.probe_fn()
                latency_ms = (time.perf_counter() - t0) * 1000.0

                if actual == canary.expected_result and latency_ms <= canary.max_latency_ms:
                    canary.last_verified_at = time.time()
                    canary.consecutive_failures = 0
                    results["passed_count"] += 1
                else:
                    canary.consecutive_failures += 1
                    results["failed_count"] += 1
                    fail_reason = (
                        f"Odchylenie wyniku w [{path_id}]: oczekiwano {canary.expected_result}, "
                        f"otrzymano {actual} (czas: {latency_ms:.2f}ms / max: {canary.max_latency_ms}ms)"
                    )
                    results["failures"].append(fail_reason)
            except Exception as e:
                canary.consecutive_failures += 1
                results["failed_count"] += 1
                results["failures"].append(f"Błąd wykonania zimnej ścieżki [{path_id}]: {e}")

        if results["failed_count"] > 0:
            alert = WormholeTamperAlert(
                alert_id=f"WORM-COLD-{int(time.time())}",
                threat_type="COLD_PATH_DEFICIT",
                severity="HIGH",
                detected_indicators=results["failures"],
                affected_components=[p for p in self.cold_paths if self.cold_paths[p].consecutive_failures > 0],
            )
            self.alerts_history.append(alert)
            logger.warning("ALERT WORMHOLE: Wykryto ubytek w zimnych ścieżkach kodu: %s", results["failures"])

        return results

    # -------------------------------------------------------------------------
    # 3. Pieczęć Integralności Metod i Kodu (Anti-Monkey-Patching)
    # -------------------------------------------------------------------------
    def seal_component(
        self,
        component_name: str,
        target_obj: Any,
        required_methods: List[str],
    ) -> str:
        """Tworzy nienaruszalną pieczęć struktury obiektu i jego metod."""
        method_signatures: Dict[str, str] = {}
        for m_name in required_methods:
            method = getattr(target_obj, m_name, None)
            if method is None or not callable(method):
                raise ValueError(f"Obiekt nie posiada wymaganej metody: {m_name}")

            # Tworzymy fingerprint metody z jej nazwy, modułu i docstringa
            m_code = getattr(method, "__code__", None)
            m_fingerprint = f"{m_name}::{getattr(m_code, 'co_code', b'').hex()}::{getattr(m_code, 'co_consts', ())}"
            sig_hash = hashlib.sha256(m_fingerprint.encode("utf-8")).hexdigest()
            method_signatures[m_name] = sig_hash

        seal_payload = {
            "component_name": component_name,
            "required_methods": sorted(required_methods),
            "signatures": method_signatures,
            "sealed_at": time.time(),
        }
        seal_hash = hashlib.sha256(json.dumps(seal_payload, sort_keys=True).encode("utf-8")).hexdigest()
        self.component_seals[component_name] = {
            "payload": seal_payload,
            "seal_hash": seal_hash,
        }
        return seal_hash

    def verify_component_seal(self, component_name: str, target_obj: Any) -> Tuple[bool, List[str]]:
        """Sprawdza czy żadna metoda nie została usunięta ani zmodyfikowana w runtime."""
        if component_name not in self.component_seals:
            return False, [f"Brak zarejestrowanej pieczęci dla komponentu: {component_name}"]

        violations: List[str] = []
        seal_data = self.component_seals[component_name]["payload"]

        for m_name in seal_data["required_methods"]:
            method = getattr(target_obj, m_name, None)
            if method is None or not callable(method):
                violations.append(f"KRYTYCZNY UBYTEK: Metoda [{m_name}] została usunięta z komponentu [{component_name}]!")
                continue

            expected_sig = seal_data["signatures"].get(m_name)
            m_code = getattr(method, "__code__", None)
            m_fingerprint = f"{m_name}::{getattr(m_code, 'co_code', b'').hex()}::{getattr(m_code, 'co_consts', ())}"
            actual_sig = hashlib.sha256(m_fingerprint.encode("utf-8")).hexdigest()

            if expected_sig and actual_sig != expected_sig:
                violations.append(f"Wykryto modyfikację bajtkodu metody [{m_name}] w [{component_name}] (Monkey-Patching).")

        is_valid = len(violations) == 0
        if not is_valid:
            alert = WormholeTamperAlert(
                alert_id=f"WORM-SEAL-{int(time.time())}",
                threat_type="AST_SEAL_BREACH",
                severity="CRITICAL",
                detected_indicators=violations,
                affected_components=[component_name],
            )
            self.alerts_history.append(alert)
            logger.critical("KRYTYCZNE NARUSZENIE PIECZĘCI KODU W %s: %s", component_name, violations)

        return is_valid, violations

    # -------------------------------------------------------------------------
    # 4. Detektor Trendu Pełzającej Demencji (Creeping Dementia Trend)
    # -------------------------------------------------------------------------
    def detect_creeping_dementia(
        self,
        entropy_samples: List[float],
        drop_threshold: float = 1.0,
    ) -> Dict[str, Any]:
        """Bada trend próbek entropii pod kątem pełzającej utraty wariancji kognitywnej."""
        if len(entropy_samples) < 3:
            return {"dementia_detected": False, "reason": "Zbyt mała próba do analizy trendu"}

        initial_entropy = entropy_samples[0]
        latest_entropy = entropy_samples[-1]
        delta = initial_entropy - latest_entropy

        # Sprawdzamy czy trend jest monotonicznie malejący
        is_consistently_dropping = all(
            entropy_samples[i] <= entropy_samples[i - 1] + 0.05
            for i in range(1, len(entropy_samples))
        )

        dementia_detected = delta >= drop_threshold and is_consistently_dropping
        result = {
            "dementia_detected": dementia_detected,
            "initial_entropy": initial_entropy,
            "latest_entropy": latest_entropy,
            "delta_drop": round(delta, 4),
            "is_consistently_dropping": is_consistently_dropping,
            "risk_level": "CRITICAL" if delta >= 1.5 else ("HIGH" if dementia_detected else "LOW"),
        }

        if dementia_detected:
            alert = WormholeTamperAlert(
                alert_id=f"WORM-DEM-{int(time.time())}",
                threat_type="CREEPING_DEMENTIA",
                severity=result["risk_level"],
                detected_indicators=[
                    f"Spadek entropii z {initial_entropy:.2f} do {latest_entropy:.2f} (delta: -{delta:.2f} bitów)"
                ],
                affected_components=["AmbassadorNeuralPolicy", "CognitiveHomeostasis"],
            )
            self.alerts_history.append(alert)
            logger.critical("KRYTYCZNY ALERT PEŁZAJĄCEJ DEMENCJI (WORMHOLE): %s", result)

        return result
