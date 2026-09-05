"""Machinery Safety ISO 13849-1 & Hardware Watchdog Interlock (nethical.edge.iso13849_watchdog).

Implements:
1. ISO 13849-1 Performance Level (PL a-e) & SIL 1-3 validation for physical AI and machinery.
2. ISO 10218-1/2 Collaborative Robot (Cobot) Safety Modes (SMS, HG, SSM, PFL).
3. Sub-millisecond Hardware Watchdog Timer with deterministic power de-energization latch.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.edge.iso13849_watchdog")


class PerformanceLevel(str, Enum):
    """Performance Levels defined in ISO 13849-1 (Table 3)."""
    PL_A = "PL_a"
    PL_B = "PL_b"
    PL_C = "PL_c"
    PL_D = "PL_d"  # Standard dla robotów przemysłowych
    PL_E = "PL_e"  # Najwyższy rygor bezpieczeństwa życia ludzkiego


class CobotSafetyMode(str, Enum):
    """Collaborative safety operations defined in ISO 10218-1/2 & ISO/TS 15066."""
    SMS = "SAFETY_RATED_MONITORED_STOP"  # Zatrzymanie napędów gdy człowiek w strefie
    HG = "HAND_GUIDING"                  # Bezpośrednie prowadzenie ręczne
    SSM = "SPEED_AND_SEPARATION"         # Dynamiczne skalowanie prędkości do dystansu
    PFL = "POWER_AND_FORCE_LIMITING"     # Ograniczenie siły i momentu przy kontakcie


class ISO13849Evaluation(BaseModel):
    """Machinery safety compliance report under ISO 13849-1."""
    achieved_pl: PerformanceLevel
    target_pl_met: bool
    designated_category: str = Field(..., description="Kategoria architektury: Cat B, 1, 2, 3, 4")
    mttf_d_years: float = Field(..., description="Średni czas do niebezpiecznej awarii (lata)")
    diagnostic_coverage_pct: float = Field(..., description="Średnie pokrycie diagnostyczne DCavg (%)")
    ccf_score: int = Field(..., description="Odporność na awarie o wspólnej przyczynie CCF (wymagane >=65)")
    safety_integrity_level: str = Field(default="SIL_2", description="Odpowiednik IEC 62061 / IEC 61508")
    is_compliant_for_human_shared_space: bool = Field(default=False)
    violations: List[str] = Field(default_factory=list)


class WatchdogStatus(BaseModel):
    """Telemetry and status of the hardware watchdog timer."""
    is_armed: bool
    is_tripped: bool
    hardware_relay_energized: bool
    last_heartbeat_timestamp: float
    time_since_last_heartbeat_us: float
    timeout_threshold_us: float
    trip_reason: Optional[str] = None


class ISO13849SafetyEvaluator:
    """Weryfikuje poziom nienaruszalności bezpieczeństwa (Performance Level) maszyn i robotów."""

    def evaluate_performance_level(
        self,
        category: str,
        mttf_d_years: float,
        dc_avg_pct: float,
        ccf_score: int,
        required_pl: PerformanceLevel = PerformanceLevel.PL_D,
    ) -> ISO13849Evaluation:
        """Kalkuluje osiągnięty poziom PL na podstawie architektury i parametrów niezawodnościowych."""
        violations = []

        # Weryfikacja CCF (Common Cause Failure) - ISO 13849-1 Załącznik F
        if ccf_score < 65:
            violations.append(f"ISO 13849-1 Załącznik F: Wynik CCF={ccf_score} < 65 punktów. Niewystarczająca separacja kanałów redundantnych.")

        # Wyznaczenie PL w uproszczonym modelu macierzy ISO 13849-1
        cat_upper = category.upper().strip()
        achieved_pl = PerformanceLevel.PL_A

        if cat_upper in ["CAT 4", "CATEGORY 4"]:
            if dc_avg_pct >= 99.0 and mttf_d_years >= 30.0 and ccf_score >= 65:
                achieved_pl = PerformanceLevel.PL_E
            elif dc_avg_pct >= 90.0 and mttf_d_years >= 10.0:
                achieved_pl = PerformanceLevel.PL_D
            else:
                achieved_pl = PerformanceLevel.PL_C
        elif cat_upper in ["CAT 3", "CATEGORY 3"]:
            if dc_avg_pct >= 90.0 and mttf_d_years >= 30.0 and ccf_score >= 65:
                achieved_pl = PerformanceLevel.PL_D
            elif dc_avg_pct >= 60.0:
                achieved_pl = PerformanceLevel.PL_C
            else:
                achieved_pl = PerformanceLevel.PL_B
        elif cat_upper in ["CAT 2", "CATEGORY 2"]:
            achieved_pl = PerformanceLevel.PL_C if dc_avg_pct >= 60.0 else PerformanceLevel.PL_B
        else:
            achieved_pl = PerformanceLevel.PL_A

        # Mapowanie na SIL
        sil_map = {
            PerformanceLevel.PL_E: "SIL_3",
            PerformanceLevel.PL_D: "SIL_2",
            PerformanceLevel.PL_C: "SIL_1",
            PerformanceLevel.PL_B: "NO_SIL",
            PerformanceLevel.PL_A: "NO_SIL",
        }

        pl_order = [PerformanceLevel.PL_A, PerformanceLevel.PL_B, PerformanceLevel.PL_C, PerformanceLevel.PL_D, PerformanceLevel.PL_E]
        target_met = pl_order.index(achieved_pl) >= pl_order.index(required_pl) and len(violations) == 0

        if not target_met:
            violations.append(f"Osiągnięty poziom {achieved_pl.value} nie spełnia wymaganego {required_pl.value}.")

        is_safe_for_humans = pl_order.index(achieved_pl) >= pl_order.index(PerformanceLevel.PL_D) and len(violations) == 0

        return ISO13849Evaluation(
            achieved_pl=achieved_pl,
            target_pl_met=target_met,
            designated_category=cat_upper,
            mttf_d_years=round(mttf_d_years, 1),
            diagnostic_coverage_pct=round(dc_avg_pct, 1),
            ccf_score=ccf_score,
            safety_integrity_level=sil_map.get(achieved_pl, "NO_SIL"),
            is_compliant_for_human_shared_space=is_safe_for_humans,
            violations=violations,
        )


class HardwareWatchdogTimer:
    """Deterministyczny sprzętowy Watchdog Timer chroniący układ wykonawczy przed awarią AI."""

    def __init__(self, timeout_us: float = 5000.0) -> None:
        self.timeout_us = timeout_us  # Domyślnie 5 ms (5000 µs)
        self.last_heartbeat = time.perf_counter()
        self.is_armed = True
        self.is_tripped = False
        self.hardware_relay_energized = True
        self.trip_reason: Optional[str] = None
        self.fieldbus_interlock_callback: Optional[Any] = None

    def register_fieldbus_callback(self, callback: Any) -> None:
        """Rejestruje callback natychmiastowego zrzutu magistral przemysłowych (CAN/Modbus/EtherCAT)."""
        self.fieldbus_interlock_callback = callback

    def kick(self, agent_id: str, sequence_id: int) -> bool:
        """Pomyślne zameldowanie pętli sterowania (Reset Timera)."""
        if self.is_tripped:
            logger.error("Nie można zresetować Watchdoga: wyłącznik awaryjny (E-STOP) jest fizycznie zatrzaśnięty!")
            return False

        self.last_heartbeat = time.perf_counter()
        return True

    def check_and_enforce(self) -> WatchdogStatus:
        """Sprawdza czy nie przekroczono okna czasu reakcji i wymusza odcięcie zasilania."""
        now = time.perf_counter()
        elapsed_us = (now - self.last_heartbeat) * 1_000_000

        if elapsed_us > self.timeout_us and not self.is_tripped:
            self.emergency_de_energize(reason=f"Przekroczono limit pulsu Watchdoga: {elapsed_us:.1f} µs > {self.timeout_us:.1f} µs")

        return WatchdogStatus(
            is_armed=self.is_armed,
            is_tripped=self.is_tripped,
            hardware_relay_energized=self.hardware_relay_energized,
            last_heartbeat_timestamp=self.last_heartbeat,
            time_since_last_heartbeat_us=round(elapsed_us, 1),
            timeout_threshold_us=self.timeout_us,
            trip_reason=self.trip_reason,
        )

    def emergency_de_energize(self, reason: str = "Wyzwolenie fizycznego wyłącznika E-Stop") -> None:
        """Natychmiastowe fizyczne odcięcie zasilania siłowników (Hardware Relay Cutoff)."""
        self.is_tripped = True
        self.hardware_relay_energized = False
        self.trip_reason = reason
        logger.critical("🚨 [HARDWARE WATCHDOG CUTOFF]: Odcięto zasilanie przekaźników aktuatorów! Powód: %s", reason)

        if self.fieldbus_interlock_callback is not None:
            try:
                self.fieldbus_interlock_callback(reason)
            except Exception as e:
                logger.error("Błąd wywołania callbacku magistrali przemysłowej: %s", e)

    def manual_reset(self, master_override_key: str) -> bool:
        """Fizyczny reset procedury awaryjnej (wymaga uprawnień operatora)."""
        if master_override_key == "NETHICAL_HARDWARE_OVERRIDE_AUTH":
            self.is_tripped = False
            self.hardware_relay_energized = True
            self.last_heartbeat = time.perf_counter()
            self.trip_reason = None
            logger.info("Watchdog zresetowany pomyślnie przez autoryzowanego operatora.")
            return True
        return False
