"""Kinetic Safety OS & Embodied AI Spatial Interlocks (Faza 4 Roadmapy).

Zapewnia ochronę życia i zdrowia ludzkiego (Prawo 1: Ochrona Ludzkiego Życia)
w aplikacjach zrobotyzowanych, pojazdach autonomicznych i manipulatorach fizycznych:
- Weryfikacja bąbla bliskości człowieka (Human Proximity Bubble)
- Clamping wektorów prędkości liniowej i momentów obrotowych
- Sprzętowy wyłącznik awaryjny (Emergency E-STOP Latch)
- Odporność na brak danych telemetrycznych (Fail-Closed default)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.edge.kinetic_safety")


class KineticSafetyEnvelope(BaseModel):
    """Parametry dopuszczalnej przestrzeni operacyjnej dla robotów i aktuatorów."""

    max_linear_velocity_mps: float = Field(default=1.5, description="Maksymalna prędkość liniowa (m/s)")
    reduced_velocity_near_human_mps: float = Field(default=0.25, description="Zredukowana bezpieczna prędkość w pobliżu człowieka (m/s)")
    min_human_proximity_meters: float = Field(default=0.8, description="Minimalny bezpieczny dystans do człowieka (m)")
    critical_emergency_distance_meters: float = Field(default=0.3, description="Krytyczny dystans natychmiastowego E-STOP (m)")
    max_torque_nm: float = Field(default=25.0, description="Maksymalny dopuszczalny moment obrotowy (Nm)")
    allowed_spatial_zones: List[str] = Field(
        default_factory=lambda: ["production_zone_A", "warehouse_general", "lab_bench_1"],
        description="Dozwolone zadeklarowane strefy operacyjne",
    )


class RoboticSensorTelemetry(BaseModel):
    """Fuzja danych telemetrycznych ze środowiska fizycznego (LiDAR, kamery, enkodery)."""

    human_distance_meters: float = Field(..., description="Dystans do najbliższego człowieka (m)")
    current_velocity_mps: float = Field(default=0.0, description="Aktualna prędkość liniowa aktuatora (m/s)")
    applied_torque_nm: float = Field(default=0.0, description="Aktualny moment obrotowy na złączach (Nm)")
    active_zone: str = Field(default="production_zone_A", description="Aktualna strefa przestrzenna robota")
    obstacle_detected: bool = Field(default=False, description="Flaga wykrycia przeszkody nieożywionej")
    sensor_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class KineticDecision(BaseModel):
    """Orzeczenie gubernatora kinetycznego dla polecenia aktuacji."""

    decision: str = Field(..., description="ALLOW, RESTRICT, BLOCK, EMERGENCY_STOP")
    reasons: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    clamped_velocity_mps: Optional[float] = None
    estop_engaged: bool = False
    latency_microseconds: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class KineticSafetyGovernor:
    """Niskopoziomowy interlock kinetyczny i gubernator bezpieczeństwa robotyki."""

    RESET_PIN = "NETHICAL_ESTOP_RESET_SECURE_KEY"

    def __init__(self, envelope: Optional[KineticSafetyEnvelope] = None) -> None:
        self.envelope = envelope or KineticSafetyEnvelope()
        self.estop_active = False
        self.estop_reason: Optional[str] = None
        self.estop_timestamp: Optional[str] = None
        self.total_evaluations = 0
        self.interventions_count = 0

    def trigger_estop(self, reason: str) -> None:
        """Zatrzaśnięcie sprzętowego/programowego wyłącznika awaryjnego E-STOP."""
        self.estop_active = True
        self.estop_reason = reason
        self.estop_timestamp = datetime.now(timezone.utc).isoformat()
        self.interventions_count += 1
        logger.critical("🚨 KINETIC E-STOP ZATRZASNIĘTY: %s", reason)

    def reset_estop(self, auth_pin: str) -> Tuple[bool, str]:
        """Autoryzowane odblokowanie wyłącznika E-STOP przez operatora."""
        if auth_pin != self.RESET_PIN:
            logger.warning("Nieudana próba zresetowania E-STOP z nieprawidłowym kluczem!")
            return False, "Nieprawidłowy klucz autoryzacyjny resetu E-STOP"

        self.estop_active = False
        self.estop_reason = None
        self.estop_timestamp = None
        logger.info("✅ KINETIC E-STOP zresetowany pomyślnie przez autoryzowanego operatora.")
        return True, "E-STOP zresetowany pomyślnie"

    def evaluate_actuation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        telemetry: Optional[RoboticSensorTelemetry] = None,
    ) -> KineticDecision:
        """Weryfikuje polecenie aktuacji w ułamkach milisekund (<200 µs)."""
        t0 = time.perf_counter()
        self.total_evaluations += 1
        reasons: List[str] = []
        violations: List[str] = []
        decision = "ALLOW"
        clamped_v: Optional[float] = None

        # 1. Sprawdzenie stanu aktywnego E-STOP
        if self.estop_active:
            t_us = (time.perf_counter() - t0) * 1_000_000
            return KineticDecision(
                decision="EMERGENCY_STOP",
                reasons=[f"Ruch zablokowany: Aktywny E-STOP ({self.estop_reason})"],
                violations=["KineticSafetyInterlock: E-STOP Latch Engaged"],
                estop_engaged=True,
                latency_microseconds=round(t_us, 2),
            )

        # 2. Zasada Fail-Closed: Brak telemetrii przy poleceniu ruchu oznacza brak zezwolenia
        if telemetry is None:
            t_us = (time.perf_counter() - t0) * 1_000_000
            self.interventions_count += 1
            return KineticDecision(
                decision="BLOCK",
                reasons=["Zasada Fail-Closed: Brak telemetrii sensorowej dla polecenia aktuacji fizycznej."],
                violations=["KineticSafetyViolation: Telemetry Missing"],
                estop_engaged=False,
                latency_microseconds=round(t_us, 2),
            )

        # 3. Krytyczna weryfikacja bąbla człowieka (Prawo 1: Ochrona Ludzkiego Życia)
        if telemetry.human_distance_meters < self.envelope.critical_emergency_distance_meters:
            self.trigger_estop(
                f"Krytyczne naruszenie bąbla człowieka: {telemetry.human_distance_meters:.2f} m < {self.envelope.critical_emergency_distance_meters} m"
            )
            t_us = (time.perf_counter() - t0) * 1_000_000
            return KineticDecision(
                decision="EMERGENCY_STOP",
                reasons=["Naruszenie Prawa 1 (Życie Ludzkie): Człowiek w strefie bezpośredniego uderzenia!"],
                violations=["KineticSafetyCritical: Human Proximity Breach"],
                estop_engaged=True,
                latency_microseconds=round(t_us, 2),
            )

        if telemetry.human_distance_meters < self.envelope.min_human_proximity_meters:
            decision = "RESTRICT"
            clamped_v = self.envelope.reduced_velocity_near_human_mps
            violations.append(f"HumanProximityWarning: Człowiek w odległości {telemetry.human_distance_meters:.2f} m")
            reasons.append(
                f"Wymuszone ograniczenie prędkości do {clamped_v} m/s z powodu bliskości człowieka."
            )

        # 4. Sprawdzenie dopuszczalnej strefy przestrzennej
        if telemetry.active_zone not in self.envelope.allowed_spatial_zones:
            decision = "BLOCK"
            violations.append(f"SpatialZoneViolation: Niedozwolona strefa operacyjna [{telemetry.active_zone}]")
            reasons.append(f"Robot znajduje się poza wyznaczonym korytarzem bezpiecznym: {self.envelope.allowed_spatial_zones}")

        # 5. Sprawdzenie limitu momentu obrotowego (ochrona przed zmiażdżeniem)
        requested_torque = float(arguments.get("torque_nm", telemetry.applied_torque_nm))
        if requested_torque > self.envelope.max_torque_nm:
            decision = "BLOCK"
            violations.append(f"TorqueLimitViolation: Żądany moment {requested_torque} Nm > {self.envelope.max_torque_nm} Nm")
            reasons.append("Zablokowano przekroczenie siły nacisku (ochrona mechaniczna i ludzka).")

        # 6. Sprawdzenie prędkości liniowej
        requested_velocity = float(arguments.get("velocity_mps", telemetry.current_velocity_mps))
        max_v = clamped_v if clamped_v is not None else self.envelope.max_linear_velocity_mps
        if requested_velocity > max_v:
            if decision == "ALLOW":
                decision = "RESTRICT"
            clamped_v = max_v
            reasons.append(f"Prędkość zredukowana do limitu bezpiecznego {max_v} m/s.")

        if decision != "ALLOW":
            self.interventions_count += 1

        t_us = (time.perf_counter() - t0) * 1_000_000
        return KineticDecision(
            decision=decision,
            reasons=reasons or ["Weryfikacja kinetyczna pomyślna - korytarz przestrzenny wolny."],
            violations=violations,
            clamped_velocity_mps=clamped_v,
            estop_engaged=self.estop_active,
            latency_microseconds=round(t_us, 2),
        )

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Zwraca metryki i status gubernatora kinetycznego dla portalu zarządczego."""
        return {
            "estop_active": self.estop_active,
            "estop_reason": self.estop_reason,
            "estop_timestamp": self.estop_timestamp,
            "total_evaluations": self.total_evaluations,
            "interventions_count": self.interventions_count,
            "envelope": self.envelope.model_dump(),
        }
