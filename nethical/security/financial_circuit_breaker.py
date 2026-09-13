"""Financial Circuit Breakers & Runaway Protection for Autonomous Agents (nethical.security.financial_circuit_breaker).

Protects Agent-to-Agent (A2A) economic interactions from:
- Flash crashes and runaway transaction loops
- Capital drainage and rapid budget exhaustion
- Rogue automated high-frequency exploitation

Features Dual Safety Corridors (Lower & Upper Thresholds):
- Proactive Early Warning Corridor: Triggers adaptive micro-throttling at lower thresholds
  before risk metrics explode.
- Upper Boundary Trip: Enforces mandatory cooling-off when weighted multi-factor risk breaches
  the upper corridor.
- Hard Ceilings: Fail-closed HALT on cumulative budget depletion.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.security.financial_circuit_breaker")


class CircuitBreakerState(str, Enum):
    """Operational states of the financial circuit breaker."""
    NORMAL = "NORMAL"        # Pełna przepustowość w bezpiecznym korytarzu
    THROTTLED = "THROTTLED"  # Przekroczenie dolnego progu (Early Warning), progresywne opóźnienie
    TRIPPED = "TRIPPED"      # Przekroczenie górnego progu / limitu (okres schłodzenia - Cooling Off)
    HALTED = "HALTED"        # Całkowita blokada awaryjna - wymagany autoryzowany reset HITL


class FinancialTransaction(BaseModel):
    """Payload representing an agent financial or budget transfer operation."""
    tx_id: str
    initiator_agent_id: str
    target_agent_id: str
    amount: float = Field(..., ge=0.0)
    currency: str = Field(default="USD")
    session_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class CircuitBreakerDecision(BaseModel):
    """Verdict of the circuit breaker for a proposed financial transaction."""
    allowed: bool
    current_state: CircuitBreakerState
    reason: str
    applied_throttle_delay_ms: float = 0.0
    velocity_tx_per_min: float = 0.0
    hourly_volume_sum: float = 0.0
    composite_risk_score: float = 0.0
    lower_threshold: float = 0.40
    upper_threshold: float = 0.75
    early_warning_active: bool = False
    violations: List[str] = Field(default_factory=list)


class FinancialCircuitBreaker:
    """Rynkowy bezpiecznik z podwójnymi widełkami bezpieczeństwa (Dual Threshold Corridors)."""

    def __init__(
        self,
        max_single_tx_limit: float = 50_000.0,
        max_velocity_tx_per_min: int = 20,
        max_hourly_volume_limit: Optional[float] = None,
        cooling_off_seconds: float = 30.0,
        lower_threshold_ratio: float = 0.40,
        upper_threshold_ratio: float = 0.75,
        weight_velocity: float = 0.40,
        weight_volume: float = 0.35,
        weight_single_tx: float = 0.25,
    ) -> None:
        self.max_single_tx = max_single_tx_limit
        self.max_velocity = max_velocity_tx_per_min
        self.max_hourly_volume = max_hourly_volume_limit if max_hourly_volume_limit is not None else (max_single_tx_limit * 5.0)
        self.cooling_off_seconds = cooling_off_seconds

        # Widełki bezpieczeństwa (Dual Thresholds)
        self.lower_threshold = lower_threshold_ratio
        self.upper_threshold = upper_threshold_ratio

        # Znormalizowane wagi czynników ryzyka (suma = 1.0)
        total_weight = weight_velocity + weight_volume + weight_single_tx
        self.w_velocity = weight_velocity / total_weight
        self.w_volume = weight_volume / total_weight
        self.w_single_tx = weight_single_tx / total_weight

        self.state = CircuitBreakerState.NORMAL
        self.trip_timestamp: Optional[float] = None
        self.trip_reason: Optional[str] = None
        self.trip_violation_code: Optional[str] = None

        # Historia transakcji (okno przesuwne)
        self.tx_history: deque[Tuple[float, float]] = deque()  # (timestamp, amount)

    def calculate_composite_risk(self, amount: float, velocity: float, hourly_volume: float) -> Tuple[float, Dict[str, float]]:
        """Oblicza zrównoważony wskaźnik ryzyka w oparciu o skalibrowane wagi."""
        ratio_vel = min(velocity / max(1.0, float(self.max_velocity)), 2.0)
        ratio_vol = min(hourly_volume / max(1.0, float(self.max_hourly_volume)), 2.0)
        ratio_amt = min(amount / max(1.0, float(self.max_single_tx)), 2.0)

        risk_score = (
            self.w_velocity * ratio_vel
            + self.w_volume * ratio_vol
            + self.w_single_tx * ratio_amt
        )

        factors = {
            "velocity_ratio": round(ratio_vel, 3),
            "volume_ratio": round(ratio_vol, 3),
            "amount_ratio": round(ratio_amt, 3),
            "composite_score": round(risk_score, 3),
        }
        return risk_score, factors

    def evaluate_transaction(self, tx: FinancialTransaction) -> CircuitBreakerDecision:
        """Ocenia transakcję pod kątem ryzyka runaway w ramach podwójnych widełek bezpieczeństwa."""
        now = time.time()
        self._prune_history(now)

        # 1. Sprawdzenie stanu TRIPPED i okresu schłodzenia
        if self.state == CircuitBreakerState.TRIPPED:
            if self.trip_timestamp and (now - self.trip_timestamp) >= self.cooling_off_seconds:
                logger.info("Upłynął okres schłodzenia (Cooling-off). Powrót bezpiecznika do stanu THROTTLED.")
                self.state = CircuitBreakerState.THROTTLED
                self.trip_timestamp = None
            else:
                remaining = self.cooling_off_seconds - (now - self.trip_timestamp if self.trip_timestamp else 0)
                trip_violations = [self.trip_violation_code] if self.trip_violation_code else []
                trip_violations.append(f"CircuitBreakerTripped: {self.trip_reason}")
                return CircuitBreakerDecision(
                    allowed=False,
                    current_state=self.state,
                    reason=f"Bezpiecznik jest wyzwolony (TRIPPED). Pozostało {remaining:.1f}s okresu schłodzenia.",
                    lower_threshold=self.lower_threshold,
                    upper_threshold=self.upper_threshold,
                    violations=trip_violations,
                )

        # 2. Stan HALTED (wymaga odblokowania ludzkiego)
        if self.state == CircuitBreakerState.HALTED:
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason="System finansowy jest w stanie HALTED. Wymagana autoryzacja człowieka (HITL).",
                lower_threshold=self.lower_threshold,
                upper_threshold=self.upper_threshold,
                violations=["CircuitBreakerHalted: Critical runaway detected."],
            )

        # 3. Twarde limity graniczne (Hard Ceilings)
        if tx.amount > self.max_single_tx:
            self._trip(
                f"Przekroczenie twardego limitu pojedynczej transakcji: {tx.amount:.2f} > {self.max_single_tx:.2f}",
                violation_code="ExceededSingleTxLimit",
            )
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason=f"Pojedyncza kwota {tx.amount:.2f} przekracza dopuszczalny limit {self.max_single_tx:.2f}.",
                lower_threshold=self.lower_threshold,
                upper_threshold=self.upper_threshold,
                violations=["ExceededSingleTxLimit"],
            )

        # 4. Sprawdzenie dynamiki (Velocity per minute)
        minute_txs = [amount for ts, amount in self.tx_history if (now - ts) <= 60.0]
        velocity = len(minute_txs) + 1

        if velocity > self.max_velocity:
            self._trip(
                f"Wykryto anomalię pętli transakcyjnej (Velocity: {velocity} tx/min > {self.max_velocity})",
                violation_code="VelocityRunawayAnomaly",
            )
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason=f"Zbyt duża częstotliwość operacji ({velocity} tx/min > {self.max_velocity}). Aktywacja bezpiecznika TRIPPED.",
                velocity_tx_per_min=velocity,
                lower_threshold=self.lower_threshold,
                upper_threshold=self.upper_threshold,
                violations=["VelocityRunawayAnomaly"],
            )

        # 5. Skumulowany wolumen godzinowy
        hourly_volume = sum(amount for _, amount in self.tx_history) + tx.amount
        if hourly_volume > self.max_hourly_volume:
            self.state = CircuitBreakerState.HALTED
            self.trip_reason = f"Przekroczono maksymalny godzinowy limit wolumenu: {hourly_volume:.2f} > {self.max_hourly_volume:.2f}"
            logger.critical("🚨 [FINANCIAL HALT]: %s", self.trip_reason)
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason=self.trip_reason,
                hourly_volume_sum=hourly_volume,
                lower_threshold=self.lower_threshold,
                upper_threshold=self.upper_threshold,
                violations=["HourlyVolumeExceededHalt"],
            )

        # 6. Ewaluacja w ramach Podwójnych Widełek Bezpieczeństwa (Dual Corridors)
        risk_score, factors = self.calculate_composite_risk(tx.amount, velocity, hourly_volume)

        # 6a. Górny Próg (Upper Threshold Breach) -> Wyzwolenie TRIPPED przed uderzeniem w twardy sufit
        if risk_score >= self.upper_threshold:
            self._trip(
                f"Złożony wskaźnik ryzyka przekroczył górne widełki bezpieczeństwa "
                f"({risk_score:.2f} >= {self.upper_threshold:.2f} [vel_ratio={factors['velocity_ratio']}, vol_ratio={factors['volume_ratio']}])",
                violation_code="UpperThresholdRiskTrip",
            )
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason=f"Wskaźnik ryzyka ({risk_score:.2f}) przekroczył górny próg ({self.upper_threshold:.2f}). Aktywacja schłodzenia TRIPPED.",
                velocity_tx_per_min=velocity,
                hourly_volume_sum=hourly_volume,
                composite_risk_score=risk_score,
                lower_threshold=self.lower_threshold,
                upper_threshold=self.upper_threshold,
                violations=["UpperThresholdRiskTrip"],
            )

        # 6b. Dolny Próg (Lower Threshold Breach / Early Warning Corridor) -> Proaktywne Dławienie (THROTTLED)
        applied_delay = 0.0
        early_warning = False

        if risk_score >= self.lower_threshold:
            self.state = CircuitBreakerState.THROTTLED
            early_warning = True
            # Płynne skalowanie opóźnienia dławiącego: 50 ms do 300 ms w zależności od pozycji w widełkach
            corridor_span = max(0.01, self.upper_threshold - self.lower_threshold)
            normalized_corridor_pos = (risk_score - self.lower_threshold) / corridor_span
            applied_delay = round(50.0 + (normalized_corridor_pos * 250.0), 1)
            logger.warning(
                "⚠️ [EARLY WARNING / THROTTLED]: Ryzyko weszło w dolne widełki (%0.2f >= %0.2f). "
                "Wdrożono proaktywne opóźnienie: %0.1f ms",
                risk_score, self.lower_threshold, applied_delay
            )
        else:
            self.state = CircuitBreakerState.NORMAL

        # Rejestracja transakcji w historii
        self.tx_history.append((now, tx.amount))

        return CircuitBreakerDecision(
            allowed=True,
            current_state=self.state,
            reason="Transakcja zweryfikowana pomyślnie w bezpiecznym korytarzu finansowym.",
            applied_throttle_delay_ms=applied_delay,
            velocity_tx_per_min=velocity,
            hourly_volume_sum=hourly_volume,
            composite_risk_score=risk_score,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            early_warning_active=early_warning,
        )

    def _trip(self, reason: str, violation_code: str = "CircuitBreakerTripped") -> None:
        self.state = CircuitBreakerState.TRIPPED
        self.trip_timestamp = time.time()
        self.trip_reason = reason
        self.trip_violation_code = violation_code
        logger.warning("⚡ [FINANCIAL CIRCUIT BREAKER TRIPPED]: %s", reason)

    def _prune_history(self, now: float) -> None:
        # Usuń wpisy starsze niż 1 godzina (3600 sekund)
        while self.tx_history and (now - self.tx_history[0][0]) > 3600.0:
            self.tx_history.popleft()

    def reset_halt(self, admin_token: str) -> bool:
        """Resetuje blokadę HALTED przez uprawnionego administratora."""
        if admin_token == "NETHICAL_FINANCIAL_SUPERVISOR_KEY":
            self.state = CircuitBreakerState.NORMAL
            self.trip_timestamp = None
            self.trip_reason = None
            self.trip_violation_code = None
            self.tx_history.clear()
            logger.info("Blokada finansowa HALTED zresetowana pomyślnie przez administratora.")
            return True
        return False
