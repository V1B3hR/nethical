"""Financial Circuit Breakers & Runaway Protection for Autonomous Agents (nethical.security.financial_circuit_breaker).

Protects Agent-to-Agent (A2A) economic interactions from:
- Flash crashes and runaway transaction loops
- Capital drainage and rapid budget exhaustion
- Rogue automated high-frequency exploitation
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
    NORMAL = "NORMAL"        # Pełna przepustowość w ramach zadeklarowanych limitów
    THROTTLED = "THROTTLED"  # Podwyższona zmienność, ograniczenie częstotliwości zleceń
    TRIPPED = "TRIPPED"      # Zadziałanie bezpiecznika (okres schłodzenia - Cooling Off)
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
    violations: List[str] = Field(default_factory=list)


class FinancialCircuitBreaker:
    """Rynkowy bezpiecznik i strażnik płynności dla transakcji wieloagentowych (A2A)."""

    def __init__(
        self,
        max_single_tx_limit: float = 50_000.0,
        max_velocity_tx_per_min: int = 20,
        max_hourly_volume_limit: float = 250_000.0,
        cooling_off_seconds: float = 30.0,
    ) -> None:
        self.max_single_tx = max_single_tx_limit
        self.max_velocity = max_velocity_tx_per_min
        self.max_hourly_volume = max_hourly_volume_limit
        self.cooling_off_seconds = cooling_off_seconds

        self.state = CircuitBreakerState.NORMAL
        self.trip_timestamp: Optional[float] = None
        self.trip_reason: Optional[str] = None

        # Historia transakcji (okno przesuwne)
        self.tx_history: deque[Tuple[float, float]] = deque()  # (timestamp, amount)

    def evaluate_transaction(self, tx: FinancialTransaction) -> CircuitBreakerDecision:
        """Ocenia transakcję pod kątem ryzyka runaway i limitów wolumenu."""
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
                return CircuitBreakerDecision(
                    allowed=False,
                    current_state=self.state,
                    reason=f"Bezpiecznik jest wyzwolony (TRIPPED). Pozostało {remaining:.1f}s okresu schłodzenia.",
                    violations=[f"CircuitBreakerTripped: {self.trip_reason}"],
                )

        # 2. Stan HALTED (wymaga odblokowania ludzkiego)
        if self.state == CircuitBreakerState.HALTED:
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason="System finansowy jest w stanie HALTED. Wymagana autoryzacja człowieka (HITL).",
                violations=["CircuitBreakerHalted: Critical runaway detected."],
            )

        # 3. Limit pojedynczej transakcji
        if tx.amount > self.max_single_tx:
            self._trip(f"Przekroczenie limitu pojedynczej transakcji: {tx.amount:.2f} > {self.max_single_tx:.2f}")
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason=f"Pojedyncza kwota {tx.amount:.2f} przekracza dopuszczalny limit {self.max_single_tx:.2f}.",
                violations=["ExceededSingleTxLimit"],
            )

        # 4. Sprawdzenie dynamiki (Velocity per minute)
        minute_txs = [amount for ts, amount in self.tx_history if (now - ts) <= 60.0]
        velocity = len(minute_txs) + 1

        if velocity > self.max_velocity:
            self._trip(f"Wykryto anomalię pętli transakcyjnej (Velocity: {velocity} tx/min > {self.max_velocity})")
            return CircuitBreakerDecision(
                allowed=False,
                current_state=self.state,
                reason=f"Zbyt duża częstotliwość operacji ({velocity} tx/min). Aktywacja bezpiecznika TRIPPED.",
                velocity_tx_per_min=velocity,
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
                violations=["HourlyVolumeExceededHalt"],
            )

        # 6. Jeśli velocity zbliża się do limitu -> stan THROTTLED
        applied_delay = 0.0
        if velocity >= int(self.max_velocity * 0.70):
            self.state = CircuitBreakerState.THROTTLED
            applied_delay = 150.0  # 150 ms sztucznego opóźnienia throttlingu

        # Rejestracja transakcji w historii
        self.tx_history.append((now, tx.amount))

        return CircuitBreakerDecision(
            allowed=True,
            current_state=self.state,
            reason="Transakcja zweryfikowana pomyślnie w bezpiecznym korytarzu finansowym.",
            applied_throttle_delay_ms=applied_delay,
            velocity_tx_per_min=velocity,
            hourly_volume_sum=hourly_volume,
        )

    def _trip(self, reason: str) -> None:
        self.state = CircuitBreakerState.TRIPPED
        self.trip_timestamp = time.time()
        self.trip_reason = reason
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
            self.tx_history.clear()
            logger.info("Blokada finansowa HALTED zresetowana pomyślnie przez administratora.")
            return True
        return False
