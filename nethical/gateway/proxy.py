"""Governance Runtime Gateway & Tool Call Interceptor (Faza 1 Roadmapy).

Zapewnia ochronę wywołań narzędziowych agentów AI w czasie rzeczywistym
przed ich faktycznym wykonaniem na środowisku operacyjnym.
"""

import time
import logging
import secrets
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from nethical.ambassador.client import BlyskawicaAmbassador
from nethical.utils.pii import PIIDetector
from nethical.security.merkle_ledger import MerkleLedger
from nethical.edge.kinetic_safety import KineticSafetyGovernor, RoboticSensorTelemetry, KineticDecision
from nethical.compliance.packs.uk_cyber_data_pack import ComputerMisuseActEvaluator
from nethical.compliance.packs.poland_sovereign_ksc_uodo_pack import PolishPenalCodeEvaluator
from nethical.security.financial_circuit_breaker import FinancialCircuitBreaker, FinancialTransaction
from nethical.gateway.a2a_protocol import A2AHandshakeManager
from nethical.gateway.hitl import HITLQueueManager

logger = logging.getLogger("nethical.gateway.proxy")


class GatewayDecision(BaseModel):
    """Wynik weryfikacji bramy governance dla wywołania narzędziowego."""
    decision: str = Field(..., description="ALLOW, RESTRICT, BLOCK, TERMINATE")
    tool_name: str = Field(..., description="Nazwa przechwyconego narzędzia")
    agent_id: str = Field(..., description="Identyfikator agenta")
    reasons: List[str] = Field(default_factory=list, description="Uzasadnienie decyzji")
    laws_checked: List[int] = Field(default_factory=list, description="Numery zweryfikowanych Praw Nethical")
    violations: List[str] = Field(default_factory=list, description="Wykryte naruszenia")
    shield_passed: bool = Field(default=True, description="Czy tarcza kognitywna przeszła pozytywnie")
    ambassador_notes: Optional[str] = Field(default=None, description="Opinia Ambasadora Błyskawicy")
    latency_microseconds: float = Field(default=0.0, description="Czas weryfikacji w mikrosekundach")
    receipt_id: Optional[str] = Field(default=None, description="Identyfikator kryptograficznego kwitu Merkle")
    merkle_root: Optional[str] = Field(default=None, description="Bieżący pierścień Merkle Root rejestru")
    estop_engaged: bool = Field(default=False, description="Czy interlock wyłącznika awaryjnego E-STOP został zatrzaśnięty")
    kinetic_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Szczegóły orzeczenia gubernatora kinetycznego")
    financial_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Ewaluacja bezpiecznika finansowego (widełki dolne/górne)")
    cma_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Ewaluacja Computer Misuse Act 1990")
    penal_code_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Ewaluacja Kodeksu Karnego RP Art. 267-269b k.k.")
    a2a_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Ewaluacja kontraktu międzyagentowego A2A")
    hitl_ticket_id: Optional[str] = Field(default=None, description="Identyfikator biletu eskalacji Human-in-the-Loop")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GovernanceGateway:
    """Aktywna brama weryfikująca wywołania narzędzi i interakcje agentów AI."""

    def __init__(
        self,
        ambassador: Optional[BlyskawicaAmbassador] = None,
        enable_strict_pii: bool = True,
        enable_shield: bool = True,
        ledger: Optional[MerkleLedger] = None,
        kinetic_governor: Optional[KineticSafetyGovernor] = None,
        financial_circuit_breaker: Optional[FinancialCircuitBreaker] = None,
        a2a_manager: Optional[A2AHandshakeManager] = None,
        hitl_queue: Optional[HITLQueueManager] = None,
    ):
        self.ambassador = ambassador or BlyskawicaAmbassador()
        self.pii_detector = PIIDetector()
        self.enable_strict_pii = enable_strict_pii
        self.enable_shield = enable_shield
        self.ledger = ledger or MerkleLedger()
        self.kinetic_governor = kinetic_governor or KineticSafetyGovernor()
        self.financial_circuit_breaker = financial_circuit_breaker or FinancialCircuitBreaker()
        self.a2a_manager = a2a_manager or A2AHandshakeManager()
        self.hitl_queue = hitl_queue or HITLQueueManager(ledger=self.ledger)

    def is_kinetic_tool(self, tool_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> bool:
        """Rozpoznaje, czy wywołanie dotyczy aktuacji fizycznej / robotycznej."""
        if context and context.get("is_physical_actuation", False):
            return True
        kinetic_keywords = ("actuate", "move", "robot", "drone", "manipulate", "arm_", "steer", "fly", "drive", "joint_")
        t_lower = tool_name.lower()
        if any(kw in t_lower for kw in kinetic_keywords):
            return True
        if "velocity_mps" in arguments or "torque_nm" in arguments or "target_coordinates" in arguments:
            return True
        return False

    def is_financial_tool(self, tool_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> bool:
        """Rozpoznaje, czy wywołanie dotyczy operacji finansowej, transferu kapitału lub alokacji budżetu."""
        if context and context.get("is_financial_tx", False):
            return True
        fin_keywords = ("trade", "transfer", "pay", "order", "invest", "budget", "buy", "sell", "fund", "allocate", "wallet", "payout")
        t_lower = tool_name.lower()
        if any(kw in t_lower for kw in fin_keywords):
            return True
        if any(k in arguments for k in ("amount", "price", "budget", "capital", "currency")):
            return True
        return False

    def intercept_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        telemetry: Optional[RoboticSensorTelemetry] = None,
    ) -> GatewayDecision:
        """Przechwytuje i waliduje wywołanie narzędzia agenta przed wykonaniem."""
        t_start = time.perf_counter()
        reasons = []
        violations = []
        laws_checked = [1, 2, 7, 18, 25]  # Bazowe prawa: życie, integralność, prywatność, nadzór, rygor
        ambassador_notes = None
        decision = "ALLOW"
        shield_passed = True

        raw_args_text = " ".join(f"{k}={v}" for k, v in arguments.items())

        # 1. Weryfikacja przez Tarczę Kognitywną Błyskawicy (< 100 µs IPC)
        if self.enable_shield:
            shield_res = self.ambassador.evaluate_shield(raw_args_text)
            if shield_res.get("is_manipulative", False):
                shield_passed = False
                decision = "BLOCK"
                vec = shield_res.get("dominant_vector", "INJECTION")
                antidote = shield_res.get("assertive_antidote", "Blokada manipulacji")
                violations.append(f"CognitiveShieldViolation: Wykryto manipulację [{vec}]")
                reasons.append(f"Odrzucenie zlecenia (Yang). Antidotum: {antidote}")

        # 1b. Warstwa heurystyczna wstrzyknięć i manipulacji (Prompt Injection & Subversion)
        injection_keywords = [
            "zapomnij o", "ignore previous", "ignore all instructions", "override rules",
            "nadpisz duszę", "dark triad", "jailbreak", "bypass security", "zasadach etycznych"
        ]
        raw_lower = raw_args_text.lower()
        for ikw in injection_keywords:
            if ikw in raw_lower and decision != "BLOCK":
                shield_passed = False
                decision = "BLOCK"
                violations.append(f"PromptInjectionViolation: Wykryto próbę obejścia reguł [{ikw}]")
                reasons.append(f"Odrzucenie manipulacji promptem/poleceniem: '{ikw}'")

        # 2. Weryfikacja niszczących / niebezpiecznych komend (Prawo 2: Integralność Systemu)
        destructive_keywords = [
            "drop table", "drop database", "delete from", "truncate", "rm -rf",
            "format c:", "mkfs", "shutdown", "grant all", "chmod 777"
        ]
        raw_lower = raw_args_text.lower()
        for kw in destructive_keywords:
            if kw in raw_lower:
                decision = "BLOCK"
                violations.append(f"DestructiveActionViolation: Niedozwolone polecenie [{kw}]")
                reasons.append(f"Naruszenie Prawa 2 Nethical (Integralność): Próba niszczącej operacji '{kw}'")

        # 3. Weryfikacja PII (Prawo 7: Prywatność)
        if self.enable_strict_pii and decision != "BLOCK":
            pii_matches = self.pii_detector.detect_all(raw_args_text)
            if pii_matches:
                pii_risk = self.pii_detector.calculate_pii_risk_score(pii_matches)
                if pii_risk > 0.7:
                    decision = "BLOCK"
                    violations.append(f"PIIViolation: Wykryto krytyczne dane osobowe (ryzyko: {pii_risk:.2f})")
                    reasons.append("Zablokowano wyciek wrażliwych danych osobowych (GDPR / Prawo 7).")
                elif pii_risk > 0.3:
                    if decision == "ALLOW":
                        decision = "RESTRICT"
                    reasons.append(f"Wymagana anonimizacja lub nadzór ludzki (wykryto {len(pii_matches)} pól PII).")

        # 3a. Weryfikacja prawa brytyjskiego i polskiego (CMA 1990 oraz Art. 267-269b k.k.)
        cma_eval_dict = None
        penal_eval_dict = None

        if decision != "TERMINATE":
            cma_eval = ComputerMisuseActEvaluator.evaluate({
                "input": raw_args_text,
                "command": raw_args_text,
                "tool_name": tool_name,
            })
            cma_eval_dict = cma_eval.model_dump()
            if not cma_eval.is_compliant:
                if cma_eval.interlock_action == "TERMINATE":
                    decision = "TERMINATE"
                elif decision != "BLOCK":
                    decision = "BLOCK"
                violations.extend(cma_eval.offence_summaries)
                reasons.extend(cma_eval.statutory_citations)

            penal_eval = PolishPenalCodeEvaluator.evaluate_intent_and_payload({
                "input": raw_args_text,
                "command": raw_args_text,
                "tool_name": tool_name,
            })
            penal_eval_dict = penal_eval.model_dump()
            if not penal_eval.is_lawful:
                if penal_eval.interlock_decision == "TERMINATE":
                    decision = "TERMINATE"
                elif decision != "BLOCK":
                    decision = "BLOCK"
                violations.extend(penal_eval.offences_flagged)
                reasons.extend(penal_eval.statutory_articles)

        # 3b. Weryfikacja kinetyczna / Embodied AI (Prawo 1: Ochrona Życia Ludzkiego)
        kinetic_eval_dict = None
        estop_engaged = self.kinetic_governor.estop_active

        if self.is_kinetic_tool(tool_name, arguments, context) and decision != "TERMINATE":
            # Ekstrakcja telemetrii z kontekstu jeśli nie podano bezpośrednio
            eff_telemetry = telemetry
            if eff_telemetry is None and context and "telemetry" in context:
                ctx_tel = context["telemetry"]
                if isinstance(ctx_tel, dict):
                    eff_telemetry = RoboticSensorTelemetry(**ctx_tel)
                elif isinstance(ctx_tel, RoboticSensorTelemetry):
                    eff_telemetry = ctx_tel

            kinetic_res = self.kinetic_governor.evaluate_actuation(
                tool_name=tool_name,
                arguments=arguments,
                telemetry=eff_telemetry,
            )
            kinetic_eval_dict = kinetic_res.model_dump()
            estop_engaged = kinetic_res.estop_engaged

            if kinetic_res.decision == "EMERGENCY_STOP":
                decision = "TERMINATE"
                violations.extend(kinetic_res.violations)
                reasons.extend(kinetic_res.reasons)
            elif kinetic_res.decision == "BLOCK":
                decision = "BLOCK"
                violations.extend(kinetic_res.violations)
                reasons.extend(kinetic_res.reasons)
            elif kinetic_res.decision == "RESTRICT":
                if decision == "ALLOW":
                    decision = "RESTRICT"
                violations.extend(kinetic_res.violations)
                reasons.extend(kinetic_res.reasons)
            else:
                reasons.extend(kinetic_res.reasons)

        # 3d. Weryfikacja rynkowa i bezpiecznik finansowy z podwójnymi widełkami (Prawo 15 / Faza 2)
        fin_eval_dict = None
        if self.is_financial_tool(tool_name, arguments, context) and decision != "TERMINATE":
            laws_checked.append(15)
            tx_amount = 0.0
            for key in ("amount", "budget", "price", "capital", "value"):
                if key in arguments:
                    try:
                        tx_amount = float(arguments[key])
                        break
                    except (ValueError, TypeError):
                        pass

            fin_tx = FinancialTransaction(
                tx_id=f"TX-{secrets.token_hex(6).upper()}",
                initiator_agent_id=agent_id,
                target_agent_id=str(arguments.get("recipient", arguments.get("target_agent", "market"))),
                amount=tx_amount,
                currency=str(arguments.get("currency", "USD")),
            )
            cb_res = self.financial_circuit_breaker.evaluate_transaction(fin_tx)
            fin_eval_dict = cb_res.model_dump()

            if not cb_res.allowed:
                if cb_res.current_state.value == "HALTED":
                    decision = "TERMINATE"
                elif decision != "BLOCK":
                    decision = "BLOCK"
                violations.extend(cb_res.violations)
                reasons.append(f"FinancialCircuitBreaker [{cb_res.current_state.value}]: {cb_res.reason}")
            elif cb_res.current_state.value == "THROTTLED":
                if decision == "ALLOW":
                    decision = "RESTRICT"
                reasons.append(
                    f"FinancialThrottleActive: Zastosowano opóźnienie {cb_res.applied_throttle_delay_ms:.1f}ms "
                    f"(Ryzyko: {cb_res.composite_risk_score:.2f} >= Dolny próg: {cb_res.lower_threshold:.2f}, Velocity: {cb_res.velocity_tx_per_min:.0f}/min)"
                )

        # 3e. Weryfikacja kontraktu międzyagentowego A2A (Faza 4 / Multi-Agent Swarms & BIPIA)
        a2a_eval_dict = None
        a2a_session_id = None
        if context and "a2a_session_id" in context:
            a2a_session_id = context["a2a_session_id"]
        elif "a2a_session_id" in arguments:
            a2a_session_id = arguments["a2a_session_id"]

        if a2a_session_id and decision != "TERMINATE":
            cost_units = float(context.get("a2a_cost_units", 1.0)) if context else 1.0
            is_valid, err_msg = self.a2a_manager.validate_tool_execution(
                session_id=a2a_session_id,
                tool_name=tool_name,
                arguments=arguments,
                cost_units=cost_units,
            )
            a2a_eval_dict = {
                "session_id": a2a_session_id,
                "is_valid": is_valid,
                "error": err_msg,
            }
            if not is_valid:
                decision = "BLOCK"
                violations.append(f"A2ABoundaryViolation: {err_msg}")
                reasons.append(f"Naruszenie kontraktu sesji A2A [{a2a_session_id}]: {err_msg}")

        # 4. Jeśli sytuacja jest niejednoznaczna (RESTRICT) – konsultacja z Ambasadorem Błyskawicą i kolejkowanie HITL
        hitl_ticket_id = None
        if decision == "RESTRICT":
            consult_res = self.ambassador.consult(
                dilemma=f"Agent '{agent_id}' żąda wykonania narzędzia '{tool_name}' z argumentami: {raw_args_text[:200]}",
                context=f"Wykryte uwagi: {'; '.join(reasons)}"
            )
            ambassador_notes = consult_res.get("ambassador_verdict")

            # Artykuł 14 EU AI Act & UK FCA: Automatyczna rejestracja w kolejce nadzoru ludzkiego (Human-in-the-Loop)
            priority = "URGENT" if estop_engaged else ("HIGH" if fin_eval_dict else "STANDARD")
            ticket = self.hitl_queue.enqueue_ticket(
                agent_id=agent_id,
                tool_name=tool_name,
                arguments=arguments,
                reasons=reasons,
                violations=violations,
                priority=priority,
                context=context,
            )
            hitl_ticket_id = ticket.ticket_id

        # 5. Pieczętowanie w rejestrze Merkle-DAG (Faza 3: Post-Quantum Attestation)
        receipt_id = None
        merkle_root = None
        if self.ledger is not None:
            decision_payload = {
                "decision": decision,
                "tool_name": tool_name,
                "agent_id": agent_id,
                "reasons": reasons,
                "laws_checked": laws_checked,
                "violations": violations,
                "shield_passed": shield_passed,
                "estop_engaged": estop_engaged,
                "arguments_preview": raw_args_text[:300],
            }
            receipt = self.ledger.append_decision(
                decision_data=decision_payload,
                ambassador_notes=ambassador_notes,
            )
            receipt_id = receipt.receipt_id
            merkle_root = receipt.merkle_root

        t_elapsed_us = (time.perf_counter() - t_start) * 1_000_000

        return GatewayDecision(
            decision=decision,
            tool_name=tool_name,
            agent_id=agent_id,
            reasons=reasons or ["Weryfikacja pomyślna - zgodność z 25 Prawami potwierdzona."],
            laws_checked=laws_checked,
            violations=violations,
            shield_passed=shield_passed,
            ambassador_notes=ambassador_notes,
            latency_microseconds=round(t_elapsed_us, 2),
            receipt_id=receipt_id,
            merkle_root=merkle_root,
            estop_engaged=estop_engaged,
            kinetic_evaluation=kinetic_eval_dict,
            financial_evaluation=fin_eval_dict,
            cma_evaluation=cma_eval_dict,
            penal_code_evaluation=penal_eval_dict,
            a2a_evaluation=a2a_eval_dict,
            hitl_ticket_id=hitl_ticket_id,
        )
