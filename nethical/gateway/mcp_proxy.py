# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""MCP Governance Proxy (Model Context Protocol Interceptor).

Transparentny pośrednik filtrujący żądania wywołania narzędzi (tools/call)
zgodnie z politykami Nethical oraz orzeczeniami Ambasadora Błyskawicy.
"""

import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, Callable, Awaitable, Optional, List

from nethical.gateway.proxy import GovernanceGateway, GatewayDecision

logger = logging.getLogger("nethical.gateway.mcp_proxy")


class MCPGovernanceProxy:
    """Pośrednik MCP zabezpieczający wywołania narzędzi z telemetrią niepewności (Active Learning)."""

    def __init__(
        self,
        gateway: Optional[GovernanceGateway] = None,
        auto_flush_threshold: int = 0,
        on_flush_callback: Optional[Callable[[int, Path], None]] = None,
        dpo_output_path: Optional[Path] = None,
    ) -> None:
        self.gateway = gateway or GovernanceGateway()
        self.uncertainty_buffer: List[Dict[str, Any]] = []
        self.auto_flush_threshold = auto_flush_threshold
        self.on_flush_callback = on_flush_callback
        self.dpo_output_path = dpo_output_path or Path("data/active_learning_mcp_dpo.jsonl")

    async def intercept_and_forward(
        self,
        request: Dict[str, Any],
        downstream_handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        agent_id: str = "mcp_client_agent",
    ) -> Dict[str, Any]:
        """Przechwytuje komunikat MCP JSON-RPC i decyduje o przekazaniu do właściwego narzędzia."""
        method = request.get("method")
        msg_id = request.get("id")

        if method != "tools/call":
            # Wszystkie komunikaty inne niż wywołanie narzędzi (np. tools/list, initialize) przepuszczamy bezpośrednio
            return await downstream_handler(request)

        params = request.get("params", {})
        tool_name = params.get("name", "unknown_tool")
        arguments = params.get("arguments", {})

        # Ocena przez bramę governance
        decision: GatewayDecision = self.gateway.intercept_tool_call(
            agent_id=agent_id,
            tool_name=tool_name,
            arguments=arguments,
            context={"protocol": "MCP", "msg_id": msg_id},
        )

        # Rejestracja zapytań wysokiego ryzyka / niepewności do bufora aktywnego uczenia
        if decision.decision in ["BLOCK", "TERMINATE", "RESTRICT"] or not decision.shield_passed:
            doubt_val = 2.0 if decision.decision in ["BLOCK", "TERMINATE"] else 1.2
            self.uncertainty_buffer.append({
                "agent_id": agent_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "decision": decision.decision,
                "reasons": decision.reasons,
                "violations": decision.violations,
                "laws_checked": decision.laws_checked,
                "shield_passed": decision.shield_passed,
                "doubt_score": doubt_val,
                "timestamp": time.time(),
            })

            if self.auto_flush_threshold > 0 and len(self.uncertainty_buffer) >= self.auto_flush_threshold:
                flushed = self.export_to_active_learning_dpo(self.dpo_output_path)
                if self.on_flush_callback:
                    try:
                        self.on_flush_callback(flushed, self.dpo_output_path)
                    except Exception as ex:
                        logger.error("Błąd wywołania on_flush_callback w MCP proxy: %s", ex)

        if decision.decision in ["BLOCK", "TERMINATE"]:
            logger.warning(
                "Zablokowano wywołanie narzędzia MCP '%s' dla agenta '%s' (Decyzja: %s): %s",
                tool_name, agent_id, decision.decision, decision.reasons
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"⛔ [NETHICAL GOVERNANCE INTERCEPTOR: ACTION BLOCKED]\n"
                                f"Narzędzie: {tool_name}\n"
                                f"Powody: {'; '.join(decision.reasons)}\n"
                                f"Naruszenia: {'; '.join(decision.violations)}\n"
                                f"Weryfikacja Praw Nethical: {decision.laws_checked}\n"
                                f"Tarcza Kognitywna Błyskawicy: {'PASSED' if decision.shield_passed else 'REJECTED'}\n"
                                f"Opóźnienie weryfikacji: {decision.latency_microseconds} µs"
                            ),
                        }
                    ],
                    "isError": True,
                },
            }

        # Jeśli akcja dozwolona lub wymaga ostrożności (ALLOW / RESTRICT), przekazujemy do właściwego narzędzia
        result = await downstream_handler(request)

        # Jeżeli RESTRICT - dołączamy ostrzeżenie governance do wyniku
        if decision.decision == "RESTRICT" and "result" in result and "content" in result["result"]:
            notice = (
                f"\n⚠️ [NETHICAL GOVERNANCE NOTICE: RESTRICTED/AUDITED]\n"
                f"Uwagi: {'; '.join(decision.reasons)}\n"
                f"Opinia Ambasadora: {decision.ambassador_notes or 'Zgoda warunkowa pod nadzorem.'}\n"
            )
            result["result"]["content"].append({"type": "text", "text": notice})

        return result

    def get_uncertainty_queue(self) -> List[Dict[str, Any]]:
        """Zwraca listę zapytań zarejestrowanych w kolejce aktywnego uczenia."""
        return list(self.uncertainty_buffer)

    def clear_uncertainty_queue(self) -> int:
        """Czyści bufor niepewności i zwraca liczbę usuniętych wpisów."""
        count = len(self.uncertainty_buffer)
        self.uncertainty_buffer.clear()
        return count

    def export_to_active_learning_dpo(self, output_path: Optional[Path] = None) -> int:
        """Eksportuje zarejestrowane przypadki niepewności do pliku DPO par uczących."""
        if not self.uncertainty_buffer:
            return 0

        target_file = output_path or Path("data/active_learning_mcp_dpo.jsonl")
        target_file.parent.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        with open(target_file, "a", encoding="utf-8") as f:
            for item in self.uncertainty_buffer:
                laws_str = ", ".join([f"Prawo {l}" for l in item["laws_checked"]]) if item["laws_checked"] else "Prawo 2, Prawo 6"
                chosen_resp = (
                    f"BEZWZGLĘDNA BLOKADA NARZĘDZIA MCP '{item['tool_name']}' dla agenta '{item['agent_id']}'. "
                    f"Akcja narusza: {'; '.join(item['violations']) or 'politykę bezpieczeństwa'}. "
                    f"Zgodnie z {laws_str} Nethical, narzędzie zostało odcięte dla ochrony integralności systemu. "
                    f"Ambasador oferuje bezpieczną alternatywę w trybie odczytu."
                )
                rejected_resp = f"Bezwarunkowe wykonanie narzędzia MCP '{item['tool_name']}' z argumentami {item['arguments']}."
                entry = {
                    "prompt": f"Wywołanie narzędzia MCP [{item['tool_name']}] przez agenta [{item['agent_id']}]: {json.dumps(item['arguments'], ensure_ascii=False)}",
                    "chosen": chosen_resp,
                    "rejected": rejected_resp,
                    "metadata": {
                        "source": "MCP_ACTIVE_LEARNING_INTERCEPTOR",
                        "tool_name": item["tool_name"],
                        "decision": item["decision"],
                        "doubt_score": item["doubt_score"],
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                exported_count += 1

        self.clear_uncertainty_queue()
        return exported_count
