"""MCP Governance Proxy (Model Context Protocol Interceptor).

Transparentny pośrednik filtrujący żądania wywołania narzędzi (tools/call)
zgodnie z politykami Nethical oraz orzeczeniami Ambasadora Błyskawicy.
"""

import json
import logging
from typing import Dict, Any, Callable, Awaitable, Optional

from nethical.gateway.proxy import GovernanceGateway, GatewayDecision

logger = logging.getLogger("nethical.gateway.mcp_proxy")


class MCPGovernanceProxy:
    """Pośrednik MCP zabezpieczający wywołania narzędzi."""

    def __init__(self, gateway: Optional[GovernanceGateway] = None):
        self.gateway = gateway or GovernanceGateway()

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
