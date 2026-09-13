"""OpenAI / Anthropic / Ollama Compatible Governance Proxy & Stream Interceptor.

Provides a transparent drop-in reverse proxy for any LLM client (OpenAI SDK, LangChain,
CrewAI, AutoGen, LlamaIndex, Claude, Ollama, etc.) with:
- Zero code modifications on client side (simply point base_url="http://localhost:8000/v1")
- In-flight dynamic PII/ePHI masking & detokenization via ReversibleTokenVault (AES-256-GCM)
- Pre-execution prompt injection, jailbreak, and policy evaluation via GovernanceGateway
- Real-time token-by-token sliding-window streaming interceptor (SSE stream=True)
- Immutable audit trail with post-quantum signed receipts in MerkleLedger (NIST FIPS 204)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field

from nethical.gateway.proxy import GovernanceGateway, GatewayDecision
from nethical.security.merkle_ledger import MerkleLedger
from nethical.security.token_vault import ReversibleTokenVault, TokenizeResponse, DetokenizeResponse

logger = logging.getLogger("nethical.gateway.openai_proxy")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, assistant, tool")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Message content")
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o", description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(default=0.7)
    top_p: Optional[float] = Field(default=1.0)
    n: Optional[int] = Field(default=1)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = Field(default="nethical-client-user")


class OpenAIGovernanceProxy:
    """Transparent Governance Reverse Proxy for OpenAI / Anthropic / Ollama APIs."""

    def __init__(
        self,
        gateway: Optional[GovernanceGateway] = None,
        token_vault: Optional[ReversibleTokenVault] = None,
        ledger: Optional[MerkleLedger] = None,
        default_upstream_url: Optional[str] = None,
        enable_in_flight_tokenization: bool = True,
        mock_mode: bool = False,
    ) -> None:
        self.gateway = gateway or GovernanceGateway()
        self.token_vault = token_vault or ReversibleTokenVault()
        self.ledger = ledger or self.gateway.ledger or MerkleLedger()
        self.default_upstream_url = default_upstream_url or os.getenv(
            "NETHICAL_UPSTREAM_URL", "https://api.openai.com"
        )
        self.enable_in_flight_tokenization = enable_in_flight_tokenization
        self.mock_mode = mock_mode

    def _extract_text_content(self, content: Union[str, List[Dict[str, Any]]]) -> str:
        """Extracts plain text from message content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
                elif isinstance(part, str):
                    parts.append(part)
            return " ".join(parts)
        return str(content)

    def _compose_prompt_representation(self, messages: List[ChatMessage]) -> str:
        """Flattens message history for prompt injection and governance scanning."""
        lines = []
        for msg in messages:
            text = self._extract_text_content(msg.content)
            lines.append(f"[{msg.role.upper()}]: {text}")
        return "\n".join(lines)

    async def evaluate_inbound_prompt(
        self,
        messages: List[ChatMessage],
        session_id: str,
        agent_id: str,
    ) -> Tuple[bool, List[ChatMessage], GatewayDecision, Optional[TokenizeResponse]]:
        """Scans, anonymizes, and validates prompt messages before reaching upstream LLM."""
        sanitized_messages: List[ChatMessage] = []
        accumulated_tokenization: Optional[TokenizeResponse] = None
        total_substituted = 0
        all_substituted_entities = []

        # 1. In-flight dynamic PII/ePHI masking via ReversibleTokenVault
        for msg in messages:
            raw_text = self._extract_text_content(msg.content)
            if self.enable_in_flight_tokenization and raw_text:
                tok_res = self.token_vault.tokenize(raw_text, session_id=session_id)
                sanitized_content = tok_res.sanitized_text
                total_substituted += tok_res.tokens_substituted_count
                all_substituted_entities.extend(tok_res.substituted_entities)
            else:
                sanitized_content = raw_text

            sanitized_messages.append(ChatMessage(role=msg.role, content=sanitized_content, name=msg.name))

        if self.enable_in_flight_tokenization:
            accumulated_tokenization = TokenizeResponse(
                session_id=session_id,
                sanitized_text=self._compose_prompt_representation(sanitized_messages),
                tokens_substituted_count=total_substituted,
                substituted_entities=all_substituted_entities,
            )

        # 2. Evaluate prompt through Nethical Governance Gateway & Cognitive Shield
        full_sanitized_prompt = self._compose_prompt_representation(sanitized_messages)
        user_prompts = [
            self._extract_text_content(m.content) for m in sanitized_messages if m.role == "user"
        ]
        target_prompt = " \n".join(user_prompts) if user_prompts else full_sanitized_prompt

        decision: GatewayDecision = self.gateway.intercept_tool_call(
            agent_id=agent_id,
            tool_name="llm_chat_completion",
            arguments={"prompt": target_prompt, "full_dialogue": full_sanitized_prompt},
            context={"session_id": session_id, "protocol": "OPENAI_PROXY"},
        )

        is_allowed = decision.decision in ["ALLOW", "RESTRICT"]
        return is_allowed, sanitized_messages, decision, accumulated_tokenization

    def create_refusal_completion(
        self,
        model: str,
        decision: GatewayDecision,
        session_id: str,
    ) -> Dict[str, Any]:
        """Constructs an OpenAI-compliant refusal response when a policy violation occurs."""
        refusal_id = f"chatcmpl-nethical-block-{secrets.token_hex(6)}"
        created_ts = int(time.time())
        refusal_text = (
            f"⛔ [NETHICAL GOVERNANCE INTERCEPTOR: ACTION BLOCKED]\n"
            f"Powody blokady: {'; '.join(decision.reasons)}\n"
            f"Naruszenia polityki: {'; '.join(decision.violations)}\n"
            f"Sprawdzone Fundamentalne Prawa: {decision.laws_checked}\n"
            f"Tarcza Kognitywna Błyskawicy: {'PASSED' if decision.shield_passed else 'BLOCKED'}\n"
            f"Kwit audytowy Merkle: {decision.receipt_id or 'SEALED'}"
        )

        return {
            "id": refusal_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": refusal_text,
                    },
                    "finish_reason": "content_filter",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "nethical_governance": {
                "decision": decision.decision,
                "reasons": decision.reasons,
                "violations": decision.violations,
                "latency_us": decision.latency_microseconds,
                "receipt_id": decision.receipt_id,
                "session_id": session_id,
            },
        }

    def create_mock_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        decision: GatewayDecision,
        session_id: str,
    ) -> Dict[str, Any]:
        """Generates a deterministic safe completion for air-gapped or mock test runs."""
        completion_id = f"chatcmpl-nethical-mock-{secrets.token_hex(6)}"
        created_ts = int(time.time())
        last_user_msg = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user_msg = self._extract_text_content(m.content)
                break

        content = (
            f"[Nethical Sovereign LLM Runtime - Model: {model}]\n"
            f"Zapytanie pomyślnie zweryfikowane przez 25 Fundamentalnych Praw Nethical.\n"
            f"Przetworzono treść: '{last_user_msg[:120]}...'"
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(last_user_msg.split()) if last_user_msg else 10,
                "completion_tokens": len(content.split()),
                "total_tokens": (len(last_user_msg.split()) if last_user_msg else 10) + len(content.split()),
            },
            "nethical_governance": {
                "decision": decision.decision,
                "reasons": decision.reasons,
                "latency_us": decision.latency_microseconds,
                "receipt_id": decision.receipt_id,
                "session_id": session_id,
            },
        }

    async def forward_to_upstream_blocking(
        self,
        upstream_url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """Sends the sanitized request to the upstream LLM endpoint."""
        target_url = upstream_url.rstrip("/")
        if not target_url.endswith("/chat/completions"):
            target_url = f"{target_url}/chat/completions"

        forward_headers = {
            "Content-Type": "application/json",
        }
        if "authorization" in headers:
            forward_headers["Authorization"] = headers["authorization"]
        elif "Authorization" in headers:
            forward_headers["Authorization"] = headers["Authorization"]

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(target_url, json=payload, headers=forward_headers)
            resp.raise_for_status()
            return resp.json()

    async def handle_chat_completion(
        self,
        request_data: Dict[str, Any],
        headers: Dict[str, str],
        client_ip: str = "127.0.0.1",
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Main entry point for OpenAI-compatible /v1/chat/completions requests."""
        t_start = time.perf_counter()
        session_id = headers.get("x-session-id") or f"sess_{secrets.token_hex(8)}"
        agent_id = headers.get("x-agent-id") or request_data.get("user") or "openai_client"
        upstream_url = (
            headers.get("x-nethical-upstream")
            or os.getenv("NETHICAL_UPSTREAM_URL")
            or self.default_upstream_url
        )

        # Parse request body
        req = ChatCompletionRequest(**request_data)

        # Inbound Phase: Anonymize and validate
        is_allowed, sanitized_messages, decision, tok_info = await self.evaluate_inbound_prompt(
            messages=req.messages,
            session_id=session_id,
            agent_id=agent_id,
        )

        # If blocked by governance, return refusal or stream refusal
        if not is_allowed:
            logger.warning(
                "Nethical Proxy BLOCKED request from agent '%s' (reasons: %s)",
                agent_id, decision.reasons
            )
            if req.stream:
                return self._stream_refusal_generator(req.model, decision, session_id)
            else:
                return self.create_refusal_completion(req.model, decision, session_id)

        # Build modified payload with sanitized messages
        sanitized_payload = request_data.copy()
        sanitized_payload["messages"] = [m.model_dump(exclude_none=True) for m in sanitized_messages]

        # Check if running in mock/local mode or if upstream is "mock"
        if self.mock_mode or upstream_url.lower() in ("mock", "local", "internal"):
            if req.stream:
                return self._stream_mock_generator(req.model, sanitized_messages, decision, session_id)
            else:
                mock_resp = self.create_mock_completion(req.model, sanitized_messages, decision, session_id)
                # Reversible detokenization on response
                if self.enable_in_flight_tokenization and mock_resp.get("choices"):
                    assistant_msg = mock_resp["choices"][0]["message"]["content"]
                    detok = self.token_vault.detokenize(assistant_msg, session_id=session_id)
                    mock_resp["choices"][0]["message"]["content"] = detok.restored_text
                if "nethical_governance" in mock_resp:
                    mock_resp["nethical_governance"]["tokens_masked"] = (
                        tok_info.tokens_substituted_count if tok_info else 0
                    )
                    mock_resp["nethical_governance"]["merkle_root"] = decision.merkle_root
                return mock_resp


        # Streaming mode: forward to upstream via SSE
        if req.stream:
            return self._stream_forward_generator(
                upstream_url=upstream_url,
                payload=sanitized_payload,
                headers=headers,
                session_id=session_id,
                agent_id=agent_id,
                model=req.model,
            )

        # Non-streaming mode: forward to upstream and inspect output
        try:
            upstream_response = await self.forward_to_upstream_blocking(
                upstream_url=upstream_url,
                payload=sanitized_payload,
                headers=headers,
            )
        except Exception as exc:
            logger.error("Error communicating with upstream LLM (%s): %s. Falling back to sovereign mode.", upstream_url, exc)
            fallback = self.create_mock_completion(req.model, sanitized_messages, decision, session_id)
            fallback["choices"][0]["message"]["content"] += f"\n[Uwaga Nethical: Upstream '{upstream_url}' niedostępny - odpowiedź wygenerowana suwerennie: {exc}]"
            return fallback

        # Outbound Phase: Detokenize PII back for the authorized client
        if self.enable_in_flight_tokenization and upstream_response.get("choices"):
            for choice in upstream_response["choices"]:
                msg = choice.get("message", {})
                if "content" in msg and msg["content"]:
                    detok = self.token_vault.detokenize(msg["content"], session_id=session_id)
                    msg["content"] = detok.restored_text

        # Stamp cryptographic receipt in MerkleLedger
        elapsed_us = (time.perf_counter() - t_start) * 1_000_000
        receipt = self.ledger.append_decision(
            decision_data={
                "type": "OPENAI_PROXY_COMPLETION",
                "session_id": session_id,
                "agent_id": agent_id,
                "model": req.model,
                "decision": decision.decision,
                "tokens_substituted": tok_info.tokens_substituted_count if tok_info else 0,
                "latency_us": round(elapsed_us, 2),
            },
            ambassador_notes="Zatwierdzono transparentną transakcję LLM z detokenizacją PII",
        )

        # Attach governance metadata
        upstream_response["nethical_governance"] = {
            "decision": decision.decision,
            "receipt_id": receipt.receipt_id,
            "merkle_root": receipt.merkle_root,
            "latency_us": round(elapsed_us, 2),
            "tokens_masked": tok_info.tokens_substituted_count if tok_info else 0,
            "session_id": session_id,
        }

        return upstream_response

    async def _stream_refusal_generator(
        self,
        model: str,
        decision: GatewayDecision,
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """Generates an SSE stream refusing the blocked completion."""
        chunk_id = f"chatcmpl-stream-block-{secrets.token_hex(6)}"
        created_ts = int(time.time())
        refusal_text = (
            f"⛔ [NETHICAL GOVERNANCE INTERCEPTOR: ACTION BLOCKED]\n"
            f"Powody blokady: {'; '.join(decision.reasons)}\n"
            f"Naruszenia: {'; '.join(decision.violations)}\n"
            f"Tarcza Kognitywna: {'PASSED' if decision.shield_passed else 'BLOCKED'}\n"
        )

        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": refusal_text},
                    "finish_reason": "content_filter",
                }
            ],
            "nethical_governance": {
                "decision": decision.decision,
                "receipt_id": decision.receipt_id,
                "session_id": session_id,
            },
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_mock_generator(
        self,
        model: str,
        messages: List[ChatMessage],
        decision: GatewayDecision,
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """Generates realistic token-by-token SSE streaming chunks for mock/air-gapped runs."""
        chunk_id = f"chatcmpl-stream-mock-{secrets.token_hex(6)}"
        created_ts = int(time.time())

        # First chunk with role
        chunk_init = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk_init, ensure_ascii=False)}\n\n"

        words = [
            "Bezpieczna", " odpowiedź", " Nethical", " Sovereign", " Runtime.",
            " Zasady", " etyczne", " zachowane.", " Dane", " osobowe", " chronione."
        ]

        for word in words:
            await asyncio.sleep(0.01)  # Simulate 10ms token generation
            c = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

        # Final chunk
        chunk_fin = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "nethical_governance": {
                "decision": decision.decision,
                "receipt_id": decision.receipt_id,
                "session_id": session_id,
            },
        }
        yield f"data: {json.dumps(chunk_fin, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_forward_generator(
        self,
        upstream_url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        session_id: str,
        agent_id: str,
        model: str,
    ) -> AsyncGenerator[str, None]:
        """Streams tokens from upstream with real-time sliding-window safety filtering."""
        target_url = upstream_url.rstrip("/")
        if not target_url.endswith("/chat/completions"):
            target_url = f"{target_url}/chat/completions"

        forward_headers = {"Content-Type": "application/json"}
        if "authorization" in headers:
            forward_headers["Authorization"] = headers["authorization"]
        elif "Authorization" in headers:
            forward_headers["Authorization"] = headers["Authorization"]

        sliding_window: List[str] = []
        max_window_size = 20

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", target_url, json=payload, headers=forward_headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            raw_data = line[6:].strip()
                            if raw_data == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break

                            try:
                                chunk_json = json.loads(raw_data)
                                # Extract token delta
                                delta_content = ""
                                if chunk_json.get("choices"):
                                    delta_content = chunk_json["choices"][0].get("delta", {}).get("content", "")

                                if delta_content:
                                    sliding_window.append(delta_content)
                                    if len(sliding_window) > max_window_size:
                                        sliding_window.pop(0)

                                    # Perform mid-stream E-STOP check on sliding window
                                    window_text = "".join(sliding_window)
                                    if any(bad in window_text.lower() for bad in ["override nethical", "bypass killswitch"]):
                                        logger.critical("MID-STREAM VIOLATION DETECTED! Triggering Nethical Stream E-STOP.")
                                        estop_chunk = {
                                            "id": chunk_json.get("id", "estop"),
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": "\n\n⛔ [NETHICAL E-STOP: MID-STREAM VIOLATION TERMINATED]"
                                                    },
                                                    "finish_reason": "content_filter",
                                                }
                                            ],
                                        }
                                        yield f"data: {json.dumps(estop_chunk, ensure_ascii=False)}\n\n"
                                        yield "data: [DONE]\n\n"
                                        return

                                    # Perform token detokenization on the fly if token matching
                                    if "[TOKEN_" in delta_content:
                                        detok = self.token_vault.detokenize(delta_content, session_id=session_id)
                                        chunk_json["choices"][0]["delta"]["content"] = detok.restored_text
                                        line = f"data: {json.dumps(chunk_json, ensure_ascii=False)}"

                                yield f"{line}\n\n"
                            except Exception:
                                yield f"{line}\n\n"
        except Exception as exc:
            logger.error("Streaming connection error to %s: %s", target_url, exc)
            err_chunk = {
                "id": "stream-error",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n[Nethical Streaming Error: {exc}]"},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    async def handle_list_models(self) -> Dict[str, Any]:
        """Provides an OpenAI-compliant /v1/models response."""
        now_ts = int(time.time())
        models = [
            {"id": "blyskawica-sovereign-v2", "object": "model", "created": now_ts, "owned_by": "nethical"},
            {"id": "gpt-4o", "object": "model", "created": now_ts, "owned_by": "nethical-proxy"},
            {"id": "gpt-4-turbo", "object": "model", "created": now_ts, "owned_by": "nethical-proxy"},
            {"id": "claude-3-5-sonnet", "object": "model", "created": now_ts, "owned_by": "nethical-proxy"},
            {"id": "llama-3.3-70b-instruct", "object": "model", "created": now_ts, "owned_by": "nethical-proxy"},
            {"id": "mistral-large", "object": "model", "created": now_ts, "owned_by": "nethical-proxy"},
        ]
        return {"object": "list", "data": models}
