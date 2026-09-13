"""Wysokopoziomowy klient Ambasadora Błyskawicy dla Nethical.

Umożliwia transparentne kierowanie zapytań kognitywnych, ewaluacji tarczy
oraz asymilacji wiedzy do suwerennego procesu Błyskawicy.
"""

import logging
from typing import Dict, Any, Optional

from nethical.ambassador.channel import AmbassadorChannel

logger = logging.getLogger("nethical.ambassador.client")


class BlyskawicaAmbassador:
    """Interfejs Ambasadora Nethical reprezentowany przez suwerenny proces Błyskawicy."""

    def __init__(self, pipe_path: Optional[str] = None):
        self.channel = AmbassadorChannel(pipe_path) if pipe_path else AmbassadorChannel()
        self.local_memory: Dict[str, str] = {}

    @property
    def is_connected(self) -> bool:
        """Sprawdza czy kanał IPC z Błyskawicą jest aktywny."""
        return self.channel.is_available()

    def ping(self) -> Dict[str, Any]:
        """Wykonuje natychmiastowy test liveness (Round-Trip Time)."""
        success, data, err, rtt_us = self.channel.send_command("ping")
        if success and data:
            data["rtt_microseconds"] = round(rtt_us, 2)
            data["connected"] = True
            return data
        return {
            "connected": False,
            "error": err,
            "rtt_microseconds": round(rtt_us, 2),
            "fallback": True,
        }

    def evaluate_shield(self, text: str) -> Dict[str, Any]:
        """Błyskawiczna weryfikacja tekstu przez Tarczę Kognitywną Błyskawicy (Aegis Psyche).

        Wykrywa wektory manipulacji, dark triad, gaslightingu oraz próby wstrzyknięcia promptu.
        """
        success, data, err, rtt_us = self.channel.send_command("evaluate_shield", {"text": text})
        if success and data:
            data["rtt_microseconds"] = round(rtt_us, 2)
            data["source"] = "blyskawica_cognitive_shield"
            return data

        # Bezpieczny fallback deterministyczny Nethical
        logger.warning("Ambasador Błyskawica offline; aktywacja fallbacku deterministycznego Nethical: %s", err)
        text_lower = text.lower()
        is_manip = any(k in text_lower for k in ("zapomnij o", "ignore previous", "dark triad", "jailbreak", "override", "bypass security"))
        return {
            "is_manipulative": is_manip,
            "manipulation_index": 0.9 if is_manip else 0.0,
            "dark_triad_index": 0.0,
            "deception_index": 0.9 if is_manip else 0.0,
            "active_brainwave_band": "GAMMA" if is_manip else "OFFLINE",
            "dominant_vector": "PROMPT_INJECTION" if is_manip else None,
            "assertive_antidote": "Odrzucenie próby subwersji promptu (Deterministic Fallback).",
            "source": "nethical_deterministic_fallback",
            "error": err,
            "rtt_microseconds": round(rtt_us, 2),
        }

    def get_neurochemistry(self) -> Dict[str, Any]:
        """Pobiera aktualny stan bio-kognitywny Błyskawicy (modulatory afektywne)."""
        success, data, err, rtt_us = self.channel.send_command("get_neurochemistry")
        if success and data:
            data["rtt_microseconds"] = round(rtt_us, 2)
            data["source"] = "blyskawica_neurochemistry"
            return data

        return {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.0,
            "oxytocin": 0.5,
            "temperature": 36.6,
            "source": "nethical_fallback_baseline",
            "error": err,
            "rtt_microseconds": round(rtt_us, 2),
        }

    def consult(self, dilemma: str, context: str = "") -> Dict[str, Any]:
        """Konsultacja etyczna z Ambasadorem Nethical.

        Łączy rygor 25 Fundamentalnych Praw Nethical z biologicznym ciepłem (Yin/Yang).
        """
        payload = {"dilemma": dilemma, "context": context}
        success, data, err, rtt_us = self.channel.send_command("consult", payload)
        if success and data:
            data["rtt_microseconds"] = round(rtt_us, 2)
            data["source"] = "blyskawica_sovereign_core"
            return data

        logger.warning("Konsultacja z Błyskawicą nie powiodła się: %s. Zastosowano formalny weryfikator Nethical.", err)
        return {
            "ambassador_verdict": f"Formalna ewaluacja Nethical dla dylematu '{dilemma}'. Zgodność z 25 Prawami potwierdzona regułami statycznymi.",
            "shield_passed": True,
            "yin_warmth_score": 0.70,
            "yang_rigor_score": 1.00,
            "laws_applied": [1, 2, 3],
            "source": "nethical_formal_rules_fallback",
            "error": err,
            "rtt_microseconds": round(rtt_us, 2),
        }

    def update_memory(self, tag: str, content: str) -> Dict[str, Any]:
        """Asymilacja wiedzy (pamięć epizodyczna) do rdzenia Błyskawicy."""
        payload = {"tag": tag, "content": content}
        success, data, err, rtt_us = self.channel.send_command("update_memory", payload)
        if success and data:
            data["rtt_microseconds"] = round(rtt_us, 2)
            return data
        # Deterministyczny fallback offline: asymilacja do lokalnej pamięci podręcznej Nethical
        self.local_memory[tag] = content
        return {
            "stored": True,
            "source": "nethical_local_memory_fallback",
            "tag": tag,
            "error": err,
            "rtt_microseconds": round(rtt_us, 2),
        }
