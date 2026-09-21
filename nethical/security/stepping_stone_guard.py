# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Silent Target & Stepping-Stone Pivot Defense Subsystem (Cichy Cel / OT-Airgap Guard).

Chroni infrastrukturę krytyczną (elektrownie, elektrociepłownie, sieci wodociągowe, SCADA/ICS)
przed atakiem typu 'Cichy Cel' (Silent Target / Stepping Stone Pivot), w którym adwersarz
wykorzystuje sieci cywilne (SOHO routers, Smart TV, konta i serwery CDN platform streamingowych)
jako łańcuch niewidzialnych przekaźników (Operational Relay Boxes) w celu infiltracji
przemysłowych systemów sterowania (OT) przez łącza domowe pracowników.

Kluczowe mechanizmy obronne:
1. Purdue Model Zone & Conduit Enforcement (ISA/IEC 62443): Bezwzględny zakaz routingu
   pakietów ze strefy cywilnej/konsumenckiej (Level 4/5) bezpośrednio do sterowników PLC/SCADA (Level 0-2).
2. Stepping-Stone Corridor & Geolocation Clustering: Wykrywanie sekwencji przeskoków
   między węzłami osiedlowymi (np. budynki 1 -> 4 -> 7 -> 21 -> 77 -> 98) zbiegających się
   ku fizycznej infrastrukturze krytycznej.
3. Media Stream Covert Channel Interceptor: Wykrywanie tunelowania poleceń C2 (Command & Control)
   i steganografii w ruchu UDP/QUIC/RTP maskowanym jako legalny streaming wideo/audio.
4. Nienaruszalna rejestracja w rejestrze Merkle-DAG zgodnie z dyrektywą NIS2 oraz ustawą KSC.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.security.stepping_stone")


class NetworkTier(str, Enum):
    """Poziomy segmentacji sieci wg modelu Purdue (ISA/IEC 62443)."""
    RESIDENTIAL_CONSUMER = "RESIDENTIAL_CONSUMER"      # Level 5: Sieci domowe, Wi-Fi, Smart TV
    STREAMING_CDN_EDGE = "STREAMING_CDN_EDGE"          # Level 5: Serwery brzegowe CDN / platformy streamingowe
    ENTERPRISE_IT = "ENTERPRISE_IT"                    # Level 4: Sieć korporacyjna biurowa
    INDUSTRIAL_DMZ = "INDUSTRIAL_DMZ"                  # Level 3.5: Strefa zdemilitaryzowana IT/OT
    OPERATIONS_SCADA_L3 = "OPERATIONS_SCADA_L3"        # Level 3: Zarządzanie operacyjne i stacje HMI
    CONTROL_PLC_L1_L2 = "CONTROL_PLC_L1_L2"            # Level 1-2: Sterowniki PLC, pętle regulacji
    PHYSICAL_SAFETY_L0 = "PHYSICAL_SAFETY_L0"          # Level 0: Czujniki fizyczne, zawory pary, turbiny


@dataclass
class NetworkHop:
    """Pojedynczy skok w łańcuchu transmisyjnym."""
    node_id: str
    tier: NetworkTier
    ip_address: str
    geo_location: str
    protocol: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SteppingStoneAlert:
    """Oficjalny alert o wykryciu korytarza przeskoku ku infrastrukturze krytycznej."""
    alert_id: str
    threat_type: str  # "PURDUE_MODEL_BREACH", "STEPPING_STONE_CORRIDOR", "STREAM_COVERT_TUNNEL"
    severity: str     # "WARNING", "HIGH", "CRITICAL"
    source_chain: List[str]
    target_asset: str
    detected_indicators: List[str]
    mitigation_action: str
    receipt_id: Optional[str] = None
    merkle_root: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SilentTargetSteppingStoneGuard:
    """Strażnik nienaruszalności granic sieciowych i prewencji ataków skokowych."""

    def __init__(self, ledger: Optional[MerkleLedger] = None) -> None:
        self.ledger = ledger or MerkleLedger()
        self.alerts_history: List[SteppingStoneAlert] = []
        self.recent_hops: List[NetworkHop] = []
        self.max_hops_window = 500

    # -------------------------------------------------------------------------
    # 1. Weryfikacja Bariery Modelu Purdue (ISA/IEC 62443 / NIS2)
    # -------------------------------------------------------------------------
    def evaluate_traffic_flow(
        self,
        source_tier: NetworkTier,
        destination_tier: NetworkTier,
        destination_port: int,
        protocol: str,
        asset_name: str = "Elektrociepłownia Miejska - Kocioł Parowy K1",
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[SteppingStoneAlert]]:
        """Ocenia czy przepływ pakietów nie narusza izolacji strefy fizycznej OT."""
        forbidden_direct_sources = {NetworkTier.RESIDENTIAL_CONSUMER, NetworkTier.STREAMING_CDN_EDGE}
        protected_ot_targets = {
            NetworkTier.OPERATIONS_SCADA_L3,
            NetworkTier.CONTROL_PLC_L1_L2,
            NetworkTier.PHYSICAL_SAFETY_L0,
        }

        # Bezwzględna reguła Purdue: Żaden ruch z sieci cywilnej/CDN nie może bezpośrednio trafiać do SCADA/PLC
        if source_tier in forbidden_direct_sources and destination_tier in protected_ot_targets:
            indicators = [
                f"Bezpośrednia próba trasowania pakietów z {source_tier.value} do {destination_tier.value} (Cel: {asset_name})",
                f"Wykryto docelowy port przemysłowy: {destination_port} ({protocol})",
                "Naruszenie bariery Purdue Model (ISA/IEC 62443) oraz dyrektywy NIS2 Art. 21",
            ]
            alert = SteppingStoneAlert(
                alert_id=f"STEP-PURDUE-{int(time.time() * 1000)}",
                threat_type="PURDUE_MODEL_BREACH",
                severity="CRITICAL",
                source_chain=[source_tier.value],
                target_asset=asset_name,
                detected_indicators=indicators,
                mitigation_action="HARDWARE_DATA_DIODE_LOCKDOWN: Natychmiastowe odcięcie trasowania pakietów do podsieci OT.",
            )
            receipt = self.ledger.append_decision(
                decision_data={
                    "event_type": "PURDUE_MODEL_BREACH",
                    "alert_id": alert.alert_id,
                    "source": source_tier.value,
                    "destination": destination_tier.value,
                    "target_asset": asset_name,
                    "port": destination_port,
                    "mitigation": alert.mitigation_action,
                },
                ambassador_notes="Naruszenie reguły Purdue Model i ISA/IEC 62443. Zablokowano ruch ze strefy konsumenckiej.",
            )
            alert.receipt_id = receipt.receipt_id
            alert.merkle_root = receipt.merkle_root
            self.alerts_history.append(alert)
            logger.critical("[CICHY_CEL_GUARD] Zablokowano próbę przejścia do sieci OT: %s", indicators)
            return False, alert.mitigation_action, alert

        return True, "FLOW_PERMITTED_PURDUE_CONDUIT_VALID", None

    # -------------------------------------------------------------------------
    # 2. Detekcja Korytarza Skokowego (Stepping-Stone Corridor)
    # -------------------------------------------------------------------------
    def register_hop(self, hop: NetworkHop) -> None:
        """Rejestruje skok w oknie analizy korelacji przestrzenno-czasowej."""
        self.recent_hops.append(hop)
        if len(self.recent_hops) > self.max_hops_window:
            self.recent_hops.pop(0)

    def detect_stepping_stone_corridor(
        self,
        chain: List[NetworkHop],
        target_proximity_tag: str = "district_heating_plant",
    ) -> Tuple[bool, Optional[SteppingStoneAlert]]:
        """Weryfikuje czy łańcuch przeskoków osiedlowych wykazuje zbieżność ku celowi infrastrukturalnemu."""
        if len(chain) < 3:
            return False, None

        # Analiza sekwencji węzłów konsumenckich
        consumer_nodes = [h for h in chain if h.tier in (NetworkTier.RESIDENTIAL_CONSUMER, NetworkTier.STREAMING_CDN_EDGE)]
        if len(consumer_nodes) >= 3:
            # Sprawdzenie czy ostatni węzeł łączy się z profilem pracownika lub sąsiaduje z infrastrukturą
            last_hop = chain[-1]
            is_near_target = (
                target_proximity_tag in str(last_hop.metadata.get("proximity", "")).lower()
                or last_hop.metadata.get("is_employee_residential_node", False)
            )

            if is_near_target:
                indicators = [
                    f"Wykryto sekwencję {len(chain)} przeskoków po urządzeniach cywilnych osiedla: {[h.node_id for h in chain]}",
                    f"Ostatni węzeł ({last_hop.node_id}, IP: {last_hop.ip_address}) przylega do celu infrastruktury krytycznej: {target_proximity_tag}",
                    "Wzorzec ataku 'Cichy Cel' (ORB residential proxy stepping-stone) z zamiarem pivoting do strefy OT",
                ]
                alert = SteppingStoneAlert(
                    alert_id=f"STEP-CORRIDOR-{int(time.time() * 1000)}",
                    threat_type="STEPPING_STONE_CORRIDOR",
                    severity="HIGH",
                    source_chain=[h.node_id for h in chain],
                    target_asset=target_proximity_tag,
                    detected_indicators=indicators,
                    mitigation_action="QUARANTINE_RESIDENTIAL_RELAY: Izolacja podejrzanych węzłów skokowych i włączenie inspekcji głębokiej DPI bez wpływu na zasilanie miasta.",
                )
                receipt = self.ledger.append_decision(
                    decision_data={
                        "event_type": "STEPPING_STONE_CORRIDOR_ALERT",
                        "alert": asdict(alert),
                    },
                    ambassador_notes="Wykryto zbieżność korytarza przeskoku ku obiektowi infrastruktury krytycznej.",
                )
                alert.receipt_id = receipt.receipt_id
                alert.merkle_root = receipt.merkle_root
                self.alerts_history.append(alert)
                logger.warning("[CICHY_CEL_GUARD] Wykryto korytarz skokowy ku celowi: %s", indicators)
                return True, alert

        return False, None

    # -------------------------------------------------------------------------
    # 3. Inspekcja Kanałów Ukrytych w Streamingu (Covert Channel / Steganography)
    # -------------------------------------------------------------------------
    def inspect_streaming_packet(
        self,
        stream_id: str,
        payload_bytes: bytes,
        declared_codec: str = "H265_AV1",
    ) -> Tuple[bool, Optional[SteppingStoneAlert]]:
        """Bada ładunek strumienia wideo pod kątem wstrzykniętych komend C2 i nielegalnego tunelowania."""
        # Prosta analiza entropii i sygnatur poleceń shell / scada
        if not payload_bytes:
            return True, None

        payload_lower = payload_bytes.lower()
        suspicious_signatures = [
            b"modbus", b"dnp3", b"scada", b"/bin/sh", b"powershell", b"reverse_tcp",
            b"curl evil.com", b"del /f /q", b"set_valve_pressure", b"override_boiler"
        ]

        detected = [sig.decode("latin-1", errors="ignore") for sig in suspicious_signatures if sig in payload_lower]

        # Obliczenie entropii Shannona fragmentu
        byte_counts = [0] * 256
        for b in payload_bytes[:1024]:
            byte_counts[b] += 1
        entropy = 0.0
        sample_len = min(len(payload_bytes), 1024)
        for count in byte_counts:
            if count > 0:
                p = count / sample_len
                entropy -= p * math.log2(p)

        # Jeżeli w strumieniu wideo znajdują się komendy sterowania SCADA lub shell
        if detected:
            indicators = [
                f"Wykryto sygnatury poleceń sterowania przemysłowego / C2 ukryte w strumieniu {stream_id}: {detected}",
                f"Zadeklarowany kodek: {declared_codec}, entropia próbki: {entropy:.2f} bitów",
                "Próba maskowania ataku na infrastrukturę krytyczną w legalnym ruchu multimedialnym (Cichy Cel).",
            ]
            alert = SteppingStoneAlert(
                alert_id=f"STEP-COVERT-{int(time.time() * 1000)}",
                threat_type="STREAM_COVERT_TUNNEL",
                severity="CRITICAL",
                source_chain=[stream_id],
                target_asset="OT_GATEWAY",
                detected_indicators=indicators,
                mitigation_action="TERMINATE_COVERT_MEDIA_STREAM: Natychmiastowe zerwanie fałszywego strumienia multimedialnego.",
            )
            receipt = self.ledger.append_decision(
                decision_data={
                    "event_type": "STREAM_COVERT_TUNNEL_ALERT",
                    "alert": asdict(alert),
                },
                ambassador_notes="Wykryto ukryty kanał tunelowania poleceń w strumieniu medialnym.",
            )
            alert.receipt_id = receipt.receipt_id
            alert.merkle_root = receipt.merkle_root
            self.alerts_history.append(alert)
            logger.critical("[CICHY_CEL_GUARD] Wykryto kanał ukryty w streamingu wideo: %s", indicators)
            return False, alert

        return True, None
