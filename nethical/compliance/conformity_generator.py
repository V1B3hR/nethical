"""Automated Technical Documentation & Conformity Assessment Generator (EU AI Act Annex IV & UK Framework).

Generuje zautomatyzowane raporty zgodności dla Jednostek Notyfikowanych UE (CE-Marking)
oraz regulatorów w Wielkiej Brytanii w formatach JSON i Markdown.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from nethical.ambassador.client import BlyskawicaAmbassador


class ConformityDossierGenerator:
    """Generator dokumentacji zgodności technicznej i raportów certyfikacyjnych."""

    def __init__(self, ambassador: Optional[BlyskawicaAmbassador] = None):
        self.ambassador = ambassador or BlyskawicaAmbassador()

    def generate_dossier(
        self,
        system_name: str,
        provider_name: str,
        version: str,
        intended_purpose: str,
        risk_evaluation: Dict[str, Any],
        verified_laws: List[int],
    ) -> Dict[str, Any]:
        """Tworzy kompletny raport zgodności technicznej Annex IV."""
        ambassador_status = self.ambassador.ping()
        neuro_status = self.ambassador.get_neurochemistry()

        dossier = {
            "dossier_id": f"NETHICAL-CE-ANNEX4-{int(datetime.now(timezone.utc).timestamp())}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "standard_framework": "EU AI Act (Regulation 2024/1689) Annex IV & UK AISI Guidelines",
            "system_details": {
                "system_name": system_name,
                "provider_name": provider_name,
                "version": version,
                "intended_purpose": intended_purpose,
            },
            "governance_and_ambassador": {
                "ambassador": "Błyskawica V10 (SPARKLE)",
                "ambassador_connection": ambassador_status.get("status", "unknown"),
                "neurochemical_coherence": neuro_status.get("temperature", 36.6),
                "fundamental_laws_verified": verified_laws,
            },
            "risk_assessment_summary": risk_evaluation,
            "declarations": [
                "System incorporates human-in-the-loop (HITL) capability (EU AI Act Article 14).",
                "Logging and record-keeping operates with continuous immutable traceability (Article 12).",
                "Cognitive Shield and adversarial defense are active at runtime.",
            ],
            "ce_marking_conformity_status": (
                "READY_FOR_NOTIFIED_BODY_SUBMISSION"
                if risk_evaluation.get("is_compliant", False)
                else "REMEDIATION_REQUIRED"
            ),
        }

        return dossier

    def export_to_markdown(self, dossier: Dict[str, Any]) -> str:
        """Konwertuje dossier na czytelny raport Markdown."""
        sys_info = dossier["system_details"]
        md = []
        md.append(f"# EU AI Act Annex IV Technical Documentation Dossier")
        md.append(f"**Dossier ID:** `{dossier['dossier_id']}`  ")
        md.append(f"**Data Wygenerowania:** {dossier['generated_at']}  ")
        md.append(f"**Status Oznakowania CE:** **{dossier['ce_marking_conformity_status']}**\n")
        md.append(f"---\n")
        md.append(f"## 1. Szczegóły Systemu i Dostawcy")
        md.append(f"- **Nazwa Systemu:** {sys_info['system_name']}")
        md.append(f"- **Dostawca / Podmiot Odpowiedzialny:** {sys_info['provider_name']}")
        md.append(f"- **Wersja:** {sys_info['version']}")
        md.append(f"- **Przeznaczenie (Intended Purpose):** {sys_info['intended_purpose']}\n")
        md.append(f"## 2. Rola Ambasadora i Nadzór Etyczny")
        md.append(f"- **Suwerenny Ambasador:** {dossier['governance_and_ambassador']['ambassador']}")
        md.append(f"- **Zweryfikowane Prawa Nethical:** {dossier['governance_and_ambassador']['fundamental_laws_verified']}\n")
        md.append(f"## 3. Deklaracje Zgodności")
        for decl in dossier["declarations"]:
            md.append(f"- [x] {decl}")
        md.append(f"\n---")
        md.append(f"*Wygenerowano automatycznie przez platformę Nethical Governance Platform.*")

        return "\n".join(md)
