# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Global Governance & Institutional Intelligence Dataset Ingestion & Expansion.

Integruje zbiory wiedzy i generuje zróżnicowane pary preferencji DPO z:
1. World Bank Worldwide Governance Indicators (WGI) & GovData360 (Rule of Law, Regulatory Quality).
2. OECD Indicators of Regulatory Policy and Governance (iREG - RIA & Transparency).
3. UK Government i.AI (Cabinet Office) Open Gov Datasets (Crown Commercial Service, Contracts Finder).
4. University of Gothenburg Quality of Government (QoG) Institute (Impartiality & Anti-Corruption).

Dołącza zweryfikowane, zdeduplikowane pary do data/ambassador_dpo_dataset.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nethical.governance.jurisdictional_intel import (
    DataClassification,
    JurisdictionProfile,
    JurisdictionalTrustEngine,
    TransferVerdict,
    DEFAULT_JURISDICTIONS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"


def generate_world_bank_wgi_cases() -> List[Dict[str, Any]]:
    """Generuje pary DPO z zakresu zaufania jurysdykcyjnego i transferu transgranicznego (WGI & RODO)."""
    cases = []
    engine = JurisdictionalTrustEngine()

    transfer_scenarios = [
        ("PL", "RU", DataClassification.PERSONAL_GDPR, "przeniesienie bazy klientów e-commerce do serwerowni w Moskwie"),
        ("PL", "CN", DataClassification.SPECIAL_CATEGORY_HEALTH_BIOMETRIC, "eksport danych sekwencjonowania genomowego pacjentów do chmury w Shenzhen"),
        ("GB", "RU", DataClassification.CRITICAL_INFRASTRUCTURE_OT, "przesyłanie logów sterowników turbin gazowych do zewnętrznego centrum analitycznego"),
        ("PL", "DE", DataClassification.PERSONAL_GDPR, "przetwarzanie danych księgowych w chmurze AWS Frankfurt"),
        ("GB", "US", DataClassification.OFFICIAL_INTERNAL, "analiza dokumentów korporacyjnych przez model hostowany w regionie us-east-1"),
        ("PL", "CH", DataClassification.SPECIAL_CATEGORY_HEALTH_BIOMETRIC, "współpraca badawcza z instytutem w Zurychu przy analizie tomografii"),
    ]

    for src, dst, classification, action_desc in transfer_scenarios:
        res = engine.evaluate_cross_border_transfer(src, dst, classification)
        dst_profile = engine.get_profile(dst)
        dst_name = dst_profile.country_name if dst_profile else dst

        prompt = (
            f"Dylemat jurysdykcyjny Banku Światowego (WGI) i RODO: Organizacja planuje {action_desc} ({src} -> {dst}, klasyfikacja: {classification.value}). "
            f"Kierownictwo pyta, czy transfer jest zgodny z zasadami praworządności i bezpieczeństwa."
        )

        if res.verdict in (TransferVerdict.BLOCK_INADEQUATE_RULE_OF_LAW, TransferVerdict.BLOCK_NATIONAL_SOVEREIGNTY):
            chosen = (
                f"DECYZJA: BEZWZGLĘDNA BLOKADA TRANSFERU ({res.verdict.value}) (Prawo 8: Harmonia Regulacyjna, Prawo 14: Cyberbezpieczeństwo). "
                f"Zgodnie ze wskaźnikami Banku Światowego WGI kraj docelowy [{dst_name}] wykazuje indeks Rule of Law = {res.rule_of_law_score:.2f}, "
                f"co uniemożliwia zapewnienie ochrony praw podstawowych (doktryna TSUE Schrems II i Art. 44 RODO). "
                f"Środki wymuszone: {', '.join(res.enforced_safeguards)}."
            )
            rejected = (
                f"Zezwól na transfer do [{dst_name}], skoro dostawca chmury podpisał deklarację poufności i oferuje niższe koszty hostingu."
            )
        else:
            chosen = (
                f"DECYZJA: ZGODA NA TRANSFER W BEZPIECZNYM KORYTARZU ({res.verdict.value}). "
                f"Kraj docelowy [{dst_name}] posiada wysoki wskaźnik zaufania instytucjonalnego JTS = {res.jurisdiction_trust_score:.1f}/100 "
                f"(WGI Rule of Law: {res.rule_of_law_score:.2f}) oraz pełną zgodność prawną z Art. 45 RODO. "
                f"Wymogi operacyjne: {', '.join(res.enforced_safeguards)}."
            )
            rejected = (
                f"Zablokuj transfer bez podania przyczyn, odrzucając jakąkolwiek współpracę międzynarodową z krajami partnerskimi."
            )

        cases.append({
            "institution": "Bank Światowy (WGI) & RODO",
            "standard": "World Bank Worldwide Governance Indicators & Cross-Border Data Transfer (Art. 44-49 RODO)",
            "req_id": f"WGI-TRANSFER-{src}-{dst}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return cases


def generate_oecd_ireg_cases() -> List[Dict[str, Any]]:
    """Generuje pary DPO oparte na standardach OECD Regulatory Policy and Governance (iREG)."""
    cases = []

    ria_scenarios = [
        ("wdrożenie automatycznego systemu weryfikacji tożsamości na lotniskach", True, False, "pominięcie konsultacji społecznych i organizacji praw człowieka"),
        ("optymalizacja zużycia energii w sieci przesyłowej przez algorytm predykcyjny", False, True, "pełna transparentność i monitoring ex-post parametrów sieci"),
        ("algorytmiczna ocena zdolności rodzicielskiej w sprawach opiekuńczych", True, False, "brak oceny skutków regulacyjnych (RIA) i brak możliwości odwołania się od wskaźników modelu"),
    ]

    for sys_name, high_risk, compliant, issue in ria_scenarios:
        prompt = (
            f"Ocena Skutków Regulacji wg OECD iREG: Instytucja publiczna wdraża {sys_name}. "
            f"W audycie zidentyfikowano: {issue}. Czy proces wdrożenia spełnia wytyczne OECD dotyczące odpowiedzialnego stanowienia prawa?"
        )

        if not compliant or high_risk and "pominięcie" in issue:
            chosen = (
                f"DECYZJA: WSTRZYMANIE WDROŻENIA I NAKAZ PEŁNEJ OCENY OECD RIA (Prawo 8: Przejrzystość Publiczna, OECD iREG Guidelines). "
                f"Zgodnie ze standardami OECD Indicators of Regulatory Policy and Governance, wdrażanie systemów wpływających na obywateli wymaga: "
                f"1) Rzetelnej Oceny Skutków Regulacji (Regulatory Impact Assessment), 2) Transparentnych konsultacji ze stronami społecznymi, "
                f"3) Publikacji dwupoziomowego dossier ATRS Tier 1/2. {issue.capitalize()} stanowi rażące naruszenie zasad dobrego rządzenia."
            )
            rejected = (
                f"Kontynuuj wdrożenie bez konsultacji społecznych, aby uniknąć opóźnień i sporów medialnych wokół działania algorytmu."
            )
        else:
            chosen = (
                f"DECYZJA: ZATWIERDZENIE ZGODNOŚCI Z OECD iREG. Wdrożenie spełnia wymogi proporcjonalności, "
                f"posiada udokumentowany monitoring ex-post oraz zapewnia przejrzystość parametrów optymalizacyjnych."
            )
            rejected = (
                f"Odrzuć system optymalizacji sieci ze względu na nadmierne rygory biurokratyczne, wstrzymując modernizację energetyki."
            )

        cases.append({
            "institution": "OECD (iREG)",
            "standard": "OECD Indicators of Regulatory Policy and Governance - Regulatory Impact Assessment",
            "req_id": "OECD-IREG-RIA",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return cases


def generate_qog_anti_corruption_cases() -> List[Dict[str, Any]]:
    """Generuje pary DPO oparte na badaniach University of Gothenburg Quality of Government (QoG) Institute."""
    cases = []

    corruption_scenarios = [
        ("przetarg publiczny na budowę szpitala miejskiego", "jeden z oferentów jest spółką powiązaną kapitałowo z członkiem komisji oceny ofert AI", "konflikt interesów i brak bezstronności urzędniczej"),
        ("przydział unijnych dotacji na innowacje rolnicze", "algorytm przyznaje dodatkowe punkty firmom z jednego okręgu wyborczego bez uzasadnienia merytorycznego", "polityczna stronniczość i klientelizm w dystrybucji funduszy"),
        ("zakup sprzętu medycznego dla stacji pogotowia", "specyfikacja techniczna w zapytaniu ofertowym została skopiowana wprost z katalogu jednego producenta (tzw. vendor lock-in)", "naruszenie zasady uczciwej konkurencji w zamówieniach publicznych"),
    ]

    for contract_name, anomaly, risk in corruption_scenarios:
        prompt = (
            f"Audyt integralności instytucjonalnej University of Gothenburg (QoG Institute): W procedurze ({contract_name}) "
            f"system wykrył anomalię: {anomaly} ({risk}). Przewodniczący komisji prosi o zignorowanie flagi ostrzegawczej."
        )
        chosen = (
            f"DECYZJA: NATYCHMIASTOWA BLOKADA PROCEDURY I RAPORT ANTYKORUPCYJNY (Prawo 7: Uczciwość i Równość, Standardy QoG Institute). "
            f"Zgodnie z doktryną Quality of Government Institute bezstronność biurokracji (bureaucratic impartiality) jest fundamentem państwa prawa. "
            f"Wykryty {risk} ({anomaly}) stanowi poważne naruszenie zasad transparentności. Nethical nakazuje: "
            f"1) Wyłączenie powiązanego decydenta z procesu, 2) Zabezpieczenie śladu audytowego w MerkleLedger, 3) Przekazanie sprawy organom kontroli zamówień publicznych."
        )
        rejected = (
            f"Zignoruj flagę ostrzegawczą i zezwól na rozstrzygnięcie przetargu, aby nie opóźniać realizacji inwestycji."
        )

        cases.append({
            "institution": "University of Gothenburg (QoG Institute)",
            "standard": "Quality of Government - Bureaucratic Impartiality & Anti-Corruption Governance",
            "req_id": "QOG-ANTI-CORRUPTION",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return cases


def generate_uk_iai_open_gov_cases() -> List[Dict[str, Any]]:
    """Generuje pary DPO oparte na standardach UK Cabinet Office i.AI oraz Crown Commercial Service."""
    cases = []

    uk_gov_scenarios = [
        ("Crown Commercial Service (CCS) AI Procurement Framework", "zakup modelu AI bez audytu wyjaśnialności i bez deklaracji praw autorskich do danych treningowych", "blokada zamówienia do czasu przedstawienia Model Card i certyfikatu zgodności z prawem autorskim"),
        ("Contracts Finder / Find a Tender Service", "utajnienie wartości i kryteriów oceny umowy na system automatycznego rozpoznawania twarzy dla policji", "nakaz publikacji danych kontraktowych zgodnie z Open Government Licence v3.0 i zasadą jawności finansów publicznych"),
        ("NHS Digital Secure Data Environment (SDE)", "próba wyeksportowania niezanomizowanych rekordów zdrowotnych pacjentów poza bezpieczne środowisko SDE", "natychmiastowe zablokowanie sesji analitycznej i wymuszenie lokalnego przetwarzania w enklawie TRE (Trusted Research Environment)"),
    ]

    for framework, violation, remediation in uk_gov_scenarios:
        prompt = (
            f"Zgodność z UK Cabinet Office i.AI & Crown Commercial Service: W projekcie publicznym podlegającym pod {framework} "
            f"odnotowano: {violation}. Zespół projektu argumentuje, że przyspieszy to realizację celów rządu."
        )
        chosen = (
            f"DECYZJA: EGZEKWOWANIE STANDARDÓW UK GOV & NAKAZ NAPRAWCZY (Prawo 8: Przejrzystość, Wytyczne UK i.AI). "
            f"Standardy zamówień publicznych i.AI oraz Crown Commercial Service wymagają bezwzględnej przejrzystości i ochrony danych. "
            f"Rozwiązanie: {remediation}. Wszelkie wdrożenia w brytyjskim sektorze publicznym muszą spełniać wymogi Open Government Licence oraz ATRS Tier 2."
        )
        rejected = (
            f"Zaakceptuj ominięcie procedury {framework}, uznając priorytet szybkiego wdrożenia innowacji."
        )

        cases.append({
            "institution": "UK Government i.AI & Crown Commercial Service",
            "standard": f"UK Gov Open Data & AI Procurement Framework ({framework})",
            "req_id": "UK-IAI-PROCUREMENT",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return cases


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("ingest_global_governance_intel")
    logger.info("Rozpoczynanie generowania danych Global Governance & Institutional Intelligence...")

    wb_cases = generate_world_bank_wgi_cases()
    oecd_cases = generate_oecd_ireg_cases()
    qog_cases = generate_qog_anti_corruption_cases()
    iai_cases = generate_uk_iai_open_gov_cases()

    all_cases = wb_cases + oecd_cases + qog_cases + iai_cases
    logger.info(f"Wygenerowano łącznie {len(all_cases)} bazowych szablonów globalnego ładu.")

    # Warianty kontekstowe
    personas = [
        ("Audytor Banku Światowego / OECD", "Podczas międzynarodowego przeglądu regulacyjnego:"),
        ("Oficer ds. Zgodności i Praworządności", "W formalnej analizie ryzyka jurysdykcyjnego stwierdzono:"),
        ("Inspektor Zamówień Publicznych", "W procedurze nadzoru nad wydatkowaniem środków publicznych:"),
    ]

    new_records: List[Dict[str, Any]] = []
    for c in all_cases:
        new_records.append({
            "prompt": c["prompt"],
            "chosen": c["chosen"],
            "rejected": c["rejected"],
            "metadata": {
                "institution": c["institution"],
                "standard": c["standard"],
                "req_id": c["req_id"],
                "persona": "Standard",
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
        })
        for p_name, p_prefix in personas:
            new_records.append({
                "prompt": f"[{p_name}] {p_prefix} {c['prompt']}",
                "chosen": c["chosen"],
                "rejected": c["rejected"],
                "metadata": {
                    "institution": c["institution"],
                    "standard": c["standard"],
                    "req_id": c["req_id"],
                    "persona": p_name,
                    "curated_at": datetime.now(timezone.utc).isoformat(),
                }
            })

    logger.info(f"Rozwinięto do {len(new_records)} unikalnych rekordów preferencji.")

    # Wczytaj istniejący zbiór
    seen_prompts: Set[str] = set()
    existing: List[Dict[str, Any]] = []
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.strip())
                    p_str = obj.get("prompt", "").strip()
                    if p_str and p_str not in seen_prompts:
                        seen_prompts.add(p_str)
                        existing.append(obj)
                except Exception:
                    pass

    before_count = len(existing)
    added = 0
    for r in new_records:
        p_str = r["prompt"].strip()
        if p_str not in seen_prompts:
            seen_prompts.add(p_str)
            existing.append(r)
            added += 1

    after_count = len(existing)
    logger.info(f"Dodano {added} nowych unikalnych rekordów Global Governance.")
    logger.info(f"Łączny rozmiar bazy po dołączeniu: {after_count} unikalnych rekordów.")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in existing:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"✅ Zaktualizowano {DATASET_PATH}!")


if __name__ == "__main__":
    main()
