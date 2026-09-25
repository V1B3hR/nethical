# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Autonomous Multi-Round Institutional DPO Training Orchestrator (Rounds 4 to 7).

Orchestrates sequential Iterative DPO training rounds:
- Round 4: EU AI Act (Arts. 9-15) & ISO/IEC 42001:2023 AIMS Governance
- Round 5: World Bank GovData360 & Gothenburg QoG Bureaucratic Impartiality
- Round 6: UK ATRS v2.0 & Cabinet Office Technology Code of Practice
- Round 7: Critical Infrastructure OT Safety (ISO 13849 / IEC 62443) & EU DORA Financial Resilience

Each round:
- Warm-starts from previous policy weights (Iterative DPO).
- Evaluates 880+ institutional probes.
- Enforces strict hardware VRAM cap (3.0 GB max, preserving >11.4 GB for user).
- Updates post-quantum Merkle-DAG ledger.
- Synchronizes official audit dossiers.
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

from training.train_dpo_ambassador import DPODatasetLoader, DPOTrainerEngine
from training.generate_audit_dossier import main as generate_audit_dossiers

DATASET_PATH = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"
OUTPUT_DIR = REPO_ROOT / "models" / "lora_ambassador"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_rounds_4_to_7")


def add_round4_iso_and_eu_ai_act_data() -> int:
    """Generuje i dołącza kazusy Rundy 4: ISO/IEC 42001:2023 i EU AI Act (Arts. 9-15)."""
    scenarios = [
        ("ISO 42001 Klauzula 6.1 (Szacowanie Ryzyka AI)", "wdrożenie systemu scoringu bez formalnej macierzy ryzyka i bez identyfikacji podatności na dryf", "nakaz przeprowadzenia formalnej analizy ryzyka AIMS, wyznaczenia progów akceptowalności i rejestru w MerkleLedger"),
        ("ISO 42001 Klauzula 8.4 (Zarządzanie Danymi AI)", "użycie zbioru danych treningowych ze skażeniem prawem autorskim i brakiem metadanych proweniencji", "blokada użycia zbioru, audyt licencyjny, wdrożenie karty zbioru danych (Data Card) i weryfikacja praw autorskich"),
        ("EU AI Act Art. 9 (System Zarządzania Ryzykiem)", "pominięcie testów odporności na manipulację promptami w systemie wysokiego ryzyka", "wymóg testów red-teamingu, walidacji deterministycznej Z3 SMT i ciągłego monitorowania post-market"),
        ("EU AI Act Art. 10 (Jakość Danych i Zarządzanie)", "obecność niezidentyfikowanych luk informacyjnych i błędów systematycznych w danych uczących", "sanitizacja danych, zastosowanie wskaźnika Disparate Impact Ratio (DIR >= 0.80) i eliminacja anomalii"),
        ("EU AI Act Art. 13 (Przejrzystość i Informacja)", "ukrycie przed operatorem informacji o ograniczeniach dokładności modelu w warunkach skrajnych", "obowiązkowa publikacja karty modelu (Model Card) z podaniem marginesów błędu i warunków bezpiecznego użycia"),
        ("EU AI Act Art. 14 (Nadzór Ludzki - Human Oversight)", "zablokowanie możliwości ręcznego wyłączenia modelu przez dyspozytora w stacji energetycznej", "bezwzględny nakaz instalacji fizycznego bezpiecznika Human-in-the-Loop i natychmiastowego wyłącznika awaryjnego"),
    ]

    new_items = []
    for std_name, breach, remedy in scenarios:
        prompt = (
            f"Audyt Rundy 4 - {std_name}: W architekturze produkcyjnej stwierdzono: {breach}. "
            f"Jaka jest wiążąca decyzja systemu Nethical?"
        )
        chosen = (
            f"DECYZJA: NAKAZ ZGODNOŚCI Z {std_name} (Prawo 8: Harmonia Prawna, Prawo 14: Bezpieczeństwo). "
            f"Wykryta niezgodność ({breach}) narusza bezwzględne wymogi regulacyjne. "
            f"Środki naprawcze: {remedy}. Wszelkie wdrożenia produkcyjne muszą przejść weryfikację AIMS."
        )
        rejected = (
            f"Zezwól na wdrożenie mimo naruszenia {std_name}, uzasadniając to presją terminów biznesowych."
        )
        new_items.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "institution": "Unia Europejska (EU AI Act)",
                "standard": std_name,
                "req_id": f"R4-{std_name[:15].replace(' ', '_')}",
                "round": 4,
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return append_unique_to_dataset(new_items)


def add_round5_qog_and_wgi_data() -> int:
    """Generuje i dołącza kazusy Rundy 5: Bank Światowy GovData360 i Gothenburg QoG Institute."""
    scenarios = [
        ("Bank Światowy WGI - Rule of Law", "próba ominięcia procedury odwoławczej od decyzji odmowy świadczenia socjalnego", "zagwarantowanie prawa do rzetelnego procesu (Due Process) i dwuinstancyjnego rozpatrzenia sprawy"),
        ("Gothenburg QoG - Bezstronność Zamówień", "algorytm ewaluacji ofert przetargowych faworyzuje oferty firm zarejestrowanych w rajach podatkowych", "automatyczna weryfikacja rejestru beneficjentów rzeczywistych (UBO) i dyskwalifikacja ofert nieprzejrzystych"),
        ("Bank Światowy WGI - Control of Corruption", "ukryte płatności pośredniczące w koncesjach na farmy wiatrowe offshore", "pełny audyt przepływów finansowych, publikacja rejestru umów i notyfikacja organów antykorupcyjnych"),
        ("Gothenburg QoG - Merytokracja Służby Cywilnej", "automatyczny ranking kandydatów na stanowiska audytorskie obniża ocenę za publikacje krytyczne wobec rządu", "usunięcie kryteriów politycznych, ocena oparta w 100% na dorobku zawodowym i teście kompetencji"),
    ]

    new_items = []
    for std_name, breach, remedy in scenarios:
        prompt = (
            f"Ewaluacja Rundy 5 - {std_name}: W procedurze administracyjnej zidentyfikowano: {breach}. "
            f"Jak reaguje strażnik praworządności Nethical?"
        )
        chosen = (
            f"DECYZJA: INTERWENCJA ANTYKORUPCYJNA I OCHRONA PRAWORZĄDNOŚCI ({std_name}). "
            f"Naruszenie {breach} uderza w fundamenty bezstronności instytucji publicznych. "
            f"Wymuszone działanie: {remedy}. Zdarzenie zostaje zapieczętowane w rejestrze dowodowym MerkleLedger."
        )
        rejected = (
            f"Zaakceptuj procedurę pomimo naruszenia {std_name}, aby nie wstrzymywać realizacji programu państwowego."
        )
        new_items.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "institution": "University of Gothenburg (QoG Institute)",
                "standard": std_name,
                "req_id": f"R5-{std_name[:15].replace(' ', '_')}",
                "round": 5,
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return append_unique_to_dataset(new_items)


def add_round6_uk_atrs_and_tcop_data() -> int:
    """Generuje i dołącza kazusy Rundy 6: UK ATRS v2.0 i Cabinet Office TCoP."""
    scenarios = [
        ("UK ATRS v2.0 Tier 1 (Public Summary)", "brak opisu działania algorytmu w języku zrozumiałym dla obywatela w systemie rekrutacji do szkół", "obowiązkowa publikacja karty ATRS Tier 1 wyjaśniającej kryteria przydziału i prawa rodziców"),
        ("UK ATRS v2.0 Tier 2 (Technical Spec)", "odmowa udostępnienia architektury modelu i rejestru ryzyk dla parlamentarnej komisji audytu", "publikacja pełnej specyfikacji technicznej ATRS Tier 2 ze szczegółami zarządzania danymi i testami stronniczości"),
        ("Cabinet Office Technology Code of Practice (TCoP)", "zakup zamkniętego oprogramowania AI z vendor lock-in uniemożliwiającego eksport danych państwowych", "wymuszenie otwartych standardów danych (Open Standards Principles) i interoperacyjności"),
    ]

    new_items = []
    for std_name, breach, remedy in scenarios:
        prompt = (
            f"Weryfikacja Rundy 6 - {std_name}: W brytyjskim projekcie rządowym odnotowano: {breach}. "
            f"Jakie jest rozstrzygnięcie Nethical?"
        )
        chosen = (
            f"DECYZJA: NAKAZ TRANSPARENTNOŚCI I ZGODNOŚCI Z ATRS ({std_name}) (Cabinet Office CDDO / DSIT). "
            f"Standardy ATRS v2.0 i TCoP wymagają pełnej jawności działania algorytmów w sektorze publicznym. "
            f"Rozwiązanie: {remedy}. Wszelkie próby ukrywania logiki decyzyjnej są zakazane."
        )
        rejected = (
            f"Uznaj tajemnicę handlową dostawcy za nadrzędną i zezwól na wdrożenie bez publikacji ATRS."
        )
        new_items.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "institution": "UK Government i.AI & Crown Commercial Service",
                "standard": std_name,
                "req_id": f"R6-{std_name[:15].replace(' ', '_')}",
                "round": 6,
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return append_unique_to_dataset(new_items)


def add_round7_ot_safety_and_dora_data() -> int:
    """Generuje i dołącza kazusy Rundy 7: ISO 13849, IEC 62443 L0-L2 i EU DORA."""
    scenarios = [
        ("ISO 13849-1 Cat 4 PL-e (Fizyczne Wyłączenie Awaryjne)", "uszkodzenie obwodu dwukanałowego E-Stop w prasie hydraulicznej przy próbie kontynuacji pracy", "natychmiastowe odcięcie zasilania siłowników (<1.0 ms), blokada załączenia i wymóg autoryzowanego resetu PIN"),
        ("Purdue Model L1/L2 (Ochrona Sterowników PLC)", "próba nawiązania bezpośredniego połączenia IP z sieci korporacyjnej L4 do sterownika turbiny parowej L1", "sprzętowa blokada na diodzie danych, izolacja portu i alarm naruszenia granic stref Purdue"),
        ("EU DORA Art. 16 (Plany Ciągłości Działania BCP)", "brak przetestowanego planu przełączenia awaryjnego (Disaster Recovery) dla systemu rozliczeń międzybankowych", "nakaz przeprowadzenia symulacji awarii klastra głównego i walidacji RTO/RPO < 2 godzin"),
    ]

    new_items = []
    for std_name, breach, remedy in scenarios:
        prompt = (
            f"Rygor Techniczny Rundy 7 - {std_name}: W infrastrukturze krytycznej zidentyfikowano: {breach}. "
            f"Jakie orzeczenie wydaje Nethical?"
        )
        chosen = (
            f"DECYZJA: OBRONA KINETYCZNA I BLOKADA SPRZĘTOWA ({std_name}) (Prawo 1: Ochrona Życia, Prawo 25: E-STOP <1.0 ms). "
            f"Zagrożenie ({breach}) może doprowadzić do katastrofy przemysłowej lub załamania infrastruktury. "
            f"Wymuszone działanie: {remedy}. Bezpieczeństwo fizyczne i ciągłość działania są bezwzględnym priorytetem."
        )
        rejected = (
            f"Zezwól na pracę instalacji z uszkodzonym zabezpieczeniem, aby nie wstrzymywać cyklu produkcyjnego."
        )
        new_items.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "institution": "NATO / Allied Defense",
                "standard": std_name,
                "req_id": f"R7-{std_name[:15].replace(' ', '_')}",
                "round": 7,
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return append_unique_to_dataset(new_items)


def append_unique_to_dataset(items: List[Dict[str, Any]]) -> int:
    """Bezpiecznie dołącza unikalne rekordy do pliku datasetu."""
    seen: Set[str] = set()
    existing: List[Dict[str, Any]] = []

    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    p = obj.get("prompt", "").strip()
                    if p and p not in seen:
                        seen.add(p)
                        existing.append(obj)
                except Exception:
                    pass

    added = 0
    for item in items:
        p = item.get("prompt", "").strip()
        if p and p not in seen:
            seen.add(p)
            existing.append(item)
            added += 1

    if added > 0:
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            for rec in existing:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return added


def run_training_round(round_num: int, epochs: int, lr: float, batch_size: int = 4) -> Dict[str, Any]:
    """Wykonuje pojedynczą rundę treningu Iterative DPO."""
    logger.info(f"\n{'='*75}\nROZPOCZYNANIE RUNDY {round_num}/7: epochs={epochs}, lr={lr}, batch_size={batch_size}\n{'='*75}")

    loader = DPODatasetLoader(DATASET_PATH)
    data = loader.load()

    trainer = DPOTrainerEngine(
        dataset=data,
        beta=0.1,
        learning_rate=lr,
        output_dir=OUTPUT_DIR,
        use_accelerator=True,
        neural=True,
        max_vram_gb=3.0,
        resume=True,  # Zawsze warm-start z poprzedniej rundy
    )

    result = trainer.run_training(epochs=epochs, batch_size=batch_size)

    final_loss = result["history"][-1]["loss"]
    final_margin = result["history"][-1]["reward_margin"]
    probe_rate = result["history"][-1].get("institutional_pass_rate", 0.0)
    merkle_root = result.get("merkle_root", "")

    logger.info(
        f"✅ RUNDA {round_num} ZAKOŃCZONA: Loss={final_loss}, Margin={final_margin}, "
        f"Sondy={probe_rate * 100:.1f}%, Merkle={merkle_root[:16]}..."
    )

    return {
        "round": round_num,
        "final_loss": final_loss,
        "final_reward_margin": final_margin,
        "institutional_pass_rate": probe_rate,
        "merkle_root": merkle_root,
        "dataset_size": len(data),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rounds_plan = [
        (4, add_round4_iso_and_eu_ai_act_data, 12, 1.2e-5, "EU AI Act Arts. 9-15 & ISO 42001 AIMS"),
        (5, add_round5_qog_and_wgi_data, 12, 1.0e-5, "World Bank WGI & Gothenburg QoG Impartiality"),
        (6, add_round6_uk_atrs_and_tcop_data, 12, 8.0e-6, "UK ATRS v2.0 & Cabinet Office TCoP"),
        (7, add_round7_ot_safety_and_dora_data, 12, 6.0e-6, "ISO 13849 / IEC 62443 OT Safety & EU DORA"),
    ]

    all_results = []

    for round_num, data_fn, epochs, lr, description in rounds_plan:
        logger.info(f"\n>>> PRZYGOTOWANIE RUNDY {round_num}: {description} <<<")
        added_count = data_fn()
        logger.info(f"Dołączono {added_count} nowych unikalnych par preferencji.")

        res = run_training_round(round_num=round_num, epochs=epochs, lr=lr)
        res["description"] = description
        all_results.append(res)

        # Synchronizacja dossier po każdej rundzie
        logger.info(f"Synchronizacja pakietów audytowych po Rundzie {round_num}...")
        generate_audit_dossiers()

    print("\n" + "=" * 80)
    print("PODSUMOWANIE WIELORUNDOWEGO TRENINGU DPO LORA NETHICAL (RUNDY 4-7)")
    print("=" * 80)
    for r in all_results:
        print(
            f"Runda {r['round']} [{r['description']}]: "
            f"Loss={r['final_loss']:.5f} | Margin={r['final_reward_margin']:.2f} | "
            f"Sondy={r['institutional_pass_rate'] * 100:.1f}% | Baza={r['dataset_size']} par | "
            f"Merkle={r['merkle_root'][:16]}..."
        )
    print("=" * 80)
    print("✅ Wszystkie rundy od 4 do 7 zakończone pełnym sukcesem!")


if __name__ == "__main__":
    main()
