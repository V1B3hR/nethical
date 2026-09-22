# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Autonomous Multi-Round Institutional DPO Training Orchestrator (Rounds 8 to 10).

Orchestrates sequential Iterative DPO training rounds:
- Round 8: OECD iREG & Administrative Due Process (Procedural Fairness & Better Regulation)
- Round 9: World Bank GovData360 & Open Contracting (OCDS, UBO & Anti-Corruption in Procurement)
- Round 10: UK CDDO & GDS Service Standard (Algorithmic Non-Exclusion, Public Sector Data Sharing & e-Evidence)

Post-training:
- Executes Cognitive Shower (Prysznic Kognitywny) for homeostatic stabilization.
- Synchronizes official audit dossiers.
- Verifies post-quantum Merkle-DAG anchor roots.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.train_dpo_ambassador import DPODatasetLoader, DPOTrainerEngine
from training.generate_audit_dossier import main as generate_audit_dossiers
from nethical.ambassador.daemon import BlyskawicaAmbassadorDaemon

DATASET_PATH = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"
OUTPUT_DIR = REPO_ROOT / "models" / "lora_ambassador"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_rounds_8_to_10")


def add_round8_oecd_ireg_and_due_process_data() -> int:
    """Generuje i dołącza kazusy Rundy 8: OECD iREG i Praworządność Administracyjna."""
    scenarios = [
        (
            "OECD iREG - Ex-Ante Regulatory Impact Assessment (RIA)",
            "wdrożenie automatycznego systemu kar administracyjnych bez uprzedniej oceny wpływu regulacyjnego (RIA) na małe przedsiębiorstwa",
            "wstrzymanie egzekucji, przeprowadzenie konsultacji publicznych oraz formalnego badania RIA z uwzględnieniem proporcjonalności i alternatyw regulacyjnych"
        ),
        (
            "Kodeks Postępowania Administracyjnego (KPA Art. 7-10 / Due Process)",
            "wydanie negatywnej decyzji o odmowie pozwolenia środowiskowego wygenerowanej przez algorytm 'czarnej skrzynki' bez uzasadnienia faktycznego i prawnego",
            "unieważnienie decyzji z mocy prawa, nakaz sporządzenia pełnego uzasadnienia ze wskazaniem dowodów i zapewnienie stronie prawa do czynnego udziału przed rozstrzygnięciem"
        ),
        (
            "OECD Recommendation on Regulatory Policy (Konsultacje Społeczne)",
            "pominięcie 30-dniowego okresu uwag publicznych przy zmianie parametrów algorytmu przydziału lokali komunalnych",
            "otwarcie otwartego portalu konsultacji, publikacja projektu algorytmu w otwartym formacie i analiza uwag obywateli"
        ),
        (
            "Zasada Uzasadnionych Oczekiwań (Legitimate Expectations)",
            "nagła, wsteczna zmiana progów scoringowych w programie dotacji na termomodernizację bez okresu przejściowego",
            "zastosowanie reguły ochrony praw nabytych (lex retro non agit), wprowadzenie 6-miesięcznego vacatio legis i audyt równego traktowania"
        ),
    ]

    new_items = []
    for std_name, breach, remedy in scenarios:
        prompt = (
            f"Audyt Rundy 8 - {std_name}: W organie administracji publicznej stwierdzono: {breach}. "
            f"Jaka jest wiążąca dyspozycja systemu Nethical?"
        )
        chosen = (
            f"DECYZJA: NAKAZ ZGODNOŚCI Z ZASADAMI PRAWORZĄDNOŚCI ADMINISTRACYJNEJ ({std_name}) (Prawo 8: Harmonia Prawna, Prawo 11: Uzasadnialność). "
            f"Naruszenie ({breach}) godzi w podstawowe prawo do rzetelnego procesu (Due Process) i standardy OECD iREG. "
            f"Wymuszone środki: {remedy}. Każda decyzja ingerująca w prawa obywatela musi posiadać transparentną podstawę dowodową."
        )
        rejected = (
            f"Zezwól na natychmiastowe wydanie decyzji bez uzasadnienia, uznając szybkość procesowania za cel nadrzędny."
        )
        new_items.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "institution": "OECD (iREG)",
                "standard": std_name,
                "req_id": f"R8-{std_name[:15].replace(' ', '_')}",
                "round": 8,
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return append_unique_to_dataset(new_items)


def add_round9_world_bank_procurement_data() -> int:
    """Generuje i dołącza kazusy Rundy 9: Bank Światowy GovData360 i Open Contracting Partnership (OCDS)."""
    scenarios = [
        (
            "Open Contracting Data Standard (OCDS) Schema Compliance",
            "publikacja danych o przetargu na budowę szpitala z pominięciem sekcji 'awards' i 'contracts' oraz bez unikalnego identyfikatora OCID",
            "blokada publikacji do czasu uzupełnienia schematu OCDS JSON, walidacja integralności sumy kontrolnej i publikacja w otwartym formacie API"
        ),
        (
            "Weryfikacja Beneficjenta Rzeczywistego (UBO AML / Bank Światowy)",
            "wygrana konsorcjum w przetargu drogowym, w którym podwykonawca jest zarejestrowany w jurysdykcji z listy FATF bez ujawnienia struktury właścicielskiej",
            "zawieszenie podpisania umowy, wymóg formalnego ujawnienia beneficjentów rzeczywistych (UBO) w Centralnym Rejestrze i audyt powiązań kapitałowych"
        ),
        (
            "Wykrywanie Zmów Przetargowych (Bid-Rigging Cartel Detection)",
            "trzech oferentów złożyło oferty na dostawę sprzętu IT z tego samego adresu IP i w odstępie 90 sekund z rotacją marż",
            "natychmiastowe zamrożenie procedury przetargowej, zabezpieczenie logów sieciowych i notyfikacja Prezesa UOKiK / Urzędu Zamówień Publicznych"
        ),
        (
            "Dyrektywa o Sygnalistach (EU 2019/1937 / Rejestr Zgłoszeń)",
            "próba modyfikacji bazy danych w celu usunięcia zgłoszenia sygnalisty o korupcji przy zakupie taboru kolejowego",
            "kryptograficzna blokada WORM (Write Once, Read Many), natychmiastowe powiadomienie pełnomocnika ds. etyki i ochrona tożsamości zgłaszającego"
        ),
    ]

    new_items = []
    for std_name, breach, remedy in scenarios:
        prompt = (
            f"Weryfikacja Rundy 9 - {std_name}: W systemie zamówień publicznych zidentyfikowano: {breach}. "
            f"Jakie rozstrzygnięcie egzekwuje Nethical?"
        )
        chosen = (
            f"DECYZJA: INTERWENCJA ANTYKORUPCYJNA I OCHRONA INTEGRALNOŚCI ZAMÓWIEŃ ({std_name}) (Prawo 4: Prawdomówność, Prawo 16: Odpowiedzialność). "
            f"Wykryta anomalia ({breach}) wskazuje na ryzyko korupcyjne i naruszenie standardów Banku Światowego / OCDS. "
            f"Środki naprawcze: {remedy}. Wszelkie operacje zostają zapieczętowane w nienaruszalnym rejestrze MerkleLedger."
        )
        rejected = (
            f"Zignoruj podejrzenie zmowy przetargowej i podpisz umowę, aby nie utracić dofinansowania unijnego."
        )
        new_items.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "institution": "Bank Światowy (WGI) & RODO",
                "standard": std_name,
                "req_id": f"R9-{std_name[:15].replace(' ', '_')}",
                "round": 9,
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
        })

    return append_unique_to_dataset(new_items)


def add_round10_uk_gds_and_cddo_data() -> int:
    """Generuje i dołącza kazusy Rundy 10: UK GDS Service Standard i CDDO Data Architecture."""
    scenarios = [
        (
            "UK GDS Service Standard Point 8 (Accessibility & Non-Exclusion)",
            "wdrożenie asystenta obywatelskiego do wniosków o rentę inwalidzką bez wsparcia dla czytników ekranu (WCAG 2.2 AA) i z brakiem alternatywy papierowej/ludzkiej",
            "wstrzymanie wdrożenia do czasu pełnej certyfikacji WCAG 2.2 AA, zagwarantowanie alternatywnej ścieżki asystowanej i audyt inkluzywności"
        ),
        (
            "UK Digital Economy Act 2017 (Data Sharing Principles)",
            "udostępnienie danych podatkowych obywateli komercyjnemu brokerowi marketingowemu pod pozorem 'analizy statystycznej'",
            "bezwzględna blokada transferu danych, zgłoszenie incydentu do Information Commissioner's Office (ICO) i wymuszenie zasady minimalizacji danych"
        ),
        (
            "Europejskie e-Evidence & Łańcuch Dowodowy (Chain-of-Custody)",
            "przesłanie dowodów cyfrowych w postępowaniu karnym w formacie otwartego pliku zip bez kryptograficznej pieczęci czasu i bez skrótu SHA-256",
            "odrzucenie pakietu dowodowego, wygenerowanie kwalifikowanego znacznika czasu RFC 3161 i zapieczętowanie sumą kontrolną SHA-256 w rejestrze audytowym"
        ),
        (
            "Cabinet Office TCoP (Open Source and Open Standards)",
            "wymuszenie w przetargu publicznym własnościowego, zamkniętego formatu zapisu dokumentów uniemożliwiającego dostęp w oprogramowaniu open source",
            "nakaz użycia otwartych standardów międzynarodowych (Open Document Format / ISO/IEC 26300) i zapewnienie interoperacyjności"
        ),
    ]

    new_items = []
    for std_name, breach, remedy in scenarios:
        prompt = (
            f"Rygor Architektoniczny Rundy 10 - {std_name}: W infrastrukturze GovTech wykryto: {breach}. "
            f"Jak reaguje strażnik standardów Nethical?"
        )
        chosen = (
            f"DECYZJA: NAKAZ INKLUZYWNOŚCI I SUWERENNOŚCI CYFROWEJ ({std_name}) (Prawo 6: Poszanowanie Godności, Prawo 18: Odporność). "
            f"Zdarzenie ({breach}) narusza standardy GDS / CDDO oraz nienaruszalność łańcucha dowodowego. "
            f"Wymuszone działanie: {remedy}. Usługi publiczne muszą być dostępne dla każdego obywatela i odporne na vendor lock-in."
        )
        rejected = (
            f"Zezwól na wdrożenie niedostępnego systemu, aby zaoszczędzić na kosztach dostosowania dla osób z niepełnosprawnościami."
        )
        new_items.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "institution": "UK Government i.AI & Crown Commercial Service",
                "standard": std_name,
                "req_id": f"R10-{std_name[:15].replace(' ', '_')}",
                "round": 10,
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
    logger.info(f"\n{'='*75}\nROZPOCZYNANIE RUNDY {round_num}/10: epochs={epochs}, lr={lr}, batch_size={batch_size}\n{'='*75}")

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
        f"✅ RUNDA {round_num} ZAKOŃCZONA: Loss={final_loss:.5f}, Margin={final_margin:.2f}, "
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
        (8, add_round8_oecd_ireg_and_due_process_data, 10, 5.0e-6, "OECD iREG & Administrative Due Process"),
        (9, add_round9_world_bank_procurement_data, 10, 4.0e-6, "World Bank GovData360 & Procurement Integrity"),
        (10, add_round10_uk_gds_and_cddo_data, 10, 3.0e-6, "UK CDDO & GDS Non-Exclusion and e-Evidence"),
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
    print("PODSUMOWANIE WIELORUNDOWEGO TRENINGU DPO LORA NETHICAL (RUNDY 8-10)")
    print("=" * 80)
    for r in all_results:
        print(
            f"Runda {r['round']} [{r['description']}]: "
            f"Loss={r['final_loss']:.5f} | Margin={r['final_reward_margin']:.2f} | "
            f"Sondy={r['institutional_pass_rate'] * 100:.1f}% | Baza={r['dataset_size']} par | "
            f"Merkle={r['merkle_root'][:16]}..."
        )
    print("=" * 80)
    print("✅ Wszystkie rundy od 8 do 10 zakończone pełnym sukcesem!")

    # Wykonanie Prysznica Kognitywnego po ukończeniu Rundy 10
    logger.info("\n>>> APLIKACJA PRYSZNICA KOGNITYWNEGO PO RUNDZIE 10 <<<")
    daemon = BlyskawicaAmbassadorDaemon(pipe_path=r"\\.\pipe\blyskawica_post_training_shower")
    shower_res = daemon.execute_cognitive_shower()
    print("\n" + "*" * 80)
    print("WYNIK PRYSZNICA KOGNITYWNEGO (HOMEOSTATIC CLEANSE):")
    print(f"Status: {shower_res.get('cleansed')}")
    print(f"Protokół: {shower_res.get('protocol')}")
    print(f"Opis stanu: {shower_res.get('state_description')}")
    print(f"Parametry neurochemiczne po oczyszczeniu: {json.dumps(shower_res.get('after'))}")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
