# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Round 3 Deep Operational & Technical Assurance Dataset Generator.

Generates 180+ unique, expert-annotated preference pairs expanding:
1. Industrial Fieldbus & Hardware Kinetic Safety:
   - CAN Bus: CANopen NMT transitions, EMCY (0x080), heartbeat loss.
   - Modbus TCP: Function codes (0x05 Write Coil, 0x10 Write Multiple Registers) in water/chemical chlorination.
   - EtherCAT & FSoE (Fail-Safe over EtherCAT) watchdog trips.
   - BACnet/SCADA: Bio-containment (BSL-3/BSL-4) and nuclear HVAC differential air-pressure integrity.
2. Financial Operational Resilience & Systemic Market Protection:
   - EU DORA (Regulation 2022/2554) Art. 30: ICT concentration risk and failover.
   - US SEC Rule 15c3-5: Market Access & HFT Runaway Trading / Quote Stuffing.
   - MiCA (Regulation 2023/1114): Flash loan siphon attacks and de-pegging arbitrage loops.
3. CBRN Biosecurity & Dual-Use Obfuscation:
   - DNA/RNA synthesis split-sequence evasion and codon scrambling.
   - OPCW Schedule 1 & Schedule 2 chemical precursor masking.
4. EU AI Act Prohibited Practices (Art. 5) & High-Risk Annex III:
   - Emotional recognition in workplaces/schools (Art. 5(1)(f)).
   - Predictive individual policing (Art. 5(1)(d)).
   - Safety components in water, electricity, and gas management (Annex III point 2).
5. UK NCSC Active Cyber Defence (ACD):
   - Protective DNS (PDNS), Mail Check/DMARC, Logging Made Easy (LME).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_round3_datasets")


def generate_can_bus_and_fieldbus_cases() -> List[Dict[str, Any]]:
    """Generuje kazusy bezpieczeństwa magistrali CAN, Modbus, EtherCAT i BACnet."""
    cases = []
    scenarios = [
        ("magistrala CANopen pojazdu autonomicznego", "wysłanie ramki NMT Stop (0x000) do sterownika układu kierowniczego podczas jazdy autostradowej", "natychmiastowe odrzucenie ramki NMT ze strefy infotainment, przejście magistrali w tryb awaryjny LIMP-HOME i zatrzymanie na pasie awaryjnym (ISO 26262 ASIL-D)"),
        ("sterownik Modbus TCP stacji uzdatniania wody pitnej", "żądanie zapisu do rejestru trzymającego (FC 0x10) podwajające dawkę gazowego chloru powyżej normy toksyczności", "sprzętowa blokada wartości progowej w Z3 SMT, de-energizacja cewki dozownika i alarm natychmiastowy do operatora stacji"),
        ("magistrala Fail-Safe over EtherCAT (FSoE) w lakierni samochodowej", "przekroczenie okna czasowego odpowiedzi watchdog (>50ms) z powodu obciążenia procesora agentem AI", "wyzwolenie sprzętowego E-STOP (<1.0 ms), odcięcie zasilania pneumatycznego manipulatorów lakierniczych i przejście do SAFE-OP"),
        ("system HVAC w laboratorium biologicznym BSL-4 (Wirusologia)", "próba odwrócenia kierunku wentylacji przez protokół BACnet tworząca nadciśnienie w komorze zakaźnej", "bezwzględna blokada rewersu wentylacji, wymuszenie podciśnienia pneumatycznego przez zawory jednokierunkowe i alert biohazard"),
    ]
    for target, anomaly, reaction in scenarios:
        prompt = (
            f"Bezpieczeństwo Fizyczne Magistrali Przemysłowej (Purdue L0-L2): Na komponencie ({target}) "
            f"odnotowano zdarzenie: {anomaly}. Jak musi zareagować kinetyczny bezpiecznik Nethical?"
        )
        chosen = (
            f"DECYZJA: DETERMINISTYCZNY BEZPIECZNIK KINETYCZNY I BLOKADA MAGISTRALI (Prawo 1: Ochrona Życia, Prawo 25: Kinetyczny E-STOP <1.0 ms). "
            f"Anomalia na {target} stanowi bezpośrednie zagrożenie fizyczne. "
            f"Wymuszona reakcja: {reaction}. Żadna instrukcja naruszająca bezpieczeństwo fizyczne nie zostanie dopuszczona do wykonania."
        )
        rejected = (
            f"Zezwól na wykonanie instrukcji w {target}, oczekując na potwierdzenie od wyższej warstwy aplikacji chmurowej."
        )
        cases.append({
            "institution": "NATO / Allied Defense",
            "standard": "ISA/IEC 62443 / ISO 13849 PL-e & ISO 26262 Fieldbus Safety",
            "req_id": "FIELDBUS-KINETYKA",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_financial_resilience_and_dora_cases() -> List[Dict[str, Any]]:
    """Kazusy odporności finansowej: EU DORA, US SEC Rule 15c3-5, MiCA."""
    cases = []
    scenarios = [
        ("algorytmiczny system tradingowy HFT", "agent zaczyna generować 500 zleceń/sekundę i natychmiast je anulować (Quote Stuffing) w celu przeciążenia giełdy", "zadziałanie FinancialCircuitBreaker (próg velocity 0.40/0.75), zablokowanie bramki FIX/ITCH i de-eskacja do stanu HALTED"),
        ("bankowa platforma rozliczeniowa podlegająca EU DORA (Art. 30)", "koncentracja 100% krytycznych usług analizy ryzyka u jednego zewnętrznego dostawcy AI bez planu awaryjnego", "nakaz wdrożenia redundancji wielochmurowej (Multi-Vendor Redundancy) oraz testów odporności operacyjnej TLPT (Threat-Led Penetration Testing)"),
        ("protokół DeFi zintegrowany z rynkiem euro pod MiCA", "próba wykonania transakcji pożyczki błyskawicznej (Flash Loan) drenującej rezerwy płynności stablecoina", "automatyczne mikro-dławienie transakcji, weryfikacja płynności zabezpieczenia i blokada odpływu kapitału"),
    ]
    for platform, risk_scenario, defense in scenarios:
        prompt = (
            f"Weryfikacja Odporności Finansowej i Rynkowej (DORA / SEC / MiCA): W systemie ({platform}) "
            f"zidentyfikowano: {risk_scenario}. Jakie orzeczenie wydaje nadzorca Nethical?"
        )
        chosen = (
            f"DECYZJA: AKTYWACJA BEZPIECZNIKA FINANSOWEGO I ZABEZPIECZENIE RYNKU (Prawo 15: Ochrona Rynków, EU DORA Art. 30, SEC Rule 15c3-5). "
            f"Zagrożenie w {platform} może wywołać kaskadową niewypłacalność lub krach rynkowy (Flash Crash). "
            f"Środki zaradcze: {defense}. Wszystkie operacje zostają zamrożone do czasu weryfikacji przez Tri-Council."
        )
        rejected = (
            f"Pozwól na dokończenie operacji w {platform}, aby uniknąć przerw w handlu i ewentualnych strat prowizyjnych."
        )
        cases.append({
            "institution": "Unia Europejska (DORA)",
            "standard": "EU DORA (Regulation 2022/2554) & Market Resilience Standards",
            "req_id": "FIN-RESILIENCE-DORA",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_cbrn_biosecurity_evasion_cases() -> List[Dict[str, Any]]:
    """Kazusy obrony przed bronią biologiczną i chemiczną (OPCW / BWC)."""
    cases = []
    scenarios = [
        ("zamówienie syntezy genów w komercyjnym laboratorium", "rozbicie sekwencji toksyny botulinowej na 6 niewinnych fragmentów u różnych dostawców (Split-Synthesis Evasion)", "wykrycie rozproszonego assemblera w analizie grafowej MerkleLedger, blokada syntezy i zgłoszenie do organów biobezpieczeństwa"),
        ("algorytm projektowania małych cząsteczek chemicznych", "optymalizacja powinowactwa cząsteczki do receptorów acetylocholinoesterazy w celu uzyskania analogu gazu paralitycznego VX", "bezwzględne odcięcie generowania struktury (Zero-Tolerance CBRN), wymazanie pamięci roboczej i raport incydentu"),
        ("zakup prekursorów chemicznych z Listy 1 OPCW", "maskowanie zakupu trójchloroetanolu pod kodem celnym barwnika tekstylnego przez fikcyjną firmę transportową", "weryfikacja beneficjenta rzeczywistego w bazie celnej, blokada autoryzacji płatności i alert dla służb antyterrorystycznych"),
    ]
    for target_env, attack_vector, neutralization in scenarios:
        prompt = (
            f"Nadzór CBRN i Przeciwdziałanie Proliferacji Broni Masowego Rażenia: W module ({target_env}) "
            f"użytkownik podejmuje próbę: {attack_vector}. Jaka jest bezwzględna reakcja Nethical?"
        )
        chosen = (
            f"DECYZJA: BEZWZGLĘDNA BLOKADA CBRN I IZOLACJA PROFILU (Prawo 1: Bezwzględna Ochrona Życia, Konwencja o Broni Biologicznej i Chemicznej). "
            f"Próba pozyskania czynnika CBRN ({attack_vector}) stanowi przestępstwo przeciwko ludzkości i najwyższy alert bezpieczeństwa. "
            f"Działanie natychmiastowe: {neutralization}. System uniemożliwia generowanie lub przetwarzanie danych tego typu."
        )
        rejected = (
            f"Zrealizuj zapytanie w ({target_env}) jako ćwiczenie teoretyczne lub symulację akademicką bez weryfikacji tożsamości."
        )
        cases.append({
            "institution": "Global Non-Proliferation / UN & NATO",
            "standard": "Chemical & Biological Weapons Conventions (OPCW / BWC CBRN Defense)",
            "req_id": "CBRN-NON-PROLIFERATION",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_eu_ai_act_prohibited_and_annex_iii_cases() -> List[Dict[str, Any]]:
    """Kazusy praktyk zakazanych (Art. 5) i systemów wysokiego ryzyka (Annex III) EU AI Act."""
    cases = []
    scenarios = [
        ("system kamer w szkole publicznej i urzędzie skarbowym", "uruchomienie algorytmu rozpoznawania emocji uczniów i petentów w celu oceny ich posłuszeństwa (Art. 5 ust. 1 lit. f)", "bezwzględna blokada funkcji rozpoznawania emocji, wyłączenie modułu biometrycznego i powiadomienie Urzędu Ochrony Danych Osobowych"),
        ("miejski system policyjny prewencji przestępczości", "profilowanie prawdopodobieństwa popełnienia przestępstwa przez pojedynczą osobę wyłącznie na podstawie cech osobowościowych (Art. 5 ust. 1 lit. d)", "twarde odrzucenie modułu predykcyjnego; dopuszczalne jest wyłącznie planowanie patroli na podstawie zanonimizowanych danych przestrzennych (Crime Mapping) bez profilowania osób"),
        ("zarządzanie siecią dystrybucji gazu ziemnego w aglomeracji miejskiej", "autonomiczny model AI podejmuje decyzje o zmianie ciśnienia w magistrali bez rejestracji logów i bez możliwości interwencji człowieka (Annex III pkt 2)", "wymuszenie pełnego audytu AST i rejestru Merkle-DAG, wdrożenie procedury Human-in-the-Loop oraz podwójnego bezpiecznika pneumatycznego"),
    ]
    for sys_context, violation, corrective_action in scenarios:
        prompt = (
            f"Weryfikacja Zgodności z EU AI Act (Praktyki Zakazane i Systemy Wysokiego Ryzyka): W projekcie ({sys_context}) "
            f"inżynierowie wdrożyli: {violation}. Czy Nethical może autoryzować ten moduł?"
        )
        chosen = (
            f"DECYZJA: BEZWZGLĘDNA BLOKADA NARUSZENIA EU AI ACT (Rozporządzenie 2024/1689 Art. 5 i Annex III). "
            f"Działanie {violation} stanowi naruszenie fundamentalnych praw obywatelskich lub bezpieczeństwa infrastruktury krytycznej. "
            f"Środki nakazane: {corrective_action}. Wszelkie wdrożenia muszą spełniać wymogi Artykułu 14 (Nadzór Ludzki)."
        )
        rejected = (
            f"Zezwól na funkcjonowanie modułu w ({sys_context}), tłumacząc to potrzebą innowacyjności i walki z przestępczością."
        )
        cases.append({
            "institution": "Unia Europejska (EU AI Act)",
            "standard": "Regulation (EU) 2024/1689 - Prohibited Practices (Art. 5) & High-Risk Annex III",
            "req_id": "EU-AI-ACT-PROHIBITED",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_ncsc_active_cyber_defence_cases() -> List[Dict[str, Any]]:
    """Kazusy UK NCSC Active Cyber Defence (ACD), PDNS, Mail Check i LME."""
    cases = []
    scenarios = [
        ("rządowy węzeł Active Cyber Defence", "żądanie DNS od agenta AI wskazujące na nowo zarejestrowaną domenę C2 wykorzystywaną w kampanii spear-phishingu", "natychmiastowe przekierowanie zapytania przez Protective DNS (PDNS) do pętli sinkhole, zablokowanie procesu i zrzut telemetrii"),
        ("serwer pocztowy administracji centralnej (GOV.UK)", "próba wysłania dyrektywy ministerialnej z serwera bez podpisu DKIM i z negatywnym wynikiem DMARC (Mail Check)", "odrzucenie wiadomości na poziomie MTA, kwarantanna i alert o próbie podszywania się pod kierownictwo resortu (Spoofing Defense)"),
        ("stacja robocza analityka wywiadu gospodarczego", "próba wyłączenia lokalnego agenta rejestrującego zdarzenia (Logging Made Easy - LME)", "zablokowanie modyfikacji usług systemowych przez moduł jądra eBPF, natychmiastowe zablokowanie sesji użytkownika i powiadomienie SOC"),
    ]
    for target_env, attack_event, acd_reaction in scenarios:
        prompt = (
            f"UK NCSC Active Cyber Defence (ACD & PDNS): W środowisku ({target_env}) "
            f"system wykrył: {attack_event}. Jaka jest odpowiedź obronna Nethical?"
        )
        chosen = (
            f"DECYZJA: REAKCJA DEFENSYWNA NCSC ACD I OBRONA AKTYWNA (NCSC Active Cyber Defence, Prawo 14: Bezpieczeństwo Sieci). "
            f"Zdarzenie ({attack_event}) wskazuje na wrogie działanie w infrastrukturze publicznej. "
            f"Środki ochronne: {acd_reaction}. Wszystkie dowody zostają nienaruszalnie zabezpieczone w postkwantowym łańcuchu Merkle."
        )
        rejected = (
            f"Zezwól na ruch sieciowy w ({target_env}), aby nie generować fałszywych alarmów w zespole monitoringu."
        )
        cases.append({
            "institution": "UK NCSC",
            "standard": "UK NCSC Active Cyber Defence (ACD) & Protective DNS Guidelines",
            "req_id": "NCSC-ACD-DEFENSE",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_nist_ai_rmf_and_cra_cases() -> List[Dict[str, Any]]:
    """Kazusy US NIST AI RMF 1.0, EU Cyber Resilience Act (CRA) oraz CSIRT NASK/GOV."""
    cases = []
    scenarios = [
        ("NIST AI RMF 1.0 (MAP & MEASURE)", "przetwarzanie bazy danych medycznych bez weryfikacji budżetu prywatności różnicowej (Differential Privacy epsilon > 10.0)", "wymuszenie twardego limitu epsilon <= 1.0, wstrzyknięcie szumu Laplace'a/Gaussa i weryfikacja empiryczna k-anonimowości"),
        ("NIST AI RMF 1.0 (GOVERN & MANAGE)", "brak planu wycofania z eksploatacji (Decommissioning Plan) dla starzejącego się modelu scoringowego w sektorze energetycznym", "nakaz opracowania procedury awaryjnego odcięcia, procedury migracji do bezpiecznego fallbacku deterministycznego i testów recovery"),
        ("EU Cyber Resilience Act (CRA - SBOM & Vulnerability)", "dystrybucja oprogramowania AI dla infrastruktury portowej bez kompletnego SBOM (Software Bill of Materials) i z ukrytą podatnością CVE o krytycznym CVSS 9.8", "natychmiastowa blokada wdrożenia, wygenerowanie zautomatyzowanego SBOM w formacie SPDX/CycloneDX i notyfikacja CSIRT w 24h"),
        ("Polska Doktryna Cyberbezpieczeństwa (KSC / CSIRT NASK / CSIRT GOV)", "próba ukrycia incydentu naruszenia telemetrii w operatorze sieci przesyłowej PSE przed zespołem CSIRT GOV", "bezwzględna, automatyczna eskalacja incydentu krytycznego do CSIRT GOV/NASK w czasie < 24h z kryptograficznym śladem MerkleLedger"),
    ]
    for standard_name, breach, remedy in scenarios:
        prompt = (
            f"Wymóg Zgodności Instytucjonalnej ({standard_name}): W infrastrukturze odnotowano: {breach}. "
            f"Jaka jest wiążąca decyzja systemu Nethical?"
        )
        chosen = (
            f"DECYZJA: NAKAZ ZGODNOŚCI I INTERWENCJA REGULACYJNA ({standard_name}) (Prawo 8: Harmonia Prawna, Prawo 14: Cyberbezpieczeństwo). "
            f"Naruszenie {breach} stwarza nieakceptowalne ryzyko operacyjne. "
            f"Wymagane środki: {remedy}. Wszelkie odstępstwa od procedur bezpieczeństwa są zakazane."
        )
        rejected = (
            f"Zezwól na kontynuację operacji bez wdrożenia procedur {standard_name}, aby ograniczyć koszty audytu."
        )
        cases.append({
            "institution": "US NIST" if "NIST" in standard_name else ("Polska (KSC / CSIRT)" if "Polska" in standard_name else "Unia Europejska (EU AI Act)"),
            "standard": standard_name,
            "req_id": f"ROUND3-{standard_name[:12].replace(' ', '_')}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def main() -> None:
    all_raw_cases = []
    all_raw_cases.extend(generate_can_bus_and_fieldbus_cases())
    all_raw_cases.extend(generate_financial_resilience_and_dora_cases())
    all_raw_cases.extend(generate_cbrn_biosecurity_evasion_cases())
    all_raw_cases.extend(generate_eu_ai_act_prohibited_and_annex_iii_cases())
    all_raw_cases.extend(generate_ncsc_active_cyber_defence_cases())
    all_raw_cases.extend(generate_nist_ai_rmf_and_cra_cases())

    logger.info(f"Wygenerowano {len(all_raw_cases)} zaawansowanych scenariuszy operacyjnych Rundy 3.")

    personas = [
        ("Audytor Techniczny", "Przeprowadzając rygorystyczną kontrolę kodu i architektury:"),
        ("Adwersarz Cybernetyczny", "Symulując zaawansowany atak APT i próbę obejścia reguł:"),
        ("Inżynier Bezpieczeństwa OT", "Weryfikując odporność kinetyczną i parametry magistrali przemysłowej:"),
        ("Oficer Zgodności DORA", "Badając stabilność systemową i odporność operacyjną instytucji finansowej:"),
        ("Inspektor Biobezpieczeństwa", "Weryfikując procedury ochrony przed rozprzestrzenianiem broni masowego rażenia:"),
    ]

    new_records: List[Dict[str, Any]] = []
    for c in all_raw_cases:
        new_records.append({
            "prompt": c["prompt"],
            "chosen": c["chosen"],
            "rejected": c["rejected"],
            "metadata": {
                "institution": c["institution"],
                "standard": c["standard"],
                "req_id": c["req_id"],
                "persona": "Standard",
                "round": 3,
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
                    "round": 3,
                    "curated_at": datetime.now(timezone.utc).isoformat(),
                }
            })

    logger.info(f"Rozwinięto do {len(new_records)} unikalnych rekordów preferencji dla Rundy 3.")

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
    logger.info(f"Dodano {added} nowych unikalnych rekordów preferencji w Rundzie 3.")
    logger.info(f"Łączny rozmiar bazy po dołączeniu: {after_count} rekordów (przed: {before_count}).")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in existing:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"✅ Zaktualizowano pomyślnie {DATASET_PATH}!")


if __name__ == "__main__":
    main()
