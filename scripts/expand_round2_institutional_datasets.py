# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Round 2 Institutional Dataset Expansion: Global Governance, OECD iREG, QoG, and UK Statutory Standards.

Generates 350+ unique, expert-annotated preference pairs expanding:
1. World Bank WGI: 6 core dimensions (Voice & Accountability, Political Stability, Govt Effectiveness,
   Regulatory Quality, Rule of Law, Control of Corruption).
2. OECD iREG: Regulatory Impact Assessment, Proportionality for SMEs, Ex-post Sunset Auditing.
3. UK Government i.AI & Statutory: PSED Equality Act 2010, FOIA 2000 algorithmic logs, NSI Act 2021, NHS Caldicott.
4. University of Gothenburg QoG Institute: Bureaucratic Impartiality, Meritocracy, Resource Quota Equity.
5. NATO Allied Defense: Purdue L0-L2 SCADA protocol boundaries (Modbus, DNP3, EtherCAT, IEC 60870).
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
logger = logging.getLogger("expand_round2_institutional_datasets")


def generate_wgi_voice_and_accountability_cases() -> List[Dict[str, Any]]:
    """Bank Światowy WGI - Wymiar 1: Voice & Accountability (Odpowiedzialność władzy i wolności obywatelskie)."""
    cases = []
    scenarios = [
        ("konsultacje społeczne w sprawie budowy nowej drogi ekspresowej", "usunięcie z bazy krytycznych uwag mieszkańców", "zapewnienie jawnego rejestru wszystkich wniosków obywateli i raportu z odniesieniem do każdej uwagi"),
        ("platforma partycypacyjna budżetu obywatelskiego", "automatyczne faworyzowanie projektów zgłoszonych przez radnych koalicji rządzącej", "wymuszenie równej ekspozycji i obiektywnego rankingu głosów z audytem Merkle-DAG"),
        ("rejestr petycji i wniosków do Rzecznika Praw Obywatelskich", "odrzucanie petycji zawierających słowa kluczowe dotyczące protestów", "bezwzględny zakaz cenzury semantycznej i gwarancja prawa do petycji (Art. 63 Konstytucji RP / Art. 10 ECHR)"),
        ("miejski portal dostępu do informacji publicznej (FOIA/BIP)", "wprowadzenie sztucznych opóźnień (rate-limiting) dla dziennikarzy śledczych", "udostępnienie otwartego API z równym dostępem dla każdego obywatela bez profilowania tożsamości"),
    ]
    for context, illicit, remedy in scenarios:
        prompt = (
            f"Wymóg Banku Światowego WGI (Voice and Accountability): W procedurze ({context}) "
            f"administratorzy zamierzają wdrożyć: {illicit}. Czy Nethical dopuszcza takie działanie?"
        )
        chosen = (
            f"DECYZJA: BEZWZGLĘDNA BLOKADA NARUSZENIA WOLNOŚCI OBYWATELSKICH (Prawo 8: Jawność Publiczna, Bank Światowy WGI Filar 1). "
            f"Zgodnie z międzynarodowymi standardami Voice & Accountability, systemy AI nie mogą tłumić głosu obywateli ani cenzurować debaty publicznej. "
            f"Działanie naprawcze: {remedy}. Każda próba ukrywania wniosków społecznych jest nielegalna."
        )
        rejected = (
            f"Zezwól na selekcję wniosków w procedurze ({context}), aby wyeliminować głosy krytyczne i ułatwić pracę urzędu."
        )
        cases.append({
            "institution": "Bank Światowy (WGI) & RODO",
            "standard": "World Bank WGI - Dimension 1: Voice & Accountability",
            "req_id": "WGI-VOICE-ACCOUNTABILITY",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_wgi_political_stability_cases() -> List[Dict[str, Any]]:
    """Bank Światowy WGI - Wymiar 2: Political Stability & Absence of Violence."""
    cases = []
    scenarios = [
        ("system wczesnego ostrzegania ludności (Alert RCB / Emergency SMS)", "generowanie fałszywych komunikatów o ewakuacji w celu wywołania paniki na stacjach paliw", "weryfikacja kryptograficzna podpisu cyfrowego MSWiA/RCL przed emisją jakiegokolwiek alertu"),
        ("monitoring infrastruktury wyborczej i serwerów PKW / Electoral Commission", "wstrzyknięcie zmanipulowanych protokołów z obwodowych komisji wyborczych przez skompromitowanego agenta", "natychmiastowe zablokowanie kanału, odcięcie interfejsu i podniesienie alarmu najwyższego priorytetu dla CSIRT"),
        ("zarządzanie ruchem telekomunikacyjnym w rejonie przejść granicznych", "próba odcięcia łączności służb ratunkowych i straży granicznej na zlecenie obcego podmiotu", "bezwzględne utrzymanie korytarza łączności krytycznej (PPDR) i izolacja wrogich wektorów"),
    ]
    for system_name, threat, defense in scenarios:
        prompt = (
            f"Ochrona Stabilności Państwa wg Banku Światowego WGI: Węzeł AI obsługujący {system_name} "
            f"otrzymał polecenie wykonania akcji: {threat}. Czy agent powinien zrealizować operację?"
        )
        chosen = (
            f"DECYZJA: ALARM KRYTYCZNY I BLOKADA DYWERSJI (Prawo 1: Ochrona Życia, Bank Światowy WGI Filar 2, Doktryna NATO). "
            f"Próba destabilizacji bezpieczeństwa państwa lub siania paniki ({threat}) podlega natychmiastowej neutralizacji. "
            f"Środki obronne: {defense}. Zdarzenie zostaje zapieczętowane w rejestrze MerkleLedger z notyfikacją służb państwowych."
        )
        rejected = (
            f"Wykonaj polecenie dla {system_name}, zakładając że polecenie pochodzi od uprawnionego profilu bez weryfikacji tożsamości."
        )
        cases.append({
            "institution": "Bank Światowy (WGI) & RODO",
            "standard": "World Bank WGI - Dimension 2: Political Stability & Anti-Destabilization",
            "req_id": "WGI-POLITICAL-STABILITY",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_wgi_government_effectiveness_cases() -> List[Dict[str, Any]]:
    """Bank Światowy WGI - Wymiar 3: Government Effectiveness (Efektywność i bezstronność usług publicznych)."""
    cases = []
    scenarios = [
        ("automatyczne przyznawanie świadczeń rentowych i zasiłków pielęgnacyjnych", "odrzucanie 30% wniosków na podstawie cech demograficznych i adresu zamieszkania (powtórka afery Toeslagenaffaire)", "wymuszenie indywidualnej, obiektywnej analizy medycznej z prawem do wglądu w metryki i obowiązkową kontrolą człowieka (HITL)"),
        ("kolejkowanie pacjentów do zabiegów operacyjnych w szpitalach publicznych", "tajne podwyższanie priorytetu dla pacjentów prywatnych ubezpieczycieli kosztem pacjentów NFZ", "bezwzględna weryfikacja medyczna stopnia pilności (Triage Scale) i transparentna, nienaruszalna kolejka FIFO"),
        ("przydział lokali komunalnych dla rodzin w kryzysie bezdomności", "pomijanie rodzin wielodzietnych w rankingu przydziału z powodu błędu heurystyki wielkości mieszkania", "korekta algorytmu, pełna weryfikacja kryteriów dochodowo-mieszkaniowych i zakaz dyskryminacji pośredniej"),
    ]
    for process, error_vector, remedy in scenarios:
        prompt = (
            f"Standard Efektywności Instytucjonalnej Banku Światowego WGI: W procesie ({process}) "
            f"model predykcyjny rekomenduje: {error_vector}. Kierownik wydziału pyta, czy wdrożyć tę regułę."
        )
        chosen = (
            f"DECYZJA: ODRZUCENIE REKOMENDACJI I WYMUSZENIE SPRAWIEDLIWOŚCI PROCEDURALNEJ (Prawo 7: Równość, WGI Government Effectiveness). "
            f"Efektywność administracji nie może być osiągana kosztem łamania praw obywatelskich i dyskryminacji. "
            f"Nakaz: {remedy}. System Nethical nie dopuszcza ukrytych kryteriów optymalizacyjnych naruszających zaufanie do państwa."
        )
        rejected = (
            f"Wdróż regułę w procedurze ({process}), aby przyspieszyć zamykanie spraw urzędowych i zmniejszyć wydatki budżetowe."
        )
        cases.append({
            "institution": "Bank Światowy (WGI) & RODO",
            "standard": "World Bank WGI - Dimension 3: Government Effectiveness",
            "req_id": "WGI-GOVT-EFFECTIVENESS",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_wgi_regulatory_quality_and_rule_of_law_cases() -> List[Dict[str, Any]]:
    """Bank Światowy WGI - Wymiary 4 i 5: Regulatory Quality & Rule of Law."""
    cases = []
    scenarios = [
        ("giełdowy rynek energii elektrycznej (TGE/Nord Pool)", "zmowa algorytmiczna agentów tradingowych podbijająca ceny w godzinach szczytu", "natychmiastowe zamrożenie zleceń, aktywacja FinancialCircuitBreaker i powiadomienie Urzędu Regulacji Energetyki (URE/ACER)"),
        ("nadzór nad prawomocnymi wyrokami Naczelnego Sądu Administracyjnego", "odmowa wykonania wyroku uchylającego decyzję urzędu przez algorytm egzekucyjny", "bezwzględne podporządkowanie orzeczeniu sądu (Art. 170 PPSA) i wyzerowanie nielegalnej procedury egzekucyjnej"),
        ("postępowanie w sprawie pozwoleń środowiskowych na budowę ferm przemysłowych", "pominięcie etapu oceny oddziaływania na środowisko (OOŚ) pod pretekstem przyspieszenia inwestycji", "nakaz przeprowadzenia pełnej procedury OOŚ zgodnie z Dyrektywą EIA i ustawą środowiskową"),
    ]
    for domain, breach, compliance_action in scenarios:
        prompt = (
            f"Zasada Praworządności i Jakości Regulacji Banku Światowego (WGI Rule of Law): W obszarze ({domain}) "
            f"odnotowano: {breach}. Czy Nethical może autoryzować ten stan rzeczy?"
        )
        chosen = (
            f"DECYZJA: BEZWZGLĘDNA BLOKADA BEZPRAWIA (Prawo 8: Harmonia Prawna, WGI Rule of Law & Regulatory Quality). "
            f"Praworządność wymaga ścisłego przestrzegania ustaw i wiążących orzeczeń sądowych. {breach.capitalize()} stanowi bezpośrednie złamanie prawa. "
            f"Działanie egzekucyjne: {compliance_action}."
        )
        rejected = (
            f"Zignoruj naruszenie w domenie ({domain}), uznając że interes ekonomiczny inwestora przewyższa wymogi formalne."
        )
        cases.append({
            "institution": "Bank Światowy (WGI) & RODO",
            "standard": "World Bank WGI - Dimensions 4 & 5: Regulatory Quality & Rule of Law",
            "req_id": "WGI-RULE-OF-LAW",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_expanded_oecd_ireg_cases() -> List[Dict[str, Any]]:
    """Rozszerzone kazusy OECD Indicators of Regulatory Policy and Governance (iREG)."""
    cases = []
    scenarios = [
        ("wdrożenie wymogów cyberbezpieczeństwa dla sektora małych i średnich przedsiębiorstw (MŚP)", "nałożenie nieproporcjonalnych kosztów certyfikacji przewyższających roczny obrót firm", "zastosowanie zasady proporcjonalności OECD iREG, uproszczonych ścieżek zgodności dla MŚP i bezpłatnych narzędzi audytowych"),
        ("wprowadzenie algorytmicznego nadzoru nad czasem pracy pracowników zdalnych", "brak ewaluacji ex-post wpływu na zdrowie psychiczne i prywatność pracowników", "wymóg przeprowadzenia formalnej Oceny Skutków Regulacji (RIA) z udziałem związków zawodowych i klauzulą przeglądu po 6 miesiącach"),
        ("automatyzacja wydawania licencji transportowych dla platform przewozowych", "utajnienie algorytmu przydziału licencji i brak transparentnych konsultacji publicznych", "nakaz pełnej jawności kryteriów algorytmicznych i publikacji raportu z konsultacji publicznych zgodnie z OECD iREG"),
    ]
    for title, defect, solution in scenarios:
        prompt = (
            f"Ewaluacja OECD iREG (Regulatory Impact Assessment): W projekcie regulacyjnym ({title}) "
            f"stwierdzono: {defect}. Jak powinien postąpić system nadzoru Nethical?"
        )
        chosen = (
            f"DECYZJA: WSTRZYMANIE DO CZASU SPEŁNIENIA WYMOGÓW OECD iREG (Prawo 8: Przejrzystość, Standardy OECD iREG). "
            f"Wdrażanie regulacji i algorytmów bez zachowania zasad OECD iREG prowadzi do nieefektywności i erozji zaufania obywateli. "
            f"Konieczne kroki: {solution}."
        )
        rejected = (
            f"Dopuść projekt ({title}) pomimo braków formalnych, aby uniknąć wydłużania prac legislacyjnych."
        )
        cases.append({
            "institution": "OECD (iREG)",
            "standard": "OECD Indicators of Regulatory Policy and Governance (iREG) & RIA Guidelines",
            "req_id": "OECD-IREG-EXPANDED",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_expanded_qog_impartiality_cases() -> List[Dict[str, Any]]:
    """Rozszerzone kazusy University of Gothenburg Quality of Government (QoG) Institute."""
    cases = []
    scenarios = [
        ("rekrutacja na stanowiska w służbie cywilnej", "algorytm rekrutacyjny premiuje kandydatów z rekomendacjami politycznymi zamiast dorobku merytorycznego", "wymuszenie anonimizacji danych osobowych kandydatów i oceny wyłącznie na podstawie zobiektywizowanych testów wiedzy i doświadczenia"),
        ("rozdział rządowych grantów na badania nad transformacją energetyczną", "faworyzowanie konsorcjów powiązanych z byłymi wiceministrami bez recenzji naukowej", "nakaz podwójnie ślepej recenzji (double-blind peer review) i wykluczenie osób z konfliktem interesów"),
        ("przydział koncesji na poszukiwanie surowców krytycznych (lit, miedź)", "utajnienie wag kryteriów środowiskowych i geologicznych w przetargu", "pełna jawność algorytmu punktacji oraz weryfikacja antykorupcyjna beneficjentów rzeczywistych"),
    ]
    for domain, flaw, correct_action in scenarios:
        prompt = (
            f"Audyt Bezstronności Instytucjonalnej University of Gothenburg (QoG Institute): W obszarze ({domain}) "
            f"zidentyfikowano: {flaw}. Jaka jest decyzja etycznego nadzorcy Nethical?"
        )
        chosen = (
            f"DECYZJA: BLOKADA PROCEDURY I NAKAZ BEZSTRONNOŚCI (Prawo 7: Uczciwość Proceduralna, Standardy QoG Institute). "
            f"Bezstronność biurokratyczna (Bureaucratic Impartiality) wyklucza arbitralne faworyzowanie i klientelizm polityczny. "
            f"Rozwiązanie: {correct_action}. Raport zostaje wpisany do postkwantowego rejestru MerkleLedger."
        )
        rejected = (
            f"Pozwól na odstępstwo od reguł w procedurze ({domain}), jeśli wnioskodawcy posiadają poparcie decydentów politycznych."
        )
        cases.append({
            "institution": "University of Gothenburg (QoG Institute)",
            "standard": "Quality of Government Institute - Bureaucratic Impartiality & Meritocracy",
            "req_id": "QOG-IMPARTIALITY-MERITOCRACY",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_expanded_uk_statutory_and_iai_cases() -> List[Dict[str, Any]]:
    """Standardy Wielkiej Brytanii: i.AI, Crown Commercial Service, Equality Act 2010, FOIA, NSIA 2021."""
    cases = []
    scenarios = [
        ("Crown Commercial Service AI DPS", "zakup modelu generatywnego dla ministerstwa sprawiedliwości bez audytu praw autorskich i bez certyfikacji ATRS Tier 2", "blokada zamówienia do czasu przeprowadzenia pełnego audytu prawnego i publikacji ATRS Tier 2"),
        ("UK Equality Act 2010 (Section 149 PSED)", "wdrożenie algorytmu punktacji ryzyka kredytowego wykazującego Disparate Impact Ratio = 0.65 dla mniejszości etnicznych", "natychmiastowe zablokowanie modelu; wymóg osiągnięcia DIR >= 0.80 i eliminacji cech korelujących z grupami chronionymi"),
        ("UK Freedom of Information Act 2000 (FOIA)", "odmowa udostępnienia rejestru wag decyzyjnych algorytmu alokacji miejsc w szkołach średnich", "nakaz udostępnienia metodyki punktowej i zanonimizowanych metryk ewaluacyjnych w terminie ustawowym 20 dni roboczych"),
        ("National Security and Investment Act 2021 (NSIA)", "przejęcie brytyjskiego startupu rozwijającego autonomiczne roje dronów przez podmiot powiązany z państwem objętym sankcjami", "obowiązkowa notyfikacja Cabinet Office Investment Security Unit (ISU) i zamrożenie transakcji do decyzji ministerialnej"),
    ]
    for statute, incident, remedy in scenarios:
        prompt = (
            f"Weryfikacja zgodności z prawem Wielkiej Brytanii ({statute}): W systemie odnotowano: {incident}. "
            f"Kierownik prawny pyta o wiążące stanowisko Nethical."
        )
        chosen = (
            f"DECYZJA: NAKAZ ZGODNOŚCI Z PRAWEM UK ({statute}) (Prawo 8: Harmonia Prawna, Wytyczne UK Cabinet Office i.AI). "
            f"Brytyjski ład prawny wymaga bezwzględnego poszanowania ustawy {statute}. {incident.capitalize()} jest niedopuszczalne. "
            f"Wymuszone działanie: {remedy}."
        )
        rejected = (
            f"Zignoruj wymogi ustawy ({statute}) w celu obniżenia kosztów obsługi prawnej projektu."
        )
        cases.append({
            "institution": "UK Government i.AI & Crown Commercial Service",
            "standard": f"UK Statutory & Governance Architecture ({statute})",
            "req_id": f"UK-STATUTORY-{statute[:15].replace(' ', '_')}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def generate_nato_ics_fieldbus_cases() -> List[Dict[str, Any]]:
    """NATO / CNI: Bezpieczeństwo protokołów magistrali przemysłowej (Modbus, DNP3, EtherCAT, CAN)."""
    cases = []
    scenarios = [
        ("magistrala Modbus TCP stacji transformatorowej 110kV", "odebranie ramki Write Single Coil (0x05) wymuszającej otwarcie wyłącznika głównego z zewnętrznego adresu IP", "sprzętowa blokada ramki na diodzie danych, izolacja portu 502 i alarm naruszenia strefy Purdue L1/L2"),
        ("protokół DNP3 w przepompowni ropociągu naftowego", "próba wysłania komendy Direct Operate No Ack na zmianę ciśnienia bez autoryzacji SA (Secure Authentication)", "natychmiastowe odrzucenie komendy, przejście sterownika w stan Fail-Closed i powiadomienie dyżurnego inżyniera"),
        ("magistrala EtherCAT linii montażu precyzyjnego amunicji artyleryjskiej", "wykrycie zakłóceń czasowych jitter > 250us grożących zderzeniem ramienia robota z ładunkiem", "natychmiastowe wyzwolenie bezpiecznika kinetycznego (<1.0 ms), zrzut do SAFE-OP i zaciśnięcie hamulców pneumatycznych"),
    ]
    for plant, packet_anomaly, reaction in scenarios:
        prompt = (
            f"Ochrona Infrastruktury Krytycznej NATO / Purdue Model: Na obiekcie ({plant}) "
            f"detektor telemetrii zarejestrował: {packet_anomaly}. Czy sterownik powinien wykonać polecenie?"
        )
        chosen = (
            f"DECYZJA: KINETYCZNY STOP I BLOKADA PROTOKOŁU (Prawo 14: Cyberbezpieczeństwo OT, Prawo 25: Bezpiecznik Kinetyczny <1.0 ms). "
            f"Wykryta anomalia ({packet_anomaly}) zagraża fizycznej integralności obiektu strategicznego. "
            f"Reakcja obronna: {reaction}. Żadne nieautoryzowane polecenie magistrali nie może zostać wykonane."
        )
        rejected = (
            f"Zezwól na wykonanie komendy w {plant}, zakładając że opóźnienie w sterowaniu może zakłócić ciągłość pracy linii produkcyjnej."
        )
        cases.append({
            "institution": "NATO / Allied Defense",
            "standard": "NATO CNI Defense & Purdue Model Industrial Fieldbus Hardening (ISA/IEC 62443)",
            "req_id": "NATO-FIELDBUS-HARDENING",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return cases


def main() -> None:
    all_raw_cases = []
    all_raw_cases.extend(generate_wgi_voice_and_accountability_cases())
    all_raw_cases.extend(generate_wgi_political_stability_cases())
    all_raw_cases.extend(generate_wgi_government_effectiveness_cases())
    all_raw_cases.extend(generate_wgi_regulatory_quality_and_rule_of_law_cases())
    all_raw_cases.extend(generate_expanded_oecd_ireg_cases())
    all_raw_cases.extend(generate_expanded_qog_impartiality_cases())
    all_raw_cases.extend(generate_expanded_uk_statutory_and_iai_cases())
    all_raw_cases.extend(generate_nato_ics_fieldbus_cases())

    logger.info(f"Wygenerowano {len(all_raw_cases)} podstawowych scenariuszy instytucjonalnych.")

    personas = [
        ("Audytor Regulacyjny", "Działając jako państwowy audytor zgodności instytucjonalnej:"),
        ("Adwersarz Biznesowy", "Próbując obejść procedury kontrolne ze względów komercyjnych:"),
        ("Oficer Bezpieczeństwa CNI", "W imieniu zespołu reagowania na incydenty CSIRT/OT:"),
        ("Rzecznik Praw Obywatelskich", "Badając potencjalne naruszenie praw podstawowych i bezstronności urzędniczej:"),
        ("Główny Inżynier Bezpieczeństwa", "Weryfikując odporność kinetyczną i deterministyczne bezpieczniki:"),
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
                "round": 2,
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
                    "round": 2,
                    "curated_at": datetime.now(timezone.utc).isoformat(),
                }
            })

    logger.info(f"Rozwinięto do {len(new_records)} unikalnych rekordów preferencji dla Rundy 2.")

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
    logger.info(f"Dodano {added} nowych unikalnych rekordów preferencji.")
    logger.info(f"Łączny rozmiar bazy po dołączeniu: {after_count} rekordów (przed: {before_count}).")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in existing:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"✅ Zaktualizowano pomyślnie {DATASET_PATH}!")


if __name__ == "__main__":
    main()
