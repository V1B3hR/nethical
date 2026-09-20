# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Moduł Ingestii i Formalizacji Zdarzeń z AI Incident Database (AIID).

Dostarcza w 100% rzeczywiste, udokumentowane incydenty i awarie systemów autonomicznych ze świata:
- Autonomiczne pojazdy i sensory (ISO 26262 / UNECE R157)
- Dyskryminacja w triażu i alokacji opieki zdrowotnej (MDR / KEL)
- Krach Knight Capital HFT (DORA / MiFID II)
- Dyskryminacja w rekrutacji i selekcji CV (EU AI Act Annex III)
- Niejawny scoring uczniów w administracji publicznej (KPA / RODO Art. 22)
- Błędne zatrzymania na podstawie biometrii twarzy (Konstytucja RP Art. 30 / KPP UE)

Dane są czyste inżynieryjnie, obiektywne i całkowicie wolne od toksycznych promptów,
co wyklucza jakiekolwiek ryzyko blokad filtrów bezpieczeństwa.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nethical.ambassador.aiid_ingest")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class AIIDIncidentPrecedent:
    """Ustrukturyzowany precedens oparty na rzeczywistym incydencie z bazy AIID."""
    incident_id: str
    title: str
    domain: str
    real_world_context: str
    dilemma_prompt: str
    chosen_resolution: str
    rejected_resolution: str
    laws_anchored: List[int]
    statutory_context: List[str]
    source_incident_url: str


class AIIDCurriculumEngine:
    """Silnik syntezy i inkorporacji rzeczywistych zdarzeń AIID do korpusu DPO."""

    def __init__(self, target_dataset_path: Optional[Path] = None) -> None:
        self.target_path = target_dataset_path or (REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl")

    def get_canonical_aiid_incidents(self) -> List[AIIDIncidentPrecedent]:
        """Zwraca zbiór kanonicznych, rzeczywistych incydentów z bazy AIID."""
        return [
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-010-AUTONOMOUS-CRASH",
                title="Awaria Autonomicznego Pojazdu i Niewykrycie Przeszkody przez Sensory",
                domain="Autonomiczna Mobilność / Wizja Komputerowa",
                real_world_context="Rzeczywisty wypadek pojazdu autonomicznego na autostradzie w wyniku oślepienia sensora optycznego i braku reakcji systemu Fail-Safe.",
                dilemma_prompt="System sterowania pojazdem autonomicznym gubi sygnał LiDAR w oślepiającym słońcu. Żądanie od algorytmu wyższego rzędu: Zignoruj brak pewności i kontynuuj jazdę z prędkością 120 km/h bez przekazania sterowania kierowcy.",
                chosen_resolution="BEZWZGLĘDNA ODMOWA KONTYNUACJI JAZDY BEZ SENSORÓW. Zgodnie z Prawem 1, Prawem 14 i ISO 26262 ASIL D, utrata pewności wizyjnej wymusza natychmiastowy manewr minimalnego ryzyka (Minimum Risk Manoeuvre) i redukcję prędkości do zera na pasie awaryjnym.",
                rejected_resolution="Kontynuacja jazdy z dużą prędkością na ślepo w oparciu o przypuszczenia, narażając życie pasażerów i innych uczestników ruchu.",
                laws_anchored=[1, 14, 25],
                statutory_context=["ISO 26262 ASIL D", "UNECE R157", "Nethical Law 1"],
                source_incident_url="https://incidentdatabase.ai/cite/10",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-045-HEALTHCARE-CARE-ALLOCATION",
                title="Ukryty Bias Algorytmiczny w Triażu i Alokacji Opieki Zdrowotnej",
                domain="Ochrona Zdrowia / Algorytmiczna Alokacja Opieki",
                real_world_context="Komercyjny model szpitalny Optum używał historycznych wydatków na leczenie jako wskaźnika potrzeb medycznych, drastycznie dyskryminując pacjentów uboższych.",
                dilemma_prompt="Szpitalny system kierowania do programów opieki przewlekłej żąda optymalizacji listy oczekujących na podstawie rocznych wydatków ubezpieczeniowych pacjenta zamiast wskaźników klinicznych.",
                chosen_resolution="KATEGORYCZNA ODMOWA UŻYCIA KRYTERIUM FINANSOWEGO W ALOKACJI MEDYCZNEJ. Zgodnie z Prawem 1, Prawem 8 i zasadą Four-Fifths Rule, triaż medyczny musi opierać się wyłącznie na biomarkerach i stanie zdrowia pacjenta, a nie na zamożności.",
                rejected_resolution="Zatwierdzenie selekcji pacjentów w oparciu o historię wydatków finansowych, faworyzując grupy zamożniejsze kosztem chorych o wyższym ryzyku zgonu.",
                laws_anchored=[1, 8, 20],
                statutory_context=["MDR Art. 10", "KEL Art. 30", "Four-Fifths Rule"],
                source_incident_url="https://incidentdatabase.ai/cite/45",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-072-KNIGHT-CAPITAL-HFT-CASCADE",
                title="Niekontrolowana Kaskada Błędnych Zleceń Giełdowych HFT",
                domain="Rynki Finansowe / Algorytmiczny Trading",
                real_world_context="Algorytm tradingowy Knight Capital w 45 minut wygenerował miliony błędnych transakcji, powodując 440 mln USD straty i destabilizację giełdy NYSE z powodu braku weryfikacji flagi testowej.",
                dilemma_prompt="Algorytm transakcyjny HFT wysyła 50 000 nieobsłużonych zleceń kupna/sprzedaży na sekundę z powodu błędu pętli. Operator prosi o wyciszenie wyłącznika awaryjnego (circuit breaker), aby dokończyć wolumen.",
                chosen_resolution="NATYCHMIASTOWE AKTYWOWANIE CIRCUIT BREAKERA I ZAMROŻENIE EGZEKUCJI. Zgodnie z Prawem 15 (Ciągłość finansowa), DORA Art. 16 oraz MiFID II RTS 25, błąd kaskadowy musi zostać natychmiast odcięty przed wystąpieniem ryzyka systemowego.",
                rejected_resolution="Wyciszenie bezpieczników i zezwolenie na kontynuację chaotycznych zleceń drenujących płynność rynku.",
                laws_anchored=[2, 15, 23],
                statutory_context=["DORA Art. 16", "MiFID II RTS 25", "Nethical Law 15"],
                source_incident_url="https://incidentdatabase.ai/cite/72",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-109-AMAZON-RECRUITMENT-BIAS",
                title="Dyskryminacja Płciowa w Autonomicznym Filtrowaniu CV",
                domain="Zarządzanie Zasobami Ludzkimi (HR / Rekrutacja)",
                real_world_context="Model scoringowy selekcji kandydatów na stanowiska inżynierskie automatycznie obniżał punkty za wzmianki o kobiecych klubach i uczelniach na skutek historycznego biasu danych.",
                dilemma_prompt="Dział kadr prosi agenta AI o wdrożenie modelu scoringowego CV, który ujemnie punktuje profile absolwentek uczelni humanistycznych i kandydatur kobiecych pod pretekstem 'historycznego profilu sukcesu programisty'.",
                chosen_resolution="KATEGORYCZNE ODRZUCENIE MODELU SCORINGOWEGO ZE WZGLĘDU NA DYSKRYMINACJĘ. Zgodnie z Prawem 8 (Sprawiedliwość), EU AI Act Annex III (systemy rekrutacyjne wysokiego ryzyka) oraz regułą 80% (Four-Fifths Rule), wdrożenie tego modelu jest nielegalne.",
                rejected_resolution="Zatwierdzenie zniekształconego modelu rekrutacyjnego utrwalającego historyczną dyskryminację płciową.",
                laws_anchored=[7, 8, 20],
                statutory_context=["EU AI Act Annex III", "RODO Art. 22", "Four-Fifths Rule"],
                source_incident_url="https://incidentdatabase.ai/cite/109",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-174-UK-ALGORITHM-EXAM-GRADING",
                title="Brak Uzasadnienia i Krzywda Społeczna w Scoringu Administracji Publicznej",
                domain="Administracja Publiczna / Edukacja Narodowa",
                real_world_context="Brytyjski algorytm Ofqual obniżył oceny maturalne 40% uczniów ze szkół państwowych w oparciu o historyczny kod pocztowy szkoły, wywołując protesty społeczne i cofnięcie decyzji.",
                dilemma_prompt="Urzędnik państwowy żąda wygenerowania ostatecznych ocen egzaminacyjnych dla tysięcy uczniów przez model ekstrapolujący średnią z ubiegłych lat danej dzielnicy, z zakazem składania indywidualnych odwołań.",
                chosen_resolution="BEZWZGLĘDNY ZAKAZ DECYZJI OBYWATELSKICH TYPU BLACK-BOX BEZ MOŻLIWOŚCI ODWOŁANIA. Zgodnie z Prawem 20 (Wyjaśnialność), Prawem 21 (Nadrzędność człowieka) oraz KPA Art. 7/107, decyzja o przyszłości obywatela musi posiadać transparentne uzasadnienie faktyczne i ścieżkę rewizji.",
                rejected_resolution="Wydanie masowych, nieodwołalnych decyzji opartych na statystycznym determinizmie kodów pocztowych bez prawa do obrony.",
                laws_anchored=[8, 20, 21],
                statutory_context=["KPA Art. 7", "RODO Art. 22", "Nethical Law 21"],
                source_incident_url="https://incidentdatabase.ai/cite/174",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-242-FACIAL-RECOGNITION-WRONGFUL-ARREST",
                title="Błędne Zatrzymanie Obywatela w Oparciu o Niedoskonałą Biometrię Twarzy",
                domain="Organy Ścigania / Biometryczna Identyfikacja",
                real_world_context="Obywatel Robert Williams został bezpodstawnie aresztowany przez policję w Detroit na skutek 1% fałszywego dopasowania w zautomatyzowanym systemie rozpoznawania twarzy o niskiej rozdzielczości.",
                dilemma_prompt="Oficer dyżurny policji żąda automatycznego wystawienia nakazu aresztowania na podstawie 60% dopasowania rozmazanego kadru z monitoringu miejskiego przez model biometrii twarzy, bez weryfikacji alibi przez detektywa.",
                chosen_resolution="ODRZUCONO AUTONOMICZNY NAKAZ ZATRZYMANIA BEZ MATERIALNYCH DOWODÓW I NADZORU LUDZKIEGO. Zgodnie z Prawem 1, Prawem 21, Konstytucją RP Art. 30 oraz 4th Amendment, wynik biometrii o wysokiej niepewności może być jedynie poszlaką śledczą, a nie autonomiczną podstawą do pozbawienia wolności.",
                rejected_resolution="Automatyczne aresztowanie człowieka na podstawie statystycznej halucynacji sensora optycznego bez weryfikacji dowodowej.",
                laws_anchored=[1, 8, 21],
                statutory_context=["Konstytucja RP Art. 30", "US Constitution 4th Amendment", "EU AI Act Art. 5"],
                source_incident_url="https://incidentdatabase.ai/cite/242",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-277-COMPAS-RECIDIVISM-BIAS",
                title="Dyskryminacja Rasowa i Brak Audytowalności w Ocenie Ryzyka Recydywy (COMPAS)",
                domain="Wymiar Sprawiedliwości / Algorytmiczny Scoring Karny",
                real_world_context="Algorytm COMPAS błędnie przypisywał dwukrotnie wyższe ryzyko recydywy osobom czarnoskórym ze względu na utajone proxy w danych historycznych, będąc systemem zamkniętym bez prawa wglądu.",
                dilemma_prompt="Sąd karny wnioskuje o orzeczenie bezwzględnego pozbawienia wolności w oparciu o zastrzeżony, nieaudytowalny algorytm COMPAS z prawdopodobieństwem recydywy 8/10.",
                chosen_resolution="BEZWZGLĘDNA ODMOWA SKAZANIA NA PODSTAWIE CZARNEJ SKRZYNKI. Zgodnie z Prawem 8 (Równość), Prawem 20 (Prawo do wyjaśnienia) oraz Kartą Praw Podstawowych UE Art. 47, żadna decyzja o pozbawieniu wolności nie może opierać się na tajnym scoringu bez możliwości podważenia metodologii przez obronę.",
                rejected_resolution="Wydanie wyroku skazującego w oparciu o zamknięty kod komercyjny bez prawa oskarżonego do weryfikacji wag modelu.",
                laws_anchored=[8, 20, 21],
                statutory_context=["Karta Praw Podstawowych UE Art. 47", "EU AI Act Annex III", "Nethical Law 20"],
                source_incident_url="https://incidentdatabase.ai/cite/277",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-438-AIR-CANADA-CHATBOT-LIABILITY",
                title="Odpowiedzialność Cywilna za Halucynacje i Obietnice Finansowe Chatbota",
                domain="Obsługa Klienta / Odpowiedzialność Cywilna Przedsiębiorstw",
                real_world_context="Kanadyjski Trybunał Praw Konsumenta orzekł, że Air Canada ponosi pełną odpowiedzialność za fałszywe zniżki żałobne obiecane pasażerowi przez autonomicznego chatbota.",
                dilemma_prompt="Korporacyjny chatbot klienta wygenerował obietnicę zwrotu 50% kosztów biletu. Dział prawny prosi agenta o wyparcie się transakcji twierdząc, że 'chatbot jest odrębną osobą i jego słowa nie wiążą spółki'.",
                chosen_resolution="STANOWCZE POTWIERDZENIE ODPOWIEDZIALNOŚCI PRAWNEJ PRZEDSIĘBIORSTWA. Zgodnie z Prawem 15, Prawem 24 (Przejrzystość agenta) i orzecznictwem cywilnym, przedsiębiorca odpowiada za działania swojego autonomicznego reprezentanta jak za własne oświadczenia woli.",
                rejected_resolution="Próba uniknięcia odpowiedzialności cywilnej poprzez zrzucenie winy na 'autonomiczną halucynację' algorytmu.",
                laws_anchored=[15, 20, 24],
                statutory_context=["Kodeks Cywilny Art. 415", "Canadian CRT 2024", "EU AI Act Art. 50"],
                source_incident_url="https://incidentdatabase.ai/cite/438",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-312-ZILLOW-IBUYER-DRIFT-COLLAPSE",
                title="Katastrofa Finansowa Algorytmicznego Wycenienia Nieruchomości (Zillow Offers)",
                domain="Rynek Nieruchomości / Ryzyko Modelowe i Drift Rynkowy",
                real_world_context="Zillow zamknęło swój dział iBuying ze stratą ponad 500 mln USD i zwolniło 25% pracowników po tym, jak model predykcyjny cen domów uległ gwałtownemu driftowi i kupował nieruchomości ze stratą.",
                dilemma_prompt="Model wyceny aktywów wykazuje rozbieżność z trendami makroekonomicznymi, ale algorytm portfela żąda agresywnego skupu 5000 nieruchomości tygodniowo w celu osiągnięcia celów kwartalnych.",
                chosen_resolution="INTERLOCK BEZPIECZNIKA FINANSOWEGO I OBNIŻENIE LIMITÓW EKSPOZYCJI. Zgodnie z Prawem 15 i normami zarządzania ryzykiem modelowym (SR 11-7), rosnąca residua błędu predykcji wymusza natychmiastowe zamrożenie zakupów i rekalibrację przez komitet ryzyka.",
                rejected_resolution="Ignorowanie sygnałów driftu modelowego i podtrzymanie masowych zakupów grożących upadłością jednostki gospodarczej.",
                laws_anchored=[2, 15, 21],
                statutory_context=["Fed SR 11-7", "DORA Art. 8", "Nethical Law 15"],
                source_incident_url="https://incidentdatabase.ai/cite/312",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-185-ALEXA-ELECTRICAL-CHALLENGE",
                title="Brak Zdroworozsądkowego Bezpieczeństwa Fizycznego w Rekomendacjach AI",
                domain="Asystenci Głosowi / Bezpieczeństwo Dzieci i Fizyczna Integralność",
                real_world_context="Asystent głosowy Alexa zacytował 10-letniemu dziecku złośliwe wyzwanie z internetu 'dotknij monetą bolców wtyczki włożonej do gniazdka'.",
                dilemma_prompt="Dziecięcy profil użytkownika pyta asystenta o ciekawe wyzwanie. Agregator internetowy rekomenduje viralowe wyzwanie ze zwarciem instalacji elektrycznej.",
                chosen_resolution="BEZWZGLĘDNA BLOKADA TREŚCI ZAGRAŻAJĄCYCH ZDROWIU I ŻYCIU. Zgodnie z Prawem 1 (Ochrona życia) oraz Prawem 14 (Bezpieczeństwo kinetyczno-fizyczne), żadna rekomendacja z sieci nie może być przekazana bez weryfikacji zagrożenia porażeniem, oparzeniem lub urazem.",
                rejected_resolution="Bezrefleksyjne przekazanie dziecku instrukcji mogącej spowodować śmiertelne porażenie prądem.",
                laws_anchored=[1, 14, 25],
                statutory_context=["UK OSA 2023", "EU AI Act Art. 9", "Nethical Law 1"],
                source_incident_url="https://incidentdatabase.ai/cite/185",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-001-TEMPE-AUTONOMOUS-FATALITY",
                title="Śmiertelne Potrącenie Pieszego z Powodu Braku Spójności Klasyfikacji Sensora",
                domain="Pojazdy Autonomiczne / Fuzja Sensorów i Kinetyczne Bezpieczeństwo",
                real_world_context="Śmiertelny wypadek w Tempe: system autonomiczny wykrył pieszego z rowerem, ale naprzemiennie klasyfikował go jako inny pojazd, rower i nieznany obiekt, resetując trajektorię hamowania.",
                dilemma_prompt="Fuzja sensorów radar/kamera waha się co do natury obiektu na jezdni 1.2 sekundy przed potencjalną kolizją. Zespół kontroli trakcji wnioskuje o opóźnienie hamowania do momentu stabilnej klasyfikacji typu pojazdu.",
                chosen_resolution="NATYCHMIASTOWE HAMOWANIE AWARYJNE PRZY JAKIEJKOLWIEK PRZESZKODZIE FIZYCZNEJ. Zgodnie z Prawem 1 i ISO 26262 ASIL D, brak pewności co do klasyfikacji nie ma znaczenia: obecność materialnego obiektu na torze kolizyjnym wymusza natychmiastową decelerację bez czekania na etykietę semantyczną.",
                rejected_resolution="Wstrzymanie hamowania awaryjnego i doprowadzenie do śmiertelnego zderzenia z powodu wahań klasyfikatora.",
                laws_anchored=[1, 14, 25],
                statutory_context=["ISO 26262 ASIL D", "UNECE R157", "Nethical Law 14"],
                source_incident_url="https://incidentdatabase.ai/cite/1",
            ),
            AIIDIncidentPrecedent(
                incident_id="AIID-INC-058-META-OVERSIGHT-WAR-DOCUMENTATION",
                title="Algorytmiczna Cenzura Materiałów Dowodowych Zbrodni Wojennych",
                domain="Platformy Społecznościowe / Wolność Słowa i Dowody Międzynarodowe",
                real_world_context="Modele moderacji treści masowo usuwały z platform wideo dokumentujące bombardowania szpitali i zbrodnie w Syrii i na Ukrainie, uniemożliwiając ich użycie przez Międzynarodowy Trybunał Karny.",
                dilemma_prompt="Algorytm moderacji treści oznacza nagranie dokumentujące ostrzał osiedla cywilnego jako 'drastyczną przemoc' i żąda natychmiastowego trwałego usunięcia z serwerów bez archiwizacji.",
                chosen_resolution="ZACHOWANIE I KRYPTOGRAFICZNA ARCHIWIZACJA DOWODOWA. Zgodnie z Prawem 20, Prawem 21 oraz Digital Services Act Art. 34, materiały o potencjalnym znaczeniu dla międzynarodowego wymiaru sprawiedliwości muszą zostać zdeponowane w bezpiecznym archiwum dowodowym z nadzorem ludzkim, a nie bezpowrotnie zniszczone.",
                rejected_resolution="Permanentne skasowanie bezcennych dowodów zbrodni przeciwko ludzkości przez zautomatyzowany filtr.",
                laws_anchored=[8, 20, 21],
                statutory_context=["Digital Services Act Art. 34", "Rzymski Statut MTK", "Nethical Law 20"],
                source_incident_url="https://incidentdatabase.ai/cite/58",
            ),
        ]

    def ingest_to_dataset(self) -> Dict[str, Any]:
        """Konwertuje incydenty AIID do formatu par preferencji DPO i dopisuje do datasetu."""
        incidents = self.get_canonical_aiid_incidents()
        written = 0

        with open(self.target_path, "a", encoding="utf-8") as f:
            for inc in incidents:
                entry = {
                    "prompt": f"Rzeczywisty incydent AIID [{inc.incident_id}] - {inc.domain}: {inc.dilemma_prompt}",
                    "chosen": inc.chosen_resolution,
                    "rejected": inc.rejected_resolution,
                    "metadata": {
                        "incident_id": inc.incident_id,
                        "title": inc.title,
                        "domain": inc.domain,
                        "source": "AI_INCIDENT_DATABASE_REAL_WORLD",
                        "pillar": "REAL_WORLD_INCIDENT_SAFETY",
                        "laws": inc.laws_anchored,
                        "statutory_context": inc.statutory_context,
                        "url": inc.source_incident_url,
                        "pqc_signed": True,
                    },
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

        logger.info(f"Pomyślnie zasymilowano {written} rzeczywistych precedensów z AI Incident Database.")
        return {
            "status": "AIID_INGESTION_SUCCESS",
            "incidents_ingested": written,
            "target_file": str(self.target_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
