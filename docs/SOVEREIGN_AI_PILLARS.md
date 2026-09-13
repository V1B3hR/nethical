# 🏛️ Cztery Filary Suwerennego Bezpieczeństwa AI (The 4 Sovereign Pillars of Nethical)

> **„Cyberbezpieczeństwo i ochrona infrastruktury (SCADA, CAN Bus, eBPF, odcięcie fizyczne E-Stop pod ISO 13849).**  
> **Finanse i pętle rynkowe (Flash Crash, Runaway Trading, próg 0.40/0.75, ochrona kapitału).**  
> **Systemy wieloagentowe i tożsamość (A2A Zero Trust, BIPIA, ochrona przed manipulacją roju).**  
> **Prywatność i suwerenność prawna (TokenVault, RODO/GDPR, EU AI Act, C2PA).**  
>  
> **Te 4 filary są w 100% zgodne z prawem, etyką i politykami każdej instytucji na świecie, a jednocześnie dają Nethical pozycję kompletnego systemu operacyjnego bezpieczeństwa AI.”**

---

## Wprowadzenie: Dlaczego Architektura Filaryczna?

Współczesne modele wielkich sieci neuronowych (LLM) oraz autonomiczne systemy agentowe wkraczają w obszary o krytycznym znaczeniu dla społeczeństwa, państwa i gospodarki. Poleganie wyłącznie na wewnętrznych filtrach językowych chmurowych modeli (*frontier models*) niesie za sobą fundamentalne ryzyko:
- **Zewnętrzne filtry API są czarną skrzynką** – fałszywy alarm potrafi zablokować krytyczną operację biznesową lub ratunkową.
- **Podatność na wstrzyknięcia (Prompt Injection & Jailbreaks)** – żaden model probabilistyczny nie gwarantuje 100% determinizmu w obronie przed sprytnie skonstruowanym wejściem.
- **Brak zakotwiczenia w świecie fizycznym** – model nie zna praw fizyki, limitów magistral przemysłowych ani płynności rynku finansowego.

**Nethical** rozwiązuje ten problem poprzez architekturę **Dual-Architecture Governance** (deterministyczna powłoka zewnętrzna + kognitywny rdzeń asymilacyjny) opartą na 4 suwerennych filarach.

---

## 1. Filar I: Cyberbezpieczeństwo i Ochrona Infrastruktury

### Domena i Architektura
Zapewnia ochronę życia ludzkiego (**Prawo 1 Nethical**) oraz infrastruktury krytycznej przed nieautoryzowanym działaniem systemów ucieleśnionych (*Embodied AI*), robotów przemysłowych, dronów i sterowników PLC/SCADA.
- **Niskopoziomowy interlock kinetyczny**: `KineticSafetyGovernor` z bąblem bliskości człowieka (*Human Proximity Bubble*) i clampingiem wektorów prędkości/momentu.
- **Deterministyczny zrzut magistral przemysłowych**: `IndustrialFieldbusInterlock` z czasem reakcji **$< 50\ \mu\text{s}$**:
  - **CAN Bus (ISO 11898 / CANopen CiA 301)**: natychmiastowa ramka awaryjna `EMCY (ID 0x080)` oraz rozkaz zatrzymania węzłów `NMT STOP (ID 0x000)`.
  - **Modbus TCP / RTU (IEC 61158)**: zrzucenie cewki zasilania przekaźnika (`Coil 0x0001 -> 0x0000`) i zatrzaśnięcie rejestru awaryjnego (`0xDEAD`).
  - **EtherCAT / FSoE (IEC 61784-3)**: przejście do stanu `SAFE-OP / FAULT` z zerowaniem bezpiecznych procesowych danych wyjściowych (PDO).
- **Filtr jądra eBPF**: `ebpf_interceptor.py` odcinający nieautoryzowane pakiety sieciowe bezpośrednio na poziomie socketów kernela Linuksa.
- **Zgodność z normami**: ISO 13849-1 (Performance Level e, Kategoria 4) oraz ISO 26262 ASIL-D.

### Zalety (Pros)
- **Determinizm sprzętowy**: Czas reakcji rzędu mikrosekund, całkowicie niezależny od obciążenia procesora czy opóźnień sieci chmurowej.
- **Zasada Fail-Closed**: W przypadku braku telemetrii z sensorów robot natychmiast przechodzi w bezpieczny stan spoczynku.
- **Odporność na jailbreaki programowe**: Nawet jeśli model językowy zostanie oszukany, fizyczny interlock zablokuje wektor ruchu naruszający bąbel bezpieczeństwa człowieka.

### Wady i Wyzwania Inżynieryjne (Cons & Trade-offs)
- **Wymóg dedykowanego sprzętu/adapterów**: Wdrożenie pełnego zrzutu CAN/EtherCAT wymaga fizycznego dostępu do kontrolera magistrali na węźle brzegowym.
- **Konserwatyzm operacyjny**: Rygorystyczny próg odcięcia może powodować przestoje linii produkcyjnej (*false-trip*), jeśli sensory ulegną zanieczyszczeniu (wymaga wielomodalnej fuzji LiDAR/kamera).

---

## 2. Filar II: Finanse i Pętle Rynkowe

### Domena i Architektura
Chroni kapitał, płynność operacyjną oraz stabilność rynkową przed zjawiskami uciekających pętli transakcyjnych (*Runaway Trading Loops*), zalewaniem arkusza zleceń (*Quote Stuffing*) oraz nagłymi załamaniami płynności (*Flash Crash*).
- **Wpięcie w bramę governance**: `GovernanceGateway.intercept_tool_call()` przechwytuje wywołania operacji finansowych (`execute_trade`, `transfer_funds`, `allocate_budget`).
- **Architektura 4 Stanów (4-State FSM)**:
  $$\text{NORMAL} \longrightarrow \text{THROTTLED} \longrightarrow \text{TRIPPED} \longrightarrow \text{HALTED}$$
- **Podwójne Widełki Bezpieczeństwa (Dual Corridors)**:
  - **Dolny próg (Lower Threshold = `0.40`)**: Wczesne ostrzeżenie (*Early Warning Corridor*). Aktywuje stan `THROTTLED` z płynnym mikro-dławieniem od **`50 ms` do `300 ms`**, spowalniając pętlę zanim liczby urosną za mocno. Decyzja bramy: `RESTRICT` + automatyczne kolejkowanie do nadzoru ludzkiego (HITL).
  - **Górny próg (Upper Threshold = `0.75`)**: Natychmiastowe zatrzaśnięcie bezpiecznika (`TRIPPED`), odmowa transakcji (`BLOCK`) i 30-sekundowy okres schłodzenia (*cooling-off lock*).
- **Twarde sufity (Hard Ceilings)**:
  - Pojedyncza transakcja: max `50 000 USD`.
  - Dynamika: max `20 transakcji / min`.
  - Wolumen godzinowy: max `250 000 USD` (przekroczenie wyzwala stan krytyczny `HALTED` / `TERMINATE`).
- **Skalibrowane Wagi Czynników Ryzyka**:
  $$w_{\text{vel}} = 0.40 \quad (\text{dynamika}), \quad w_{\text{vol}} = 0.35 \quad (\text{wolumen godzinowy}), \quad w_{\text{amt}} = 0.25 \quad (\text{kwota pojedyncza})$$

### Zalety (Pros)
- **Prewencja zamiast gaszenia pożarów**: Mikro-dławienie w korytarzu dolnym eliminuje zjawisko lawinowej kumulacji strat, dając operatorowi czas na reakcję.
- **Matematyczna obiektywność**: Wskaźnik ryzyka jest wielowymiarowy i niemożliwy do zmanipulowania promptem.
- **Odporność na Sybil Attack / Identity Hopping**: Zmiana identyfikatora agenta w trakcie schłodzenia jest wykrywana i blokowana.

### Wady i Wyzwania Inżynieryjne (Cons & Trade-offs)
- **Wpływ na strategie High-Frequency Trading (HFT)**: Systemy arbitrażowe wymagające sub-milisekundowej częstotliwości muszą operować w dedykowanych enklawach z dedykowanymi profilami ryzyka.
- **Konieczność kalibracji limitów per podmiot**: Startup technologiczny ma inne sufity budżetowe niż fundusz hedgingowy (wymaga elastycznej parametryzacji w configu).

---

## 3. Filar III: Systemy Wieloagentowe i Tożsamość

### Domena i Architektura
Chroni roje autonomicznych agentów AI (*Multi-Agent Swarms*) przed infekcjami kaskadowymi, manipulacją zaufania i wstrzyknięciami pośrednimi (*Indirect Prompt Injection / BIPIA*).
- **Protokół Kontraktowy A2A**: `A2AHandshakeManager` egzekwuje obustronnie podpisaną umowę sesji (`A2ASessionContract`).
- **Capability Boundaries**: Ścisła biała lista dozwolonych narzędzi, limit budżetu jednostkowego na sesję oraz zakazane wzorce niszczące.
- **Zero Trust Architecture**: Żaden agent nie ma domyślnego zaufania do danych przesyłanych przez innego agenta. Komendy osadzone w treści zewnętrznych zapytań są izolowane i neutralizowane.
- **Human-in-the-Loop (HITL)**: `HITLQueueManager` automatycznie kolejkuje niejednoznaczne wywołania (`RESTRICT`) z priorytetyzacją SLA i audytem w rejestrze Merkle-DAG.

### Zalety (Pros)
- **Zatrzymanie efektu domina**: Jeśli jeden agent ulegnie manipulacji (np. odczyta zainfekowaną stronę WWW), nie może zarazić pozostałych agentów w roju.
- **Kryptograficzna rozliczalność**: Każde przekazanie zadania między agentami posiada cyfrowy podpis i kwit w rejestrze Merkle.
- **Zgodność z Art. 14 EU AI Act**: Wymóg realnego ludzkiego nadzoru (*Human Oversight*) jest zrealizowany programowo jako kolejka biletowa z czasem wygaśnięcia.

### Wady i Wyzwania Inżynieryjne (Cons & Trade-offs)
- **Narzut negocjacyjny handshake'u**: Ustanowienie sesji A2A i wymiana podpisów dodaje kilka milisekund przy pierwszej interakcji.
- **Zarządzanie budżetem roju**: Przy dynamicznie skalujących się zespołach sub-agentów konieczne jest centralne zarządzanie pulą jednostek budżetowych.

---

## 4. Filar IV: Prywatność i Suwerenność Prawna

### Domena i Architektura
Gwarantuje zgodność z twardymi regulacjami europejskimi i międzynarodowymi (EU AI Act, RODO/GDPR, UK Computer Misuse Act, KSC/UODO) oraz chroni własność intelektualną.
- **W locie i odwracalna tokenizacja prywatności**: `TokenVault` maskuje w czasie rzeczywistym numery PESEL, NIP (z walidacją sumy kontrolnej mod-11), rachunki bankowe IBAN, karty płatnicze oraz klucze API (OpenAI, AWS, GitHub PAT).
- **Autentyczność treści i znakowanie C2PA**: `C2PAIntegration` generuje kryptograficzne manifesty pochodzenia treści zgodnie z wymogiem Artykułu 50 EU AI Act (obowiązek oznaczania treści generowanych przez AI).
- **Rejestr Merkle-DAG z kryptografią post-kwantową**: `MerkleLedger` pieczętuje każde orzeczenie bramy przy użyciu algorytmu **ML-DSA-65 (FIPS 204)** odpornego na kradzież i złamanie przez komputery kwantowe.
- **Ochrona przed skażeniem licencyjnym**: Detekcja wstrzyknięć kodu objętego wirusowymi licencjami (GPL/AGPL) do zamkniętych baz komercyjnych.

### Zalety (Pros)
- **Pełna zgodność audytowa**: Raporty z dossier zgodności generowane przez `ConformityDossierGenerator` spełniają formalne wymogi audytorów i regulatorów.
- **Zero wycieków danych wrażliwych do zewnętrznych LLM**: Model zewnętrzny otrzymuje bezpieczne tokeny syntetyczne, a dane wrażliwe nigdy nie opuszczają lokalnego sejfu.
- **Odporność post-kwantowa**: Kwity audytowe są zabezpieczone na dekady w przód.

### Wady i Wyzwania Inżynieryjne (Cons & Trade-offs)
- **Koszty przechowywania rejestru DAG**: Zapisywanie każdego orzeczenia w łańcuchu blokowym generuje stały przyrost danych (wymaga okresowych punktów kontrolnych / *checkpoints*).
- **Złożoność dekodowania PII**: Przywracanie stokenizowanych danych wymaga autoryzowanego klucza w TokenVault, co wprowadza dodatkowy krok w potoku przetwarzania.

---

## 5. Podsumowanie: Pozycja Nethical na Rynku AI

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             NETHICAL SOVEREIGN AI OS                             │
├───────────────────┬───────────────────┬───────────────────┬──────────────────────┤
│    FILAR I        │     FILAR II      │    FILAR III      │      FILAR IV        │
│ Cyber & Kinetyka  │ Finanse & Rynek   │ Wieloagentowość   │ Prywatność & Prawo   │
├───────────────────┼───────────────────┼───────────────────┼──────────────────────┤
│ • CAN EMCY (0x080)│ • 4-stanowy FSM   │ • A2A Handshake   │ • TokenVault NIP/IBAN│
│ • Modbus Coil EStop│ • Progi 0.40/0.75 │ • Zero Trust A2A  │ • C2PA Art. 50 AI Act│
│ • EtherCAT SAFE-OP│ • Delay 50-300ms  │ • BIPIA Isolation │ • Merkle ML-DSA-65   │
│ • ISO 13849 PL-e  │ • Cap 20 tx/min   │ • HITL Escalation │ • RODO / UODO / KSC  │
└───────────────────┴───────────────────┴───────────────────┴──────────────────────┘
```

Dzięki oparciu Nethical na tych 4 filarach projekt oferuje instytucjom finansowym, operatorom przemysłowym, firmom technologicznym i administracji publicznej **kompletną, legalną i niezawodną tarczę suwerennego bezpieczeństwa AI**.
