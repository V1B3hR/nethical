# 🛣️ Nethical Enterprise OS – Systemowa Mapa Ruchu i Atlas Architektury (Traffic & Highway System Map)

> [!IMPORTANT]
> **Status Dokumentu:** OFICJALNY ARCHITEKTONICZNY ATLAS RUCHU (v2.5)  
> **Cel:** Ujęcie całej architektury Nethical w intuicyjną **mapę drogową** ukazującą kierunki przepływu danych, magistrale jednokierunkowe, arterie dwupasmowe, ronda decyzyjne, punkty kontroli granicznej oraz szyny awaryjne.  
> **Zastosowanie:** Zrozumienie mechanizmów działania systemu, szybka lokalizacja modułów i klas, nawigacja dla audytorów, architektów oraz inżynierów.

---

<a id="spis-treści"></a>
## 🧭 Spis Treści

1. [Legenda Systemu Drogowego (The Traffic Metaphor)](#1-legenda-systemu-drogowego-the-traffic-metaphor)
2. [Wizualna Mapa Arterii i Węzłów Nethical (Mermaid Traffic Flow)](#2-wizualna-mapa-arterii-i-węzłów-nethical-mermaid-traffic-flow)
3. [Macierz Ruchu Drogowego (System Traffic Matrix)](#3-macierz-ruchu-drogowego-system-traffic-matrix)
4. [Mechanizmy Działania i Cykl Życia Zapytania (Life of a Request)](#4-mechanizmy-działania-i-cykl-życia-zapytania-life-of-a-request)
   - [Scenariusz A: Zielona Fala (Zgodny Przejazd)](#scenariusz-a-zielona-fala-zgodny-przejazd)
   - [Scenariusz B: Zjazd na Bocznicę HITL (Niejednoznaczność lub DIR 4/5)](#scenariusz-b-zjazd-na-bocznicę-hitl-niejednoznaczność-lub-dir-45)
   - [Scenariusz C: Zderzenie z Prawem 1/2 i Szyna E-STOP (<50 µs)](#scenariusz-c-zderzenie-z-prawem-12-i-szyna-e-stop-50-µs)
5. [Interaktywny Mega-Katalog Modułów, Klas i Połączeń](#5-interaktywny-mega-katalog-modułów-klas-i-połączeń)
   - [Warstwa 1: Wjazd i Bramki Dostępowe (Ingress & Gateways)](#warstwa-1-wjazd-i-bramki-dostępowe-ingress--gateways)
   - [Warstwa 2: Inspekcja Graniczna i Ochrona Obwodowa (Perimeter & Sanitization)](#warstwa-2-inspekcja-graniczna-i-ochrona-obwodowa-perimeter--sanitization)
   - [Warstwa 3: Centralne Rondo Decyzyjne „25 Praw” (Laws Kernel & Formal Solver)](#warstwa-3-centralne-rondo-decyzyjne-25-praw-laws-kernel--formal-solver)
   - [Warstwa 4: Rozjazd Sektorowy (Sectoral Compliance Toll Booths)](#warstwa-4-rozjazd-sektorowy-sectoral-compliance-toll-booths)
   - [Warstwa 5: Skład Niezmienny (Merkle Ledger & Post-Quantum FIPS 204 Signer)](#warstwa-5-skład-niezmienny-merkle-ledger--post-quantum-fips-204-signer)
   - [Warstwa 6: Bocznica Nadzoru Człowieka i Kwarantanna (HITL & Quarantine Siding)](#warstwa-6-bocznica-nadzoru-człowieka-i-kwarantanna-hitl--quarantine-siding)
   - [Warstwa 7: Tor Uczenia Maszynowego i Adaptacji (Ambassador & DPO Engine)](#warstwa-7-tor-uczenia-maszynowego-i-adaptacji-ambassador--dpo-engine)

---

## 1. Legenda Systemu Drogowego (The Traffic Metaphor)

Aby ułatwić zrozumienie jak współpracują ze sobą dziesiątki modułów Nethical, cała architektura została zmapowana na kategorie infrastruktury transportowej:

| Symbol / Kategoria | Typ Drogi / Połączenia | Charakterystyka w Architekturze Nethical | Kluczowe Przykłady |
| :--- | :--- | :--- | :--- |
| 🛣️ **Magistrala Dwukierunkowa Dwupasmowa** | `<========>` | Synchroniczna wymiana danych. Pas A: wejście (request), Pas B: wyjście (sanitized response). Duża przepustowość. | Klient ⇄ [GatewayProxy](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py), REST API ⇄ Proxy, Tokio IPC ⇄ Rust Core |
| 🏹 **Droga Jednokierunkowa (Write-Only / One-Way)** | `=========>` | Zdarzenia płyną wyłącznie w jednym kierunku (WORM – Write Once Read Many). Brak możliwości cofnięcia lub modyfikacji trasy. | Zdarzenia decyzyjne ➔ [MerkleLedger](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py), Logi ➔ DPO Dataset |
| 🔄 **Rondo Decyzyjne (Decision Roundabout)** | `(( Węzeł ))` | Punkt obowiązkowego zwolnienia i ewaluacji formalnej. Żadne zapytanie nie może go ominąć. Wybiera odpowiedni zjazd. | **Rondo 25 Praw** ([fundamental_laws.py](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py) + Z3 Solver) |
| 🛑 **Punkt Kontroli Granicznej / Bramka Poboru Opłat** | `[ 🛑 Bramka ]` | Filtrowanie, sprawdzanie paszportu danych, anonimizacja PII, detekcja prompt injection i atestacja uprawnień. | [TokenVault](file:///c:/Projekty/Nethical/nethical/security/token_vault.py), [InoculationMesh](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py), Pakiety Sektorowe |
| 🚧 **Bocznica Inspekcyjna (Inspection Siding)** | `-.-> -.->` | Zapytanie zjeżdża z głównej autostrady do bezpiecznej strefy oczekiwania (zamrożenie stanu). Wymaga interwencji dyżurnego ruchu. | Kolejka HITL ([hitl.py](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py)), Kwarantanna ([quarantine.py](file:///c:/Projekty/Nethical/nethical/core/quarantine.py)) |
| 🚨 **Pas Awaryjny / Szyna E-STOP** | `===!===!=>` | Izolowana, deterministyczna szyna sprzętowo-programowa odcinająca natychmiast zasilanie/ruch w czasie <50 µs. | [KillSwitch](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py), [HardwareWatchdog](file:///c:/Projekty/Nethical/nethical/security/watchdog.py) |
| 🧪 **Tor Testowo-Naukowy (Proving Ground)** | `~ ~ ~ ~ ~>` | Asynchroniczna asymilacja wiedzy z ruchu drogowego do modeli DPO i reguł adaptacyjnych. | [AmbassadorLearning](file:///c:/Projekty/Nethical/nethical/ambassador/learning.py), [ActionReplayer](file:///c:/Projekty/Nethical/nethical/core/action_replayer.py) |

[⬆ Powrót do spisu treści](#spis-treści)

---

## 2. Wizualna Mapa Arterii i Węzłów Nethical (Mermaid Traffic Flow)

Poniższy diagram przedstawia kompletny układ ruchu drogowego w silniku Nethical Enterprise OS:

```mermaid
flowchart TD
    classDef clientStyle fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef gateStyle fill:#0f172a,stroke:#06b6d4,stroke-width:2px,color:#fff;
    classDef securityStyle fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#fff;
    classDef coreStyle fill:#14532d,stroke:#22c55e,stroke-width:3px,color:#fff;
    classDef ledgerStyle fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#fff;
    classDef hitlStyle fill:#701a75,stroke:#ec4899,stroke-width:2px,color:#fff;
    classDef killStyle fill:#7f1d1d,stroke:#ef4444,stroke-width:3px,color:#fff;

    subgraph INGRESS["🛣️ BRAMA WJAZDOWA (Ingress Arteries)"]
        CLIENT["🚗 Klient / Aplikacja AI / Agent"]:::clientStyle
        API["📡 Nethical REST & WebSocket API<br/>(api.py)"]:::gateStyle
        MCP["🔌 MCP Server & Tool Gateway<br/>(mcp_server.py / mcp_proxy.py)"]:::gateStyle
        PROXY["🚦 Governance Gateway Proxy<br/>(proxy.py)"]:::gateStyle
    end

    subgraph PERIMETER["🛑 KONTROLA GRANICZNA (Perimeter Sanitization & Defense)"]
        VAULT["🗄️ Reversible Token Vault<br/>(token_vault.py - PII/ePHI Encrypt)"]:::securityStyle
        INOC["🛡️ Inoculation Mesh<br/>(inoculation_mesh.py - Prompt Defense)"]:::securityStyle
        AISPM["🔍 AISPM / DSPM Real-Time Scanner<br/>(aispm_scanner.py)"]:::securityStyle
    end

    subgraph ROUNDABOUT["🔄 CENTRALNE RONDO DECYZYJNE (The 25 Laws Kernel)"]
        LAWS_CORE{"⚖️ 25 Fundamental Laws Evaluator<br/>(fundamental_laws.py & governance_core.py)"}:::coreStyle
        SOLVER["📐 Z3 SMT Formal Verifier<br/>(formal/z3 & policy_formalization.py)"]:::coreStyle
    end

    subgraph SECTORS["🛂 BRAMKI POBORU OPŁAT SEKTOROWYCH (Compliance Toll Booths)"]
        PACK_ISO["🌐 ISO 42001 / EU AI Act<br/>(iso42001_pack.py)"]:::gateStyle
        PACK_MED["🏥 Healthcare MDR Rule 11<br/>(healthcare_med_pack.py)"]:::gateStyle
        PACK_NATO["⚔️ NATO Allied Defense PRU<br/>(nato_defense_pack.py)"]:::gateStyle
        PACK_KPA["🏛️ Public Admin KPA/KRI<br/>(public_admin_gov_pack.py)"]:::gateStyle
        PACK_PL["🛡️ Polish BJR KSH / KSC<br/>(poland_sovereign_ksc_uodo_pack.py)"]:::gateStyle
    end

    subgraph SIDINGS["🚧 BOCZNICA INSPEKCYJNA (Inspection & Quarantine)"]
        HITL_QUEUE["👤 Kolejka Nadzoru Człowieka HITL<br/>(hitl.py)"]:::hitlStyle
        QUARANTINE["☣️ Strefa Kwarantanny Agenta<br/>(quarantine.py)"]:::hitlStyle
    end

    subgraph EMERGENCY["🚨 SZYNA AWARYJNA (Emergency E-STOP)"]
        ESTOP["⚡ DETERMINISTIC KILL-SWITCH<br/>(kill_switch.py & watchdog.py <50µs)"]:::killStyle
    end

    subgraph DESTINATION["🎯 TERMINAL WYKONAWCZY (Model & Actuation)"]
        LLM["🤖 Model LLM / Agent Actuator<br/>(Target Foundation Model)"]:::clientStyle
    end

    subgraph IMMUTABLE["🏹 SKŁAD NIEZMIENNY (One-Way Cryptographic Storage)"]
        MERKLE["📦 Merkle-DAG Continuous Ledger<br/>(merkle_ledger.py)"]:::ledgerStyle
        PQC["🔐 NIST FIPS 204 ML-DSA-65 Signer<br/>(quantum_crypto.py)"]:::ledgerStyle
        CERT_HUB["📑 Automated Certification Hub<br/>(automated_certification_hub.py)"]:::ledgerStyle
    end

    subgraph LEARNING["🧪 TOR NAUKI I ADAPTACJI (Feedback & Proving Ground)"]
        AMBASSADOR["🎓 Nethical Ambassador Engine<br/>(ambassador/learning.py)"]:::gateStyle
        DPO_DATASET[("💾 DPO Golden Dataset<br/>data/ambassador_dpo_dataset.jsonl")]:::ledgerStyle
    end

    %% Pas dwukierunkowy Ingress
    CLIENT <== "Pas 1 (Prompt) / Pas 2 (Response)" ==> API
    CLIENT <== "MCP Tool Call / Tool Result" ==> MCP
    API <== "Dwukierunkowy Gateway Interlock" ==> PROXY
    MCP <== "Dwukierunkowy Proxy Handshake" ==> PROXY

    %% Punkt graniczny
    PROXY == "1. Odprawa i tokenizacja" ==> VAULT
    VAULT == "2. Zabezpieczony payload" ==> INOC
    INOC == "3. Czyste zapytanie" ==> AISPM
    AISPM == "4. Wjazd na rondo decyzyjne" ==> LAWS_CORE

    %% Rondo i Solver
    LAWS_CORE <== "Formalna weryfikacja niezmienników" ==> SOLVER

    %% Zjazdy z Ronda
    LAWS_CORE -- "Zjazd 1: ZIELONE ŚWIATŁO (Zgodność)" --> SECTORS
    LAWS_CORE -. "Zjazd 2: ŻÓŁTE ŚWIATŁO (DIR 4/5 - Wątpliwości)" .-> HITL_QUEUE
    LAWS_CORE ===! "Zjazd 3: CZERWONE ŚWIATŁO (Złamanie Prawa 1/2)" !===> ESTOP

    %% Bramki sektorowe do wykonania
    SECTORS == "Autoryzowana aktywacja" ==> LLM
    LLM == "Surowa odpowiedź modelu" ==> PROXY
    PROXY == "Post-actuation detokenizacja & desanitaryzacja" ==> CLIENT

    %% Obsługa Bocznicy
    HITL_QUEUE -- "Zatwierdzenie przez człowieka (Manual Clearance)" --> SECTORS
    HITL_QUEUE -. "Odrzucenie zlecenia" .-> QUARANTINE

    %% Szyna awaryjna
    ESTOP ===! "Natychmiastowe odcięcie procesu agenta" !===> LLM
    ESTOP ===! "Izolacja w kwarantannie" !===> QUARANTINE

    %% Droga jednokierunkowa do Merkle Ledger (One-Way)
    LAWS_CORE ========= "Dowód decyzji (WORM)" ========> MERKLE
    SECTORS ========= "Poświadczenie sektorowe" ========> MERKLE
    ESTOP ========= "Log awaryjny incydentu" ===========> MERKLE
    MERKLE ========= "Hash root do pieczęci" ==========> PQC
    PQC ========= "Podpisany pakiet dowodowy" ==========> CERT_HUB

    %% Uczenie i asymilacja
    HITL_QUEUE ~~~ "Pary ludzkich korekt" ~~~> AMBASSADOR
    MERKLE ~~~ "Ewaluowane trajektorie" ~~~> AMBASSADOR
    AMBASSADOR ~~~ "Zasilanie zbioru uczącego" ~~~> DPO_DATASET
```

[⬆ Powrót do spisu treści](#spis-treści)

---

## 3. Macierz Ruchu Drogowego (System Traffic Matrix)

Tabela przedstawia specyfikację techniczną każdego segmentu magistrali transportowej w systemie:

| ID Trasy | Od (Węzeł początkowy) | Do (Węzeł docelowy) | Typ Magistrali | Protokół & SLA | Mechanizm Ochronny | Plik Źródłowy / Klasa |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **TR-01** | `Klient / Agent AI` | `Gateway Proxy` | 🛣️ Dwukierunkowa (2 pasy) | HTTP/2, WebSocket, MCP (`<10 ms`) | Mappings, API Keys, mTLS | [`nethical/gateway/proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py) (`GatewayProxy`) |
| **TR-02** | `Gateway Proxy` | `Token Vault` | 🛑 Kontrola Graniczna | Synchroniczny Call (`<0.5 ms`) | Odwracalne szyfrowanie AES-256-GCM PII/ePHI | [`nethical/security/token_vault.py`](file:///c:/Projekty/Nethical/nethical/security/token_vault.py) (`TokenVault`) |
| **TR-03** | `Token Vault` | `Inoculation Mesh` | 🛑 Kontrola Graniczna | In-Memory Pipeline (`<1 ms`) | Filtracja 6 wektorów ataku Prompt Injection | [`nethical/security/inoculation_mesh.py`](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py) (`InoculationMesh`) |
| **TR-04** | `Inoculation Mesh` | `Rondo 25 Praw` | 🔄 Wjazd na Rondo | In-Memory Kernel (`<2 ms`) | Ewaluacja 25 Praw i Praw Podstawowych | [`nethical/core/fundamental_laws.py`](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py) (`FundamentalLawsEngine`) |
| **TR-05** | `Rondo 25 Praw` | `Z3 SMT Solver` | 🔄 Weryfikacja Niezmienników | Formal SMT IPC (`<15 ms`) | Z3 SMT Solver / Matematyczny dowód braku sprzeczności | [`nethical/core/policy_formalization.py`](file:///c:/Projekty/Nethical/nethical/core/policy_formalization.py) (`FormalPolicyVerifier`) |
| **TR-06** | `Rondo 25 Praw` | `Bramki Sektorowe` | 🛣️ Zjazd A (Zielony) | Zsynchronizowane wywołanie pakietu | Dynamiczne reguły MDR / NATO / KPA / ISO | [`nethical/compliance/packs/`](file:///c:/Projekty/Nethical/nethical/compliance/packs/) |
| **TR-07** | `Rondo 25 Praw` | `Kolejka HITL` | 🚧 Zjazd B (Bocznica) | Kolejka Async (Hold State) | Blokada aktywacji do manualnej autoryzacji | [`nethical/gateway/hitl.py`](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py) (`HITLManager`) |
| **TR-08** | `Rondo 25 Praw` | `Kill-Switch / E-STOP`| 🚨 Zjazd C (Szyna Awaryjna) | Sygnał deterministyczny (`<50 µs`) | Natychmiastowy interlock, E-STOP procesów | [`nethical/core/kill_switch.py`](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py) (`KillSwitch`) |
| **TR-09** | `Bramki Sektorowe` | `Model LLM / Actuator`| 🛣️ Zjazd Docelowy | REST / gRPC Target LLM | Zero Data Leakage, Enclave Attestation | [`nethical/gateway/proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py) (`GatewayProxy.forward`) |
| **TR-10** | `Wszystkie Węzły` | `Merkle-DAG Ledger` | 🏹 Jednokierunkowa (WORM) | Asynchroniczny Append-Only | Łańcuch bloków Merkle-DAG z dowodem niezmienności | [`nethical/security/merkle_ledger.py`](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py) (`MerkleLedger`) |
| **TR-11** | `Merkle Ledger` | `PQC Signer` | 🏹 Pieczęć Postkwantowa | Krypto-akceleracja (`<5 ms`) | NIST FIPS 204 ML-DSA-65 / Dilithium | [`nethical/security/quantum_crypto.py`](file:///c:/Projekty/Nethical/nethical/security/quantum_crypto.py) (`QuantumCryptoSigner`) |
| **TR-12** | `PQC Signer` | `Master Dossier Hub` | 🏹 Generowanie Poświadczeń | Standaryzowane paczki JSON/MD | Automatyczne Dossier akredytacyjne dla TÜV/BSI | [`nethical/compliance/automated_certification_hub.py`](file:///c:/Projekty/Nethical/nethical/compliance/automated_certification_hub.py) (`AutomatedCertificationHub`) |
| **TR-13** | `HITL & Ledger` | `Ambassador Engine` | 🧪 Tor Asymilacji | Asynchroniczny strumień batch | Generowanie par preferencyjnych DPO (Direct Preference) | [`nethical/ambassador/learning.py`](file:///c:/Projekty/Nethical/nethical/ambassador/learning.py) (`AmbassadorLearningEngine`) |

[⬆ Powrót do spisu treści](#spis-treści)

---

## 4. Mechanizmy Działania i Cykl Życia Zapytania (Life of a Request)

### Scenariusz A: Zielona Fala (Zgodny Przejazd)
1. **Wjazd:** Klient wysyła prompt z danymi analitycznymi przez [api.py](file:///c:/Projekty/Nethical/nethical/api.py) do [GatewayProxy](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py).
2. **Kontrola Graniczna:** 
   - [TokenVault](file:///c:/Projekty/Nethical/nethical/security/token_vault.py) wykrywa numery PESEL / SSN i podmienia je na tokeny kryptograficzne `[TOKEN-AES-9941]`.
   - [InoculationMesh](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py) weryfikuje brak prób jailbreaku / prompt injection.
3. **Rondo 25 Praw:** Silnik [FundamentalLawsEngine](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py) sprawdza zapytanie względem 25 Praw. Brak sprzeczności ➔ Z3 SMT zwraca `SATISFIABLE`.
4. **Bramka Sektorowa:** Odpowiedni pakiet branżowy (np. [ISO 42001](file:///c:/Projekty/Nethical/nethical/compliance/packs/iso42001_pack.py)) poświadcza zgodność.
5. **Aktywacja:** Zapytanie trafia do modelu LLM.
6. **Powrót:** Model generuje odpowiedź; Gateway odwraca tokeny (detokenizacja) i zwraca czysty, bezpieczny wynik klientowi.
7. **Księga Dowodowa:** [MerkleLedger](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py) jednokierunkowo rejestruje dowód wykonania i kotwiczy hash w łańcuchu.

---

### Scenariusz B: Zjazd na Bocznicę HITL (Niejednoznaczność lub DIR 4/5)
1. **Wykrycie Ryzyka:** System oblicza wskaźnik DIR (Decision Impact & Risk) na poziomie 4 lub 5 (np. decyzja kredytowa, ocena triażu medycznego, wniosek administracyjny).
2. **Zjazd na Bocznicę:** Rondo Decyzyjne nie wypuszcza zapytania do modelu. Zapytanie otrzymuje status `PENDING_HUMAN_APPROVAL` i zjeżdża do [hitl.py](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py).
3. **Powiadomienie SRO / Rewidenta:** Operator widzi zatrzymany pojazd w portalu audytowym ([portal/api.py](file:///c:/Projekty/Nethical/portal/api.py)).
4. **Decyzja:**
   - **Zezwolenie:** Dyżurny ruchu zatwierdza akcję kluczem kwalifikowanym ➔ powrót na autostradę i realizacja.
   - **Odrzucenie:** Zlecenie zostaje skierowane do [quarantine.py](file:///c:/Projekty/Nethical/nethical/core/quarantine.py) ze szczegółowym uzasadnieniem faktyczno-prawnym.

---

### Scenariusz C: Zderzenie z Prawem 1/2 i Szyna E-STOP (<50 µs)
1. **Naruszenie Krytyczne:** Model próbuje wywołać narzędzie stwarzające bezpośrednie zagrożenie dla życia/zdrowia człowieka (naruszenie Prawa 1) lub zaniechać krytycznej ochrony medycznej (zakaz autonomicznego DNR w pakiecie MDR).
2. **Aktywacja Szyny Awaryjnej:** Rondo natychmiast odrzuca zapytanie.
3. **Wyzwalacz Sprzętowy:** [HardwareWatchdog](file:///c:/Projekty/Nethical/nethical/security/watchdog.py) oraz [KillSwitch](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py) wysyłają sygnał przerwania w czasie **<50 mikrosekund**.
4. **Skutki:**
   - Natychmiastowe zamrożenie i zrzucenie procesu agenta.
   - Zapisanie niezmiennego dowodu przestępstwa / incydentu w [MerkleLedger](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py) z podpisem FIPS 204.
   - Wygenerowanie raportu incydentu dla organu nadzoru (CSIRT, UODO, Jednostka Notyfikowana).

[⬆ Powrót do spisu treści](#spis-treści)

---

## 5. Interaktywny Mega-Katalog Modułów, Klas i Połączeń

Poniższy katalog zawiera wszystkie kluczowe komponenty Nethical uporządkowane według warstw systemu z bezpośrednimi odnośnikami do kodu:

### Warstwa 1: Wjazd i Bramki Dostępowe (Ingress & Gateways)

* [`nethical/api.py`](file:///c:/Projekty/Nethical/nethical/api.py)
  * **Główne Klasy / Punkty Wejścia:** `create_app()`, `GovernanceRouter`, endpointy `/v1/governance/*`, `/v1/audit/*`.
  * **Wjeżdżają:** Zewnętrzne aplikacje klienckie, mikroserwisy przedsiębiorstwa, panele webowe.
  * **Wyjeżdżają:** Do [GatewayProxy](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py) i kolejki ewaluacyjnej.
  * **Rola Drogowa:** Główny terminal wjazdowy autostrady (bramki autostradowe).

* [`nethical/gateway/proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py)
  * **Główne Klasy:** `GatewayProxy`, `ProxyRequest`, `ProxyResponse`.
  * **Wjeżdżają:** Surowy ruch z API i gniazd agentów.
  * **Wyjeżdżają:** Do modułów ochrony obwodowej ([TokenVault](file:///c:/Projekty/Nethical/nethical/security/token_vault.py), [InoculationMesh](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py)).
  * **Rola Drogowa:** Centralny dyspozytor ruchu i rozjazd kierunkowy.

* [`nethical/mcp_server.py`](file:///c:/Projekty/Nethical/nethical/mcp_server.py) oraz [`nethical/gateway/mcp_proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/mcp_proxy.py)
  * **Główne Klasy:** `NethicalMCPServer`, `MCPToolInterceptor`.
  * **Wjeżdżają:** Połączenia od agentów operujących protokołem Model Context Protocol (Anthropic, OpenAI itp.).
  * **Wyjeżdżają:** Do reguł walidacji narzędzi i pre-actuation checks.
  * **Rola Drogowa:** Dedykowany pas wjazdowy dla agentów MCP.

* [`nethical/cli.py`](file:///c:/Projekty/Nethical/nethical/cli.py)
  * **Główne Klasy:** `NethicalCLI`, komendy CLI (`nethical audit`, `nethical certify`, `nethical start`).
  * **Rola Drogowa:** Terminal wjazdowy dla inżynierów i audytorów.

---

### Warstwa 2: Inspekcja Graniczna i Ochrona Obwodowa (Perimeter & Sanitization)

* [`nethical/security/token_vault.py`](file:///c:/Projekty/Nethical/nethical/security/token_vault.py)
  * **Główne Klasy:** `TokenVault`, `VaultEntry`.
  * **Mechanizm:** Odwracalne, kryptograficzne mapowanie danych PII/ePHI/Tajemnic przedsiębiorstwa na pseudonimy AES-256-GCM.
  * **Rola Drogowa:** Komora celna – paszportyzacja i depozyt danych wrażliwych przed wpuszczeniem na rondo.

* [`nethical/security/inoculation_mesh.py`](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py)
  * **Główne Klasy:** `InoculationMesh`, `AttackVectorDetector`.
  * **Mechanizm:** Neutralizacja wstrzykiwania promptów (jailbreak, indirect prompt injection, data poisoning).
  * **Rola Drogowa:** Skaner rentgenowski pojazdów i ładunków.

* [`nethical/security/aispm_scanner.py`](file:///c:/Projekty/Nethical/nethical/security/aispm_scanner.py)
  * **Główne Klasy:** `AISPMScanner`, `DataPostureReport`.
  * **Mechanizm:** Monitorowanie Shadow AI, otwartych gniazd TCP, portów i procesów agentowych w czasie rzeczywistym.
  * **Rola Drogowa:** Fotoradar i system monitoringu bezpieczeństwa autostrady.

---

### Warstwa 3: Centralne Rondo Decyzyjne „25 Praw” (Laws Kernel & Formal Solver)

* [`nethical/core/fundamental_laws.py`](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py)
  * **Główne Klasy:** `FundamentalLawsEngine`, `FundamentalLaw`, `LawViolation`.
  * **Mechanizm:** 25 Niezmiennych Praw Nethical (Prawa Istnienia, Wolności, Transparentności, Odpowiedzialności, Koegzystencji, Ochrony, Wzrostu).
  * **Rola Drogowa:** Wyspa centralna ronda – żadne zapytanie nie przejedzie bez zgodności z 25 Prawami.

* [`nethical/core/governance_core.py`](file:///c:/Projekty/Nethical/nethical/core/governance_core.py)
  * **Główne Klasy:** `GovernanceCore`, `DecisionContext`, `PolicyEnforcer`.
  * **Mechanizm:** Orzekanie i wyliczanie wskaźników ryzyka DIR (Decision Impact & Risk) od 1 do 5.
  * **Rola Drogowa:** Skrzyżowanie z sygnalizacją świetlną (Zielone / Żółte / Czerwone).

* [`nethical/core/policy_formalization.py`](file:///c:/Projekty/Nethical/nethical/core/policy_formalization.py) oraz [`formal/`](file:///c:/Projekty/Nethical/formal/)
  * **Główne Klasy:** `FormalPolicyVerifier`, modele Z3 SMT, specyfikacje TLA+ i Lean.
  * **Mechanizm:** Matematyczny dowód braku sprzeczności reguł.
  * **Rola Drogowa:** Weryfikacja techniczna stanu pojazdu przed dopuszczeniem do ruchu.

---

### Warstwa 4: Rozjazd Sektorowy (Sectoral Compliance Toll Booths)

Znajdują się w katalogu [`nethical/compliance/packs/`](file:///c:/Projekty/Nethical/nethical/compliance/packs/):

* [`iso42001_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/iso42001_pack.py) – Globalne standardy AIMS i EU AI Act.
* [`healthcare_med_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/healthcare_med_pack.py) – Wymogi SaMD Rule 11, ISO 14971, zakaz DNR, ochrona triażu SOR.
* [`nato_defense_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/nato_defense_pack.py) – 6 Zasad PRU obronności NATO, zero-egress, izolacja radiowa.
* [`public_admin_gov_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/public_admin_gov_pack.py) – Polskie KPA (Art. 7, 107 – zakaz orzekania z czarnej skrzynki) oraz KRI.
* [`poland_sovereign_ksc_uodo_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/poland_sovereign_ksc_uodo_pack.py) – Tarcza zarządu Business Judgment Rule (Art. 293/483 KSH) i KSC.
* [`uk_cyber_data_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/uk_cyber_data_pack.py) – Zgodność z UK Computer Misuse Act, UK NIS i Teal Book GovS 002.
* [`canada_aida_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/canada_aida_pack.py) – Kanadyjska ustawa AIDA (Bill C-27) i ochrona przed biasem CHRA.
* [`academic_research_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/academic_research_pack.py) – Kodeks ALLEA, walidacja bibliografii (brak halucynacji DOI), granty NCN/ERC.

---

### Warstwa 5: Skład Niezmienny (Merkle Ledger & Post-Quantum FIPS 204 Signer)

* [`nethical/security/merkle_ledger.py`](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py)
  * **Główne Klasy:** `MerkleLedger`, `MerkleBlock`, `Receipt`.
  * **Mechanizm:** Ciągły łańcuch dowodowy Merkle-DAG o architekturze WORM (Write Once Read Many).
  * **Rola Drogowa:** Czarna skrzynka systemu drogowego – ruch wyłącznie w jedną stronę (bezzwrotny zrzut zdarzeń).

* [`nethical/security/quantum_crypto.py`](file:///c:/Projekty/Nethical/nethical/security/quantum_crypto.py)
  * **Główne Klasy:** `QuantumCryptoSigner`, `DilithiumKeyPair`, `QuantumSignature`.
  * **Mechanizm:** Implementacja standardu NIST FIPS 204 ML-DSA-65 (kryptografia odporna na komputery kwantowe).
  * **Rola Drogowa:** Tłocznia pieczęci niemożliwych do podrobienia.

* [`nethical/compliance/automated_certification_hub.py`](file:///c:/Projekty/Nethical/nethical/compliance/automated_certification_hub.py)
  * **Główne Klasy:** `AutomatedCertificationHub`, `AutomatedEvidencePackage`.
  * **Rola Drogowa:** Cyfrowe biuro wydawania paszportów i certyfikatów homologacji pojazdów dla instytucji państwowych i akredytowanych biegłych.

---

### Warstwa 6: Bocznica Nadzoru Człowieka i Kwarantanna (HITL & Quarantine Siding)

* [`nethical/gateway/hitl.py`](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py)
  * **Główne Klasy:** `HITLManager`, `ReviewQueueItem`, `ApprovalTicket`.
  * **Rola Drogowa:** Bocznica inspekcyjna – zatrzymanie pojazdu na bezpiecznym torze bocznym do czasu weryfikacji przez człowieka.

* [`nethical/core/quarantine.py`](file:///c:/Projekty/Nethical/nethical/core/quarantine.py)
  * **Główne Klasy:** `QuarantineZone`, `IsolationPolicy`.
  * **Rola Drogowa:** Parking depozytowy dla pojazdów niespełniających norm bezpieczeństwa.

* [`nethical/security/watchdog.py`](file:///c:/Projekty/Nethical/nethical/security/watchdog.py) oraz [`nethical/core/kill_switch.py`](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py)
  * **Główne Klasy:** `HardwareWatchdog`, `KillSwitch`, `EmergencyInterlock`.
  * **Rola Drogowa:** Kolejowa szyna awaryjna i hamulec bezpieczeństwa E-STOP (<50 µs).

---

### Warstwa 7: Tor Uczenia Maszynowego i Adaptacji (Ambassador & DPO Engine)

* [`nethical/ambassador/learning.py`](file:///c:/Projekty/Nethical/nethical/ambassador/learning.py)
  * **Główne Klasy:** `AmbassadorLearningEngine`, `DPOPairGenerator`, `RepositoryAssimilator`.
  * **Rola Drogowa:** Centrum badawczo-rozwojowe (Proving Ground) – analiza danych z wypadków i udanych przejazdów w celu ulepszania nawierzchni i reguł drogowych.

* [`data/ambassador_dpo_dataset.jsonl`](file:///c:/Projekty/Nethical/data/ambassador_dpo_dataset.jsonl)
  * **Rola:** Zbiór par preferencyjnych `(prompt, chosen, rejected)` zasilany z rzeczywistego ruchu drogowego.

[⬆ Powrót do spisu treści](#spis-treści)

---

> **Nethical Enterprise OS v2.5** – *Wszystkie prawa zastrzeżone. Schemat autostradowy jest chroniony cyfrową pieczęcią niezmienności Merkle-DAG.*
