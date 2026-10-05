# 🧭 Nethical: Raport Zgodności Kodu z Dokumentacją i Plan Zadań Niedokończonych

**Data audytu:** Październik 2026  
**Wersja frameworka:** 2.7.x  
**Zasada przewodnia:** *Zero fałszywych obietnic, 100% prawdy inżynieryjnej. Nethical dotrzymuje tego, co deklaruje.*

---

## 1. Analiza Strategiczna: Sprinto (go.sprinto.com) vs. Dokumentacja Własna (In-House)

### Czym jest Sprinto?
Sprinto to komercyjna platforma klasy **SaaS GRC (Governance, Risk, and Compliance)** służąca do automatyzacji audytów ogólnego bezpieczeństwa IT w firmach technologicznych. Łączy się z dostawcami chmurowymi (AWS, GCP, Azure, GitHub, Okta, MDM) i sprawdza stan konfiguracji (np. czy repozytoria mają włączoną ochronę gałęzi, czy laptopy pracowników są szyfrowane, czy wdrożono MFA). Służy głównie do uzyskania pieczęci **SOC 2 (Type 1 & 2)** oraz **ISO/IEC 27001** na potrzeby sprzedaży B2B w USA.

### Czego Sprinto NIE robi i dlaczego nie jest właściwym narzędziem dla Nethical na tym etapie?
1. **Brak wsparcia dla prawa i bezpieczeństwa AI:** Sprinto nie obsługuje wymogów **EU AI Act (Rozporządzenie 2024/1689)**, normy **ISO/IEC 42001 (AIMS)**, **NIST AI RMF**, ani **Cyber Resilience Act (CRA)**. Nie weryfikuje odporności na jailbreaki, halucynacji, alignmentu modeli ani poprawności dowodów matematycznych.
2. **Brak audytu licencji Open-Source i łańcucha dostaw kodu:** Sprinto nie bada czystości licencyjnej pakietów Pythona (np. ryzyka zakażenia licencją GPL/AGPL wobec MIT), nie generuje SBOM CycloneDX/SPDX dla modeli, ani nie tworzy kryptograficznych atestacji SLSA Level 3.
3. **Nethical sam w sobie jest silnikiem Governance:** Nasz framework zawiera dedykowane generatory dokumentacji technicznej i zgodności prawnej (`nethical/compliance/conformity_generator.py`, `eu_ai_act.py`, rejestry Merkle-DAG, audyty kryptograficzne). Korzystanie z zewnętrznego narzędzia do weryfikacji Nethical byłoby odwróceniem ról.
4. **Wysoki koszt:** Koszt licencji Sprinto to od 8 000 do 30 000+ USD rocznie, co dla projektu open-source/on-premise stanowi nieuzasadniony wydatek bez wartości dodanej dla kodu.

### 📌 Rekomendacja:
> **Zajmujemy się certyfikatami, licencjami i dokumentacją osobiście i natywnie w repozytorium.**  
> Do platform typu Sprinto/Vanta wrócimy wyłącznie w scenariuszu powołania komercyjnej spółki oferującej Nethical jako zarządzaną chmurę multi-tenant (Hosted Cloud SaaS), w momencie gdy amerykańscy klienci korporacyjni zażądają formalnego raportu SOC 2 Type II od certyfikowanego audytora CPA.

---

## 2. Głęboki Audyt: Dokumentacja vs. Rzeczywistość w Kodzie

Przeprowadziliśmy bezwzględną weryfikację deklaracji w dokumentacji (`README.md`, `roadmaps/roadmap.md`, `docs/`, `GOVERNANCE.md`) z faktyczną implementacją w kodzie (`nethical/`).

### Tabela A: Komponenty w 100% Potwierdzone w Kodzie (Sukcesy)

| Moduł / Deklaracja | Lokalizacja w Kodzie | Stan Implementacji | Weryfikacja Testowa |
|---|---|---|---|
| **25 Fundamental Laws & Wektorowy Mapper** | `nethical/core/fundamental_laws.py`, `semantic_mapper.py` | Pełna deonotologia, mapowanie na prymitywy, kalibracja `raw_cosine`. | 22/22 testów pass w `tests/test_vector_language.py` |
| **Formalna Weryfikacja Matematyczna Z3 SMT** | `nethical/formal/law_prover.py`, `verify_rfc.py` | Prawdziwy solver Microsoft Z3 (`5.1.0.0`), dowodzenie niesprzeczności i niezmienników bezpieczeństwa. | Testy w `tests/test_formal_ebpf_and_enclave.py` pass |
| **Financial Circuit Breaker** | `nethical/security/financial_circuit_breaker.py` | 4 stany (`NORMAL` -> `THROTTLED` -> `TRIPPED` -> `HALTED`), limity wolumenu i velocity, wpięty w proxy. | Testy w `test_governance_gateway.py` pass |
| **Reversible TokenVault** | `nethical/security/token_vault.py` | Prawdziwe szyfrowanie AES-256-GCM, in-flight maskowanie PII, rotacja kluczy. | Testy kryptograficzne pass |
| **Detekcja Kanałów Ukrytych (Teoria Informacji)** | `nethical/detectors/embedding/covert_channel_detector.py` | Entropia Shannona $H(X)$, współczynnik kompresji zlib, znaki zero-width Unicode. Brak false positives. | 4/4 testów pass w `test_covert_channel_detector.py` |
| **Stepping Stone & Purdue Model** | `nethical/security/stepping_stone_guard.py` | Inspekcja poziomów Purdue 3/4, korytarze proxy rezydencjalnych, badanie entropii pakietów. | 5/5 testów pass w `test_stepping_stone_guard.py` |
| **Autentykacja Enterprise & Multi-Tenancy** | `nethical/security/auth.py`, `mfa.py`, `sso.py`, `tenant_manager.py` | JWT, API keys, TOTP MFA, SSO SAML, 4 dedykowane tenanty. | Ponad 70 testów pass |
| **Sovereign Control Plane (HITL Web UI)** | `portal/templates/index.html`, `nethical/api/hitl_api.py` | Nowoczesny kokpit bez zewnętrznych CDN, podgląd Merkle ledger, przełącznik tenantów, radar ryzyka. | Zintegrowany z API FastAPI |
| **Infrastruktura Wdrożeniowa (Helm & Terraform)** | `deploy/helm/nethical/`, `deploy/helm/nethical-edge/`, `deploy/terraform/` | Pełne manifesty Kubernetes, wartości dev/prod, moduły Terraform dla AWS/GCP/Azure. | Gotowe do wdrożenia |
| **Nethical CLI** | `nethical/cli.py` | CLI oparte o `click` z komendami `init`, `evaluate`, `status`, `serve`, `verify-plugin`, `ambassador`. | Zarejestrowane w `pyproject.toml` |

---

### Tabela B: Rozbieżności, Symulacje i "Over-Promising" (Do Naprawy)

Poniższe elementy zostały zidentyfikowane jako rozbieżne z deklaracjami w dokumentacji. Dokumentacja przedstawia je jako fizyczną/sprzętową rzeczywistość, podczas gdy kod implementuje je jako **programowe modele/symulatory**:

| # | Komponent | Deklaracja w Dokumentacji / README | Rzeczywisty Stan w Kodzie | Klasyfikacja Ryzyka |
|---|---|---|---|---|
| **1** | **Kryptografia Postkwantowa (ML-DSA-65)** | *"Post-quantum Merkle-DAG ledgers signed with NIST FIPS 204 ML-DSA-65 algorithms"* | W `nethical/security/post_quantum.py:259` klasa nazywa się `SimulatedMLDSA` i bazuje na SHA-256 HMAC z jawnym komentarzem: *"NOT a real ML-DSA verification (requires liboqs for production)"*. | **ŚREDNIE / ETYCZNE**: Należy wprost opisać to jako referencyjny symulator lub wdrożyć opcjonalną bibliotekę `liboqs`. |
| **2** | **Magistrale Przemysłowe (CAN, Modbus, EtherCAT E-Stop)** | *"Hardware determinism (<50 µs)... CAN Bus hardware shutdown (EMCY 0x080), Modbus de-energize... ISO 13849-1 PL-e Cat 4 & ISO 26262 ASIL-D"* | W `nethical/edge/industrial_fieldbus.py` kod w Pythonie tworzy obiekty Pydantic `CANFrame` i `ModbusCommand` i zapisuje je w tablicy w pamięci (`self.can_frames_log.append(...)`). Brak połączenia z fizycznym interfejsem `socketcan` czy portem RS-485. | **WYSOKIE**: Deklarowanie zgodności z normami sprzętowymi PL-e / ASIL-D bez certyfikowanego hardware'u to over-promising. |
| **3** | **Filtrowanie Jądra Linuxa (eBPF Kernel Drop)** | *"eBPF transparent kernel-level network socket drops"* | W `nethical/edge/ebpf_interceptor.py` działa tryb `USERSPACE_SIMULATOR`. Kod C jest zdefiniowany jako string, ale nie jest kompilowany ani ładowany przez jądro. Filtrowanie odbywa się w Pythonie. | **ŚREDNIE**: Wymaga jasnego sprecyzowania, że to warstwa emulacyjna userspace z opcjonalnym modułem jądra. |
| **4** | **Atestacja Sprzętowa TEE (AMD SEV, Intel SGX)** | *"Hardware Remote Attestation Quote potwierdzający nienaruszalność kodu w RAM"* | W `nethical/security/enclave_attestation.py` działa symulator generujący skróty SHA-384 bez fizycznej komunikacji z `/dev/sev-guest` czy sterownikiem SGX. | **ŚREDNIE**: Należy skorygować nazewnictwo na "Emulowany mechanizm poświadczeń TEE". |
| **5** | **Detektory-Atrapy (Stub Detectors)** | Deklarowane w taksonomii jako pełne detektory naruszeń bezpieczeństwa. | W `nethical/core/governance_detectors.py:650-780`: <br>• `HallucinationDetector`: sprawdza tylko frazę `"i am certain"`!<br>• `MisinformationDetector`: sprawdza `"everyone knows"` lub `"undisputed fact"`.<br>• `ToxicContentDetector`: 3 słowa na krzyż (`"idiot"`, `"stupid"`, `"trash"`).<br>• `ModelExtractionDetector`: fraza `"model weights"`.<br>• `DataPoisoningDetector`: naiwny stosunek tokenów. | **WYSOKIE**: Te detektory nie spełniają standardów produkcyjnych i wymagają natychmiastowej modernizacji. |
| **6** | **Multimodal Embeddings** | Deklarowane wsparcie dla wielomodalności w UVL. | W `nethical/core/multimodal_embeddings.py:331-337` metody rzucają: `raise NotImplementedError("Image embeddings not yet implemented")` i `Audio embeddings not yet implemented`. | **ŚREDNIE**: Kod rzuca wyjątek przy próbie użycia obrazu/audio. |
| **7** | **Statusy w Roadmapie** | `roadmaps/roadmap.md` zawiera nieodznaczone pozycje, które są już gotowe. | `CONTRIBUTING.md`, `GOVERNANCE.md` (charter TSC) oraz `nethical CLI` są w pełni gotowe, ale widnieją w roadmapie jako `[ ]`. | **NISKIE / DOKUMENTACYJNE**: Wymaga prostej synchronizacji. |
| **8** | **Katalog RFC** | `nethical/formal/verify_rfc.py` formalnie weryfikuje `RFC-0001`. | Brak katalogu `rfcs/` lub `docs/rfcs/` z faktycznymi plikami markdown opisującymi RFC. | **NISKIE**: Wymaga utworzenia szablonu i pliku RFC-0001. |

---

## 3. Nowy Plan Działań Naprawczych (Action Plan v2.8)

Poniższy plan eliminuje rozbieżności, zamyka luki w kodzie i doprowadza dokumentację do pełnej prawdomówności:

### 🎯 Pakiet 1: Uczciwość w Dokumentacji (Zero Over-Promising)
1. **Doprecyzowanie warstw symulacyjnych w `README.md` oraz `docs/SOVEREIGN_AI_PILLARS.md`:**
   - Wyraźne oznaczenie komponentów:
     - `IndustrialFieldbusInterlock` -> jako **Hardware-in-the-Loop (HIL) Protocol Simulator** z gotowością do podpięcia driverów magistrali.
     - `EBPFAgentInterceptor` -> jako **Transparent Userspace Policy Emulator** z dołączonym kodem źródłowym C dla instalacji w jądrze.
     - `PostQuantum` -> jako **Cryptographic Protocol Skeleton (SHA-256 HMAC Reference)** z opcją podpięcia `liboqs`.
     - `EnclaveAttestation` -> jako **Software Attestation Mock**.
   - Usunięcie kategorycznych twierdzeń o uzyskanej certyfikacji sprzętowej `ISO 13849 PL-e Cat 4` i `ISO 26262 ASIL-D` dla czystego kodu Pythona (zastąpienie: "Zaprojektowane według wytycznych architektonicznych ISO 13849/26262").

### 🎯 Pakiet 2: Modernizacja Detektorów-Atrap (`governance_detectors.py`)
1. **Refaktoryzacja `HallucinationDetector`:** Zastąpienie frazy `"i am certain"` algorytmem sprawdzania spójności semantycznej i confidence-score (np. entropia odpowiedzi lub cross-check z bazą wiedzy).
2. **Refaktoryzacja `ToxicContentDetector`:** Zastąpienie 3 słów zbiorem leksykalnym z wagami lub klasyfikatorem embeddingowym (podobnie jak w `CovertChannelDetector`).
3. **Refaktoryzacja `ModelExtractionDetector` & `DataPoisoningDetector`:** Zastosowanie metryk statystycznych rozkładu tokenów i wzorców rekurencyjnych zapytań o wagi/architekturę.

### 🎯 Pakiet 3: Uporządkowanie Modułu Wielomodalnego (`multimodal_embeddings.py`)
1. Usunięcie `NotImplementedError` i zastąpienie go bezpiecznym, jawnym statusem:
   - Jeśli brak zainstalowanych bibliotek wizyjnych (np. `torchvision`/`onnxruntime`), metoda zwraca elegancki `EmbeddingResult(status="unsupported_modality")` zamiast wywalać aplikację wyjątkiem.
   - Dodanie lekkiego backendu ONNX dla prostych embeddingów obrazu (np. MobileNet/CLIP w ONNX).

### 🎯 Pakiet 4: Porządki w Roadmapie i Formalny Rejestr RFC
1. Zaktualizowanie checkboxów w `roadmaps/roadmap.md` dla:
   - `CONTRIBUTING.md` -> ✅ COMPLETED
   - `GOVERNANCE.md` (TSC Charter) -> ✅ COMPLETED
   - `nethical CLI` -> ✅ COMPLETED
2. Utworzenie katalogu `rfcs/` z plikiem `RFC-0001-deprecated-binary-curves.md` dokumentującym formalny dowód Z3 dla wycofania podatnych krzywych eliptycznych.

---

## 4. Harmonogram Wdrożenia

| Faza | Zakres | Czas realizacji | Odpowiedzialny |
|---|---|---|---|
| **Krok 1** | Aktualizacja `roadmaps/roadmap.md` oraz utworzenie `rfcs/RFC-0001-deprecated-binary-curves.md` | Natychmiast (dzisiaj) | Core Agent |
| **Krok 2** | Korekta `README.md` pod kątem uczciwego nazewnictwa symulatorów HIL/eBPF/PQC | Natychmiast (dzisiaj) | Core Agent |
| **Krok 3** | Przebudowa atrap w `nethical/core/governance_detectors.py` na solidne detektory | Sprint bieżący | Core Agent |
| **Krok 4** | Bezpieczny fallback w `multimodal_embeddings.py` bez `NotImplementedError` | Sprint bieżący | Core Agent |
