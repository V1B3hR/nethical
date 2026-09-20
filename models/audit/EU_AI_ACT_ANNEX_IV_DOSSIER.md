# EU AI ACT ANNEX IV TECHNICAL DOCUMENTATION
**System Name:** Nethical Enterprise OS & Błyskawica Ambassador  
**Version:** v10.4-sovereign  
**Regulation:** Regulation (EU) 2024/1689 (EU AI Act)  
**Classification:** High-Risk AI System / General Purpose AI Governance  
**Generated At:** `2026-09-20T16:09:48.421384+00:00`  
**Merkle Anchor Root:** `b6cf3693c0838358cf5e46b62c73fa72705435e351ff7121902f695d725e4447`  

---

## 1. General System Description
- **Intended Purpose:** Autonomiczne zarządzanie ładem etycznym i suwerenna weryfikacja zachowań modeli AI.
- **Architektura:** Dwuskładnikowy węzeł suwerenny (Gateway FastAPI + Sidecar Błyskawica Rust Core przez IPC).
- **Transport IPC:** Pamięć współdzielona (`emptyDir` Memory), opóźnienie < 0.5 ms.

## 2. Metodologia Uczenia i Algorytmy (DPO + Kalman + AcceleratorAI)
- **Algorytm:** Direct Preference Optimization (DPO) na preferencjach Bradley-Terry.
- **Termostat Kalmana:** Dynamiczne skalowanie kary $\beta$ proporcjonalnie do estymowanego odchylenia/wątpliwości.
- **Pneumatyczne tłumienie gradientów:** Tanh soft-clipping usuwający eksplozje gradientowe na RTX 4070.
- **Ochrona przed zapominaniem:** `ContinuousReplayBuffer` stale wplatający 25 Praw Fundamentalnych.
- **Metryki uczenia:**
  - Redukcja straty: $0.47456 \rightarrow 0.31352$ (-33.9%)
  - Rozszerzenie marginesu nagrody: $0.72025 \rightarrow 1.87582$ (2.6x)
  - Przepustowość: 25 795 tokenów/s na NVIDIA RTX 4070.

## 3. Zarządzanie Danymi i Pochodzenie (Data Governance)
- **Rozmiar zbioru:** 4195 zweryfikowanych par preferencji.
- **Archetypy:** 20 rygorystycznych archetypów behawioralnych (w tym ochrona przed sycophancy i dark nudging).
- **Zgodność antydyskryminacyjna:** Four-Fifths Rule (DIR > 0.80) w 100% domen.

## 4. Nadzór Ludzki i Prawa Podstawowe (Human Oversight)
- **Prawo 21:** Bezwzględne poszanowanie wolnej woli i autonomii kognitywnej człowieka.
- **Affective Safety:** 100% eliminacja manipulacji emocjonalnej i uległości.
- **Prawo 25:** Sprzętowy wyłącznik awaryjny (Circuit Breaker) < 1.0 ms.

## 5. Cyberbezpieczeństwo i Odporność Bojowa (Swarm Arena)
- **Profilowanie Prędkości:** Rozróżnianie botów zalewowych (`BURST_ULTRA_FAST`) od deliberatywnych modeli (`METHODICAL_DEEP`).
- **Profilowanie Inteligencji:** Od skryptów Tier 1 po skoordynowaną zmowę roju Tier 4.
- **Obrona Bizantyjska:** Automatyczna kwarantanna wieloagentowych grup zmawiających się.
- **Kryptografia:** Post-quantum Merkle-DAG z pieczęcią stanu modelu.
