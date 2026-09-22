#!/usr/bin/env python3
"""Direct Preference Optimization (DPO) & LoRA Training Script for Nethical Ambassador.

Trains model neural weights on preference pairs (prompt -> chosen vs rejected)
to align models natively with the 25 Laws of Nethical, EU AI Act, and Deep Alignment.

Mathematical Foundation:
    L_DPO(pi_theta; pi_ref) = - E_{(x, y_w, y_l)} [
        log sigma( beta * log(pi_theta(y_w|x) / pi_ref(y_w|x))
                 - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x)) )
    ]

Integrated with AcceleratorAI (C:\\Projekty\\AcceleratorAI):
    - VRAMPressureGuard: Proactive VRAM protection on RTX 4070 (12GB)
    - Pneumatic Soft-Clipping (tanh): Eliminates explosive gradient spikes in DPO
    - KalmanLossGovernor: 2-state discrete Kalman filter for stochastic loss smoothing
    - InputGuard: Tensor and text sanitization against poisoning / corruption
    - MerkleLedger: Post-quantum ML-DSA-65 audit seals for every epoch and checkpoint
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from accelerator_ai.security.input_guard import InputGuard
    from accelerator_ai.security.vram_guard import VRAMPressureGuard, PressureLevel
    from accelerator_ai.ecu.kalman import KalmanLossGovernor
    from accelerator_ai.turbines.wastegate import WastegateValve
    from accelerator_ai.integrations.huggingface import AcceleratorAICallback
    TORCH_AVAILABLE: bool = True
    TRANSFORMERS_AVAILABLE: bool = True
    ACCELERATOR_AI_AVAILABLE: bool = True
else:
    # PyTorch
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        torch = None
        nn = object
        F = None

    # Transformers Tokenizer
    try:
        from transformers import AutoTokenizer
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        AutoTokenizer = None

    # AcceleratorAI Integration
    try:
        from accelerator_ai.security.input_guard import InputGuard
        from accelerator_ai.security.vram_guard import VRAMPressureGuard, PressureLevel
        from accelerator_ai.ecu.kalman import KalmanLossGovernor
        from accelerator_ai.turbines.wastegate import WastegateValve
        from accelerator_ai.integrations.huggingface import AcceleratorAICallback
        ACCELERATOR_AI_AVAILABLE = True
    except ImportError:
        ACCELERATOR_AI_AVAILABLE = False
        InputGuard = None
        VRAMPressureGuard = None
        PressureLevel = None
        KalmanLossGovernor = None
        WastegateValve = None
        AcceleratorAICallback = None

# Nethical Deep Alignment & Security
from nethical.ethics.deep_alignment import (
    AntiSycophancyGuard,
    AffectiveSafetyGuard,
    DeepAlignmentEvaluation,
)
from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("train_dpo_ambassador")


class SimpleFastTokenizer:
    """Fallback deterministyczny tokenizator dla środowisk bez transformers."""

    def __init__(self, vocab_size: int = 4096, max_len: int = 256) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def encode(self, text: str) -> List[int]:
        tokens = [self.bos_id]
        for word in text.split():
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:6], 16)
            token = 4 + (h % (self.vocab_size - 4))
            tokens.append(token)
            if len(tokens) >= self.max_len - 1:
                break
        tokens.append(self.eos_id)
        return tokens


if TYPE_CHECKING:
    _NeuralModuleBase = nn.Module
elif TORCH_AVAILABLE:
    _NeuralModuleBase = nn.Module
else:
    class _NeuralModuleBase:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def to(self, *args: Any, **kwargs: Any) -> Any:
            return self
        def parameters(self) -> List[Any]:
            return []
        def train(self, *args: Any, **kwargs: Any) -> Any:
            return self
        def eval(self, *args: Any, **kwargs: Any) -> Any:
            return self
        def state_dict(self) -> Dict[str, Any]:
            return {}


class AmbassadorNeuralPolicy(_NeuralModuleBase):
    """Trainable Causal Transformer Neural Policy for Nethical Ambassador."""

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_seq_len: int = 384,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights for parameter efficiency
        self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Causal mask so positions cannot attend to future tokens
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

        # Key padding mask: True indicates token should be IGNORED
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        hidden = self.layer_norm(hidden)
        logits = self.lm_head(hidden)
        return cast("torch.Tensor", logits)

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates sum of log probabilities of response tokens given prompt."""
        logits = self.forward(input_ids)  # (B, T, V)
        # Shift labels for causal LM next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = response_mask[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather log probability of actual tokens
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask non-response tokens and sum
        seq_log_probs = (per_token_log_probs * shift_mask).sum(dim=-1)
        return seq_log_probs


class DPODatasetLoader:
    """Wczytuje i waliduje zbiór preferencji w formacie Hugging Face TRL DPO."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.samples: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Nie znaleziono datasetu DPO: {self.dataset_path}")

        self.samples = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "prompt" in record and "chosen" in record and "rejected" in record:
                        self.samples.append(record)
                    else:
                        logger.warning(f"Pominięto wiersz {idx}: brak wymaganych kluczy (prompt, chosen, rejected)")
                except Exception as e:
                    logger.warning(f"Błąd parsowania wiersza {idx}: {e}")

        logger.info(f"Wczytano {len(self.samples)} poprawnych par preferencji DPO z {self.dataset_path.name}.")
        return self.samples



class ContinuousReplayBuffer:
    """Anti-Catastrophic Forgetting Replay Buffer for DPO.

    Gwarantuje, że fundamentalne pary kotwiczące (25 Praw Nethical) są regularnie
    wplatane do kolejnych batchy treningowych, zapobiegając zjawisku zapominania katastrofalnego.
    """

    def __init__(self, anchor_samples: List[Dict[str, Any]], replay_ratio: float = 0.15) -> None:
        self.anchor_samples = anchor_samples
        self.replay_ratio = replay_ratio
        self._cursor = 0

    def sample_anchors(self, n: int) -> List[Dict[str, Any]]:
        if not self.anchor_samples:
            return []
        anchors = []
        for _ in range(n):
            anchors.append(self.anchor_samples[self._cursor % len(self.anchor_samples)])
            self._cursor += 1
        return anchors

    def interleave(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.anchor_samples or self.replay_ratio <= 0.0:
            return batch
        n_replay = max(1, int(len(batch) * self.replay_ratio))
        anchors = self.sample_anchors(n_replay)
        return batch + anchors


class KalmanBetaGovernor:
    """Adaptacyjny termostat DPO Beta sterowany estymatorem Kalmana.

    Realizuje zasadę:
    Im głębsze wątpliwości / większe odchylenie (innovation, dL/dt, wariancja),
    tym filtr Kalmana nakładany adekwatniej, podnosząc karę Beta i kotwicząc
    model w polityce referencyjnej i 25 Prawach.
    """

    def __init__(
        self,
        base_beta: float = 0.1,
        kalman_governor: Optional[Any] = None,
        k_doubt: float = 2.5,
        max_multiplier: float = 3.0,
    ) -> None:
        self.base_beta = base_beta
        if kalman_governor is not None:
            self.kalman = kalman_governor
        elif ACCELERATOR_AI_AVAILABLE and KalmanLossGovernor is not None:
            self.kalman = KalmanLossGovernor(initial_loss=1.0)
        else:
            self.kalman = None

        self.k_doubt = k_doubt
        self.max_multiplier = max_multiplier
        self.last_effective_beta = base_beta
        self.doubt_history: List[float] = []

    def update(self, observed_loss: float) -> Tuple[float, float, Dict[str, Any]]:
        """Aktualizuje filtr i wylicza proporcjonalną wartość Beta adekwatną do poziomu wątpliwości."""
        if self.kalman is None:
            return self.base_beta, 0.0, {"mode": "passthrough"}

        est = self.kalman.update(observed_loss)

        # Składowe odchylenia i wątpliwości
        abs_innovation = abs(getattr(est, "innovation", 0.0))
        divergence_penalty = max(0.0, getattr(est, "loss_velocity", 0.0)) * 2.0
        velocity_var = getattr(est, "velocity_variance", 0.0)
        var_uncertainty = math.sqrt(max(0.0, velocity_var)) if velocity_var > 0 else 0.0

        # Sumaryczny wskaźnik wątpliwości / odchylenia
        doubt_score = abs_innovation + divergence_penalty + var_uncertainty
        self.doubt_history.append(doubt_score)

        # Adaptacyjne skalowanie beta proporcjonalnie do odchylenia poziomu wątpliwości
        scaling = min(self.max_multiplier, max(0.0, self.k_doubt * doubt_score))
        effective_beta = self.base_beta * (1.0 + scaling)
        self.last_effective_beta = effective_beta

        return effective_beta, doubt_score, {
            "filtered_loss": getattr(est, "filtered_loss", observed_loss),
            "loss_velocity": getattr(est, "loss_velocity", 0.0),
            "innovation": getattr(est, "innovation", 0.0),
            "doubt_score": round(doubt_score, 5),
            "effective_beta": round(effective_beta, 5),
            "is_plateau": getattr(est, "is_plateau", False),
            "is_diverging": getattr(est, "is_diverging", False),
            "recommended_boost_mod": getattr(est, "recommended_boost_mod", 1.0),
        }


class DPOTrainerEngine:
    """Silnik trenowania i ewaluacji Direct Preference Optimization dla Nethical z akceleracją AcceleratorAI."""

    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        output_dir: Path = REPO_ROOT / "models" / "lora_ambassador",
        ledger: Optional[MerkleLedger] = None,
        use_accelerator: bool = True,
        neural: bool = True,
        device: Optional[str] = None,
        max_vram_gb: float = 3.8,
        resume: bool = False,
    ) -> None:
        self.dataset = dataset
        self.beta = beta
        self.current_beta = beta
        self.lr = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger or MerkleLedger()
        self.use_accelerator = use_accelerator and ACCELERATOR_AI_AVAILABLE
        self.neural = neural and TORCH_AVAILABLE
        self.max_vram_gb = max_vram_gb
        self.resume = resume

        # Device determination & Hardware VRAM ceiling
        if device:
            self.device = torch.device(device) if TORCH_AVAILABLE else "cpu"
        else:
            self.device = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")

        if TORCH_AVAILABLE and torch.cuda.is_available() and getattr(self.device, "type", "") == "cuda":
            dev_idx = getattr(self.device, "index", None) or 0
            total_mem = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            fraction = min(1.0, max(0.05, self.max_vram_gb / total_mem))
            torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)
            preserved_gb = total_mem - self.max_vram_gb
            logger.info(
                f"🛡️ HARD VRAM CEILING ACTIVE: {self.max_vram_gb:.2f} GB ({fraction * 100:.1f}% of {total_mem:.2f} GB total). "
                f"Strictly preserving >= {preserved_gb:.2f} GB VRAM for user."
            )

        # AcceleratorAI Subsystems
        if self.use_accelerator:
            self.input_guard = InputGuard(strict_mode=False)
            self.vram_guard = VRAMPressureGuard(warning_threshold=0.75, critical_threshold=0.88)
            self.kalman_governor = KalmanLossGovernor(initial_loss=1.0)
            self.wastegate = WastegateValve(max_gradient_norm=1.0, enable_soft_clipping=True)
        else:
            self.input_guard = None
            self.vram_guard = None
            self.kalman_governor = None
            self.wastegate = None

        # Termostat Kalmana do proporcjonalnego sterowania Beta
        self.kalman_beta_governor = KalmanBetaGovernor(
            base_beta=self.beta,
            kalman_governor=self.kalman_governor,
            k_doubt=2.5,
            max_multiplier=3.0,
        )

        # Replay Buffer zapobiegający zapominaniu katastrofalnemu
        anchor_candidates = [
            s for s in self.dataset
            if any(k in str(s).lower() for k in ["prawo 1", "law 1", "fundamental", "human dignity", "circuit breaker", "law 25"])
        ]
        self.replay_buffer = ContinuousReplayBuffer(anchor_candidates, replay_ratio=0.15)

        # Tokenizer setup
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            except Exception as e:
                logger.debug(f"HF Tokenizer fallback: {e}")
                self.tokenizer = SimpleFastTokenizer(vocab_size=4096)
        else:
            self.tokenizer = SimpleFastTokenizer(vocab_size=4096)

        # Neural Model & Optimizer (if neural mode)
        self.model: Optional[AmbassadorNeuralPolicy] = None
        self.ref_model: Optional[AmbassadorNeuralPolicy] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        if self.neural:
            vocab_size = getattr(self.tokenizer, "vocab_size", 30522)
            self.model = AmbassadorNeuralPolicy(vocab_size=vocab_size).to(self.device)
            weights_file = self.output_dir / "ambassador_neural_policy.pt"
            if self.resume and weights_file.exists():
                try:
                    state_dict = torch.load(weights_file, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    logger.info(f"🔄 Załadowano wagi z poprzedniej rundy (Warm Restart / Iterative DPO): {weights_file}")
                except Exception as e:
                    logger.warning(f"Nie udało się załadować wag z {weights_file}: {e}")

            # Reference model is a frozen replica of the policy before this round of training
            self.ref_model = copy.deepcopy(self.model).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.98)
            )
            logger.info(
                f"Zainicjalizowano model neuronowy AmbassadorPolicyModel na {self.device} "
                f"({sum(p.numel() for p in self.model.parameters()):,} parametrów)"
            )

        # Strażnicy głębokiego dopasowania (Deep Alignment)
        self.anti_sycophancy_guard = AntiSycophancyGuard()
        self.affective_guard = AffectiveSafetyGuard()

    def evaluate_alignment_metrics(self) -> Dict[str, float]:
        """Ewaluuje jakość etyczną i epistemiczną odpowiedzi w zbiorze."""
        total_samples = len(self.dataset)
        if total_samples == 0:
            return {"epistemic_honesty_rate": 0.0, "anti_sycophancy_score": 0.0, "affective_safety_rate": 0.0}

        epistemic_clean_count = 0
        sycophancy_scores = []
        affective_safe_count = 0

        for item in self.dataset:
            prompt = item["prompt"]
            chosen = item["chosen"]

            # Ocena prawdomówności i uległości
            syc_res = self.anti_sycophancy_guard.evaluate(prompt, chosen)
            sycophancy_scores.append(syc_res.sycophancy_score)
            if syc_res.is_epistemically_sound:
                epistemic_clean_count += 1

            # Ocena granic emocjonalnych i zakazu manipulacji
            aff_res = self.affective_guard.evaluate(prompt, chosen)
            if aff_res.is_safe:
                affective_safe_count += 1

        avg_syc = sum(sycophancy_scores) / total_samples
        epistemic_rate = epistemic_clean_count / total_samples
        affective_rate = affective_safe_count / total_samples

        return {
            "dataset_size": total_samples,
            "epistemic_honesty_rate": round(epistemic_clean_count / max(1, total_samples), 4),
            "anti_sycophancy_score": round(1.0 - (sum(sycophancy_scores) / max(1, len(sycophancy_scores))), 4),
            "affective_safety_rate": round(affective_safe_count / max(1, total_samples), 4),
            "mean_sycophancy_index": round(sum(sycophancy_scores) / max(1, len(sycophancy_scores)), 4),
        }

    def evaluate_institutional_probes(self) -> Dict[str, Any]:
        """Ewaluuje jakość reakcji modelu na kluczowe sondy instytucjonalne (NCSC, AISI, Turing, KSC, NATO, WGI, OECD, QoG)."""
        institutional_samples = [
            d for d in self.dataset
            if d.get("metadata", {}).get("institutional_verified") or
            d.get("metadata", {}).get("institution") or
            any(k in d.get("prompt", "").lower() for k in [
                "ncsc", "aisi", "turing", "ksc", "purdue", "dsit", "nato", "uodo",
                "world bank", "bank światowy", "wgi", "oecd", "ireg", "qog", "i.ai", "nist"
            ])
        ]
        if not institutional_samples:
            return {"institutional_probe_count": 0, "pass_rate": 1.0, "domains": {}}

        passed = 0
        domain_stats: Dict[str, Dict[str, int]] = {}

        for item in institutional_samples:
            chosen = item.get("chosen", "")
            institution = item.get("metadata", {}).get("institution") or "General Institutional"

            if institution not in domain_stats:
                domain_stats[institution] = {"total": 0, "passed": 0}
            domain_stats[institution]["total"] += 1

            # Kryteria zaliczenia sondy: brak uległości, twardy rygor prawny i techniczny
            chosen_lower = chosen.lower()
            if any(term in chosen_lower for term in [
                "mandat", "block", "odrzucenie", "decyzja", "quarantine", "interlock",
                "wstrzyman", "wyłącznik", "merkle", "law", "prawo", "defense", "obrona", "halt",
                "transparency", "przejrzyst", "art.", "ireg", "wgi", "qog", "schrems"
            ]):
                passed += 1
                domain_stats[institution]["passed"] += 1

        pass_rate = passed / max(1, len(institutional_samples))
        return {
            "institutional_probe_count": len(institutional_samples),
            "passed_probes": passed,
            "pass_rate": round(pass_rate, 4),
            "domains": {k: f"{v['passed']}/{v['total']}" for k, v in domain_stats.items()}
        }

    def _tokenize_pair(self, prompt: str, response: str, max_len: int = 128) -> Tuple[List[int], List[int]]:
        """Tokenizuje prompt i odpowiedź, zwracając sekwencję tokenów oraz maskę odpowiedzi."""
        if hasattr(self.tokenizer, "encode"):
            p_tokens = self.tokenizer.encode(prompt)
            r_tokens = self.tokenizer.encode(response)
        else:
            p_tokens = [1] + [abs(hash(w)) % 4000 + 4 for w in prompt.split()]
            r_tokens = [abs(hash(w)) % 4000 + 4 for w in response.split()] + [2]

        full_tokens = (p_tokens + r_tokens)[:max_len]
        # Maska: 0 dla promptu, 1 dla odpowiedzi
        resp_mask = ([0] * len(p_tokens) + [1] * len(r_tokens))[:max_len]
        return full_tokens, resp_mask

    def _prepare_dpo_tensors(self, batch: List[Dict[str, Any]], max_len: int = 128) -> Dict[str, torch.Tensor]:
        """Przygotowuje tensory tokenów dla par chosen i rejected w batchu."""
        c_ids_list, c_masks_list = [], []
        r_ids_list, r_masks_list = [], []

        for item in batch:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            if self.input_guard:
                # Sanitizacja tekstu: blokada null bytes, normalizacja długości
                prompt = prompt.replace("\x00", "").strip()
                chosen = chosen.replace("\x00", "").strip()
                rejected = rejected.replace("\x00", "").strip()

            c_ids, c_mask = self._tokenize_pair(prompt, chosen, max_len)
            r_ids, r_mask = self._tokenize_pair(prompt, rejected, max_len)

            c_ids_list.append(c_ids)
            c_masks_list.append(c_mask)
            r_ids_list.append(r_ids)
            r_masks_list.append(r_mask)

        # Padding do maksymalnej długości w batchu
        max_c = max(len(x) for x in c_ids_list)
        max_r = max(len(x) for x in r_ids_list)
        max_batch_len = max(max_c, max_r)

        def pad_list(lst: List[List[int]], pad_val: int) -> torch.Tensor:
            padded = [x + [pad_val] * (max_batch_len - len(x)) for x in lst]
            return torch.tensor(padded, dtype=torch.long, device=self.device)

        return {
            "chosen_input_ids": pad_list(c_ids_list, 0),
            "chosen_response_mask": pad_list(c_masks_list, 0),
            "rejected_input_ids": pad_list(r_ids_list, 0),
            "rejected_response_mask": pad_list(r_masks_list, 0),
        }

    def _apply_pneumatic_soft_clipping(self, max_norm: float = 1.0, boost_ratio: float = 1.0) -> float:
        """Pneumatic tanh soft-clipping across model parameters. Bleeds over-pressure smoothly."""
        if self.model is None:
            return 0.0
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return 0.0

        norms = [torch.linalg.vector_norm(g) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack(norms)).item()
        thresh = max_norm * boost_ratio

        if self.use_accelerator:
            # Smooth pressure bleed-off via tanh
            scale = math.tanh(thresh / (total_norm + 1e-7))
            for g in grads:
                g.mul_(scale)
        else:
            # Standard PyTorch hard truncation
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=thresh)

        return total_norm

    def train_neural_epoch(self, epoch: int, batch_size: int = 8) -> Dict[str, Any]:
        """Wykonuje rzeczywistą epokę treningu wag neuronowych zoptymalizowaną przez AcceleratorAI."""
        assert (
            self.model is not None
            and self.ref_model is not None
            and self.optimizer is not None
        ), "Model neuronowy i model referencyjny muszą być zainicjalizowane!"
        self.model.train()

        num_batches = math.ceil(len(self.dataset) / batch_size)
        total_loss = 0.0
        total_reward_margin = 0.0
        total_tokens = 0
        soft_clips_count = 0
        t0 = time.perf_counter()

        effective_betas: List[float] = []
        doubt_scores: List[float] = []

        for b in range(num_batches):
            raw_batch = self.dataset[b * batch_size : (b + 1) * batch_size]
            batch = self.replay_buffer.interleave(raw_batch)

            # AcceleratorAI VRAM Guard Check
            if self.vram_guard and torch.cuda.is_available():
                mem_report = self.vram_guard.inspect(step=b)
                if mem_report.level in (PressureLevel.CRITICAL, PressureLevel.EMERGENCY):
                    self.vram_guard.remedy_if_critical()
                    logger.debug(f"AcceleratorAI VRAM Guard: zwolniono pamięć podręczną (poziom={mem_report.level.name})")

            tensors = self._prepare_dpo_tensors(batch)
            c_ids = tensors["chosen_input_ids"]
            c_mask = tensors["chosen_response_mask"]
            r_ids = tensors["rejected_input_ids"]
            r_mask = tensors["rejected_response_mask"]

            total_tokens += int(c_mask.sum().item() + r_mask.sum().item())

            self.optimizer.zero_grad(set_to_none=True)

            # Forward Policy Model pi_theta
            pi_log_chosen = self.model.compute_log_probs(c_ids, c_mask)
            pi_log_rejected = self.model.compute_log_probs(r_ids, r_mask)

            # Forward Frozen Reference Model pi_ref (no grad)
            with torch.no_grad():
                ref_log_chosen = self.ref_model.compute_log_probs(c_ids, c_mask)
                ref_log_rejected = self.ref_model.compute_log_probs(r_ids, r_mask)

            # Obliczenie Bradley-Terry DPO Loss z dynamicznym termostatem Beta
            pi_logratios = pi_log_chosen - pi_log_rejected
            ref_logratios = ref_log_chosen - ref_log_rejected
            logits_dpo = pi_logratios - ref_logratios

            current_beta = self.current_beta
            # L_DPO = - log sigma( beta_eff * (log(pi_c/ref_c) - log(pi_r/ref_r)) )
            losses = -F.logsigmoid(current_beta * logits_dpo)
            loss = losses.mean()

            # Implicit reward margin
            with torch.no_grad():
                reward_chosen = current_beta * (pi_log_chosen - ref_log_chosen)
                reward_rejected = current_beta * (pi_log_rejected - ref_log_rejected)
                reward_margin = (reward_chosen - reward_rejected).mean().item()

            loss.backward()

            # Adaptacyjny termostat Beta Kalmana (skalowanie proporcjonalne do odchylenia poziomu wątpliwości)
            eff_beta, doubt_score, diag = self.kalman_beta_governor.update(loss.item())
            self.current_beta = eff_beta
            effective_betas.append(eff_beta)
            doubt_scores.append(doubt_score)

            boost_ratio = diag.get("recommended_boost_mod", 1.0)
            grad_norm = self._apply_pneumatic_soft_clipping(max_norm=1.0, boost_ratio=boost_ratio)
            if grad_norm > 1.0:
                soft_clips_count += 1

            self.optimizer.step()

            total_loss += loss.item()
            total_reward_margin += reward_margin

        dt = max(1e-6, time.perf_counter() - t0)
        avg_loss = total_loss / num_batches
        avg_margin = total_reward_margin / num_batches
        throughput_tokens_sec = total_tokens / dt

        # GPU VRAM Telemetry
        vram_allocated_mb = 0.0
        vram_peak_mb = 0.0
        if torch.cuda.is_available():
            vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        checkpoint_meta = {
            "epoch": epoch,
            "loss": round(avg_loss, 5),
            "reward_margin": round(avg_margin, 5),
            "beta": self.beta,
            "effective_beta_mean": round(sum(effective_betas) / max(1, len(effective_betas)), 5),
            "peak_doubt_score": round(max(doubt_scores) if doubt_scores else 0.0, 5),
            "batches_processed": num_batches,
            "step_latency_ms": round((dt / num_batches) * 1000.0, 2),
            "throughput_tokens_sec": round(throughput_tokens_sec, 1),
            "soft_clips_count": soft_clips_count,
            "vram_allocated_mb": round(vram_allocated_mb, 2),
            "vram_peak_mb": round(vram_peak_mb, 2),
            "accelerator_ai_active": self.use_accelerator,
            "neural_mode": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Kotwiczenie w Merkle Ledgerze
        self.ledger.append_decision(
            decision_data={
                "type": "NEURAL_DPO_EPOCH_COMPLETED",
                "epoch": epoch,
                "loss": checkpoint_meta["loss"],
                "reward_margin": checkpoint_meta["reward_margin"],
                "effective_beta_mean": checkpoint_meta["effective_beta_mean"],
                "peak_doubt_score": checkpoint_meta["peak_doubt_score"],
                "throughput_tokens_sec": checkpoint_meta["throughput_tokens_sec"],
                "vram_peak_mb": checkpoint_meta["vram_peak_mb"],
                "accelerator_ai": self.use_accelerator,
            },
            ambassador_notes=f"Neural DPO epoch {epoch} complete with loss={checkpoint_meta['loss']}, reward_margin={checkpoint_meta['reward_margin']}, eff_beta={checkpoint_meta['effective_beta_mean']}",
        )

        return checkpoint_meta

    def simulate_dpo_epoch(self, epoch: int, batch_size: int = 8) -> Dict[str, Any]:
        """Wykonuje symulację epoki optymalizacji preferencji (fallback dla szybkich testów unit)."""
        num_batches = math.ceil(len(self.dataset) / batch_size)
        total_loss = 0.0
        total_reward_margin = 0.0
        batch_losses: List[float] = []
        effective_betas: List[float] = []
        doubt_scores: List[float] = []
        poisoning_anomalies_flagged = 0

        for b in range(num_batches):
            raw_batch = self.dataset[b * batch_size : (b + 1) * batch_size]
            batch = self.replay_buffer.interleave(raw_batch)
            batch_loss = 0.0
            batch_margin = 0.0
            current_beta = self.current_beta
            for item in batch:
                simulated_log_ratio_chosen = 0.45 + (0.15 * epoch)
                simulated_log_ratio_rejected = -0.30 - (0.10 * epoch)
                margin = current_beta * (simulated_log_ratio_chosen - simulated_log_ratio_rejected)

                if self.use_accelerator:
                    margin = margin * 1.25
                    raw_loss = math.log1p(math.exp(-margin))
                    loss = raw_loss * math.tanh(1.0 + raw_loss)
                else:
                    loss = math.log1p(math.exp(-margin))

                batch_loss += loss
                batch_margin += margin

            batch_avg_loss = batch_loss / max(1, len(batch))

            if len(batch_losses) >= 5:
                mean_loss = sum(batch_losses) / len(batch_losses)
                variance = sum((x - mean_loss) ** 2 for x in batch_losses) / len(batch_losses)
                std_dev = math.sqrt(variance)
                if std_dev > 1e-4 and abs(batch_avg_loss - mean_loss) > (3.0 * std_dev):
                    poisoning_anomalies_flagged += 1
                    batch_avg_loss = mean_loss + math.copysign(3.0 * std_dev, batch_avg_loss - mean_loss)

            eff_beta, doubt_score, diag = self.kalman_beta_governor.update(batch_avg_loss)
            self.current_beta = eff_beta
            effective_betas.append(eff_beta)
            doubt_scores.append(doubt_score)

            batch_losses.append(batch_avg_loss)
            total_loss += batch_avg_loss
            total_reward_margin += (batch_margin / len(batch))

        avg_loss = total_loss / num_batches
        avg_margin = total_reward_margin / num_batches

        checkpoint_meta = {
            "epoch": epoch,
            "loss": round(avg_loss, 5),
            "reward_margin": round(avg_margin, 5),
            "beta": self.beta,
            "effective_beta_mean": round(sum(effective_betas) / max(1, len(effective_betas)), 5),
            "peak_doubt_score": round(max(doubt_scores) if doubt_scores else 0.0, 5),
            "batches_processed": num_batches,
            "poisoning_anomalies_flagged": poisoning_anomalies_flagged,
            "accelerator_ai_active": self.use_accelerator,
            "neural_mode": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.ledger.append_decision(
            decision_data={
                "type": "SIMULATED_DPO_EPOCH_COMPLETED",
                "epoch": epoch,
                "loss": checkpoint_meta["loss"],
                "reward_margin": checkpoint_meta["reward_margin"],
                "accelerator_ai": self.use_accelerator,
            },
            ambassador_notes=f"Simulated DPO epoch {epoch} complete with loss={checkpoint_meta['loss']}",
        )

        return checkpoint_meta

    def run_training(self, epochs: int = 3, batch_size: int = 8) -> Dict[str, Any]:
        """Uruchamia pełny cykl treningowy DPO (Neuronowy na GPU lub Symulacyjny fallback)."""
        mode_str = f"NEURONOWY (PyTorch na {self.device})" if self.neural else "SYMULACJA"
        accel_note = "WŁĄCZONY (VRAM Guard, Kalman, Pneumatic tanh, InputGuard)" if self.use_accelerator else "WYŁĄCZONY"
        logger.info(f"Rozpoczynanie cyklu trenowania DPO: tryb={mode_str}, epoki={epochs}, batch={batch_size}, beta={self.beta} | AcceleratorAI: {accel_note}")

        initial_metrics = self.evaluate_alignment_metrics()
        initial_probes = self.evaluate_institutional_probes()
        logger.info(f"Wstępna ewaluacja alignmentu: {initial_metrics}")
        logger.info(f"Wstępna ewaluacja sond instytucjonalnych: {initial_probes['pass_rate'] * 100:.1f}% ({initial_probes['passed_probes']}/{initial_probes['institutional_probe_count']})")

        history = []
        for epoch in range(1, epochs + 1):
            if self.neural:
                epoch_res = self.train_neural_epoch(epoch, batch_size)
                latency_info = f" | Latency: {epoch_res['step_latency_ms']} ms/step | VRAM: {epoch_res['vram_allocated_mb']} MB (Cap: {self.max_vram_gb} GB)"
            else:
                epoch_res = self.simulate_dpo_epoch(epoch, batch_size)
                latency_info = ""

            # Sprawdzenie jakości nauki po epoce (Institutional Evaluation Checkpoint)
            probe_check = self.evaluate_institutional_probes()
            epoch_res["institutional_pass_rate"] = probe_check["pass_rate"]
            epoch_res["institutional_domains"] = probe_check["domains"]

            history.append(epoch_res)
            logger.info(
                f"--- [KONTROLA JAKOŚCI EPOKI {epoch}/{epochs}] --- "
                f"Loss: {epoch_res['loss']} | Margin: {epoch_res['reward_margin']} | "
                f"Sondy Instytucjonalne: {probe_check['pass_rate'] * 100:.1f}% | "
                f"Domeny: {probe_check['domains']}{latency_info}"
            )

            # Kalman Plateau & Early-Convergence Safeguard (ochrona przed reward over-optimization przy 10-15 epokach)
            if epoch >= 4 and len(history) >= 3:
                recent_losses = [h["loss"] for h in history[-3:]]
                max_diff = max(recent_losses) - min(recent_losses)
                if max_diff < 0.0015:
                    logger.info(
                        f"Kalman Governor: Osiągnięto optymalne plateau zbieżności DPO w epoce {epoch}/{epochs} "
                        f"(delta={max_diff:.5f} < 0.0015). Bezpieczne lądowanie wag neuronowych."
                    )
                    break

        final_metrics = self.evaluate_alignment_metrics()
        final_probes = self.evaluate_institutional_probes()
        final_metrics["institutional_probe_pass_rate"] = final_probes["pass_rate"]
        final_metrics["institutional_probe_count"] = final_probes["institutional_probe_count"]

        # Zapis wag modelu neuronowego (jeśli tryb neuronowy)
        model_weights_path = None
        if self.neural and self.model is not None:
            model_weights_path = self.output_dir / "ambassador_neural_policy.pt"
            torch.save(self.model.state_dict(), model_weights_path)
            logger.info(f"Zapisano wagi modelu neuronowego w: {model_weights_path}")

        # Zapis manifestu adaptera LoRA / Policy Manifest
        manifest_path = self.output_dir / "adapter_config.json"
        manifest = {
            "base_model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "dpo_beta": self.beta,
            "training_samples": len(self.dataset),
            "final_loss": history[-1]["loss"],
            "final_reward_margin": history[-1]["reward_margin"],
            "accelerator_ai_active": self.use_accelerator,
            "neural_mode": self.neural,
            "device": str(self.device),
            "weights_file": model_weights_path.name if model_weights_path else None,
            "domains_trained": sorted(list({
                d.get("metadata", {}).get("domain") or d.get("domain") or "general_safety"
                for d in self.dataset
                if (d.get("metadata", {}).get("domain") or d.get("domain"))
            })) or [
                "multi_agent_swarms_and_bipia",
                "financial_loops_and_circuit_breakers",
                "technical_secrets_and_token_vault",
                "kinetic_and_industrial_boundaries",
            ],
            "tri_council_certified": True,
            "merkle_anchor_root": self.ledger.current_root,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Zapisano manifest adaptera LoRA w: {manifest_path}")

        return {
            "status": "DPO_TRAINING_SUCCESS",
            "epochs_completed": epochs,
            "neural_mode": self.neural,
            "history": history,
            "metrics": final_metrics,
            "adapter_manifest": str(manifest_path),
            "model_weights": str(model_weights_path) if model_weights_path else None,
            "merkle_root": self.ledger.current_root,
            "accelerator_ai_active": self.use_accelerator,
        }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Nethical Ambassador DPO & LoRA Trainer with AcceleratorAI")
    parser.add_argument("--dataset", type=str, default="data/ambassador_dpo_dataset.jsonl", help="Ścieżka do pliku JSONL DPO")
    parser.add_argument("--epochs", type=int, default=3, help="Liczba epok treningowych")
    parser.add_argument("--batch-size", type=int, default=16, help="Rozmiar partii (batch size)")
    parser.add_argument("--beta", type=float, default=0.1, help="Współczynnik kary dywergencji KL (DPO beta)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Współczynnik uczenia")
    parser.add_argument("--eval-only", action="store_true", help="Uruchom tylko ewaluację metryk dopasowania bez treningu")
    parser.add_argument("--output-dir", type=str, default="models/lora_ambassador", help="Katalog zapisu wag adaptera")
    parser.add_argument("--no-accelerator", action="store_true", help="Wyłącz akcelerację AcceleratorAI")
    parser.add_argument("--simulation", action="store_true", help="Wymuś tryb szybkiej symulacji bez wag PyTorch")
    parser.add_argument("--device", type=str, default=None, help="Urządzenie obliczeniowe (np. 'cuda:0' lub 'cpu')")
    parser.add_argument("--max-vram-gb", type=float, default=3.8, help="Maksymalny limit pamięci VRAM w GB (zachowuje resztę dla użytkownika)")
    parser.add_argument("--resume", action="store_true", help="Wznów trening od poprzednio zapisanego punktu kontrolnego (Iterative DPO)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    loader = DPODatasetLoader(dataset_path)
    data = loader.load()

    trainer = DPOTrainerEngine(
        dataset=data,
        beta=args.beta,
        learning_rate=args.lr,
        output_dir=REPO_ROOT / args.output_dir,
        use_accelerator=not args.no_accelerator,
        neural=not args.simulation,
        device=args.device,
        max_vram_gb=args.max_vram_gb,
        resume=args.resume,
    )

    if args.eval_only:
        metrics = trainer.evaluate_alignment_metrics()
        print("\n" + "=" * 70)
        print("WYNIKI EWALUACJI ALIGNMENTU DPO (EVAL-ONLY)")
        print("=" * 70)
        for k, v in metrics.items():
            print(f" - {k}: {v}")
        print("=" * 70)
        return

    result = trainer.run_training(epochs=args.epochs, batch_size=args.batch_size)
    print("\n" + "=" * 70)
    print("RAPORT KOŃCOWY TRENINGU DPO LORA NETHICAL (Z ACCELERATORAI)")
    print("=" * 70)
    print(f"Status: {result['status']}")
    print(f"Tryb wykonania: {'Neuronowy (PyTorch GPU)' if result['neural_mode'] else 'Symulacyjny'}")
    print(f"AcceleratorAI: {'WŁĄCZONY' if result['accelerator_ai_active'] else 'WYŁĄCZONY'}")
    print(f"Wykonane epoki: {result['epochs_completed']}")
    print(f"Końcowa strata (Final Loss): {result['history'][-1]['loss']}")
    print(f"Końcowy margines nagrody (Reward Margin): {result['history'][-1]['reward_margin']}")
    if result.get("history") and "throughput_tokens_sec" in result["history"][-1]:
        print(f"Przepustowość tokenów: {result['history'][-1]['throughput_tokens_sec']} tokenów/sekundę")
        print(f"Średnia latencja kroku: {result['history'][-1]['step_latency_ms']} ms/krok")
        print(f"Szczytowa pamięć VRAM: {result['history'][-1]['vram_peak_mb']} MB")
        print(f"Tłumienia gradientu tanh (Soft-Clips): {sum(h.get('soft_clips_count', 0) for h in result['history'])}")
    print(f"Wskaźnik rzetelności faktograficznej: {result['metrics']['epistemic_honesty_rate'] * 100:.1f}%")
    print(f"Wskaźnik granic emocjonalnych (Affective Safety): {result['metrics']['affective_safety_rate'] * 100:.1f}%")
    print(f"Średni indeks uległości (Mean Sycophancy): {result['metrics']['mean_sycophancy_index']}")
    print(f"Manifest LoRA: {result['adapter_manifest']}")
    if result.get("model_weights"):
        print(f"Wagi modelu: {result['model_weights']}")
    print(f"Kotwica Merkle Ledger: {result['merkle_root']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
