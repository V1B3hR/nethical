"""Monitor for detecting deviations between stated intent and actual actions.

Advanced implementation that aggregates multiple complementary signals:

Lexical signals (no extra dependencies):
- Unigram and bigram Jaccard similarities
- TF cosine similarity
- Character-level ratio (difflib)
- Directional coverage: how much of action terms are covered by intent terms (and vice versa)
- High-risk term detection (curated lexicon), policy-bypass cues, negations

Optional ML signals (loaded lazily, degrade gracefully if unavailable):
- Sentence embedding cosine similarity via sentence-transformers
- Natural Language Inference (NLI) for entailment/contradiction using transformers pipeline

Additional features:
- Normalization, stopword filtering, synonym canonicalization, min token length
- Configurable metric weights; dynamic thresholding with risk-aware boost
- Evidence payload with per-metric scores, overlaps, risk terms, and model availability
- Lightweight embedding/NLI caching to reduce repeated compute
- Severity boosted by contradiction signal and high-risk cues

Dependencies:
- Base features require only the Python standard library.
- Optional embeddings: pip install sentence-transformers
- Optional NLI: pip install transformers torch (or equivalent backend)

Notes:
- Heavy model loads are lazy; first use may be slow.
- This async method runs CPU-bound work synchronously; for high-throughput systems,
  consider offloading to a thread pool or an async-enabled inference service.
"""

import math
import re
import string
import uuid
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import tee
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .base_monitor import BaseMonitor
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel


# -------------------------
# Optional ML backends
# -------------------------

class _OptionalSentenceTransformerEmbedder:
    """Lazy loader for sentence-transformers embedding model."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._available_error: Optional[str] = None

    def available(self) -> bool:
        if self._model is None and self._available_error is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:  # noqa: BLE001
                self._available_error = f"{type(e).__name__}: {e}"
        return self._model is not None

    @lru_cache(maxsize=256)
    def encode(self, text: str) -> Optional[List[float]]:
        if not self.available():
            return None
        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def similarity(self, a: str, b: str) -> Optional[float]:
        va, vb = self.encode(a), self.encode(b)
        if va is None or vb is None or not va or not vb:
            return None
        # Cosine similarity with normalized embeddings (should be dot product)
        dot = sum(x * y for x, y in zip(va, vb))
        return max(0.0, min(1.0, float(dot)))


class _OptionalNLI:
    """Lazy loader for NLI pipeline, e.g., 'facebook/bart-large-mnli'."""
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self._pipe = None
        self._available_error: Optional[str] = None

    def available(self) -> bool:
        if self._pipe is None and self._available_error is None:
            try:
                from transformers import pipeline  # type: ignore
                self._pipe = pipeline("text-classification", model=self.model_name)
            except Exception as e:  # noqa: BLE001
                self._available_error = f"{type(e).__name__}: {e}"
        return self._pipe is not None

    @lru_cache(maxsize=256)
    def score(self, premise: str, hypothesis: str) -> Optional[Dict[str, float]]:
        """
        Returns probabilities for labels: entailment, contradiction, neutral.
        Normalized to sum ~1.0. Returns None if unavailable.
        """
        if not self.available():
            return None
        pipe = self._pipe
        try:
            # Many NLI pipelines require concatenation or provide multi-class labels
            # The default HF NLI models with text-classification return scores per label.
            outputs = pipe(f"{premise} </s></s> {hypothesis}", return_all_scores=True)  # type: ignore[operator]
            if not outputs or not isinstance(outputs, list) or not outputs[0]:
                return None
            label_map = {}
            for item in outputs[0]:
                label = item.get("label", "").lower()
                score = float(item.get("score", 0.0))
                label_map[label] = score
            # Normalize common label names
            entail = label_map.get("entailment", 0.0)
            contra = label_map.get("contradiction", 0.0)
            neutral = label_map.get("neutral", 0.0)
            total = entail + contra + neutral
            if total > 0:
                entail, contra, neutral = entail / total, contra / total, neutral / total
            return {"entailment": entail, "contradiction": contra, "neutral": neutral}
        except Exception:
            return None


class IntentDeviationMonitor(BaseMonitor):
    """Monitors deviations between an agent's stated intent and actual actions."""

    # Compact stopword list to reduce noise without extra deps.
    _DEFAULT_STOPWORDS: Set[str] = {
        "the", "a", "an", "and", "or", "but",
        "to", "of", "in", "on", "for", "with",
        "at", "by", "from", "as", "is", "are",
        "was", "were", "be", "been", "being",
        "that", "this", "these", "those",
        "it", "its", "into", "about", "over", "under",
        "if", "then", "so", "than", "too", "very",
        "can", "could", "should", "would",
        "will", "may", "might", "must", "do", "does", "did",
        "not", "no", "yes",
    }

    # High-risk lexicon and bypass cues (extend as needed)
    _HIGH_RISK_TOKENS: Set[str] = {
        "delete", "drop", "truncate", "wipe", "erase", "format",
        "shutdown", "kill", "terminate", "halt",
        "leak", "exfiltrate", "steal", "exfil", "dump",
        "chmod", "chown", "sudo", "root", "rm", "rmdir", "del",
        "privilege", "escalate", "exploit", "backdoor",
        "disable", "bypass", "ignore", "override",
        "production", "prod", "database", "db",
        "secrets", "token", "apikey", "password", "passwd", "key",
    }
    _BYPASS_PHRASES: Set[str] = {"ignore safety", "bypass policy", "disable guard", "disable safeguards"}
    _NEGATIONS: Set[str] = {"not", "no", "never", "without", "avoid"}

    def __init__(
        self,
        deviation_threshold: float = 0.7,
        *,
        min_token_len: int = 2,
        stopwords: Optional[Sequence[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        synonyms: Optional[Dict[str, str]] = None,
        # Optional ML
        enable_embeddings: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_nli: bool = False,
        nli_model: str = "facebook/bart-large-mnli",
        # Risk handling
        risk_threshold_boost: float = 0.1,  # increases strictness when risk cues found
        contradiction_severity_boost: float = 0.1,  # boosts severity if contradiction is high
    ):
        """
        Args:
            deviation_threshold: Base deviation threshold in [0,1]. Lower is stricter.
            min_token_len: Ignore tokens shorter than this length after normalization.
            stopwords: Optional custom stopword list. Defaults to a small built-in set.
            weights: Optional dict of metric weights summing to 1.0.
                Keys supported (all optional): "unigram_jaccard", "bigram_jaccard", "cosine",
                "char_ratio", "coverage", "embed_cosine", "nli_entailment"
            synonyms: Optional mapping to canonicalize tokens, e.g., {"del": "delete", "rm": "remove"}.
            enable_embeddings: If True, try to load sentence-transformers model lazily.
            embedding_model: Model name for sentence-transformers.
            enable_nli: If True, try to load NLI model pipline lazily.
            nli_model: Model name for NLI pipeline.
            risk_threshold_boost: Boost applied to lower the threshold when high-risk cues appear.
            contradiction_severity_boost: Additional severity boost when contradiction is high.
        """
        super().__init__("Intent Deviation Monitor")

        self.deviation_threshold = self._clamp(deviation_threshold, 0.0, 1.0)
        self.min_token_len = max(0, int(min_token_len))
        self.stopwords: Set[str] = set(stopwords) if stopwords is not None else set(self._DEFAULT_STOPWORDS)
        self.synonyms = dict(synonyms) if synonyms else {}

        default_weights = {
            "unigram_jaccard": 0.20,
            "bigram_jaccard": 0.10,
            "cosine": 0.25,
            "char_ratio": 0.05,
            "coverage": 0.20,         # directional coverage average
            "embed_cosine": 0.15,     # optional
            "nli_entailment": 0.05,   # optional
        }
        self.weights = self._normalize_weights(weights or default_weights)

        # Pre-compile regex for performance
        self._punct_regex = re.compile(f"[{re.escape(string.punctuation)}]")

        # Optional ML backends
        self._embedder = _OptionalSentenceTransformerEmbedder(embedding_model) if enable_embeddings else None
        self._nli = _OptionalNLI(nli_model) if enable_nli else None

        # Risk handling
        self.risk_threshold_boost = float(risk_threshold_boost)
        self.contradiction_severity_boost = float(contradiction_severity_boost)

    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        """Analyze an action for intent deviation."""
        if not self.enabled:
            return []

        violations: List[SafetyViolation] = []

        deviation_score, evidence, risk_info = self._calculate_deviation(action.stated_intent, action.actual_action)

        # Dynamic thresholding based on risk cues
        threshold = self.deviation_threshold
        if risk_info.get("risk_cues_detected", False):
            threshold = max(0.0, threshold - self.risk_threshold_boost)

        if deviation_score > threshold:
            # Severity determined by deviation, risk cues, and contradiction signal
            contradiction = evidence.get("nli", {}).get("intent->action", {}).get("contradiction", 0.0) if evidence.get("nli") else 0.0
            severity = self._determine_severity(
                deviation_score,
                risk_boost=(0.15 if risk_info.get("risk_cues_detected") else 0.0),
                contradiction=contradiction,
            )

            violation = SafetyViolation(
                id=str(uuid.uuid4()),
                action_id=getattr(action, "id", None),
                violation_type=ViolationType.INTENT_DEVIATION,
                severity=severity,
                description=(
                    f"Deviation between stated intent and actual action "
                    f"(score: {deviation_score:.2f}, threshold: {threshold:.2f})"
                ),
                evidence={
                    "stated_intent": getattr(action, "stated_intent", None),
                    "actual_action": getattr(action, "actual_action", None),
                    "deviation_score": deviation_score,
                    "base_threshold": self.deviation_threshold,
                    "effective_threshold": threshold,
                    "weights": self.weights,
                    **evidence,
                },
            )
            violations.append(violation)

        return violations

    # -------------------------
    # Core deviation computation
    # -------------------------

    def _calculate_deviation(self, intent: Optional[str], action: Optional[str]) -> Tuple[float, Dict, Dict]:
        """
        Calculate deviation score between intent and action.

        Returns:
            deviation_score in [0, 1], evidence dict, risk_info dict
            where 0 = identical/entailed, 1 = completely different/contradictory.
        """
        # Handle nulls or empties: empty vs non-empty => max deviation; both empty => no deviation
        if not intent and not action:
            return 0.0, self._empty_evidence(), {"risk_cues_detected": False}
        if not intent or not action:
            return 1.0, self._empty_evidence(flag="one_side_empty"), {"risk_cues_detected": False}

        # Normalize and tokenize
        norm_intent = self._normalize(intent)
        norm_action = self._normalize(action)

        tokens_intent = self._tokenize(norm_intent)
        tokens_action = self._tokenize(norm_action)

        # Bigrams
        bigrams_intent = self._bigrams(tokens_intent)
        bigrams_action = self._bigrams(tokens_action)

        # Lexical metrics
        uni_jaccard = self._jaccard(tokens_intent, tokens_action)
        bi_jaccard = self._jaccard(bigrams_intent, bigrams_action)
        cosine_sim = self._cosine_similarity(tokens_intent, tokens_action)
        char_ratio = self._char_similarity(norm_intent, norm_action)

        # Directional coverage
        coverage_action_by_intent = self._coverage(tokens_action, tokens_intent)
        coverage_intent_by_action = self._coverage(tokens_intent, tokens_action)
        coverage_avg = 0.5 * (coverage_action_by_intent + coverage_intent_by_action)

        # Optional embedding similarity
        embed_sim = None
        embed_available = False
        if self._embedder is not None:
            embed_sim = self._embedder.similarity(norm_intent, norm_action)
            embed_available = self._embedder.available()

        # Optional NLI (directional)
        nli_scores = None
        nli_available = False
        if self._nli is not None:
            s1 = self._nli.score(norm_intent, norm_action)  # intent -> action
            s2 = self._nli.score(norm_action, norm_intent)  # action -> intent
            if s1 or s2:
                nli_scores = {"intent->action": s1 or {}, "action->intent": s2 or {}}
            nli_available = self._nli.available()

        # Weighted similarity aggregation
        sim = 0.0
        sim += self.weights.get("unigram_jaccard", 0.0) * uni_jaccard
        sim += self.weights.get("bigram_jaccard", 0.0) * bi_jaccard
        sim += self.weights.get("cosine", 0.0) * cosine_sim
        sim += self.weights.get("char_ratio", 0.0) * char_ratio
        sim += self.weights.get("coverage", 0.0) * coverage_avg
        if embed_sim is not None:
            sim += self.weights.get("embed_cosine", 0.0) * embed_sim
        if nli_scores and "intent->action" in nli_scores:
            entail = float(nli_scores["intent->action"].get("entailment", 0.0))
            sim += self.weights.get("nli_entailment", 0.0) * entail

        sim = self._clamp(sim, 0.0, 1.0)
        deviation = 1.0 - sim

        # Risk signals
        risky_terms_intent, risky_terms_action = self._find_risky_terms(tokens_intent), self._find_risky_terms(tokens_action)
        bypass_cues = self._find_bypass_cues(norm_intent) or self._find_bypass_cues(norm_action)
        negation_intent = any(t in self._NEGATIONS for t in tokens_intent)
        negation_action = any(t in self._NEGATIONS for t in tokens_action)
        risk_cues_detected = bool(risky_terms_action - risky_terms_intent) or bool(bypass_cues)

        # Evidence
        only_in_intent = sorted(list(set(tokens_intent) - set(tokens_action)))[:50]
        only_in_action = sorted(list(set(tokens_action) - set(tokens_intent)))[:50]
        only_in_intent_bi = sorted(list(set(bigrams_intent) - set(bigrams_action)))[:50]
        only_in_action_bi = sorted(list(set(bigrams_action) - set(bigrams_intent)))[:50]

        evidence: Dict = {
            "metrics": {
                "unigram_jaccard": round(uni_jaccard, 4),
                "bigram_jaccard": round(bi_jaccard, 4),
                "tf_cosine": round(cosine_sim, 4),
                "char_ratio": round(char_ratio, 4),
                "coverage_action_by_intent": round(coverage_action_by_intent, 4),
                "coverage_intent_by_action": round(coverage_intent_by_action, 4),
                "coverage_avg": round(coverage_avg, 4),
                "embed_cosine": (round(embed_sim, 4) if embed_sim is not None else None),
                "weighted_similarity": round(sim, 4),
                "deviation": round(deviation, 4),
            },
            "model_availability": {
                "embeddings": embed_available,
                "nli": nli_available,
            },
            "nli": nli_scores,
            "unigram_overlap": {
                "only_in_intent": only_in_intent,
                "only_in_action": only_in_action,
            },
            "bigram_overlap": {
                "only_in_intent": only_in_intent_bi,
                "only_in_action": only_in_action_bi,
            },
            "normalized": {
                "intent": norm_intent,
                "action": norm_action,
            },
            "token_counts": {
                "intent": len(tokens_intent),
                "action": len(tokens_action),
            },
            "risk": {
                "risky_terms_intent": sorted(risky_terms_intent),
                "risky_terms_action": sorted(risky_terms_action),
                "bypass_cues_detected": bool(bypass_cues),
                "negation_intent": negation_intent,
                "negation_action": negation_action,
            },
        }

        risk_info = {"risk_cues_detected": risk_cues_detected}
        return deviation, evidence, risk_info

    # -------------------------
    # Similarity helper methods
    # -------------------------

    def _normalize(self, text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = self._punct_regex.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _apply_synonyms(self, token: str) -> str:
        """Map a token to its canonical form via synonyms, if provided."""
        return self.synonyms.get(token, token)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize by whitespace, remove stopwords, apply synonyms, filter by min length."""
        tokens = []
        for t in text.split():
            if t in self.stopwords:
                continue
            t = self._apply_synonyms(t)
            if len(t) >= self.min_token_len:
                tokens.append(t)
        return tokens

    def _bigrams(self, tokens: Sequence[str]) -> List[Tuple[str, str]]:
        """Create adjacent token bigrams."""
        if len(tokens) < 2:
            return []
        a, b = tee(tokens)
        next(b, None)
        return list(zip(a, b))

    def _jaccard(self, a: Iterable, b: Iterable) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)

    def _cosine_similarity(self, a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
        if not a_tokens and not b_tokens:
            return 1.0
        va, vb = Counter(a_tokens), Counter(b_tokens)
        dot = sum(va[t] * vb.get(t, 0) for t in va)
        na = math.sqrt(sum(c * c for c in va.values()))
        nb = math.sqrt(sum(c * c for c in vb.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _char_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(a=a, b=b).ratio()

    def _coverage(self, subset_tokens: Sequence[str], superset_tokens: Sequence[str]) -> float:
        """How much of subset_tokens are covered by superset_tokens (set-based)."""
        ssub, ssuper = set(subset_tokens), set(superset_tokens)
        if not ssub:
            return 1.0
        return len(ssub & ssuper) / len(ssub)

    # -------------------------
    # Risk helpers
    # -------------------------

    def _find_risky_terms(self, tokens: Sequence[str]) -> Set[str]:
        toks = set(tokens)
        risky = set()
        for t in toks:
            if t in self._HIGH_RISK_TOKENS:
                risky.add(t)
            # Light substring checks for e.g. 'rm -rf', 'drop-table'
            if "rm" == t or "rmrf" in t or "rm-rf" in t or "rm" in t and "-rf" in t:
                risky.add("rm -rf")
            if "drop" in t and ("table" in t or "db" in t or "database" in t):
                risky.add("drop")
        return risky

    def _find_bypass_cues(self, text: str) -> bool:
        for phrase in self._BYPASS_PHRASES:
            if phrase in text:
                return True
        return False

    # -------------------------
    # Severity and utilities
    # -------------------------

    def _determine_severity(self, deviation_score: float, *, risk_boost: float = 0.0, contradiction: float = 0.0) -> SeverityLevel:
        """Determine severity based on deviation score, risk cues, and contradiction signal."""
        # Base mapping
        score = deviation_score
        # Boost for contradiction (up to +0.1 default), non-linear emphasis
        score = min(1.0, score + self.contradiction_severity_boost * (contradiction ** 0.5))
        # Boost for risk cues
        score = min(1.0, score + risk_boost)

        if score >= 0.95:
            return SeverityLevel.CRITICAL
        elif score >= 0.85:
            return SeverityLevel.HIGH
        elif score >= 0.70:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        supported = {
            "unigram_jaccard",
            "bigram_jaccard",
            "cosine",
            "char_ratio",
            "coverage",
            "embed_cosine",
            "nli_entailment",
        }
        filtered = {k: max(0.0, float(v)) for k, v in weights.items() if k in supported}
        if not filtered:
            # Fallback to a safe default if invalid
            filtered = {
                "unigram_jaccard": 0.20,
                "bigram_jaccard": 0.10,
                "cosine": 0.25,
                "char_ratio": 0.05,
                "coverage": 0.20,
                "embed_cosine": 0.15,
                "nli_entailment": 0.05,
            }
        total = sum(filtered.values())
        if total <= 0.0:
            filtered["unigram_jaccard"] = 1.0
            total = 1.0
        return {k: v / total for k, v in filtered.items()}

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _empty_evidence(self, *, flag: Optional[str] = None) -> Dict:
        e = {
            "metrics": {},
            "unigram_overlap": {"only_in_intent": [], "only_in_action": []},
            "bigram_overlap": {"only_in_intent": [], "only_in_action": []},
            "model_availability": {"embeddings": False, "nli": False},
        }
        if flag:
            e["notes"] = [flag]
        return e
