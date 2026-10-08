"""FinBERT sentiment analysis.

Model: ProsusAI/finbert  (Apache-2.0)
  — Fine-tuned on financial news: 10-K filings, earnings calls, Bloomberg.

Score formula
-------------
  score = positive_prob - negative_prob   ∈ [-1.0, 1.0]

  positive_prob ≈ 1  →  score ≈ +1  (strongly positive sentiment)
  negative_prob ≈ 1  →  score ≈ -1  (strongly negative sentiment)
  neutral text       →  score ≈  0

Confidence is max(positive_prob, negative_prob, neutral_prob).

Both the model and per-text results are cached:
  - The HuggingFace pipeline is loaded lazily once and held in a module global.
  - Text → (score, confidence) pairs are cached in an LRU dict of configurable
    size (default 512 entries) to avoid re-running inference on duplicate text.

Thread safety: the module-level lock serialises model loading; inference calls
are independently thread-safe (HuggingFace pipelines release the GIL during
tokenisation).
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import NamedTuple

from src.config import get as cfg

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_MODEL_NAME  = cfg("nlp", "sentiment_model",    default="ProsusAI/finbert")
_MAX_LENGTH  = cfg("nlp", "sentiment_max_length", default=512)
_BATCH_SIZE  = cfg("nlp", "sentiment_batch_size", default=8)
_CACHE_SIZE  = cfg("nlp", "sentiment_cache_size", default=512)
_DEVICE      = cfg("nlp", "device",              default="cpu")

# ── Model state ───────────────────────────────────────────────────────────────
_pipe       = None
_load_lock  = threading.Lock()

# ── LRU text cache ────────────────────────────────────────────────────────────
_cache: OrderedDict[str, tuple[float, float]] = OrderedDict()
_cache_lock = threading.Lock()


class SentimentResult(NamedTuple):
    score: float        # [-1.0, 1.0]  positive_prob - negative_prob
    confidence: float   # [0.0,  1.0]  max label probability


# ── Model loader ──────────────────────────────────────────────────────────────

def _load() -> None:
    """Lazy-load FinBERT pipeline (thread-safe, called at most once)."""
    global _pipe
    if _pipe is not None:
        return
    with _load_lock:
        if _pipe is not None:
            return
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        logger.info("Loading FinBERT: %s (device=%s) …", _MODEL_NAME, _DEVICE)
        _pipe = hf_pipeline(
            "text-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            device=0 if _DEVICE == "cuda" else -1,
            max_length=_MAX_LENGTH,
            truncation=True,
            top_k=None,          # return all 3 labels every time
        )
        logger.info("FinBERT ready.")


# ── Internal inference ────────────────────────────────────────────────────────

def _infer_one(text: str) -> SentimentResult:
    """Run FinBERT on a single text string (no caching)."""
    _load()
    label_scores: list[dict] = _pipe(text[:_MAX_LENGTH])[0]  # type: ignore[index]

    probs: dict[str, float] = {d["label"].lower(): d["score"] for d in label_scores}
    p_pos = probs.get("positive", 0.0)
    p_neg = probs.get("negative", 0.0)
    p_neu = probs.get("neutral",  0.0)

    score      = round(p_pos - p_neg, 4)
    confidence = round(max(p_pos, p_neg, p_neu), 4)
    return SentimentResult(score=score, confidence=confidence)


def _cache_get(key: str) -> SentimentResult | None:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return SentimentResult(*_cache[key])
    return None


def _cache_set(key: str, result: SentimentResult) -> None:
    with _cache_lock:
        _cache[key] = (result.score, result.confidence)
        _cache.move_to_end(key)
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)  # evict LRU


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(text: str) -> SentimentResult:
    """Return (score, confidence) for a single text string.

    Results are cached; duplicate texts incur no model cost.
    """
    text = text.strip()
    cached = _cache_get(text)
    if cached is not None:
        return cached
    result = _infer_one(text)
    _cache_set(text, result)
    return result


def analyse_batch(texts: list[str]) -> list[SentimentResult]:
    """Batch inference — skips cache hits, runs model only on misses.

    Maintains input order in output.
    """
    texts = [t.strip() for t in texts]
    results: list[SentimentResult | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts:   list[str] = []

    for i, text in enumerate(texts):
        cached = _cache_get(text)
        if cached is not None:
            results[i] = cached
        else:
            miss_indices.append(i)
            miss_texts.append(text)

    if miss_texts:
        _load()
        raw_batch = _pipe(miss_texts, batch_size=_BATCH_SIZE)  # type: ignore[call-arg,misc]
        for idx, label_scores in zip(miss_indices, raw_batch):
            probs = {d["label"].lower(): d["score"] for d in label_scores}
            p_pos = probs.get("positive", 0.0)
            p_neg = probs.get("negative", 0.0)
            p_neu = probs.get("neutral",  0.0)
            r = SentimentResult(
                score=round(p_pos - p_neg, 4),
                confidence=round(max(p_pos, p_neg, p_neu), 4),
            )
            _cache_set(texts[idx], r)
            results[idx] = r

    return results  # type: ignore[return-value]


def cache_info() -> dict[str, int]:
    """Return current cache occupancy stats."""
    with _cache_lock:
        return {"size": len(_cache), "maxsize": _CACHE_SIZE}
