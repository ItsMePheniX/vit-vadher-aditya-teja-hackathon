"""Zero-shot event-type classifier using a small NLI model.

Model: typeform/distilbert-base-uncased-mnli  (Apache-2.0, ~66 M params)
  — CPU-friendly DistilBERT fine-tuned on MultiNLI for entailment.
  — Swap for "facebook/bart-large-mnli" in config for higher accuracy (slower).

How it works
------------
The HuggingFace zero-shot-classification pipeline runs NLI between the
input text and a hypothesis for each candidate label.  The hypothesis is:

    "This financial news is about {description}."

where {description} is the human-readable label description from config.yaml.
The label with the highest entailment probability is returned.

Label mapping
-------------
config.yaml  →  nlp.event_labels  →  dict[label_name, description]
  "Earnings" → "quarterly earnings results, revenue beats or misses …"

The classifier maps between description strings (NLI candidates) and canonical
label names (Signal.event_type).

Cache
-----
Input text → (label_name, probability) pairs cached in LRU OrderedDict.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import NamedTuple

from src.config import get as cfg
from src.engine.schema import EventType

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_MODEL_NAME   = cfg("nlp", "classifier_model",    default="typeform/distilbert-base-uncased-mnli")
_HYPOTHESIS   = cfg("nlp", "classifier_hypothesis", default="This financial news is about {}.")
_CACHE_SIZE   = cfg("nlp", "classifier_cache_size", default=256)
_DEVICE       = cfg("nlp", "device",               default="cpu")

# Load label map from config; fall back to bare label names
_EVENT_LABELS: dict[str, str] = cfg("nlp", "event_labels", default={}) or {}
_ALL_LABELS: list[str] = [
    "Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition",
    "Earnings", "Regulatory", "Product Launch", "Other",
]
if not _EVENT_LABELS:
    # bare label names as descriptions (fallback)
    _EVENT_LABELS = {lbl: lbl for lbl in _ALL_LABELS}

# description → canonical label name (reverse map)
_DESC_TO_LABEL: dict[str, str] = {desc: name for name, desc in _EVENT_LABELS.items()}
_DESCRIPTIONS: list[str] = list(_EVENT_LABELS.values())

# ── Model state ───────────────────────────────────────────────────────────────
_pipe      = None
_load_lock = threading.Lock()

# ── LRU cache ─────────────────────────────────────────────────────────────────
_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
_cache_lock = threading.Lock()


class ClassifyResult(NamedTuple):
    label: EventType       # canonical label name
    probability: float     # NLI entailment score [0.0, 1.0]


# ── Model loader ──────────────────────────────────────────────────────────────

def _load() -> None:
    global _pipe
    if _pipe is not None:
        return
    with _load_lock:
        if _pipe is not None:
            return
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        logger.info("Loading NLI classifier: %s (device=%s) …", _MODEL_NAME, _DEVICE)
        _pipe = hf_pipeline(
            "zero-shot-classification",
            model=_MODEL_NAME,
            device=0 if _DEVICE == "cuda" else -1,
        )
        logger.info("NLI classifier ready.")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_get(key: str) -> ClassifyResult | None:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return ClassifyResult(*_cache[key])
    return None


def _cache_set(key: str, result: ClassifyResult) -> None:
    with _cache_lock:
        _cache[key] = (result.label, result.probability)
        _cache.move_to_end(key)
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)


# ── Public API ────────────────────────────────────────────────────────────────

def classify(text: str) -> ClassifyResult:
    """Classify text into one of the 8 event types.

    Returns the label with the highest NLI entailment probability.
    Falls back to "Other" (probability=0.0) if the model is unavailable.
    """
    text = text.strip()
    cached = _cache_get(text)
    if cached is not None:
        return cached

    try:
        _load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLI model unavailable (%s); returning 'Other'.", exc)
        return ClassifyResult(label="Other", probability=0.0)  # type: ignore[arg-type]

    # Build hypothesis-per-description candidates
    # We use the descriptions as candidate labels so the NLI model has full context
    result = _pipe(  # type: ignore[call-arg]
        text,
        candidate_labels=_DESCRIPTIONS,
        hypothesis_template=_HYPOTHESIS,
        multi_label=False,
    )

    # Best label = first in returned list (highest score)
    best_desc  = result["labels"][0]
    best_score = float(result["scores"][0])

    # Map description back to canonical label name
    label_name = _DESC_TO_LABEL.get(best_desc, "Other")

    out = ClassifyResult(label=label_name, probability=round(best_score, 4))  # type: ignore[arg-type]
    _cache_set(text, out)
    return out


def classify_batch(texts: list[str]) -> list[ClassifyResult]:
    """Classify multiple texts (each independently, not batched by the NLI pipeline)."""
    return [classify(t) for t in texts]


def label_descriptions() -> dict[str, str]:
    """Return the current label → description mapping (from config)."""
    return dict(_EVENT_LABELS)
