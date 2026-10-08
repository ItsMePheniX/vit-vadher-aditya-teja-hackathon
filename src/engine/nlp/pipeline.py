"""Pipeline: process(RawItem) -> list[Signal] + JSONL writer.

Processing steps for each RawItem
----------------------------------
1. Entity linking  — find all tickers mentioned (cashtag + alias + context).
   Fall back to entity="MARKET" if none found.
2. Sentiment       — FinBERT score ∈ [-1, 1] on the full text.
3. Event classify  — zero-shot NLI over 8 event types.
4. Evidence        — top 1-2 sentences per entity.
5. Half-life       — looked up from config.yaml nlp.half_life_hours.
6. Impact          — calibrated event-study GB model (src.engine.impact).
   Falls back to rule-based formula when model is not yet trained.
   Train with: python scripts/train_impact.py --offline
7. Signal assembly — construct frozen Signal with ULID id.

One Signal is produced per linked entity (often just one).
If a RawItem mentions JPM and BAC, two Signals are returned.

JSONL Writer
------------
write_signals(signals) appends each signal as a JSON line to the path
configured in nlp.signals_path (default: data/signals.jsonl).
A module-level threading.Lock ensures safety when called from thread pools.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from ulid import ULID

from src.config import get as cfg
from src.engine.impact import score_impact
from src.engine.ingest.base import RawItem
from src.engine.linker import EntityLinker
from src.engine.nlp import classifier as cls_mod
from src.engine.nlp import evidence as ev_mod
from src.engine.nlp import sentiment as sent_mod
from src.engine.schema import EventType, Signal

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_SIGNALS_PATH = Path(cfg("nlp", "signals_path", default="data/signals.jsonl"))
_HALF_LIFE: dict[str, float] = cfg("nlp", "half_life_hours", default={}) or {}

_DEFAULT_HALF_LIFE: dict[str, float] = {
    "Geopolitical":       72.0,
    "Macroeconomic":      48.0,
    "Credit Event":       96.0,
    "Merger/Acquisition": 168.0,
    "Earnings":           12.0,
    "Regulatory":         96.0,
    "Product Launch":     36.0,
    "Other":              24.0,
}

_linker   = EntityLinker()
_write_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _half_life(event_type: str) -> float:
    """Return half-life hours from config, with hard-coded fallback."""
    hl = _HALF_LIFE or _DEFAULT_HALF_LIFE
    return float(hl.get(event_type, _DEFAULT_HALF_LIFE.get(event_type, 24.0)))


_impact_warned = False


def _impact(
    sentiment_score: float,
    event_type: str,
    source_count: int = 1,
    velocity: float = 1.0,
    entity: str = "MARKET",
) -> tuple[int, str]:
    """Return (impact, interpretation) from the calibrated event-study model.

    Delegates to src.engine.impact.score_impact which uses a trained GB regressor
    when available, or falls back to a rule-based formula.
    See docs/ASSUMPTIONS.md §Impact Score Calibration.
    """
    global _impact_warned
    from src.engine.impact.scorer import FALLBACK_MODE  # noqa: PLC0415
    if FALLBACK_MODE and not _impact_warned:
        logger.warning(
            "Impact model not trained — using rule-based fallback. "
            "Run `python scripts/train_impact.py --offline` to calibrate."
        )
        _impact_warned = True
    return score_impact(sentiment_score, event_type, source_count, velocity, entity)


def _make_signal(
    item: RawItem,
    entity: str,
    text: str,
    sentiment_result: sent_mod.SentimentResult,
    classify_result: cls_mod.ClassifyResult,
) -> Signal:
    evidence   = ev_mod.extract(text, entity)
    half_life  = _half_life(classify_result.label)

    # Derive velocity from the item if available (RawItem may carry metadata)
    source_count = getattr(item, "source_count", 1) or 1
    velocity     = getattr(item, "velocity", 1.0) or 1.0

    impact, impact_interp = _impact(
        sentiment_result.score,
        classify_result.label,
        source_count=source_count,
        velocity=velocity,
        entity=entity,
    )

    # Blend model confidences: weight sentiment 60 %, classification 40 %
    blended_conf = round(
        0.6 * sentiment_result.confidence + 0.4 * classify_result.probability, 4
    )

    sources: list[str] = [item.source]
    if item.url:
        sources.append(item.url)

    return Signal(
        id=str(ULID()),
        ts=item.ts if item.ts.tzinfo else item.ts.replace(tzinfo=timezone.utc),
        entity=entity,
        sentiment=sentiment_result.score,
        event_type=classify_result.label,
        impact=impact,
        confidence=blended_conf,
        half_life_hours=half_life,
        evidence=evidence,
        sources=sources,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def process(item: RawItem) -> list[Signal]:
    """Run a single RawItem through the full NLP pipeline.

    Returns a list of Signals — one per linked entity (minimum one for MARKET).

    The function is CPU-bound (model inference); call it via asyncio.to_thread
    from async contexts to avoid blocking the event loop.
    """
    text = item.text.strip()
    if not text:
        logger.warning("process() called with empty text (id=%s)", item.id)
        return []

    # Shared NLP work: same text → same sentiment and event type
    # (per-entity evidence differs, so we extract it inside the loop)
    try:
        sentiment_result = sent_mod.analyse(text)
    except Exception as exc:  # noqa: BLE001
        logger.error("Sentiment analysis failed for %s: %s", item.id, exc)
        sentiment_result = sent_mod.SentimentResult(score=0.0, confidence=0.0)

    try:
        classify_result = cls_mod.classify(text)
    except Exception as exc:  # noqa: BLE001
        import traceback
        logger.error("Event classification failed for %s: %s\n%s", item.id, exc, traceback.format_exc())
        classify_result = cls_mod.ClassifyResult(label="Other", probability=0.0)  # type: ignore[arg-type]

    # Entity linking
    links = _linker.link(text)

    if not links:
        # No entity found → single MARKET-wide signal
        signal = _make_signal(item, "MARKET", text, sentiment_result, classify_result)
        logger.debug("Signal [MARKET]: %s  impact=%d", classify_result.label, signal.impact)
        return [signal]

    signals: list[Signal] = []
    for link in links:
        signal = _make_signal(item, link.ticker, text, sentiment_result, classify_result)
        logger.debug(
            "Signal [%s]: sent=%.3f event=%s impact=%d",
            link.ticker, signal.sentiment, signal.event_type, signal.impact,
        )
        signals.append(signal)

    return signals


def write_signals(signals: list[Signal], path: Path | None = None) -> None:
    """Append signals to the JSONL file (append-only, thread-safe).

    Args:
        signals: List of Signal objects to persist.
        path:    Override output path (defaults to config nlp.signals_path).
    """
    out = path or _SIGNALS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    with _write_lock:
        with out.open("a", encoding="utf-8") as fh:
            for signal in signals:
                fh.write(signal.model_dump_json() + "\n")

    logger.debug("Wrote %d signal(s) to %s", len(signals), out)
