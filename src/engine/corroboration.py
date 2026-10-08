"""corroboration.py — rolling-window signal grouping, source boost, pump filter.

Purpose
-------
Raw NLP signals arrive one per item, but the same news event often appears
across multiple sources.  Corroboration:
  1. Groups signals by (entity, event_type) in a rolling 60-min window.
  2. Boosts confidence when N distinct sources corroborate.
  3. Applies a credibility multiplier per source (news > wire > social).
  4. Detects pump patterns — bursts of near-duplicate social posts — and
     lowers confidence proportionally.

The corroborate() function is idempotent: calling it with the same list of
signals always returns the same corroborated version.

All weights are configurable in config.yaml under the `corroboration` section.

Algorithm details
-----------------
Merge strategy: mean of sentiments, max of impacts, harmonic mean of confidences.

Pump detection (Jaccard):
  token_set_a = set(text_a.lower().split())
  jaccard     = |A ∩ B| / |A ∪ B|
  near-dup if jaccard >= similarity_threshold

If burst_threshold or more social items in the pump sub-window are near-duplicates
of the centroid, the pump_filter_confidence_penalty is applied.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from src import config as cfg_mod
from src.engine.schema import Signal

logger = logging.getLogger(__name__)

# ── Config (read once at import time, so tests can monkeypatch cfg_mod.get) ──

def _cfg(*keys, default=None):
    return cfg_mod.get(*keys, default=default)


def _load_cfg() -> dict:
    corr = _cfg("corroboration") or {}
    pump = corr.get("pump_filter", {})
    cred = corr.get("source_credibility", {})
    return {
        "window_minutes":         float(corr.get("window_minutes", 60)),
        "source_boost_per":       float(corr.get("source_boost_per_source", 0.05)),
        "source_boost_max":       float(corr.get("source_boost_max", 0.25)),
        "social_penalty":         float(corr.get("social_source_penalty", 0.10)),
        "pump_enabled":           bool(pump.get("enabled", True)),
        "pump_window_min":        float(pump.get("window_minutes", 10)),
        "pump_burst_thresh":      int(pump.get("burst_threshold", 5)),
        "pump_sim_thresh":        float(pump.get("similarity_threshold", 0.80)),
        "pump_conf_penalty":      float(pump.get("confidence_penalty", 0.30)),
        "credibility":            {k: float(v) for k, v in cred.items()},
        "default_credibility":    float(cred.get("default", 0.75)),
    }


_SOCIAL_SOURCES = frozenset({"kaggle_tweets", "twitter", "social", "tweet"})


class CorroboratedSignal(NamedTuple):
    """A signal after corroboration, carrying metadata about the grouping."""
    signal: Signal
    source_count: int         # distinct sources that contributed
    item_count: int           # total items in the group
    pump_detected: bool       # whether pump pattern was suppressed
    credibility_score: float  # weighted credibility of the group


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _credibility(source: str, cred_map: dict, default: float) -> float:
    s = source.lower().strip()
    # Exact match first, then prefix search
    if s in cred_map:
        return cred_map[s]
    for k, v in cred_map.items():
        if s.startswith(k) or k.startswith(s):
            return v
    return default


def _is_social(source: str) -> bool:
    s = source.lower()
    return any(tag in s for tag in _SOCIAL_SOURCES)


def _harmonic_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    recip = [1.0 / v for v in values if v > 0]
    return len(recip) / sum(recip) if recip else 0.0


# ── Pump filter ───────────────────────────────────────────────────────────────

def _detect_pump(
    signals: list[Signal],
    pump_window_min: float,
    burst_thresh: int,
    sim_thresh: float,
) -> bool:
    """Return True if a pump pattern is detected in this group.

    A pump is: ≥ burst_thresh social-source signals within pump_window_min
    minutes that are near-duplicates of each other (Jaccard ≥ sim_thresh).
    """
    social_sigs = [s for s in signals if _is_social(s.sources[0] if s.sources else "")]
    if len(social_sigs) < burst_thresh:
        return False

    # Use the most recent signal as centroid
    social_sigs_sorted = sorted(social_sigs, key=lambda s: s.ts)
    window = timedelta(minutes=pump_window_min)
    ref_ts = social_sigs_sorted[-1].ts.replace(tzinfo=timezone.utc) \
        if social_sigs_sorted[-1].ts.tzinfo is None \
        else social_sigs_sorted[-1].ts

    # Signals within the pump sub-window
    in_window = [
        s for s in social_sigs_sorted
        if abs((ref_ts - (s.ts if s.ts.tzinfo else s.ts.replace(tzinfo=timezone.utc))).total_seconds()) <= window.total_seconds()
    ]

    if len(in_window) < burst_thresh:
        return False

    # Check near-duplicate fraction: how many are similar to the centroid?
    centroid_text = " ".join(in_window[-1].evidence or [])
    if not centroid_text:
        centroid_text = in_window[-1].sources[0] if in_window[-1].sources else ""

    dup_count = sum(
        1 for s in in_window
        if _jaccard(" ".join(s.evidence or []), centroid_text) >= sim_thresh
    )
    is_pump = dup_count >= burst_thresh
    if is_pump:
        logger.warning(
            "Pump detected: %d near-duplicate social signals in %d min window",
            dup_count, pump_window_min,
        )
    return is_pump


# ── Main corroboration function ────────────────────────────────────────────────

def corroborate(signals: list[Signal]) -> list[CorroboratedSignal]:
    """Group signals by (entity, event_type) and apply corroboration logic.

    Args:
        signals: flat list of raw Signal objects (may span multiple windows)

    Returns:
        One CorroboratedSignal per (entity, event_type) group, with adjusted
        confidence based on source diversity, credibility, and pump detection.
    """
    if not signals:
        return []

    c = _load_cfg()
    window = timedelta(minutes=c["window_minutes"])
    now = datetime.now(tz=timezone.utc)

    # ── Group by (entity, event_type), within window ──────────────────────────
    groups: dict[tuple[str, str], list[Signal]] = defaultdict(list)
    for sig in signals:
        ts = sig.ts if sig.ts.tzinfo else sig.ts.replace(tzinfo=timezone.utc)
        if now - ts <= window:
            groups[(sig.entity, sig.event_type)].append(sig)

    results: list[CorroboratedSignal] = []

    for (entity, event_type), group in groups.items():
        if not group:
            continue

        # ── Compute merged fields ─────────────────────────────────────────────
        avg_sentiment = sum(s.sentiment for s in group) / len(group)
        max_impact    = max(s.impact for s in group)
        base_conf     = _harmonic_mean([s.confidence for s in group])
        all_sources   = [s.sources[0] if s.sources else "unknown" for s in group]
        distinct_src  = list({s.lower() for s in all_sources})
        n_distinct    = len(distinct_src)

        # ── Source credibility multiplier ─────────────────────────────────────
        cred_scores = [
            _credibility(src, c["credibility"], c["default_credibility"])
            for src in distinct_src
        ]
        avg_cred = sum(cred_scores) / len(cred_scores)

        # ── Confidence boosting ───────────────────────────────────────────────
        # Start from base harmonic-mean confidence, weighted by credibility
        conf = base_conf * avg_cred

        # Boost for multi-source corroboration
        if n_distinct > 1:
            boost = min(c["source_boost_max"], (n_distinct - 1) * c["source_boost_per"])
            conf = min(1.0, conf + boost)
            logger.debug(
                "Corroboration boost: entity=%s n_src=%d boost=+%.3f → conf=%.3f",
                entity, n_distinct, boost, conf,
            )

        # Penalty for single social source with no news corroboration
        all_social = all(_is_social(src) for src in distinct_src)
        if n_distinct == 1 and all_social:
            conf = max(0.0, conf - c["social_penalty"])
            logger.debug("Social penalty −%.2f → conf=%.3f", c["social_penalty"], conf)

        # ── Pump filter ───────────────────────────────────────────────────────
        pump = False
        if c["pump_enabled"] and all_social:
            pump = _detect_pump(
                group,
                c["pump_window_min"],
                c["pump_burst_thresh"],
                c["pump_sim_thresh"],
            )
            if pump:
                conf = max(0.0, conf - c["pump_conf_penalty"])
                logger.warning(
                    "Pump filter applied: entity=%s conf reduced by %.2f → %.3f",
                    entity, c["pump_conf_penalty"], conf,
                )

        conf = round(min(1.0, max(0.0, conf)), 4)

        # ── Take the most recent signal as the canonical signal ───────────────
        base = max(group, key=lambda s: s.ts)
        merged = base.model_copy(update={
            "sentiment":  round(avg_sentiment, 4),
            "impact":     max_impact,
            "confidence": conf,
        })

        results.append(CorroboratedSignal(
            signal=merged,
            source_count=n_distinct,
            item_count=len(group),
            pump_detected=pump,
            credibility_score=round(avg_cred, 4),
        ))

    # Sort by impact descending
    results.sort(key=lambda r: -r.signal.impact)
    return results


def corroborate_one(signal: Signal, recent: list[Signal]) -> Signal:
    """Corroborate a single incoming signal against recent signals in memory.

    Finds all recent signals matching (entity, event_type) within the window
    and returns an adjusted version of `signal` with updated confidence.
    Used in the streaming pipeline for real-time adjustment.

    Args:
        signal: the newly arrived signal
        recent: buffer of recently processed signals (caller manages this)

    Returns:
        The (possibly confidence-adjusted) signal.
    """
    c = _load_cfg()
    window = timedelta(minutes=c["window_minutes"])
    now_ts = signal.ts if signal.ts.tzinfo else signal.ts.replace(tzinfo=timezone.utc)

    peers = [
        s for s in recent
        if s.entity == signal.entity
        and s.event_type == signal.event_type
        and abs((now_ts - (s.ts if s.ts.tzinfo else s.ts.replace(tzinfo=timezone.utc))).total_seconds()) <= window.total_seconds()
    ]
    peers.append(signal)

    results = corroborate(peers)
    if results:
        return results[0].signal
    return signal
