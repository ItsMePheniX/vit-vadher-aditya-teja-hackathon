"""scorer.py — load trained impact model at runtime; rule-based fallback.

Behaviour
---------
On first call to score_impact(), this module tries to load:
    data/impact_model/gb_model.joblib
    data/impact_model/quantile_map.json

If either is missing, FALLBACK_MODE = True and the rule-based formula is used:

    raw   = |sentiment| × event_weight × 10
    impact = clamp(ceil(raw), 1, 10)

The rule-based fallback intentionally replicates the method in the original
placeholder but with per-event-type weights, so it degrades gracefully.

When the model IS loaded:
    features = features_from_signal(sentiment, event_type, source_count, velocity)
    predicted_ar = model.predict(features)[0]
    impact = quantile_bucket(predicted_ar, mapping)

Thread safety: module-level lazy load protected by a threading.Lock.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_PATH    = Path("data/impact_model/gb_model.joblib")
_QUANTILE_PATH = Path("data/impact_model/quantile_map.json")

_model   = None
_mapping: list[dict] = []
_loaded  = False
_lock    = threading.Lock()

# Public flag: True when using fallback, False when using trained model
FALLBACK_MODE: bool = True


# ── Rule-based fallback ───────────────────────────────────────────────────────

_FALLBACK_WEIGHTS: dict[str, float] = {
    "Geopolitical":       1.40,
    "Credit Event":       1.35,
    "Regulatory":         1.20,
    "Macroeconomic":      1.15,
    "Merger/Acquisition": 1.20,
    "Earnings":           1.10,
    "Product Launch":     0.90,
    "Other":              1.00,
}


def _fallback_impact(sentiment: float, event_type: str) -> int:
    """Rule-based impact: |sentiment| × event_weight × 10, clamped to [1,10]."""
    weight = _FALLBACK_WEIGHTS.get(event_type, 1.0)
    raw    = abs(sentiment) * weight * 10.0
    return int(max(1, min(10, math.ceil(raw))))


def _fallback_interp(impact: int) -> str:
    return f"impact {impact} (rule-based fallback — train model with `make train`)"


# ── Model loading ─────────────────────────────────────────────────────────────

def _load() -> None:
    global _model, _mapping, _loaded, FALLBACK_MODE
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        if _MODEL_PATH.exists() and _QUANTILE_PATH.exists():
            try:
                import joblib  # noqa: PLC0415
                _model   = joblib.load(_MODEL_PATH)
                _mapping = json.loads(_QUANTILE_PATH.read_text())
                FALLBACK_MODE = False
                logger.info(
                    "Impact model loaded from %s (buckets: %d)",
                    _MODEL_PATH, len(_mapping),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load impact model (%s) — using fallback.", exc)
                FALLBACK_MODE = True
        else:
            logger.info(
                "No trained impact model found at %s — using rule-based fallback. "
                "Run `make train` to calibrate.", _MODEL_PATH,
            )
            FALLBACK_MODE = True
        _loaded = True


# ── Public API ────────────────────────────────────────────────────────────────

def score_impact(
    sentiment: float,
    event_type: str,
    source_count: int = 1,
    velocity: float = 1.0,
    entity: str = "MARKET",
    **_kwargs,
) -> tuple[int, str]:
    """Return (impact: int [1,10], interpretation: str).

    Uses the trained GB model when available; falls back to rule-based formula.

    Args:
        sentiment:    FinBERT score [-1, 1]
        event_type:   One of the 8 EventType labels
        source_count: Distinct news sources covering this event
        velocity:     Signals per day in the 24h window before the event
        entity:       Ticker symbol (unused currently; reserved for per-entity bias)

    Returns:
        (impact, interpretation_string)
    """
    _load()

    if FALLBACK_MODE or _model is None:
        impact = _fallback_impact(sentiment, event_type)
        return impact, _fallback_interp(impact)

    from src.engine.impact.features import features_from_signal  # noqa: PLC0415
    from src.engine.impact.train import predicted_to_impact       # noqa: PLC0415

    try:
        X = features_from_signal(sentiment, event_type, source_count, velocity)
        y_pred = float(_model.predict(X)[0])
        impact = predicted_to_impact(y_pred, _mapping)

        # Find interpretation from mapping
        interp = f"impact {impact}: model predicts ~{y_pred*100:.2f}% abnormal move vs SPY"
        for bucket in _mapping:
            if bucket["impact"] == impact:
                interp = bucket["interp"]
                break

        return impact, interp

    except Exception as exc:  # noqa: BLE001
        logger.warning("Model inference failed (%s) — falling back to rule-based.", exc)
        impact = _fallback_impact(sentiment, event_type)
        return impact, _fallback_interp(impact)


def reload_model() -> None:
    """Force a model reload (useful after retraining without restarting)."""
    global _loaded
    with _lock:
        _loaded = False
    _load()
