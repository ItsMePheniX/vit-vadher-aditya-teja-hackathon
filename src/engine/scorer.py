"""Signal scorer: computes impact, half_life_hours from sentiment + event_type.

Design rationale (see docs/ASSUMPTIONS.md):
- Impact [1-10] is a weighted function of |sentiment| * confidence * event_weight.
- Half-life is looked up from config.yaml per event type.
  It represents how many hours before the signal is expected to halve
  in market relevance, inspired by information decay models in finance.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from src import config
from src.engine.schema import EventType

# Event-type → config key for half-life lookup
_HALF_LIFE_KEY: dict[EventType, str] = {
    "Geopolitical": "geopolitical_hours",
    "Macroeconomic": "macroeconomic_hours",
    "Credit Event": "credit_event_hours",
    "Merger/Acquisition": "merger_acquisition_hours",
    "Earnings": "earnings_hours",
    "Regulatory": "regulatory_hours",
    "Product Launch": "product_launch_hours",
    "Other": "other_hours",
}

_IMPACT_WEIGHT: dict[EventType, float] = {
    "Geopolitical": config.get("scoring", "impact", "geopolitical_weight", default=1.4),
    "Credit Event": config.get("scoring", "impact", "credit_event_weight", default=1.3),
    "Earnings": config.get("scoring", "impact", "earnings_weight", default=1.1),
    "Macroeconomic": 1.0,
    "Merger/Acquisition": 1.2,
    "Regulatory": 1.1,
    "Product Launch": 0.9,
    "Other": config.get("scoring", "impact", "default_weight", default=1.0),
}


class ScoreResult(NamedTuple):
    impact: int               # [1, 10]
    half_life_hours: float    # > 0


def score(
    sentiment: float,
    confidence: float,
    event_type: EventType,
) -> ScoreResult:
    """Compute impact and half_life_hours from model outputs.

    Impact formula:
        raw = |sentiment| * confidence * event_weight * 10
        impact = clamp(ceil(raw), 1, 10)

    The ceil ensures even weak signals register at least 1.
    """
    weight = _IMPACT_WEIGHT.get(event_type, 1.0)
    raw_impact = abs(sentiment) * confidence * weight * 10.0
    impact = int(max(1, min(10, math.ceil(raw_impact))))

    # Read from new nlp.half_life_hours (flat dict: label_name → hours)
    hl_map: dict[str, float] = config.get("nlp", "half_life_hours", default={}) or {}
    _FALLBACK: dict[str, float] = {
        "Geopolitical": 72.0, "Macroeconomic": 48.0, "Credit Event": 96.0,
        "Merger/Acquisition": 168.0, "Earnings": 12.0, "Regulatory": 96.0,
        "Product Launch": 36.0, "Other": 24.0,
    }
    half_life = float(hl_map.get(event_type, _FALLBACK.get(event_type, 24.0)))

    return ScoreResult(impact=impact, half_life_hours=half_life)
