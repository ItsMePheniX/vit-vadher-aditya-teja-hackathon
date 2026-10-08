"""src/engine/impact — event-study calibration for the impact score.

Public API
----------
    from src.engine.impact import score_impact, FALLBACK_MODE

    # Returns int [1,10] and the interpretation string
    impact, interp = score_impact(
        sentiment=0.72,
        event_type="Earnings",
        source_count=3,
        velocity=5.0,
        entity="JPM",
        event_ts=datetime.now(tz=timezone.utc),
    )

Modules
-------
    event_study  — download prices, compute abnormal returns, cache to CSV
    features     — leakage-safe feature engineering
    train        — GB regressor + quantile bucket mapping, time-split eval
    scorer       — load model/mapping at runtime, fallback if missing
"""
from src.engine.impact.scorer import score_impact, FALLBACK_MODE

__all__ = ["score_impact", "FALLBACK_MODE"]
