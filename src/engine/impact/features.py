"""features.py — leakage-safe feature engineering for the impact model.

Feature Set
-----------
All features MUST be computable from information available AT OR BEFORE
the event date.  Each feature is annotated with its leakage class:

    [SAFE]    — derived from the news text itself (before event)
    [SAFE]    — derived from event metadata (known at publication time)
    [LEAKAGE] — uses price data from on or after event date (FORBIDDEN)

Features used (all [SAFE]):
    sentiment_abs   — |FinBERT score|, captures magnitude
    sentiment       — signed score, captures direction
    sentiment_sq    — sentiment², captures non-linearity
    is_positive     — 1 if sentiment > 0.05 else 0
    source_count    — distinct sources covering the event (known at t)
    velocity        — signals/day in 24h window (known at t)
    log_velocity    — log1p(velocity), reduces skew
    event_*         — one-hot encoding of 8 event types

Leakage Audit
-------------
The function `leakage_check(df)` inspects the DataFrame for any columns
that appear to encode forward-looking price information.  It logs a warning
and returns the list of suspect columns.  This is called automatically by
`build_feature_matrix()`.

Column naming convention: any column starting with "ar", "ret", "price",
or "future" is flagged.  The model never sees these columns as features.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical event types (same as schema.py EventType)
EVENT_TYPES = [
    "Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition",
    "Earnings", "Regulatory", "Product Launch", "Other",
]

# Features the model is trained on — FROZEN (changing breaks saved models)
FEATURE_COLUMNS: list[str] = [
    "sentiment_abs",
    "sentiment",
    "sentiment_sq",
    "is_positive",
    "source_count",
    "velocity",
    "log_velocity",
] + [f"event_{t.lower().replace(' ', '_').replace('/', '_')}" for t in EVENT_TYPES]

# Patterns that indicate potential leakage (used by leakage_check)
_LEAKAGE_PATTERNS = ("ar", "ret", "price", "future", "next_day", "t1", "t2", "t3")


def leakage_check(df: pd.DataFrame) -> list[str]:
    """Scan df for columns that might encode post-event information.

    Logs a WARNING for each suspect column.  Returns list of suspect column names.
    This should be called BEFORE passing any DataFrame to the model.
    """
    suspect = [
        col for col in df.columns
        if any(col.lower().startswith(p) for p in _LEAKAGE_PATTERNS)
    ]
    if suspect:
        msg = (
            f"⚠️  LEAKAGE WARNING: {len(suspect)} column(s) may encode "
            f"forward-looking information: {suspect}. "
            "These must NOT be passed as model features."
        )
        logger.warning(msg)
        warnings.warn(msg, stacklevel=3)
    return suspect


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y1, y3) from an event-study DataFrame.

    Args:
        df: Output of event_study.get_abnormal_returns() — contains both
            features AND labels.  We separate them here to prevent leakage.

    Returns:
        X   — feature DataFrame (FEATURE_COLUMNS only)
        y1  — abs_ar1 (|abnormal return| at t+1) — regression target
        y3  — abs_ar3 (|CAR| over t+1..t+3) — secondary target

    Leakage guarantee:
        The returned X contains ONLY columns from FEATURE_COLUMNS.
        Any price-derived column (ar1, ar3, abs_ar1, abs_ar3) is excluded.
        leakage_check() is called on the full df and warns if suspicious
        columns are detected.
    """
    # ── Leakage audit on the raw df ───────────────────────────────────────────
    leakage_check(df)

    # ── Feature engineering ───────────────────────────────────────────────────
    feat = pd.DataFrame(index=df.index)

    # Sentiment features (all from text, pre-event)
    sent = df["sentiment"].clip(-1.0, 1.0)
    feat["sentiment_abs"] = sent.abs()
    feat["sentiment"]     = sent
    feat["sentiment_sq"]  = sent ** 2
    feat["is_positive"]   = (sent > 0.05).astype(float)

    # Metadata features (known at publication time)
    feat["source_count"] = df["source_count"].clip(1, 20).astype(float)
    feat["velocity"]     = df["velocity"].clip(0, 100).astype(float)
    feat["log_velocity"] = np.log1p(feat["velocity"])

    # Event-type one-hot encoding
    for et in EVENT_TYPES:
        col = f"event_{et.lower().replace(' ', '_').replace('/', '_')}"
        feat[col] = (df["event_type"] == et).astype(float)

    # ── Labels (post-event prices — never used as features) ───────────────────
    y1 = df["abs_ar1"].clip(0.0)   # |AR1|
    y3 = df["abs_ar3"].clip(0.0)   # |CAR3|

    # Verify column set matches frozen list
    missing = set(FEATURE_COLUMNS) - set(feat.columns)
    extra   = set(feat.columns) - set(FEATURE_COLUMNS)
    if missing:
        logger.warning("Missing expected features: %s — filling with 0", missing)
        for col in missing:
            feat[col] = 0.0
    if extra:
        logger.warning("Extra features dropped to prevent leakage: %s", extra)
        feat = feat[FEATURE_COLUMNS]

    return feat[FEATURE_COLUMNS], y1, y3


def features_from_signal(
    sentiment: float,
    event_type: str,
    source_count: int = 1,
    velocity: float = 1.0,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for inference at runtime.

    Used by scorer.score_impact() to score live signals.
    No leakage possible here — all inputs are from the signal itself.
    """
    sent = float(np.clip(sentiment, -1.0, 1.0))
    row: dict[str, float] = {
        "sentiment_abs": abs(sent),
        "sentiment":     sent,
        "sentiment_sq":  sent ** 2,
        "is_positive":   float(sent > 0.05),
        "source_count":  float(min(max(source_count, 1), 20)),
        "velocity":      float(min(max(velocity, 0), 100)),
        "log_velocity":  float(np.log1p(velocity)),
    }
    for et in EVENT_TYPES:
        col = f"event_{et.lower().replace(' ', '_').replace('/', '_')}"
        row[col] = float(event_type == et)

    return pd.DataFrame([row])[FEATURE_COLUMNS]
