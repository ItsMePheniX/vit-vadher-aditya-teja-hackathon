"""train.py — gradient boosting training, quantile bucket mapping, time-split eval.

Pipeline
--------
1. Load events from news CSV (or accept a pre-built DataFrame).
2. Fetch abnormal returns via event_study (uses price cache).
3. Build feature matrix via features.build_feature_matrix().
4. Time-based train/test split (no leakage):
       train = events before SPLIT_DATE
       test  = events on/after SPLIT_DATE
5. Train sklearn GradientBoostingRegressor on abs_ar1.
6. Compute quantile bucket mapping: divide predicted-move space into 10 deciles,
   map each decile to impact level 1-10.
7. Evaluate on held-out test set:
       MAE, R², Spearman ρ (rank correlation), bucket accuracy.
8. Save model + mapping as joblib artifacts in data/impact_model/.

Artifacts saved
---------------
    data/impact_model/gb_model.joblib      — trained GBR
    data/impact_model/quantile_map.json    — predicted_move_pct → impact 1-10
    data/impact_model/meta.json            — training metadata, metrics, split date

Usage
-----
    python scripts/train_impact.py [--news data/svb_news_sample.csv] [--target ar1|ar3]
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

from src.engine.impact.features import build_feature_matrix, FEATURE_COLUMNS, leakage_check

logger = logging.getLogger(__name__)

_MODEL_DIR = Path("data/impact_model")
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH     = _MODEL_DIR / "gb_model.joblib"
QUANTILE_PATH  = _MODEL_DIR / "quantile_map.json"
META_PATH      = _MODEL_DIR / "meta.json"

# Time split: train on events before this date, test on after
DEFAULT_SPLIT_DATE = date(2023, 3, 13)   # mid-SVB crisis

# GBR hyperparameters (tuned for small datasets)
GBR_PARAMS = {
    "n_estimators":    200,
    "max_depth":       3,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "min_samples_leaf": 2,
    "random_state":    42,
}

N_IMPACT_BUCKETS = 10


# ── Quantile mapping ──────────────────────────────────────────────────────────

def _build_quantile_map(
    y_pred_train: np.ndarray,
) -> list[dict]:
    """Divide predicted |AR| into N_IMPACT_BUCKETS quantile buckets.

    Returns a list of N dicts (sorted ascending), each with:
        impact       — int [1, N]
        threshold    — lower bound of predicted |AR| for this bucket (as %)
        upper        — upper bound (as %)
        interp       — human-readable string, e.g. "impact 7 ≈ 2.1–3.4% abnormal move"
    """
    quantiles = np.percentile(y_pred_train, np.linspace(0, 100, N_IMPACT_BUCKETS + 1))

    mapping = []
    for i in range(N_IMPACT_BUCKETS):
        lo = float(quantiles[i])
        hi = float(quantiles[i + 1]) if i < N_IMPACT_BUCKETS - 1 else float("inf")
        impact = i + 1
        lo_pct = lo * 100
        hi_pct = hi * 100 if hi != float("inf") else None
        hi_str = f"{hi_pct:.2f}%" if hi_pct else "∞"
        mapping.append({
            "impact":    impact,
            "threshold": lo,
            "upper":     hi,
            "lo_pct":    lo_pct,
            "hi_pct":    hi_pct,
            "interp":    f"impact {impact} ≈ {lo_pct:.2f}–{hi_str} abnormal move vs SPY",
        })
    return mapping


def predicted_to_impact(y_pred: float, mapping: list[dict]) -> int:
    """Map a single predicted |AR| to impact [1, 10] using the quantile map."""
    for bucket in reversed(mapping):
        if y_pred >= bucket["threshold"]:
            return int(bucket["impact"])
    return 1


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    split_date: date = DEFAULT_SPLIT_DATE,
    target: str = "abs_ar1",
) -> dict:
    """Train the gradient boosting impact model.

    Args:
        df:         event-study DataFrame (output of event_study.get_abnormal_returns)
        split_date: events before this date → train; on/after → test
        target:     "abs_ar1" or "abs_ar3"

    Returns:
        metrics dict with keys: mae_train, mae_test, r2_test, spearman_rho,
        n_train, n_test, split_date, target, bucket_map.

    Leakage guarantee:
        The split is done on event_date (the date of the news, not price data).
        No test-set labels influence the model or the quantile mapping.
        The quantile map is fit ONLY on the training set predictions.
    """
    if len(df) < 5:
        raise ValueError(
            f"Need at least 5 events to train; got {len(df)}. "
            "Run scripts/make_sample_data.py to expand the dataset."
        )

    # ── Leakage audit ─────────────────────────────────────────────────────────
    leakage_check(df)

    X, y1, y3 = build_feature_matrix(df)
    y = y1 if target == "abs_ar1" else y3

    # ── Time-based split ──────────────────────────────────────────────────────
    train_mask = pd.to_datetime(df["event_date"]).dt.date < split_date
    test_mask  = ~train_mask

    n_train = int(train_mask.sum())
    n_test  = int(test_mask.sum())

    logger.info(
        "Time split on %s: %d train / %d test events",
        split_date, n_train, n_test,
    )

    if n_train < 3:
        raise ValueError(
            f"Only {n_train} training events before {split_date}. "
            "Adjust split_date or gather more data."
        )

    X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
    X_test,  y_test  = X[test_mask].copy(),  y[test_mask].copy()

    # ── Model training ────────────────────────────────────────────────────────
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test) if n_test > 0 else np.array([])

    # ── Quantile mapping (fit on TRAIN predictions only — no leakage) ─────────
    mapping = _build_quantile_map(y_pred_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    mae_train = float(mean_absolute_error(y_train, y_pred_train))
    mae_test  = float(mean_absolute_error(y_test, y_pred_test)) if n_test > 0 else None
    r2_test   = float(r2_score(y_test, y_pred_test)) if n_test > 1 else None
    rho, _    = spearmanr(y_test, y_pred_test) if n_test > 1 else (None, None)

    # Bucket accuracy: predicted bucket matches label bucket?
    if n_test > 0:
        label_impacts = [predicted_to_impact(v, mapping) for v in y_test.values]
        pred_impacts  = [predicted_to_impact(v, mapping) for v in y_pred_test]
        bucket_acc = float(sum(a == b for a, b in zip(label_impacts, pred_impacts)) / n_test)
    else:
        bucket_acc = None

    # ── Feature importance ────────────────────────────────────────────────────
    feat_imp = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    top_feats = sorted(feat_imp.items(), key=lambda x: -x[1])[:5]

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Impact Model — {target}  |  split: {split_date}")
    print(f"{'─'*60}")
    print(f"  Train: {n_train} events,  Test: {n_test} events")
    print(f"  MAE train: {mae_train:.4f} ({mae_train*100:.2f}% |AR|)")
    if mae_test is not None:
        print(f"  MAE test:  {mae_test:.4f} ({mae_test*100:.2f}% |AR|)")
    if r2_test is not None:
        print(f"  R² test:   {r2_test:.3f}")
    if rho is not None:
        print(f"  Spearman ρ:{rho:.3f}  (rank correlation)")
    if bucket_acc is not None:
        print(f"  Bucket acc:{bucket_acc:.1%}  (exact impact match)")
    print(f"\n  Top features by importance:")
    for feat, imp in top_feats:
        print(f"    {feat:<28} {imp:.3f}")
    print(f"\n  Impact bucket interpretation:")
    for b in mapping:
        print(f"    {b['interp']}")
    print(f"{'─'*60}\n")

    metrics = {
        "mae_train":    mae_train,
        "mae_test":     mae_test,
        "r2_test":      r2_test,
        "spearman_rho": float(rho) if rho is not None else None,
        "bucket_accuracy": bucket_acc,
        "n_train":      n_train,
        "n_test":       n_test,
        "split_date":   str(split_date),
        "target":       target,
        "top_features": [{"feature": f, "importance": round(i, 4)} for f, i in top_feats],
        "bucket_map":   mapping,
    }

    return model, mapping, metrics


def save_artifacts(model, mapping: list[dict], metrics: dict) -> None:
    """Persist model, quantile map, and metadata."""
    import joblib  # noqa: PLC0415
    joblib.dump(model, MODEL_PATH)
    QUANTILE_PATH.write_text(json.dumps(mapping, indent=2))
    META_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("Saved: %s, %s, %s", MODEL_PATH, QUANTILE_PATH, META_PATH)
    print(f"✓ Model saved to {MODEL_PATH}")
    print(f"✓ Quantile map saved to {QUANTILE_PATH}")
    print(f"✓ Metadata saved to {META_PATH}")
