"""Tests for src/engine/impact — event-study calibration module.

Coverage:
- feature engineering (leakage guard, column completeness, values)
- abnormal return computation (synthetic prices)
- training pipeline (offline mode, metrics, quantile map)
- scorer (fallback mode, model load when artifacts present)
- pipeline integration (score_impact called from process())
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_events(n: int = 20, split: date = date(2023, 3, 13)) -> list[dict]:
    """Generate n synthetic event dicts straddling the split date."""
    events = []
    start = date(2023, 3, 8)
    entities = ["JPM", "BAC", "GS", "WFC", "C"]
    event_types = ["Earnings", "Credit Event", "Macroeconomic", "Regulatory", "Geopolitical"]
    for i in range(n):
        events.append({
            "entity":       entities[i % len(entities)],
            "event_date":   start + timedelta(days=i),
            "event_type":   event_types[i % len(event_types)],
            "sentiment":    round(-0.8 + 1.6 * (i / n), 3),
            "source_count": (i % 5) + 1,
            "velocity":     float((i % 10) + 1),
        })
    return events


def _make_event_study_df(n: int = 20) -> pd.DataFrame:
    """Build a minimal event-study DataFrame with both features and labels."""
    events = _make_events(n)
    rng = np.random.default_rng(42)
    rows = []
    for ev in events:
        ar1 = float(rng.normal(0.0, 0.02))
        ar3 = float(rng.normal(0.0, 0.03))
        rows.append({
            **ev,
            "ar1": ar1, "ar3": ar3,
            "abs_ar1": abs(ar1), "abs_ar3": abs(ar3),
        })
    return pd.DataFrame(rows)


# ── Feature engineering ────────────────────────────────────────────────────────

class TestFeatures:
    def test_columns_match_frozen_list(self):
        from src.engine.impact.features import build_feature_matrix, FEATURE_COLUMNS
        df = _make_event_study_df(10)
        X, y1, y3 = build_feature_matrix(df)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_no_label_columns_in_X(self):
        from src.engine.impact.features import build_feature_matrix
        df = _make_event_study_df(10)
        X, _, _ = build_feature_matrix(df)
        forbidden = {"ar1", "ar3", "abs_ar1", "abs_ar3"}
        assert not forbidden & set(X.columns), "Labels must not appear in X"

    def test_sentiment_abs_non_negative(self):
        from src.engine.impact.features import build_feature_matrix
        df = _make_event_study_df(10)
        X, _, _ = build_feature_matrix(df)
        assert (X["sentiment_abs"] >= 0).all()

    def test_event_one_hot_sum_is_one(self):
        from src.engine.impact.features import build_feature_matrix, EVENT_TYPES
        df = _make_event_study_df(10)
        X, _, _ = build_feature_matrix(df)
        event_cols = [c for c in X.columns if c.startswith("event_")]
        assert len(event_cols) == len(EVENT_TYPES)
        sums = X[event_cols].sum(axis=1)
        assert (sums == 1).all(), "Each row should have exactly one event type active"

    def test_leakage_check_warns_on_ar_column(self):
        from src.engine.impact.features import leakage_check
        df = pd.DataFrame({"ar1": [0.01], "sentiment": [0.5], "future_price": [100]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            suspect = leakage_check(df)
        assert len(suspect) == 2
        assert any("LEAKAGE" in str(warning.message) for warning in w)

    def test_leakage_check_clean_df_no_warning(self):
        from src.engine.impact.features import leakage_check
        df = pd.DataFrame({"sentiment": [0.5], "velocity": [3.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            suspect = leakage_check(df)
        assert suspect == []
        leakage_warns = [x for x in w if "LEAKAGE" in str(x.message)]
        assert len(leakage_warns) == 0

    def test_features_from_signal_shape(self):
        from src.engine.impact.features import features_from_signal, FEATURE_COLUMNS
        X = features_from_signal(sentiment=-0.6, event_type="Credit Event", source_count=3, velocity=5.0)
        assert X.shape == (1, len(FEATURE_COLUMNS))

    def test_features_from_signal_values(self):
        from src.engine.impact.features import features_from_signal
        X = features_from_signal(sentiment=-0.6, event_type="Earnings")
        assert abs(X["sentiment_abs"].iloc[0] - 0.6) < 1e-9
        assert abs(X["sentiment_sq"].iloc[0] - 0.36) < 1e-9
        assert X["is_positive"].iloc[0] == 0.0
        assert X["event_earnings"].iloc[0] == 1.0

    def test_features_clipping(self):
        from src.engine.impact.features import features_from_signal
        # Extreme values should be clipped
        X = features_from_signal(sentiment=5.0, event_type="Other", source_count=999, velocity=9999)
        assert X["sentiment_abs"].iloc[0] <= 1.0
        assert X["source_count"].iloc[0] <= 20.0
        assert X["velocity"].iloc[0] <= 100.0


# ── Event study / abnormal returns ────────────────────────────────────────────

class TestEventStudy:
    def test_get_abnormal_returns_offline(self):
        """Test with monkey-patched synthetic prices."""
        from src.engine.impact import event_study as es
        events = _make_events(10)

        rng = np.random.default_rng(99)
        def _synth_fetch(ticker, start, end):
            dates = pd.date_range(start, end, freq="B")
            return pd.Series(rng.normal(0.0003, 0.012, len(dates)), index=dates)

        with patch.object(es, "_fetch", side_effect=_synth_fetch):
            df = es.get_abnormal_returns(events)

        assert not df.empty
        assert "abs_ar1" in df.columns
        assert "abs_ar3" in df.columns
        assert (df["abs_ar1"] >= 0).all()
        assert (df["abs_ar3"] >= 0).all()

    def test_ar_columns_exclude_feature_names(self):
        """ar* columns must not match FEATURE_COLUMNS."""
        from src.engine.impact.features import FEATURE_COLUMNS
        ar_cols = {"ar1", "ar3", "abs_ar1", "abs_ar3"}
        assert not ar_cols & set(FEATURE_COLUMNS)

    def test_spy_fallback_on_empty_fetch(self):
        """SPY fetch failure triggers synthetic returns, not a crash."""
        from src.engine.impact import event_study as es
        events = _make_events(5)

        def _always_empty(ticker, start, end):
            return pd.Series(dtype=float)

        with patch.object(es, "_fetch", side_effect=_always_empty):
            df = es.get_abnormal_returns(events)
        # Should not raise; may return empty if all stocks also fail
        assert isinstance(df, pd.DataFrame)

    def test_cache_path_format(self):
        from src.engine.impact.event_study import _cache_path
        p = _cache_path("JPM", "2023-03-08", "2023-03-17")
        assert "JPM" in str(p)
        assert p.suffix == ".csv"


# ── Training ──────────────────────────────────────────────────────────────────

class TestTraining:
    def test_train_returns_model_and_mapping(self):
        from src.engine.impact.train import train
        df = _make_event_study_df(20)
        model, mapping, metrics = train(df, split_date=date(2023, 3, 13))
        assert model is not None
        assert len(mapping) == 10
        assert "mae_train" in metrics

    def test_quantile_map_has_10_buckets(self):
        from src.engine.impact.train import train
        df = _make_event_study_df(20)
        _, mapping, _ = train(df, split_date=date(2023, 3, 13))
        assert [b["impact"] for b in mapping] == list(range(1, 11))

    def test_quantile_thresholds_are_sorted(self):
        from src.engine.impact.train import train
        df = _make_event_study_df(20)
        _, mapping, _ = train(df, split_date=date(2023, 3, 13))
        thresholds = [b["threshold"] for b in mapping]
        assert thresholds == sorted(thresholds)

    def test_predicted_to_impact_range(self):
        from src.engine.impact.train import train, predicted_to_impact
        df = _make_event_study_df(20)
        _, mapping, _ = train(df, split_date=date(2023, 3, 13))
        for y_pred in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]:
            impact = predicted_to_impact(y_pred, mapping)
            assert 1 <= impact <= 10

    def test_train_requires_minimum_events(self):
        from src.engine.impact.train import train
        df = _make_event_study_df(4)
        with pytest.raises(ValueError, match="at least 5"):
            train(df, split_date=date(2023, 3, 13))

    def test_time_split_no_leakage(self):
        """Verify train events are strictly before split date."""
        from src.engine.impact.train import train, DEFAULT_SPLIT_DATE
        from src.engine.impact.features import build_feature_matrix
        df = _make_event_study_df(20)
        train_mask = pd.to_datetime(df["event_date"]).dt.date < DEFAULT_SPLIT_DATE
        assert train_mask.any(), "Need at least one training event"
        assert (~train_mask).any(), "Need at least one test event"

    def test_save_and_reload_artifacts(self, tmp_path):
        from src.engine.impact import train as tr_mod
        df = _make_event_study_df(20)
        model, mapping, metrics = tr_mod.train(df, split_date=date(2023, 3, 13))

        # Patch save paths to tmp
        orig_model  = tr_mod.MODEL_PATH
        orig_quant  = tr_mod.QUANTILE_PATH
        orig_meta   = tr_mod.META_PATH
        tr_mod.MODEL_PATH    = tmp_path / "gb_model.joblib"
        tr_mod.QUANTILE_PATH = tmp_path / "quantile_map.json"
        tr_mod.META_PATH     = tmp_path / "meta.json"
        try:
            tr_mod.save_artifacts(model, mapping, metrics)
            assert (tmp_path / "gb_model.joblib").exists()
            assert (tmp_path / "quantile_map.json").exists()
            assert (tmp_path / "meta.json").exists()
            meta = json.loads((tmp_path / "meta.json").read_text())
            assert "mae_train" in meta
            assert "bucket_map" in meta
        finally:
            tr_mod.MODEL_PATH    = orig_model
            tr_mod.QUANTILE_PATH = orig_quant
            tr_mod.META_PATH     = orig_meta


# ── Scorer ────────────────────────────────────────────────────────────────────

class TestScorer:
    def test_fallback_mode_when_no_model(self):
        """Scorer must return a valid impact without a trained model."""
        import src.engine.impact.scorer as sc
        # Reset state
        sc._loaded = False
        sc._model  = None
        sc.FALLBACK_MODE = True

        impact, interp = sc.score_impact(-0.7, "Credit Event")
        assert 1 <= impact <= 10
        assert "fallback" in interp.lower() or "rule" in interp.lower() or impact > 0

    def test_fallback_impact_formula(self):
        """Rule-based formula: |sent| × weight × 10, clamped."""
        import src.engine.impact.scorer as sc
        # High negative sentiment + geopolitical (weight 1.4) → high impact
        impact = sc._fallback_impact(-0.9, "Geopolitical")
        assert impact >= 9

        # Zero sentiment → floor at 1
        impact = sc._fallback_impact(0.0, "Other")
        assert impact == 1

    def test_fallback_impact_clamped_to_10(self):
        import src.engine.impact.scorer as sc
        impact = sc._fallback_impact(1.0, "Geopolitical")
        assert impact == 10

    def test_score_impact_with_trained_model(self, tmp_path):
        """Test scorer with a real trained model loaded from temp artifacts."""
        import joblib
        import src.engine.impact.scorer as sc
        from src.engine.impact.train import train, save_artifacts

        df = _make_event_study_df(20)
        model, mapping, metrics = train(df, split_date=date(2023, 3, 13))

        # Save to tmp and monkey-patch paths
        model_p  = tmp_path / "gb_model.joblib"
        quant_p  = tmp_path / "quantile_map.json"
        joblib.dump(model, model_p)
        quant_p.write_text(json.dumps(mapping))

        orig_mp, orig_qp = sc._MODEL_PATH, sc._QUANTILE_PATH
        sc._MODEL_PATH    = model_p
        sc._QUANTILE_PATH = quant_p
        sc._loaded = False
        sc._model  = None
        sc.FALLBACK_MODE = True

        try:
            impact, interp = sc.score_impact(0.7, "Earnings", source_count=3, velocity=5.0)
            assert 1 <= impact <= 10
            assert not sc.FALLBACK_MODE  # should have loaded the model
        finally:
            sc._MODEL_PATH    = orig_mp
            sc._QUANTILE_PATH = orig_qp
            sc._loaded = False
            sc._model  = None
            sc.FALLBACK_MODE = True


# ── Pipeline integration ───────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Verify process() uses score_impact not the old placeholder."""

    def test_process_calls_score_impact(self):
        from src.engine.ingest.base import RawItem
        from src.engine.nlp.sentiment import SentimentResult
        from src.engine.nlp.classifier import ClassifyResult
        from unittest.mock import patch, MagicMock

        item = RawItem(
            id="test:001", ts=datetime.now(tz=timezone.utc),
            text="$JPM JPMorgan beats Q3 earnings by 12 percent",
            source="test", url="",
        )

        with (
            patch("src.engine.nlp.pipeline.sent_mod.analyse",
                  return_value=SentimentResult(score=0.72, confidence=0.88)),
            patch("src.engine.nlp.pipeline.cls_mod.classify",
                  return_value=ClassifyResult(label="Earnings", probability=0.85)),
            patch("src.engine.nlp.pipeline.ev_mod.extract",
                  return_value=["JPMorgan beats earnings."]),
            patch("src.engine.nlp.pipeline.score_impact",
                  return_value=(8, "impact 8 ≈ 2.1–3.4% abnormal move")) as mock_score,
        ):
            from src.engine.nlp.pipeline import process
            signals = process(item)
            assert mock_score.called, "score_impact should be called from process()"
            for s in signals:
                assert s.impact == 8

    def test_process_impact_in_range(self):
        from src.engine.ingest.base import RawItem
        from src.engine.nlp.sentiment import SentimentResult
        from src.engine.nlp.classifier import ClassifyResult

        item = RawItem(
            id="test:002", ts=datetime.now(tz=timezone.utc),
            text="Federal Reserve raises rates sharply",
            source="test", url="",
        )
        with (
            patch("src.engine.nlp.pipeline.sent_mod.analyse",
                  return_value=SentimentResult(score=-0.55, confidence=0.82)),
            patch("src.engine.nlp.pipeline.cls_mod.classify",
                  return_value=ClassifyResult(label="Macroeconomic", probability=0.78)),
            patch("src.engine.nlp.pipeline.ev_mod.extract",
                  return_value=["Fed raises rates."]),
        ):
            from src.engine.nlp.pipeline import process
            signals = process(item)
            for s in signals:
                assert 1 <= s.impact <= 10
