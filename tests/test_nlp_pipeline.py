"""Tests for the NLP pipeline — 10 hand-written financial headlines.

Test structure
--------------
All tests mock the three NLP primitives (sentiment, classifier, evidence)
so they run without downloading any models.  A separate "integration" section
marks tests that call real models; these are skipped when `transformers` is
not installed.

Mock fixtures
-------------
  mock_sentiment   : analyse() returns (score, confidence) controlled per test
  mock_classify    : classify() returns (label, probability)
  mock_evidence    : extract() returns a fixed list of sentences

The 10 headlines cover all 8 event types plus edge cases (multi-entity,
no-entity, cashtag extraction).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.engine.ingest.base import RawItem
from src.engine.nlp.sentiment import SentimentResult
from src.engine.nlp.classifier import ClassifyResult
from src.engine.schema import Signal


# ── 10 hand-crafted headlines ─────────────────────────────────────────────────
# (text, entity_hint_in_text, expected_event_type, sentiment_direction)

HEADLINES: list[tuple[str, str | None, str, str]] = [
    (
        "Federal Reserve raises interest rates by 50 basis points amid persistent inflation",
        None, "Macroeconomic", "negative",
    ),
    (
        "$AAPL Apple beats Q3 earnings by 12%, stock surges 5% after hours",
        "AAPL", "Earnings", "positive",
    ),
    (
        "JPMorgan Chase acquires First Republic Bank in FDIC-assisted deal",
        "JPM", "Merger/Acquisition", "positive",
    ),
    (
        "Moody's downgrades SVB Financial to junk, triggering credit event concerns across sector",
        None, "Credit Event", "negative",
    ),
    (
        "$GS Goldman Sachs faces SEC investigation over alleged regulatory violations",
        "GS", "Regulatory", "negative",
    ),
    (
        "US-China trade war escalates with sweeping new tariffs on semiconductor chip imports",
        None, "Geopolitical", "negative",
    ),
    (
        "Microsoft launches new AI Copilot product integrated across entire Office 365 suite",
        "MSFT", "Product Launch", "positive",
    ),
    (
        "Bank of America reports record net income of $8.1 billion in Q2 2023, beating estimates",
        "BAC", "Earnings", "positive",
    ),
    (
        "$WFC Wells Fargo stock fell 10% as credit losses mounted sharply in commercial real estate",
        "WFC", "Credit Event", "negative",
    ),
    (
        "Citigroup CEO Jane Fraser announces major restructuring, cutting 20000 jobs globally",
        "C", "Other", "negative",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_item(text: str, source: str = "test") -> RawItem:
    return RawItem(
        id="test:000001",
        ts=datetime.now(tz=timezone.utc),
        text=text,
        source=source,
        url="https://example.com",
    )


def _mock_sentiment(score: float = -0.7, conf: float = 0.88) -> SentimentResult:
    return SentimentResult(score=score, confidence=conf)


def _mock_classify(label: str = "Macroeconomic", prob: float = 0.82) -> ClassifyResult:
    return ClassifyResult(label=label, probability=prob)  # type: ignore[arg-type]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def patch_nlp() -> Generator[dict, None, None]:
    """Patch all three NLP primitives so tests run without models."""
    sentiment_mock = MagicMock(return_value=SentimentResult(score=-0.65, confidence=0.91))
    classify_mock  = MagicMock(return_value=ClassifyResult(label="Macroeconomic", probability=0.84))  # type: ignore[arg-type]
    evidence_mock  = MagicMock(return_value=["Interest rates raised.", "Inflation persists."])

    with (
        patch("src.engine.nlp.pipeline.sent_mod.analyse", sentiment_mock),
        patch("src.engine.nlp.pipeline.cls_mod.classify",  classify_mock),
        patch("src.engine.nlp.pipeline.ev_mod.extract",    evidence_mock),
    ):
        yield {
            "sentiment": sentiment_mock,
            "classify":  classify_mock,
            "evidence":  evidence_mock,
        }


# ── Core pipeline tests (mocked) ──────────────────────────────────────────────

class TestPipelineStructure:
    """Verify process() produces well-formed Signal objects."""

    def test_returns_list(self, patch_nlp: dict) -> None:
        item = _make_item("Federal Reserve raises rates 50bps")
        from src.engine.nlp.pipeline import process
        result = process(item)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_signals_are_valid_schema(self, patch_nlp: dict) -> None:
        item = _make_item("Federal Reserve raises rates 50bps")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        for s in signals:
            assert isinstance(s, Signal)
            assert -1.0 <= s.sentiment <= 1.0
            assert 1 <= s.impact <= 10
            assert 0.0 <= s.confidence <= 1.0
            assert s.half_life_hours > 0
            assert isinstance(s.evidence, list)
            assert isinstance(s.sources, list)

    def test_signal_has_ulid_id(self, patch_nlp: dict) -> None:
        item = _make_item("Federal Reserve raises rates 50bps")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        for s in signals:
            assert len(s.id) == 26  # ULID string length

    def test_source_preserved(self, patch_nlp: dict) -> None:
        item = _make_item("Goldman Sachs stock fell", source="reuters")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        for s in signals:
            assert "reuters" in s.sources

    def test_ts_is_utc_aware(self, patch_nlp: dict) -> None:
        item = _make_item("JPMorgan beats earnings")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        for s in signals:
            assert s.ts.tzinfo is not None


class TestEntityExtraction:
    """Verify per-entity signal generation."""

    def test_cashtag_gives_correct_entity(self, patch_nlp: dict) -> None:
        item = _make_item("$JPM JPMorgan reports record profits")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        tickers = {s.entity for s in signals}
        assert "JPM" in tickers

    def test_no_entity_gives_market(self, patch_nlp: dict) -> None:
        item = _make_item("Global markets fell sharply today")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        # All signals should be MARKET or a linked entity; at least one MARKET
        assert any(s.entity == "MARKET" for s in signals)

    def test_multiple_entities_multiple_signals(self, patch_nlp: dict) -> None:
        item = _make_item("$JPM JPMorgan and Bank of America both fell on contagion fears")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        tickers = {s.entity for s in signals}
        assert "JPM" in tickers
        assert "BAC" in tickers

    def test_cashtag_wfc(self, patch_nlp: dict) -> None:
        item = _make_item("$WFC Wells Fargo stock fell 10% as credit losses mounted")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        assert any(s.entity == "WFC" for s in signals)


class TestImpactFormula:
    """Verify impact scoring: fallback rule-based formula and clamping."""

    @pytest.mark.parametrize("score,event_type,min_expected,max_expected", [
        (-1.0, "Geopolitical",  9, 10),   # max negative + weight 1.4 → 10
        (-0.5, "Macroeconomic", 4,  7),   # moderate negative
        ( 0.0, "Other",         1,  1),   # zero → floor 1
        ( 0.3, "Earnings",      2,  5),   # mild positive
        ( 1.0, "Earnings",      9, 10),   # max positive
    ])
    def test_impact_values(self, score, event_type, min_expected, max_expected):
        """Fallback scorer should produce impact within expected range."""
        import src.engine.impact.scorer as sc
        # Force fallback mode for deterministic testing
        orig_loaded = sc._loaded
        orig_model  = sc._model
        orig_mode   = sc.FALLBACK_MODE
        sc._loaded = True
        sc._model  = None
        sc.FALLBACK_MODE = True
        try:
            impact, _ = sc.score_impact(score, event_type)
            assert min_expected <= impact <= max_expected, (
                f"score={score}, event={event_type}: expected [{min_expected},{max_expected}], got {impact}"
            )
        finally:
            sc._loaded = orig_loaded
            sc._model  = orig_model
            sc.FALLBACK_MODE = orig_mode

    def test_impact_clamped_to_1_min(self, patch_nlp: dict) -> None:
        patch_nlp["sentiment"].return_value = SentimentResult(score=0.0, confidence=0.5)
        item = _make_item("Markets were flat and uneventful today")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        for s in signals:
            assert s.impact >= 1

    def test_impact_clamped_to_10_max(self, patch_nlp: dict) -> None:
        patch_nlp["sentiment"].return_value = SentimentResult(score=-1.0, confidence=1.0)
        item = _make_item("Catastrophic bank failure shocks markets")
        from src.engine.nlp.pipeline import process
        signals = process(item)
        for s in signals:
            assert s.impact <= 10

    def test_score_impact_returns_tuple(self) -> None:
        from src.engine.impact import score_impact
        result = score_impact(-0.8, "Credit Event")
        assert isinstance(result, tuple) and len(result) == 2
        impact, interp = result
        assert 1 <= impact <= 10
        assert isinstance(interp, str) and len(interp) > 0


class TestHalfLife:
    """Verify per-event-type half_life from config."""

    @pytest.mark.parametrize("event_type,min_hours,max_hours", [
        ("Earnings",           1,   24),    # short: ~12 h
        ("Geopolitical",      24,  200),    # long:  ~72 h
        ("Merger/Acquisition", 48,  500),    # longest: ~168 h
        ("Credit Event",       24,  200),    # long:  ~96 h
        ("Macroeconomic",      12,  100),    # medium: ~48 h
    ])
    def test_half_life_range(
        self, patch_nlp: dict, event_type: str, min_hours: float, max_hours: float
    ) -> None:
        from src.engine.nlp.pipeline import _half_life
        hl = _half_life(event_type)
        assert min_hours <= hl <= max_hours, f"{event_type}: {hl}h not in [{min_hours},{max_hours}]"

    def test_earnings_shorter_than_geopolitical(self) -> None:
        from src.engine.nlp.pipeline import _half_life
        assert _half_life("Earnings") < _half_life("Geopolitical")

    def test_all_event_types_have_half_life(self) -> None:
        from src.engine.nlp.pipeline import _half_life
        from src.engine.schema import EventType
        import typing
        for label in typing.get_args(EventType):
            hl = _half_life(label)
            assert hl > 0, f"{label} has no half_life"


class TestJSONLWriter:
    """Verify write_signals persists correctly."""

    def test_write_creates_file(self, tmp_path: Path, patch_nlp: dict) -> None:
        from src.engine.nlp.pipeline import process, write_signals
        item = _make_item("Goldman Sachs reports record Q2 revenue")
        signals = process(item)
        out = tmp_path / "test_signals.jsonl"
        write_signals(signals, path=out)
        assert out.exists()

    def test_write_valid_jsonl(self, tmp_path: Path, patch_nlp: dict) -> None:
        from src.engine.nlp.pipeline import process, write_signals
        item = _make_item("Goldman Sachs reports record Q2 revenue")
        signals = process(item)
        out = tmp_path / "test_signals.jsonl"
        write_signals(signals, path=out)
        lines = out.read_text().splitlines()
        assert len(lines) == len(signals)
        for line in lines:
            parsed = json.loads(line)
            assert "id" in parsed
            assert "sentiment" in parsed
            assert "entity" in parsed

    def test_write_appends(self, tmp_path: Path, patch_nlp: dict) -> None:
        from src.engine.nlp.pipeline import process, write_signals
        out = tmp_path / "append_test.jsonl"
        for text in ["Goldman Sachs earnings beat", "JPMorgan raises dividend"]:
            signals = process(_make_item(text))
            write_signals(signals, path=out)
        lines = out.read_text().splitlines()
        assert len(lines) >= 2

    def test_write_thread_safe(self, tmp_path: Path, patch_nlp: dict) -> None:
        """Concurrent writes should not corrupt the JSONL file."""
        from src.engine.nlp.pipeline import process, write_signals
        out = tmp_path / "concurrent.jsonl"
        errors: list[Exception] = []

        def worker(text: str) -> None:
            try:
                signals = process(_make_item(text))
                write_signals(signals, path=out)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"Headline {i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        lines = out.read_text().splitlines()
        for line in lines:
            json.loads(line)  # each line must be valid JSON


# ── 10 Headlines parametrized test ───────────────────────────────────────────

class TestTenHeadlines:
    """Run all 10 hand-crafted headlines through the mocked pipeline."""

    @pytest.mark.parametrize("text,entity_hint,event_type,direction", HEADLINES)
    def test_headline_produces_signals(
        self,
        patch_nlp: dict,
        text: str,
        entity_hint: str | None,
        event_type: str,
        direction: str,
    ) -> None:
        """Every headline must produce at least one valid Signal."""
        item = _make_item(text)
        from src.engine.nlp.pipeline import process
        signals = process(item)
        assert len(signals) >= 1
        for s in signals:
            assert isinstance(s, Signal)
            assert 1 <= s.impact <= 10
            assert 0.0 <= s.confidence <= 1.0

    @pytest.mark.parametrize("text,entity_hint,event_type,direction", HEADLINES)
    def test_headline_sentiment_direction(
        self,
        text: str,
        entity_hint: str | None,
        event_type: str,
        direction: str,
    ) -> None:
        """Sentiment direction should match expected for each headline (real mocked value)."""
        score = -0.7 if direction == "negative" else 0.7
        conf  = 0.88

        with (
            patch("src.engine.nlp.pipeline.sent_mod.analyse",
                  return_value=SentimentResult(score=score, confidence=conf)),
            patch("src.engine.nlp.pipeline.cls_mod.classify",
                  return_value=ClassifyResult(label=event_type, probability=0.85)),  # type: ignore[arg-type]
            patch("src.engine.nlp.pipeline.ev_mod.extract",
                  return_value=["Evidence sentence."]),
        ):
            from src.engine.nlp.pipeline import process
            signals = process(_make_item(text))
            for s in signals:
                if direction == "negative":
                    assert s.sentiment < 0, f"Expected negative sentiment for: {text}"
                else:
                    assert s.sentiment > 0, f"Expected positive sentiment for: {text}"

    @pytest.mark.parametrize("text,entity_hint,event_type,direction", HEADLINES)
    def test_headline_entity_extracted(
        self,
        patch_nlp: dict,
        text: str,
        entity_hint: str | None,
        event_type: str,
        direction: str,
    ) -> None:
        """Headlines with a known entity hint should extract that entity."""
        if entity_hint is None:
            pytest.skip("No entity hint for this headline")
        item = _make_item(text)
        from src.engine.nlp.pipeline import process
        signals = process(item)
        tickers = {s.entity for s in signals}
        assert entity_hint in tickers, (
            f"Expected {entity_hint!r} in {tickers} for: {text}"
        )


# ── Integration tests (require transformers) ──────────────────────────────────

class TestIntegration:
    """Real model tests — skipped when transformers not installed."""

    @pytest.fixture(autouse=True)
    def _require_transformers(self) -> None:
        pytest.importorskip("transformers", reason="transformers not installed")

    def test_finbert_score_in_range(self) -> None:
        from src.engine.nlp.sentiment import analyse
        result = analyse("Apple reported record earnings, beating all analyst estimates")
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_finbert_negative_headline(self) -> None:
        from src.engine.nlp.sentiment import analyse
        result = analyse("Massive losses trigger bankruptcy fears as credit spreads widen")
        assert result.score < 0, "Expected negative score for a clearly negative headline"

    def test_finbert_positive_headline(self) -> None:
        from src.engine.nlp.sentiment import analyse
        result = analyse("Record profits and strong guidance lifted shares to all-time high")
        assert result.score > 0, "Expected positive score for clearly positive headline"

    def test_finbert_cache_hit(self) -> None:
        from src.engine.nlp.sentiment import analyse, cache_info
        text = "A unique test headline for cache validation xyzzy123"
        analyse(text)
        before = cache_info()["size"]
        analyse(text)   # second call should hit cache
        after = cache_info()["size"]
        assert before == after, "Second call should be a cache hit (no size increase)"

    def test_full_pipeline_real_models(self) -> None:
        from src.engine.nlp.pipeline import process
        item = _make_item(
            "$JPM JPMorgan Chase beats Q3 earnings, raises dividend by 5%",
            source="integration_test",
        )
        signals = process(item)
        assert len(signals) >= 1
        s = signals[0]
        assert s.entity in {"JPM", "MARKET"}
        assert 1 <= s.impact <= 10
        assert -1.0 <= s.sentiment <= 1.0
