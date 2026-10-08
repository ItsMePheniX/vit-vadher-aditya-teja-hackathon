"""Tests for EntityLinker — alias resolution, ambiguity, cashtags."""
from __future__ import annotations

import pytest

from src.engine.linker import EntityLinker, LinkResult


@pytest.fixture(scope="module")
def linker() -> EntityLinker:
    return EntityLinker()


# ── Cashtag matches (confidence 0.99) ─────────────────────────────────────────

class TestCashtagMatches:
    def test_simple_cashtag(self, linker: EntityLinker) -> None:
        results = linker.link("$JPM reported record profits this quarter")
        assert any(r.ticker == "JPM" and r.confidence == 0.99 for r in results)

    def test_multiple_cashtags(self, linker: EntityLinker) -> None:
        results = linker.link("$BAC and $GS both fell on contagion fears")
        tickers = {r.ticker for r in results}
        assert "BAC" in tickers
        assert "GS" in tickers

    def test_cashtag_beats_alias(self, linker: EntityLinker) -> None:
        """$JPM cashtag should win over any alias match for the same ticker."""
        results = linker.link("$JPM JPMorgan Chase earnings beat estimates")
        jpm = [r for r in results if r.ticker == "JPM"]
        assert len(jpm) == 1          # de-duplicated
        assert jpm[0].confidence == 0.99   # cashtag confidence kept

    def test_unknown_cashtag_ignored(self, linker: EntityLinker) -> None:
        results = linker.link("$SIVB collapsed on Friday")
        assert not any(r.ticker == "SIVB" for r in results)


# ── Unambiguous alias matches (confidence 0.90) ────────────────────────────────

class TestUnambiguousAliases:
    def test_full_company_name(self, linker: EntityLinker) -> None:
        r = linker.link("Goldman Sachs cut its dividend forecast")
        assert any(r.ticker == "GS" for r in r)

    def test_short_alias(self, linker: EntityLinker) -> None:
        r = linker.link("Citi downgraded by Moody's credit rating team")
        assert any(r.ticker == "C" for r in r)

    def test_alias_case_insensitive(self, linker: EntityLinker) -> None:
        r = linker.link("wells fargo issued a regulatory warning")
        assert any(r.ticker == "WFC" for r in r)

    def test_bank_of_america_alias(self, linker: EntityLinker) -> None:
        r = linker.link("BofA raised its dividend by 15 percent")
        assert any(r.ticker == "BAC" for r in r)

    def test_jpmorgan_chase_alias(self, linker: EntityLinker) -> None:
        r = linker.link("JPMorgan Chase reported Q4 earnings")
        assert any(r.ticker == "JPM" for r in r)

    def test_microsoft_unambiguous(self, linker: EntityLinker) -> None:
        r = linker.link("Microsoft revenue grew 15% year-on-year")
        assert any(r.ticker == "MSFT" for r in r)

    def test_confidence_090(self, linker: EntityLinker) -> None:
        r = linker.link("Goldman Sachs upgraded its earnings outlook")
        gs = [x for x in r if x.ticker == "GS"]
        assert gs[0].confidence == 0.90


# ── Ambiguous aliases — context check required ────────────────────────────────

class TestAmbiguousAliases:
    def test_apple_with_financial_context(self, linker: EntityLinker) -> None:
        r = linker.link("Apple stock fell 5% after earnings miss")
        assert any(r.ticker == "AAPL" for r in r), "Apple in financial context should resolve"

    def test_apple_without_context_ignored(self, linker: EntityLinker) -> None:
        r = linker.link("I ate an apple and a banana for breakfast")
        assert not any(r.ticker == "AAPL" for r in r), "Apple in food context should NOT resolve"

    def test_meta_with_financial_context(self, linker: EntityLinker) -> None:
        r = linker.link("Meta shares rallied 8% on strong ad revenue growth")
        assert any(r.ticker == "META" for r in r)

    def test_meta_without_context_ignored(self, linker: EntityLinker) -> None:
        r = linker.link("Adding meta tags to the HTML page improves SEO")
        assert not any(r.ticker == "META" for r in r)

    def test_berkshire_with_earnings_context(self, linker: EntityLinker) -> None:
        r = linker.link("Berkshire Hathaway reported earnings; shares gained")
        assert any(r.ticker == "BRK-B" for r in r)

    def test_ambiguous_confidence_075(self, linker: EntityLinker) -> None:
        r = linker.link("Apple stock surged on strong iPhone sales")
        aapl = [x for x in r if x.ticker == "AAPL"]
        assert aapl[0].confidence == 0.75


# ── Multiple entities in one text ────────────────────────────────────────────

class TestMultipleEntities:
    def test_multiple_banks_in_sentence(self, linker: EntityLinker) -> None:
        text = "JPMorgan Chase, Bank of America and Citigroup all fell on credit fears"
        results = linker.link(text)
        tickers = {r.ticker for r in results}
        assert "JPM" in tickers
        assert "BAC" in tickers
        assert "C" in tickers

    def test_deduplication(self, linker: EntityLinker) -> None:
        """Same ticker mentioned twice should appear once in results."""
        text = "Goldman Sachs GS both mentioned twice in this headline"
        results = linker.link(text)
        gs_hits = [r for r in results if r.ticker == "GS"]
        assert len(gs_hits) == 1

    def test_sorted_by_confidence(self, linker: EntityLinker) -> None:
        text = "$JPM JPMorgan and Apple stock fell"  # JPM cashtag 0.99, AAPL ambiguous 0.75
        results = linker.link(text)
        assert results[0].confidence >= results[-1].confidence


# ── primary() helper ─────────────────────────────────────────────────────────

class TestPrimaryHelper:
    def test_returns_highest_confidence_ticker(self, linker: EntityLinker) -> None:
        primary = linker.primary("$GS Goldman Sachs earnings beat")
        assert primary == "GS"

    def test_returns_none_for_unknown(self, linker: EntityLinker) -> None:
        primary = linker.primary("nothing financial here at all whatsoever")
        assert primary is None


# ── get_entry() helper ────────────────────────────────────────────────────────

class TestGetEntry:
    def test_known_ticker(self, linker: EntityLinker) -> None:
        entry = linker.get_entry("JPM")
        assert entry is not None
        assert entry.ticker == "JPM"
        assert entry.sector == "Financials"

    def test_case_insensitive(self, linker: EntityLinker) -> None:
        assert linker.get_entry("jpm") is not None

    def test_unknown_returns_none(self, linker: EntityLinker) -> None:
        assert linker.get_entry("UNKNOWN") is None
