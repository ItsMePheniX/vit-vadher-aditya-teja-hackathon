"""Entity linker: maps company names / aliases from free text to ticker symbols.

Algorithm
---------
1. Build an alias → ticker reverse map from tickers.yaml (all lower-cased).
2. For each candidate span in text (cashtag → entity name extraction):
   a. Check for "$TICKER" patterns first (highest confidence, 0.99).
   b. Try full phrase match in the reverse map.
   c. For aliases marked ``ambiguous=true``, verify that at least one financial
      context word appears within a ±50-token window of the mention.
3. Return a list of (ticker, confidence) pairs, de-duplicated by ticker.

Confidence levels:
    0.99  exact cashtag match   e.g. "$JPM"
    0.90  exact alias match     (unambiguous ticker)
    0.75  exact alias match     (ambiguous ticker, context present)
    0.00  alias found but no context (ambiguous) → not returned

Context words (financial domain — see docs/ASSUMPTIONS.md §Entity Linker):
    stock, share, equity, market, price, revenue, earnings, dividend,
    bank, financial, fund, asset, portfolio, trade, invest, quarter,
    CEO, CFO, management, board, analyst, rating, downgrade, upgrade

Usage::

    linker = EntityLinker()
    results = linker.link("JPMorgan beat earnings estimates; Apple stock fell")
    # → [LinkResult(ticker='JPM', confidence=0.90),
    #    LinkResult(ticker='AAPL', confidence=0.75)]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import yaml

_TICKERS_PATH = Path(__file__).parent / "tickers.yaml"

# Financial context words used to disambiguate ambiguous aliases
_CONTEXT_WORDS: frozenset[str] = frozenset({
    "stock", "share", "shares", "equity", "market", "price", "prices",
    "revenue", "earnings", "earning", "dividend", "dividends",
    "bank", "banking", "banks", "financial", "finance", "fund", "funds",
    "asset", "assets", "portfolio", "trade", "trades", "invest", "investment",
    "quarter", "quarterly", "ceo", "cfo", "cto", "coo", "management",
    "board", "directors", "analyst", "analysts", "rating", "ratings",
    "downgrade", "upgrade", "ipo", "merger", "acquisition", "buyout",
    "debt", "credit", "loan", "bonds", "yield", "inflation", "fed",
    "nasdaq", "nyse", "s&p", "index", "etf", "hedge", "volatility",
    "sec", "regulator", "regulatory", "fine", "penalty", "lawsuit",
})

_CONTEXT_WINDOW = 100  # characters on each side of the match to search for context


@dataclass(frozen=True)
class TickerEntry:
    ticker: str
    name: str
    sector: str
    aliases: list[str]
    ambiguous: bool


class LinkResult(NamedTuple):
    ticker: str
    confidence: float
    matched_text: str


@lru_cache(maxsize=1)
def _load_entries() -> list[TickerEntry]:
    with _TICKERS_PATH.open() as fh:
        data = yaml.safe_load(fh)
    return [
        TickerEntry(
            ticker=t["ticker"],
            name=t["name"],
            sector=t["sector"],
            aliases=t.get("aliases", []),
            ambiguous=t.get("ambiguous", False),
        )
        for t in data["tickers"]
    ]


def _build_maps() -> tuple[
    dict[str, tuple[str, bool]],     # alias_lower → (ticker, ambiguous)
    dict[str, TickerEntry],          # ticker → entry
]:
    """Build lookup maps from loaded tickers.yaml."""
    entries = _load_entries()
    alias_map: dict[str, tuple[str, bool]] = {}
    ticker_map: dict[str, TickerEntry] = {}

    for entry in entries:
        ticker_map[entry.ticker.upper()] = entry
        # Ticker itself as alias — inherits the entry's ambiguous flag only when
        # the ticker text itself is a common English word (e.g. "V" for Visa is fine,
        # but lowercase "meta" could match META ambiguously)
        ticker_lower = entry.ticker.lower().replace("-", "")
        alias_map[ticker_lower] = (entry.ticker, entry.ambiguous)
        for alias in entry.aliases:
            key = alias.lower().strip()
            if key not in alias_map:  # first definition wins
                alias_map[key] = (entry.ticker, entry.ambiguous)

    return alias_map, ticker_map


# Module-level caches (populated on first use)
_alias_map: dict[str, tuple[str, bool]] = {}
_ticker_map: dict[str, TickerEntry] = {}


def _ensure_loaded() -> None:
    global _alias_map, _ticker_map
    if not _alias_map:
        _alias_map, _ticker_map = _build_maps()


def _has_financial_context(text: str, match_start: int, match_end: int) -> bool:
    """Check for a financial context word near the match position."""
    lo = max(0, match_start - _CONTEXT_WINDOW)
    hi = min(len(text), match_end + _CONTEXT_WINDOW)
    window = text[lo:hi].lower()
    return any(word in window for word in _CONTEXT_WORDS)


def _cashtag_results(text: str) -> list[LinkResult]:
    """Extract "$TICKER" cashtag mentions (confidence 0.99)."""
    _ensure_loaded()
    results: list[LinkResult] = []
    for m in re.finditer(r"\$([A-Z]{1,5}(?:-[A-B])?)\b", text):
        raw = m.group(1).upper()
        if raw in _ticker_map:
            results.append(LinkResult(ticker=raw, confidence=0.99, matched_text=m.group(0)))
    return results


def _alias_results(text: str) -> list[LinkResult]:
    """Match known aliases in text; apply context check for ambiguous aliases."""
    _ensure_loaded()
    results: list[LinkResult] = []

    # Sort aliases longest-first to prefer "Goldman Sachs" over "Goldman"
    sorted_aliases = sorted(_alias_map.keys(), key=len, reverse=True)

    text_lower = text.lower()
    for alias in sorted_aliases:
        ticker, ambiguous = _alias_map[alias]
        # Word-boundary match (handle multi-word aliases with spaces)
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        for m in re.finditer(pattern, text_lower):
            if ambiguous:
                if not _has_financial_context(text_lower, m.start(), m.end()):
                    continue
                conf = 0.75
            else:
                conf = 0.90
            results.append(
                LinkResult(ticker=ticker, confidence=conf, matched_text=text[m.start():m.end()])
            )

    return results


class EntityLinker:
    """Link entity mentions in text to canonical ticker symbols.

    The linker is stateless and thread-safe; instantiate once and reuse.
    """

    def link(self, text: str) -> list[LinkResult]:
        """Return de-duplicated (ticker, confidence) pairs for all mentions.

        De-duplication: if the same ticker is found multiple times, keep the
        highest-confidence result.

        Args:
            text: Free text (news headline, tweet, etc.)

        Returns:
            List of :class:`LinkResult`, sorted by confidence descending.
        """
        candidates = _cashtag_results(text) + _alias_results(text)

        # De-duplicate: keep max confidence per ticker
        best: dict[str, LinkResult] = {}
        for r in candidates:
            if r.ticker not in best or r.confidence > best[r.ticker].confidence:
                best[r.ticker] = r

        return sorted(best.values(), key=lambda r: r.confidence, reverse=True)

    def primary(self, text: str) -> str | None:
        """Return the ticker of the highest-confidence mention, or None."""
        results = self.link(text)
        return results[0].ticker if results else None

    def get_entry(self, ticker: str) -> TickerEntry | None:
        """Return the full TickerEntry for a ticker, or None."""
        _ensure_loaded()
        return _ticker_map.get(ticker.upper())
