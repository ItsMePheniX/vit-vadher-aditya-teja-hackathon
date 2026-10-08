"""SocialSource: reads financial tweets from a local Kaggle-format CSV.

Supported column aliases (auto-detected):
    date   ← post_date, created_at, date, timestamp, Date
    text   ← body, tweet, text, content, Body, Tweet
    ticker ← ticker, ticker_symbol, Ticker, symbol, stock
    writer ← writer, user, author, screen_name

Dataset assumptions:
  - Kaggle "Stock Market Tweet" dataset (tweet_id, writer, post_date, body, ticker)
  - Kaggle "Twitter Financial News Sentiment" (tweet, label, ticker)
  - Any CSV where a text column + date column are present

See data/DATA_SOURCES.md for provenance.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiofiles

from src.engine.ingest.base import RawItem, Source
from src.engine.ingest.news import _parse_ts  # reuse the flexible date parser

logger = logging.getLogger(__name__)

_DATE_COLS   = {"post_date", "created_at", "date", "timestamp", "Date", "publish_date"}
_TEXT_COLS   = {"body", "tweet", "text", "content", "Body", "Tweet", "tweet_text"}
_TICKER_COLS = {"ticker", "ticker_symbol", "Ticker", "symbol", "stock", "cashtag"}
_WRITER_COLS = {"writer", "user", "author", "screen_name", "username", "Writer"}


def _pick(row: dict, candidates: set[str], default: str = "") -> str:
    for key in candidates:
        if key in row and row[key]:
            return str(row[key]).strip()
    return default


def _tweet_id(idx: int, text: str) -> str:
    digest = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"tweet:{idx:06d}:{digest}"


class SocialSource(Source):
    """Read financial tweets from a Kaggle stock-tweets CSV.

    Args:
        path:        Path to the tweets CSV file.
        source_name: Feed label (defaults to 'kaggle_tweets').
        encoding:    File encoding (default utf-8).
    """

    def __init__(
        self,
        path: str | Path,
        source_name: str = "kaggle_tweets",
        encoding: str = "utf-8",
    ) -> None:
        self._path = Path(path)
        self._source = source_name
        self._encoding = encoding

    async def items(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        logger.info("SocialSource: reading %s", self._path)
        async with aiofiles.open(self._path, encoding=self._encoding) as fh:
            content = await fh.read()

        reader = csv.DictReader(content.splitlines())
        for idx, row in enumerate(reader):
            row = dict(row)

            raw_date = _pick(row, _DATE_COLS)
            if not raw_date:
                continue
            try:
                ts = _parse_ts(raw_date)
            except ValueError as e:
                logger.debug("Skipping tweet %d: %s", idx, e)
                continue

            text = _pick(row, _TEXT_COLS)
            if not text:
                continue

            ticker = _pick(row, _TICKER_COLS)
            writer = _pick(row, _WRITER_COLS)

            # Prefix ticker into text so the entity linker can pick it up
            # e.g.  "$JPM …tweet text…"  →  ticker hint
            if ticker and not text.startswith(f"${ticker}"):
                text = f"${ticker} {text}"

            # Build a profile-style URL if no real URL is available
            url = f"https://twitter.com/{writer}" if writer else ""

            yield RawItem(
                id=_tweet_id(idx, text),
                ts=ts,
                text=text,
                source=self._source,
                url=url,
            )

    def __repr__(self) -> str:
        return f"<SocialSource path={self._path}>"
