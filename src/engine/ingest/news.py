"""NewsSource: reads financial news from a local CSV/JSON or live GDELT feed.

Supported local formats
-----------------------
CSV (auto-detected by extension):
    Must have columns mappable to: date, title, body (optional), source, url (optional).
    Common Kaggle financial-news column names are handled automatically.

JSON / JSONL:
    Each record must be a dict with the same fields as above.

GDELT (optional, async HTTP):
    Uses the GDELT v2 DOC API (public, no auth required).
    Query: financial news for a given keyword and date range.
    Results are cached to data/gdelt_cache.jsonl to avoid re-fetching.

Column-name aliases handled transparently:
    date     ← publish_date, published_at, date, Date, timestamp
    title    ← title, headline, Title, Headline
    body     ← text, body, content, article, description, Body
    source   ← publication, source, publisher, Source
    url      ← url, link, URL, source_url

See data/DATA_SOURCES.md for dataset provenance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiofiles

from src.engine.ingest.base import RawItem, Source

logger = logging.getLogger(__name__)

# ── Column-name normalisation ──────────────────────────────────────────────────

_DATE_COLS   = {"publish_date", "published_at", "date", "Date", "timestamp", "created_at"}
_TITLE_COLS  = {"title", "headline", "Title", "Headline"}
_BODY_COLS   = {"text", "body", "content", "article", "description", "Body", "content_text"}
_SOURCE_COLS = {"publication", "source", "publisher", "Source", "feed"}
_URL_COLS    = {"url", "link", "URL", "source_url", "article_url"}


def _pick(row: dict, candidates: set[str], default: str = "") -> str:
    for key in candidates:
        if key in row and row[key]:
            return str(row[key]).strip()
    return default


def _parse_ts(raw: str) -> datetime:
    """Parse a wide variety of date string formats into a UTC-aware datetime."""
    raw = raw.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Last resort: epoch parsing
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except (ValueError, OSError):
        pass
    raise ValueError(f"Cannot parse date: {raw!r}")


def _item_id(source_name: str, idx: int, text: str) -> str:
    digest = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{source_name}:{idx:06d}:{digest}"


def _row_to_item(row: dict, idx: int, source_name: str) -> RawItem | None:
    raw_date = _pick(row, _DATE_COLS)
    if not raw_date:
        return None
    try:
        ts = _parse_ts(raw_date)
    except ValueError as e:
        logger.debug("Skipping row %d: %s", idx, e)
        return None

    title = _pick(row, _TITLE_COLS)
    body  = _pick(row, _BODY_COLS)
    text  = f"{title} {body}".strip() if body else title
    if not text:
        return None

    return RawItem(
        id=_item_id(source_name, idx, text),
        ts=ts,
        text=text,
        source=_pick(row, _SOURCE_COLS, default=source_name),
        url=_pick(row, _URL_COLS),
    )


# ── NewsSource ─────────────────────────────────────────────────────────────────

class NewsSource(Source):
    """Read financial news from a local CSV or JSON/JSONL file.

    Args:
        path:        Path to the news file (CSV or JSON/JSONL).
        source_name: Override feed label (defaults to file stem).
        encoding:    File encoding (default utf-8).
    """

    def __init__(
        self,
        path: str | Path,
        source_name: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self._path = Path(path)
        self._source = source_name or self._path.stem
        self._encoding = encoding

    async def items(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        suffix = self._path.suffix.lower()
        if suffix == ".csv":
            async for item in self._read_csv():
                yield item
        elif suffix in {".json", ".jsonl", ".ndjson"}:
            async for item in self._read_json():
                yield item
        else:
            raise ValueError(f"Unsupported file type: {self._path.suffix}")

    async def _read_csv(self) -> AsyncIterator[RawItem]:
        logger.info("NewsSource: reading CSV %s", self._path)
        async with aiofiles.open(self._path, encoding=self._encoding) as fh:
            content = await fh.read()

        reader = csv.DictReader(content.splitlines())
        for idx, row in enumerate(reader):
            item = _row_to_item(dict(row), idx, self._source)
            if item:
                yield item

    async def _read_json(self) -> AsyncIterator[RawItem]:
        logger.info("NewsSource: reading JSON %s", self._path)
        async with aiofiles.open(self._path, encoding=self._encoding) as fh:
            content = await fh.read()

        # Handle both JSON array and JSONL (one object per line)
        content = content.strip()
        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

        for idx, record in enumerate(records):
            item = _row_to_item(record, idx, self._source)
            if item:
                yield item

    def __repr__(self) -> str:
        return f"<NewsSource path={self._path}>"


# ── GDELTSource ────────────────────────────────────────────────────────────────

_GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}&mode=ArtList&maxrecords=250&format=json"
    "&startdatetime={start}&enddatetime={end}"
)
_GDELT_CACHE = Path("data/gdelt_cache.jsonl")


class GDELTSource(Source):
    """Fetch financial news headlines from the GDELT v2 DOC API (no auth needed).

    Args:
        query:      Search query (e.g. "bank earnings").
        start:      Start datetime (UTC).
        end:        End datetime (UTC).
        use_cache:  If True, skip fetch if cache file exists.
    """

    def __init__(
        self,
        query: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> None:
        self._query = query
        self._start = start
        self._end = end
        self._use_cache = use_cache

    async def items(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        import httpx  # noqa: PLC0415

        cache_key = (
            f"{self._query}_{self._start.strftime('%Y%m%d')}_{self._end.strftime('%Y%m%d')}"
        )

        if self._use_cache and _GDELT_CACHE.exists():
            logger.info("GDELTSource: using cache %s", _GDELT_CACHE)
            async for item in self._read_cache(cache_key):
                yield item
            return

        url = _GDELT_URL.format(
            query=self._query.replace(" ", "%20"),
            start=self._start.strftime("%Y%m%d%H%M%S"),
            end=self._end.strftime("%Y%m%d%H%M%S"),
        )
        logger.info("GDELTSource: fetching %s", url)

        async with httpx.AsyncClient(timeout=20) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.error("GDELT fetch failed: %s", exc)
                return

        articles = data.get("articles", [])
        _GDELT_CACHE.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(_GDELT_CACHE, "a") as fh:
            for idx, art in enumerate(articles):
                raw_ts = art.get("seendate", "")
                try:
                    ts = datetime.strptime(raw_ts, "%Y%m%dT%H%M%SZ").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    continue

                title = art.get("title", "")
                domain = art.get("domain", "gdelt")
                art_url = art.get("url", "")
                text = title

                if not text:
                    continue

                item = RawItem(
                    id=_item_id("gdelt", idx, text),
                    ts=ts,
                    text=text,
                    source=domain,
                    url=art_url,
                )
                record = {
                    "cache_key": cache_key,
                    "id": item.id,
                    "ts": item.ts.isoformat(),
                    "text": item.text,
                    "source": item.source,
                    "url": item.url,
                }
                await fh.write(json.dumps(record) + "\n")
                yield item

    async def _read_cache(self, cache_key: str) -> AsyncIterator[RawItem]:
        async with aiofiles.open(_GDELT_CACHE) as fh:
            async for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("cache_key") != cache_key:
                    continue
                ts = datetime.fromisoformat(rec["ts"])
                yield RawItem(
                    id=rec["id"],
                    ts=ts,
                    text=rec["text"],
                    source=rec["source"],
                    url=rec["url"],
                )
