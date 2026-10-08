"""Async ingestor: feeds raw news into the NLP pipeline.

Architecture:
- asyncio.Queue (no Kafka) as specified.
- Synthetic feed when NEWS_API_KEY absent or synthetic_feed=true in config.yaml.
- Pipeline now returns list[Signal] (one per entity); all are put on the queue.
- JSONL persistence is handled by nlp.pipeline.write_signals (thread-safe).
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from src import config
from src.engine.nlp.pipeline import process as nlp_process, write_signals
from src.engine.schema import Signal

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = config.get("ingestor", "queue_maxsize", default=1000)
_POLL_INTERVAL = config.get("ingestor", "poll_interval_seconds", default=5)
_USE_SYNTHETIC = config.get("ingestor", "synthetic_feed", default=True)

# Module-level asyncio queue — shared between producer and consumer
signal_queue: asyncio.Queue[Signal] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)


# ── Synthetic feed ─────────────────────────────────────────────────────────────

_SYNTHETIC_HEADLINES: list[tuple[str, str | None]] = [
    ("Federal Reserve raises interest rates by 25bps amid persistent inflation", None),
    ("AAPL reported record quarterly earnings, beating analyst estimates", "AAPL"),
    ("Geopolitical tensions escalate as trade war tariffs extended to semiconductors", None),
    ("JPM downgraded to junk rating by Moody's following credit losses", "JPM"),
    ("MSFT acquires AI startup in $2.5B takeover deal", "MSFT"),
    ("SEC launches investigation into XOM over accounting irregularities", "XOM"),
    ("GS announces record revenue in investment banking division", "GS"),
    ("Oil prices surge 8% as OPEC cuts production amid conflict", None),
    ("Inflation CPI data shows cooling at 3.2%, below Fed forecast", None),
    ("Bankruptcy filing by regional bank triggers credit event concerns", None),
    ("Amazon launches new AI product line targeting enterprise clients", "AMZN"),
    ("EU regulators impose €4B antitrust fine on Google", "GOOGL"),
    ("US GDP growth revised downward to 1.8%, recession fears mount", None),
    ("Tesla reports earnings miss, cuts guidance on weak demand", "TSLA"),
    ("Central bank emergency rate cut signals macroeconomic concern", None),
]


@dataclass
class _RawItem:
    text: str
    source: str
    entity_hint: str | None


async def _synthetic_producer() -> None:
    """Emit a synthetic headline every poll interval."""
    logger.info("Synthetic feed started (interval=%ss)", _POLL_INTERVAL)
    while True:
        text, entity_hint = random.choice(_SYNTHETIC_HEADLINES)
        item = _RawItem(
            text=text,
            source="synthetic",
            entity_hint=entity_hint,
        )
        await _enqueue(item)
        await asyncio.sleep(_POLL_INTERVAL)


async def _newsapi_producer() -> None:
    """Real NewsAPI producer (requires NEWS_API_KEY in .env)."""
    import httpx  # noqa: PLC0415

    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        logger.warning("NEWS_API_KEY not set — falling back to synthetic feed.")
        await _synthetic_producer()
        return

    base_url = "https://newsapi.org/v2/everything"
    query = "finance OR stock OR earnings OR inflation OR central bank"
    logger.info("NewsAPI feed started.")

    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(
                    base_url,
                    params={"q": query, "language": "en", "pageSize": 10, "apiKey": api_key},
                    timeout=10,
                )
                resp.raise_for_status()
                articles = resp.json().get("articles", [])
                for art in articles:
                    text = (art.get("title") or "") + " " + (art.get("description") or "")
                    if text.strip():
                        item = _RawItem(
                            text=text.strip(),
                            source=art.get("url", "newsapi"),
                            entity_hint=None,
                        )
                        await _enqueue(item)
            except Exception as exc:  # noqa: BLE001
                logger.error("NewsAPI error: %s", exc)
            await asyncio.sleep(_POLL_INTERVAL)


async def _enqueue(item: _RawItem) -> None:
    """Run pipeline in thread-pool; put each Signal on the queue + persist."""
    try:
        signals: list[Signal] = await asyncio.to_thread(nlp_process, item)
        for signal in signals:
            await signal_queue.put(signal)
            logger.info(
                "Enqueued signal %s [%s / %s  impact=%d]",
                signal.id, signal.entity, signal.event_type, signal.impact,
            )
        # Persist synchronously in the thread pool result
        if signals:
            await asyncio.to_thread(write_signals, signals)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline error: %s", exc)


async def run() -> None:
    """Start producer. Never returns."""
    producer = _synthetic_producer() if _USE_SYNTHETIC else _newsapi_producer()
    await producer
