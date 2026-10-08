"""Tests for ReplayStreamer — timestamp ordering, date windowing, speed, multi-source."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone, timedelta
from typing import AsyncIterator

import pytest

from src.engine.ingest.base import RawItem, Source
from src.engine.ingest.replay import ReplayStreamer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dt(day: int, hour: int = 12) -> datetime:
    """Create a UTC-aware datetime in March 2023."""
    return datetime(2023, 3, day, hour, 0, 0, tzinfo=timezone.utc)


def _item(idx: int, day: int, hour: int = 12, source: str = "test") -> RawItem:
    return RawItem(
        id=f"item-{idx:03d}",
        ts=_dt(day, hour),
        text=f"Test item {idx} on day {day} hour {hour}",
        source=source,
        url="",
    )


class _ListSource(Source):
    """Simple in-memory Source for testing."""

    def __init__(self, items: list[RawItem]) -> None:
        self._items = items

    async def items(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        for item in self._items:
            yield item


async def _drain(streamer: ReplayStreamer) -> list[RawItem]:
    """Run the streamer and collect all items from its queue."""
    received: list[RawItem] = []

    async def consumer() -> None:
        while True:
            item: RawItem = await streamer.queue.get()
            received.append(item)
            streamer.queue.task_done()

    # Run producer and consumer concurrently; producer finishes, then cancel consumer
    prod = asyncio.create_task(streamer.run())
    cons = asyncio.create_task(consumer())
    await prod
    await asyncio.sleep(0.05)   # let consumer drain remaining items
    cons.cancel()
    try:
        await cons
    except asyncio.CancelledError:
        pass
    return received


# ── Ordering tests ────────────────────────────────────────────────────────────

class TestReplayOrdering:
    @pytest.mark.asyncio
    async def test_items_emitted_in_timestamp_order(self) -> None:
        """Items fed in reverse order must come out in ascending timestamp order."""
        items_reversed = [
            _item(5, 10, 15),
            _item(4, 10, 8),
            _item(3, 10, 3),
            _item(2, 9, 22),
            _item(1, 9, 6),
        ]
        source = _ListSource(items_reversed)
        streamer = ReplayStreamer(source, start=date(2023, 3, 9), end=date(2023, 3, 10), speed=1_000_000)
        received = await _drain(streamer)

        assert len(received) == 5
        timestamps = [r.ts for r in received]
        assert timestamps == sorted(timestamps), "Items must be in ascending timestamp order"

    @pytest.mark.asyncio
    async def test_deterministic_tie_breaking(self) -> None:
        """Items with identical timestamps must always come out in the same order (by id)."""
        same_ts = [
            RawItem(id="z-last", ts=_dt(10, 9), text="z", source="test", url=""),
            RawItem(id="a-first", ts=_dt(10, 9), text="a", source="test", url=""),
            RawItem(id="m-middle", ts=_dt(10, 9), text="m", source="test", url=""),
        ]
        source = _ListSource(same_ts)
        streamer = ReplayStreamer(source, start=date(2023, 3, 10), end=date(2023, 3, 10), speed=1_000_000)
        received = await _drain(streamer)

        assert len(received) == 3
        ids = [r.id for r in received]
        assert ids == sorted(ids), "Tie-breaking must be by id (alphabetical)"

    @pytest.mark.asyncio
    async def test_empty_source(self) -> None:
        source = _ListSource([])
        streamer = ReplayStreamer(source, start=date(2023, 3, 8), end=date(2023, 3, 10), speed=1_000_000)
        received = await _drain(streamer)
        assert received == []

    @pytest.mark.asyncio
    async def test_single_item(self) -> None:
        source = _ListSource([_item(1, 9, 10)])
        streamer = ReplayStreamer(source, start=date(2023, 3, 8), end=date(2023, 3, 10), speed=1_000_000)
        received = await _drain(streamer)
        assert len(received) == 1


# ── Date windowing tests ──────────────────────────────────────────────────────

class TestDateWindowing:
    @pytest.mark.asyncio
    async def test_items_outside_window_excluded(self) -> None:
        items = [
            _item(1, 7, 12),    # Mar 7 — before window
            _item(2, 8, 12),    # Mar 8 — in window (start)
            _item(3, 10, 12),   # Mar 10 — in window
            _item(4, 17, 12),   # Mar 17 — in window (end)
            _item(5, 18, 12),   # Mar 18 — after window
        ]
        source = _ListSource(items)
        streamer = ReplayStreamer(
            source,
            start=date(2023, 3, 8),
            end=date(2023, 3, 17),
            speed=1_000_000,
        )
        received = await _drain(streamer)
        assert len(received) == 3
        assert all(date(2023, 3, 8) <= r.ts.date() <= date(2023, 3, 17) for r in received)

    @pytest.mark.asyncio
    async def test_single_day_window(self) -> None:
        items = [_item(i, 10, i) for i in range(5)]
        source = _ListSource(items)
        streamer = ReplayStreamer(
            source,
            start=date(2023, 3, 10),
            end=date(2023, 3, 10),
            speed=1_000_000,
        )
        received = await _drain(streamer)
        assert all(r.ts.date() == date(2023, 3, 10) for r in received)

    @pytest.mark.asyncio
    async def test_datetime_window_precision(self) -> None:
        """Passing datetime instead of date should honour precise boundaries."""
        items = [
            RawItem(id="early", ts=_dt(10, 7), text="early", source="test", url=""),
            RawItem(id="in",    ts=_dt(10, 9), text="in",    source="test", url=""),
            RawItem(id="late",  ts=_dt(10, 18), text="late", source="test", url=""),
        ]
        source = _ListSource(items)
        streamer = ReplayStreamer(
            source,
            start=datetime(2023, 3, 10, 8, tzinfo=timezone.utc),
            end=datetime(2023, 3, 10, 17, tzinfo=timezone.utc),
            speed=1_000_000,
        )
        received = await _drain(streamer)
        assert len(received) == 1
        assert received[0].id == "in"


# ── Multi-source merge tests ──────────────────────────────────────────────────

class TestMultiSource:
    @pytest.mark.asyncio
    async def test_two_sources_merged_in_order(self) -> None:
        source_a = _ListSource([_item(1, 8, 6, "news"), _item(3, 9, 14, "news")])
        source_b = _ListSource([_item(2, 8, 10, "tweet"), _item(4, 10, 8, "tweet")])

        streamer = ReplayStreamer(
            [source_a, source_b],
            start=date(2023, 3, 8),
            end=date(2023, 3, 10),
            speed=1_000_000,
        )
        received = await _drain(streamer)
        assert len(received) == 4
        timestamps = [r.ts for r in received]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_both_sources_represented(self) -> None:
        source_a = _ListSource([_item(1, 8, 9, "news")])
        source_b = _ListSource([_item(2, 8, 11, "tweet")])
        streamer = ReplayStreamer(
            [source_a, source_b],
            start=date(2023, 3, 8),
            end=date(2023, 3, 8),
            speed=1_000_000,
        )
        received = await _drain(streamer)
        sources = {r.source for r in received}
        assert "news" in sources
        assert "tweet" in sources


# ── Speed multiplier tests (timing) ──────────────────────────────────────────

class TestSpeedMultiplier:
    @pytest.mark.asyncio
    async def test_high_speed_completes_quickly(self) -> None:
        """60 minutes of event-time at 1_000_000× should finish in < 0.1s."""
        items = [
            _item(1, 8, 0),   # 08:00
            _item(2, 8, 1),   # 09:00  (1 hr later in event-time)
        ]
        source = _ListSource(items)
        streamer = ReplayStreamer(
            source, start=date(2023, 3, 8), end=date(2023, 3, 8), speed=1_000_000
        )
        import time  # noqa: PLC0415
        t0 = time.monotonic()
        received = await _drain(streamer)
        elapsed = time.monotonic() - t0

        assert len(received) == 2
        assert elapsed < 0.5, f"1_000_000× replay took too long: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_queue_receives_items(self) -> None:
        """Verify items land on queue, not just in internal list."""
        source = _ListSource([_item(1, 8, 12)])
        q: asyncio.Queue[RawItem] = asyncio.Queue()
        streamer = ReplayStreamer(
            source, start=date(2023, 3, 8), end=date(2023, 3, 8),
            speed=1_000_000, queue=q,
        )
        await streamer.run()
        assert not q.empty()
        item = q.get_nowait()
        assert item.id == "item-001"


# ── Real sample data integration test ─────────────────────────────────────────

class TestWithSampleData:
    @pytest.mark.asyncio
    async def test_news_csv_replay(self) -> None:
        """Replay the committed SVB news sample; verify ordering and count."""
        from pathlib import Path  # noqa: PLC0415
        from src.engine.ingest.news import NewsSource  # noqa: PLC0415

        sample = Path("data/svb_news_sample.csv")
        if not sample.exists():
            pytest.skip("svb_news_sample.csv not found")

        source = NewsSource(sample, source_name="svb_news_test")
        streamer = ReplayStreamer(
            source,
            start=date(2023, 3, 8),
            end=date(2023, 3, 17),
            speed=1_000_000,
        )
        received = await _drain(streamer)

        assert len(received) >= 10, "Expected at least 10 news items in SVB window"
        timestamps = [r.ts for r in received]
        assert timestamps == sorted(timestamps), "News items must be timestamp-ordered"

    @pytest.mark.asyncio
    async def test_tweets_csv_replay(self) -> None:
        """Replay the committed SVB tweets sample; verify ordering."""
        from pathlib import Path  # noqa: PLC0415
        from src.engine.ingest.social import SocialSource  # noqa: PLC0415

        sample = Path("data/svb_tweets_sample.csv")
        if not sample.exists():
            pytest.skip("svb_tweets_sample.csv not found")

        source = SocialSource(sample)
        streamer = ReplayStreamer(
            source,
            start=date(2023, 3, 8),
            end=date(2023, 3, 17),
            speed=1_000_000,
        )
        received = await _drain(streamer)

        assert len(received) >= 5
        timestamps = [r.ts for r in received]
        assert timestamps == sorted(timestamps)
