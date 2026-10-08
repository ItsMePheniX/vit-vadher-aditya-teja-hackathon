"""ReplayStreamer: time-ordered async replay of any Source.

The ReplayStreamer:
1. Drains a Source (or multiple sources) completely into memory.
2. Filters items to [start, end] date window.
3. Sorts all items by ascending timestamp (deterministic ordering for equal ts).
4. Emits items on an asyncio.Queue<RawItem> at wall-clock delays scaled by
   (1 / speed_multiplier), so a 60× replay emits 1 hour of event-time every
   60 real seconds.

Design notes (see docs/ASSUMPTIONS.md):
- Full in-memory buffering is acceptable because sample data is ≤ a few MB.
  For production use, replace with a merge-sorted streaming approach.
- The queue is bounded by maxsize to provide backpressure if the consumer
  (NLP pipeline) is slower than the producer.
- Wall-clock start is recorded when run() is called, not at construction time.

CLI usage:
    python -m src.engine.ingest.replay \\
        --news   data/svb_news_sample.csv \\
        --tweets data/svb_tweets_sample.csv \\
        --start  2023-03-08 \\
        --end    2023-03-17 \\
        --speed  60
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timezone
from typing import Sequence

from src.engine.ingest.base import RawItem, Source

logger = logging.getLogger(__name__)


def _to_utc_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ReplayStreamer:
    """Replay events from one or more Sources in chronological order.

    Args:
        sources:          One or more Source instances to drain.
        start:            Only emit items with ts >= start (date or datetime, UTC).
        end:              Only emit items with ts <= end (date or datetime, UTC).
        speed:            Speed multiplier (default 1.0 = real-time, 60.0 = 60× faster).
        queue:            Caller-supplied asyncio.Queue; created internally if None.
        queue_maxsize:    Bound for the internal queue (default 1 000).
        log_every:        Log a progress line every N emitted items (0 = silent).
    """

    def __init__(
        self,
        sources: Source | Sequence[Source],
        start: date | datetime,
        end: date | datetime,
        speed: float = 1.0,
        queue: asyncio.Queue[RawItem] | None = None,
        queue_maxsize: int = 1_000,
        log_every: int = 10,
    ) -> None:
        self._sources: list[Source] = (
            [sources] if isinstance(sources, Source) else list(sources)  # type: ignore[arg-type]
        )
        self._start = self._coerce_dt(start, end_of_day=False)
        self._end   = self._coerce_dt(end,   end_of_day=True)
        self._speed = max(speed, 0.001)  # guard against div-by-zero
        self.queue: asyncio.Queue[RawItem] = queue or asyncio.Queue(maxsize=queue_maxsize)
        self._log_every = log_every

    @staticmethod
    def _coerce_dt(d: date | datetime, end_of_day: bool) -> datetime:
        if isinstance(d, datetime):
            return _to_utc_aware(d)
        # date only: expand to midnight or 23:59:59
        if end_of_day:
            return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    async def _collect(self) -> list[RawItem]:
        """Drain all sources, filter to date window, sort by ts."""
        all_items: list[RawItem] = []
        for source in self._sources:
            async for item in source.items():
                ts = _to_utc_aware(item.ts)
                if self._start <= ts <= self._end:
                    all_items.append(item)

        # Primary: timestamp ascending; secondary: id (deterministic tie-break)
        all_items.sort(key=lambda i: (i.ts, i.id))
        logger.info(
            "ReplayStreamer: collected %d items [%s → %s] across %d source(s)",
            len(all_items), self._start.date(), self._end.date(), len(self._sources),
        )
        return all_items

    async def run(self) -> None:
        """Start the replay; puts items on self.queue until exhausted.

        The coroutine returns when all items have been enqueued.
        Callers should await self.queue.join() if they need to wait for
        all items to be fully consumed downstream.
        """
        items = await self._collect()
        if not items:
            logger.warning("ReplayStreamer: no items in window — nothing to replay.")
            return

        first_event_ts = items[0].ts
        wall_start     = asyncio.get_event_loop().time()
        emitted        = 0

        for item in items:
            # How many real seconds should have elapsed by the time this item fires?
            event_offset_s = (item.ts - first_event_ts).total_seconds()
            target_wall_s  = event_offset_s / self._speed

            # Current wall-clock elapsed
            now_wall_s = asyncio.get_event_loop().time() - wall_start
            sleep_for  = target_wall_s - now_wall_s

            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            await self.queue.put(item)
            emitted += 1

            if self._log_every and emitted % self._log_every == 0:
                logger.info(
                    "Replay %d/%d — %s [%s] %s",
                    emitted, len(items), item.ts.isoformat(), item.source, item.text[:60],
                )

        logger.info("ReplayStreamer: finished — %d items emitted.", emitted)


# ── CLI entry-point ────────────────────────────────────────────────────────────

async def _cli_main(args: argparse.Namespace) -> None:
    """Run replay and print items to stdout (for demo / debugging)."""
    from src.engine.ingest.news   import NewsSource    # noqa: PLC0415
    from src.engine.ingest.social import SocialSource  # noqa: PLC0415

    sources: list[Source] = []
    if args.news:
        sources.append(NewsSource(args.news))
    if args.tweets:
        sources.append(SocialSource(args.tweets))

    if not sources:
        print("Error: provide at least --news or --tweets")
        return

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    streamer = ReplayStreamer(sources, start=start, end=end, speed=args.speed, log_every=1)

    # Consumer: print each item as it arrives
    async def consumer() -> None:
        received = 0
        while True:
            item: RawItem = await streamer.queue.get()
            print(f"[{item.ts.isoformat()}] [{item.source}] {item.text[:120]}")
            streamer.queue.task_done()
            received += 1

    await asyncio.gather(streamer.run(), consumer())


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Replay financial events on an asyncio queue")
    parser.add_argument("--news",   default=None, help="Path to news CSV")
    parser.add_argument("--tweets", default=None, help="Path to tweets CSV")
    parser.add_argument("--start",  required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    required=True, help="End date   YYYY-MM-DD")
    parser.add_argument("--speed",  type=float, default=60.0,
                        help="Speed multiplier (60 = 60× faster than real-time)")
    args = parser.parse_args()

    try:
        asyncio.run(_cli_main(args))
    except KeyboardInterrupt:
        sys.exit(0)
