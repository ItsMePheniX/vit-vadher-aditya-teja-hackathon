"""mode_manager.py — controls Live vs Replay engine mode.

The app can run in one of three modes at a time:

    IDLE     — no data flowing (default on startup)
    LIVE     — ingestor feeds synthetic/live news continuously
    REPLAY   — ReplayStreamer feeds a historical episode at configurable speed

Only one mode can be active at a time.  Switching cancels the previous task.

Public API
----------
    from src.engine.mode_manager import manager, Mode

    await manager.start_live()
    await manager.start_replay(episode="svb_march_2023", speed=60.0)
    await manager.stop()
    status = manager.status()   # dict with mode, episode, speed, running_since, ...

Episode registry (episodes available for replay)
-------------------------------------------------
Keyed by episode name.  Each entry points to CSV files in data/.
Add new episodes here or load from config.yaml in future.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Episode registry ──────────────────────────────────────────────────────────

EPISODES: dict[str, dict] = {
    "svb_march_2023": {
        "description": "SVB bank run — 8–17 March 2023",
        "news_csv":    "data/svb_news_sample.csv",
        "tweets_csv":  "data/svb_tweets_sample.csv",
        "start_date":  "2023-03-08",
        "end_date":    "2023-03-17",
    },
}


class Mode(str, Enum):
    IDLE   = "idle"
    LIVE   = "live"
    REPLAY = "replay"


# ── Manager ───────────────────────────────────────────────────────────────────

class ModeManager:
    """Singleton that manages the engine's operating mode."""

    def __init__(self) -> None:
        self._mode: Mode           = Mode.IDLE
        self._task: asyncio.Task | None = None
        self._episode: str | None  = None
        self._speed: float         = 1.0
        self._started_at: datetime | None = None
        self._signal_queue: asyncio.Queue | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def set_queue(self, q: asyncio.Queue) -> None:
        """Inject the shared signal_queue used by the broadcaster."""
        self._signal_queue = q

    # ── Control ───────────────────────────────────────────────────────────────

    async def start_live(self) -> None:
        """Start the synthetic/live ingestor."""
        await self._cancel()
        from src.engine import ingestor  # noqa: PLC0415
        self._mode       = Mode.LIVE
        self._episode    = None
        self._speed      = 1.0
        self._started_at = datetime.now(tz=timezone.utc)
        self._task = asyncio.create_task(ingestor.run(), name="live-ingestor")
        logger.info("Live mode started")

    async def start_replay(self, episode: str, speed: float = 60.0) -> None:
        """Start replay of a historical episode at `speed`× real-time."""
        if episode not in EPISODES:
            raise ValueError(f"Unknown episode: {episode!r}. Available: {list(EPISODES)}")
        ep = EPISODES[episode]

        await self._cancel()

        news_path   = Path(ep["news_csv"])
        tweets_path = Path(ep.get("tweets_csv", ""))

        if not news_path.exists():
            raise FileNotFoundError(
                f"Replay data not found: {news_path}. Run `make sample` first."
            )

        self._mode       = Mode.REPLAY
        self._episode    = episode
        self._speed      = speed
        self._started_at = datetime.now(tz=timezone.utc)
        self._task       = asyncio.create_task(
            self._run_replay(ep, news_path, tweets_path, speed),
            name=f"replay-{episode}",
        )
        logger.info("Replay mode started: episode=%s speed=%.0fx", episode, speed)

    async def stop(self) -> None:
        """Stop whatever is running; return to IDLE."""
        await self._cancel()
        self._mode       = Mode.IDLE
        self._episode    = None
        self._started_at = None
        logger.info("Engine stopped → IDLE")

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        uptime = None
        if self._started_at:
            uptime = (datetime.now(tz=timezone.utc) - self._started_at).total_seconds()
        return {
            "mode":        self._mode.value,
            "running":     self.is_running,
            "episode":     self._episode,
            "speed":       self._speed,
            "started_at":  self._started_at.isoformat() if self._started_at else None,
            "uptime_s":    round(uptime, 1) if uptime else None,
            "episodes":    list(EPISODES.keys()),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    async def _run_replay(
        self,
        ep: dict,
        news_path: Path,
        tweets_path: Path,
        speed: float,
    ) -> None:
        """Stream replay items through the NLP pipeline → shared signal queue."""
        from src.engine.ingest.news   import NewsSource         # noqa: PLC0415
        from src.engine.ingest.social import SocialSource       # noqa: PLC0415
        from src.engine.ingest.replay import ReplayStreamer     # noqa: PLC0415
        from src.engine.nlp.pipeline  import process            # noqa: PLC0415
        from src.engine.state         import get_store          # noqa: PLC0415
        from src.engine.corroboration import corroborate_one    # noqa: PLC0415
        from src.engine.nlp.pipeline  import write_signals      # noqa: PLC0415

        start = date.fromisoformat(ep["start_date"])
        end   = date.fromisoformat(ep["end_date"])

        sources = [NewsSource(path=news_path)]
        if tweets_path.exists():
            sources.append(SocialSource(path=tweets_path))

        item_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        streamer = ReplayStreamer(
            sources=sources,
            start=start,
            end=end,
            speed=speed,
            queue=item_queue,
        )

        recent_buffer: list = []
        store = get_store()

        # Run the streamer; it populates item_queue then returns
        replay_task = asyncio.create_task(streamer.run(), name="replay-streamer")

        try:
            while not replay_task.done() or not item_queue.empty():
                try:
                    item = await asyncio.wait_for(item_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                try:
                    signals = await asyncio.to_thread(process, item)
                    for sig in signals:
                        sig = corroborate_one(sig, recent_buffer)
                        store.ingest(sig)
                        recent_buffer.append(sig)
                        if len(recent_buffer) > 500:
                            recent_buffer = recent_buffer[-200:]
                        if self._signal_queue is not None:
                            try:
                                self._signal_queue.put_nowait(sig)
                            except asyncio.QueueFull:
                                pass
                    await asyncio.to_thread(write_signals, signals)
                    item_queue.task_done()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("Replay pipeline error: %s", exc)
                    item_queue.task_done()
        except asyncio.CancelledError:
            replay_task.cancel()
            raise

        logger.info("Replay complete: episode=%s", self._episode)
        self._mode = Mode.IDLE


# Module-level singleton
manager = ModeManager()
