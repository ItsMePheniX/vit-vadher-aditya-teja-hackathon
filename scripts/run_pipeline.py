"""run_pipeline.py — run any ingest source through the NLP pipeline and print signals.

Usage examples
--------------
# News CSV (default: SVB sample):
    python scripts/run_pipeline.py --news data/svb_news_sample.csv

# Tweets CSV:
    python scripts/run_pipeline.py --tweets data/svb_tweets_sample.csv

# Both sources, filtered to SVB window, 10× speed:
    python scripts/run_pipeline.py \\
        --news   data/svb_news_sample.csv \\
        --tweets data/svb_tweets_sample.csv \\
        --start  2023-03-08  --end 2023-03-17 \\
        --speed  10

# Single headline (quick test without loading a file):
    python scripts/run_pipeline.py --text "JPMorgan beats Q3 earnings by 12 percent"

# Write signals to a custom JSONL file:
    python scripts/run_pipeline.py --news data/svb_news_sample.csv --out /tmp/out.jsonl

Options
-------
  --news FILE      Path to a news CSV/JSON file (NewsSource)
  --tweets FILE    Path to a tweets CSV file (SocialSource)
  --text TEXT      Process a single headline instead of a file
  --start DATE     Start date filter YYYY-MM-DD (default: 2000-01-01)
  --end   DATE     End date filter   YYYY-MM-DD (default: today)
  --speed FLOAT    Replay speed multiplier (default: 1_000_000 = near-instant)
  --out   FILE     Output JSONL path (default: data/signals.jsonl from config)
  --no-write       Don't persist signals (print only)
  --limit N        Stop after N signals
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure repo root on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.engine.ingest.base import RawItem
from src.engine.nlp.pipeline import process, write_signals
from src.engine.schema import Signal

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


# ── Single-headline mode ───────────────────────────────────────────────────────

def run_single(text: str, source: str = "cli", out: Path | None = None, no_write: bool = False) -> list[Signal]:
    item = RawItem(
        id="cli:000000",
        ts=datetime.now(tz=timezone.utc),
        text=text,
        source=source,
        url="",
    )
    signals = process(item)
    _print_signals(signals)
    if signals and not no_write:
        write_signals(signals, path=out)
    return signals


# ── File / replay mode ─────────────────────────────────────────────────────────

async def run_files(
    news_path: Path | None,
    tweets_path: Path | None,
    start: date,
    end: date,
    speed: float,
    out: Path | None,
    no_write: bool,
    limit: int,
) -> None:
    from src.engine.ingest.news import NewsSource    # noqa: PLC0415
    from src.engine.ingest.social import SocialSource  # noqa: PLC0415
    from src.engine.ingest.replay import ReplayStreamer  # noqa: PLC0415

    sources = []
    if news_path:
        sources.append(NewsSource(news_path))
    if tweets_path:
        sources.append(SocialSource(tweets_path))

    if not sources:
        console.print("[red]Error: provide --news, --tweets, or --text[/red]")
        return

    streamer = ReplayStreamer(sources, start=start, end=end, speed=speed, log_every=0)
    total = 0

    async def consumer() -> None:
        nonlocal total
        while True:
            item: RawItem = await streamer.queue.get()
            signals = await asyncio.to_thread(process, item)
            if signals:
                _print_signals(signals)
                if not no_write:
                    await asyncio.to_thread(write_signals, signals, out)
                total += len(signals)
                if limit and total >= limit:
                    return
            streamer.queue.task_done()

    prod = asyncio.create_task(streamer.run())
    cons = asyncio.create_task(consumer())
    try:
        await asyncio.gather(prod, cons)
    except asyncio.CancelledError:
        pass

    console.print(f"\n[green]Done — {total} signal(s) produced.[/green]")


# ── Display ────────────────────────────────────────────────────────────────────

def _print_signals(signals: list[Signal]) -> None:
    if not signals:
        return

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Entity", style="bold yellow", width=8)
    table.add_column("Event",  width=20)
    table.add_column("Sent",   width=7, justify="right")
    table.add_column("Imp",    width=4, justify="right")
    table.add_column("Conf",   width=6, justify="right")
    table.add_column("HL(h)",  width=6, justify="right")
    table.add_column("Evidence", no_wrap=False, max_width=70)

    for s in signals:
        sent_color = "green" if s.sentiment > 0.1 else ("red" if s.sentiment < -0.1 else "white")
        table.add_row(
            s.entity,
            s.event_type,
            f"[{sent_color}]{s.sentiment:+.3f}[/{sent_color}]",
            str(s.impact),
            f"{s.confidence:.2f}",
            f"{s.half_life_hours:.0f}",
            " | ".join(s.evidence)[:70],
        )

    console.print(table)


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run NLP pipeline on news/tweets and print signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--news",     type=Path, default=None)
    p.add_argument("--tweets",   type=Path, default=None)
    p.add_argument("--text",     type=str,  default=None, help="Single headline")
    p.add_argument("--start",    type=str,  default="2000-01-01")
    p.add_argument("--end",      type=str,  default=date.today().isoformat())
    p.add_argument("--speed",    type=float, default=1_000_000)
    p.add_argument("--out",      type=Path, default=None, help="Output JSONL path")
    p.add_argument("--no-write", action="store_true", help="Don't persist to JSONL")
    p.add_argument("--limit",    type=int,  default=0,   help="Stop after N signals")
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    console.print("[bold]Financial Risk NLP Engine — Pipeline Runner[/bold]\n")

    if args.text:
        run_single(args.text, out=args.out, no_write=args.no_write)
        return

    if not args.news and not args.tweets:
        # Default: run SVB sample news
        default = Path("data/svb_news_sample.csv")
        if default.exists():
            console.print(f"No source specified — using default: {default}\n")
            args.news = default
        else:
            console.print("[red]No source. Use --news, --tweets, or --text[/red]")
            sys.exit(1)

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    asyncio.run(run_files(
        news_path=args.news,
        tweets_path=args.tweets,
        start=start,
        end=end,
        speed=args.speed,
        out=args.out,
        no_write=args.no_write,
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()
