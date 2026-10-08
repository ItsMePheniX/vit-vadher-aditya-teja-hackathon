"""train_impact.py — CLI to calibrate the event-study impact model.

Usage
-----
    # Train on SVB sample (default, offline-capable):
    python scripts/train_impact.py

    # Train on a different news CSV:
    python scripts/train_impact.py --news data/my_news.csv

    # Use 3-day cumulative AR as target:
    python scripts/train_impact.py --target ar3

    # Use a different time split:
    python scripts/train_impact.py --split 2023-03-12

Options
-------
    --news FILE     Path to news CSV (default: data/svb_news_sample.csv)
    --target        ar1 (1-day AR, default) | ar3 (3-day cumulative)
    --split DATE    Train/test split date YYYY-MM-DD (default: 2023-03-13)
    --no-cache      Force re-download of prices (ignore local price cache)
    --offline       Use synthetic prices only (no network calls)
    --verbose       Enable DEBUG logging

Leakage note
------------
All features are computed from text BEFORE the event date.
All labels (abnormal returns) are computed from prices AFTER the event.
The train/test split is chronological — test events come after train events.
See docs/ASSUMPTIONS.md §Impact Score Calibration for full methodology.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train event-study impact model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--news",    type=Path, default=Path("data/svb_news_sample.csv"))
    parser.add_argument("--target",  choices=["ar1", "ar3"], default="ar1",
                        help="Regression target: 1-day (ar1) or 3-day (ar3) abnormal return")
    parser.add_argument("--split",   type=str,  default="2023-03-13",
                        help="YYYY-MM-DD — events before this are train, after are test")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download of prices")
    parser.add_argument("--offline",  action="store_true",
                        help="Skip yfinance; use synthetic returns for price cache")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("train_impact")

    # ── 0. Optionally wipe price cache ────────────────────────────────────────
    if args.no_cache:
        import shutil
        cache_dir = Path("data/prices_cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Cleared price cache at %s", cache_dir)

    # ── 1. Build event list from CSV ──────────────────────────────────────────
    print(f"\n[1/4] Loading events from: {args.news}")
    from src.engine.impact.event_study import build_events_from_csv, get_abnormal_returns

    if not args.news.exists():
        print(f"ERROR: {args.news} not found. Run `make sample` first.")
        sys.exit(1)

    events = build_events_from_csv(args.news)
    print(f"      {len(events)} event rows extracted")

    # ── 2. Fetch abnormal returns ──────────────────────────────────────────────
    if args.offline:
        # Monkey-patch yfinance to always raise so we use synthetic fallback
        import src.engine.impact.event_study as es_mod
        original_fetch = es_mod._fetch

        def _synthetic_fetch(ticker, start, end):
            import numpy as np, pandas as pd
            rng = np.random.default_rng(hash(ticker) & 0xFFFF)
            dates = pd.date_range(start, end, freq="B")
            vol = 0.018 if ticker != "SPY" else 0.009
            returns = pd.Series(rng.normal(0.0003, vol, len(dates)), index=dates)
            return returns

        es_mod._fetch = _synthetic_fetch
        print("[2/4] Offline mode — using synthetic price returns")
    else:
        print("[2/4] Fetching abnormal returns (may take a moment…)")

    df = get_abnormal_returns(events)
    print(f"      {len(df)} events with complete return data")

    if len(df) < 5:
        print(
            "ERROR: Not enough events with return data to train. "
            "Try --offline for synthetic prices or expand the news CSV."
        )
        sys.exit(1)

    # ── 3. Train model ────────────────────────────────────────────────────────
    print(f"[3/4] Training gradient boosting on target=abs_{args.target}")
    from src.engine.impact.train import train, save_artifacts

    split_date = date.fromisoformat(args.split)
    target_col = f"abs_{args.target}"
    model, mapping, metrics = train(df, split_date=split_date, target=target_col)

    # ── 4. Save artifacts ─────────────────────────────────────────────────────
    print("[4/4] Saving artifacts…")
    save_artifacts(model, mapping, metrics)

    print("\n✓ Training complete. The pipeline will now use the calibrated model.")
    print("  Reload the running API with `make run` to pick up the new model.\n")


if __name__ == "__main__":
    main()
