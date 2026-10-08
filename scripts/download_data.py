"""download_data.py — download full public datasets used by this project.

Datasets
--------
1. All the News (Kaggle)      → data/all_the_news.csv
   URL:    https://www.kaggle.com/datasets/snapcrack/all-the-news
   Licence: CC0 Public Domain
   Size:   ~200 MB compressed
   Format: CSV, columns: id, title, publication, author, date, year, month, url, content

2. Stock Market Tweets (Kaggle) → data/stock_tweets.csv
   URL:    https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction
   Licence: CC BY-SA 4.0
   Size:   ~50 MB
   Format: CSV, columns: tweet_id, writer, post_date, body, ticker_symbol

Both datasets require a free Kaggle account and the Kaggle CLI (pip install kaggle).
After downloading, run `make sample` to extract the SVB window.

Usage:
    python scripts/download_data.py [--news] [--tweets] [--all]

Or set KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR  = _REPO_ROOT / "data"

_DATASETS = {
    "news": {
        "kaggle_slug": "snapcrack/all-the-news",
        "description": "All the News — 200k+ financial/general articles (CC0)",
        "output_files": ["articles1.csv", "articles2.csv", "articles3.csv"],
        "rename_to": "all_the_news.csv",
    },
    "tweets": {
        "kaggle_slug": "equinxx/stock-tweets-for-sentiment-analysis-and-prediction",
        "description": "Stock Market Tweets for Sentiment Analysis (CC BY-SA 4.0)",
        "output_files": ["stock_tweets.csv"],
        "rename_to": "stock_tweets.csv",
    },
}


def _check_kaggle_cli() -> bool:
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _check_kaggle_auth() -> bool:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    return cfg.exists() or has_env


def download_dataset(key: str) -> None:
    meta = _DATASETS[key]
    slug = meta["kaggle_slug"]
    print(f"\nDownloading: {meta['description']}")
    print(f"  Kaggle slug: {slug}")

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(_DATA_DIR), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print(f"  ERROR: kaggle download failed for {slug}")
        print("  Ensure you have: pip install kaggle && ~/.kaggle/kaggle.json")
        return

    # Rename / consolidate output files
    rename_to = meta["rename_to"]
    out_files  = meta["output_files"]

    existing = [(_DATA_DIR / f) for f in out_files if (_DATA_DIR / f).exists()]
    if not existing:
        print(f"  WARNING: expected output files not found: {out_files}")
        return

    if len(existing) == 1:
        existing[0].rename(_DATA_DIR / rename_to)
    else:
        # Concatenate multiple parts (e.g. All the News has 3 CSVs)
        import csv  # noqa: PLC0415
        print(f"  Concatenating {len(existing)} part files → {rename_to}")
        with open(_DATA_DIR / rename_to, "w", newline="", encoding="utf-8") as out_fh:
            writer = None
            for part in existing:
                with open(part, encoding="utf-8", errors="replace") as in_fh:
                    reader = csv.DictReader(in_fh)
                    if writer is None:
                        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames or [])
                        writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
                part.unlink()

    final_path = _DATA_DIR / rename_to
    size_mb = final_path.stat().st_size / 1_048_576
    print(f"  Saved: {final_path} ({size_mb:.1f} MB)")
    print(f"  Next step: run `make sample` to extract the SVB window.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public datasets for the Financial Risk NLP Engine"
    )
    parser.add_argument("--news",   action="store_true", help="Download All the News (Kaggle)")
    parser.add_argument("--tweets", action="store_true", help="Download stock tweets (Kaggle)")
    parser.add_argument("--all",    action="store_true", help="Download all datasets")
    args = parser.parse_args()

    if not (args.news or args.tweets or args.all):
        parser.print_help()
        print("\nTip: use --all to download everything, then run `make sample`")
        sys.exit(0)

    if not _check_kaggle_cli():
        print("ERROR: kaggle CLI not found.")
        print("  Install with: uv pip install kaggle")
        sys.exit(1)

    if not _check_kaggle_auth():
        print("ERROR: Kaggle credentials not found.")
        print("  Download your API token from https://www.kaggle.com/account")
        print("  Place it at ~/.kaggle/kaggle.json  OR  set KAGGLE_USERNAME + KAGGLE_KEY")
        sys.exit(1)

    if args.all or args.news:
        download_dataset("news")
    if args.all or args.tweets:
        download_dataset("tweets")

    print("\nAll downloads complete. Run `make sample` to generate SVB window files.")


if __name__ == "__main__":
    main()
