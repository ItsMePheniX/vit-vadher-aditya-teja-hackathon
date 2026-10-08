"""evaluate.py — evaluate NLP pipeline outputs against labelled gold data.

This is a stub for Phase 2. It will:
1. Load labelled signals from data/gold_labels.jsonl
2. Load predicted signals from data/signals.jsonl
3. Compute precision / recall / F1 per event_type
4. Compute sentiment MAE vs. human labels
5. Print a formatted report

Run with: make evaluate
"""
from __future__ import annotations

import sys


def main() -> None:
    print("evaluate.py — stub (Phase 2)")
    print("Gold labels not yet available; this script will be extended.")
    print("Run `make replay` to generate signals, then evaluate them here.")


if __name__ == "__main__":
    main()
