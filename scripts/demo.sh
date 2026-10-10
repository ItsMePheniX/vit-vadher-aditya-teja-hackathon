#!/usr/bin/env bash
set -e

echo "==========================================="
echo " Starting FinRisk NLP Engine - SVB Replay "
echo "==========================================="

echo "1. Warming up ML model cache (offline support)..."
python scripts/warm_cache.py

echo "2. Starting the backend and triggering replay..."
python -m src.run --mode replay --episode svb_march_2023 --speed 60
