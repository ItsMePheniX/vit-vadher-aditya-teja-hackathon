"""Unified Runner for the Financial Risk NLP Engine.

Usage:
    python -m src.run --mode replay --episode svb_collapse_2023 --speed 60
    python -m src.run --mode live

Starts the FastAPI backend and automatically triggers the selected mode.
"""
import argparse
import asyncio
import logging
import sys
import threading
import time

import httpx
import uvicorn

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def start_server():
    """Start uvicorn in this thread."""
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="warning",
    )


def trigger_mode(mode: str, episode: str, speed: float):
    """Wait for the server to come up, then trigger the API endpoint."""
    base_url = f"http://{config.API_HOST}:{config.API_PORT}"
    
    # Wait for health check
    for _ in range(10):
        try:
            r = httpx.get(f"{base_url}/health")
            if r.status_code == 200:
                break
        except httpx.RequestError:
            time.sleep(1)
    else:
        logger.error("Timeout waiting for API to start.")
        sys.exit(1)

    logger.info("API is up. Triggering mode: %s...", mode)
    
    try:
        if mode == "replay":
            payload = {"episode": episode, "speed": speed}
            r = httpx.post(f"{base_url}/replay/start", json=payload)
            r.raise_for_status()
            logger.info("✅ Replay started successfully: %s", r.json())
        elif mode == "live":
            r = httpx.post(f"{base_url}/live/start")
            r.raise_for_status()
            logger.info("✅ Live mode started successfully.")
    except Exception as e:
        logger.error("Failed to start mode %s: %s", mode, e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Unified Runner for FinRisk NLP Engine")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay", help="Mode to start in")
    parser.add_argument("--episode", type=str, default="svb_march_2023", help="Episode ID for replay")
    parser.add_argument("--speed", type=float, default=60.0, help="Playback speed for replay")
    
    args = parser.parse_args()

    # Start the server in a daemon thread so it runs in the background of this script
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Trigger the mode from the main thread
    trigger_mode(args.mode, args.episode, args.speed)
    
    # Keep the main thread alive as long as the server thread is running
    try:
        while server_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")

if __name__ == "__main__":
    main()
