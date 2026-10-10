"""Pre-download ML models for offline demo mode.

Run this script once while connected to the internet. It forces 
the HuggingFace pipeline to download the weights and cache them 
locally. Afterwards, you can run the demo completely offline.
"""
import logging
from pathlib import Path

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.nlp.sentiment import _load as load_sentiment
from src.engine.nlp.classifier import _load as load_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Warming up ML caches for offline mode...")
    
    logger.info("Downloading/Loading FinBERT Sentiment model...")
    load_sentiment()
    
    logger.info("Downloading/Loading Zero-Shot NLI classifier model...")
    load_classifier()
    
    logger.info("✅ All models successfully cached. You can now run the demo offline!")

if __name__ == "__main__":
    main()
