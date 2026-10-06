"""Root conftest: ensure repo root is on sys.path for all test imports."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from src.engine...` everywhere without PYTHONPATH hacks
sys.path.insert(0, str(Path(__file__).parent))
