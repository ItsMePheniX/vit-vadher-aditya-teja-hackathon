"""Shared config loader — reads config.yaml + .env once at import time."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "config.yaml"

load_dotenv(_REPO_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def get(section: str, *keys: str, default: Any = None) -> Any:
    """Dot-path lookup into config.yaml, e.g. get('model', 'name')."""
    cfg = load_config()
    node: Any = cfg.get(section, {})
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


# Convenience: env-overrideable settings
API_HOST: str = os.getenv("API_HOST", get("api", "host", default="0.0.0.0"))
API_PORT: int = int(os.getenv("API_PORT", get("api", "port", default=8000)))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", get("api", "log_level", default="info"))
