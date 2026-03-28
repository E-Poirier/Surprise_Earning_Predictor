"""Project configuration: YAML + helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache ``config.yaml``. Path defaults to ``config/config.yaml`` next to this package."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def project_root() -> Path:
    """Repository root (parent of ``config/``)."""
    return _PROJECT_ROOT


def resolve_path(key: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve a path from ``config['paths']`` relative to project root."""
    cfg = config if config is not None else load_config()
    rel = cfg["paths"][key]
    p = Path(rel)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p
