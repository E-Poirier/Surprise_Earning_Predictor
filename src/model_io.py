"""Load the serialized classifier and training metadata.

Kept free of Finnhub, feature builders, and yfinance so the API and tests can
import it with minimal dependency surface.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config, resolve_path


def load_model_bundle(
    *,
    config: dict[str, Any] | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Load ``train_metadata.json`` and the corresponding ``.joblib`` model.

    Returns a dict with keys: ``model``, ``metadata``, ``feature_columns``,
    ``label_order``, ``models_dir``.

    If ``metadata["model_path"]`` points at a missing file (e.g. absolute path
    from another machine), falls back to ``<models_dir>/xgb_classifier.joblib``.
    """
    cfg = config if config is not None else load_config()
    mdir = models_dir if models_dir is not None else resolve_path("models_dir", cfg)
    meta_path = mdir / "train_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Train the model first: python -m src.train"
        )
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    # Prefer ``models_dir/xgb_classifier.joblib`` when metadata stores a stale absolute
    # ``model_path`` from another machine or moved checkout.
    default_model = mdir / "xgb_classifier.joblib"
    raw = metadata.get("model_path")
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = _PROJECT_ROOT / candidate
        model_path = candidate if candidate.exists() else default_model
    else:
        model_path = default_model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file (looked for {default_model} and metadata path {raw!r})"
        )
    model = joblib.load(model_path)
    feature_columns: list[str] = list(metadata["feature_columns"])
    label_order: list[str] = list(metadata["label_order"])
    return {
        "model": model,
        "metadata": metadata,
        "feature_columns": feature_columns,
        "label_order": label_order,
        "models_dir": mdir,
    }
