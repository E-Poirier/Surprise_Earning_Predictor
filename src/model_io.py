"""Load serialized model + training metadata (no Finnhub / feature pipeline imports)."""

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
    """Load ``xgb_classifier.joblib`` and ``train_metadata.json``."""
    cfg = config if config is not None else load_config()
    mdir = models_dir if models_dir is not None else resolve_path("models_dir", cfg)
    meta_path = mdir / "train_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Train the model first: python -m src.train"
        )
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    model_path = Path(metadata.get("model_path", str(mdir / "xgb_classifier.joblib")))
    if not model_path.is_absolute():
        model_path = _PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
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
