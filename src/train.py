"""Time-based split, multiclass XGBoost with sample weights, save to ``models/`` (Phase 4).

Train / validation / test are **quarter-based** (``fiscal_label``), never shuffled.
Optional **validation-only** hyperparameter grid (macro-F1); test set is never used for selection.

Run from project root::

    python -m src.train
    python -m src.train --no-tune
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config, resolve_path

logger = logging.getLogger(__name__)

_FISCAL_RE = re.compile(r"^(\d{4})-Q([1-4])$")


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays for ``json.dump``."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def _import_xgb_classifier():
    """Import ``XGBClassifier``; raise with install hints if the native library fails (common on macOS)."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier
    except Exception as e:
        hint = (
            "The XGBoost wheel needs OpenMP (libomp) at runtime.\n"
            "  • macOS + Homebrew:  brew install libomp\n"
            "    (install Homebrew from https://brew.sh if you do not have brew)\n"
            "  • macOS + conda (no brew):  conda install -c conda-forge llvm-openmp\n"
            "    If you use a .venv built with Apple/system Python, either recreate the venv\n"
            "    with your conda Python, or run once per shell:\n"
            "      export DYLD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${DYLD_LIBRARY_PATH:-}\"\n"
            "Then run:  python -m src.train"
        )
        raise RuntimeError(f"Failed to load XGBoost.\n{hint}\n\nUnderlying error:\n{e}") from e


BASE_FEATURE_COLS = [
    "beat_rate_4q",
    "beat_rate_8q",
    "surprise_magnitude_avg",
    "surprise_magnitude_trend",
    "momentum_30d",
    "momentum_60d",
    "hist_vol_30d",
    "quarter",
    "sentiment_score",
]

TUNE_PARAM_KEYS = ("max_depth", "n_estimators", "learning_rate", "min_child_weight")


def parse_fiscal_label(s: str) -> tuple[int, int] | None:
    """Parse ``YYYY-Qn`` → ``(year, quarter)``; ``None`` if invalid."""
    if not s or not isinstance(s, str):
        return None
    m = _FISCAL_RE.match(s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_quarter_key(key: str) -> tuple[int, int]:
    """Parse split boundary from config (e.g. ``2022-Q1``)."""
    t = parse_fiscal_label(key)
    if t is None:
        raise ValueError(f"Invalid quarter key in config: {key!r}")
    return t


def fiscal_cmp(a: tuple[int, int], b: tuple[int, int]) -> int:
    """``-1`` if ``a < b``, ``0`` if equal, ``1`` if ``a > b``."""
    if a[0] != b[0]:
        return -1 if a[0] < b[0] else 1
    if a[1] != b[1]:
        return -1 if a[1] < b[1] else 1
    return 0


def assign_split(
    fiscal_label: str,
    *,
    val_start: tuple[int, int],
    val_end: tuple[int, int],
    test_start: tuple[int, int],
) -> str | None:
    """``train`` | ``val`` | ``test`` | ``None`` (unknown quarter)."""
    t = parse_fiscal_label(fiscal_label)
    if t is None:
        return None
    if fiscal_cmp(t, val_start) < 0:
        return "train"
    if fiscal_cmp(t, val_end) <= 0:
        return "val"
    if fiscal_cmp(t, test_start) >= 0:
        return "test"
    return None


def feature_columns_for_df(df: pd.DataFrame) -> list[str]:
    """Stable column order: base features then sorted ``sector_*``."""
    missing = [c for c in BASE_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Features parquet missing columns: {missing}. Re-run: python -m src.features"
        )
    sector_cols = sorted(c for c in df.columns if c.startswith("sector_"))
    return list(BASE_FEATURE_COLS) + sector_cols


def _fit_xgb(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    fit_kwargs: dict[str, Any] = {
        "sample_weight": sample_weight,
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    if len(y_val) > 0:
        fit_kwargs["early_stopping_rounds"] = 40
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("early_stopping_rounds", None)
        model.fit(X_train, y_train, **fit_kwargs)


def _tune_grid_product(tune_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    lists: list[list[Any]] = []
    for k in TUNE_PARAM_KEYS:
        v = tune_cfg.get(k)
        if not isinstance(v, list) or not v:
            raise ValueError(f"model.tune.{k} must be a non-empty list when tuning is enabled")
        lists.append(v)
    combos: list[dict[str, Any]] = []
    for tup in itertools.product(*lists):
        combos.append(dict(zip(TUNE_PARAM_KEYS, tup)))
    return combos


def _grid_search_xgb(
    XGBClassifier: Any,
    base_xgb: dict[str, Any],
    tune_cfg: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label_order: list[str],
) -> tuple[Any, dict[str, Any], float, list[dict[str, Any]]]:
    """Fit a grid on train; score with macro-F1 on validation only. Refit once with best params."""
    combos = _tune_grid_product(tune_cfg)
    labels = list(range(len(label_order)))
    best_score = -1.0
    best_merged: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    logger.info("Hyperparameter search: %d candidates (validation macro-F1)", len(combos))
    for overrides in combos:
        merged = {**base_xgb, **overrides}
        model = XGBClassifier(**merged)
        _fit_xgb(model, X_train, y_train, sample_weight, X_val, y_val)
        pred = model.predict(X_val)
        score = float(
            f1_score(np.asarray(y_val), pred, average="macro", labels=labels, zero_division=0)
        )
        row = {k: merged[k] for k in TUNE_PARAM_KEYS}
        row["val_macro_f1"] = score
        results.append(row)
        if score > best_score:
            best_score = score
            best_merged = merged

    logger.info("Best validation macro-F1: %.4f %s", best_score, {k: best_merged[k] for k in TUNE_PARAM_KEYS})

    final = XGBClassifier(**best_merged)
    _fit_xgb(final, X_train, y_train, sample_weight, X_val, y_val)
    results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    return final, best_merged, best_score, results


def run_training(
    *,
    config: dict[str, Any] | None = None,
    features_path: Path | None = None,
    no_tune: bool = False,
) -> dict[str, Any]:
    """Load features, time-split, fit XGBoost, save model + metadata."""
    cfg = config if config is not None else load_config()
    paths = cfg["paths"]
    splits_cfg = cfg["splits"]
    label_order: list[str] = list(cfg["labels"]["order"])
    if len(label_order) != 3:
        raise ValueError("config.labels.order must list exactly 3 classes")

    val_start = parse_quarter_key(splits_cfg["validation_start_quarter"])
    val_end = parse_quarter_key(splits_cfg["validation_end_quarter"])
    test_start = parse_quarter_key(splits_cfg["test_start_quarter"])

    if fiscal_cmp(val_start, val_end) > 0:
        raise ValueError("validation_start_quarter must be <= validation_end_quarter")
    if fiscal_cmp(val_end, test_start) >= 0:
        raise ValueError("test_start_quarter must be after validation_end_quarter")

    feat_path = features_path if features_path is not None else resolve_path("features_file", cfg)
    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}. Run src.features first.")

    df = pd.read_parquet(feat_path)
    if df.empty:
        raise ValueError(f"No rows in {feat_path}")

    df = df.copy()
    df["_split"] = df["fiscal_label"].astype(str).map(
        lambda s: assign_split(s, val_start=val_start, val_end=val_end, test_start=test_start)
    )
    n_unassigned = int(df["_split"].isna().sum())
    if n_unassigned:
        logger.warning("Dropping %d rows with missing/invalid fiscal_label or gap split", n_unassigned)
    df = df.dropna(subset=["_split"])

    label_to_idx = {name: i for i, name in enumerate(label_order)}
    if df["target"].isna().any():
        df = df.dropna(subset=["target"])
    unknown = set(df["target"].unique()) - set(label_order)
    if unknown:
        raise ValueError(f"Unexpected target labels: {unknown}")

    feature_cols = feature_columns_for_df(df)
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df["target"].map(label_to_idx).astype(np.int32)

    split_masks = {
        "train": df["_split"] == "train",
        "val": df["_split"] == "val",
        "test": df["_split"] == "test",
    }

    for name, m in split_masks.items():
        n = int(m.sum())
        logger.info("Split %s: %d rows", name, n)
        if n == 0 and name in ("train", "val"):
            raise ValueError(
                f"No rows in {name} split — check fiscal_label coverage vs config splits "
                f"(need data before {val_start} for train, and between {val_start} and {val_end} for val)."
            )

    X_train, y_train = X[split_masks["train"]], y[split_masks["train"]]
    X_val, y_val = X[split_masks["val"]], y[split_masks["val"]]
    X_test, y_test = X[split_masks["test"]], y[split_masks["test"]]

    sample_weight = compute_sample_weight("balanced", y_train)

    XGBClassifier = _import_xgb_classifier()

    base_xgb = dict(cfg["model"]["xgboost"])
    tune_cfg = dict(cfg["model"].get("tune") or {})
    tune_on = (
        tune_cfg.get("enabled", False)
        and not no_tune
        and len(y_val) > 0
    )
    if tune_cfg.get("enabled", False) and not no_tune and len(y_val) == 0:
        logger.warning("Tuning enabled but validation set is empty; using base model.xgboost only.")

    tuning_meta: dict[str, Any]
    if tune_on:
        model, xgb_params, best_f1, tune_results = _grid_search_xgb(
            XGBClassifier,
            base_xgb,
            tune_cfg,
            X_train,
            y_train,
            sample_weight,
            X_val,
            y_val,
            label_order,
        )
        tuning_meta = {
            "enabled": True,
            "metric": tune_cfg.get("metric", "macro_f1"),
            "best_val_macro_f1": best_f1,
            "candidates_evaluated": len(tune_results),
            "top_candidates": tune_results[:10],
        }
    else:
        model = XGBClassifier(**base_xgb)
        _fit_xgb(model, X_train, y_train, sample_weight, X_val, y_val)
        xgb_params = base_xgb
        tuning_meta = {"enabled": False}

    def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, split_name: str) -> dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(label_order))),
            target_names=label_order,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_order))))
        return {
            "split": split_name,
            "n_samples": int(len(y_true)),
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    y_val_pred = model.predict(X_val)
    metrics_val = metrics_dict(np.asarray(y_val), y_val_pred, "validation")

    metrics_test: dict[str, Any] | None = None
    if len(y_test) > 0:
        y_test_pred = model.predict(X_test)
        metrics_test = metrics_dict(np.asarray(y_test), y_test_pred, "test")
        logger.info(
            "Test accuracy (single evaluation, no tuning): %.4f",
            metrics_test["accuracy"],
        )
    else:
        logger.warning("No test split rows — skipping test metrics")

    models_dir = resolve_path("models_dir", cfg)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "xgb_classifier.joblib"
    metadata_path = models_dir / "train_metadata.json"

    joblib.dump(model, model_path)

    metadata: dict[str, Any] = {
        "label_order": label_order,
        "feature_columns": feature_cols,
        "xgb_params": xgb_params,
        "hyperparameter_search": tuning_meta,
        "splits": {
            "validation_start_quarter": splits_cfg["validation_start_quarter"],
            "validation_end_quarter": splits_cfg["validation_end_quarter"],
            "test_start_quarter": splits_cfg["test_start_quarter"],
        },
        "metrics_validation": metrics_val,
        "metrics_test": metrics_test,
        "features_file": str(feat_path),
        "model_path": str(model_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metadata to %s", metadata_path)

    return metadata


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiclass XGBoost on features.parquet (time split).")
    p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Override path to features parquet (default: config paths.features_file)",
    )
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip validation grid search; use model.xgboost from config only",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    run_training(features_path=args.features, no_tune=args.no_tune)


if __name__ == "__main__":
    main()
