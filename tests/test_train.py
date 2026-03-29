"""Unit tests for time-based splits and fiscal parsing (no training on real data)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train import (
    assign_split,
    feature_columns_for_df,
    fiscal_cmp,
    parse_fiscal_label,
    parse_quarter_key,
    run_training,
)


def test_parse_fiscal_label():
    assert parse_fiscal_label("2022-Q1") == (2022, 1)
    assert parse_fiscal_label(" 2024-Q4 ") == (2024, 4)
    assert parse_fiscal_label("") is None
    assert parse_fiscal_label("2024-Q5") is None
    assert parse_fiscal_label("bad") is None


def test_parse_quarter_key():
    assert parse_quarter_key("2022-Q1") == (2022, 1)


def test_fiscal_cmp():
    assert fiscal_cmp((2021, 4), (2022, 1)) == -1
    assert fiscal_cmp((2022, 1), (2022, 1)) == 0
    assert fiscal_cmp((2023, 4), (2023, 2)) == 1


def test_assign_split():
    vs, ve, ts = (2022, 1), (2023, 4), (2024, 1)
    assert assign_split("2021-Q4", val_start=vs, val_end=ve, test_start=ts) == "train"
    assert assign_split("2022-Q1", val_start=vs, val_end=ve, test_start=ts) == "val"
    assert assign_split("2023-Q4", val_start=vs, val_end=ve, test_start=ts) == "val"
    assert assign_split("2024-Q1", val_start=vs, val_end=ve, test_start=ts) == "test"


def test_feature_columns_for_df():
    df = pd.DataFrame(
        {
            "beat_rate_4q": [1.0],
            "beat_rate_8q": [1.0],
            "surprise_magnitude_avg": [1.0],
            "surprise_magnitude_trend": [0.0],
            "momentum_30d": [0.0],
            "momentum_60d": [0.0],
            "hist_vol_30d": [0.1],
            "quarter": [1],
            "sentiment_score": [0.5],
            "sector_Tech": [1.0],
            "sector_X": [0.0],
        }
    )
    cols = feature_columns_for_df(df)
    assert cols[:9] == [
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
    assert cols[9:] == ["sector_Tech", "sector_X"]


def test_run_training_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end train on synthetic parquet with known splits."""
    try:
        import xgboost  # noqa: F401 — may raise if libomp missing on macOS
    except Exception as e:
        pytest.skip(f"xgboost not loadable: {e}")

    from config import load_config

    cfg = load_config()
    cfg = {**cfg, "splits": {**cfg["splits"]}}

    rows = []
    # Train: before 2022-Q1
    for y, q in [(2020, 1), (2020, 2), (2021, 4)]:
        rows.append(_synthetic_row(f"{y}-Q{q}", "BEAT"))
    # Val
    for y, q in [(2022, 1), (2022, 4), (2023, 2)]:
        rows.append(_synthetic_row(f"{y}-Q{q}", "IN_LINE"))
    # Test
    for y, q in [(2024, 1), (2024, 3)]:
        rows.append(_synthetic_row(f"{y}-Q{q}", "MISS"))

    feat_path = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(feat_path, index=False)

    models_dir = tmp_path / "models"
    monkeypatch.setitem(cfg["paths"], "models_dir", str(models_dir))

    meta = run_training(config=cfg, features_path=feat_path, no_tune=True)

    assert meta["metrics_validation"]["n_samples"] == 3
    assert meta["metrics_test"]["n_samples"] == 2
    assert (models_dir / "xgb_classifier.joblib").exists()
    with open(models_dir / "train_metadata.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["label_order"] == ["BEAT", "IN_LINE", "MISS"]
    assert len(loaded["feature_columns"]) >= 9
    assert loaded["hyperparameter_search"]["enabled"] is False


def _synthetic_row(fiscal_label: str, target: str) -> dict:
    return {
        "ticker": "SYN",
        "fiscal_label": fiscal_label,
        "earnings_date": np.datetime64("2020-01-01"),
        "beat_rate_4q": 0.5,
        "beat_rate_8q": 0.5,
        "surprise_magnitude_avg": 2.0,
        "surprise_magnitude_trend": 0.0,
        "momentum_30d": 0.01,
        "momentum_60d": 0.02,
        "hist_vol_30d": 0.2,
        "quarter": 1,
        "sentiment_score": 0.5,
        "sector_Test": 1.0,
        "target": target,
    }
