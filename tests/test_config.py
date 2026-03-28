"""Smoke tests for Phase 0 configuration."""

from config import load_config, project_root, resolve_path
from config.tickers import TICKERS, TICKER_SET


def test_ticker_universe_size():
    # Plan targets ~50 names; current list is 52 unique large-caps
    assert len(TICKERS) == 52
    assert len(TICKER_SET) == 52


def test_load_config():
    cfg = load_config()
    assert cfg["paths"]["data_root"] == "data"
    assert cfg["target"]["surprise_threshold_pct"] == 2.0
    assert "xgboost" in cfg["model"]


def test_resolve_path():
    cfg = load_config()
    p = resolve_path("features_file", cfg)
    assert p == project_root() / "data" / "features" / "features.parquet"
