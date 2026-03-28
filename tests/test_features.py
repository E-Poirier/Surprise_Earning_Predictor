"""Point-in-time feature tests: no lookahead, pure helpers (no network)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features import (
    abs_surprise_pct,
    beat_rate,
    build_features_for_ticker,
    hist_vol_30d,
    last_close_on_or_before,
    momentum_calendar_return,
    pit_anchor_date,
    pit_price_end_date,
    prepare_prices_df,
    sector_one_hot,
    surprise_label,
)


def test_surprise_label_beat_miss_inline():
    assert surprise_label(1.1, 1.0, 2.0) == "BEAT"
    assert surprise_label(0.9, 1.0, 2.0) == "MISS"
    assert surprise_label(1.01, 1.0, 2.0) == "IN_LINE"
    assert surprise_label(0.99, 1.0, 2.0) == "IN_LINE"


def test_surprise_label_invalid():
    assert surprise_label(None, 1.0, 2.0) is None
    assert surprise_label(1.0, 0.0, 2.0) is None
    assert surprise_label(np.nan, 1.0, 2.0) is None


def test_pit_price_end_date():
    d = date(2024, 6, 30)
    assert pit_price_end_date(d) == date(2024, 6, 29)


def test_pit_anchor_date_from_row():
    row = pd.Series({"period": "2024-03-31"})
    assert pit_anchor_date(row) == date(2024, 3, 31)


def test_last_close_on_or_before_excludes_future_rows():
    prices = prepare_prices_df(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-05-01", "2024-07-01"]),
                "Close": [10.0, 11.0, 999.0],
            }
        )
    )
    end = date(2024, 6, 1)
    assert last_close_on_or_before(prices, end) == 11.0


def test_momentum_calendar_return_no_lookahead():
    """Prices after ``D-1`` must not affect momentum."""
    prices = prepare_prices_df(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-05-20", "2024-06-15"]),
                "Close": [100.0, 110.0, 500.0],
            }
        )
    )
    anchor = date(2024, 6, 1)
    end_d = pit_price_end_date(anchor)
    assert end_d == date(2024, 5, 31)
    assert last_close_on_or_before(prices, end_d) == 110.0
    start_d = end_d - timedelta(days=30)
    assert last_close_on_or_before(prices, start_d) == 100.0
    mom = momentum_calendar_return(prices, anchor, lookback_days=30)
    assert mom is not None
    assert mom == pytest.approx(110.0 / 100.0 - 1.0)


def test_hist_vol_30d_uses_only_trading_rows_before_d_minus_one():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=len(dates)))
    prices = prepare_prices_df(pd.DataFrame({"date": dates, "Close": close}))
    anchor = date(2024, 3, 15)
    end_d = pit_price_end_date(anchor)
    v1 = hist_vol_30d(prices, anchor)
    # Inject an artificial spike *after* D-1 — volatility should not change
    prices2 = prices.copy()
    mask = prices2["date"] > pd.Timestamp(end_d)
    prices2.loc[mask, "Close"] = prices2.loc[mask, "Close"] * 10.0
    v2 = hist_vol_30d(prices2, anchor)
    assert v1 is not None and v2 is not None
    assert v1 == pytest.approx(v2, rel=1e-9)


def test_beat_rate():
    prior = pd.DataFrame(
        {
            "actual": [1.1, 0.9, 1.0, 1.0],
            "estimate": [1.0, 1.0, 1.0, 1.0],
        }
    )
    br = beat_rate(prior, 2.0)
    assert br == pytest.approx(0.25)


def test_sector_one_hot():
    u = ["Technology", "Unknown"]
    d = sector_one_hot("Technology", u)
    assert d["sector_Technology"] == 1.0
    assert d["sector_Unknown"] == 0.0
    d2 = sector_one_hot("Weird Sector", u)
    assert d2["sector_Unknown"] == 1.0


def test_abs_surprise_pct_fallback():
    row = pd.Series({"actual": 1.1, "estimate": 1.0, "surprise_percent": np.nan})
    assert abs_surprise_pct(row) == pytest.approx(10.0)


def test_build_features_for_ticker_end_to_end():
    """Synthetic path: 9 quarters of earnings + long price history; sentiment skipped."""
    spec = [
        (2020, 1, "2020-03-31"),
        (2020, 2, "2020-06-30"),
        (2020, 3, "2020-09-30"),
        (2020, 4, "2020-12-31"),
        (2021, 1, "2021-03-31"),
        (2021, 2, "2021-06-30"),
        (2021, 3, "2021-09-30"),
        (2021, 4, "2021-12-31"),
        (2022, 1, "2022-03-31"),
    ]
    rows = []
    for year, quarter, period in spec:
        rows.append(
            {
                "actual": 1.05,
                "estimate": 1.0,
                "period": period,
                "quarter": quarter,
                "year": year,
                "surprise_percent": 5.0,
                "symbol": "TEST",
                "fiscal_label": f"{year}-Q{quarter}",
            }
        )

    earnings_df = pd.DataFrame(rows)
    start = date(2018, 1, 1)
    dates = pd.date_range(start, periods=900, freq="B")
    close = 100.0 * (1.0 + np.linspace(0, 0.5, len(dates)))
    prices = prepare_prices_df(pd.DataFrame({"date": dates, "Close": close}))

    out = build_features_for_ticker(
        "TEST",
        {"sentiment": {"neutral_score": 0.5}},
        earnings_df=earnings_df,
        prices=prices,
        sector="Technology",
        sector_universe=["Technology", "Unknown"],
        threshold_pct=2.0,
        finnhub_client=None,
        hf_client=None,
        sentiment_cache=None,
        skip_sentiment=True,
    )
    assert len(out) >= 1
    row0 = out[0]
    assert row0["target"] == "BEAT"
    assert row0["beat_rate_4q"] == 1.0
    assert row0["beat_rate_8q"] == 1.0
    assert "sector_Technology" in row0 and row0["sector_Technology"] == 1.0
