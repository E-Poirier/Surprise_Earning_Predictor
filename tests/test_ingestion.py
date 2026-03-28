"""Unit tests for ingestion helpers (no network)."""

import pytest

from datetime import date

from src.ingestion import (
    _payload_to_dataframe,
    _quarter_end_before_announcement,
    merge_earnings_by_period,
    merge_earnings_finnhub_yfinance,
    normalize_earnings_rows,
    normalize_period_key,
    yfinance_symbol,
)


def test_yfinance_symbol_brk():
    assert yfinance_symbol("BRK-B") == "BRK-B"
    assert yfinance_symbol("AAPL") == "AAPL"


def test_normalize_earnings_rows():
    rows = [
        {
            "actual": 1.5,
            "estimate": 1.4,
            "period": "2024-03",
            "quarter": 1,
            "surprisePercent": 7.14,
            "symbol": "TEST",
            "year": 2024,
        }
    ]
    df = normalize_earnings_rows(rows)
    assert len(df) == 1
    assert df.loc[0, "surprise_percent"] == 7.14
    assert df.loc[0, "fiscal_label"] == "2024-Q1"


def test_normalize_earnings_empty():
    assert normalize_earnings_rows([]).empty
    assert normalize_earnings_rows(None).empty


def test_payload_to_dataframe_list():
    df = _payload_to_dataframe([{"a": 1}, {"a": 2}])
    assert list(df["a"]) == [1, 2]


def test_payload_to_dataframe_wrapped():
    df = _payload_to_dataframe({"data": [{"epsAvg": 1.0}]})
    assert df.loc[0, "epsAvg"] == 1.0


def test_normalize_period_key():
    assert normalize_period_key("2024-03-31") == "2024-03-31"


def test_quarter_end_before_announcement():
    # Announcement Nov 1 2025 → prior quarter end Sep 30 2025
    qe = _quarter_end_before_announcement(date(2025, 11, 1))
    assert qe == date(2025, 9, 30)


def test_merge_earnings_finnhub_yfinance_adds_only_new_periods():
    fh = [
        {
            "period": "2025-09-30",
            "year": 2025,
            "quarter": 4,
            "actual": 1.0,
            "estimate": 1.0,
            "symbol": "X",
        }
    ]
    yf = [
        {
            "period": "2024-06-30",
            "year": 2024,
            "quarter": 2,
            "actual": 0.9,
            "estimate": 0.88,
            "symbol": "X",
        }
    ]
    merged = merge_earnings_by_period(fh, yf)
    assert len(merged) == 2
    assert merged[0]["period"] == "2024-06-30"
    assert merged[1]["period"] == "2025-09-30"


def test_merge_earnings_finnhub_wins_on_overlap():
    fh = [{"period": "2025-09-30", "actual": 2.0, "estimate": 1.9, "year": 2025, "quarter": 4, "symbol": "X"}]
    yf = [{"period": "2025-09-30", "actual": 99.0, "estimate": 99.0, "year": 2025, "quarter": 4, "symbol": "X"}]
    merged = merge_earnings_by_period(fh, yf)
    assert len(merged) == 1
    assert merged[0]["actual"] == 2.0


def test_normalize_fills_surprise_percent_when_missing():
    rows = [
        {
            "actual": 1.1,
            "estimate": 1.0,
            "period": "2024-06-30",
            "quarter": 2,
            "year": 2024,
            "symbol": "T",
        }
    ]
    df = normalize_earnings_rows(rows)
    assert df.loc[0, "surprise_percent"] == pytest.approx(10.0)
