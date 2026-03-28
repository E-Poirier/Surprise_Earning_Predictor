"""Unit tests for sentiment (no Hugging Face network calls)."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.sentiment import (
    SentimentCache,
    aggregate_sentiment,
    earnings_news_window_dates,
    headline_cache_key,
    headlines_from_finnhub_news,
    positive_score_from_classification,
    score_headline,
)


class _FakeEl:
    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = score


def test_positive_score_from_classification():
    out = [
        _FakeEl("negative", 0.1),
        _FakeEl("positive", 0.85),
        _FakeEl("neutral", 0.05),
    ]
    assert positive_score_from_classification(out) == pytest.approx(0.85)


def test_positive_score_from_classification_fallback():
    out = [_FakeEl("label_positive", 0.7)]
    assert positive_score_from_classification(out) == pytest.approx(0.7)


def test_positive_score_finbert_tone_label_1():
    out = [_FakeEl("LABEL_0", 0.2), _FakeEl("LABEL_1", 0.7), _FakeEl("LABEL_2", 0.1)]
    assert positive_score_from_classification(out) == pytest.approx(0.7)


def test_headline_cache_key_stable():
    assert headline_cache_key("  hello  ") == headline_cache_key("hello")


def test_earnings_news_window_dates():
    d = date(2024, 5, 15)
    start, end = earnings_news_window_dates(d, lookback_days=14)
    assert end == date(2024, 5, 14)
    assert start == date(2024, 5, 1)


def test_earnings_news_window_dates_uses_cfg_lookback():
    d = date(2024, 5, 15)
    start, end = earnings_news_window_dates(d, cfg={"news": {"lookback_days": 7}})
    assert end == date(2024, 5, 14)
    assert start == date(2024, 5, 8)


def test_headlines_from_finnhub_news_filters_by_window():
    # 2024-01-10 12:00 UTC, 2024-01-05 12:00 UTC
    items = [
        {"datetime": 1704888000, "headline": "A"},
        {"datetime": 1704456000, "headline": "B"},
        {"datetime": 1600000000, "headline": "too old"},
    ]
    win_start = date(2024, 1, 1)
    win_end = date(2024, 1, 31)
    hs = headlines_from_finnhub_news(items, win_start, win_end)
    assert "A" in hs and "B" in hs
    assert "too old" not in hs


def test_aggregate_sentiment_empty():
    cfg = {
        "sentiment": {"neutral_score": 0.5},
        "paths": {"sentiment_cache": "data/sentiment_cache.json"},
    }
    m = MagicMock()
    assert aggregate_sentiment([], client=m, cfg=cfg) == 0.5
    m.text_classification.assert_not_called()


def test_aggregate_sentiment_mean_and_cache(tmp_path: Path):
    cfg = {
        "sentiment": {
            "neutral_score": 0.5,
            "model_id": "ProsusAI/finbert",
            "max_retries": 3,
            "cold_start_retry_sleep_seconds": 0,
        },
        "paths": {"sentiment_cache": str(tmp_path / "cache.json")},
    }
    client = MagicMock()
    client.text_classification.side_effect = [
        [_FakeEl("positive", 0.8)],
        [_FakeEl("positive", 0.8)],
    ]

    cache = SentimentCache(tmp_path / "cache.json")
    cache.load()

    a = aggregate_sentiment(["a", "b"], client=client, cfg=cfg, cache=cache)
    assert a == pytest.approx(0.8)
    assert client.text_classification.call_count == 2

    # Second aggregate with same headlines: cache hits
    b = aggregate_sentiment(["a", "b"], client=client, cfg=cfg, cache=cache)
    assert b == pytest.approx(0.8)
    assert client.text_classification.call_count == 2


def test_score_headline_empty_whitespace():
    cfg = {"sentiment": {"neutral_score": 0.42}}
    client = MagicMock()
    assert score_headline("   ", client=client, cache=None, cfg=cfg) == 0.42
    client.text_classification.assert_not_called()
