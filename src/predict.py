"""Load trained model and build features for one ticker (used by API).

Implementation lives in :mod:`src.predict_core` so tests can mock
``predict_for_ticker`` without importing Finnhub/yfinance/pandas first.
"""

from __future__ import annotations

from typing import Any

from src.errors import InsufficientHistoryError

__all__ = ["predict_for_ticker", "InsufficientHistoryError"]


def predict_for_ticker(
    ticker: str,
    *,
    config: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src import predict_core

    return predict_core.predict_for_ticker(ticker, config=config, bundle=bundle)
