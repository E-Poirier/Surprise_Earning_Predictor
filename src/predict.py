"""Public inference entrypoint used by the API (thin wrapper).

Heavy dependencies (pandas, Finnhub, Yahoo, HF) load only when
:func:`predict_for_ticker` runs. Implementation is in :mod:`src.predict_core` so
tests can mock ``predict_for_ticker`` without pulling those imports.
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
