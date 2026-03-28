"""Pull Finnhub + yfinance data into ``data/raw`` and ``data/processed`` (Phase 1).

Run from project root::

    python -m src.ingestion
    python -m src.ingestion --spike AAPL
    python -m src.ingestion --tickers AAPL MSFT

Bulk company news for every historical quarter is **not** downloaded here (too many API
calls). Use :func:`fetch_company_news_range` from feature/sentiment steps with a narrow
date window per earnings event. Optional ``--news-sample`` fetches one recent window per
ticker to validate the Finnhub news endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import finnhub
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config, project_root, resolve_path
from config.tickers import TICKERS

logger = logging.getLogger(__name__)


def yfinance_symbol(ticker: str) -> str:
    """Map configured symbol to Yahoo ticker when they differ."""
    if ticker == "BRK-B":
        return "BRK-B"
    return ticker


def _sleep_after_finnhub(cfg: dict[str, Any]) -> None:
    time.sleep(float(cfg["finnhub"]["rate_limit_sleep_seconds"]))


def _payload_to_dataframe(payload: Any) -> pd.DataFrame:
    """Normalize Finnhub JSON (list, dict with ``data``, or empty) to a DataFrame."""
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        inner = payload.get("data")
        if isinstance(inner, list):
            return pd.DataFrame(inner)
        return pd.DataFrame([payload])
    return pd.DataFrame()


def normalize_earnings_rows(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Standardize ``/stock/earnings`` rows for downstream features."""
    df = pd.DataFrame([] if rows is None else rows)
    if df.empty:
        return df
    rename = {"surprisePercent": "surprise_percent"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "year" in df.columns and "quarter" in df.columns:
        df["fiscal_label"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    return df.sort_values(["year", "quarter"], ascending=False).reset_index(drop=True)


def fetch_company_news_range(
    client: finnhub.Client,
    symbol: str,
    date_from: date,
    date_to: date,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Finnhub ``company_news`` for ``[date_from, date_to]`` (inclusive)."""
    news = client.company_news(
        symbol,
        _from=date_from.isoformat(),
        to=date_to.isoformat(),
    )
    _sleep_after_finnhub(cfg)
    return list(news or [])


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _ingest_yfinance(
    ticker: str,
    yf_ticker: str,
    period: str,
    processed_dir: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    """Daily OHLCV + metadata from Yahoo. Returns a small metadata dict."""
    meta: dict[str, Any] = {"symbol": ticker, "yahoo_symbol": yf_ticker}
    try:
        t = yf.Ticker(yf_ticker)
        hist = t.history(period=period, auto_adjust=True)
    except Exception as e:
        logger.warning("yfinance history failed for %s (%s): %s", ticker, yf_ticker, e)
        return meta

    if hist is not None and not hist.empty:
        hist = hist.reset_index()
        date_col = hist.columns[0]
        hist[date_col] = pd.to_datetime(hist[date_col], utc=True).dt.tz_localize(None)
        hist = hist.rename(columns={date_col: "date"})
        out = processed_dir / f"{ticker}_prices.parquet"
        hist.to_parquet(out, index=False)
        raw_csv = raw_dir / f"{ticker}_yfinance_prices.csv"
        hist.to_csv(raw_csv, index=False)
    else:
        logger.warning("No price history for %s (%s)", ticker, yf_ticker)

    try:
        info = t.info or {}
    except Exception as e:
        logger.warning("yfinance .info failed for %s: %s", ticker, e)
        info = {}

    meta["sector"] = info.get("sector") or "Unknown"
    meta["industry"] = info.get("industry") or "Unknown"
    meta["exchange"] = info.get("exchange")
    meta["currency"] = info.get("currency")
    _write_json(raw_dir / f"{ticker}_yfinance_info.json", info)
    return meta


def ingest_one_ticker(
    client: finnhub.Client,
    ticker: str,
    cfg: dict[str, Any],
    raw_dir: Path,
    processed_dir: Path,
    *,
    news_sample: bool = False,
    news_sample_days: int = 30,
) -> dict[str, Any]:
    """Pull Finnhub earnings + yfinance prices; optional news sample.

    We do **not** call ``company_eps_estimates`` — Finnhub free tier returns 403; EPS
    revision features (``eps_revision_*``) are dropped from the model per project plan.
    """
    out: dict[str, Any] = {"symbol": ticker, "ok": True, "error": None}
    ig = cfg.get("ingestion", {})
    limit = int(ig.get("company_earnings_limit", 40))
    yf_period = str(ig.get("yfinance_history_period", "max"))

    yf_sym = yfinance_symbol(ticker)

    try:
        earnings_raw = client.company_earnings(ticker, limit=limit)
        _sleep_after_finnhub(cfg)
        _write_json(raw_dir / f"{ticker}_earnings.json", earnings_raw)
        earnings_list = earnings_raw if isinstance(earnings_raw, list) else []
        earnings_df = normalize_earnings_rows(earnings_list)
        earnings_df.to_parquet(processed_dir / f"{ticker}_earnings.parquet", index=False)

        if news_sample:
            end = date.today()
            start = end - timedelta(days=news_sample_days)
            news = fetch_company_news_range(client, ticker, start, end, cfg)
            _write_json(raw_dir / f"{ticker}_news_sample.json", news)

        meta = _ingest_yfinance(ticker, yf_sym, yf_period, processed_dir, raw_dir)
        out["metadata"] = meta

    except Exception as e:
        logger.exception("Ingest failed for %s", ticker)
        out["ok"] = False
        out["error"] = str(e)

    return out


def run_ingestion(
    tickers: list[str],
    *,
    news_sample: bool = False,
) -> dict[str, Any]:
    """Ingest all symbols; returns a manifest dict."""
    load_dotenv(project_root() / ".env")
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is not set. Add it to .env")

    cfg = load_config()
    raw_dir = resolve_path("raw", cfg)
    processed_dir = resolve_path("processed", cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    client = finnhub.Client(api_key=api_key)
    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for i, ticker in enumerate(tickers):
        logger.info("Ingesting %s (%d/%d)", ticker, i + 1, len(tickers))
        row = ingest_one_ticker(
            client,
            ticker,
            cfg,
            raw_dir,
            processed_dir,
            news_sample=news_sample,
        )
        results.append(row)
        if not row.get("ok"):
            errors[ticker] = str(row.get("error") or "unknown")

    metadata_rows = [r["metadata"] for r in results if r.get("ok") and "metadata" in r]
    meta_df = pd.DataFrame(metadata_rows)
    if not meta_df.empty:
        meta_df.to_parquet(processed_dir / "ticker_metadata.parquet", index=False)

    manifest = {
        "tickers_requested": tickers,
        "succeeded": [r["symbol"] for r in results if r.get("ok")],
        "failed": errors,
        "news_sample": news_sample,
    }
    _write_json(processed_dir / "ingestion_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Finnhub + yfinance data.")
    p.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Symbols (default: config.tickers.TICKERS)",
    )
    p.add_argument(
        "--spike",
        type=str,
        default=None,
        metavar="SYMBOL",
        help="Ingest only one symbol (quick Finnhub/yfinance check)",
    )
    p.add_argument(
        "--news-sample",
        action="store_true",
        help="Also fetch a recent Finnhub news window per ticker (extra API calls)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    if args.spike:
        tickers = [args.spike.strip().upper()]
    else:
        tickers = list(args.tickers) if args.tickers else list(TICKERS)
    manifest = run_ingestion(tickers, news_sample=args.news_sample)
    logger.info("Done. Succeeded: %d, failed: %d", len(manifest["succeeded"]), len(manifest["failed"]))
    if manifest["failed"]:
        logger.warning("Failures: %s", manifest["failed"])


if __name__ == "__main__":
    main()
