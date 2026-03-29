"""Point-in-time feature matrix → ``data/features/features.parquet`` (Phase 3).

Uses processed earnings (Finnhub) + daily prices (yfinance). **Earnings anchor date**
``D`` is the Finnhub ``period`` fiscal quarter-end date (announcement dates are not in
the free earnings payload); prices and news use data **through D−1** only.

Run from project root::

    python -m src.features
    python -m src.features --tickers AAPL MSFT
    python -m src.features --skip-sentiment   # tabular only; sentiment_score = neutral
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import finnhub
import numpy as np
import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config, project_root, resolve_path
from config.tickers import TICKERS
from src.ingestion import fetch_company_news_range
from src.sentiment import (
    SentimentCache,
    aggregate_sentiment,
    build_inference_client,
    earnings_news_window_dates,
    headlines_from_finnhub_news,
)

logger = logging.getLogger(__name__)

_CLOSE = "Close"


def pit_anchor_date(row: pd.Series) -> date:
    """Point-in-time calendar anchor ``D`` from Finnhub ``period`` (fiscal quarter end)."""
    return pd.to_datetime(row["period"], errors="coerce").date()


def pit_price_end_date(anchor: date) -> date:
    """Last calendar day usable for prices/news: ``D - 1``."""
    return anchor - timedelta(days=1)


def surprise_label(
    actual: float | None,
    estimate: float | None,
    threshold_pct: float,
) -> str | None:
    """``BEAT`` / ``MISS`` / ``IN_LINE`` from actual vs estimate; ``None`` if unusable."""
    if actual is None or estimate is None:
        return None
    if isinstance(actual, (float, np.floating)) and isinstance(estimate, (float, np.floating)):
        if np.isnan(actual) or np.isnan(estimate):
            return None
    try:
        a = float(actual)
        e = float(estimate)
    except (TypeError, ValueError):
        return None
    if abs(e) < 1e-12:
        return None
    rel = (a - e) / abs(e) * 100.0
    if rel > threshold_pct:
        return "BEAT"
    if rel < -threshold_pct:
        return "MISS"
    return "IN_LINE"


def surprise_magnitude_trend(magnitudes: list[float]) -> float | None:
    """Slope of absolute surprise percentage over prior quarters (oldest → newest), linear fit.

    Uses only past quarters already in ``magnitudes`` (typically four prior rows).
    """
    if len(magnitudes) < 4:
        return None
    x = np.arange(len(magnitudes), dtype=float)
    y = np.asarray(magnitudes, dtype=float)
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def abs_surprise_pct(row: pd.Series) -> float | None:
    """|surprise %| for a history row; falls back to recomputing from actual/estimate."""
    sp = row.get("surprise_percent")
    if sp is not None and pd.notna(sp):
        return abs(float(sp))
    a = row.get("actual")
    e = row.get("estimate")
    if a is None or e is None or pd.isna(a) or pd.isna(e):
        return None
    try:
        a = float(a)
        e = float(e)
    except (TypeError, ValueError):
        return None
    if abs(e) < 1e-12:
        return None
    return abs((a - e) / abs(e) * 100.0)


def prepare_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``date`` column to naive UTC-stripped datetimes."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date", _CLOSE])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def last_close_on_or_before(prices: pd.DataFrame, d: date) -> float | None:
    """Last ``Close`` on or before calendar day ``d``."""
    ts = pd.Timestamp(d)
    sub = prices.loc[prices["date"] <= ts]
    if sub.empty:
        return None
    return float(sub.iloc[-1][_CLOSE])


def momentum_calendar_return(
    prices: pd.DataFrame,
    anchor: date,
    *,
    lookback_days: int,
) -> float | None:
    """Total return from last close on/before ``anchor - lookback_days`` to last close on/before ``D-1``."""
    end_d = pit_price_end_date(anchor)
    start_d = end_d - timedelta(days=lookback_days)
    end_px = last_close_on_or_before(prices, end_d)
    start_px = last_close_on_or_before(prices, start_d)
    if end_px is None or start_px is None or start_px <= 0:
        return None
    return (end_px / start_px) - 1.0


def hist_vol_30d(prices: pd.DataFrame, anchor: date) -> float | None:
    """Std dev of daily log returns over the last 30 **trading** rows ending at ``D-1``."""
    end_d = pit_price_end_date(anchor)
    ts = pd.Timestamp(end_d)
    sub = prices.loc[prices["date"] <= ts]
    if len(sub) < 32:
        return None
    tail = sub.tail(31)
    c = tail[_CLOSE].astype(float)
    lr = np.log(c / c.shift(1)).dropna()
    if len(lr) < 30:
        return None
    return float(lr.iloc[-30:].std(ddof=0))


def beat_rate(prior: pd.DataFrame, threshold_pct: float) -> float | None:
    """Fraction of rows labeled ``BEAT`` among rows with a valid ``surprise_label``."""
    if prior.empty:
        return None
    labels: list[str] = []
    for _, r in prior.iterrows():
        lab = surprise_label(
            r.get("actual"),
            r.get("estimate"),
            threshold_pct,
        )
        if lab is not None:
            labels.append(lab)
    if not labels:
        return None
    return sum(1 for x in labels if x == "BEAT") / len(labels)


def _sector_column_name(sector: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (sector or "Unknown").strip()).strip("_")
    return f"sector_{s}" if s else "sector_Unknown"


def sector_one_hot(sector: str, sector_universe: list[str]) -> dict[str, float]:
    """Fixed dict of ``sector_*`` → 0/1 for tree models."""
    keys = [_sector_column_name(s) for s in sector_universe]
    out: dict[str, float] = {k: 0.0 for k in keys}
    target = _sector_column_name(sector)
    if target not in out:
        target = _sector_column_name("Unknown")
    if target in out:
        out[target] = 1.0
    return out


def _sentiment_for_row(
    client: finnhub.Client | None,
    hf_client: Any,
    cache: SentimentCache | None,
    cfg: dict[str, Any],
    ticker: str,
    anchor: date,
) -> float:
    """Mean FinBERT positive score over pre-earnings headlines; neutral on failure."""
    neutral = float(cfg.get("sentiment", {}).get("neutral_score", 0.5))
    if client is None or hf_client is None:
        return neutral
    win_start, win_end = earnings_news_window_dates(anchor, cfg=cfg)
    try:
        news = fetch_company_news_range(client, ticker, win_start, win_end, cfg)
    except Exception as e:
        logger.warning("Finnhub news failed for %s %s: %s", ticker, anchor, e)
        return neutral
    headlines = headlines_from_finnhub_news(news, win_start, win_end)
    try:
        return aggregate_sentiment(
            headlines,
            client=hf_client,
            cache=cache,
            cfg=cfg,
        )
    except Exception as e:
        logger.warning("aggregate_sentiment failed for %s %s: %s", ticker, anchor, e)
        return neutral


def build_features_for_ticker(
    ticker: str,
    cfg: dict[str, Any],
    *,
    earnings_df: pd.DataFrame,
    prices: pd.DataFrame,
    sector: str,
    sector_universe: list[str],
    threshold_pct: float,
    finnhub_client: finnhub.Client | None,
    hf_client: Any,
    sentiment_cache: SentimentCache | None,
    skip_sentiment: bool,
) -> list[dict[str, Any]]:
    """One row per company-quarter with enough prior history and a realized outcome."""
    earnings = earnings_df.copy()
    earnings["_pit_order"] = pd.to_datetime(earnings["period"], errors="coerce")
    earnings = earnings.sort_values("_pit_order", ascending=True).drop(columns=["_pit_order"]).reset_index(drop=True)
    n = len(earnings)
    rows: list[dict[str, Any]] = []
    if n < 9:
        logger.warning(
            "%s: need at least 9 earnings rows (8 prior quarters + current); have %d",
            ticker,
            n,
        )
        return rows

    for i in range(n):
        if i < 8:
            continue
        cur = earnings.iloc[i]
        act = cur.get("actual")
        est = cur.get("estimate")
        if act is None or est is None or pd.isna(act) or pd.isna(est):
            continue
        tgt = surprise_label(act, est, threshold_pct)
        if tgt is None:
            continue

        prior8 = earnings.iloc[i - 8 : i]
        prior4 = earnings.iloc[i - 4 : i]

        br8 = beat_rate(prior8, threshold_pct)
        br4 = beat_rate(prior4, threshold_pct)
        if br8 is None or br4 is None:
            continue

        mag_vals: list[float] = []
        for _, r in prior4.iterrows():
            v = abs_surprise_pct(r)
            if v is not None:
                mag_vals.append(v)
        if len(mag_vals) < 4:
            continue
        surprise_mag = float(np.mean(mag_vals))
        mag_trend = surprise_magnitude_trend(mag_vals)
        if mag_trend is None:
            continue

        anchor = pit_anchor_date(cur)
        q = int(cur["quarter"]) if pd.notna(cur.get("quarter")) else 1

        mom30 = momentum_calendar_return(prices, anchor, lookback_days=30)
        mom60 = momentum_calendar_return(prices, anchor, lookback_days=60)
        hv = hist_vol_30d(prices, anchor)
        if mom30 is None or mom60 is None or hv is None:
            logger.debug("Skipping %s %s: insufficient price history for momentum/vol", ticker, cur.get("fiscal_label"))
            continue

        if skip_sentiment:
            sent = float(cfg.get("sentiment", {}).get("neutral_score", 0.5))
        else:
            sent = _sentiment_for_row(
                finnhub_client,
                hf_client,
                sentiment_cache,
                cfg,
                ticker,
                anchor,
            )

        row: dict[str, Any] = {
            "ticker": ticker,
            "fiscal_label": str(cur.get("fiscal_label") or ""),
            "earnings_date": anchor,
            "beat_rate_4q": br4,
            "beat_rate_8q": br8,
            "surprise_magnitude_avg": surprise_mag,
            "surprise_magnitude_trend": mag_trend,
            "momentum_30d": mom30,
            "momentum_60d": mom60,
            "hist_vol_30d": hv,
            "quarter": q,
            "sentiment_score": sent,
            "target": tgt,
        }
        for k, v in sector_one_hot(sector, sector_universe).items():
            row[k] = v
        rows.append(row)

    return rows


def run_feature_pipeline(
    tickers: list[str],
    *,
    skip_sentiment: bool = False,
) -> dict[str, Any]:
    """Build ``features.parquet`` for ``tickers``; returns a small manifest."""
    load_dotenv(project_root() / ".env")
    cfg = load_config()
    processed_dir = resolve_path("processed", cfg)
    features_dir = resolve_path("features", cfg)
    features_path = resolve_path("features_file", cfg)
    features_dir.mkdir(parents=True, exist_ok=True)

    threshold_pct = float(cfg["target"]["surprise_threshold_pct"])
    meta_path = processed_dir / "ticker_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run ingestion first.")

    meta_df = pd.read_parquet(meta_path)
    sector_universe = sorted({str(s) for s in meta_df["sector"].dropna().unique()} or {"Unknown"})
    if "Unknown" not in sector_universe:
        sector_universe.append("Unknown")

    finnhub_client: finnhub.Client | None = None
    hf_client: Any = None
    sentiment_cache: SentimentCache | None = None

    if not skip_sentiment:
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            raise RuntimeError("FINNHUB_API_KEY is not set (required unless --skip-sentiment)")
        finnhub_client = finnhub.Client(api_key=api_key)
        hf_client = build_inference_client(cfg)
        cache_path = resolve_path("sentiment_cache", cfg)
        sentiment_cache = SentimentCache(cache_path)
        sentiment_cache.load()

    all_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}

    for i, ticker in enumerate(tickers):
        logger.info("Features %s (%d/%d)", ticker, i + 1, len(tickers))
        earn_path = processed_dir / f"{ticker}_earnings.parquet"
        px_path = processed_dir / f"{ticker}_prices.parquet"
        if not earn_path.exists() or not px_path.exists():
            skipped[ticker] = "missing earnings or prices parquet"
            logger.warning("%s: %s", ticker, skipped[ticker])
            continue

        earnings_df = pd.read_parquet(earn_path)
        if earnings_df.empty:
            skipped[ticker] = "empty earnings"
            continue

        prices_raw = pd.read_parquet(px_path)
        prices = prepare_prices_df(prices_raw)

        m = meta_df.loc[meta_df["symbol"] == ticker]
        sector = str(m.iloc[0]["sector"]) if not m.empty else "Unknown"

        rows = build_features_for_ticker(
            ticker,
            cfg,
            earnings_df=earnings_df,
            prices=prices,
            sector=sector,
            sector_universe=sector_universe,
            threshold_pct=threshold_pct,
            finnhub_client=finnhub_client,
            hf_client=hf_client,
            sentiment_cache=sentiment_cache,
            skip_sentiment=skip_sentiment,
        )
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    if not out_df.empty:
        out_df.to_parquet(features_path, index=False)
    else:
        logger.warning("No feature rows produced; not writing %s", features_path)

    manifest = {
        "tickers_requested": tickers,
        "rows": int(len(out_df)),
        "output": str(features_path),
        "skipped_tickers": skipped,
        "skip_sentiment": skip_sentiment,
    }
    manifest_path = features_dir / "features_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build point-in-time features parquet.")
    p.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Symbols (default: config.tickers.TICKERS)",
    )
    p.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Set sentiment_score to neutral; no Finnhub news / HF calls",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    tickers = list(args.tickers) if args.tickers else list(TICKERS)
    manifest = run_feature_pipeline(tickers, skip_sentiment=args.skip_sentiment)
    logger.info("Wrote %d rows to %s", manifest["rows"], manifest["output"])


if __name__ == "__main__":
    main()
