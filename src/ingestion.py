"""Pull Finnhub + yfinance data into ``data/raw`` and ``data/processed`` (Phase 1).

Earnings: Finnhub ``company_earnings`` is merged with Yahoo ``earnings_history`` when
``ingestion.yfinance_earnings_backfill`` is true (default). Rows match on fiscal period
date; Finnhub values take precedence on overlap. Yahoo fills history Finnhub omits on the
free tier.

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
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import finnhub
import pandas as pd
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


def normalize_period_key(period: Any) -> str | None:
    """Canonical ``YYYY-MM-DD`` key for merging earnings rows."""
    if period is None or (isinstance(period, float) and pd.isna(period)):
        return None
    ts = pd.to_datetime(period, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def merge_earnings_by_period(
    preferred: list[dict[str, Any]],
    supplemental: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge by ``period`` (``YYYY-MM-DD``); **preferred** rows win on overlap."""
    by_period: dict[str, dict[str, Any]] = {}
    for r in preferred:
        if not isinstance(r, dict):
            continue
        p = normalize_period_key(r.get("period"))
        if p:
            by_period[p] = r
    for r in supplemental:
        if not isinstance(r, dict):
            continue
        p = normalize_period_key(r.get("period"))
        if p and p not in by_period:
            by_period[p] = r
    return sorted(by_period.values(), key=lambda x: normalize_period_key(x.get("period")) or "")


def merge_earnings_finnhub_yfinance(
    finnhub_rows: list[dict[str, Any]],
    yfinance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backward-compatible name for :func:`merge_earnings_by_period`."""
    return merge_earnings_by_period(finnhub_rows, yfinance_rows)


def _quarter_end_before_announcement(ann: pd.Timestamp | datetime | date) -> date:
    """Most recent Mar/Jun/Sep/Dec month-end strictly before the earnings announcement date."""
    ad = pd.Timestamp(ann).date()
    candidates: list[date] = []
    for y in range(ad.year - 3, ad.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            try:
                c = date(y, m, d)
            except ValueError:
                continue
            if c < ad:
                candidates.append(c)
    return max(candidates) if candidates else ad


def yfinance_earnings_history_rows(yf_ticker: str, ticker: str) -> list[dict[str, Any]]:
    """Quarterly EPS from Yahoo ``earnings_history`` (supplemental to Finnhub). Fragile — best-effort."""
    import yfinance as yf

    try:
        t = yf.Ticker(yf_ticker)
        eh = t.earnings_history
    except Exception as e:
        logger.warning("yfinance earnings_history failed for %s: %s", yf_ticker, e)
        return []
    if eh is None or eh.empty:
        return []

    rows: list[dict[str, Any]] = []
    for idx, row in eh.iterrows():
        ts = pd.Timestamp(idx)
        period_str = ts.strftime("%Y-%m-%d")
        d = ts.date()
        year = int(d.year)
        quarter = int((d.month - 1) // 3 + 1)

        act = row.get("epsActual")
        est = row.get("epsEstimate")
        sp = row.get("surprisePercent")
        if pd.isna(act) and pd.isna(est):
            continue

        rows.append(
            {
                "actual": float(act) if pd.notna(act) else None,
                "estimate": float(est) if pd.notna(est) else None,
                "period": period_str,
                "quarter": quarter,
                "year": year,
                "surprise_percent": float(sp) if pd.notna(sp) else None,
                "symbol": ticker,
                "_source": "yfinance_earnings_history",
            }
        )
    return rows


# yfinance raises ValueError("Yahoo caps limit at 100") if limit > 100 (see yfinance/base.py).
YAHOO_EARNINGS_CALENDAR_MAX_LIMIT = 100


def yfinance_earnings_calendar_rows(yf_ticker: str, ticker: str, *, limit: int = 100) -> list[dict[str, Any]]:
    """EPS rows from Yahoo ``get_earnings_dates`` (deeper than ``earnings_history``).

    Maps each announcement to a fiscal **period** = last calendar quarter-end before the
    announcement (US large-cap heuristic). Requires ``lxml`` for ``pd.read_html``.
    Includes **upcoming** quarters that have an EPS estimate but no reported EPS yet.
    """
    import yfinance as yf

    cap = min(max(1, int(limit)), YAHOO_EARNINGS_CALENDAR_MAX_LIMIT)
    if int(limit) > YAHOO_EARNINGS_CALENDAR_MAX_LIMIT:
        logger.warning(
            "yfinance get_earnings_dates limit=%s exceeds Yahoo max %s; using %s",
            limit,
            YAHOO_EARNINGS_CALENDAR_MAX_LIMIT,
            cap,
        )

    try:
        t = yf.Ticker(yf_ticker)
        df = t.get_earnings_dates(limit=cap)
    except ImportError as e:
        logger.warning("yfinance earnings calendar skipped for %s (install lxml): %s", yf_ticker, e)
        return []
    except Exception as e:
        logger.warning("yfinance get_earnings_dates failed for %s: %s", yf_ticker, e)
        return []
    if df is None or df.empty:
        return []

    df = df.sort_index(ascending=True)

    def _col(row: pd.Series, *candidates: str) -> Any:
        for c in candidates:
            if c in row.index:
                return row[c]
        for c in row.index:
            cl = str(c).lower().replace(" ", "")
            for want in candidates:
                if want.lower().replace(" ", "") in cl:
                    return row[c]
        return None

    def _eps_cell_missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and pd.isna(v):
            return True
        if isinstance(v, str) and v.strip() in ("", "-", "—", "N/A", "n/a"):
            return True
        return False

    def _eps_to_float(v: Any) -> float | None:
        if _eps_cell_missing(v):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    by_period: dict[str, dict[str, Any]] = {}
    for idx, row in df.iterrows():
        ann = pd.Timestamp(idx)
        period_end = _quarter_end_before_announcement(ann)
        period_str = period_end.isoformat()

        rep = _col(row, "Reported EPS", "ReportedEPS")
        est = _col(row, "EPS Estimate", "EPSEstimate")
        sur = _col(row, "Surprise(%)", "Surprise (%)", "Surprise(%)")

        actual = _eps_to_float(rep)
        estimate = _eps_to_float(est)
        # Historical rows always had reported EPS; we also need **upcoming** rows (estimate only)
        # so inference can find a quarter without ``actual`` (see ``find_upcoming_earnings_index``).
        if actual is None and estimate is None:
            continue

        sp: float | None = None
        if sur is not None and not (isinstance(sur, float) and pd.isna(sur)):
            try:
                sp = float(sur)
            except (TypeError, ValueError):
                sp = None

        year = int(period_end.year)
        quarter = int((period_end.month - 1) // 3 + 1)

        by_period[period_str] = {
            "actual": actual,
            "estimate": estimate,
            "period": period_str,
            "quarter": quarter,
            "year": year,
            "surprise_percent": sp,
            "symbol": ticker,
            "_source": "yfinance_earnings_calendar",
        }
    return list(by_period.values())


def normalize_earnings_rows(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Standardize ``/stock/earnings`` rows for downstream features."""
    df = pd.DataFrame([] if rows is None else rows)
    if df.empty:
        return df
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "period" in df.columns:
        pk = df["period"].map(lambda x: normalize_period_key(x) or None)
        df = df.assign(_period_key=pk).dropna(subset=["_period_key"])
        df = df.drop_duplicates(subset=["_period_key"], keep="first").drop(columns=["_period_key"])
    df = df.reset_index(drop=True)
    rename = {"surprisePercent": "surprise_percent"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "surprise_percent" not in df.columns:
        df["surprise_percent"] = pd.NA

    if "actual" in df.columns and "estimate" in df.columns:
        sp = df["surprise_percent"]
        if isinstance(sp, pd.DataFrame):
            df["surprise_percent"] = sp.iloc[:, 0]
            sp = df["surprise_percent"]
        need = sp.isna().to_numpy() & df["actual"].notna().to_numpy() & df["estimate"].notna().to_numpy()
        if need.any():
            est = df["estimate"].astype(float)
            ok = need & (est.abs().to_numpy() > 1e-12)
            df.loc[ok, "surprise_percent"] = (
                (df.loc[ok, "actual"].astype(float) - df.loc[ok, "estimate"].astype(float))
                / df.loc[ok, "estimate"].astype(float).abs()
                * 100.0
            )

    if "year" in df.columns and "quarter" in df.columns:
        df["fiscal_label"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)

    if "period" in df.columns:
        po = pd.to_datetime(df["period"], errors="coerce")
        df = df.assign(_sort=po).sort_values("_sort", ascending=True).drop(columns=["_sort"])
    elif "year" in df.columns and "quarter" in df.columns:
        df = df.sort_values(["year", "quarter"], ascending=True)
    df = df.reset_index(drop=True)

    drop_meta = [c for c in ("_source",) if c in df.columns]
    if drop_meta:
        df = df.drop(columns=drop_meta)

    return df


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
    import yfinance as yf

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
    fh_lock: threading.Lock | None = None,
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

    def _finnhub_earnings() -> Any:
        raw = client.company_earnings(ticker, limit=limit)
        _sleep_after_finnhub(cfg)
        return raw

    try:
        if fh_lock is not None:
            with fh_lock:
                earnings_raw = _finnhub_earnings()
        else:
            earnings_raw = _finnhub_earnings()
        _write_json(raw_dir / f"{ticker}_earnings.json", earnings_raw)
        earnings_list = earnings_raw if isinstance(earnings_raw, list) else []
        if ig.get("yfinance_earnings_backfill", True):
            yf_hist = yfinance_earnings_history_rows(yf_sym, ticker)
            cal_limit = int(ig.get("yfinance_earnings_calendar_limit", 100))
            yf_cal = yfinance_earnings_calendar_rows(yf_sym, ticker, limit=cal_limit)
            _write_json(raw_dir / f"{ticker}_earnings_yfinance_backfill.json", yf_hist)
            _write_json(raw_dir / f"{ticker}_earnings_yfinance_calendar.json", yf_cal)
            n_fh = len(earnings_list)
            merged = merge_earnings_by_period(earnings_list, yf_hist)
            merged = merge_earnings_by_period(merged, yf_cal)
            earnings_list = merged
            logger.info(
                "%s earnings: %d Finnhub → %d after Yahoo merge (history + calendar)",
                ticker,
                n_fh,
                len(earnings_list),
            )
        earnings_df = normalize_earnings_rows(earnings_list)
        earnings_df.to_parquet(processed_dir / f"{ticker}_earnings.parquet", index=False)

        if news_sample:
            end = date.today()
            start = end - timedelta(days=news_sample_days)

            def _finnhub_news() -> list[dict[str, Any]]:
                return fetch_company_news_range(client, ticker, start, end, cfg)

            if fh_lock is not None:
                with fh_lock:
                    news = _finnhub_news()
            else:
                news = _finnhub_news()
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
    jobs: int = 1,
) -> dict[str, Any]:
    """Ingest all symbols; returns a manifest dict.

    When ``jobs`` > 1, tickers are processed concurrently; a shared lock serializes Finnhub
    calls so rate limits are respected while Yahoo Finance work can overlap.
    """
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

    fh_lock = threading.Lock() if jobs > 1 else None
    if jobs < 1:
        jobs = 1

    if jobs == 1:
        for i, ticker in enumerate(tickers):
            logger.info("Ingesting %s (%d/%d)", ticker, i + 1, len(tickers))
            row = ingest_one_ticker(
                client,
                ticker,
                cfg,
                raw_dir,
                processed_dir,
                news_sample=news_sample,
                fh_lock=None,
            )
            results.append(row)
            if not row.get("ok"):
                errors[ticker] = str(row.get("error") or "unknown")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _one(sym: str) -> tuple[str, dict[str, Any]]:
            row = ingest_one_ticker(
                client,
                sym,
                cfg,
                raw_dir,
                processed_dir,
                news_sample=news_sample,
                fh_lock=fh_lock,
            )
            return sym, row

        logger.info("Ingesting %d tickers with jobs=%d (Finnhub serialized)", len(tickers), jobs)
        results_by: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_one, t): t for t in tickers}
            for fut in as_completed(futs):
                sym, row = fut.result()
                results_by[sym] = row
                if not row.get("ok"):
                    errors[sym] = str(row.get("error") or "unknown")
        results = [results_by[t] for t in tickers]

    metadata_rows = [r["metadata"] for r in results if r.get("ok") and "metadata" in r]
    meta_df = pd.DataFrame(metadata_rows)
    if not meta_df.empty:
        meta_df.to_parquet(processed_dir / "ticker_metadata.parquet", index=False)

    manifest = {
        "tickers_requested": tickers,
        "succeeded": [r["symbol"] for r in results if r.get("ok")],
        "failed": errors,
        "news_sample": news_sample,
        "jobs": jobs,
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
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent tickers (default 1). Finnhub calls are globally serialized; Yahoo work can overlap.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    if args.spike:
        tickers = [args.spike.strip().upper()]
    else:
        tickers = list(args.tickers) if args.tickers else list(TICKERS)
    manifest = run_ingestion(tickers, news_sample=args.news_sample, jobs=max(1, int(args.jobs)))
    logger.info("Done. Succeeded: %d, failed: %d", len(manifest["succeeded"]), len(manifest["failed"]))
    if manifest["failed"]:
        logger.warning("Failures: %s", manifest["failed"])


if __name__ == "__main__":
    main()
