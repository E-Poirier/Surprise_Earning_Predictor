"""Full inference implementation (heavy imports). Imported by ``src.predict`` on demand."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import finnhub
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config, project_root, resolve_path
from src.errors import InsufficientHistoryError
from src.features import (
    build_upcoming_inference_row,
    prepare_prices_df,
    surprise_label,
)
from src.ingestion import (
    merge_earnings_by_period,
    normalize_earnings_rows,
    yfinance_earnings_calendar_rows,
    yfinance_symbol,
)
from src.model_io import load_model_bundle
from src.sentiment import SentimentCache, build_inference_client

logger = logging.getLogger(__name__)


def _sleep_after_finnhub(cfg: dict[str, Any]) -> None:
    time.sleep(float(cfg["finnhub"]["rate_limit_sleep_seconds"]))


def _sector_universe_from_metadata(meta_df: pd.DataFrame) -> list[str]:
    """Same sector list strategy as ``run_feature_pipeline`` (one-hot alignment)."""
    sector_universe = sorted({str(s) for s in meta_df["sector"].dropna().unique()} or {"Unknown"})
    if "Unknown" not in sector_universe:
        sector_universe.append("Unknown")
    return sector_universe


def refresh_earnings_with_finnhub(
    ticker: str,
    existing: pd.DataFrame,
    client: finnhub.Client,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Merge latest Finnhub ``company_earnings`` with on-disk processed rows (Finnhub wins on overlap)."""
    limit = int(cfg.get("ingestion", {}).get("company_earnings_limit", 40))
    raw = client.company_earnings(ticker, limit=limit)
    _sleep_after_finnhub(cfg)
    finnhub_rows = raw if isinstance(raw, list) else []
    existing_rows = existing.to_dict("records") if not existing.empty else []
    merged = merge_earnings_by_period(finnhub_rows, existing_rows)
    return normalize_earnings_rows(merged)


def _direction_for_feature(name: str, value: float) -> str:
    if name.startswith("sector_"):
        return "neutral"
    if name in ("beat_rate_4q", "beat_rate_8q", "sentiment_score"):
        return "positive" if float(value) >= 0.5 else "negative"
    if name in ("momentum_30d", "momentum_60d", "surprise_magnitude_trend"):
        return "positive" if float(value) >= 0.0 else "negative"
    if name == "surprise_magnitude_avg":
        return "positive" if float(value) >= 5.0 else "negative"
    if name == "hist_vol_30d":
        return "positive" if float(value) >= 0.03 else "negative"
    if name == "quarter":
        return "neutral"
    return "positive" if float(value) >= 0.0 else "negative"


def _top_features(
    model: Any,
    feature_columns: list[str],
    x_row: np.ndarray,
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Pick top-``k`` features by global gain importance with heuristic direction labels."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None or len(imp) != len(feature_columns):
        order = np.arange(min(len(feature_columns), len(x_row[0])))
    else:
        order = np.argsort(-imp)
    out: list[dict[str, Any]] = []
    for j in order:
        ji = int(j)
        if ji >= len(feature_columns):
            continue
        name = feature_columns[ji]
        val = float(x_row[0, ji])
        out.append({"feature": name, "value": val, "direction": _direction_for_feature(name, val)})
        if len(out) >= k:
            break
    return out


def _last_quarters_table(
    earnings: pd.DataFrame,
    upcoming_idx: int,
    threshold_pct: float,
    *,
    max_rows: int = 4,
) -> list[dict[str, Any]]:
    """Most recent completed quarters before the upcoming row."""
    e = earnings.copy()
    e["_pit_order"] = pd.to_datetime(e["period"], errors="coerce")
    e = e.sort_values("_pit_order", ascending=True).drop(columns=["_pit_order"]).reset_index(drop=True)
    start = max(0, upcoming_idx - max_rows)
    slice_ = e.iloc[start:upcoming_idx]
    rows: list[dict[str, Any]] = []
    for _, r in slice_.iterrows():
        est = r.get("estimate")
        act = r.get("actual")
        sp = r.get("surprise_percent")
        if sp is not None and pd.notna(sp):
            try:
                spf = float(sp)
            except (TypeError, ValueError):
                spf = None
        else:
            spf = None
        if spf is None and est is not None and act is not None and pd.notna(est) and pd.notna(act):
            try:
                ae = float(act)
                ee = float(est)
                if abs(ee) > 1e-12:
                    spf = (ae - ee) / abs(ee) * 100.0
            except (TypeError, ValueError):
                spf = None
        lab = surprise_label(act, est, threshold_pct)
        rows.append(
            {
                "quarter": str(r.get("fiscal_label") or ""),
                "estimate": float(est) if est is not None and pd.notna(est) else None,
                "actual": float(act) if act is not None and pd.notna(act) else None,
                "surprise_pct": float(spf) if spf is not None else None,
                "label": lab,
            }
        )
    return rows


def predict_for_ticker(
    ticker: str,
    *,
    config: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run full inference for ``ticker``; returns a JSON-serializable dict (API response body)."""
    load_dotenv(project_root() / ".env")
    cfg = config if config is not None else load_config()
    b = bundle if bundle is not None else load_model_bundle(config=cfg)
    model = b["model"]
    feature_columns: list[str] = b["feature_columns"]
    label_order: list[str] = b["label_order"]

    processed_dir = resolve_path("processed", cfg)
    earn_path = processed_dir / f"{ticker}_earnings.parquet"
    px_path = processed_dir / f"{ticker}_prices.parquet"
    meta_path = processed_dir / "ticker_metadata.parquet"
    if not earn_path.exists() or not px_path.exists():
        raise InsufficientHistoryError("missing processed earnings or prices parquet; run ingestion")
    if not meta_path.exists():
        raise InsufficientHistoryError("missing ticker_metadata.parquet; run ingestion")

    earnings_df = pd.read_parquet(earn_path)
    api_key = os.environ.get("FINNHUB_API_KEY")
    if api_key:
        try:
            client = finnhub.Client(api_key=api_key)
            earnings_df = refresh_earnings_with_finnhub(ticker, earnings_df, client, cfg)
        except Exception as e:
            logger.warning("Finnhub refresh failed for %s; using on-disk earnings: %s", ticker, e)

    # Match ingestion: Yahoo earnings calendar adds **future** quarters (estimate, no actual) so
    # ``find_upcoming_earnings_index`` can resolve the next report; calendar scrape alone is not in Finnhub refresh.
    ig = cfg.get("ingestion", {})
    if ig.get("yfinance_earnings_backfill", True):
        cal_limit = int(ig.get("yfinance_earnings_calendar_limit", 100))
        try:
            yf_cal = yfinance_earnings_calendar_rows(yfinance_symbol(ticker), ticker, limit=cal_limit)
            merged = merge_earnings_by_period(earnings_df.to_dict("records"), yf_cal)
            earnings_df = normalize_earnings_rows(merged)
        except Exception as e:
            logger.warning("Yahoo earnings calendar merge failed for %s: %s", ticker, e)

    prices_raw = pd.read_parquet(px_path)
    prices = prepare_prices_df(prices_raw)

    meta_df = pd.read_parquet(meta_path)
    sector_universe = _sector_universe_from_metadata(meta_df)
    m = meta_df.loc[meta_df["symbol"] == ticker]
    sector = str(m.iloc[0]["sector"]) if not m.empty else "Unknown"

    threshold_pct = float(cfg["target"]["surprise_threshold_pct"])
    hf_key = os.environ.get("HF_API_KEY")
    skip_sentiment = not bool(hf_key)
    finnhub_client: finnhub.Client | None = finnhub.Client(api_key) if api_key else None
    hf_client: Any = build_inference_client(cfg) if not skip_sentiment else None
    sentiment_cache: SentimentCache | None = None
    if not skip_sentiment:
        sentiment_cache = SentimentCache(resolve_path("sentiment_cache", cfg))
        sentiment_cache.load()

    built = build_upcoming_inference_row(
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
    if built is None:
        raise InsufficientHistoryError(
            "need at least 8 completed quarters before an upcoming quarter with estimate; refresh ingestion"
        )
    row_dict, upcoming_idx = built

    X = np.zeros((1, len(feature_columns)), dtype=np.float64)
    for i, col in enumerate(feature_columns):
        if col not in row_dict:
            X[0, i] = 0.0
        else:
            X[0, i] = float(row_dict[col])

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    prediction = label_order[pred_idx]
    confidence = float(proba[pred_idx])
    probs = {label_order[j]: float(proba[j]) for j in range(len(label_order))}

    top = _top_features(model, feature_columns, X, k=3)
    last_q = _last_quarters_table(earnings_df, upcoming_idx, threshold_pct, max_rows=4)

    return {
        "ticker": ticker,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probs,
        "top_features": top,
        "last_quarters": last_q,
    }
