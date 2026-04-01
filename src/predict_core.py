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
from src.errors import (
    REASON_DETAIL_MESSAGES,
    REASON_MISSING_PROCESSED_DATA,
    InsufficientHistoryError,
)
from src.features import (
    build_upcoming_inference_row,
    prepare_prices_df,
    surprise_label,
)
from src.ingestion import (
    merge_earnings_by_period,
    normalize_earnings_rows,
    yfinance_earnings_calendar_rows,
    yfinance_earnings_history_rows,
    yfinance_symbol,
)
from src.model_io import load_model_bundle
from src.shap_explain import build_shap_explanation
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


def _price_history_for_chart(prices: pd.DataFrame, *, calendar_days: int = 90) -> list[dict[str, Any]]:
    """Daily closes for the UI chart (ingestion Yahoo OHLCV, ``Close`` column)."""
    if prices is None or prices.empty or "Close" not in prices.columns:
        return []
    px = prices.sort_values("date")
    end = px["date"].max()
    if pd.isna(end):
        return []
    start = end - pd.Timedelta(days=calendar_days)
    sub = px.loc[(px["date"] >= start) & (px["date"] <= end)]
    out: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        ts = pd.Timestamp(r["date"])
        out.append({"date": ts.date().isoformat(), "close": float(r["Close"])})
    return out


def try_load_prediction_context(
    ticker: str,
    cfg: dict[str, Any],
    *,
    refresh_finnhub: bool = True,
) -> dict[str, Any] | None:
    """Load earnings/prices and clients for inference.

    Returns ``None`` if processed parquet files are missing. Applies optional Finnhub refresh
    and Yahoo earnings calendar merge (same order as :func:`predict_for_ticker`).
    """
    processed_dir = resolve_path("processed", cfg)
    earn_path = processed_dir / f"{ticker}_earnings.parquet"
    px_path = processed_dir / f"{ticker}_prices.parquet"
    meta_path = processed_dir / "ticker_metadata.parquet"
    if not earn_path.exists() or not px_path.exists() or not meta_path.exists():
        return None

    earnings_df = pd.read_parquet(earn_path)
    api_key = os.environ.get("FINNHUB_API_KEY")
    if refresh_finnhub and api_key:
        try:
            client = finnhub.Client(api_key=api_key)
            earnings_df = refresh_earnings_with_finnhub(ticker, earnings_df, client, cfg)
        except Exception as e:
            logger.warning("Finnhub refresh failed for %s; using on-disk earnings: %s", ticker, e)

    ig = cfg.get("ingestion", {})
    if ig.get("yfinance_earnings_backfill", True):
        cal_limit = int(ig.get("yfinance_earnings_calendar_limit", 100))
        yf_sym = yfinance_symbol(ticker)
        try:
            # Match ingestion: earnings_history then calendar (both add quarters Finnhub may omit).
            rows_in = earnings_df.to_dict("records")
            yf_hist = yfinance_earnings_history_rows(yf_sym, ticker)
            merged = merge_earnings_by_period(rows_in, yf_hist)
            yf_cal = yfinance_earnings_calendar_rows(yf_sym, ticker, limit=cal_limit)
            merged = merge_earnings_by_period(merged, yf_cal)
            earnings_df = normalize_earnings_rows(merged)
            logger.info(
                "%s earnings at predict: %d rows after Yahoo merge (history + calendar)",
                ticker,
                len(earnings_df),
            )
        except Exception as e:
            logger.warning("Yahoo earnings merge failed for %s: %s", ticker, e)

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

    return {
        "earnings_df": earnings_df,
        "prices": prices,
        "sector": sector,
        "sector_universe": sector_universe,
        "threshold_pct": threshold_pct,
        "finnhub_client": finnhub_client,
        "hf_client": hf_client,
        "sentiment_cache": sentiment_cache,
        "skip_sentiment": skip_sentiment,
    }


def predictability_for_ticker(
    ticker: str,
    *,
    config: dict[str, Any] | None = None,
    refresh_finnhub: bool = False,
) -> tuple[bool, str | None]:
    """Return whether a full feature row can be built (disk + merge path; no model).

    ``refresh_finnhub=False`` matches a fast bulk check from last ingestion; ``True`` matches
    :func:`predict_for_ticker` data freshness more closely but is slower (Finnhub per ticker).
    """
    cfg = config if config is not None else load_config()
    ctx = try_load_prediction_context(ticker, cfg, refresh_finnhub=refresh_finnhub)
    if ctx is None:
        return False, REASON_MISSING_PROCESSED_DATA
    built = build_upcoming_inference_row(
        ticker,
        cfg,
        earnings_df=ctx["earnings_df"],
        prices=ctx["prices"],
        sector=ctx["sector"],
        sector_universe=ctx["sector_universe"],
        threshold_pct=ctx["threshold_pct"],
        finnhub_client=ctx["finnhub_client"],
        hf_client=ctx["hf_client"],
        sentiment_cache=ctx["sentiment_cache"],
        skip_sentiment=ctx["skip_sentiment"],
    )
    if built[0] is None:
        return False, built[1]
    return True, None


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

    ctx = try_load_prediction_context(ticker, cfg, refresh_finnhub=True)
    if ctx is None:
        raise InsufficientHistoryError(
            REASON_DETAIL_MESSAGES[REASON_MISSING_PROCESSED_DATA],
            reason_code=REASON_MISSING_PROCESSED_DATA,
        )

    built = build_upcoming_inference_row(
        ticker,
        cfg,
        earnings_df=ctx["earnings_df"],
        prices=ctx["prices"],
        sector=ctx["sector"],
        sector_universe=ctx["sector_universe"],
        threshold_pct=ctx["threshold_pct"],
        finnhub_client=ctx["finnhub_client"],
        hf_client=ctx["hf_client"],
        sentiment_cache=ctx["sentiment_cache"],
        skip_sentiment=ctx["skip_sentiment"],
    )
    if built[0] is None:
        code = built[1]
        msg = REASON_DETAIL_MESSAGES.get(code, "Insufficient history for prediction.")
        raise InsufficientHistoryError(msg, reason_code=code)
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
    explainer = b.get("shap_explainer")
    shap_explanation = build_shap_explanation(
        explainer,
        model,
        X,
        feature_columns,
        label_order,
        pred_idx,
        top_n=12,
    )
    last_q = _last_quarters_table(ctx["earnings_df"], upcoming_idx, ctx["threshold_pct"], max_rows=4)
    price_hist = _price_history_for_chart(ctx["prices"], calendar_days=90)

    anchor = row_dict.get("earnings_date")
    anchor_iso: str | None
    if anchor is None:
        anchor_iso = None
    elif hasattr(anchor, "isoformat"):
        anchor_iso = anchor.isoformat()
    else:
        anchor_iso = str(anchor)

    return {
        "ticker": ticker,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probs,
        "top_features": top,
        "last_quarters": last_q,
        "upcoming_fiscal_quarter": str(row_dict.get("fiscal_label") or ""),
        "earnings_anchor_date": anchor_iso,
        "price_history": price_hist,
        "shap_explanation": shap_explanation,
    }
