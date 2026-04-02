"""FinBERT (HF Inference) + JSON cache + pre-earnings news helpers (Phase 2).

``aggregate_sentiment`` returns the **mean positive class score** from the
configured FinBERT model (default ``ProsusAI/finbert``) over headline strings. Empty input → ``neutral_score``
(default 0.5). Headlines are cached by ``md5`` of trimmed text in
``data/sentiment_cache.json`` (see ``config/config.yaml`` paths).

Cold start: on HTTP 503, ``InferenceTimeoutError``, or a body mentioning
``estimated_time``, we sleep ``cold_start_retry_sleep_seconds`` and retry up to
``max_retries``.

Run from project root::

    python -m src.sentiment --headline "Markets rally on earnings optimism"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config, project_root, resolve_path

logger = logging.getLogger(__name__)

# FinBERT / BERT-style models: avoid pathological headline lengths
_MAX_HEADLINE_CHARS = 2000

# ---------------------------------------------------------------------------
# Cache keys & HF classification output parsing
# ---------------------------------------------------------------------------


def headline_cache_key(text: str) -> str:
    """Stable cache key: hex ``md5`` of UTF-8 trimmed headline."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def positive_score_from_classification(
    outputs: list[Any],
) -> float:
    """Return the **positive** class probability from HF text-classification output.

    ``ProsusAI/finbert`` / typical FinBERT heads use string labels ``positive``,
    ``neutral``, ``negative``. Some checkpoints emit ``LABEL_0`` / ``LABEL_1`` / ``LABEL_2``
    instead (``LABEL_1`` = positive for ``yiyanghkust/finbert-tone``).
    """
    if not outputs:
        return 0.5
    for el in outputs:
        label = (getattr(el, "label", "") or "").strip()
        lu = label.lower()
        if lu in ("positive", "label_1", "label1"):
            return float(getattr(el, "score", 0.5))
    for el in outputs:
        label = (getattr(el, "label", "") or "").strip()
        lu = label.lower()
        if "positive" in lu:
            return float(getattr(el, "score", 0.5))
    for el in outputs:
        label = (getattr(el, "label", "") or "").strip()
        if label.upper() == "LABEL_1":
            return float(getattr(el, "score", 0.5))
    logger.warning("No positive label in classifier output: %s", outputs)
    return 0.5


def _truncate(text: str) -> str:
    if len(text) <= _MAX_HEADLINE_CHARS:
        return text
    return text[:_MAX_HEADLINE_CHARS]


def _should_retry_inference(exc: BaseException) -> bool:
    from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

    if isinstance(exc, InferenceTimeoutError):
        return True
    if isinstance(exc, HfHubHTTPError):
        sc = getattr(exc.response, "status_code", None)
        if sc == 503:
            return True
        blob = " ".join(
            filter(
                None,
                [
                    str(exc.server_message or ""),
                    str(exc),
                ],
            )
        ).lower()
        if "estimated_time" in blob:
            return True
    return False


# ---------------------------------------------------------------------------
# On-disk score cache (md5 headline → positive probability)
# ---------------------------------------------------------------------------


@dataclass
class SentimentCache:
    """JSON file cache: ``{ md5_hex: float_positive_score }``."""

    path: Path
    _data: dict[str, float] | None = field(default=None, repr=False)

    def load(self) -> None:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            self._data = {str(k): float(v) for k, v in raw.items()}
        else:
            self._data = {}

    @property
    def data(self) -> dict[str, float]:
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data

    def get(self, key: str) -> float | None:
        v = self.data.get(key)
        return None if v is None else float(v)

    def set(self, key: str, value: float) -> None:
        self.data[key] = float(value)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Hugging Face Inference API client (FinBERT)
# ---------------------------------------------------------------------------


def build_inference_client(cfg: dict[str, Any] | None = None) -> Any:
    """``InferenceClient`` + ``HF_API_KEY``. Provider from ``sentiment.hf_provider`` (default ``auto``).

    ``hf-inference`` alone can return **400** for some BERT checkpoints on the router;
    ``auto`` lets Hugging Face choose a working inference route.
    """
    from huggingface_hub import InferenceClient

    load_dotenv(project_root() / ".env")
    token = os.environ.get("HF_API_KEY")
    if not token:
        raise RuntimeError("HF_API_KEY is not set. Add it to .env")
    cfg = cfg if cfg is not None else load_config()
    sentiment_cfg = cfg.get("sentiment", {})
    model_id = str(sentiment_cfg.get("model_id", "ProsusAI/finbert"))
    provider = sentiment_cfg.get("hf_provider", "auto")
    p = str(provider).strip().lower()
    if p in ("auto", "", "none", "default"):
        return InferenceClient(model_id, token=token)
    return InferenceClient(model_id, token=token, provider=p)


# ---------------------------------------------------------------------------
# Finnhub news → headline strings (point-in-time window before earnings)
# ---------------------------------------------------------------------------


def earnings_news_window_dates(
    earnings_date: date,
    *,
    lookback_days: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[date, date]:
    """Inclusive calendar window for news: ``[D - lookback_days, D - 1]`` (point-in-time).

    If ``lookback_days`` is omitted, uses ``news.lookback_days`` from ``config.yaml``
    (default 14).
    """
    if lookback_days is None:
        cfg = cfg if cfg is not None else load_config()
        lookback_days = int(cfg.get("news", {}).get("lookback_days", 14))
    end = earnings_date - timedelta(days=1)
    start = earnings_date - timedelta(days=lookback_days)
    return start, end


def headlines_from_finnhub_news(
    news_items: Iterable[dict[str, Any]],
    window_start: date,
    window_end: date,
) -> list[str]:
    """Keep Finnhub ``company_news`` rows whose UTC calendar day falls in ``[window_start, window_end]``."""
    out: list[str] = []
    for item in news_items:
        ts = item.get("datetime")
        if ts is None:
            continue
        try:
            d = datetime.utcfromtimestamp(int(ts)).date()
        except (TypeError, ValueError, OSError):
            continue
        if window_start <= d <= window_end:
            h = item.get("headline")
            if isinstance(h, str):
                t = h.strip()
                if t:
                    out.append(t)
    return out


# ---------------------------------------------------------------------------
# Per-headline and aggregate sentiment (retries, cache, neutral fallback)
# ---------------------------------------------------------------------------


def score_headline(
    text: str,
    *,
    client: Any,
    cache: SentimentCache | None,
    cfg: dict[str, Any],
) -> float:
    """Single headline → positive FinBERT score; uses cache; neutral on failure after retries."""
    from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

    sentiment_cfg = cfg.get("sentiment", {})
    neutral = float(sentiment_cfg.get("neutral_score", 0.5))
    max_retries = int(sentiment_cfg.get("max_retries", 3))
    sleep_s = float(sentiment_cfg.get("cold_start_retry_sleep_seconds", 20))
    model_id = str(sentiment_cfg.get("model_id", "ProsusAI/finbert"))

    t = _truncate(text.strip())
    if not t:
        return neutral

    key = headline_cache_key(t)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            outputs = client.text_classification(t, model=model_id)
            score = positive_score_from_classification(outputs)
            if cache is not None:
                cache.set(key, score)
                cache.save()
            return score
        except (InferenceTimeoutError, HfHubHTTPError) as e:
            last_exc = e
            if _should_retry_inference(e) and attempt < max_retries - 1:
                logger.warning(
                    "FinBERT inference retry %d/%d after %s; sleeping %.1fs",
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            break
        except Exception as e:
            last_exc = e
            logger.exception("FinBERT unexpected error")
            break

    logger.warning(
        "FinBERT failed for headline (using neutral %.3f): %s",
        neutral,
        last_exc,
    )
    # Do not cache failure fallback — avoids locking in 0.5 after a bad run (e.g. router 400).
    return neutral


def aggregate_sentiment(
    headlines: list[str],
    *,
    client: Any | None = None,
    cfg: dict[str, Any] | None = None,
    cache: SentimentCache | None = None,
) -> float:
    """Mean positive score over ``headlines``; empty → ``neutral_score`` from config."""
    cfg = cfg if cfg is not None else load_config()
    neutral = float(cfg.get("sentiment", {}).get("neutral_score", 0.5))
    cleaned = [h.strip() for h in headlines if isinstance(h, str) and h.strip()]
    if not cleaned:
        return neutral

    if client is None:
        client = build_inference_client(cfg)
    if cache is None:
        cache_path = resolve_path("sentiment_cache", cfg)
        cache = SentimentCache(cache_path)
        cache.load()

    scores = [score_headline(h, client=client, cache=cache, cfg=cfg) for h in cleaned]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score headline(s) with FinBERT (HF Inference).")
    p.add_argument(
        "--headline",
        nargs="+",
        help="One or more headline strings (joined with space if multiple words split)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    if not args.headline:
        print("Usage: python -m src.sentiment --headline \"Your headline text\"")
        raise SystemExit(2)
    text = " ".join(args.headline)
    cfg = load_config()
    client = build_inference_client(cfg)
    cache_path = resolve_path("sentiment_cache", cfg)
    cache = SentimentCache(cache_path)
    cache.load()
    s = score_headline(text, client=client, cache=cache, cfg=cfg)
    print(f"positive_score={s:.6f}")


if __name__ == "__main__":
    main()
