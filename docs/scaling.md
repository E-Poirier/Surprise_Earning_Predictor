# Scaling ingestion and the ticker universe

## Finnhub rate limits

- Configure `finnhub.rate_limit_sleep_seconds` in [config/config.yaml](../config/config.yaml) so each Finnhub call is followed by a pause that stays under your plan (free tier is often on the order of 60 calls per minute).
- Ingestion serializes **Finnhub** requests when you use `--jobs` greater than 1: a shared lock ensures only one Finnhub request runs at a time, while Yahoo Finance (prices, earnings backfill, calendar) can overlap across workers.

## Parallel ingestion (`--jobs`)

Run:

```bash
python -m src.ingestion --jobs 4
```

- `jobs=1` (default) is sequential; behavior matches older versions.
- `jobs>1` uses a thread pool. Finnhub calls remain globally serialized; yfinance work can run concurrently. Use this when the bottleneck is network latency to Yahoo, not Finnhub.
- Do not set `jobs` so high that CPU or memory on the host becomes the limit; start with 2–4.

## Larger universes

- Add symbols to [config/tickers.py](../config/tickers.py). Full ingestion time grows roughly linearly with the number of tickers (Finnhub is still one-at-a-time under the lock when using parallel jobs).
- Consider a paid Finnhub tier for higher throughput or richer endpoints if you outgrow the free cap.
- After changing limits in `config.yaml` (`ingestion.company_earnings_limit`, `yfinance_earnings_calendar_limit`), re-run ingestion so processed parquets include deeper history. **Note:** Yahoo’s `get_earnings_dates` is capped at **100** by yfinance; values above 100 are clamped (previously caused silent failure and “4 Finnhub → 4” merges).

## Predictable tickers vs configured universe

- `GET /api/tickers` returns the full configured list.
- `GET /api/tickers/predictable` returns symbols that currently pass the same point-in-time feature checks as prediction (no model call). Use `?live=true` to refresh Finnhub per symbol first (slow for large universes); default is fast and uses on-disk data plus the Yahoo calendar merge.

## Failure reason codes (422)

When prediction cannot build a feature row, the API returns `insufficient_history` with `reason` and `detail`. Codes are defined in [src/errors.py](../src/errors.py) (for example `short_history`, `no_upcoming_quarter`, `price_features`).
