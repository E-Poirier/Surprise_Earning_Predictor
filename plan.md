# Earnings Surprise Predictor — Build Plan

Single reference for implementing this hackathon project (FastAPI + React/Vite + Docker). Stack: **Finnhub** + **yfinance** + **FinBERT (Hugging Face Inference)** + **XGBoost**.

---

## 1. Overview

Predict whether a public company will **beat**, **miss**, or come **in-line** with Wall Street EPS consensus for an **upcoming** earnings release.

- **Batch inference only** (not real-time trading).
- **Not financial advice.**
- **Universe:** fixed ~50 large-cap tickers (not full S&P 500) to respect Finnhub rate limits.

---

## 2. Problem Definition

### 2.1 Target variable (three-class, per company-quarter)

| Label      | Rule                                      |
|-----------|--------------------------------------------|
| `BEAT`    | actual EPS > estimate by **> 2%**         |
| `MISS`    | actual EPS < estimate by **> 2%**         |
| `IN_LINE` | within **±2%** of estimate                 |

Threshold is configurable in `config/config.yaml`.

### 2.2 Prediction scope

Given a **ticker** and an **upcoming earnings date**, predict surprise direction using only information available **before** the release (**point-in-time** — no lookahead).

### 2.3 Out of scope

- Trading signals
- Real-time streaming
- Training on the full S&P 500

---

## 3. Ticker Universe

Hard-code in `config/tickers.py`. Example (~50 names):

```python
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ORCL", "ADBE",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "BLK", "SCHW",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY",
    # Consumer
    "AMZN", "WMT", "COST", "TGT", "MCD", "NKE", "SBUX",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials
    "BA", "CAT", "GE", "HON", "MMM",
    # Telecom / Media
    "DIS", "NFLX", "CMCSA", "T", "VZ",
    # Other
    "TSLA", "BRK-B", "V", "MA", "PG", "KO", "PEP",
]
```

**Note:** For yfinance, normalize symbols if needed (e.g. `BRK-B` vs `BRK.B`) when history is empty.

---

## 4. Data Sources

### 4.1 Finnhub (primary)

- **Rate limit:** ~60 calls/minute — use ~`time.sleep(1.1)` between calls in ingestion loops.
- **API key:** `FINNHUB_API_KEY` (never commit).
- **Endpoints (Python client):**
  - `company_earnings(symbol, limit=40)` — historical EPS actuals, estimates, surprise %
  - `company_eps_estimates(symbol, freq="quarterly")` — estimate trends (validate fields for revision features)
  - `company_news(symbol, _from=..., to=...)` — headlines for FinBERT
- **Package:** `pip install finnhub-python`

### 4.2 yfinance (supplementary)

- Daily prices, historical vol proxy, sector/industry.
- **Fragile** (scrapes Yahoo) — wrap in try/except; acceptable for a hackathon.
- **Do not** use yfinance for earnings numbers — use Finnhub.

### 4.3 Hugging Face — FinBERT (`yiyanghkust/finbert-tone`)

- Use **`InferenceClient`** with **`provider="hf-inference"`** via `huggingface_hub` (no local GPU).
- **API key:** `HF_API_KEY` — fine-grained token with **“Make calls to Inference Providers”**.
- **Caching:** key by `md5(headline)` in `data/sentiment_cache.json`; avoid repeat calls (HF has usage limits).
- **Cold start:** on `503` / `estimated_time`, retry after ~20s.
- **Aggregation:** headlines from **14 days** before earnings date **through D−1**; feature = **mean Positive score**; if no headlines → **0.5**.

Verify the exact client method name (`text_classification` vs API route) against current `huggingface_hub` docs at implementation time.

---

## 5. Feature Engineering

All features are **per company-quarter**, known **before** earnings date `D`.

### 5.1 Tabular features

| Feature                 | Description                                              | Source        |
|-------------------------|----------------------------------------------------------|---------------|
| `eps_revision_30d`      | Consensus EPS change in last 30d before earnings       | Finnhub est.  |
| `eps_revision_60d`      | Consensus EPS change in last 60d before earnings         | Finnhub est.  |
| `beat_rate_4q`          | Fraction of last 4 quarters beating                      | Finnhub       |
| `beat_rate_8q`          | Fraction of last 8 quarters beating                      | Finnhub       |
| `surprise_magnitude_avg`| Mean \|surprise %\| over last 4 quarters                 | Finnhub       |
| `momentum_30d`          | Stock return 30d before earnings                         | yfinance      |
| `momentum_60d`          | Stock return 60d before earnings                         | yfinance      |
| `hist_vol_30d`          | 30d historical volatility (std of log returns)           | yfinance      |
| `sector`                | One-hot sector                                           | yfinance info |
| `quarter`               | Q1–Q4 seasonality                                        | derived       |

**Risk:** `eps_revision_*` needs historical estimate snapshots. **Spike on one ticker** in Phase 1; if unavailable on free tier, **drop these two** and ship beat-rates + momentum + sentiment + sector + quarter.

### 5.2 NLP feature

| Feature            | Description                                                |
|--------------------|------------------------------------------------------------|
| `sentiment_score`  | Mean FinBERT positive score over pre-earnings headlines   |

### 5.3 Point-in-time rules

For earnings date `D`:

- Prices and returns: use data **through D−1** only.
- News: **through D−1** only.
- Estimates: only snapshots **before D**.
- **Never** put actual EPS into features.

### 5.4 Output

- `data/features/features.parquet` — one row per company-quarter: features + `target` label.

---

## 6. Train / Validation / Test

**Time-based split only — never shuffle.**

| Split        | Range              | Use                                      |
|-------------|--------------------|------------------------------------------|
| Train       | Before 2022-Q1     | Fit model                                |
| Validation  | 2022-Q1 – 2023-Q4  | Hyperparameters / early stopping         |
| Test        | 2024-Q1 onward     | **Final metrics once** — no tuning       |

Class imbalance: `BEAT` is often majority. Use **per-sample `sample_weight`** (or equivalent) derived from class counts — **not** `scale_pos_weight` (binary-only in XGBoost).

Report accuracy, per-class precision/recall/F1, confusion matrix.

---

## 7. Model

### 7.1 Algorithm

- **XGBoost** `XGBClassifier`, `objective="multi:softprob"`, `num_class=3`.

### 7.2 Label encoding (fixed everywhere)

Define once (config or `train.py`), use in training **and** API responses:

- Example: `BEAT → 0`, `IN_LINE → 1`, `MISS → 2` (adjust if you prefer, but stay consistent with `probabilities` JSON keys).

### 7.3 Hyperparameters (starting point)

```yaml
n_estimators: 300
max_depth: 4
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
# Imbalance: compute sample_weight in train.py from class frequencies — do not use scale_pos_weight for multiclass
```

### 7.4 Metrics

Accuracy; per-class P/R/F1; confusion matrix; feature importance (optional: SHAP if time).

---

## 8. Repository Layout

```
Surprise_Earning_Predictor/
├── plan.md                  # this file
├── config/
│   ├── config.yaml          # paths, thresholds, model params, split dates
│   └── tickers.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/            # features.parquet
│   └── sentiment_cache.json
├── models/                  # trained artifacts (gitignored)
├── src/
│   ├── ingestion.py
│   ├── sentiment.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── api/
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── TickerSearch.jsx
│   │       ├── PredictionCard.jsx
│   │       └── HistoryTable.jsx
│   ├── index.html
│   └── vite.config.js       # proxy /api → backend :8000
├── notebooks/
│   └── eda.ipynb
├── tests/
│   ├── test_features.py     # point-in-time / no lookahead (required)
│   ├── test_sentiment.py
│   └── test_api.py
├── .env.example             # FINNHUB_API_KEY, HF_API_KEY, API_KEY
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
└── README.md
```

**Env paths:** Prefer **root `.env`** and `env_file: .env` in Docker Compose (document in README).

---

## 9. API (FastAPI)

### 9.1 Auth

Header: `x-api-key` → must match `API_KEY`.

### 9.2 Endpoints

**`GET /api/health`** — `{ "status": "ok", "model_version": "v1" }`

**`GET /api/tickers`** — supported ticker list.

**`POST /api/predict`**

Request:

```json
{ "ticker": "AAPL" }
```

Response (shape):

```json
{
  "ticker": "AAPL",
  "prediction": "BEAT",
  "confidence": 0.71,
  "probabilities": { "BEAT": 0.71, "IN_LINE": 0.19, "MISS": 0.10 },
  "top_features": [
    { "feature": "beat_rate_8q", "value": 0.875, "direction": "positive" },
    { "feature": "sentiment_score", "value": 0.63, "direction": "positive" },
    { "feature": "eps_revision_30d", "value": -0.02, "direction": "negative" }
  ],
  "last_quarters": [
    {
      "quarter": "2024-Q3",
      "estimate": 1.43,
      "actual": 1.51,
      "surprise_pct": 5.6,
      "label": "BEAT"
    }
  ]
}
```

### 9.3 Errors

- Unknown ticker → **404** `{ "error": "ticker_not_supported" }`
- Insufficient history (for example fewer than 8 quarters) → **422** `{ "error": "insufficient_history" }`

---

## 10. Frontend

- Vite + React; **Tailwind** utilities only.
- Components: `TickerSearch`, `PredictionCard`, `HistoryTable`.
- Colors: BEAT = green, MISS = red, IN_LINE = gray; loading spinner; friendly errors.

---

## 11. Docker

- Services: **backend** (FastAPI, port 8000), **frontend** (Vite dev server, port 5173).
- Mount `./data` and `./models` so artifacts persist.
- **Workflow:** run `ingestion → features → train` **outside** Docker first (or one-off job), then `docker compose up` for demo.

---

## 12. Environment Variables

```bash
# .env (see .env.example)
FINNHUB_API_KEY=...
HF_API_KEY=...
API_KEY=...          # FastAPI x-api-key
```

---

## 13. Implementation Checklist (master order)

Use this sequence when building; skip §15 items if time is short.

| Phase | Task | Checkpoint |
|-------|------|------------|
| **0** | Create tree, `requirements*.txt`, `.gitignore`, `.env.example`, `config/tickers.py`, `config/config.yaml` | Config imports; ~50 tickers |
| **1** | `src/ingestion.py` — Finnhub + yfinance, rate limits, persisted raw/processed data | One-ticker spike; revision fields validated or scoped down |
| **2** | `src/sentiment.py` — HF client, cache, 503 retry, 14d aggregation | Second run hits cache only |
| **3** | `src/features.py` — PIT assembly → `data/features/features.parquet` | `test_features.py` guards lookahead |
| **4** | `src/train.py` — time split, multiclass XGBoost + **sample weights**, save to `models/` | Val metrics; test evaluated once |
| **5** | `api/` — health, tickers, predict + auth; `tests/test_api.py` | `curl` matches contract |
| **6** | `frontend/` — search, card, table, proxy | End-to-end in browser |
| **7** | `Dockerfile`, `docker-compose.yml`, `README.md` | `docker compose up` works with mounts |
| **8** (opt) | `notebooks/eda.ipynb`, SHAP | Nice-to-have |

**If blocked on estimate revisions:** drop `eps_revision_30d` / `eps_revision_60d` first; keep the rest.

---

## 14. Design Decisions (summary)

| Decision | Why |
|----------|-----|
| ~50 tickers | Finnhub free tier + time |
| HF Inference (not local) | No GPU; fast to integrate |
| FinBERT feature + XGBoost | Sentiment + tabular strengths |
| Time split | Prevents leakage |
| ±2% bands | Reduces noise labels |

---

## 15. De-scope if time is short

- SHAP
- SEC EDGAR
- Batch predict endpoint
- CI/CD
- Full test suite — **keep `test_features.py`**

---

## 16. Sentiment reference (implementation sketch)

Logic to implement in `src/sentiment.py` (adjust API calls to match current SDK):

- Load/save JSON cache at `data/sentiment_cache.json`.
- Hash headline → cache key.
- On repeated failure after retries, return **0.5** and log (do not crash batch ingestion).
- `aggregate_sentiment(headlines)` → mean positive score; empty list → **0.5**.

---

*This file is the canonical build spec for the Surprise Earning Predictor repo.*
