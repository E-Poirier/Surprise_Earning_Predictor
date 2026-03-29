import { useCallback, useState } from "react";

import { predictTicker } from "./api.js";
import HistoryTable from "./components/HistoryTable.jsx";
import PredictionCard from "./components/PredictionCard.jsx";
import TickerSearch from "./components/TickerSearch.jsx";

function LoadingSpinner() {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-sm text-slate-300">
      <span
        className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-400"
        aria-hidden
      />
      <span>Running prediction…</span>
    </div>
  );
}

function friendlyError(err) {
  if (!err) return "Something went wrong.";
  if (err.message === "missing_api_key") {
    return "Add VITE_API_KEY to your project .env (same value as API_KEY) so the browser can authenticate.";
  }
  if (err.message === "tickers_fetch_failed") {
    return "Could not reach the API. Start the backend (uvicorn on port 8000) and keep the Vite proxy enabled.";
  }
  const status = err.status;
  const code = err.body?.error;
  if (status === 401) {
    return "Unauthorized: check that VITE_API_KEY matches the server’s API_KEY.";
  }
  if (status === 404 && code === "ticker_not_supported") {
    return "That ticker is not in the supported universe.";
  }
  if (status === 422 && code === "insufficient_history") {
    return "Not enough earnings history to build a prediction for this ticker.";
  }
  if (status === 503) {
    // API sends FastAPI ``detail`` (e.g. missing file vs missing xgboost) — show it.
    if (err.message && err.message !== "request_failed") {
      return err.message;
    }
    return "Model is not loaded on the server. Train the model and ensure artifacts exist under models/.";
  }
  if (status === 500) {
    if (err.message && err.message !== "request_failed") {
      return err.message;
    }
    return "Server error while predicting. Check API logs.";
  }
  return err.message || "Request failed.";
}

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tickersKey, setTickersKey] = useState(0);

  const onSubmit = useCallback(async (ticker) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictTicker(ticker);
      setResult(data);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 py-10 sm:px-6 lg:px-8">
      <header className="mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-400/90">
          Hackathon demo
        </p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
          Earnings surprise predictor
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
          Pick a supported ticker to get a three-class forecast (beat / in-line / miss vs consensus). Not
          financial advice.
        </p>
      </header>

      <TickerSearch
        key={tickersKey}
        onSubmit={onSubmit}
        disabled={loading}
        onRetryLoad={() => setTickersKey((k) => k + 1)}
      />

      <div className="mt-8 space-y-6">
        {loading && <LoadingSpinner />}
        {error && (
          <div
            role="alert"
            className="rounded-xl border border-red-500/40 bg-red-950/40 px-4 py-3 text-sm text-red-100"
          >
            {friendlyError(error)}
          </div>
        )}
        {!loading && !error && result && (
          <>
            <PredictionCard data={result} />
            <HistoryTable rows={result.last_quarters} />
          </>
        )}
      </div>

      <footer className="mt-auto pt-16 text-center text-xs text-slate-600">
        Batch inference only. Past performance does not guarantee future results.
      </footer>
    </div>
  );
}
