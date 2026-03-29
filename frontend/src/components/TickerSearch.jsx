import { useEffect, useMemo, useState } from "react";

import { fetchTickers } from "../api.js";

function normalizeQuery(q) {
  return q.trim().toUpperCase();
}

export default function TickerSearch({
  onSubmit,
  disabled,
  loadError,
  onRetryLoad,
}) {
  const [tickers, setTickers] = useState([]);
  const [loadState, setLoadState] = useState("idle");
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoadState("loading");
    fetchTickers()
      .then((list) => {
        if (!cancelled) {
          setTickers(Array.isArray(list) ? list : []);
          setLoadState("ok");
        }
      })
      .catch(() => {
        if (!cancelled) setLoadState("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const q = normalizeQuery(query);
    if (!q) return tickers.slice(0, 12);
    return tickers.filter((t) => t.includes(q)).slice(0, 12);
  }, [tickers, query]);

  const canSubmit = normalizeQuery(query).length > 0 && !disabled;

  function handleSubmit(e) {
    e.preventDefault();
    const t = normalizeQuery(query);
    if (!t || disabled) return;
    onSubmit(t);
    setOpen(false);
  }

  function pickTicker(symbol) {
    setQuery(symbol);
    setOpen(false);
  }

  return (
    <div className="w-full max-w-xl">
      <form onSubmit={handleSubmit} className="relative">
        <label htmlFor="ticker-input" className="mb-2 block text-sm font-medium text-slate-300">
          Ticker
        </label>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <input
              id="ticker-input"
              type="text"
              autoComplete="off"
              placeholder="e.g. AAPL"
              value={query}
              disabled={disabled || loadState === "loading"}
              onChange={(e) => {
                setQuery(e.target.value.toUpperCase());
                setOpen(true);
              }}
              onFocus={() => setOpen(true)}
              onBlur={() => {
                window.setTimeout(() => setOpen(false), 150);
              }}
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 text-slate-100 placeholder:text-slate-500 outline-none ring-emerald-500/0 transition focus:border-emerald-500/60 focus:ring-2 focus:ring-emerald-500/30 disabled:opacity-50"
            />
            {open && filtered.length > 0 && loadState === "ok" && (
              <ul
                className="absolute z-20 mt-1 max-h-56 w-full overflow-auto rounded-lg border border-slate-700 bg-slate-900 py-1 shadow-xl"
                role="listbox"
              >
                {filtered.map((sym) => (
                  <li key={sym}>
                    <button
                      type="button"
                      className="w-full px-3 py-2 text-left text-sm text-slate-200 hover:bg-slate-800"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => pickTicker(sym)}
                    >
                      {sym}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button
            type="submit"
            disabled={!canSubmit}
            className="shrink-0 rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
          >
            Predict
          </button>
        </div>
      </form>

      {loadState === "loading" && (
        <p className="mt-2 text-xs text-slate-500">Loading supported tickers…</p>
      )}
      {loadState === "error" && (
        <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
          <p>{loadError || "Could not load the ticker list. Is the API running on port 8000?"}</p>
          {onRetryLoad && (
            <button
              type="button"
              onClick={onRetryLoad}
              className="mt-2 text-xs font-medium text-amber-200 underline hover:text-white"
            >
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}
