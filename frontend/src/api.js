/**
 * All requests go through the Vite dev proxy as same-origin `/api/*` → FastAPI :8000.
 */

export async function fetchTickers() {
  const res = await fetch("/api/tickers");
  if (!res.ok) {
    const err = new Error("tickers_fetch_failed");
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function predictTicker(ticker) {
  const key = import.meta.env.VITE_API_KEY;
  if (!key) {
    const err = new Error("missing_api_key");
    err.status = 0;
    throw err;
  }
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": key,
    },
    body: JSON.stringify({ ticker }),
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    data = {};
  }
  if (!res.ok) {
    const msg =
      data.error ||
      (typeof data.detail === "string" ? data.detail : null) ||
      "request_failed";
    const err = new Error(msg);
    err.status = res.status;
    err.body = data;
    throw err;
  }
  return data;
}
