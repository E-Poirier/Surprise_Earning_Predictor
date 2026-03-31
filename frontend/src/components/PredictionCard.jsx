import ShapWaterfall from "./ShapWaterfall.jsx";

const labelStyles = {
  BEAT: {
    ring: "ring-emerald-500/40",
    border: "border-emerald-500/50",
    bg: "bg-emerald-950/50",
    badge: "bg-emerald-600/90 text-white",
    text: "text-emerald-200",
  },
  MISS: {
    ring: "ring-red-500/40",
    border: "border-red-500/50",
    bg: "bg-red-950/50",
    badge: "bg-red-600/90 text-white",
    text: "text-red-200",
  },
  IN_LINE: {
    ring: "ring-slate-500/40",
    border: "border-slate-500/50",
    bg: "bg-slate-900/80",
    badge: "bg-slate-600 text-slate-100",
    text: "text-slate-200",
  },
};

function pct(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

function formatAnchor(iso) {
  if (!iso || typeof iso !== "string") return null;
  const d = new Date(`${iso}T12:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export default function PredictionCard({ data }) {
  if (!data) return null;

  const pred = data.prediction;
  const styles = labelStyles[pred] || labelStyles.IN_LINE;
  const probs = data.probabilities || {};
  const order = ["BEAT", "IN_LINE", "MISS"];
  const fq = data.upcoming_fiscal_quarter;
  const anchorLabel = formatAnchor(data.earnings_anchor_date);

  return (
    <div
      className={`rounded-2xl border ${styles.border} ${styles.bg} p-6 shadow-lg ring-1 ${styles.ring}`}
    >
      {(fq || anchorLabel) && (
        <p className="mb-4 text-xs leading-relaxed text-slate-400">
          {fq && (
            <>
              <span className="font-medium text-slate-300">Upcoming quarter:</span>{" "}
              <span className="font-mono text-slate-200">{fq}</span>
            </>
          )}
          {fq && anchorLabel && <span className="text-slate-600"> · </span>}
          {anchorLabel && (
            <>
              <span className="font-medium text-slate-300">Fiscal period end (D):</span>{" "}
              <span className="text-slate-300">{anchorLabel}</span>
            </>
          )}
        </p>
      )}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Prediction</p>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <span className={`inline-flex rounded-full px-3 py-1 text-sm font-semibold ${styles.badge}`}>
              {pred}
            </span>
            <span className={`text-lg font-semibold ${styles.text}`}>{data.ticker}</span>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Confidence</p>
          <p className="mt-1 text-2xl font-bold tabular-nums text-slate-50">
            {pct(data.confidence)}
          </p>
        </div>
      </div>

      <div className="mt-6 space-y-3">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Class probabilities</p>
        {order.map((key) => {
          const v = typeof probs[key] === "number" ? probs[key] : 0;
          const bar =
            key === "BEAT"
              ? "bg-emerald-500"
              : key === "MISS"
                ? "bg-red-500"
                : "bg-slate-400";
          return (
            <div key={key}>
              <div className="mb-1 flex justify-between text-xs text-slate-400">
                <span>{key}</span>
                <span className="tabular-nums text-slate-300">{pct(v)}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full ${bar} transition-all`}
                  style={{ width: `${Math.min(100, Math.max(0, v * 100))}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {(data.shap_explanation ||
        (Array.isArray(data.top_features) && data.top_features.length > 0)) && (
        <details className="group mt-8 border-t border-slate-700/80 pt-2">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-xl border border-slate-600/40 bg-slate-950/50 px-3 py-3 text-left text-sm text-slate-300 outline-none transition hover:border-slate-500/50 hover:bg-slate-900/60 focus-visible:ring-2 focus-visible:ring-slate-500 [&::-webkit-details-marker]:hidden">
            <span>
              <span className="font-medium text-slate-200">Optional: technical breakdown</span>
              <span className="mt-0.5 block text-xs font-normal text-slate-500">
                SHAP explanation and model feature highlights — expand if you want detail
              </span>
            </span>
            <span
              className="shrink-0 text-slate-500 transition group-open:rotate-180"
              aria-hidden
            >
              ▼
            </span>
          </summary>
          <div className="mt-4 space-y-6">
            {data.shap_explanation && (
              <ShapWaterfall explanation={data.shap_explanation} embedded />
            )}
            {Array.isArray(data.top_features) && data.top_features.length > 0 && (
              <div
                className={
                  data.shap_explanation ? "border-t border-slate-700/80 pt-6" : "pt-1"
                }
              >
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  Top features (global importance)
                </p>
                <ul className="mt-3 space-y-2">
                  {data.top_features.map((f) => (
                    <li
                      key={f.feature}
                      className="flex flex-wrap items-baseline justify-between gap-2 rounded-lg bg-slate-950/40 px-3 py-2 text-sm"
                    >
                      <span className="font-mono text-slate-200">{f.feature}</span>
                      <span className="text-slate-400">
                        <span className="tabular-nums text-slate-200">{f.value}</span>
                        <span className="ml-2 text-xs text-slate-500">({f.direction})</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );
}
