/**
 * Cumulative SHAP waterfall in margin space: expected value, then each feature, ending at model output.
 */
function fmt(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  return n.toFixed(3);
}

export default function ShapWaterfall({ explanation }) {
  if (!explanation || !Array.isArray(explanation.rows)) return null;

  const { base_value: base, model_output: out, explained_class: ec, rows } = explanation;

  const steps = [];
  steps.push({
    key: "base",
    label: "E[f(x)]",
    start: 0,
    end: base,
    delta: base,
    isBase: true,
  });
  let cum = base;
  for (const r of rows) {
    const start = cum;
    const end = cum + r.shap;
    steps.push({
      key: r.feature,
      label: r.feature,
      start,
      end,
      delta: r.shap,
      rawValue: r.value,
    });
    cum = end;
  }

  const allEnds = [0, base, out, cum, ...steps.map((s) => s.end)];
  const minV = Math.min(...allEnds);
  const maxV = Math.max(...allEnds);
  const span = maxV - minV || 1;

  const scale = (v) => ((v - minV) / span) * 100;

  return (
    <div className="mt-6 border-t border-slate-700/80 pt-6">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">SHAP explanation</p>
      <p className="mt-1 text-xs leading-relaxed text-slate-500">
        Margin contributions for predicted class <span className="font-mono text-slate-400">{ec}</span> (before
        softmax). Expected margin <span className="tabular-nums text-slate-400">{fmt(base)}</span> → output{" "}
        <span className="tabular-nums text-slate-400">{fmt(out)}</span>.
      </p>

      <div className="mt-4 space-y-2">
        {steps.map((s) => {
          const left = scale(Math.min(s.start, s.end));
          const right = scale(Math.max(s.start, s.end));
          const width = Math.max(right - left, 0.4);
          const positive = s.delta >= 0;
          const fill = s.isBase
            ? "bg-slate-600/90"
            : positive
              ? "bg-emerald-600/85"
              : "bg-red-600/85";

          return (
            <div key={s.key} className="flex min-h-[2rem] items-center gap-2 text-xs sm:gap-3">
              <span className="w-[40%] shrink-0 truncate font-mono text-slate-400 sm:w-[44%]" title={s.label}>
                {s.label}
              </span>
              <div className="relative h-6 flex-1 rounded bg-slate-900/80">
                <div
                  className={`absolute top-1 bottom-1 rounded ${fill} opacity-95`}
                  style={{ left: `${left}%`, width: `${width}%` }}
                />
              </div>
              <span
                className={`w-[22%] shrink-0 text-right tabular-nums sm:w-[18%] ${
                  s.isBase ? "text-slate-400" : positive ? "text-emerald-300" : "text-red-300"
                }`}
              >
                {s.isBase ? fmt(s.delta) : `${s.delta >= 0 ? "+" : ""}${fmt(s.delta)}`}
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-3 text-[11px] leading-relaxed text-slate-600">
        Bars show how each segment moves the cumulative margin from the global expected value through this
        instance’s features. “Other (remaining)” aggregates smaller contributions when many features are present.
      </p>
    </div>
  );
}
