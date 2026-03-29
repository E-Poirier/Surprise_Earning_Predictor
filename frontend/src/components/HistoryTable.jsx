const labelClass = {
  BEAT: "text-emerald-400",
  MISS: "text-red-400",
  IN_LINE: "text-slate-400",
};

function fmtNum(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function fmtPct(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${Number(n).toFixed(2)}%`;
}

export default function HistoryTable({ rows }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-700 bg-slate-900/40 px-4 py-8 text-center text-sm text-slate-500">
        No recent quarters returned for this prediction.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/50 shadow-inner">
      <div className="border-b border-slate-800 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-200">Recent earnings history</h2>
        <p className="mt-0.5 text-xs text-slate-500">Used for context (labels are realized outcomes).</p>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
              <th className="whitespace-nowrap px-4 py-3 font-medium">Quarter</th>
              <th className="whitespace-nowrap px-4 py-3 font-medium">Estimate</th>
              <th className="whitespace-nowrap px-4 py-3 font-medium">Actual</th>
              <th className="whitespace-nowrap px-4 py-3 font-medium">Surprise %</th>
              <th className="whitespace-nowrap px-4 py-3 font-medium">Label</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {rows.map((row) => {
              const lbl = row.label || "—";
              const lc = labelClass[lbl] || "text-slate-300";
              return (
                <tr key={row.quarter} className="hover:bg-slate-800/40">
                  <td className="whitespace-nowrap px-4 py-3 font-mono text-slate-200">{row.quarter}</td>
                  <td className="whitespace-nowrap px-4 py-3 tabular-nums text-slate-300">
                    {fmtNum(row.estimate)}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 tabular-nums text-slate-300">
                    {fmtNum(row.actual)}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 tabular-nums text-slate-300">
                    {fmtPct(row.surprise_pct)}
                  </td>
                  <td className={`whitespace-nowrap px-4 py-3 font-medium ${lc}`}>{lbl}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
