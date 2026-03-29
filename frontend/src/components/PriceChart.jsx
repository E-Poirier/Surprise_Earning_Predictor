import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const RANGES = [
  { id: "1w", label: "1W", days: 7 },
  { id: "1m", label: "1M", days: 30 },
  { id: "3m", label: "3M", days: 90 },
];

function filterByDays(points, calendarDays) {
  if (!points || points.length === 0) return [];
  const times = points.map((p) => new Date(`${p.date}T12:00:00`).getTime());
  const last = Math.max(...times);
  const cutoff = last - calendarDays * 86400000;
  return points.filter((p) => new Date(`${p.date}T12:00:00`).getTime() >= cutoff);
}

function formatTick(iso) {
  const d = new Date(`${iso}T12:00:00`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function PriceTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const v = payload[0]?.value;
  const dateLabel = typeof label === "string" ? formatTick(label) : String(label ?? "");
  return (
    <div className="rounded-lg border border-slate-600 bg-slate-900/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      <p className="font-medium text-slate-300">{dateLabel}</p>
      <p className="mt-1 font-mono text-emerald-300">
        {typeof v === "number" ? v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"}
      </p>
    </div>
  );
}

function gradientId(ticker) {
  return `priceFill-${String(ticker).replace(/[^a-zA-Z0-9_-]/g, "_")}`;
}

function lastCloseDateIso(points) {
  const pts = Array.isArray(points) ? points : [];
  if (pts.length === 0) return null;
  return pts.reduce((best, p) => (p.date > best ? p.date : best), pts[0].date);
}

function formatAsOf(iso) {
  if (!iso) return null;
  const d = new Date(`${iso}T12:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export default function PriceChart({ ticker, priceHistory }) {
  const [rangeId, setRangeId] = useState("1m");
  const days = RANGES.find((r) => r.id === rangeId)?.days ?? 30;
  const gid = gradientId(ticker);
  const asOfIso = useMemo(() => lastCloseDateIso(priceHistory), [priceHistory]);
  const asOfLabel = formatAsOf(asOfIso);

  const data = useMemo(() => {
    const pts = Array.isArray(priceHistory) ? priceHistory : [];
    const sliced = filterByDays(pts, days);
    return sliced.map((p) => ({
      ...p,
      label: p.date,
    }));
  }, [priceHistory, days]);

  if (!priceHistory || priceHistory.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 px-4 py-10 text-center text-sm text-slate-500">
        No recent price history available for {ticker}.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900/80 to-slate-950/90 p-5 shadow-lg ring-1 ring-slate-700/40">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">Price (daily close)</h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Yahoo-adjusted closes from ingestion — not live.
            {asOfLabel && (
              <>
                {" "}
                <span className="text-slate-400">As of last bar:</span>{" "}
                <span className="font-medium text-slate-300">{asOfLabel}</span>
              </>
            )}
          </p>
        </div>
        <div className="flex rounded-lg border border-slate-700 bg-slate-950/80 p-0.5">
          {RANGES.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => setRangeId(r.id)}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                rangeId === r.id
                  ? "bg-emerald-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 h-[260px] w-full min-w-0">
        {data.length === 0 ? (
          <p className="py-12 text-center text-sm text-slate-500">Not enough bars in this window.</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 6" stroke="#334155" opacity={0.6} vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={formatTick}
                stroke="#64748b"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: "#475569" }}
                minTickGap={28}
              />
              <YAxis
                domain={["auto", "auto"]}
                stroke="#64748b"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={56}
                tickFormatter={(v) =>
                  typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v
                }
              />
              <Tooltip content={(props) => <PriceTooltip {...props} />} />
              <Area
                type="monotone"
                dataKey="close"
                name="Close"
                stroke="#34d399"
                strokeWidth={2}
                fill={`url(#${gid})`}
                dot={false}
                activeDot={{ r: 4, fill: "#6ee7b7", stroke: "#064e3b", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
