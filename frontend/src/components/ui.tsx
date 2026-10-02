import type { ReactNode } from "react";
import { pct, severityColor } from "../api";

export function Kpi({ label, value, hint, tone = "default" }: {
  label: string; value: ReactNode; hint?: string; tone?: "default" | "danger" | "ok" | "warn";
}) {
  const color = { default: "text-slate-100", danger: "text-rose-400", ok: "text-emerald-400", warn: "text-amber-300" }[tone];
  return (
    <div className="panel">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-1 text-3xl font-semibold ${color}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export function Panel({ title, actions, children, className = "" }: {
  title?: string; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">{title}</h2>}
          <div className="flex gap-2">{actions}</div>
        </div>
      )}
      {children}
    </section>
  );
}

export function RiskBar({ value, label }: { value: number; label?: string }) {
  const color = value >= 0.6 ? "bg-rose-500" : value >= 0.3 ? "bg-amber-400" : "bg-emerald-500";
  return (
    <div>
      {label && (
        <div className="mb-1 flex justify-between text-xs text-slate-400">
          <span>{label}</span>
          <span className="font-mono text-slate-200">{pct(value)}</span>
        </div>
      )}
      <div className="h-3 w-full overflow-hidden rounded bg-slate-800">
        <div className={`h-full ${color} transition-all duration-700`} style={{ width: `${Math.min(100, value * 100)}%` }} />
      </div>
    </div>
  );
}

export function Badge({ level }: { level: string }) {
  return <span className={`rounded px-2 py-0.5 text-xs font-semibold ${severityColor[level] ?? "text-slate-300"}`}>{level}</span>;
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="rounded-md border border-rose-800 bg-rose-950/50 px-3 py-2 text-sm text-rose-300">{error}</div>;
}

export function Spinner({ label = "Working" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-cyan-300">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-cyan-300 border-t-transparent" />
      {label}…
    </div>
  );
}
