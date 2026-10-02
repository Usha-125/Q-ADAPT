import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct, type Overview as OverviewT } from "../api";
import { ErrorBox, Kpi, Panel, RiskBar } from "../components/ui";

export default function Overview({ overview, onOverview, onNavigate }: {
  overview: OverviewT | null; onOverview: (o: OverviewT) => void; onNavigate: (tab: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenario, setScenario] = useState("demo");

  const run = async (fn: () => Promise<OverviewT>) => {
    setBusy(true);
    setError(null);
    try { onOverview(await fn()); } catch (e) { setError(String(e)); } finally { setBusy(false); }
  };

  if (!overview) return <div className="text-slate-400">Loading…</div>;
  const o = overview;
  const riskTone = o.current_risk >= 0.6 ? "danger" : o.current_risk >= 0.3 ? "warn" : "ok";
  const nextStage = o.stage < o.stages.length ? o.stages[o.stage] : null;

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Q-ADAPT Security Center</h1>
          <p className="text-sm text-slate-400">
            Scenario <span className="font-mono text-slate-200">{o.scenario}</span> · {o.assets} assets ·{" "}
            {o.graph.edges} attack edges
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select className="input w-32" value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {["demo", "small", "medium", "large"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <button className="btn btn-ghost" disabled={busy} onClick={() => run(() => api.reset(scenario))}>
            Reset scenario
          </button>
        </div>
      </header>
      <ErrorBox error={error} />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Kpi label="Threats detected" value={o.threats_detected} hint="hosts with ML evidence" tone={o.threats_detected ? "warn" : "ok"} />
        <Kpi label="Critical assets" value={o.critical_assets} hint={`${o.critical_at_risk} at high risk`} tone={o.critical_at_risk ? "danger" : "ok"} />
        <Kpi label="Attack paths" value={o.attack_paths} hint="to critical assets, p ≥ 5%" tone={o.attack_paths > 5 ? "danger" : "default"} />
        <Kpi label="Current risk" value={pct(o.current_risk, 0)} hint="criticality-weighted" tone={riskTone} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Risk timeline (adaptive loop)" className="lg:col-span-2">
          <div className="h-64">
            <ResponsiveContainer>
              <LineChart data={o.timeline.map((t) => ({ ...t, risk: +(t.risk * 100).toFixed(1) }))}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="step" stroke="#64748b" />
                <YAxis stroke="#64748b" domain={[0, 100]} unit="%" />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                  labelFormatter={(s) => o.timeline[Number(s)]?.label ?? s}
                />
                <Line type="stepAfter" dataKey="risk" stroke="#22d3ee" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <ol className="mt-2 space-y-1 text-xs text-slate-400">
            {o.timeline.map((t) => (
              <li key={t.step}><span className="font-mono text-slate-500">#{t.step}</span> {t.label} — risk {pct(t.risk)}</li>
            ))}
          </ol>
        </Panel>

        <div className="space-y-4">
          <Panel title="Network risk">
            <RiskBar value={o.current_risk} label="Total risk" />
            <div className="mt-4 text-xs text-slate-400">Attacker activity prior</div>
            <RiskBar value={o.graph.attacker_activity} />
          </Panel>
          {o.stages.length > 0 && (
            <Panel title="Staged attack scenario">
              <ol className="space-y-1 text-sm">
                {o.stages.map((s, i) => (
                  <li key={s} className={i < o.stage ? "text-rose-300" : "text-slate-500"}>
                    {i < o.stage ? "●" : "○"} T{i + 1}: {s}
                  </li>
                ))}
              </ol>
              <button className="btn btn-ghost mt-3 w-full" disabled={busy || !nextStage} onClick={() => run(api.nextStage)}>
                {nextStage ? `Advance → ${nextStage}` : "Scenario complete"}
              </button>
            </Panel>
          )}
          <Panel title="Applied defenses">
            {o.applied_actions.length === 0 ? (
              <div className="text-sm text-slate-500">None yet.</div>
            ) : (
              <ul className="space-y-1 text-sm">
                {o.applied_actions.map((a) => <li key={a.id} className="text-emerald-300">✓ {a.description}</li>)}
              </ul>
            )}
            <button className="btn btn-primary mt-3 w-full" onClick={() => onNavigate("optimizer")}>
              Optimize defense →
            </button>
          </Panel>
        </div>
      </div>
    </div>
  );
}
