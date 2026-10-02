import { useState } from "react";
import { api, pct, type DecisionReport, type Overview } from "../api";
import { ErrorBox, Kpi, Panel, RiskBar } from "../components/ui";

const lvl = (v: number, lo = 0.1, hi = 0.3) => (v < lo ? "Low" : v < hi ? "Moderate" : "High");

export default function Recommendation({ report, overview, onOverview, onNavigate }: {
  report: DecisionReport | null; overview: Overview | null; onOverview: (o: Overview) => void; onNavigate: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decided, setDecided] = useState<Record<string, "approve" | "reject">>({});

  if (!report) {
    return (
      <Panel title="Defense recommendation">
        <div className="text-sm text-slate-500">No recommendation yet.</div>
        <button className="btn btn-primary mt-3" onClick={onNavigate}>Go to optimizer</button>
      </Panel>
    );
  }
  const m = report.result.metrics;
  const pending = overview?.pending_run === report.run_id;

  const decide = async (decision: "approve" | "reject", ids?: string[]) => {
    setBusy(true);
    setError(null);
    try {
      onOverview(await api.decide(report.run_id, decision, ids));
      const all = ids ?? report.selected.map((a) => a.id);
      setDecided((d) => ({ ...d, ...Object.fromEntries(all.map((i) => [i, decision])) }));
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Defense Recommendation</h1>
          <p className="text-sm text-slate-400">Run #{report.run_id} · {report.result.solver} · {report.n_candidates} candidate actions</p>
        </div>
        <div className="flex gap-2">
          <button className="btn btn-ok" disabled={busy || !pending} onClick={() => decide("approve")}>Approve all</button>
          <button className="btn btn-bad" disabled={busy || !pending} onClick={() => decide("reject")}>Reject all</button>
        </div>
      </div>
      <ErrorBox error={error} />
      {!pending && <div className="text-xs text-amber-300">This recommendation is no longer pending — re-run the optimizer for the current state.</div>}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Before vs after">
          <div className="space-y-4">
            <RiskBar value={report.risk_before} label="Total risk — before" />
            <RiskBar value={report.risk_after} label="Total risk — after optimized defense" />
          </div>
        </Panel>
        <div className="grid grid-cols-2 gap-4">
          <Kpi label="Risk reduction" value={pct(m.risk_reduction, 0)} tone="ok" />
          <Kpi label="Attack paths" value={`${report.paths_before} → ${report.paths_after}`} hint="to critical assets, p ≥ 5%" />
          <Kpi label="Defense cost" value={lvl(m.cost, 0.2, 0.5)} hint={`${m.cost.toFixed(2)} (normalised)`} />
          <Kpi label="Business impact" value={lvl(m.disruption, 0.2, 0.5)} hint={`disruption ${m.disruption.toFixed(2)} · time ${m.time_to_effect.toFixed(2)}`} />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Recommended actions — why selected?">
          <ul className="space-y-3">
            {report.explanations.filter((e) => e.selected).map((e) => (
              <li key={e.action_id} className="rounded-lg border border-emerald-900/60 bg-emerald-950/20 p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="font-medium text-emerald-300">✓ {e.description}</div>
                  {decided[e.action_id] ? (
                    <span className="text-xs text-slate-400">{decided[e.action_id]}d</span>
                  ) : (
                    <div className="flex gap-1">
                      <button className="btn btn-ok px-2 py-0.5 text-xs" disabled={busy || !pending} onClick={() => decide("approve", [e.action_id])}>Approve</button>
                      <button className="btn btn-bad px-2 py-0.5 text-xs" disabled={busy || !pending} onClick={() => decide("reject", [e.action_id])}>Reject</button>
                    </div>
                  )}
                </div>
                <ul className="mt-2 list-disc space-y-0.5 pl-5 text-sm text-slate-300">
                  {e.reasons.map((r) => <li key={r}>{r}</li>)}
                </ul>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel title="Not selected — why not?">
          <ul className="space-y-3">
            {report.explanations.filter((e) => !e.selected).map((e) => (
              <li key={e.action_id} className="rounded-lg border border-slate-800 p-3">
                <div className="font-medium text-slate-300">✗ {e.description}</div>
                <ul className="mt-2 list-disc space-y-0.5 pl-5 text-sm text-slate-400">
                  {e.reasons.map((r) => <li key={r}>{r}</li>)}
                </ul>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
      <p className="text-xs text-slate-500">
        Human-in-the-loop: no action is executed automatically. Approved actions are applied to the attack-graph model and the
        risk picture is re-computed; they are not pushed to production security controls.
      </p>
    </div>
  );
}
