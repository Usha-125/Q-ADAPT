import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, defaultParams, pct, type SolveResult } from "../api";
import { ErrorBox, Panel, Spinner } from "../components/ui";

const ALL = ["qaoa", "exhaustive", "milp", "greedy", "simulated_annealing", "genetic", "score_ranking", "random_sampling"];

export default function Benchmark() {
  const [rows, setRows] = useState<SolveResult[]>([]);
  const [meta, setMeta] = useState<{ n_actions: number; base_risk: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [maxQ, setMaxQ] = useState(12);
  const [budget, setBudget] = useState(0.4);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.compare({ ...defaultParams, max_qubits: maxQ, budget }, ALL);
      setRows(r.results);
      setMeta({ n_actions: r.n_actions, base_risk: r.base_risk });
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Solver Benchmark</h1>
      <Panel title="QAOA vs classical baselines on the current threat state">
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-400">Actions (qubits)
            <input className="input mt-1 w-24" type="number" value={maxQ} min={4} max={18} onChange={(e) => setMaxQ(Number(e.target.value))} />
          </label>
          <label className="text-xs text-slate-400">Budget
            <input className="input mt-1 w-24" type="number" step={0.05} value={budget} onChange={(e) => setBudget(Number(e.target.value))} />
          </label>
          <button className="btn btn-primary" onClick={run} disabled={busy}>Run all solvers</button>
          {busy && <Spinner label="Benchmarking" />}
        </div>
        <div className="mt-2"><ErrorBox error={error} /></div>
        <p className="mt-2 text-xs text-slate-500">
          All solvers are scored on the same ground-truth objective (propagated residual risk + weighted cost, time,
          disruption). "exhaustive" is the true optimum; "random_sampling" uses the same sample budget as QAOA shots.
        </p>
      </Panel>
      {rows.length > 0 && meta && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title={`Results (${meta.n_actions} actions, base risk ${pct(meta.base_risk)})`}>
            <table className="data">
              <thead><tr><th>Solver</th><th>Objective</th><th>Gap</th><th>Risk ↓</th><th>Feasible</th><th>Time</th></tr></thead>
              <tbody>
                {[...rows].sort((a, b) => a.objective - b.objective).map((r) => (
                  <tr key={r.solver}>
                    <td className="font-mono">{r.solver}</td>
                    <td className="font-mono">{r.objective.toFixed(4)}</td>
                    <td className="font-mono">{r.gap_to_best != null ? r.gap_to_best.toFixed(4) : "—"}</td>
                    <td className="font-mono">{pct(r.metrics.risk_reduction)}</td>
                    <td>{r.feasible ? "✓" : <span className="text-rose-400">✗</span>}</td>
                    <td className="font-mono">{r.runtime_s.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
          <Panel title="Objective (lower is better)">
            <div className="h-72">
              <ResponsiveContainer>
                <BarChart data={rows.map((r) => ({ solver: r.solver, objective: +r.objective.toFixed(4) }))}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="solver" stroke="#64748b" fontSize={10} angle={-20} textAnchor="end" height={60} />
                  <YAxis stroke="#64748b" domain={["auto", "auto"]} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                  <Bar dataKey="objective" fill="#22d3ee" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
