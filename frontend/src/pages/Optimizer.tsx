import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, defaultParams, pct, type DecisionReport, type OptimizeParams } from "../api";
import { ErrorBox, Kpi, Panel, Spinner } from "../components/ui";

function QuboHeatmap({ matrix, names }: { matrix: number[][]; names: string[] }) {
  const max = Math.max(1e-9, ...matrix.flat().map(Math.abs));
  const n = matrix.length;
  const cell = Math.max(10, Math.min(28, Math.floor(420 / n)));
  return (
    <div className="overflow-auto">
      <div className="inline-grid gap-px bg-slate-800" style={{ gridTemplateColumns: `repeat(${n}, ${cell}px)` }}>
        {matrix.flatMap((row, i) =>
          row.map((v, j) => {
            const a = Math.min(1, Math.abs(v) / max) ** 0.5;
            const bg = j < i ? "#020617" : v >= 0 ? `rgba(244,63,94,${a})` : `rgba(34,211,238,${a})`;
            return <div key={`${i}-${j}`} title={`${names[i]} × ${names[j]} = ${v.toFixed(4)}`} style={{ width: cell, height: cell, background: bg }} />;
          }),
        )}
      </div>
      <div className="mt-1 text-xs text-slate-500">
        Upper-triangular Q (diagonal = linear terms). <span className="text-cyan-400">blue</span> = favourable / synergy,{" "}
        <span className="text-rose-400">red</span> = cost, redundancy or penalty.
      </div>
    </div>
  );
}

function NumField({ label, value, onChange, step = 0.05, nullable = false }: {
  label: string; value: number | null; onChange: (v: number | null) => void; step?: number; nullable?: boolean;
}) {
  return (
    <label className="text-xs text-slate-400">
      {label}
      <input
        className="input mt-1"
        type="number"
        step={step}
        value={value ?? ""}
        placeholder={nullable ? "none" : ""}
        onChange={(e) => onChange(e.target.value === "" ? (nullable ? null : 0) : Number(e.target.value))}
      />
    </label>
  );
}

export default function Optimizer({ report, onReport, onNavigate }: {
  report: DecisionReport | null; onReport: (r: DecisionReport) => void; onNavigate: () => void;
}) {
  const [params, setParams] = useState<OptimizeParams>(defaultParams);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = <K extends keyof OptimizeParams>(k: K, v: OptimizeParams[K]) => setParams((p) => ({ ...p, [k]: v }));
  const setW = (k: keyof OptimizeParams["weights"], v: number) => setParams((p) => ({ ...p, weights: { ...p.weights, [k]: v } }));

  const run = async () => {
    setBusy(true);
    setError(null);
    try { onReport(await api.optimize(params)); } catch (e) { setError(String(e)); } finally { setBusy(false); }
  };

  const info = report?.result.info ?? {};
  const isQaoa = report?.result.solver.startsWith("qaoa");
  const conv = (info.convergence ?? []).map((v: number, i: number) => ({ i, v }));

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Quantum Defense Optimizer</h1>
      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="Problem & solver">
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-slate-400">Solver
              <select className="input mt-1" value={params.solver} onChange={(e) => set("solver", e.target.value)}>
                {["qaoa", "exhaustive", "milp", "greedy", "simulated_annealing", "genetic", "score_ranking", "random_sampling"].map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <label className="text-xs text-slate-400">QAOA depth p
              <select className="input mt-1" value={params.p} onChange={(e) => set("p", Number(e.target.value))}>
                {[1, 2, 3, 4].map((p) => <option key={p}>{p}</option>)}
              </select>
            </label>
            <label className="text-xs text-slate-400">Backend
              <select className="input mt-1" value={params.backend} onChange={(e) => set("backend", e.target.value as any)}>
                <option value="statevector">state-vector (fast)</option>
                <option value="qiskit">Qiskit Aer (shots)</option>
              </select>
            </label>
            <label className="text-xs text-slate-400">Noise
              <select className="input mt-1" value={params.noise} onChange={(e) => set("noise", e.target.value as any)}>
                {["ideal", "low", "medium", "high"].map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <NumField label="Max qubits (actions)" value={params.max_qubits} step={1} onChange={(v) => set("max_qubits", v ?? 12)} />
            <NumField label="Shots" value={params.shots} step={256} onChange={(v) => set("shots", v ?? 1024)} />
            <label className="text-xs text-slate-400">Constraint encoding
              <select className="input mt-1" value={params.encoding} onChange={(e) => set("encoding", e.target.value as any)}>
                <option value="unbalanced">unbalanced penalty</option>
                <option value="slack">slack qubits</option>
              </select>
            </label>
            <label className="text-xs text-slate-400">Risk surrogate
              <select className="input mt-1" value={params.surrogate} onChange={(e) => set("surrogate", e.target.value as any)}>
                <option value="regression">regression fit</option>
                <option value="expansion">2nd-order expansion</option>
              </select>
            </label>
          </div>
          <div className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">Constraints</div>
          <div className="mt-2 grid grid-cols-2 gap-3">
            <NumField label="Budget (cost)" value={params.budget} nullable onChange={(v) => set("budget", v)} />
            <NumField label="Max response time" value={params.max_time} nullable onChange={(v) => set("max_time", v)} />
            <NumField label="Max disruption" value={params.max_disruption} nullable onChange={(v) => set("max_disruption", v)} />
            <NumField label="Max actions" value={params.max_actions} step={1} nullable onChange={(v) => set("max_actions", v)} />
          </div>
          <div className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">Objective weights</div>
          <div className="mt-2 grid grid-cols-4 gap-2">
            <NumField label="α risk" value={params.weights.alpha} onChange={(v) => setW("alpha", v ?? 0)} />
            <NumField label="β cost" value={params.weights.beta} onChange={(v) => setW("beta", v ?? 0)} />
            <NumField label="γ time" value={params.weights.gamma} onChange={(v) => setW("gamma", v ?? 0)} />
            <NumField label="δ disr." value={params.weights.delta} onChange={(v) => setW("delta", v ?? 0)} />
          </div>
          <div className="mt-4 flex items-center gap-3">
            <button className="btn btn-primary" onClick={run} disabled={busy}>Run optimization</button>
            {busy && <Spinner label="Optimizing" />}
          </div>
          <div className="mt-2"><ErrorBox error={error} /></div>
        </Panel>

        <div className="space-y-4 xl:col-span-2">
          {!report ? (
            <Panel><div className="text-sm text-slate-500">Configure the problem and run the optimizer.</div></Panel>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <Kpi label="Binary variables" value={report.qubo.n_variables ?? report.n_candidates} hint={`${report.qubo.n_slack ?? 0} slack`} />
                <Kpi label="Algorithm" value={report.result.solver} hint={isQaoa ? `${info.backend}, noise ${info.noise}` : "classical"} />
                <Kpi label="Objective" value={report.result.objective.toFixed(4)} hint={report.result.feasible ? "feasible" : "INFEASIBLE"} tone={report.result.feasible ? "ok" : "danger"} />
                <Kpi label="Runtime" value={`${report.result.runtime_s.toFixed(2)}s`} hint={isQaoa ? `${info.function_evals} circuit evaluations` : ""} />
              </div>
              {isQaoa && (
                <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                  <Kpi label="Circuit depth" value={info.circuit_depth} hint={`p = ${info.p}`} />
                  <Kpi label="Approx. ratio" value={info.approximation_ratio != null ? info.approximation_ratio.toFixed(3) : "n/a"} hint="⟨E⟩ vs QUBO range" />
                  <Kpi label="P(optimal state)" value={info.p_optimal_state != null ? pct(info.p_optimal_state, 2) : "n/a"} hint={info.optimal_state_amplification ? `${info.optimal_state_amplification.toFixed(0)}× uniform` : ""} />
                  <Kpi label="Surrogate fidelity" value={report.qubo.fidelity ? report.qubo.fidelity.spearman.toFixed(3) : "n/a"} hint="Spearman vs true risk" />
                </div>
              )}
              <div className="grid gap-4 lg:grid-cols-2">
                <Panel title="QUBO matrix">
                  {report.qubo.matrix ? <QuboHeatmap matrix={report.qubo.matrix} names={report.qubo.var_names} /> : <div className="text-sm text-slate-500">Solver did not build a QUBO.</div>}
                </Panel>
                <Panel title={isQaoa ? "QAOA convergence (normalised energy)" : "Selected actions"}>
                  {isQaoa ? (
                    <div className="h-64">
                      <ResponsiveContainer>
                        <LineChart data={conv}>
                          <CartesianGrid stroke="#1e293b" />
                          <XAxis dataKey="i" stroke="#64748b" />
                          <YAxis stroke="#64748b" domain={["auto", "auto"]} />
                          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                          <Line dataKey="v" stroke="#a78bfa" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : null}
                  <ul className="mt-2 space-y-1 text-sm">
                    {report.selected.map((a) => <li key={a.id} className="text-emerald-300">✓ {a.description}</li>)}
                  </ul>
                  <button className="btn btn-primary mt-3" onClick={onNavigate}>Review defense plan →</button>
                </Panel>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
