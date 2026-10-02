import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct, type RiskReport } from "../api";
import { Badge, Panel, RiskBar } from "../components/ui";

const MODELS = [
  { id: "propagation", label: "Propagation (C × P)" },
  { id: "multiplicative", label: "Multiplicative (T × V × C × P)" },
  { id: "severity_only", label: "Severity only (no graph)" },
];

export default function RiskAnalysis({ version }: { version: number }) {
  const [model, setModel] = useState("propagation");
  const [rep, setRep] = useState<RiskReport | null>(null);

  useEffect(() => { api.risk(model).then(setRep); }, [model, version]);
  if (!rep) return <div className="text-slate-400">Loading…</div>;
  const top = rep.assets.slice(0, 12).map((a) => ({ ...a, riskPct: +(a.risk * 100).toFixed(1) }));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h1 className="text-2xl font-semibold">Risk Analysis</h1>
        <select className="input w-72" value={model} onChange={(e) => setModel(e.target.value)}>
          {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Total network risk">
          <RiskBar value={rep.total_risk} label={rep.model} />
          <div className="mt-3 text-sm text-slate-400">Critical assets at risk: <span className="text-rose-300">{rep.critical_at_risk}</span></div>
        </Panel>
        <Panel title="Asset risk (top 12)" className="lg:col-span-2">
          <div className="h-56">
            <ResponsiveContainer>
              <BarChart data={top}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="asset_id" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" unit="%" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                <Bar dataKey="riskPct" name="risk %" fill="#f43f5e" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Asset risk register">
          <table className="data">
            <thead><tr><th>Asset</th><th>Threat</th><th>Vuln.</th><th>Crit.</th><th>P(comp.)</th><th>Risk</th><th>Level</th></tr></thead>
            <tbody>
              {rep.assets.map((a) => (
                <tr key={a.asset_id}>
                  <td className="font-mono">{a.asset_id}</td>
                  <td className="font-mono">{pct(a.threat, 0)}</td>
                  <td className="font-mono">{pct(a.vulnerability, 0)}</td>
                  <td className="font-mono">{a.criticality.toFixed(2)}</td>
                  <td className="font-mono">{pct(a.compromise_p, 0)}</td>
                  <td className="font-mono">{pct(a.risk)}</td>
                  <td><Badge level={a.level} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <Panel title="Most likely attack paths to critical assets">
          <ul className="space-y-2">
            {rep.paths.map((p, i) => (
              <li key={i} className="rounded-md border border-slate-800 p-2">
                <div className="flex justify-between text-xs text-slate-400"><span>path #{i + 1}</span><span className="font-mono">{pct(p.probability, 2)}</span></div>
                <div className="mt-1 font-mono text-sm">{p.nodes.join(" → ")}</div>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
