import { useCallback, useEffect, useState } from "react";
import { api, type DecisionReport, type Overview as OverviewT } from "./api";
import Overview from "./pages/Overview";
import Threats from "./pages/Threats";
import AttackGraph from "./pages/AttackGraph";
import RiskAnalysis from "./pages/RiskAnalysis";
import Optimizer from "./pages/Optimizer";
import Recommendation from "./pages/Recommendation";
import Benchmark from "./pages/Benchmark";

const TABS = [
  { id: "overview", label: "SOC Overview", icon: "◉" },
  { id: "threats", label: "Threat Detection", icon: "⚠" },
  { id: "graph", label: "Attack Graph", icon: "⛓" },
  { id: "risk", label: "Risk Analysis", icon: "▤" },
  { id: "optimizer", label: "Quantum Optimizer", icon: "⚛" },
  { id: "recommendation", label: "Defense Plan", icon: "🛡" },
  { id: "benchmark", label: "Solver Benchmark", icon: "⚖" },
] as const;
type Tab = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [overview, setOverview] = useState<OverviewT | null>(null);
  const [report, setReport] = useState<DecisionReport | null>(null);
  const [health, setHealth] = useState<{ version: string; qiskit: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0); // bump to make pages refetch

  const refresh = useCallback(async () => {
    try {
      setOverview(await api.overview());
      setVersion((v) => v + 1);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e)));
    refresh();
  }, [refresh]);

  const onOverview = (o: OverviewT) => {
    setOverview(o);
    setVersion((v) => v + 1);
  };

  return (
    <div className="flex min-h-screen">
      <aside className="w-60 shrink-0 border-r border-slate-800 bg-slate-950 p-4">
        <div className="mb-6">
          <div className="text-lg font-bold tracking-tight text-cyan-300">Q-ADAPT</div>
          <div className="text-xs text-slate-500">Quantum-assisted adaptive cyber defense</div>
        </div>
        <nav className="space-y-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm ${
                tab === t.id ? "bg-cyan-900/40 text-cyan-200" : "text-slate-400 hover:bg-slate-900"
              }`}
            >
              <span className="w-4 text-center">{t.icon}</span>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="mt-8 space-y-1 text-xs text-slate-500">
          <div>API {health ? `v${health.version}` : "offline"}</div>
          <div>Qiskit: {health?.qiskit ? "available" : "not installed"}</div>
          <div>Mode: simulation (no quantum advantage claimed)</div>
        </div>
      </aside>
      <main className="flex-1 space-y-4 overflow-x-hidden p-6">
        {error && (
          <div className="rounded-md border border-rose-800 bg-rose-950/50 px-3 py-2 text-sm text-rose-300">
            {error} — is the API running? (<code>qadapt serve</code>)
          </div>
        )}
        {tab === "overview" && <Overview overview={overview} onOverview={onOverview} onNavigate={(t) => setTab(t as Tab)} />}
        {tab === "threats" && <Threats version={version} onChange={refresh} />}
        {tab === "graph" && <AttackGraph version={version} />}
        {tab === "risk" && <RiskAnalysis version={version} />}
        {tab === "optimizer" && (
          <Optimizer report={report} onReport={(r) => { setReport(r); refresh(); }} onNavigate={() => setTab("recommendation")} />
        )}
        {tab === "recommendation" && (
          <Recommendation report={report} overview={overview} onOverview={onOverview} onNavigate={() => setTab("optimizer")} />
        )}
        {tab === "benchmark" && <Benchmark />}
      </main>
    </div>
  );
}
