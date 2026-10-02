import { useEffect, useState } from "react";
import { api, pct, type Threat } from "../api";
import { Badge, ErrorBox, Panel, Spinner } from "../components/ui";

const CATEGORIES = ["DDOS", "DOS", "BRUTE_FORCE", "WEB_ATTACK", "BOTNET", "INFILTRATION", "RECON", "EXPLOIT"];

export default function Threats({ version, onChange }: { version: number; onChange: () => void }) {
  const [threats, setThreats] = useState<Threat[]>([]);
  const [assets, setAssets] = useState<string[]>([]);
  const [host, setHost] = useState("");
  const [cat, setCat] = useState("INFILTRATION");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastDetect, setLastDetect] = useState<string | null>(null);

  useEffect(() => {
    api.threats().then(setThreats).catch((e) => setError(String(e)));
    api.graph().then((g) => {
      const ids = g.nodes.map((n) => n.data.id).filter((i: string) => i !== "ATTACKER");
      setAssets(ids);
      setHost((h) => h || ids[0]);
    });
  }, [version]);

  const detect = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.detect({ [host]: cat }, Math.floor(Math.random() * 1e6));
      setLastDetect(`${r.model}: ${r.flows} flows analysed, ${r.detections.length} host(s) flagged`);
      onChange();
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Threat Detection</h1>
      <Panel title="Simulate live traffic through the ML threat engine">
        <p className="mb-3 text-sm text-slate-400">
          Generates CICFlowMeter-style flows (attack traffic towards the selected host) and scores them with the trained
          classifier. Host threat = top-k flow maliciousness × evidence factor.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-400">Target host
            <select className="input mt-1 w-40" value={host} onChange={(e) => setHost(e.target.value)}>
              {assets.map((a) => <option key={a}>{a}</option>)}
            </select>
          </label>
          <label className="text-xs text-slate-400">Attack category
            <select className="input mt-1 w-44" value={cat} onChange={(e) => setCat(e.target.value)}>
              {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
            </select>
          </label>
          <button className="btn btn-primary" disabled={busy || !host} onClick={detect}>Run detection</button>
          {busy && <Spinner label="Training / scoring" />}
        </div>
        {lastDetect && <div className="mt-2 text-xs text-cyan-300">{lastDetect}</div>}
        <div className="mt-2"><ErrorBox error={error} /></div>
      </Panel>

      <Panel title={`Active threats (${threats.length})`}>
        <table className="data">
          <thead>
            <tr><th>Asset</th><th>Attack</th><th>Probability</th><th>Confidence</th><th>Events</th><th>Source</th><th>Severity</th><th>State</th></tr>
          </thead>
          <tbody>
            {threats.map((t) => (
              <tr key={t.host_id}>
                <td><span className="font-mono">{t.host_id}</span> <span className="text-xs text-slate-500">{t.asset_name}</span></td>
                <td>{t.attack_type}</td>
                <td className="font-mono">{pct(t.probability)}</td>
                <td className="font-mono">{pct(t.confidence)}</td>
                <td>{t.n_events}</td>
                <td className="font-mono text-xs">{t.source || "—"}</td>
                <td><Badge level={t.severity} /></td>
                <td>{t.compromised ? <span className="text-rose-400">compromised</span> : <span className="text-slate-500">suspected</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
