import { useEffect, useRef, useState } from "react";
import cytoscape, { type Core } from "cytoscape";
import { api, pct } from "../api";
import { Panel } from "../components/ui";

const ZONE_COL: Record<string, number> = { external: 0, perimeter: 1, dmz: 2, app: 3, corp: 3, data: 4, mgmt: 5 };

function riskColor(p: number) {
  if (p >= 0.6) return "#f43f5e";
  if (p >= 0.3) return "#f59e0b";
  return "#10b981";
}

export default function AttackGraph({ version }: { version: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const [selected, setSelected] = useState<any>(null);

  useEffect(() => {
    let cancelled = false;
    api.graph().then((g) => {
      if (cancelled || !ref.current) return;
      // deterministic tiered layout: columns = zones, rows = order within zone
      const rows: Record<number, number> = {};
      const nodes = g.nodes.map((n) => {
        const col = ZONE_COL[n.data.zone] ?? 3;
        const row = (rows[col] = (rows[col] ?? -1) + 1);
        return { ...n, position: { x: 60 + col * 190, y: 50 + row * 70 + (col % 2) * 25 } };
      });
      cy.current?.destroy();
      cy.current = cytoscape({
        container: ref.current,
        elements: [...nodes, ...g.edges],
        layout: { name: "preset" },
        wheelSensitivity: 0.2,
        style: [
          {
            selector: "node",
            style: {
              label: "data(id)",
              color: "#e2e8f0",
              "font-size": 10,
              "text-valign": "bottom",
              "text-margin-y": 4,
              "background-color": (n: any) => riskColor(n.data("compromise_p")),
              width: (n: any) => 18 + 26 * n.data("criticality"),
              height: (n: any) => 18 + 26 * n.data("criticality"),
              "border-width": (n: any) => (n.data("compromised") ? 4 : n.data("threat") > 0 ? 2 : 0),
              "border-color": "#fde047",
            },
          },
          { selector: 'node[type = "attacker"]', style: { "background-color": "#a855f7", shape: "diamond", width: 34, height: 34 } },
          {
            selector: "edge",
            style: {
              width: (e: any) => 1 + 4 * e.data("p"),
              "line-color": "#475569",
              "target-arrow-color": "#475569",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              opacity: 0.8,
            },
          },
          { selector: "edge[?on_path]", style: { "line-color": "#f43f5e", "target-arrow-color": "#f43f5e", width: 5 } },
          { selector: "edge[defended > 0.3]", style: { "line-style": "dashed", "line-color": "#22d3ee", "target-arrow-color": "#22d3ee" } },
        ],
      });
      cy.current.on("tap", "node, edge", (evt) => setSelected(evt.target.data()));
    });
    return () => { cancelled = true; };
  }, [version]);

  useEffect(() => () => cy.current?.destroy(), []);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Attack Graph</h1>
      <div className="grid gap-4 lg:grid-cols-4">
        <Panel className="lg:col-span-3">
          <div ref={ref} className="h-[620px] w-full rounded-lg bg-slate-950" />
          <div className="mt-2 flex flex-wrap gap-4 text-xs text-slate-400">
            <span><span className="text-purple-400">◆</span> attacker</span>
            <span><span className="text-rose-500">●</span> P(compromise) ≥ 60%</span>
            <span><span className="text-amber-500">●</span> 30–60%</span>
            <span><span className="text-emerald-500">●</span> &lt; 30%</span>
            <span>node size = criticality · yellow ring = ML evidence / compromised</span>
            <span><span className="text-rose-500">━</span> most likely path</span>
            <span><span className="text-cyan-400">┅</span> defended edge</span>
          </div>
        </Panel>
        <Panel title="Details">
          {!selected ? (
            <div className="text-sm text-slate-500">Click a node or edge.</div>
          ) : selected.source ? (
            <dl className="space-y-1 text-sm">
              <dt className="text-slate-400">Edge</dt><dd className="font-mono">{selected.source} → {selected.target}</dd>
              <dt className="text-slate-400">Kind</dt><dd>{selected.kind}</dd>
              <dt className="text-slate-400">Exploit probability</dt><dd className="font-mono">{pct(selected.p)}</dd>
              <dt className="text-slate-400">Reduced by defenses</dt><dd className="font-mono">{pct(selected.defended)}</dd>
            </dl>
          ) : (
            <dl className="space-y-1 text-sm">
              <dt className="text-slate-400">Asset</dt><dd className="font-mono">{selected.id} — {selected.label}</dd>
              <dt className="text-slate-400">Type / zone</dt><dd>{selected.type} / {selected.zone}</dd>
              <dt className="text-slate-400">Criticality</dt><dd className="font-mono">{selected.criticality}</dd>
              <dt className="text-slate-400">ML threat evidence</dt><dd className="font-mono">{pct(selected.threat)}</dd>
              <dt className="text-slate-400">P(compromise)</dt><dd className="font-mono">{pct(selected.compromise_p)}</dd>
              <dt className="text-slate-400">Risk</dt><dd className="font-mono">{pct(selected.risk)}</dd>
              <dt className="text-slate-400">CVEs</dt><dd className="font-mono text-xs">{selected.cves?.join(", ") || "none known"}</dd>
            </dl>
          )}
        </Panel>
      </div>
    </div>
  );
}
