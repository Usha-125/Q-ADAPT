// Typed client for the Q-ADAPT FastAPI backend.

export interface Threat {
  host_id: string;
  asset_name: string;
  attack_type: string;
  probability: number;
  confidence: number;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  n_events: number;
  source: string;
  compromised: boolean;
}

export interface Action {
  id: string;
  type: string;
  target: string;
  description: string;
  cost: number;
  time: number;
  disruption: number;
  effectiveness: number;
}

export interface TimelinePoint {
  step: number;
  label: string;
  risk: number;
  applied: string[];
  compromised: string[];
}

export interface Overview {
  scenario: string;
  threats_detected: number;
  critical_assets: number;
  critical_at_risk: number;
  attack_paths: number;
  current_risk: number;
  assets: number;
  applied_actions: Action[];
  pending_run: number | null;
  stage: number;
  stages: string[];
  timeline: TimelinePoint[];
  graph: { nodes: number; edges: number; density: number; attacker_activity: number };
}

export interface AssetRisk {
  asset_id: string;
  name: string;
  threat: number;
  vulnerability: number;
  criticality: number;
  compromise_p: number;
  risk: number;
  level: string;
}

export interface AttackPath {
  nodes: string[];
  probability: number;
}

export interface RiskReport {
  model: string;
  total_risk: number;
  critical_at_risk: number;
  assets: AssetRisk[];
  paths: AttackPath[];
}

export interface Metrics {
  residual_risk: number;
  risk_reduction: number;
  cost: number;
  time_total: number;
  time_to_effect: number;
  disruption: number;
  n_actions: number;
}

export interface SolveResult {
  solver: string;
  x: number[];
  objective: number;
  feasible: boolean;
  runtime_s: number;
  metrics: Metrics;
  selected: string[];
  info: Record<string, any>;
  gap_to_best?: number | null;
}

export interface Explanation {
  action_id: string;
  description: string;
  selected: boolean;
  reasons: string[];
  factors: Record<string, any>;
}

export interface DecisionReport {
  run_id: number;
  risk_before: number;
  risk_after: number;
  risk_reduction: number;
  paths_before: number;
  paths_after: number;
  n_candidates: number;
  candidates: Action[];
  selected: Action[];
  result: SolveResult;
  explanations: Explanation[];
  qubo: Record<string, any>;
  top_paths: AttackPath[];
}

export interface OptimizeParams {
  solver: string;
  p: number;
  noise: "ideal" | "low" | "medium" | "high";
  backend: "statevector" | "qiskit";
  shots: number;
  max_qubits: number;
  budget: number | null;
  max_time: number | null;
  max_disruption: number | null;
  max_actions: number | null;
  encoding: "unbalanced" | "slack";
  surrogate: "regression" | "expansion";
  weights: { alpha: number; beta: number; gamma: number; delta: number };
  protected_assets: string[];
  seed: number;
}

export const defaultParams: OptimizeParams = {
  solver: "qaoa",
  p: 2,
  noise: "ideal",
  backend: "statevector",
  shots: 1024,
  max_qubits: 12,
  budget: 0.5,
  max_time: null,
  max_disruption: null,
  max_actions: null,
  encoding: "unbalanced",
  surrogate: "regression",
  weights: { alpha: 1.0, beta: 0.15, gamma: 0.1, delta: 0.2 },
  protected_assets: [],
  seed: 0,
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail ? String(body.detail) : `${r.status} ${r.statusText}`);
  }
  return r.json();
}

const post = <T,>(path: string, body: unknown) =>
  req<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => req<{ status: string; version: string; qiskit: boolean; solvers: string[] }>("/health"),
  overview: () => req<Overview>("/overview"),
  reset: (scenario: string, seed = 0) => post<Overview>("/session/reset", { scenario, seed }),
  threats: () => req<Threat[]>("/threats"),
  injectThreat: (t: Partial<Threat>) => post<Threat[]>("/threats", t),
  detect: (attacked: Record<string, string>, seed = 0) =>
    post<{ detections: Threat[]; flows: number; model: string }>("/threats/detect", { attacked, seed }),
  graph: () => req<{ nodes: any[]; edges: any[] }>("/graph"),
  risk: (model = "propagation") => req<RiskReport>(`/risk?model=${model}`),
  optimize: (p: OptimizeParams) => post<DecisionReport>("/optimize", p),
  compare: (p: OptimizeParams, solvers: string[]) =>
    post<{ n_actions: number; base_risk: number; results: SolveResult[]; errors: Record<string, string> }>(
      "/compare", { ...p, solvers }),
  decide: (runId: number, decision: "approve" | "reject", action_ids?: string[]) =>
    post<Overview>(`/runs/${runId}/decision`, { decision, action_ids }),
  nextStage: () => post<Overview>("/adaptive/next-stage", {}),
};

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

export const severityColor: Record<string, string> = {
  LOW: "text-emerald-400 bg-emerald-400/10",
  MEDIUM: "text-amber-300 bg-amber-300/10",
  HIGH: "text-orange-400 bg-orange-400/10",
  CRITICAL: "text-rose-400 bg-rose-400/10",
};
