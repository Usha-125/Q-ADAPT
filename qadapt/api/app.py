"""Q-ADAPT REST API (FastAPI) backing the SOC dashboard.

Run:  uvicorn qadapt.api.app:app --reload   (or: qadapt serve)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from qadapt import __version__
from qadapt.adaptive_engine.scenarios import STAGED_DEMO
from qadapt.api.schemas import (
    CompareRequest,
    DecisionIn,
    DetectRequest,
    OptimizeRequest,
    ResetRequest,
    ThreatIn,
)
from qadapt.api.session import Pending, SOCSession, to_jsonable
from qadapt.api.storage import Storage
from qadapt.attack_graph import critical_targets, top_attack_paths
from qadapt.attack_graph.visualizer import to_cytoscape
from qadapt.core.config import ObjectiveWeights, OptimizationConfig
from qadapt.core.models import ThreatAssessment
from qadapt.pipeline import SOLVERS, QAdaptPipeline, make_solver
from qadapt.quantum_engine import qiskit_available
from qadapt.risk_engine import RiskModel

# Import Qiskit / Aer on the main thread: initialising their native extensions
# for the first time from a request worker thread has caused segfaults.
QISKIT = qiskit_available()

app = FastAPI(title="Q-ADAPT API", version=__version__,
              description="Hybrid quantum-classical adaptive cyber-defense decision engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

storage = Storage(os.environ.get("QADAPT_DB", ":memory:"))  # set QADAPT_DB to persist
session = SOCSession()
session.reset()


# ---------------------------------------------------------------------------
def _config(req: OptimizeRequest) -> OptimizationConfig:
    return OptimizationConfig(
        weights=ObjectiveWeights(**req.weights.model_dump()),
        budget=req.budget, max_time=req.max_time, max_disruption=req.max_disruption,
        max_actions=req.max_actions, constraint_encoding=req.encoding,
        protected_assets=tuple(req.protected_assets))


def _solver_kw(name: str, req: OptimizeRequest) -> dict:
    if name == "qaoa":
        return {"p": req.p, "noise": req.noise, "backend": req.backend, "shots": req.shots,
                "seed": req.seed, "qubo_kwargs": {"surrogate": req.surrogate},
                **({"restarts": 1, "maxiter": 80} if req.backend == "qiskit" else {})}
    if name in ("simulated_annealing", "genetic", "random_sampling"):
        return {"seed": req.seed, **({"n_samples": req.shots} if name == "random_sampling" else {})}
    if name == "milp":
        return {"surrogate": req.surrogate}
    return {}


def _paths(limit: int = 8):
    sources = sorted(session.ag.compromised | {h for h, t in session.ag.threat.items() if t >= 0.5})
    return top_attack_paths(session.ag.g, critical_targets(session.ag.g), k=4, sources=sources)[:limit]


# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "qiskit": QISKIT,
            "solvers": list(SOLVERS)}


@app.post("/api/session/reset")
def reset(req: ResetRequest):
    session.reset(req.scenario, req.seed, req.with_initial_threats)
    return overview()


@app.get("/api/overview")
def overview():
    with session.lock:
        rep = RiskModel().report(session.ag)
        paths = _paths(50)
        return to_jsonable({
            "scenario": session.scenario,
            "threats_detected": len(session.ag.threat_info),
            "critical_assets": sum(a.criticality >= 0.85 for a in session.ag.assets.values()),
            "critical_at_risk": rep.critical_at_risk(),
            "attack_paths": sum(p.probability >= 0.05 for p in paths),
            "current_risk": rep.total_risk,
            "assets": len(session.ag.assets),
            "applied_actions": [a.to_dict() for a in session.applied],
            "pending_run": session.pending.run_id if session.pending else None,
            "stage": session.stage,
            "stages": [s["label"] for s in STAGED_DEMO] if session.scenario == "demo" else [],
            "timeline": session.timeline,
            "graph": session.ag.stats(),
        })


@app.get("/api/threats")
def threats():
    return session.threats()


@app.post("/api/threats")
def inject_threat(t: ThreatIn):
    with session.lock:
        if t.host_id not in session.ag.assets:
            raise HTTPException(404, f"unknown asset {t.host_id}")
        ta = ThreatAssessment(t.host_id, t.attack_type, t.probability, t.confidence, 1, t.source)
        session.ag.apply_threats([ta])
        storage.log_event(ta.host_id, ta.attack_type, ta.probability, ta.confidence, ta.source)
        session.snapshot(f"Threat injected on {t.host_id}")
    return session.threats()


@app.post("/api/threats/detect")
def detect(req: DetectRequest):
    """Run the ML threat engine on synthetic live traffic towards the given hosts."""
    from qadapt.ml_engine import ThreatDetector, generate_flows
    with session.lock:
        if session.detector is None:
            model_path = Path(os.environ.get("QADAPT_MODEL", "models/detector.joblib"))
            session.detector = (ThreatDetector.load(model_path) if model_path.exists()
                                else ThreatDetector.train(generate_flows(4000, seed=11)))
        assets = session.ag.assets
        unknown = [h for h in req.attacked if h not in assets]
        if unknown:
            raise HTTPException(404, f"unknown assets {unknown}")
        attacked = {assets[h].ip: c for h, c in req.attacked.items()}
        flows = generate_flows(req.n_flows, attack_fraction=0.2 if attacked else 0.0,
                               hosts=[a.ip for a in assets.values()],
                               attacked_hosts=attacked or None, seed=req.seed)
        found = session.detector.assess_hosts(flows, session.ag.topology.ip_map())
        session.ag.apply_threats(found)
        for t in found:
            storage.log_event(t.host_id, t.attack_type, t.probability, t.confidence, t.source)
        session.snapshot(f"ML detection: {len(found)} host(s) flagged")
        return {"detections": [t.to_dict() for t in found], "flows": len(flows),
                "model": session.detector.model_name}


@app.get("/api/graph")
def graph():
    with session.lock:
        cg = session.ag.compile()
        from qadapt.attack_graph import propagate
        P = propagate(cg)
        paths = _paths()
        return to_jsonable(to_cytoscape(session.ag, cg, P, paths[0].nodes if paths else None,
                                        session.defended_edges()))


@app.get("/api/risk")
def risk(model: str = "propagation"):
    with session.lock:
        try:
            rep = RiskModel(model).report(session.ag)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return to_jsonable({**rep.to_dict(), "paths": [p.to_dict() for p in _paths()]})


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    if req.solver not in SOLVERS:
        raise HTTPException(400, f"unknown solver {req.solver}")
    with session.lock:
        pipe = QAdaptPipeline(session.ag, _config(req), max_qubits=req.max_qubits)
        problem = pipe.build_problem(exclude={a.id for a in session.applied})
        if problem.n == 0:
            raise HTTPException(409, "no candidate actions left")
        if req.solver == "exhaustive" and problem.n > 22:
            raise HTTPException(400, "exhaustive search limited to 22 actions")
        problem.qubo(surrogate=req.surrogate, encoding=req.encoding)
        report = pipe.decide(make_solver(req.solver, **_solver_kw(req.solver, req)), problem)
        out = to_jsonable(report.to_dict())
        run_id = storage.save_run(req.solver, req.model_dump(), out)
        session.pending = Pending(run_id, problem, report.selected_actions)
        out["run_id"] = run_id
        return out


@app.post("/api/compare")
def compare(req: CompareRequest):
    with session.lock:
        pipe = QAdaptPipeline(session.ag, _config(req), max_qubits=req.max_qubits)
        problem = pipe.build_problem(exclude={a.id for a in session.applied})
        problem.qubo(surrogate=req.surrogate, encoding=req.encoding)
        rows = []
        for name in req.solvers:
            if name not in SOLVERS or (name == "exhaustive" and problem.n > 22):
                continue
            r = make_solver(name, **_solver_kw(name, req)).solve(problem)
            rows.append(to_jsonable(r.to_dict()))
        best = min((r["objective"] for r in rows if r["feasible"]), default=None)
        for r in rows:
            r["gap_to_best"] = None if best is None else r["objective"] - best
        return {"n_actions": problem.n, "base_risk": problem.evaluator.base_risk, "results": rows}


@app.get("/api/runs")
def runs():
    return storage.list_runs()


@app.get("/api/runs/{run_id}")
def run(run_id: int):
    r = storage.get_run(run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    r["decisions"] = storage.decisions(run_id)
    return r


@app.post("/api/runs/{run_id}/decision")
def decide(run_id: int, d: DecisionIn):
    with session.lock:
        p = session.pending
        if p is None or p.run_id != run_id:
            raise HTTPException(409, "run is not the pending recommendation")
        ids = d.action_ids or [a.id for a in p.recommended]
        by_id = {a.id: a for a in p.problem.actions}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise HTTPException(404, f"unknown actions {unknown}")
        for i in ids:
            storage.log_decision(run_id, i, d.decision, d.analyst)
        if d.decision == "approve":
            session.apply([by_id[i] for i in ids])
            session.snapshot(f"Approved {len(ids)} action(s) from run {run_id}")
        remaining = [a for a in p.recommended if a.id not in ids]
        session.pending = Pending(run_id, p.problem, remaining) if remaining else None
        return overview()


@app.post("/api/adaptive/next-stage")
def next_stage():
    """Advance the scripted multi-stage attack (demo scenario)."""
    with session.lock:
        if session.scenario != "demo":
            raise HTTPException(400, "staged attack only available for the demo scenario")
        if session.stage >= len(STAGED_DEMO):
            raise HTTPException(409, "attack scenario already complete")
        st = STAGED_DEMO[session.stage]
        for h in st.get("compromised", []):
            session.ag.mark_compromised(h)
        session.ag.apply_threats(st["threats"])
        session.ag.apply_defenses(session.applied)
        for t in st["threats"]:
            storage.log_event(t.host_id, t.attack_type, t.probability, t.confidence, t.source)
        session.stage += 1
        session.pending = None
        session.snapshot(f"T{st['t']}: {st['label']}")
        return overview()


@app.get("/api/audit")
def audit():
    return {"events": storage.events(), "decisions": storage.decisions()}


# ---- static dashboard (built React app) -------------------------------------------
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = _DIST / path
        return FileResponse(f if f.is_file() else _DIST / "index.html")
