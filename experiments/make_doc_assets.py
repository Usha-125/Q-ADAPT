"""Generate documentation assets: the package import graph and dashboard screenshots.

    python -m experiments.make_doc_assets               # import graph + screenshots
    python -m experiments.make_doc_assets --no-screens  # import graph only

Screenshots need the built dashboard (cd frontend && npm run build) and Playwright.
"""

from __future__ import annotations

import argparse
import ast
import json
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "qadapt"
FIG = ROOT / "docs" / "figures"
SHOTS = ROOT / "docs" / "screenshots"

INK, INK2, SURFACE, EDGE = "#0b0b0b", "#52514e", "#fcfcfb", "#b9b8b2"
# one colour per architectural layer (validated categorical order)
LAYERS = [
    ("Interface", ["api", "cli"], "#2a78d6"),
    ("Orchestration", ["adaptive_engine", "pipeline", "benchmarks"], "#eb6834"),
    ("Optimisation", ["quantum_engine", "classical_baselines", "optimization", "explainability"], "#1baf7a"),
    ("Decision model", ["defense_engine", "risk_engine"], "#eda100"),
    ("Threat & graph", ["ml_engine", "attack_graph"], "#e87ba4"),
    ("Core", ["core"], "#4a3aa7"),
]


def subpackage(path: Path) -> str:
    rel = path.relative_to(PKG).parts
    return rel[0].removesuffix(".py") if rel else "qadapt"


def import_graph() -> dict[str, dict[str, int]]:
    """Edges sub-package -> sub-package with the number of import statements."""
    edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in PKG.rglob("*.py"):
        src = subpackage(f)
        if src in ("__init__", "qadapt"):
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("qadapt."):
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names if a.name.startswith("qadapt.")]
            for m in mods:
                dst = m.split(".")[1]
                if dst != src:
                    edges[src][dst] += 1
    return {k: dict(v) for k, v in edges.items()}


def draw_import_graph(edges) -> Path:
    pos, colour = {}, {}
    width = 15.0
    for row, (_, mods, c) in enumerate(LAYERS):
        y = (len(LAYERS) - 1 - row) * 1.6
        for i, m in enumerate(mods):
            pos[m] = ((i + 1) * width / (len(mods) + 1), y)
            colour[m] = c
    fig, ax = plt.subplots(figsize=(15, 10.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    for row, (name, _, _c) in enumerate(LAYERS):
        y = (len(LAYERS) - 1 - row) * 1.6
        ax.text(-0.3, y, name, ha="right", va="center", fontsize=11, color=INK2, fontweight="bold")
    for src, dsts in edges.items():
        for dst, n in dsts.items():
            if src not in pos or dst not in pos:
                continue
            (x1, y1), (x2, y2) = pos[src], pos[dst]
            ax.add_patch(FancyArrowPatch((x1, y1 - 0.28), (x2, y2 + 0.28), arrowstyle="-|>", mutation_scale=12,
                                         color=EDGE, lw=0.6 + 0.35 * min(n, 6), alpha=0.85,
                                         connectionstyle="arc3,rad=0.08", zorder=1))
    for m, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - 1.05, y - 0.28), 2.1, 0.56, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=colour[m], ec=SURFACE, lw=2, zorder=2))
        ax.text(x, y, m, ha="center", va="center", fontsize=10.5, color="white", fontweight="bold", zorder=3)
    ax.set_xlim(-2.6, width + 0.4)
    ax.set_ylim(-0.7, (len(LAYERS) - 1) * 1.6 + 0.7)
    ax.set_title("qadapt package import graph (arrow = imports from; width = number of import statements)",
                 loc="left", fontsize=13, fontweight="bold", color=INK)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "import_graph.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def check_layering(edges) -> list[str]:
    """Imports that point *up* the layer stack (should be none, or only lazy imports)."""
    rank = {m: r for r, (_, mods, _) in enumerate(LAYERS) for m in mods}
    return [f"{s} -> {d}" for s, ds in edges.items() for d in ds
            if s in rank and d in rank and rank[d] < rank[s]]


def screenshots() -> list[Path]:
    from playwright.sync_api import sync_playwright
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise SystemExit("build the dashboard first: cd frontend && npm run build")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "qadapt.api.app:app", "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    SHOTS.mkdir(parents=True, exist_ok=True)
    out = []
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1440, "height": 900})
            pg.goto(url)
            pg.wait_for_selector("text=Q-ADAPT Security Center")

            def shot(name, full=False):
                pg.wait_for_timeout(900)
                path = SHOTS / f"{name}.png"
                pg.screenshot(path=path, full_page=full)
                out.append(path)

            def nav(label):
                pg.click(f"nav >> text={label}")

            shot("01-soc-overview")
            nav("Threat Detection")
            pg.select_option("select >> nth=0", "FS-01")
            pg.click("text=Run detection")
            pg.wait_for_selector("text=host(s) flagged", timeout=60000)
            shot("02-threat-detection")
            nav("Attack Graph")
            pg.wait_for_selector("canvas")
            pg.wait_for_timeout(800)
            pg.mouse.click(768, 325)  # select APP-01 so the details panel is populated
            shot("03-attack-graph")
            nav("Risk Analysis")
            shot("04-risk-analysis", full=True)
            nav("Quantum Optimizer")
            pg.click("text=Run optimization")
            pg.wait_for_selector("text=Circuit depth", timeout=60000)
            shot("05-quantum-optimizer", full=True)
            pg.click("text=Review defense plan")
            pg.wait_for_selector("text=why selected?")
            shot("06-defense-plan", full=True)
            pg.click("text=Approve all")
            pg.wait_for_timeout(600)
            nav("SOC Overview")
            pg.click("text=Advance →")
            pg.wait_for_timeout(600)
            shot("07-overview-after-approval-and-stage-2")
            nav("Solver Benchmark")
            pg.fill("input >> nth=0", "10")
            pg.click("text=Run all solvers")
            pg.wait_for_selector("text=Objective (lower is better)", timeout=120000)
            shot("08-solver-benchmark", full=True)
            b.close()
    finally:
        proc.terminate()
        proc.wait(10)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-screens", action="store_true")
    a = ap.parse_args(argv)
    edges = import_graph()
    print("import graph:", draw_import_graph(edges))
    (FIG / "import_graph.json").write_text(json.dumps(edges, indent=2, sort_keys=True))
    up = check_layering(edges)
    print("upward (lazy) imports:", up or "none")
    if not a.no_screens:
        for p in screenshots():
            print("screenshot:", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
