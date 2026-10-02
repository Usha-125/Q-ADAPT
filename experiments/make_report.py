"""Build docs/RESULTS_REPORT.md with figures from a fresh training run, the experiment
results in results/*.json, and one end-to-end example decision.

    python -m experiments.make_report            # after: python -m experiments.run_all
"""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, roc_curve  # noqa: E402
from sklearn.model_selection import learning_curve, train_test_split  # noqa: E402

from experiments.common import md_table  # noqa: E402
from qadapt.adaptive_engine import STAGED_DEMO, run_episode  # noqa: E402
from qadapt.attack_graph import AttackGraph, demo_topology  # noqa: E402
from qadapt.core.config import OptimizationConfig  # noqa: E402
from qadapt.ml_engine import BENIGN, generate_flows  # noqa: E402
from qadapt.ml_engine.models import (  # noqa: E402
    build_model,
    malicious_probability,
    train_and_evaluate,
)
from qadapt.ml_engine.preprocessing import split_xy  # noqa: E402
from qadapt.pipeline import QAdaptPipeline  # noqa: E402
from qadapt.quantum_engine import QAOASolver  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
RES = ROOT / "results"

# Validated categorical order (dataviz reference palette, light mode) + text tokens.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Blues"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 10, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 7, "legend.frameon": False, "legend.labelcolor": INK2,
})


def save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    place_labels(fig)
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    return f"figures/{name}"


_PENDING: dict = {}


def label_end(ax, x, y, text, color=INK2):
    """Queue a direct label at the end of a line; flushed (de-collided) by place_labels."""
    _PENDING.setdefault(ax, []).append((float(x[-1]), float(y[-1]), text, color))


def place_labels(fig, min_gap_pt: float = 11.0) -> None:
    """Draw queued end labels, nudging them apart vertically so they never overlap."""
    fig.canvas.draw()
    for ax, items in list(_PENDING.items()):
        if ax.figure is not fig:
            continue
        to_px = ax.transData.transform
        pts = sorted(((to_px((x, y))[1], x, y, t, c) for x, y, t, c in items), key=lambda r: r[0])
        gap = min_gap_pt * fig.dpi / 72.0
        placed = []
        for py, x, y, t, c in pts:
            target = max(py, placed[-1] + gap) if placed else py
            placed.append(target)
            ax.annotate(t, (x, y), xytext=(6, (target - py) * 72.0 / fig.dpi),
                        textcoords="offset points", va="center", fontsize=9, color=c)
        del _PENDING[ax]


# ---------------------------------------------------------------------------
# 1. ML training
# ---------------------------------------------------------------------------
def ml_section() -> str:
    df = generate_flows(12000, seed=0)
    train, test = train_test_split(df, test_size=0.25, random_state=0, stratify=df["category"])
    names = {"random_forest": "Random Forest", "hist_gradient_boosting": "Gradient Boosting",
             "mlp": "MLP neural net"}
    models, reports, rows = {}, {}, []
    for k, label in names.items():
        m, rep = train_and_evaluate(k, train, test)
        Xtr, ytr = split_xy(train)
        train_acc = float((m.predict(Xtr) == ytr).mean())
        models[k], reports[k] = m, rep
        rows.append([label, f"{train_acc:.4f}", f"{rep.accuracy:.4f}", f"{rep.precision_macro:.4f}",
                     f"{rep.recall_macro:.4f}", f"{rep.f1_macro:.4f}", f"{rep.roc_auc_binary:.4f}",
                     f"{rep.false_positive_rate:.4f}", f"{rep.false_negative_rate:.4f}",
                     f"{rep.train_time_s:.2f}"])
        reports[k].train_accuracy = train_acc

    # learning curves (train vs cross-validated accuracy)
    X, y = split_xy(train)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    lc_rows = []
    for ax, (k, label) in zip(axes, names.items(), strict=True):
        sizes, tr, va = learning_curve(build_model(k, 0), X, np.unique(y, return_inverse=True)[1],
                                       train_sizes=np.linspace(0.1, 1.0, 6), cv=3,
                                       scoring="accuracy", n_jobs=-1)
        for s, mean, color, name in ((tr, tr.mean(1), C[0], "training"),
                                     (va, va.mean(1), C[1], "validation")):
            ax.plot(sizes, mean, marker="o", color=color, label=name)
            ax.fill_between(sizes, mean - s.std(1), mean + s.std(1), color=color, alpha=0.12, lw=0)
            label_end(ax, sizes, mean, f"{mean[-1]:.3f}")
        ax.set_title(label)
        ax.set_xlabel("training flows")
        lc_rows.append([label, f"{tr.mean(1)[-1]:.4f}", f"{va.mean(1)[-1]:.4f}",
                        f"{tr.mean(1)[-1] - va.mean(1)[-1]:+.4f}"])
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="lower right")
    for ax in axes:
        ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    f_lc = save(fig, "ml_learning_curves.png")

    # MLP training loss / validation accuracy per epoch
    mlp = models["mlp"].pipeline.named_steps["clf"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ep = np.arange(1, len(mlp.loss_curve_) + 1)
    axes[0].plot(ep, mlp.loss_curve_, color=C[0])
    label_end(axes[0], ep, mlp.loss_curve_, f"{mlp.loss_curve_[-1]:.3f}")
    axes[0].set_title("MLP training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    vs = np.asarray(mlp.validation_scores_)
    axes[1].plot(ep, vs, color=C[1])
    label_end(axes[1], ep, vs, f"{vs[-1]:.3f}")
    axes[1].set_title("MLP validation accuracy (early-stopping split)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    f_mlp = save(fig, "ml_mlp_training.png")

    # confusion matrix + per-class F1 for the best model
    best = max(reports, key=lambda k: reports[k].f1_macro)
    Xte, yte = split_xy(test)
    pred = models[best].predict(Xte)
    classes = list(models[best].classes_)
    cm = confusion_matrix(yte, pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.6 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalised) - {names[best]}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    f_cm = save(fig, "ml_confusion_matrix.png")

    f1s = f1_score(yte, pred, labels=classes, average=None)
    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(np.array(classes)[order], f1s[order], color=C[0], height=0.6)
    for i, v in enumerate(f1s[order]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 1.1)
    ax.set_title(f"Per-class F1 - {names[best]}")
    ax.grid(axis="y", visible=False)
    f_f1 = save(fig, "ml_per_class_f1.png")

    # ROC (malicious vs benign)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    y_bin = (yte != BENIGN).astype(int)
    for i, (k, label) in enumerate(names.items()):
        fpr, tpr, _ = roc_curve(y_bin, malicious_probability(models[k], Xte))
        ax.plot(fpr, tpr, color=C[i], label=f"{label} (AUC {reports[k].roc_auc_binary:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRID, lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC - malicious vs benign")
    ax.legend(loc="lower right")
    f_roc = save(fig, "ml_roc.png")

    # feature importance (RF)
    rf = models["random_forest"].pipeline
    imp = rf.named_steps["clf"].feature_importances_
    cols = rf.named_steps["prep"].columns_
    top = np.argsort(imp)[-12:]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(np.array(cols)[top], imp[top], color=C[0], height=0.6)
    ax.set_title("Random Forest - top 12 feature importances")
    ax.grid(axis="y", visible=False)
    f_fi = save(fig, "ml_feature_importance.png")

    return f"""## 1. ML threat detection - training and evaluation

> **Data notice.** These numbers come from the built-in *synthetic* CICFlowMeter-style generator
> (12,000 flows, 9 classes, stratified 75/25 split). They validate the training pipeline; they are
> **not** CSE-CIC-IDS2018 / CIC-IDS2017 results. Run `python -m experiments.experiment1_ml
> --train-csv data/raw/...` on the real datasets for publishable numbers (see `docs/DATASETS.md`).

### Scores

{md_table(["model", "train accuracy", "test accuracy", "precision (macro)", "recall (macro)",
           "F1 (macro)", "ROC-AUC", "FPR", "FNR", "train s"], rows)}

Best model by macro-F1: **{names[best]}**.

### Learning curves (training vs 3-fold validation accuracy)

![learning curves]({f_lc})

{md_table(["model", "final train acc.", "final validation acc.", "generalisation gap"], lc_rows)}

Both tree ensembles reach 100 % training accuracy. Gradient boosting generalises best (validation
{lc_rows[1][2]}) with half the generalisation gap of Random Forest; the MLP has the smallest gap but
lower accuracy. Random Forest and the MLP are still improving at the largest training size
(more data would help); gradient boosting has plateaued.

### MLP training history

![MLP training]({f_mlp})

### Confusion matrix and per-class F1

![confusion matrix]({f_cm})

![per-class F1]({f_f1})

Weakest class: **{classes[int(np.argmin(f1s))]}** (F1 {f1s.min():.3f}). Most of its errors are
predicted as BENIGN, because web attacks look like ordinary HTTP flows in flow-level features. This
is also a known difficulty on the real CIC datasets, and payload-aware features would be needed to
fix it.

### ROC curves and feature importance

![ROC]({f_roc})

![feature importance]({f_fi})
"""


# ---------------------------------------------------------------------------
# 2-8. experiment results
# ---------------------------------------------------------------------------
def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def graph_section() -> str:
    d = load("experiment2_graph")["results"]
    n = [r["nodes"] for r in d]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(n, [r["edges"] for r in d], marker="o", color=C[0])
    label_end(axes[0], n, [r["edges"] for r in d], f"{d[-1]['edges']:.0f} edges")
    axes[0].set_title("Attack-graph size")
    axes[0].set_xlabel("nodes")
    axes[0].set_ylabel("edges")
    for i, (k, lab) in enumerate((("gen_s", "build"), ("prop_s", "propagation"))):
        y = [r[k] * 1000 for r in d]
        axes[1].plot(n, y, marker="o", color=C[i], label=lab)
        label_end(axes[1], n, y, f"{lab} {y[-1]:.1f} ms")
    axes[1].set_title("Runtime")
    axes[1].set_xlabel("nodes")
    axes[1].set_ylabel("milliseconds")
    axes[1].set_xlim(right=max(n) * 1.35)
    axes[1].legend(loc="upper left")
    f = save(fig, "graph_scalability.png")
    return f"""## 2. Attack-graph scalability (Experiment 2)

![graph scalability]({f})

{(RES / "experiment2_graph.md").read_text().split(chr(10), 3)[3]}
"""


def optimization_section() -> str:
    recs = load("experiment3_optimization")["records"]
    sizes = sorted({r["n"] for r in recs})

    def mean(solver, key, n):
        v = [r[key] for r in recs if r["solver"] == solver and r["n"] == n and r[key] is not None]
        return float(np.mean(v)) if v else np.nan

    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = [("qaoa_p2", "QAOA p=2"), ("simulated_annealing", "Simulated annealing"),
              ("greedy", "Greedy"), ("random_sampling", "Random sampling (same shots)")]
    for i, (s, lab) in enumerate(series):
        y = [100 * mean(s, "gap", n) for n in sizes]
        ax.plot(sizes, y, marker="o", color=C[i], label=lab)
        label_end(ax, sizes, y, lab)
    ax.set_xlim(right=sizes[-1] + 4)
    ax.set_xlabel("candidate actions (= qubits for QAOA)")
    ax.set_ylabel("optimality gap (%)")
    ax.set_title("Gap to the exact optimum (lower is better)")
    ax.legend(loc="upper left")
    f_gap = save(fig, "opt_gap_vs_size.png")

    solvers = ["qaoa_p1", "qaoa_p2", "qaoa_p3", "simulated_annealing", "genetic", "greedy", "milp",
               "score_ranking", "random_sampling"]
    opt_rate = [100 * np.mean([r["optimal"] for r in recs if r["solver"] == s]) for s in solvers]
    order = np.argsort(opt_rate)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(np.array(solvers)[order], np.array(opt_rate)[order], color=C[0], height=0.6)
    for i, v in enumerate(np.array(opt_rate)[order]):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 112)
    ax.set_xlabel("instances solved to optimality (%) - all sizes, 50 instances")
    ax.set_title("How often each solver found the true optimum")
    ax.grid(axis="y", visible=False)
    f_opt = save(fig, "opt_optimal_rate.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, p in enumerate((1, 2, 3)):
        ar = [mean(f"qaoa_p{p}", "approximation_ratio", n) for n in sizes]
        amp = [mean(f"qaoa_p{p}", "optimal_state_amplification", n) for n in sizes]
        axes[0].plot(sizes, ar, marker="o", color=C[i], label=f"p = {p}")
        axes[1].plot(sizes, amp, marker="o", color=C[i], label=f"p = {p}")
        label_end(axes[0], sizes, ar, f"p={p}")
        label_end(axes[1], sizes, amp, f"p={p}: {amp[-1]:.0f}x")
    axes[0].set_title("QAOA approximation ratio")
    axes[0].set_ylabel("<E> relative to QUBO range")
    axes[1].set_title("Amplification of the optimal state vs uniform")
    axes[1].set_ylabel("P(opt) / P_uniform(opt)")
    for ax in axes:
        ax.set_xlabel("qubits")
        ax.legend(loc="upper left")
        ax.set_xlim(right=sizes[-1] + 2.5)
    f_q = save(fig, "qaoa_metrics.png")

    md = (RES / "experiment3_optimization.md").read_text()
    return f"""## 3. QAOA vs classical optimisation (Experiments 3 & 4)

![optimality gap]({f_gap})

![optimal rate]({f_opt})

![QAOA metrics]({f_q})

{md.split(chr(10), 3)[3]}
"""


def noise_section() -> str:
    recs = load("experiment5_noise")["records"]
    levels = ["ideal", "low", "medium", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(levels))
    for i, p in enumerate((1, 2, 3)):
        ar = [np.mean([r["approximation_ratio"] for r in recs if r["p"] == p and r["noise"] == lv])
              for lv in levels]
        same = [100 * np.mean([r["same_as_ideal"] for r in recs if r["p"] == p and r["noise"] == lv])
                for lv in levels]
        axes[0].plot(x, ar, marker="o", color=C[i], label=f"p = {p}")
        axes[1].plot(x, same, marker="o", color=C[i], label=f"p = {p}")
        label_end(axes[0], x, ar, f"p={p}")
        label_end(axes[1], x, same, f"p={p}")
    for ax in axes:
        ax.set_xlim(-0.2, len(levels) - 1 + 0.4)
        ax.set_xticks(x, levels)
        ax.set_xlabel("noise level")
        ax.legend(loc="lower left")
    axes[0].set_title("Approximation ratio under noise")
    axes[1].set_title("Plan identical to the noise-free plan (%)")
    f = save(fig, "noise.png")
    return f"""## 4. Quantum noise sensitivity (Experiment 5)

![noise]({f})

{(RES / "experiment5_noise.md").read_text().split(chr(10), 3)[3]}
"""


def adaptive_section() -> str:
    cfg = OptimizationConfig(budget=0.4, max_actions=3)
    from experiments.experiment6_adaptive import TRUE_EFF
    curves = {}
    for pol in ("none", "static", "reoptimize", "aqdo"):
        hist = [run_episode(demo_topology(), pol, 8, cfg, solver="greedy", true_effectiveness=TRUE_EFF,
                            seed=s, max_qubits=10)["history"] for s in range(20)]
        curves[pol] = np.mean([[h["true_loss"] for h in hh] for hh in hist], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(1, 9)
    names = {"none": "No defense", "static": "Static (decide once)",
             "reoptimize": "Re-optimise on change", "aqdo": "Full AQDO"}
    for i, (pol, y) in enumerate(curves.items()):
        ax.plot(steps, 100 * y, marker="o", color=C[i], label=names[pol],
                ls="--" if pol == "aqdo" else "-")
        label_end(ax, steps, 100 * y, f"{names[pol]} {100 * y[-1]:.0f}%")
    ax.set_xlim(right=10.6)
    ax.set_xlabel("decision step")
    ax.set_ylabel("assets compromised (criticality-weighted %)")
    ax.set_title("Attack campaign on the demo network (mean of 20 runs)")
    ax.legend(loc="upper left")
    f = save(fig, "adaptive_loss_over_time.png")
    md = (RES / "experiment6_adaptive.md").read_text()
    return f"""## 5. Dynamic adaptation (Experiment 6)

![loss over time]({f})

Curves use the greedy solver for speed (QAOA gives statistically identical plans here; the table
below uses QAOA). Re-optimisation and full AQDO overlap: the learning and re-weighting components
added no measurable benefit.

{md.split(chr(10), 2)[2]}
"""


def constraints_section() -> str:
    return f"""## 6. Resource constraints and encodings (Experiment 7)

{(RES / "experiment7_constraints.md").read_text().split(chr(10), 3)[3]}
"""


def ablation_section() -> str:
    recs = load("experiment8_ablation")["records"]
    models = list(dict.fromkeys(r["model"] for r in recs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, env in zip(axes, ("demo (14 nodes)", "enterprise-40"), strict=True):
        vals = [np.mean([r["cumulative_loss"] for r in recs if r["env"] == env and r["model"] == m])
                for m in models]
        ax.barh(models[::-1], vals[::-1], color=C[0], height=0.6)
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9, color=INK2)
        ax.set_title(env)
        ax.set_xlabel("cumulative loss (lower is better)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(right=max(vals) * 1.2)
    f = save(fig, "ablation.png")
    return f"""## 7. Ablation study (Experiment 8)

![ablation]({f})

{(RES / "experiment8_ablation.md").read_text().split("## ", 1)[0].split(chr(10), 2)[2]}
{"## ".join([""] + (RES / "experiment8_ablation.md").read_text().split("## ")[1:]).replace("## ", "### ")}
"""


# ---------------------------------------------------------------------------
# one end-to-end example
# ---------------------------------------------------------------------------
def example_section() -> str:
    ag = AttackGraph(demo_topology())
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.5), max_qubits=12)
    threats = STAGED_DEMO[0]["threats"] + STAGED_DEMO[1]["threats"]
    pipe.ingest(threats)
    from qadapt.risk_engine import RiskModel
    before = {a.asset_id: a.risk for a in RiskModel().report(ag).assets}
    t0 = time.perf_counter()
    rep = pipe.decide(QAOASolver(p=2, shots=1024, seed=0))
    elapsed = time.perf_counter() - t0
    r = rep.result
    sel = [rep.actions[i] for i in np.flatnonzero(r.x)]
    ag.apply_defenses(sel)
    after = {a.asset_id: a.risk for a in RiskModel().report(ag).assets}

    ids = sorted(before, key=lambda k: -before[k])
    y = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y + 0.2, [before[i] for i in ids], height=0.38, color=C[1], label="before")
    ax.barh(y - 0.2, [after[i] for i in ids], height=0.38, color=C[0], label="after recommended defense")
    ax.set_yticks(y, ids)
    ax.invert_yaxis()
    ax.set_xlabel("asset risk (criticality x P(compromise))")
    ax.set_title(f"Example: total risk {rep.risk_before:.1%} -> {rep.risk_after:.1%}")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    f_risk = save(fig, "example_asset_risk.png")

    conv = r.info["convergence"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(np.arange(1, len(conv) + 1), conv, color=C[0])
    label_end(ax, np.arange(1, len(conv) + 1), conv, f"{conv[-1]:.4f}")
    ax.set_xlabel("COBYLA iteration (final depth p = 2)")
    ax.set_ylabel("normalised <H_C>")
    ax.set_title("QAOA parameter optimisation")
    f_conv = save(fig, "example_qaoa_convergence.png")

    buf = io.StringIO()
    with redirect_stdout(buf):
        from qadapt.cli import main as cli
        cli(["demo"])
    cli_out = buf.getvalue().strip()

    explain_rows = [[("✅ " if e.selected else "✖ ") + e.description, "<br>".join(e.reasons)]
                    for e in rep.explanations]
    th_rows = [[t.host_id, t.attack_type, f"{t.probability:.0%}", f"{t.confidence:.0%}", t.source]
               for t in threats]
    return f"""## 0. Worked example - one complete Q-ADAPT run

Command: `qadapt demo` (demo enterprise, 14 assets, QAOA p = 2, budget 0.5). Decision
time {elapsed:.1f} s (QUBO build + QAOA simulation) in a cloud CPU container.

**Input - ML threat assessments**

{md_table(["host", "attack type", "probability", "confidence", "source"], th_rows)}

**Quantum optimisation**

| item | value |
|---|---|
| candidate actions (qubits) | {r.info['n_qubits']} |
| QUBO surrogate fidelity (Spearman vs true risk) | {r.info['qubo_meta']['fidelity']['spearman']:.3f} |
| circuit depth (scheduled) | {r.info['circuit_depth']} |
| function evaluations | {r.info['function_evals']} |
| approximation ratio | {r.info['approximation_ratio']:.3f} |
| P(optimal QUBO state) | {r.info['p_optimal_state']:.3%} ({r.info['optimal_state_amplification']:.0f}x uniform) |
| plan feasible | {r.feasible} |

**Outcome**

| metric | value |
|---|---|
| network risk | {rep.risk_before:.1%} -> {rep.risk_after:.1%} (-{r.metrics.risk_reduction:.1%}) |
| attack paths to critical assets (p >= 5%) | {rep.paths_before} -> {rep.paths_after} |
| cost / response time / disruption | {r.metrics.cost:.2f} / {r.metrics.time_total:.2f} / {r.metrics.disruption:.2f} |

![asset risk before and after]({f_risk})

![QAOA convergence]({f_conv})

**Recommendation and explanation**

{md_table(["action", "why"], explain_rows)}

<details><summary>Raw CLI output</summary>

```
{cli_out}
```
</details>
"""


def main():
    t0 = time.time()
    parts = [
        "# Q-ADAPT - Results Report\n",
        f"Generated by `python -m experiments.make_report` on {time.strftime('%Y-%m-%d')}. "
        "All figures and tables are produced from code in this repository with fixed seeds; "
        "experiment tables come from `results/*.json` (`python -m experiments.run_all`). "
        "QAOA results are classical simulations - no quantum speed-up is claimed.\n",
        "## Contents\n\n0. Worked example\n1. ML threat detection (training, accuracy, curves)\n"
        "2. Attack-graph scalability\n3. QAOA vs classical optimisation\n4. Noise sensitivity\n"
        "5. Dynamic adaptation\n6. Resource constraints\n7. Ablation study\n8. Conclusions\n",
        example_section(), ml_section(), graph_section(), optimization_section(), noise_section(),
        adaptive_section(), constraints_section(), ablation_section(),
        """## 8. Conclusions

* **Formulation works.** The fitted QUBO surrogate tracks the propagated risk with Spearman
  0.997-0.999, and unbalanced penalisation kept the QUBO minimiser feasible in every constraint
  scenario without extra qubits.
* **QAOA is viable but not superior.** It solves 6-10-action instances (near-)optimally and clearly
  beats naive ranking and equal-budget random sampling, but simulated annealing and the genetic
  algorithm were optimal on every instance; at 12-14 qubits QAOA averages a ~5 % gap (p = 2).
* **Noise matters.** High noise lowers the approximation ratio from ~0.97 to ~0.79 and changes the
  recommended plan in most runs; deeper circuits degrade more.
* **Adaptation is the biggest lever.** Re-optimising on state change roughly halves (demo) to thirds
  (enterprise) cumulative loss vs a static plan and nearly eliminates critical-asset loss
  (p < 0.002). Effectiveness learning and re-weighting added no significant gain.
* **Graph awareness matters.** Adding the attack graph to ML-only alerting significantly reduces loss.
* **Open item:** ML numbers above are on synthetic flows - rerun Experiment 1 on the public datasets.
""",
    ]
    import re
    text = "\n".join(parts)
    # embedded experiment tables carry their own "## " headings: demote them below the sections
    text = re.sub(r"^## (?!\d+\. |Contents)", "### ", text, flags=re.M)
    out = ROOT / "docs" / "RESULTS_REPORT.md"
    out.write_text(text)
    print(f"wrote {out} and {len(list(FIG.glob('*.png')))} figures in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
