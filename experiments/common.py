"""Shared helpers for the Q-ADAPT experiment suite."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy import stats

from qadapt.benchmarks import make_instance, random_threats  # noqa: F401 (re-exported)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def mean_ci(values, conf: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    m, s = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    h = float(stats.t.ppf((1 + conf) / 2, len(v) - 1) * s / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"mean": m, "std": s, "ci95": h, "n": int(len(v))}


def wilcoxon(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"statistic": None, "p_value": 1.0, "note": "identical"}
    r = stats.wilcoxon(a, b, zero_method="zsplit")
    return {"statistic": float(r.statistic), "p_value": float(r.pvalue)}


def fmt(stat: dict, digits: int = 4) -> str:
    if stat["mean"] is None:
        return "n/a"
    return f"{stat['mean']:.{digits}f} ± {stat['ci95']:.{digits}f}"


def save(name: str, payload: dict, markdown: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    payload = {"experiment": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "platform": platform.platform(), "python": platform.python_version(), **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_default))
    (RESULTS / f"{name}.md").write_text(markdown)
    print(markdown)
    return path


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
