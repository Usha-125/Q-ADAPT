"""ThreatDetector: turns flow-level ML predictions into per-host threat assessments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qadapt.core.models import ThreatAssessment
from qadapt.ml_engine.datasets import BENIGN
from qadapt.ml_engine.models import _LabelEncodedPipeline, fit_model, malicious_probability
from qadapt.ml_engine.preprocessing import split_xy


class ThreatDetector:
    """Flow classifier plus host-level evidence aggregation.

    Host threat probability combines *how malicious* the worst flows look with
    *how much* evidence there is::

        p_host = mean(top-k flow maliciousness) * (1 - exp(-n_flagged / tau))

    so a single suspicious flow cannot saturate a host's threat score, while a
    sustained attack quickly approaches the classifier's confidence.
    """

    def __init__(self, model: _LabelEncodedPipeline, model_name: str = "",
                 flag_threshold: float = 0.5, top_k: int = 5, tau: float = 3.0):
        self.model = model
        self.model_name = model_name
        self.flag_threshold = flag_threshold
        self.top_k = top_k
        self.tau = tau

    @classmethod
    def train(cls, flows: pd.DataFrame, model_name: str = "random_forest",
              seed: int = 0, **kw) -> ThreatDetector:
        return cls(fit_model(model_name, flows, seed), model_name, **kw)

    def score_flows(self, flows: pd.DataFrame) -> pd.DataFrame:
        X, _ = split_xy(flows) if "category" in flows else (flows, None)
        proba = self.model.predict_proba(X)
        out = pd.DataFrame(index=flows.index)
        out["predicted"] = self.model.classes_[np.argmax(proba, axis=1)]
        out["p_malicious"] = malicious_probability(self.model, X)
        out["confidence"] = proba.max(axis=1)
        for c in ("src_ip", "dst_ip"):
            if c in flows:
                out[c] = flows[c].values
        return out

    def assess_hosts(self, flows: pd.DataFrame,
                     ip_to_asset: dict[str, str] | None = None) -> list[ThreatAssessment]:
        """Aggregate flow scores per destination host into ThreatAssessments."""
        scored = self.score_flows(flows)
        if "dst_ip" not in scored:
            raise ValueError("flows need a 'dst_ip' column for host attribution")
        results = []
        for ip, grp in scored.groupby("dst_ip"):
            host = (ip_to_asset or {}).get(ip, ip)
            flagged = grp[grp["p_malicious"] >= self.flag_threshold]
            if flagged.empty:
                continue
            top = np.sort(grp["p_malicious"].to_numpy())[::-1][: self.top_k]
            evidence = 1.0 - np.exp(-len(flagged) / self.tau)
            attack_types = Counter(t for t in flagged["predicted"] if t != BENIGN)
            attack = attack_types.most_common(1)[0][0] if attack_types else "ANOMALY"
            src = flagged["src_ip"].mode().iloc[0] if "src_ip" in flagged else ""
            results.append(ThreatAssessment(
                host_id=host,
                attack_type=attack,
                probability=float(top.mean() * evidence),
                confidence=float(flagged["confidence"].mean()),
                n_events=int(len(flagged)),
                source=str(src),
            ))
        return sorted(results, key=lambda t: -t.probability)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> ThreatDetector:
        return joblib.load(path)
