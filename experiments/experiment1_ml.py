"""Experiment 1 - ML threat detection: RF vs gradient boosting vs MLP (vs XGBoost).

Real data:      python -m experiments.experiment1_ml --train-csv data/raw/2018/*.csv
Cross-dataset:  ... --train-csv <CSE-CIC-IDS2018 files> --test-csv <CIC-IDS2017 files>
Without CSVs the experiment runs on the synthetic generator and is labelled as such.
"""

from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split

from experiments.common import md_table, save
from qadapt.ml_engine import MODEL_FACTORIES, generate_flows, load_cic_csvs, train_and_evaluate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", nargs="*")
    ap.add_argument("--test-csv", nargs="*")
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--n-synthetic", type=int, default=12000)
    ap.add_argument("--models", nargs="*", default=["random_forest", "hist_gradient_boosting", "mlp", "xgboost"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    if a.train_csv:
        df = load_cic_csvs(a.train_csv, a.sample)
        source = "CIC CSVs: " + ", ".join(a.train_csv)
    else:
        df = generate_flows(3000 if a.quick else a.n_synthetic, seed=0)
        source = "SYNTHETIC flows (pipeline validation only - not a dataset result)"
    if a.test_csv:
        train, test = df, load_cic_csvs(a.test_csv, a.sample)
        protocol = "cross-dataset (train/test from different datasets)"
    else:
        train, test = train_test_split(df, test_size=0.25, random_state=0, stratify=df["category"])
        protocol = "stratified 75/25 hold-out"

    reports = []
    for m in a.models:
        if m not in MODEL_FACTORIES:
            continue
        try:
            _, rep = train_and_evaluate(m, train, test)
        except ImportError as e:
            print(f"skip {m}: {e}")
            continue
        reports.append(rep.to_dict())
    rows = [[r["model"], f"{r['accuracy']:.4f}", f"{r['precision_macro']:.4f}", f"{r['recall_macro']:.4f}",
             f"{r['f1_macro']:.4f}", f"{r['roc_auc_binary']:.4f}", f"{r['false_positive_rate']:.4f}",
             f"{r['false_negative_rate']:.4f}", f"{r['train_time_s']:.2f}", f"{r['inference_time_ms_per_1k']:.2f}"]
            for r in reports]
    md = (f"# Experiment 1 - ML threat detection\n\nData: {source}  \nProtocol: {protocol}  \n"
          f"Train/test: {len(train)}/{len(test)} flows\n\n"
          + md_table(["model", "accuracy", "precision (macro)", "recall (macro)", "F1 (macro)",
                      "ROC-AUC (malicious)", "FPR", "FNR", "train s", "infer ms / 1k"], rows) + "\n")
    save("experiment1_ml", {"source": source, "protocol": protocol, "reports": reports}, md)


if __name__ == "__main__":
    main()
