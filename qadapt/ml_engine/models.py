"""Model registry and training/evaluation for the ML threat engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from qadapt.ml_engine.datasets import BENIGN
from qadapt.ml_engine.preprocessing import FlowPreprocessor, split_xy


def _xgboost(seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("xgboost is not installed: pip install 'qadapt[xgboost]'") from exc
    return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                         tree_method="hist", random_state=seed, n_jobs=-1)


MODEL_FACTORIES = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced_subsample",
        random_state=seed),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, random_state=seed),
    "mlp": lambda seed: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                      early_stopping=True, random_state=seed),
    "xgboost": _xgboost,
}


class _LabelEncodedPipeline:
    """Pipeline wrapper that handles string labels for every backend (incl. XGBoost)."""

    def __init__(self, pipeline: Pipeline, classes: np.ndarray):
        self.pipeline = pipeline
        self.classes_ = classes

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model(name: str, seed: int = 0) -> Pipeline:
    if name not in MODEL_FACTORIES:
        raise KeyError(f"unknown model '{name}', choose from {sorted(MODEL_FACTORIES)}")
    return Pipeline([("prep", FlowPreprocessor()), ("clf", MODEL_FACTORIES[name](seed))])


def fit_model(name: str, train: pd.DataFrame, seed: int = 0) -> _LabelEncodedPipeline:
    X, y = split_xy(train)
    classes, y_idx = np.unique(y, return_inverse=True)
    pipe = build_model(name, seed)
    pipe.fit(X, y_idx)
    return _LabelEncodedPipeline(pipe, classes)


@dataclass
class EvaluationReport:
    model: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_binary: float
    false_positive_rate: float
    false_negative_rate: float
    train_time_s: float
    inference_time_ms_per_1k: float
    n_train: int
    n_test: int
    classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def malicious_probability(model: _LabelEncodedPipeline, X) -> np.ndarray:
    proba = model.predict_proba(X)
    if BENIGN in model.classes_:
        return 1.0 - proba[:, list(model.classes_).index(BENIGN)]
    return proba.max(axis=1)


def evaluate(name: str, model: _LabelEncodedPipeline, test: pd.DataFrame,
             train_time: float = 0.0, n_train: int = 0) -> EvaluationReport:
    X, y = split_xy(test)
    t0 = time.perf_counter()
    proba = model.predict_proba(X)
    infer = time.perf_counter() - t0
    pred = model.classes_[np.argmax(proba, axis=1)]
    y_bin = (y != BENIGN).astype(int)
    p_mal = malicious_probability(model, X)
    pred_bin = (pred != BENIGN).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_bin, p_mal) if len(np.unique(y_bin)) == 2 else float("nan")
    return EvaluationReport(
        model=name,
        accuracy=accuracy_score(y, pred),
        precision_macro=precision_score(y, pred, average="macro", zero_division=0),
        recall_macro=recall_score(y, pred, average="macro", zero_division=0),
        f1_macro=f1_score(y, pred, average="macro", zero_division=0),
        roc_auc_binary=float(auc),
        false_positive_rate=fp / max(fp + tn, 1),
        false_negative_rate=fn / max(fn + tp, 1),
        train_time_s=train_time,
        inference_time_ms_per_1k=1000.0 * infer / max(len(X), 1) * 1000.0,
        n_train=n_train,
        n_test=len(X),
        classes=[str(c) for c in model.classes_],
    )


def train_and_evaluate(name: str, train: pd.DataFrame, test: pd.DataFrame,
                       seed: int = 0) -> tuple[_LabelEncodedPipeline, EvaluationReport]:
    t0 = time.perf_counter()
    model = fit_model(name, train, seed)
    return model, evaluate(name, model, test, time.perf_counter() - t0, len(train))
