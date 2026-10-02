"""Feature preprocessing for flow-based threat detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

META_COLUMNS = ("category", "src_ip", "dst_ip", "label")


class FlowPreprocessor(BaseEstimator, TransformerMixin):
    """Clean -> signed log1p -> drop constant columns -> standardise."""

    def fit(self, X: pd.DataFrame, y=None):
        X = self._clean(X)
        self.columns_ = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
        self.scaler_ = StandardScaler().fit(self._transform_raw(X[self.columns_]))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._clean(X)
        for c in self.columns_:
            if c not in X:
                X[c] = 0.0
        return self.scaler_.transform(self._transform_raw(X[self.columns_]))

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=[c for c in META_COLUMNS if c in X.columns])
        X = X.apply(pd.to_numeric, errors="coerce")
        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _transform_raw(X: pd.DataFrame) -> np.ndarray:
        v = X.to_numpy(dtype=float)
        return np.sign(v) * np.log1p(np.abs(v))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.drop(columns=[c for c in META_COLUMNS if c in df.columns]), df["category"].to_numpy()
