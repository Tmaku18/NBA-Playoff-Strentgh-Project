"""OOF stacking: build OOF for A/XGB/RF, fit RidgeCV on pooled OOF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV


def build_oof(
    oof_deep_set: np.ndarray,
    oof_xgb: np.ndarray,
    oof_rf: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Stack OOF predictions (B, 3) -> (B,). For meta we use as X, not the final prediction."""
    return np.column_stack([np.asarray(oof_deep_set).ravel(), np.asarray(oof_xgb).ravel(), np.asarray(oof_rf).ravel()])


def fit_ridgecv_on_oof(
    X_oof: np.ndarray,
    y: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
    cv: int = 5,
) -> RidgeCV:
    """X_oof: (N, 3) from build_oof. y: (N,)."""
    meta = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
    meta.fit(X_oof, np.asarray(y).ravel())
    return meta


def save_oof(oof_deep_set: np.ndarray, oof_xgb: np.ndarray, oof_rf: np.ndarray, y: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "oof_deep_set": np.asarray(oof_deep_set).ravel(),
        "oof_xgb": np.asarray(oof_xgb).ravel(),
        "oof_rf": np.asarray(oof_rf).ravel(),
        "y": np.asarray(y).ravel(),
    }).to_parquet(path, index=False)


def load_oof(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    return df["oof_deep_set"].values, df["oof_xgb"].values, df["oof_rf"].values, df["y"].values
