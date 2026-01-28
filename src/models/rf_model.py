"""Random Forest model for Model B. No net_rating in features."""

from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.ensemble import RandomForestRegressor


def build_rf(config: dict | None = None) -> Any:
    cfg = config or {}
    return RandomForestRegressor(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", 12),
        min_samples_leaf=cfg.get("min_samples_leaf", 5),
        random_state=cfg.get("random_state", 42),
    )


def fit_rf(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    model.fit(X_train, y_train)
    return model
