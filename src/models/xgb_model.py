"""XGBoost ranker/regressor with early stopping. No net_rating in features."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def build_xgb(config: dict | None = None) -> Any:
    cfg = config or {}
    if not _HAS_XGB:
        raise ImportError("xgboost is required")
    return xgb.XGBRegressor(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        random_state=cfg.get("random_state", 42),
    )


def fit_xgb(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    eval_set: list[tuple] | None = None,
    early_stopping_rounds: int = 20,
) -> Any:
    if eval_set is None and X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
    kwargs: dict = {"verbose": False}
    if eval_set:
        kwargs["eval_set"] = eval_set
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    try:
        model.fit(X_train, y_train, **kwargs)
    except TypeError:
        kwargs.pop("early_stopping_rounds", None)
        model.fit(X_train, y_train, **kwargs)
    return model
