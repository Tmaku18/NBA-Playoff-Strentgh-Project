"""Train Model B (XGB + RF) on team-context features. Enforce no net_rating."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.models.rf_model import build_rf, fit_rf
from src.models.xgb_model import build_xgb, fit_xgb
from src.utils.leakage_tests import test_model_b_excludes_net_rating


def train_model_b(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    config: dict,
    feature_names: list[str],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    test_model_b_excludes_net_rating()
    for n in feature_names:
        assert "net_rating" not in str(n).lower(), f"Model B must not use net_rating; found: {n}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mb = config.get("model_b", {})
    xgb_cfg = mb.get("xgb", {})
    rf_cfg = mb.get("rf", {})
    es = xgb_cfg.get("early_stopping_rounds", 20)

    xgb_m = build_xgb(xgb_cfg)
    fit_xgb(xgb_m, X_train, y_train, X_val, y_val, early_stopping_rounds=es)

    rf_m = build_rf(rf_cfg)
    fit_rf(rf_m, X_train, y_train)

    p1 = output_dir / "xgb_model.joblib"
    p2 = output_dir / "rf_model.joblib"
    joblib.dump(xgb_m, p1)
    joblib.dump(rf_m, p2)
    return p1, p2
