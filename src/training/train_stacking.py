"""Train stacking: OOF for A/XGB/RF, fit RidgeCV, persist oof_*.parquet and meta."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.models.stacking import build_oof, fit_ridgecv_on_oof, save_oof


def train_stacking(
    oof_deep_set: np.ndarray,
    oof_xgb: np.ndarray,
    oof_rf: np.ndarray,
    y: np.ndarray,
    config: dict,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = build_oof(oof_deep_set, oof_xgb, oof_rf, y)
    y = np.asarray(y).ravel()
    meta = fit_ridgecv_on_oof(X, y, cv=config.get("training", {}).get("n_folds", 5))

    oof_path = output_dir / "oof_pooled.parquet"
    save_oof(oof_deep_set, oof_xgb, oof_rf, y, oof_path)

    meta_path = output_dir / "ridgecv_meta.joblib"
    joblib.dump(meta, meta_path)
    return meta_path
