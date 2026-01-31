"""Train Model B (XGB + RF) on real DB team-context features. Option A: K-fold OOF, then final models."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.db_loader import load_training_data
from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates
from src.training.build_lists import build_lists
from src.training.train_model_b import train_model_b

from src.models.xgb_model import build_xgb, fit_xgb
from src.models.rf_model import build_rf, fit_rf


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)
    games, tgl, teams, pgl = load_training_data(db_path)
    lists = build_lists(tgl, games, teams)
    if not lists:
        print("No lists from build_lists (empty games/tgl?). Exiting.", file=sys.stderr)
        sys.exit(1)
    rows = []
    for lst in lists:
        for tid, wr in zip(lst["team_ids"], lst["win_rates"]):
            rows.append({"team_id": int(tid), "as_of_date": lst["as_of_date"], "y": float(wr)})
    flat = pd.DataFrame(rows)
    team_dates = [(int(a), str(b)) for a, b in flat[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
    feat_df = build_team_context_as_of_dates(tgl, games, team_dates)
    df = flat.merge(feat_df, on=["team_id", "as_of_date"], how="inner")
    feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in df.columns]
    if not feat_cols:
        print("No feature columns. Exiting.", file=sys.stderr)
        sys.exit(1)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    n_folds = config.get("training", {}).get("n_folds", 5)
    dates_sorted = sorted(df["as_of_date"].unique())
    n_folds = min(n_folds, len(dates_sorted))
    if n_folds < 2:
        X = df[feat_cols].values.astype(np.float32)
        y = df["y"].values.astype(np.float32)
        p1, p2 = train_model_b(X, y, None, None, config, feat_cols, out)
        print(f"Saved {p1}, {p2} (too few dates for OOF)")
        return

    # Assign fold by date (same time-based split as script 3)
    fold_size = (len(dates_sorted) + n_folds - 1) // n_folds
    date_to_fold = {}
    for fold in range(n_folds):
        start = fold * fold_size
        end = min((fold + 1) * fold_size, len(dates_sorted))
        for i in range(start, end):
            date_to_fold[dates_sorted[i]] = fold
    df["_fold"] = df["as_of_date"].map(date_to_fold)

    mb = config.get("model_b", {})
    xgb_cfg = mb.get("xgb", {})
    rf_cfg = mb.get("rf", {})
    es = xgb_cfg.get("early_stopping_rounds", 20)

    oof_rows = []
    for fold in range(n_folds):
        train_mask = df["_fold"] != fold
        val_mask = df["_fold"] == fold
        X_train = df.loc[train_mask, feat_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, "y"].values.astype(np.float32)
        X_val = df.loc[val_mask, feat_cols].values.astype(np.float32)
        y_val = df.loc[val_mask, "y"].values.astype(np.float32)
        if X_train.size == 0 or X_val.size == 0:
            continue
        xgb_m = build_xgb(xgb_cfg)
        fit_xgb(xgb_m, X_train, y_train, X_val, y_val, early_stopping_rounds=es)
        rf_m = build_rf(rf_cfg)
        fit_rf(rf_m, X_train, y_train)
        oof_xgb = xgb_m.predict(X_val).astype(np.float32)
        oof_rf = rf_m.predict(X_val).astype(np.float32)
        val_df = df.loc[val_mask, ["team_id", "as_of_date", "y"]].copy()
        val_df["oof_xgb"] = oof_xgb
        val_df["oof_rf"] = oof_rf
        oof_rows.append(val_df)
        print(f"Fold {fold+1}/{n_folds} OOF collected {len(val_df)} rows")

    if oof_rows:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_path = out / "oof_model_b.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"Wrote {oof_path} ({len(oof_df)} rows)")
    else:
        print(
            "No OOF rows collected (need at least 2 folds with non-empty train/val).",
            file=sys.stderr,
        )

    # Final models on full data
    X = df[feat_cols].values.astype(np.float32)
    y = df["y"].values.astype(np.float32)
    dates_sorted_full = sorted(df["as_of_date"].unique())
    n_val = max(1, int(0.2 * len(dates_sorted_full)))
    val_dates = set(dates_sorted_full[-n_val:])
    val_mask = df["as_of_date"].isin(val_dates)
    X_train = X[~val_mask]
    y_train = y[~val_mask]
    X_val = X[val_mask] if val_mask.any() else None
    y_val = y[val_mask] if val_mask.any() else None
    p1, p2 = train_model_b(X_train, y_train, X_val, y_val, config, feat_cols, out)
    print(f"Saved {p1}, {p2}")


if __name__ == "__main__":
    main()
