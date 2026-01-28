"""Inference: load A/B/stacker, produce per-team JSON (predicted_rank, true_strength, delta, ensemble, contributors)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_models(
    model_a_path: str | Path | None = None,
    xgb_path: str | Path | None = None,
    rf_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    config: dict | None = None,
):
    """Load Model A, XGB, RF, RidgeCV meta. Returns (model_a, xgb, rf, meta) or Nones."""
    from src.models.deep_set_rank import DeepSetRank

    model_a, xgb, rf, meta = None, None, None, None
    cfg = config or {}
    ma = cfg.get("model_a", {})

    if model_a_path and Path(model_a_path).exists():
        ck = torch.load(model_a_path, map_location="cpu", weights_only=False)
        model_a = DeepSetRank(
            ma.get("num_embeddings", 500),
            ma.get("embedding_dim", 32),
            7,
            ma.get("encoder_hidden", [128, 64]),
            ma.get("attention_heads", 4),
            ma.get("dropout", 0.2),
        )
        if "model_state" in ck:
            model_a.load_state_dict(ck["model_state"])
        model_a.eval()

    if xgb_path and Path(xgb_path).exists():
        import joblib
        xgb = joblib.load(xgb_path)
    if rf_path and Path(rf_path).exists():
        import joblib
        rf = joblib.load(rf_path)
    if meta_path and Path(meta_path).exists():
        import joblib
        meta = joblib.load(meta_path)

    return model_a, xgb, rf, meta


def predict_teams(
    team_ids: list[int],
    team_names: list[str],
    model_a_scores: np.ndarray | None = None,
    xgb_scores: np.ndarray | None = None,
    rf_scores: np.ndarray | None = None,
    meta_model: Any = None,
    actual_ranks: dict[int, int] | None = None,
    attention_by_team: dict[int, list[tuple[str, float]]] | None = None,
    *,
    true_strength_scale: str = "percentile",
) -> list[dict]:
    """
    Combine base scores, run meta if present. For each team output:
    predicted_rank, true_strength_score, delta (actual - predicted), classification, ensemble_diagnostics, primary_contributors.
    """
    n = len(team_ids)
    if model_a_scores is not None and len(model_a_scores) == n:
        sa = np.asarray(model_a_scores).ravel()
    else:
        sa = np.zeros(n)
    if xgb_scores is not None and len(xgb_scores) == n:
        sx = np.asarray(xgb_scores).ravel()
    else:
        sx = np.zeros(n)
    if rf_scores is not None and len(rf_scores) == n:
        sr = np.asarray(rf_scores).ravel()
    else:
        sr = np.zeros(n)

    X = np.column_stack([sa, sx, sr])
    if meta_model is not None:
        ens = meta_model.predict(X).ravel()
    else:
        ens = (sa + sx + sr) / 3.0

    pred_rank = np.argsort(np.argsort(-ens)) + 1
    if true_strength_scale == "percentile":
        tss = (np.argsort(np.argsort(ens)) + 1).astype(float) / (n + 1)
    else:
        tss = (ens - ens.min()) / (ens.max() - ens.min() + 1e-12)

    actual_ranks = actual_ranks or {}
    attention_by_team = attention_by_team or {}

    out = []
    for i, (tid, tname) in enumerate(zip(team_ids, team_names)):
        act = actual_ranks.get(tid)
        delta = (act - pred_rank[i]) if act is not None else None
        if delta is not None:
            if delta > 0:
                classification = f"Sleeper (Under-ranked by {delta} slots)"
            elif delta < 0:
                classification = f"Paper Tiger (Over-ranked by {-delta} slots)"
            else:
                classification = "Aligned"
        else:
            classification = "Unknown"

        r_a = np.argsort(np.argsort(-sa))[i] + 1 if len(sa) == n else None
        r_x = np.argsort(np.argsort(-sx))[i] + 1 if len(sx) == n else None
        r_r = np.argsort(np.argsort(-sr))[i] + 1 if len(sr) == n else None
        spread = max(r or 0 for r in [r_a, r_x, r_r]) - min(r or 0 for r in [r_a, r_x, r_r]) if any([r_a, r_x, r_r]) else 0
        agreement = "High" if spread <= 2 else "Low"

        contrib = attention_by_team.get(tid, [])

        out.append({
            "team_id": int(tid),
            "team_name": tname,
            "prediction": {"predicted_rank": int(pred_rank[i]), "true_strength_score": float(tss[i])},
            "analysis": {"actual_rank": int(act) if act is not None else None, "classification": classification},
            "ensemble_diagnostics": {"model_agreement": agreement, "deep_set_rank": int(r_a) if r_a is not None else None, "xgboost_rank": int(r_x) if r_x is not None else None, "random_forest_rank": int(r_r) if r_r is not None else None},
            "roster_dependence": {"primary_contributors": [{"player": str(p), "attention_weight": float(w)} for p, w in contrib]},
        })
    return out


def run_inference(output_dir: str | Path, config: dict, run_id: str | None = None) -> Path:
    """Dummy: 3 teams, placeholder scores. Writes predictions.json and a scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    if run_id:
        out = out / run_id
    out.mkdir(parents=True, exist_ok=True)

    paths = config.get("paths", {})
    model_a, xgb, rf, meta = load_models(
        model_a_path=Path(paths.get("outputs", "outputs")) / "best_deep_set.pt",
        xgb_path=Path(paths.get("outputs", "outputs")) / "xgb_model.joblib",
        rf_path=Path(paths.get("outputs", "outputs")) / "rf_model.joblib",
        meta_path=Path(paths.get("outputs", "outputs")) / "ridgecv_meta.joblib",
        config=config,
    )

    team_ids = [1, 2, 3]
    team_names = ["Team A", "Team B", "Team C"]
    sa = np.array([0.5, 0.3, 0.8])
    sx = np.array([0.4, 0.5, 0.6])
    sr = np.array([0.6, 0.4, 0.5])
    actual = {1: 2, 2: 3, 3: 1}

    preds = predict_teams(team_ids, team_names, sa, sx, sr, meta, actual_ranks=actual, true_strength_scale=config.get("output", {}).get("true_strength_scale", "percentile"))

    pj = out / "predictions.json"
    with open(pj, "w", encoding="utf-8") as f:
        json.dump({"teams": preds}, f, indent=2)

    # predicted vs actual scatter
    fig, ax = plt.subplots()
    pr = [t["prediction"]["predicted_rank"] for t in preds]
    ar = [t["analysis"]["actual_rank"] for t in preds]
    ar = [a if a is not None else 0 for a in ar]
    ax.scatter(ar, pr, label="teams")
    ax.plot([0, 4], [0, 4], "k--", alpha=0.5, label="identity")
    ax.set_xlabel("Actual rank")
    ax.set_ylabel("Predicted rank")
    ax.legend()
    ax.set_title("Predicted vs actual rank")
    fig.savefig(out / "pred_vs_actual.png", bbox_inches="tight")
    plt.close()

    return pj
