"""Run evaluation on real predictions; write outputs/eval_report.json. Requires predictions.json from script 6."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_ranking, evaluate_upset
from src.evaluation.metrics import brier_champion, ndcg_at_4, spearman


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out_dir = Path(config["paths"]["outputs"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_id = "run_001"
    pred_path = out_dir / run_id / "predictions.json"
    if not pred_path.exists():
        print("Predictions not found. Run inference (script 6) first.", file=sys.stderr)
        sys.exit(1)
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    teams = data.get("teams", [])
    if not teams:
        print("No teams in predictions.json.", file=sys.stderr)
        sys.exit(1)
    actual_ranks = []
    pred_scores = []
    pred_ranks = []
    for t in teams:
        act = t.get("analysis", {}).get("actual_rank")
        pred_rank = t.get("prediction", {}).get("predicted_rank")
        tss = t.get("prediction", {}).get("true_strength_score")
        if act is not None:
            actual_ranks.append(act)
        else:
            actual_ranks.append(0)
        pred_ranks.append(pred_rank if pred_rank is not None else 0)
        pred_scores.append(tss if tss is not None else 0.0)
    y_actual = np.array(actual_ranks, dtype=np.float32)
    y_score = np.array(pred_scores, dtype=np.float32)
    n = len(y_actual)
    # Relevance for NDCG: higher = better team; 1st rank is best so use (n - rank + 1)
    y_true_relevance = (n - y_actual + 1).clip(1, n)
    valid = y_actual > 0
    if valid.sum() < 2:
        print("Too few valid actual ranks for evaluation.", file=sys.stderr)
        sys.exit(1)
    m = evaluate_ranking(y_true_relevance, y_score, k=min(10, n))
    # Upset: sleeper = actual_rank > predicted_rank (under-ranked by standings)
    delta = y_actual - np.array(pred_ranks, dtype=np.float32)
    y_bin = (delta > 0).astype(np.float32)
    if np.unique(y_bin).size >= 2:
        m2 = evaluate_upset(y_bin, y_score)
        m.update(m2)
    else:
        m["roc_auc_upset"] = 0.5
    m["notes"] = {
        "upset_definition": "sleeper = actual_rank > predicted_rank (under-ranked by standings)",
        "mrr": "top_k=2; 1/rank of first max-relevance item in predicted order (two conferences).",
    }

    # Playoff metrics (when playoff_rank and championship_odds present)
    playoff_ranks = [t.get("analysis", {}).get("playoff_rank") for t in teams]
    if any(r is not None for r in playoff_ranks):
        p_rank = np.array([r if r is not None else 0 for r in playoff_ranks], dtype=np.float32)
        g_rank = np.array([t.get("prediction", {}).get("global_rank") or t.get("prediction", {}).get("predicted_rank") or 0 for t in teams], dtype=np.float32)
        odds_str = [t.get("prediction", {}).get("championship_odds", "0%") for t in teams]
        odds_pct = np.array([float(s.rstrip("%")) / 100.0 for s in odds_str], dtype=np.float32)
        champion_onehot = (p_rank == 1).astype(np.float32)
        m["playoff_metrics"] = {
            "spearman_pred_vs_playoff_rank": float(spearman(p_rank, g_rank)),
            "ndcg_at_4_final_four": float(ndcg_at_4(p_rank, -g_rank)),
            "brier_championship_odds": float(brier_champion(champion_onehot, odds_pct)),
        }
        m["notes"]["playoff_metrics"] = "Spearman (pred global vs playoff rank), NDCG@4 (final four), Brier (champion vs odds)."

    out = out_dir / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
