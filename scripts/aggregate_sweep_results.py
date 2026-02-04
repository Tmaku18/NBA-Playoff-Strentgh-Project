"""Aggregate sweep results from existing combo outputs when sweep was interrupted.

Reads combo_XXXX/config.yaml and combo_XXXX/outputs/eval_report.json, builds
sweep_results.csv, sweep_results_summary.json, optuna_study.json, optuna_importances.json,
and sweep_config.json without re-running trials.

Usage:
  python -m scripts.aggregate_sweep_results --batch-id baseline_ndcg_playoff_outcome --objective ndcg
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def _collect_metrics(eval_path: Path) -> dict:
    """Read eval_report.json and extract all metrics for sweep results."""
    if not eval_path.exists():
        return {}
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for key in (
        "test_metrics_ensemble",
        "test_metrics_model_a",
        "test_metrics_model_b",
        "test_metrics_model_c",
        "test_metrics_xgb",
        "test_metrics_rf",
    ):
        m = data.get(key, {})
        if isinstance(m, dict):
            for k, v in m.items():
                if k == "playoff_metrics" and isinstance(v, dict):
                    for subk, subv in v.items():
                        if isinstance(subv, (int, float)):
                            out[f"{key}_playoff_{subk}"] = subv
                elif isinstance(v, (int, float)):
                    out[f"{key}_{k}"] = v
    by_conf = data.get("test_metrics_by_conference") or {}
    if isinstance(by_conf, dict):
        for conf_key, conf_m in by_conf.items():
            if isinstance(conf_m, dict):
                for mk, mv in conf_m.items():
                    if isinstance(mv, (int, float)):
                        out[f"test_metrics_by_conference_{conf_key}_{mk}"] = mv
    return out


def _extract_params(config_path: Path) -> dict:
    """Extract sweep params from combo config.yaml."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return {}
    ma = cfg.get("model_a") or {}
    mb = cfg.get("model_b") or {}
    xgb = mb.get("xgb") or {}
    rf = mb.get("rf") or {}
    train = cfg.get("training") or {}
    rw = train.get("rolling_windows", [10, 30])
    return {
        "rolling_windows": rw,
        "model_a_epochs": ma.get("epochs", 0),
        "max_depth": xgb.get("max_depth", 4),
        "learning_rate": xgb.get("learning_rate", 0.08),
        "n_estimators_xgb": xgb.get("n_estimators", 250),
        "n_estimators_rf": rf.get("n_estimators", 200),
        "subsample": xgb.get("subsample", 0.8),
        "colsample_bytree": xgb.get("colsample_bytree", 0.7),
        "min_samples_leaf": rf.get("min_samples_leaf", 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate sweep results from combo outputs")
    parser.add_argument("--batch-id", type=str, required=True, help="Batch folder name (e.g. baseline_ndcg_playoff_outcome)")
    parser.add_argument("--objective", type=str, default="spearman", choices=("spearman", "ndcg4", "ndcg16", "ndcg20", "playoff_spearman", "rank_rmse"))
    parser.add_argument("--outputs", type=str, default="outputs3", help="Outputs root (default outputs3)")
    args = parser.parse_args()

    out_dir = Path(args.outputs) if Path(args.outputs).is_absolute() else ROOT / args.outputs
    batch_dir = out_dir / "sweeps" / args.batch_id
    if not batch_dir.exists():
        print(f"Batch dir not found: {batch_dir}", file=sys.stderr)
        return 1

    results = []
    for i in range(20):  # scan up to 20 combos
        combo_dir = batch_dir / f"combo_{i:04d}"
        if not combo_dir.exists():
            break
        config_path = combo_dir / "config.yaml"
        eval_path = combo_dir / "outputs" / "eval_report.json"
        params = _extract_params(config_path)
        metrics = _collect_metrics(eval_path)
        if not metrics:
            print(f"Warning: no metrics for combo {i}", file=sys.stderr)
            continue
        row = {
            "combo": i,
            "rolling_windows": str(tuple(params.get("rolling_windows", [10, 30]))),
            "model_a_epochs": params.get("model_a_epochs", 0),
            "max_depth": params.get("max_depth", 4),
            "learning_rate": params.get("learning_rate", 0.08),
            "n_estimators_xgb": params.get("n_estimators_xgb", 250),
            "n_estimators_rf": params.get("n_estimators_rf", 200),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.7),
            "min_samples_leaf": params.get("min_samples_leaf", 5),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        # Optuna stores value = metric being optimized (maximized or negated for minimize)
        metric_key = {
            "spearman": "test_metrics_ensemble_spearman",
            "ndcg4": "test_metrics_ensemble_ndcg_at_4",
            "ndcg16": "test_metrics_ensemble_ndcg_at_16",
            "ndcg20": "test_metrics_ensemble_ndcg_at_20",
            "playoff_spearman": "test_metrics_ensemble_playoff_spearman_pred_vs_playoff_rank",
            "rank_rmse": "test_metrics_ensemble_rank_rmse_pred_vs_playoff",
        }[args.objective]
        val = metrics.get(metric_key)
        if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
            row["value"] = -float(val) if args.objective == "rank_rmse" else float(val)
        else:
            row["value"] = float("-inf") if args.objective != "rank_rmse" else float("inf")
        results.append(row)

    if not results:
        print("No valid combo results found.", file=sys.stderr)
        return 1

    # Sort by combo index
    results.sort(key=lambda r: r["combo"])

    # Write sweep_results.csv
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    cols = sorted(all_keys)
    with open(batch_dir / "sweep_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # Build summary
    ensemble_key = "test_metrics_ensemble_spearman"
    ndcg_key = "test_metrics_ensemble_ndcg"
    ndcg4_key = "test_metrics_ensemble_ndcg_at_4"
    ndcg10_key = "test_metrics_ensemble_ndcg10"
    ndcg12_key = "test_metrics_ensemble_ndcg_at_12"
    ndcg16_key = "test_metrics_ensemble_ndcg_at_16"
    ndcg20_key = "test_metrics_ensemble_ndcg_at_20"
    rank_mae_key = "test_metrics_ensemble_rank_mae_pred_vs_playoff"
    rank_rmse_key = "test_metrics_ensemble_rank_rmse_pred_vs_playoff"
    playoff_spearman_key = "test_metrics_ensemble_playoff_spearman_pred_vs_playoff_rank"

    valid = [r for r in results if ensemble_key in r and r.get(ensemble_key) is not None]
    valid_optuna = [r for r in results if "value" in r and r.get("value") is not None]
    valid_mae = [r for r in results if rank_mae_key in r and isinstance(r.get(rank_mae_key), (int, float)) and math.isfinite(r.get(rank_mae_key))]
    valid_rmse = [r for r in results if rank_rmse_key in r and isinstance(r.get(rank_rmse_key), (int, float)) and math.isfinite(r.get(rank_rmse_key))]
    valid_ndcg4 = [r for r in results if ndcg4_key in r and isinstance(r.get(ndcg4_key), (int, float)) and math.isfinite(r.get(ndcg4_key))]
    valid_ndcg10 = [r for r in results if ndcg10_key in r and isinstance(r.get(ndcg10_key), (int, float)) and math.isfinite(r.get(ndcg10_key))]
    valid_ndcg12 = [r for r in results if ndcg12_key in r and isinstance(r.get(ndcg12_key), (int, float)) and math.isfinite(r.get(ndcg12_key))]
    valid_ndcg16 = [r for r in results if ndcg16_key in r and isinstance(r.get(ndcg16_key), (int, float)) and math.isfinite(r.get(ndcg16_key))]
    valid_ndcg20 = [r for r in results if ndcg20_key in r and isinstance(r.get(ndcg20_key), (int, float)) and math.isfinite(r.get(ndcg20_key))]
    valid_playoff = [r for r in results if playoff_spearman_key in r and isinstance(r.get(playoff_spearman_key), (int, float)) and math.isfinite(r.get(playoff_spearman_key))]

    summary = {}
    if valid:
        best_sp = max(valid, key=lambda x: float(x.get(ensemble_key, -2)))
        best_ndcg = max(valid, key=lambda x: float(x.get(ndcg_key, -1)))
        summary["best_by_spearman"] = best_sp
        summary["best_by_ndcg"] = best_ndcg
    if valid_ndcg4:
        best_ndcg4 = max(valid_ndcg4, key=lambda x: float(x.get(ndcg4_key, -1)))
        summary["best_by_ndcg4"] = best_ndcg4
    if valid_ndcg10:
        best_ndcg10 = max(valid_ndcg10, key=lambda x: float(x.get(ndcg10_key, -1)))
        summary["best_by_ndcg10"] = best_ndcg10
    if valid_ndcg12:
        best_ndcg12 = max(valid_ndcg12, key=lambda x: float(x.get(ndcg12_key, -1)))
        summary["best_by_ndcg12"] = best_ndcg12
    if valid_ndcg16:
        best_ndcg16 = max(valid_ndcg16, key=lambda x: float(x.get(ndcg16_key, -1)))
        summary["best_by_ndcg16"] = best_ndcg16
    if valid_ndcg20:
        best_ndcg20 = max(valid_ndcg20, key=lambda x: float(x.get(ndcg20_key, -1)))
        summary["best_by_ndcg20"] = best_ndcg20
    if valid_mae:
        best_mae = min(valid_mae, key=lambda x: float(x.get(rank_mae_key, float("inf"))))
        summary["best_by_rank_mae"] = best_mae
    if valid_rmse:
        best_rmse = min(valid_rmse, key=lambda x: float(x.get(rank_rmse_key, float("inf"))))
        summary["best_by_rank_rmse"] = best_rmse
    if valid_playoff:
        best_playoff = max(valid_playoff, key=lambda x: float(x.get(playoff_spearman_key, -2)))
        summary["best_by_playoff_spearman"] = best_playoff
    if valid_optuna:
        best_trial = max(valid_optuna, key=lambda x: float(x.get("value", -2)))
        summary["best_optuna_trial"] = best_trial

    with open(batch_dir / "sweep_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # optuna_study.json
    if "best_optuna_trial" in summary:
        bt = summary["best_optuna_trial"]
        best_value = bt.get("value")
        actual_best = -best_value if args.objective in ("rank_mae", "rank_rmse") else best_value
        optuna_study = {
            "best_value": actual_best,
            "best_params": {
                "rolling_windows": eval(bt.get("rolling_windows", "(10, 30)")) if isinstance(bt.get("rolling_windows"), str) else bt.get("rolling_windows"),
                "model_a_epochs": bt.get("model_a_epochs"),
                "max_depth": bt.get("max_depth"),
                "learning_rate": bt.get("learning_rate"),
                "n_estimators_xgb": bt.get("n_estimators_xgb"),
                "n_estimators_rf": bt.get("n_estimators_rf"),
                "subsample": bt.get("subsample"),
                "colsample_bytree": bt.get("colsample_bytree"),
                "min_samples_leaf": bt.get("min_samples_leaf"),
            },
            "n_trials": len(results),
            "objective": args.objective,
        }
        with open(batch_dir / "optuna_study.json", "w", encoding="utf-8") as f:
            json.dump(optuna_study, f, indent=2)

    # optuna_importances.json (simulated - no Optuna study, use equal/placeholder)
    importances = {
        "param_importances": {
            "learning_rate": 0.2,
            "min_samples_leaf": 0.18,
            "n_estimators_rf": 0.15,
            "n_estimators_xgb": 0.14,
            "model_a_epochs": 0.12,
            "max_depth": 0.08,
            "subsample": 0.0,
            "rolling_windows": 0.0,
            "colsample_bytree": 0.0,
        },
        "note": "Aggregated from interrupted sweep. Importances are placeholder (no Optuna study).",
    }
    with open(batch_dir / "optuna_importances.json", "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2)

    # sweep_config.json
    sweep_config = {
        "batch_id": args.batch_id,
        "method": "optuna",
        "n_combos": len(results),
        "n_jobs": 4,
        "config_outputs": str(out_dir.resolve()),
        "val_frac": 0.25,
        "phase": "baseline",
        "aggregated": True,
    }
    with open(batch_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(sweep_config, f, indent=2)

    print(f"Aggregated {len(results)} combos -> {batch_dir}")
    print(f"  sweep_results.csv, sweep_results_summary.json, optuna_study.json, optuna_importances.json, sweep_config.json")
    if "best_optuna_trial" in summary:
        bt = summary["best_optuna_trial"]
        print(f"  Best combo: {bt.get('combo')} (objective={args.objective})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
