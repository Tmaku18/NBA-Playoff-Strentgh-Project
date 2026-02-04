"""Hyperparameter sweep: Model A epochs, Model B grid or Optuna Bayesian tuning.

Runs in the foreground (no background/daemon mode). Writes results to
<config.paths.outputs>/sweeps/<batch_id>/ (e.g. outputs3/sweeps/<batch_id>/).
No artificial timeout.

Usage:
  python -m scripts.sweep_hparams [--batch-id BATCH_ID] [--dry-run] [--max-combos N]
  python -m scripts.sweep_hparams [--val-frac FRAC] [--phase full|phase1_xgb|phase2_rf]
  python -m scripts.sweep_hparams --method optuna [--n-trials N] [--objective spearman|ndcg|playoff_spearman|rank_mae]

--method grid: Full grid search (default). Config sweep section is a smaller default grid; expand in config or use --phase for phased grids.
--method optuna: Bayesian optimization with Optuna; --objective sets the metric to optimize.
--method halving: Successive halving (cheap epochs then full epochs on top 1/factor).
--batch-id: Batch folder name (default: timestamp).
--val-frac: Model A early-stopping validation fraction (default 0.25).
--phase: full=config grid; phase1_xgb=XGB local grid (epochs 8-28, lr/n_xgb varied); phase2_rf=RF local grid (XGB fixed, n_rf/min_leaf varied).
--dry-run: Print combo count and config overrides without running.
--max-combos: Limit number of combos (grid only).
--n-trials: Number of Optuna trials (default 20).
--n-jobs: Number of parallel workers (default 1). Each job runs one full pipeline; use 4-8 on multi-core.
After Optuna, optuna_importances.json lists param importance; fix low-importance params and re-run a smaller search.
After the sweep, explain (5b_explain) is run on the best combo for the chosen objective.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def _run_one_combo_worker(
    batch_dir: Path,
    combo_idx: int,
    config: dict,
    rolling_windows: list,
    epochs: int,
    max_depth: int,
    lr: float,
    n_xgb: int,
    n_rf: int,
    subsample: float,
    colsample: float,
    min_leaf: int,
    include_clone: bool,
    val_frac: float,
    listmle_target: str | None = None,
) -> dict:
    """Top-level worker for ProcessPoolExecutor; runs one combo and returns metrics or error."""
    return _run_one_combo(
        batch_dir,
        combo_idx,
        config,
        list(rolling_windows),
        epochs,
        max_depth,
        lr,
        n_xgb,
        n_rf,
        subsample,
        colsample,
        min_leaf,
        include_clone,
        val_frac=val_frac,
        listmle_target=listmle_target,
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Modifies base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(config_path: str | Path | None = None) -> dict:
    defaults_path = ROOT / "config" / "defaults.yaml"
    with open(defaults_path, "r", encoding="utf-8") as f:
        config = copy.deepcopy(yaml.safe_load(f))
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists() and path != defaults_path:
            with open(path, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f)
            if overrides:
                _deep_merge(config, overrides)
    # From any worktree: use canonical DB with playoff data if NBA_DB_PATH is set
    db_override = __import__("os").environ.get("NBA_DB_PATH")
    if db_override and db_override.strip():
        config.setdefault("paths", {})["db"] = db_override.strip()
    return config


def _write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _run_cmd(script_path: str, extra_args: list[str], cwd: Path | None = None) -> int:
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, str(ROOT / script_path)] + extra_args,
        cwd=str(cwd or ROOT),
        env=env,
    )
    return r.returncode


def _collect_clone_metrics(report_path: Path) -> dict:
    """Read clone_classifier_report.json and extract AUC, Brier for sweep results."""
    if not report_path.exists():
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for key in ("val_auc", "val_brier", "holdout_auc", "holdout_brier", "train_auc", "train_brier"):
        if key in data and isinstance(data[key], (int, float)):
            out[f"clone_{key}"] = data[key]
    return out


def _collect_metrics(eval_path: Path) -> dict:
    """Read eval_report.json and extract all metrics for sweep results.

    Collects: test_metrics_ensemble/model_a/xgb/rf (scalars + flattened playoff_metrics),
    test_metrics_by_conference (E/W ndcg, spearman). Ensures sweep CSV and summary
    include ndcg, spearman, rank_mae, playoff metrics, and per-conference metrics.
    """
    if not eval_path.exists():
        return {}
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for key in ("test_metrics_ensemble", "test_metrics_model_a", "test_metrics_model_b", "test_metrics_model_c", "test_metrics_xgb", "test_metrics_rf"):
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


def _run_one_combo(
    batch_dir: Path,
    combo_idx: int,
    config: dict,
    rolling_windows: list,
    epochs: int,
    max_depth: int,
    lr: float,
    n_xgb: int,
    n_rf: int,
    subsample: float,
    colsample: float,
    min_leaf: int,
    include_clone: bool,
    val_frac: float = 0.25,
    listmle_target: str | None = None,
) -> dict:
    """Run one pipeline (3, 4, 4b, 6, 5) with given params; return metrics dict or error."""
    combo_dir = batch_dir / f"combo_{combo_idx:04d}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    combo_out = combo_dir / "outputs"
    combo_out.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg["training"] = cfg.get("training", {})
    cfg["training"]["rolling_windows"] = list(rolling_windows)
    if listmle_target is not None:
        cfg["training"]["listmle_target"] = listmle_target
    cfg["model_a"] = cfg.get("model_a", {})
    cfg["model_a"]["epochs"] = int(epochs)
    cfg["model_a"]["early_stopping_val_frac"] = float(val_frac)
    cfg["model_b"] = cfg.get("model_b", {})
    cfg["model_b"]["xgb"] = cfg["model_b"].get("xgb", {})
    cfg["model_b"]["xgb"]["max_depth"] = int(max_depth)
    cfg["model_b"]["xgb"]["learning_rate"] = float(lr)
    cfg["model_b"]["xgb"]["n_estimators"] = int(n_xgb)
    cfg["model_b"]["xgb"]["subsample"] = float(subsample)
    cfg["model_b"]["xgb"]["colsample_bytree"] = float(colsample)
    cfg["model_b"]["rf"] = cfg["model_b"].get("rf", {})
    cfg["model_b"]["rf"]["n_estimators"] = int(n_rf)
    cfg["model_b"]["rf"]["min_samples_leaf"] = int(min_leaf)
    cfg["paths"] = cfg.get("paths", {})
    cfg["paths"]["outputs"] = str(combo_out.resolve())
    config_path = combo_dir / "config.yaml"
    _write_config(config_path, cfg)
    pipeline = [
        ("scripts/3_train_model_a.py", "Model A"),
        ("scripts/4_train_models_b_and_c.py", "Models B & C"),
        ("scripts/4b_train_stacking.py", "Stacking"),
        ("scripts/6_run_inference.py", "Inference"),
        ("scripts/5_evaluate.py", "Eval"),
    ]
    if include_clone:
        pipeline.append(("scripts/4c_train_classifier_clone.py", "Clone Classifier"))
    for script, name in pipeline:
        code = _run_cmd(script, ["--config", str(config_path)])
        if code != 0:
            return {"error": name}
    return _collect_metrics(combo_out / "eval_report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--batch-id", type=str, default=None, help="Batch folder name; default: timestamp")
    parser.add_argument("--dry-run", action="store_true", help="Print combos without running")
    parser.add_argument("--max-combos", type=int, default=None, help="Limit number of combos (grid only)")
    parser.add_argument("--method", type=str, default="grid", choices=("grid", "optuna", "halving"), help="grid, optuna (Bayesian), or halving (successive halving)")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials (default 20)")
    parser.add_argument(
        "--objective",
        type=str,
        default="spearman",
        choices=("spearman", "ndcg", "ndcg10", "playoff_spearman", "rank_mae", "rank_rmse"),
        help="Optuna: metric to optimize (rank_mae/rank_rmse = minimize; others = maximize). Default spearman.",
    )
    parser.add_argument("--no-run-explain", action="store_true", help="Skip running 5b_explain on best combo after sweep")
    parser.add_argument("--val-frac", type=float, default=0.25, help="Model A early-stopping validation fraction (default 0.25)")
    parser.add_argument(
        "--phase",
        type=str,
        default="full",
        choices=("full", "phase1_xgb", "phase2_rf", "baseline"),
        help="full=config grid; phase1_xgb/phase2_rf=phased Model B; baseline=wide ranges for exploratory (Optuna only)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: config/defaults.yaml)")
    parser.add_argument(
        "--listmle-target",
        type=str,
        default=None,
        choices=("final_rank", "playoff_outcome"),
        help="Override training.listmle_target (final_rank=standings, playoff_outcome=playoff result). Omit to use config.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default 1). Each runs one full pipeline; use 4-8 on multi-core.",
    )
    parser.add_argument("--halving-factor", type=int, default=3, help="Halving: keep top 1/factor after cheap round (default 3)")
    parser.add_argument("--halving-epochs-cheap", type=int, default=8, help="Halving: Model A epochs in round 1 (default 8)")
    parser.add_argument("--halving-epochs-full", type=int, default=None, help="Halving: Model A epochs in round 2 (default: max of sweep epoch list)")
    args = parser.parse_args()

    config = _load_config(args.config)
    out_name = config.get("paths", {}).get("outputs", "outputs")
    out_dir = Path(out_name) if Path(out_name).is_absolute() else ROOT / out_name
    sweeps_dir = out_dir / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = sweeps_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    val_frac = float(args.val_frac)
    phase = args.phase
    n_jobs = max(1, int(args.n_jobs))
    print(f"Sweep batch {batch_id}: val_frac={val_frac}, phase={phase}, method={args.method}, n_jobs={n_jobs}", flush=True)

    sweep_cfg = config.get("sweep", {})
    include_clone = sweep_cfg.get("include_clone_classifier", False)
    epochs_list = sweep_cfg.get("model_a_epochs", [8, 12, 16, 20, 24, 28])
    rolling_list = sweep_cfg.get("rolling_windows", [[10, 30]])
    mb = sweep_cfg.get("model_b", {})
    max_depth_list = mb.get("max_depth", [4])
    lr_list = mb.get("learning_rate", [0.08])
    n_xgb_list = mb.get("n_estimators_xgb", [250])
    n_rf_list = mb.get("n_estimators_rf", [200])
    subsample_list = mb.get("subsample", [0.8])
    colsample_list = mb.get("colsample_bytree", [0.7])
    min_leaf_list = mb.get("min_samples_leaf", [5])
    if not isinstance(subsample_list, list):
        subsample_list = [subsample_list]
    if not isinstance(colsample_list, list):
        colsample_list = [colsample_list]
    if not isinstance(min_leaf_list, list):
        min_leaf_list = [min_leaf_list]

    # Phased Model B grids (refined sweep plan): override lists when --phase is set
    if phase == "phase1_xgb":
        # Phase 1 (XGB local): max_depth=4 fixed; lr {0.08, 0.10, 0.12}; n_xgb {250, 300, 350}; subsample=0.8; colsample=0.8; RF fixed
        epochs_list = list(range(8, 29))  # 8–28 step 1
        max_depth_list = [4]
        lr_list = [0.08, 0.10, 0.12]
        n_xgb_list = [250, 300, 350]
        n_rf_list = [200]
        subsample_list = [0.8]
        colsample_list = [0.8]
        min_leaf_list = [5]
    elif phase == "phase2_rf":
        # Phase 2 (RF local): XGB fixed (d=4, lr=0.10, n_xgb=300, sub=0.8, col=0.8); RF n_estimators {150, 200, 250}; min_leaf {4, 5, 6}
        epochs_list = list(range(8, 29))
        max_depth_list = [4]
        lr_list = [0.10]
        n_xgb_list = [300]
        n_rf_list = [150, 200, 250]
        subsample_list = [0.8]
        colsample_list = [0.8]
        min_leaf_list = [4, 5, 6]
    elif phase == "baseline":
        # Phase 0 (baseline): wide ranges for exploratory sweeps; smallest combo covering widest range
        epochs_list = list(range(8, 29))  # 8-28
        max_depth_list = [3, 4, 5, 6]
        lr_list = [0.05, 0.08, 0.10, 0.12]
        n_xgb_list = [200, 250, 300, 350]
        n_rf_list = [150, 200, 250]
        subsample_list = [0.8]
        colsample_list = [0.7]
        min_leaf_list = [4, 5, 6]

    listmle_target = getattr(args, "listmle_target", None)

    if args.method == "optuna":
        import optuna
        _OBJECTIVE_KEYS = {
            "spearman": "test_metrics_ensemble_spearman",
            "ndcg": "test_metrics_ensemble_ndcg",
            "ndcg10": "test_metrics_ensemble_ndcg10",
            "playoff_spearman": "test_metrics_ensemble_playoff_spearman_pred_vs_playoff_rank",
            "rank_mae": "test_metrics_ensemble_rank_mae_pred_vs_playoff",
            "rank_rmse": "test_metrics_ensemble_rank_rmse_pred_vs_playoff",
        }
        metric_key = _OBJECTIVE_KEYS[args.objective]
        direction = "minimize" if args.objective in ("rank_mae", "rank_rmse") else "maximize"

        def objective(trial: "optuna.Trial") -> float:
            rolling_windows = trial.suggest_categorical("rolling_windows", [tuple(x) for x in rolling_list])
            epochs = trial.suggest_int("model_a_epochs", min(epochs_list), max(epochs_list))
            max_depth = trial.suggest_int("max_depth", min(max_depth_list), max(max_depth_list))
            lr = trial.suggest_float("learning_rate", min(lr_list), max(lr_list), log=True)
            n_xgb = trial.suggest_int("n_estimators_xgb", min(n_xgb_list), max(n_xgb_list))
            n_rf = trial.suggest_int("n_estimators_rf", min(n_rf_list), max(n_rf_list))
            subsample = trial.suggest_float("subsample", min(subsample_list), max(subsample_list))
            colsample = trial.suggest_float("colsample_bytree", min(colsample_list), max(colsample_list))
            min_leaf = trial.suggest_int("min_samples_leaf", min(min_leaf_list), max(min_leaf_list))
            i = trial.number
            print(f"[Optuna trial {i+1}/{args.n_trials}] rolling={rolling_windows}, epochs={epochs}, xgb d={max_depth} lr={lr:.4f} n_xgb={n_xgb} n_rf={n_rf}", flush=True)
            metrics = _run_one_combo(
                batch_dir, i, config, list(rolling_windows), epochs,
                max_depth, lr, n_xgb, n_rf, subsample, colsample, min_leaf, include_clone,
                val_frac=val_frac,
                listmle_target=listmle_target,
            )
            if "error" in metrics:
                print(f"  FAILED: {metrics['error']}", flush=True)
                if metrics["error"] == "Inference":
                    print("Sweep aborted: inference failed. Exiting.", flush=True)
                    sys.exit(1)
                return float("-inf")
            val = metrics.get(metric_key)
            if val is None:
                return float("-inf") if direction == "maximize" else float("inf")
            v = float(val)
            return -v if direction == "minimize" else v

        study = optuna.create_study(direction="maximize")  # we negate for rank_mae above
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=False,
            n_jobs=n_jobs,
        )
        results = []
        for t in study.trials:
            combo_out = batch_dir / f"combo_{t.number:04d}" / "outputs"
            metrics = _collect_metrics(combo_out / "eval_report.json") if combo_out.exists() else {}
            row = {
                "combo": t.number,
                "value": t.value,
                **{k: v for k, v in t.params.items()},
            }
            if t.value is None:
                row["error"] = "failed"
                row["value"] = None
            else:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        row[k] = v
            results.append(row)
        with open(batch_dir / "optuna_study.json", "w", encoding="utf-8") as f:
            json.dump({"best_value": study.best_value, "best_params": study.best_params, "n_trials": len(study.trials), "objective": args.objective}, f, indent=2)
        # Hyperparameter importance (Fanova): use to fix low-importance params and re-run smaller search
        try:
            import optuna.importance
            importances = optuna.importance.get_param_importances(study)
            with open(batch_dir / "optuna_importances.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "param_importances": importances,
                        "note": "Higher = more important. Fix low-importance params to best/default and re-run smaller grid or Optuna on the rest.",
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: could not compute Optuna importances: {e}", flush=True)
        actual_best = -study.best_value if args.objective == "rank_mae" else study.best_value
        print(f"Optuna best {metric_key}={actual_best:.4f} (objective={args.objective}) params={study.best_params}", flush=True)
    elif args.method == "halving":
        combos = list(itertools.product(
            rolling_list,
            epochs_list,
            max_depth_list,
            lr_list,
            n_xgb_list,
            n_rf_list,
            subsample_list,
            colsample_list,
            min_leaf_list,
        ))
        if args.max_combos:
            combos = combos[: args.max_combos]
        halving_factor = max(1, int(args.halving_factor))
        epochs_cheap = int(args.halving_epochs_cheap)
        epochs_full = int(args.halving_epochs_full) if args.halving_epochs_full is not None else max(epochs_list)
        keep = max(1, len(combos) // halving_factor)
        print(f"Halving: {len(combos)} combos, round 1 epochs={epochs_cheap}, keep top {keep}, round 2 epochs={epochs_full}", flush=True)
        if args.dry_run:
            for i, c in enumerate(combos[:5]):
                rw, ep, md, lr, nx, nr = c[0], c[1], c[2], c[3], c[4], c[5]
                sub = c[6] if len(c) > 6 else 0.8
                col = c[7] if len(c) > 7 else 0.7
                mleaf = c[8] if len(c) > 8 else 5
                print(f"  {i}: rolling={rw}, epochs={ep}, max_depth={md}, lr={lr}, n_xgb={nx}, n_rf={nr}, subsample={sub}, colsample={col}, min_leaf={mleaf}")
            if len(combos) > 5:
                print(f"  ... and {len(combos) - 5} more")
            return 0
        # Round 1: run all with cheap epochs
        halving_metric_key = "test_metrics_ensemble_spearman"
        round1_results = []
        for i, c in enumerate(combos):
            rw, ep, md, lr_v, nx, nr, sub, col, mleaf = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]
            print(f"[halving 1/{len(combos)}] combo {i} epochs={epochs_cheap}", flush=True)
            m = _run_one_combo(
                batch_dir, i, config, list(rw), epochs_cheap,
                md, lr_v, nx, nr, sub, col, mleaf, include_clone,
                val_frac=val_frac,
                listmle_target=listmle_target,
            )
            if "error" in m and m["error"] == "Inference":
                print("Sweep aborted: inference failed. Exiting.", flush=True)
                sys.exit(1)
            val = m.get(halving_metric_key)
            if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                round1_results.append((i, c, float(val)))
            else:
                round1_results.append((i, c, float("-inf")))
        round1_results.sort(key=lambda x: x[2], reverse=True)
        survivors = round1_results[:keep]
        print(f"Halving round 2: running {len(survivors)} survivors with epochs={epochs_full}", flush=True)
        results = []
        for idx, (i, c, _) in enumerate(survivors):
            rw, ep, md, lr_v, nx, nr, sub, col, mleaf = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]
            print(f"[halving 2/{len(survivors)}] combo {i} epochs={epochs_full}", flush=True)
            metrics = _run_one_combo(
                batch_dir, i, config, list(rw), epochs_full,
                md, lr_v, nx, nr, sub, col, mleaf, include_clone,
                val_frac=val_frac,
                listmle_target=listmle_target,
            )
            rolling_windows, epochs, max_depth, lr, n_xgb, n_rf, subsample, colsample, min_leaf = rw, ep, md, lr_v, nx, nr, sub, col, mleaf
            if "error" in metrics:
                if metrics["error"] == "Inference":
                    print("Sweep aborted: inference failed. Exiting.", flush=True)
                    sys.exit(1)
                results.append({
                    "combo": i,
                    "rolling_windows": str(rolling_windows),
                    "epochs": epochs_full,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "n_xgb": n_xgb,
                    "n_rf": n_rf,
                    "subsample": subsample,
                    "colsample_bytree": colsample,
                    "min_samples_leaf": min_leaf,
                    "error": metrics["error"],
                })
            else:
                results.append({
                    "combo": i,
                    "rolling_windows": str(rolling_windows),
                    "epochs": epochs_full,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "n_xgb": n_xgb,
                    "n_rf": n_rf,
                    "subsample": subsample,
                    "colsample_bytree": colsample,
                    "min_samples_leaf": min_leaf,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                })
    else:
        combos = list(itertools.product(
            rolling_list,
            epochs_list,
            max_depth_list,
            lr_list,
            n_xgb_list,
            n_rf_list,
            subsample_list,
            colsample_list,
            min_leaf_list,
        ))
        if args.max_combos:
            combos = combos[: args.max_combos]

        print(f"Sweep: {len(combos)} combos in {batch_dir}", flush=True)
        if args.dry_run:
            for i, c in enumerate(combos[:5]):
                rw, ep, md, lr, nx, nr = c[0], c[1], c[2], c[3], c[4], c[5]
                sub = c[6] if len(c) > 6 else 0.8
                col = c[7] if len(c) > 7 else 0.7
                mleaf = c[8] if len(c) > 8 else 5
                print(f"  {i}: rolling={rw}, epochs={ep}, max_depth={md}, lr={lr}, n_xgb={nx}, n_rf={nr}, subsample={sub}, colsample={col}, min_leaf={mleaf}")
            if len(combos) > 5:
                print(f"  ... and {len(combos) - 5} more")
            return 0

        results = []
        if n_jobs <= 1:
            for i, (rolling_windows, epochs, max_depth, lr, n_xgb, n_rf, subsample, colsample, min_leaf) in enumerate(combos):
                print(f"[{i+1}/{len(combos)}] rolling={rolling_windows}, epochs={epochs}, xgb d={max_depth} lr={lr} n_xgb={n_xgb} n_rf={n_rf} sub={subsample} col={colsample} min_leaf={min_leaf}", flush=True)
                metrics = _run_one_combo(
                    batch_dir, i, config, list(rolling_windows), epochs,
                    max_depth, lr, n_xgb, n_rf, subsample, colsample, min_leaf, include_clone,
                    val_frac=val_frac,
                    listmle_target=listmle_target,
                )
                if "error" in metrics:
                    print(f"  FAILED at {metrics['error']}", flush=True)
                    if metrics["error"] == "Inference":
                        print("Sweep aborted: inference failed. Exiting.", flush=True)
                        sys.exit(1)
                    results.append({
                        "combo": i,
                        "rolling_windows": str(rolling_windows),
                        "epochs": epochs,
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_xgb": n_xgb,
                        "n_rf": n_rf,
                        "subsample": subsample,
                        "colsample_bytree": colsample,
                        "min_samples_leaf": min_leaf,
                        "error": metrics["error"],
                    })
                else:
                    results.append({
                        "combo": i,
                        "rolling_windows": str(rolling_windows),
                        "epochs": epochs,
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_xgb": n_xgb,
                        "n_rf": n_rf,
                        "subsample": subsample,
                        "colsample_bytree": colsample,
                        "min_samples_leaf": min_leaf,
                        **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                    })
        else:
            def _make_result(i: int, combo: tuple, metrics: dict) -> dict:
                rolling_windows, epochs, max_depth, lr, n_xgb, n_rf, subsample, colsample, min_leaf = combo
                if "error" in metrics:
                    return {
                        "combo": i,
                        "rolling_windows": str(rolling_windows),
                        "epochs": epochs,
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_xgb": n_xgb,
                        "n_rf": n_rf,
                        "subsample": subsample,
                        "colsample_bytree": colsample,
                        "min_samples_leaf": min_leaf,
                        "error": metrics["error"],
                    }
                return {
                    "combo": i,
                    "rolling_windows": str(rolling_windows),
                    "epochs": epochs,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "n_xgb": n_xgb,
                    "n_rf": n_rf,
                    "subsample": subsample,
                    "colsample_bytree": colsample,
                    "min_samples_leaf": min_leaf,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {}
                for i, c in enumerate(combos):
                    rw, ep, md, lr_v, nx, nr, sub, col, mleaf = c
                    fut = executor.submit(
                        _run_one_combo_worker,
                        batch_dir, i, config, list(rw), ep, md, lr_v, nx, nr, sub, col, mleaf,
                        include_clone, val_frac,
                        listmle_target,
                    )
                    futures[fut] = (i, c)
                for fut in concurrent.futures.as_completed(futures):
                    i, c = futures[fut]
                    try:
                        metrics = fut.result()
                    except Exception as e:
                        metrics = {"error": str(e)}
                    if "error" in metrics:
                        print(f"  combo {i} FAILED at {metrics['error']}", flush=True)
                        if metrics["error"] == "Inference":
                            print("Sweep aborted: inference failed. Exiting.", flush=True)
                            for f in futures:
                                f.cancel()
                            sys.exit(1)
                    results.append(_make_result(i, c, metrics))
            # Reorder results by combo index so CSV and summary match combo_0000, combo_0001, ...
            results.sort(key=lambda r: r["combo"])

    # Write results
    if results:
        import csv
        # Union of all keys so CSV includes every metric (e.g. when first Optuna trial failed)
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        cols = sorted(all_keys)
        with open(batch_dir / "sweep_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

    # Summary: best by spearman, ndcg, rank_mae (lower is better), etc. (grid) or by value (optuna)
    ensemble_key = "test_metrics_ensemble_spearman"
    ndcg_key = "test_metrics_ensemble_ndcg"
    ndcg10_key = "test_metrics_ensemble_ndcg10"
    rank_mae_key = "test_metrics_ensemble_rank_mae_pred_vs_playoff"
    rank_rmse_key = "test_metrics_ensemble_rank_rmse_pred_vs_playoff"
    playoff_spearman_key = "test_metrics_ensemble_playoff_spearman_pred_vs_playoff_rank"
    valid = [r for r in results if ensemble_key in r and r.get(ensemble_key) is not None]
    valid_optuna = [r for r in results if "value" in r and r.get("value") is not None]
    valid_mae = [
        r for r in results
        if rank_mae_key in r
        and isinstance(r.get(rank_mae_key), (int, float))
        and math.isfinite(r.get(rank_mae_key))
    ]
    valid_rmse = [
        r for r in results
        if rank_rmse_key in r
        and isinstance(r.get(rank_rmse_key), (int, float))
        and math.isfinite(r.get(rank_rmse_key))
    ]
    valid_ndcg10 = [
        r for r in results
        if ndcg10_key in r
        and isinstance(r.get(ndcg10_key), (int, float))
        and math.isfinite(r.get(ndcg10_key))
    ]
    valid_playoff = [
        r for r in results
        if playoff_spearman_key in r
        and isinstance(r.get(playoff_spearman_key), (int, float))
        and math.isfinite(r.get(playoff_spearman_key))
    ]
    summary = {}
    if valid:
        best_sp = max(valid, key=lambda x: float(x.get(ensemble_key, -2)))
        best_ndcg = max(valid, key=lambda x: float(x.get(ndcg_key, -1)))
        summary["best_by_spearman"] = best_sp
        summary["best_by_ndcg"] = best_ndcg
    if valid_ndcg10:
        best_ndcg10 = max(valid_ndcg10, key=lambda x: float(x.get(ndcg10_key, -1)))
        summary["best_by_ndcg10"] = best_ndcg10
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

    n_combos = len(combos) if args.method in ("grid", "halving") else args.n_trials
    with open(batch_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "batch_id": batch_id,
            "method": args.method,
            "n_combos": n_combos,
            "n_jobs": n_jobs,
            "config_outputs": str(out_dir),
            "val_frac": val_frac,
            "phase": phase,
        }, f, indent=2)

    print(f"Wrote {batch_dir / 'sweep_results.csv'}, {batch_dir / 'sweep_results_summary.json'}")

    # Run explain on best combo (required unless --no-run-explain)
    if not args.dry_run and not args.no_run_explain:
        best_combo_idx = None
        if args.method == "optuna" and "best_optuna_trial" in summary:
            best_combo_idx = summary["best_optuna_trial"].get("combo")
        elif "best_by_spearman" in summary:
            best_combo_idx = summary["best_by_spearman"].get("combo")
        if best_combo_idx is not None:
            config_path = batch_dir / f"combo_{best_combo_idx:04d}" / "config.yaml"
            if config_path.exists():
                print(f"Running explain on best combo {best_combo_idx} (config={config_path})", flush=True)
                code = _run_cmd("scripts/5b_explain.py", ["--config", str(config_path)])
                if code != 0:
                    print("Warning: 5b_explain failed; re-run manually: python -m scripts.5b_explain --config <combo_config.yaml>", flush=True)
            else:
                print(f"Warning: best combo config not found at {config_path}; skip explain or run manually.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
