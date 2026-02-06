"""Train Model A (DeepSet + ListMLE) on real DB data. Option A: K-fold OOF, then final model."""
import os

# Set thread count for PyTorch/numpy before importing torch (helps CPU parallelism)
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "14"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "14"

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

try:
    import torch
except ImportError:
    torch = None


def _compute_batch_cache_key(config: dict, db_path: Path) -> str:
    """Compute SHA256-based cache key from config + DB identity (path, mtime, size)."""
    training = config.get("training", {})
    model_a = config.get("model_a", {})
    key_data = {
        "listmle_target": training.get("listmle_target"),
        "rolling_windows": tuple(training.get("rolling_windows", [10, 30])),
        "train_seasons": tuple(sorted(training.get("train_seasons", []))),
        "max_lists_oof": training.get("max_lists_oof", 30),
        "max_final_batches": training.get("max_final_batches", 50),
        "n_folds": training.get("n_folds", 5),
        "roster_size": training.get("roster_size", 15),
        "use_prior_season_baseline": training.get("use_prior_season_baseline", False),
        "prior_season_lookback_days": training.get("prior_season_lookback_days", 365),
        "stat_dim": model_a.get("stat_dim"),
        "num_embeddings": model_a.get("num_embeddings", 500),
        "db_path": str(db_path.resolve()),
    }
    stat = db_path.stat() if db_path.exists() else None
    if stat:
        key_data["db_mtime"] = stat.st_mtime
        key_data["db_size"] = stat.st_size
    js = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(js.encode()).hexdigest()[:16]


def _resolve_batch_cache_dir(config: dict) -> Path | None:
    """Resolve batch cache directory; return None if caching disabled."""
    p = config.get("paths", {}).get("batch_cache")
    if p is None or (isinstance(p, str) and p.strip().lower() in ("null", "")):
        p = ROOT / "data" / "processed" / "batch_cache"
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _move_batches_to_device(batches: list, device) -> None:
    """Move tensor values in batch dicts to device (in place)."""
    if torch is None:
        return
    for b in batches:
        for k, v in list(b.items()):
            if isinstance(v, torch.Tensor):
                b[k] = v.to(device)


def _copy_batches_to_cpu(batches: list) -> list:
    """Return shallow copy of batches with tensors moved to CPU (for cache save)."""
    if torch is None:
        return batches
    out = []
    for b in batches:
        nb = {}
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                nb[k] = v.cpu().clone()
            else:
                nb[k] = v
        out.append(nb)
    return out


def _next_run_id(outputs_dir: Path, run_id_base: int | None = None) -> str:
    """Same logic as script 6: next run_NNN; if no run_* and base set, return run_{base:03d}."""
    outputs_dir = Path(outputs_dir)
    pattern = re.compile(r"^run_(\d+)$", re.I)
    numbers = []
    if outputs_dir.exists():
        for p in outputs_dir.iterdir():
            if p.is_dir() and pattern.match(p.name):
                numbers.append(int(pattern.match(p.name).group(1)))
    if not numbers and run_id_base is not None:
        return f"run_{run_id_base:03d}"
    next_n = max(numbers, default=0) + 1
    return f"run_{next_n:03d}"


def _reserve_run_id(outputs_dir: Path, config: dict) -> None:
    """Reserve the next run_id for this pipeline run so script 6 uses the same folder.
    When inference.run_id is explicitly set (e.g. run_024 for phase1), use it directly."""
    inf = config.get("inference") or {}
    run_id = inf.get("run_id")
    if run_id and isinstance(run_id, str) and re.match(r"^run_\d+$", run_id.strip(), re.I):
        run_id = run_id.strip()
    else:
        run_id_base = inf.get("run_id_base")
        run_id = _next_run_id(outputs_dir, run_id_base=run_id_base)
    path = outputs_dir / ".current_run"
    path.write_text(run_id.strip(), encoding="utf-8")

from src.data.db_loader import load_playoff_data, load_training_data
from src.training.data_model_a import build_batches_from_db, build_batches_from_lists
from src.training.train_model_a import predict_batches, train_model_a, train_model_a_on_batches
from src.training.build_lists import build_lists
from src.utils.split import compute_split, get_train_seasons_ordered, group_lists_by_season, write_split_info


def _run_walk_forward(config, train_lists, games, tgl, teams, pgl, out, root, playoff_games=None, playoff_tgl=None):
    """Per-season walk-forward: train on 1..k, validate on k+1; final step trains on all and saves."""
    import torch

    seasons_cfg = config.get("seasons") or {}
    train_seasons_ordered = get_train_seasons_ordered(config)
    if not train_seasons_ordered or not seasons_cfg:
        print("Walk-forward: no train_seasons in config. Falling back to pooled training.", flush=True)
        return
    grouped = group_lists_by_season(train_lists, seasons_cfg)
    if not grouped:
        print("Walk-forward: no lists grouped by season. Falling back to pooled.", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_final = int(config.get("training", {}).get("max_final_batches", 50))
    epochs = int((config.get("model_a") or {}).get("epochs", 20))
    oof_rows = []
    n_steps = len(train_seasons_ordered)

    for step, k in enumerate(range(1, n_steps + 1), 1):
        train_season_set = set(train_seasons_ordered[:k])
        val_season = train_seasons_ordered[k] if k < n_steps else None
        step_train_lists = [lst for s in train_season_set for lst in grouped.get(s, [])]
        step_val_lists = list(grouped.get(val_season, [])) if val_season else []

        if not step_train_lists:
            print(f"Walk-forward step {step}/{n_steps}: no train lists, skip", flush=True)
            continue

        # Subsample if needed
        if len(step_train_lists) > max_final:
            step_idx = sorted(
                range(len(step_train_lists)),
                key=lambda i: (step_train_lists[i]["as_of_date"], step_train_lists[i].get("conference", "")),
            )[:: max(1, len(step_train_lists) // max_final)][:max_final]
            step_train_lists = [step_train_lists[i] for i in sorted(step_idx)]

        train_batches, _ = build_batches_from_lists(
            step_train_lists, games, tgl, teams, pgl, config, device=device,
        )
        if not train_batches:
            train_batches = build_batches_from_db(
                games, tgl, teams, pgl, config,
                playoff_games=playoff_games,
                playoff_tgl=playoff_tgl,
            )

        val_batches = None
        val_metas = []
        if step_val_lists:
            val_batches, val_metas = build_batches_from_lists(
                step_val_lists, games, tgl, teams, pgl, config, device=device,
            )

        model = train_model_a_on_batches(
            config, train_batches, device, max_epochs=epochs, val_batches=val_batches or None
        )
        if val_batches and val_metas:
            scores_list = predict_batches(model, val_batches, device)
            for score_tensor, meta in zip(scores_list, val_metas):
                K = score_tensor.shape[1]
                for ki in range(K):
                    oof_rows.append({
                        "team_id": meta["team_ids"][ki],
                        "as_of_date": meta["as_of_date"],
                        "oof_a": float(score_tensor[0, ki].item()),
                        "y": meta["win_rates"][ki],
                    })
            print(
                f"Walk-forward step {step}/{n_steps}: trained on seasons "
                f"{train_seasons_ordered[0]}..{train_seasons_ordered[k-1]}, validated on {val_season}, "
                f"OOF {len(val_batches)} lists",
                flush=True,
            )
        else:
            print(
                f"Walk-forward step {step}/{n_steps}: trained on seasons "
                f"{train_seasons_ordered[0]}..{train_seasons_ordered[k-1]} (final, no next season)",
                flush=True,
            )

        # Last step: save final model (trained on all train seasons)
        if k == n_steps:
            path = out / "best_deep_set.pt"
            torch.save({"model_state": model.state_dict(), "config": config}, path)
            print(f"Saved {path} (final model from walk-forward)", flush=True)

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = out / "oof_model_a.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"Wrote {oof_path} ({len(oof_rows)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: config/defaults.yaml)")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    print("Script 3: loading config and DB...", flush=True)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)
    games, tgl, teams, pgl = load_training_data(db_path)
    listmle_target = (config.get("training") or {}).get("listmle_target")
    playoff_games, playoff_tgl = None, None
    if listmle_target == "playoff_outcome":
        try:
            pg, ptgl, _ = load_playoff_data(db_path)
            if not pg.empty and not ptgl.empty:
                playoff_games, playoff_tgl = pg, ptgl
                print("Loaded playoff data for listmle_target=playoff_outcome", flush=True)
            else:
                print("Warning: playoff data empty; falling back to standings for ListMLE.", flush=True)
        except Exception as e:
            print(f"Warning: could not load playoff data ({e}); falling back to standings.", flush=True)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    # Reserve run_id for this pipeline run so inference (script 6) writes to the same folder
    _reserve_run_id(out, config)

    lists = build_lists(
        tgl, games, teams,
        config=config,
        playoff_games=playoff_games,
        playoff_tgl=playoff_tgl,
    )
    print(f"build_lists: {len(lists)} lists", flush=True)
    if not lists:
        batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no lists for OOF)")
        return

    valid_lists = [lst for lst in lists if len(lst["team_ids"]) >= 2]
    if not valid_lists:
        batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no valid lists for OOF)")
        return

    # 75/25 train-test split: compute and persist; use only train lists for OOF and final model
    train_lists, test_lists, split_info = compute_split(valid_lists, config)
    write_split_info(split_info, out)
    print(f"Split: {split_info['split_mode']} — train {split_info['n_train_lists']} lists, test {split_info['n_test_lists']} lists", flush=True)
    if not train_lists:
        batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no train lists after split)")
        return

    walk_forward = bool(config.get("training", {}).get("walk_forward", False))
    if walk_forward:
        _run_walk_forward(config, train_lists, games, tgl, teams, pgl, out, ROOT, playoff_games, playoff_tgl)
        return

    n_folds = config.get("training", {}).get("n_folds", 5)
    n_folds = min(n_folds, len(train_lists))
    if n_folds < 2:
        batches, _ = build_batches_from_lists(train_lists, games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (too few lists for OOF)")
        return

    # Subsample lists for OOF to keep runtime manageable (time-stratified, within train only)
    max_lists_oof = config.get("training", {}).get("max_lists_oof", 30)
    oof_lists = train_lists
    if len(train_lists) > max_lists_oof:
        step = max(1, len(train_lists) // max_lists_oof)
        sorted_by_date = sorted(range(len(train_lists)), key=lambda i: (train_lists[i]["as_of_date"], train_lists[i].get("conference", "")))
        oof_indices = sorted_by_date[::step][:max_lists_oof]
        oof_lists = [train_lists[i] for i in sorted(oof_indices)]
        print(f"OOF: using {len(oof_lists)} lists (subsampled from {len(train_lists)} train)", flush=True)
    n_folds = min(n_folds, len(oof_lists))
    if n_folds < 2:
        batches, _ = build_batches_from_lists(oof_lists, games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (too few lists for OOF)")
        return

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = None
    list_metas = None
    all_batches = None

    cache_dir = _resolve_batch_cache_dir(config)
    cache_key = _compute_batch_cache_key(config, db_path) if cache_dir else None
    cache_file = (cache_dir / f"{cache_key}.pt") if cache_dir and cache_key else None

    if cache_file and cache_file.exists():
        print(f"Batch cache hit: {cache_file.name}", flush=True)
        payload = torch.load(cache_file, map_location="cpu")
        batches = payload.get("oof_batches", [])
        list_metas = payload.get("list_metas", [])
        all_batches = payload.get("all_batches", [])
        _move_batches_to_device(batches, device)
        _move_batches_to_device(all_batches, device)

    if batches is None or list_metas is None:
        print("Building batches for OOF...", flush=True)
        batches, list_metas = build_batches_from_lists(oof_lists, games, tgl, teams, pgl, config, device=device)

    if not batches or not list_metas:
        print(
            "No batches from build_batches_from_lists (player_game_logs required). Skipping OOF; training final model only.",
            file=sys.stderr,
        )
        all_batches = build_batches_from_db(
            games, tgl, teams, pgl, config,
            playoff_games=playoff_games, playoff_tgl=playoff_tgl,
        )
        path = train_model_a(config, out, batches=all_batches)
        print(f"Saved {path} (no oof_model_a.parquet)")
        return

    # On cache miss: build all_batches and save to cache
    if all_batches is None:
        all_lists_for_cache = train_lists
        max_final = config.get("training", {}).get("max_final_batches", 50)
        if len(all_lists_for_cache) > max_final:
            step = max(1, len(all_lists_for_cache) // max_final)
            idx = sorted(
                range(len(all_lists_for_cache)),
                key=lambda i: (all_lists_for_cache[i]["as_of_date"], all_lists_for_cache[i].get("conference", "")),
            )[::step][:max_final]
            all_lists_for_cache = [all_lists_for_cache[i] for i in sorted(idx)]
        all_batches, _ = build_batches_from_lists(
            all_lists_for_cache, games, tgl, teams, pgl, config, device=device
        )
        if not all_batches:
            all_batches = build_batches_from_db(
                games, tgl, teams, pgl, config,
                playoff_games=playoff_games, playoff_tgl=playoff_tgl,
            )
        if cache_dir and cache_key and all_batches:
            cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "split_info": split_info,
                "oof_lists": oof_lists,
                "all_lists": all_lists_for_cache,
                "oof_batches": _copy_batches_to_cpu(batches),
                "list_metas": list_metas,
                "all_batches": _copy_batches_to_cpu(all_batches),
            }
            tmp = cache_dir / f".{cache_key}.tmp"
            torch.save(payload, tmp)
            tmp.rename(cache_dir / f"{cache_key}.pt")
            print(f"Batch cache saved: {cache_key}.pt", flush=True)

    # Time-based fold split: sort by as_of_date, chunk into n_folds (within train only)
    sorted_indices = sorted(range(len(oof_lists)), key=lambda i: (oof_lists[i]["as_of_date"], oof_lists[i].get("conference", "")))
    fold_size = (len(sorted_indices) + n_folds - 1) // n_folds
    oof_rows = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, len(sorted_indices))
        val_idx = sorted_indices[val_start:val_end]
        train_idx = [i for i in sorted_indices if i not in val_idx]
        train_batches = [batches[i] for i in train_idx]
        val_batches = [batches[i] for i in val_idx]
        val_metas = [list_metas[i] for i in val_idx]
        if not train_batches or not val_batches:
            continue
        epochs = int((config.get("model_a") or {}).get("epochs", 20))
        model = train_model_a_on_batches(
            config,
            train_batches,
            device,
            max_epochs=epochs,
            val_batches=val_batches,
        )
        scores_list = predict_batches(model, val_batches, device)
        for score_tensor, meta in zip(scores_list, val_metas):
            K = score_tensor.shape[1]
            for k in range(K):
                y_val = meta.get("rel_values", meta["win_rates"])[k]
                oof_rows.append({
                    "team_id": meta["team_ids"][k],
                    "as_of_date": meta["as_of_date"],
                    "oof_a": float(score_tensor[0, k].item()),
                    "y": y_val,
                })
        print(f"Fold {fold+1}/{n_folds} OOF collected {len(val_batches)} lists")

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = out / "oof_model_a.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"Wrote {oof_path} ({len(oof_rows)} rows)")
    else:
        print("No OOF rows collected (every fold had empty train or val batches).", file=sys.stderr)

    # Final model: use cached all_batches if available (from batch cache), else build
    if all_batches is None:
        all_lists = train_lists
        max_final = config.get("training", {}).get("max_final_batches", 50)
        if len(all_lists) > max_final:
            step = max(1, len(all_lists) // max_final)
            idx = sorted(range(len(all_lists)), key=lambda i: (all_lists[i]["as_of_date"], all_lists[i].get("conference", "")))[::step][:max_final]
            all_lists = [all_lists[i] for i in sorted(idx)]
            print(f"Final model: training on {len(all_lists)} lists (subsampled from {len(train_lists)} train)", flush=True)
        print("Building final batches...", flush=True)
        all_batches, _ = build_batches_from_lists(all_lists, games, tgl, teams, pgl, config, device=device)
        if not all_batches:
            all_batches = build_batches_from_db(
                games, tgl, teams, pgl, config,
                playoff_games=playoff_games, playoff_tgl=playoff_tgl,
            )
    path = train_model_a(config, out, batches=all_batches)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
