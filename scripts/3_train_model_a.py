"""Train Model A (DeepSet + ListMLE) on real DB data. Option A: K-fold OOF, then final model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.data.db_loader import load_training_data
from src.training.data_model_a import build_batches_from_db, build_batches_from_lists
from src.training.train_model_a import predict_batches, train_model_a, train_model_a_on_batches
from src.training.build_lists import build_lists


def main():
    print("Script 3: loading config and DB...", flush=True)
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)
    games, tgl, teams, pgl = load_training_data(db_path)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    lists = build_lists(tgl, games, teams)
    print(f"build_lists: {len(lists)} lists", flush=True)
    if not lists:
        batches = build_batches_from_db(games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no lists for OOF)")
        return

    valid_lists = [lst for lst in lists if len(lst["team_ids"]) >= 2]
    if not valid_lists:
        batches = build_batches_from_db(games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (no valid lists for OOF)")
        return

    n_folds = config.get("training", {}).get("n_folds", 5)
    n_folds = min(n_folds, len(valid_lists))
    if n_folds < 2:
        batches, _ = build_batches_from_lists(valid_lists, games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (too few lists for OOF)")
        return

    # Subsample lists for OOF to keep runtime manageable (time-stratified)
    max_lists_oof = config.get("training", {}).get("max_lists_oof", 30)
    if len(valid_lists) > max_lists_oof:
        step = max(1, len(valid_lists) // max_lists_oof)
        sorted_by_date = sorted(range(len(valid_lists)), key=lambda i: (valid_lists[i]["as_of_date"], valid_lists[i].get("conference", "")))
        oof_indices = sorted_by_date[::step][:max_lists_oof]
        valid_lists = [valid_lists[i] for i in sorted(oof_indices)]
        print(f"OOF: using {len(valid_lists)} lists (subsampled from more)", flush=True)
    n_folds = min(n_folds, len(valid_lists))
    if n_folds < 2:
        batches, _ = build_batches_from_lists(valid_lists, games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=batches)
        print(f"Saved {path} (too few lists for OOF)")
        return

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Building batches for OOF...", flush=True)
    batches, list_metas = build_batches_from_lists(valid_lists, games, tgl, teams, pgl, config, device=device)

    if not batches or not list_metas:
        print(
            "No batches from build_batches_from_lists (player_game_logs required). Skipping OOF; training final model only.",
            file=sys.stderr,
        )
        all_batches = build_batches_from_db(games, tgl, teams, pgl, config)
        path = train_model_a(config, out, batches=all_batches)
        print(f"Saved {path} (no oof_model_a.parquet)")
        return

    # Time-based fold split: sort by as_of_date, chunk into n_folds
    sorted_indices = sorted(range(len(valid_lists)), key=lambda i: (valid_lists[i]["as_of_date"], valid_lists[i].get("conference", "")))
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
                oof_rows.append({
                    "team_id": meta["team_ids"][k],
                    "as_of_date": meta["as_of_date"],
                    "oof_a": float(score_tensor[0, k].item()),
                    "y": meta["win_rates"][k],
                })
        print(f"Fold {fold+1}/{n_folds} OOF collected {len(val_batches)} lists")

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = out / "oof_model_a.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"Wrote {oof_path} ({len(oof_rows)} rows)")
    else:
        print("No OOF rows collected (every fold had empty train or val batches).", file=sys.stderr)

    # Final model: use full lists, cap batch count for feasible runtime
    all_lists = build_lists(tgl, games, teams)
    all_lists = [lst for lst in all_lists if len(lst["team_ids"]) >= 2]
    max_final = config.get("training", {}).get("max_final_batches", 50)
    if len(all_lists) > max_final:
        step = max(1, len(all_lists) // max_final)
        idx = sorted(range(len(all_lists)), key=lambda i: (all_lists[i]["as_of_date"], all_lists[i].get("conference", "")))[::step][:max_final]
        all_lists = [all_lists[i] for i in sorted(idx)]
        print(f"Final model: training on {len(all_lists)} lists (subsampled)", flush=True)
    print("Building final batches...", flush=True)
    all_batches, _ = build_batches_from_lists(all_lists, games, tgl, teams, pgl, config, device=device)
    if not all_batches:
        all_batches = build_batches_from_db(games, tgl, teams, pgl, config)
    path = train_model_a(config, out, batches=all_batches)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
