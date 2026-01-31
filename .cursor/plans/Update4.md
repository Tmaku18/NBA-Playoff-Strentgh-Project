---
name: sweep-script-exec
overview: Create a real-data sweep script (epochs + Model B grid) in the `hbf` worktree, refactor the current dummy training pipeline to use DuckDB data, then run the first batch sequentially.
todos:
  - id: data-loaders
    content: Add DB data loaders + Model A/B feature builders
    status: pending
  - id: train-refactor
    content: Refactor Model A/B training for real data + epochs
    status: pending
  - id: sweep-script
    content: Create scripts/sweep_hparams.py and logging
    status: pending
  - id: run-batch1
    content: Run epochs+Model B grid batch
    status: pending
isProject: false
---

# Executable Sweep Script (hbf)

## Context and constraints

- Worktree: `C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf`
- Use **real DB data** (not the dummy training/inference currently in scripts).
- First batch: **epoch sweep + Model B grid**, no Model A hyperparam tuning until attention is fixed.

## Why refactor is needed

The current training scripts are placeholders:

```95:99:C:\Users\tmaku.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\training\train_model_a.py
    batches = [get_dummy_batch(4, 10, 15, stat_dim, num_emb, device) for _ in range(5)]
    for epoch in range(3):
        loss = train_epoch(model, batches, optimizer, device)
```

and `scripts/4_train_model_b.py` uses random data. The sweep must be wired to DuckDB tables instead.

## Plan

### 1) Add real-data loading helpers

- Add a loader in `[src/data/db_loader.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\data\db_loader.py)` to read `games`, `team_game_logs`, `teams`, `player_game_logs` into pandas using `duckdb`.
- Add a new helper module (e.g., `[src/training/model_a_data.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\training\model_a_data.py)`) that:
  - Builds training lists using `[src/training/build_lists.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\training\build_lists.py)`
  - Computes rolling player stats via `[src/features/rolling.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\features\rolling.py)`
  - Constructs roster sets using `[src/features/build_roster_set.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\features\build_roster_set.py)`
  - Returns `train_batches` and `val_batches` with `embedding_indices`, `player_stats`, `minutes`, `key_padding_mask`, `rel`.
- Extend `[src/features/team_context.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\features\team_context.py)` with a `build_team_context_as_of_dates()` helper to aggregate features by `team_id, as_of_date` for Model B.

### 2) Refactor Model A/B training to accept real data + epochs

- Update `[src/training/train_model_a.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\training\train_model_a.py)` to:
  - Accept `epochs`, `train_batches`, `val_batches` and compute validation loss with `model.eval()` (PyTorch guidance).
  - Optional early stopping based on validation loss (small budget friendly).
- Update `[config/defaults.yaml](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\config\defaults.yaml)` with `model_a.epochs`, `model_a.early_stopping_*` fields used by the sweep.
- Update or bypass `[scripts/4_train_model_b.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\4_train_model_b.py)` so the sweep uses real features and `eval_set` for XGBoost early stopping.

### 3) Implement `scripts/sweep_hparams.py`

- Create `[scripts/sweep_hparams.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\sweep_hparams.py)` with a CLI:
  - `--batch epochs_plus_model_b` (default)
  - `--epochs 5,10,15,20,25`
  - `--xgb-grid max_depth=4,6;learning_rate=0.05,0.1;n_estimators=300,500;subsample=0.8,1.0;colsample_bytree=0.8,1.0`
  - `--rf-grid n_estimators=200,400;max_depth=10,16;min_samples_leaf=2,5`
- Build data once, reuse across sweep iterations.
- Log results to `outputs/sweeps/sweep_results.csv` and `outputs/sweeps/sweep_results.json` including:
  - Model A: epoch, train/val ListMLE loss, optional NDCG on validation lists.
  - Model B: grid params, val RMSE/R2 (or correlation) from a held-out date split.

### 4) Run the first batch (sequential)

- Precondition: DuckDB exists at `config.paths.db` (run `scripts/1_download_raw.py` + `scripts/2_build_db.py` if missing).
- Execute:
  - `python scripts/sweep_hparams.py --batch epochs_plus_model_b`
- Collect sweep outputs under `outputs/sweeps/` for review.

## Deliverables

- New sweep script wired to real data.
- Training pipeline refactor to enable real DB batches and configurable epochs.
- First batch run results saved to `outputs/sweeps/`.

