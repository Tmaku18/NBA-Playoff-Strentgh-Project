---
name: Refined Sweep Rerun
overview: Rerun the sweep around the new sweet spots with tighter increments for Model A epochs (8–28) and a two-phase local Model B grid, using a more robust validation split (val_frac=0.25) and writing outputs to new batch folders to avoid overwrites.
todos: []
isProject: false
---

# Refined Sweep Rerun Plan

## Context7 notes

- XGBoost tuning: `learning_rate` (eta) trades off with `n_estimators`; smaller steps often need more estimators. `subsample`/`colsample_bytree` are standard levers to control overfitting while keeping depth fixed. This supports a local sweep around the current best values rather than a wide grid. 
- scikit-learn ensembles: explicitly set `n_estimators` (don’t rely on defaults) and treat `max_depth`/`min_samples_leaf` as the primary complexity controls for RandomForestRegressor.

## Plan

- Update the sweep script to accept a configurable output batch id and validation fraction so we can run multiple local sweeps without overwriting prior results, while keeping defaults stable.
  - File: [C:\Users\tmakucursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\sweep_hparams.py](C:\Users\tmaku.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\sweep_hparams.py)
  - Add CLI args for `--batch-id` (or `--out-dir`) and `--val-frac`, and use them to set `out_root` and the Model B date split (`val_frac`) used in the sweep.
- Run the refined Model A epoch sweep using the new batch id and `val_frac=0.25`:
  - Epochs: 8–28 step 1 (`--epochs 8,9,10,...,28`).
  - Output folder: `outputs/sweeps/<batch_id>/`.
- Run the two-phase local Model B sweep (each into its own batch id):
  - Phase 1 (XGB local): max_depth=4 fixed; learning_rate {0.08, 0.10, 0.12}; n_estimators {250, 300, 350}; subsample=0.8; colsample_bytree=0.8; RF fixed at n_estimators=200, max_depth=12, min_samples_leaf=5.
  - Phase 2 (RF local): XGB fixed at max_depth=4, learning_rate=0.10, n_estimators=300, subsample=0.8, colsample_bytree=0.8; RF n_estimators {150, 200, 250}; min_samples_leaf {4, 5, 6}; max_depth=12.
- Summarize the new sweep results by selecting the best Model A epoch by NDCG/Spearman and the best Model B combo by RMSE and Spearman from each phase.

## Key files

- [C:\Users\tmakucursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\sweep_hparams.py](C:\Users\tmaku.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\sweep_hparams.py)
- [C:\Users\tmakucursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\config\defaults.yaml](C:\Users\tmaku.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\config\defaults.yaml)

## Todos

- sweep-cli-opts: Add `--batch-id`/`--val-frac` and wire to outputs/split
- run-epochs: Run Model A epochs 8–28 step 1 with val_frac=0.25
- run-model-b: Run Model B phase-1 and phase-2 local grids
- report-results: Summarize best metrics and configs from the new sweeps

