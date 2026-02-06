# Phase 1 Sweep Analysis (outputs3 + outputs4)

Analysis of Phase 1 hyperparameter sweeps. **outputs3** sweeps completed successfully; **outputs4** sweeps encountered Triton errors on Windows (fixed: `torch.compile` now skipped on Windows in `train_model_a.py`).

---

## Data Sources

| Location | Sweeps | Status |
|----------|--------|--------|
| **outputs3/sweeps/** | phase1_spearman_final_rank, phase1_spearman_playoff_outcome | Complete; full metrics |
| **outputs4/sweeps/** | phase1_spearman_*, phase1_ndcg4_* | Failed (Triton); all trials -inf. Re-run on WSL or after Triton fix. |

**Authoritative Phase 1 results:** outputs3 (run_024; config paths.outputs=outputs3).

---

## outputs3 Phase 1 Results

### phase1_spearman_final_rank (listmle_target=final_rank)

**Best by Spearman:** combo 10

| Param | Value |
|-------|-------|
| model_a_epochs | 21 |
| max_depth | 5 |
| learning_rate | 0.0704 |
| n_estimators_xgb | 291 |
| n_estimators_rf | 164 |
| min_samples_leaf | 4 |

| Metric | Best (combo 10) | run_022 |
|--------|-----------------|---------|
| **Spearman** | **0.499** | 0.430 |
| **playoff_spearman** | **0.518** | 0.461 |
| **NDCG** | 0.485 | 0.482 |
| **NDCG@4** | 0.473 | 0.46 |
| **rank_mae_pred_vs_playoff_final_results** | **6.73** | 7.53 |
| **rank_rmse_pred_vs_playoff_final_results** | **8.67** | 9.24 |
| **ROC-AUC upset** | 0.759 | 0.728 |
| **Brier championship** | 0.032 | 0.032 |

**Best by playoff_spearman:** combo 5 — playoff_spearman **0.544**, spearman 0.497, rank_mae 6.80.

**Best by NDCG/NDCG@4/NDCG@16/NDCG@20:** combo 6 — NDCG 0.490, spearman 0.452, playoff_spearman 0.459.

**Optuna importances:** n_estimators_rf 0.26, n_estimators_xgb 0.21, learning_rate 0.21, model_a_epochs 0.19, max_depth 0.13. Fix subsample, rolling_windows, colsample_bytree.

---

### phase1_spearman_playoff_outcome (listmle_target=playoff_outcome)

**Best by Spearman:** combo 3

| Param | Value |
|-------|-------|
| model_a_epochs | 21 |
| max_depth | 4 |
| learning_rate | 0.0669 |
| n_estimators_xgb | 280 |
| n_estimators_rf | 151 |
| min_samples_leaf | 5 |

| Metric | Best (combo 3) | phase1_spearman_final_rank best | run_022 |
|--------|----------------|---------------------------------|---------|
| **Spearman** | 0.482 | **0.499** | 0.430 |
| **playoff_spearman** | 0.487 | **0.518** | 0.461 |
| **NDCG** | 0.483 | 0.485 | 0.482 |
| **rank_mae_pred_vs_playoff_final_results** | 6.87 | **6.73** | 7.53 |
| **rank_rmse_pred_vs_playoff_final_results** | 8.81 | **8.67** | 9.24 |

**Interpretation:** For Phase 1 spearman objective, **final_rank (standings) target outperformed playoff_outcome target** on Spearman, playoff_spearman, rank_mae, and rank_rmse. This is the reverse of Phase 0 baseline, where playoff_outcome led. Phase 1 narrowed ranges may favor standings-trained configs for this objective.

**Optuna importances:** n_estimators_xgb 0.33, model_a_epochs 0.26, n_estimators_rf 0.19, learning_rate 0.16, min_samples_leaf 0.06. max_depth low (0.01).

---

## Standings vs Playoff Target Comparison (Phase 1 spearman)

| Sweep | listmle_target | Best spearman | Best playoff_spearman | rank_mae | rank_rmse |
|-------|----------------|---------------|------------------------|----------|-----------|
| phase1_spearman_final_rank | final_rank | **0.499** | **0.518** | **6.73** | **8.67** |
| phase1_spearman_playoff_outcome | playoff_outcome | 0.482 | 0.487 | 6.87 | 8.81 |

**Conclusion:** Phase 1 spearman sweeps favor **final_rank** (standings) for this objective. Use phase1_spearman_final_rank best combo (10 or 5 for playoff_spearman) as reference for rolling sweep and Phase 2.

---

## outputs4 Status

outputs4 sweeps (phase1_spearman_*, phase1_ndcg4_*) failed with TritonMissing on Windows. Fix applied: `torch.compile` is skipped when `sys.platform == "win32"` in `src/training/train_model_a.py`. Re-run outputs4 sweeps on WSL (Linux) or Windows after the fix to populate outputs4 with run_025+.

---

## Next: Rolling Sweep and Phase 2

Per [phased_sweep_execution_plan](.cursor/plans/phased_sweep_execution_plan_d0f3e0a3.plan.md):

1. **Rolling windows sweep** — Test [10], [10,30], [15,30], [20,30] with best objective (spearman) and listmle_target (final_rank from Phase 1).
2. **Phase 2** — Narrow ranges around high-importance params; fix low-importance; run focused sweeps. See `.cursor/plans/PHASE2_SWEEP_PLAN.md`.
