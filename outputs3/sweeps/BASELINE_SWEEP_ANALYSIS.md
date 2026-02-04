# Phase 0 Baseline Sweep Analysis

This document analyzes the Phase 0 baseline exploratory sweeps and compares results to run_022/run_023.

---

## Sweep Workflow Rules (consistency and visibility)

1. **Before beginning a sweep**
   - Ensure `outputs3/sweeps/` is tracked by Git and pushed to `main` (sweep results, BASELINE_SWEEP_ANALYSIS.md, optuna_*, sweep_*).
   - Confirm no uncommitted changes that would be overwritten.

2. **Sweep execution**
   - Run in **foreground** (never background).
   - Use **n_jobs=4** (4 parallel workers).
   - Use **no timeout** — let the sweep run until completion.
   - **Phase 1 robust settings:** n_trials=12, --no-run-explain — keeps each objective batch (2 sweeps + analysis) under 4 hours.

3. **After each sweep**
   - Update this document with results, best combo, and interpretation.
   - Update the standings vs playoff comparison table.
   - Commit and push to `main` before starting the next sweep.

### How to speed up sweep time

- **`--n-jobs 4`** — Run 4 trials in parallel; wall time ≈ (trials/4) × per-trial time.
- **`--n-trials 6`** — Use for Phase 0 exploratory; increase to 20 for Phase 1.
- **`--no-run-explain`** — Skip SHAP/IG on best combo (~5–10 min saved).
- **Config caps** — Lower `max_lists_oof` and `max_final_batches` in `baseline_max_features.yaml` for faster Model A training (trade: fewer lists).

**Invocation template:** Valid objectives: `spearman`, `ndcg4`, `ndcg16`, `ndcg20`, `playoff_spearman`, `rank_rmse`.
- **Phase 0 (baseline):** n_trials=6, phase=baseline
- **Phase 1 (robust, &lt; 4 h per objective):** n_trials=12, phase=phase1, --no-run-explain

```powershell
# Phase 0
python -m scripts.sweep_hparams --method optuna --n-trials 6 --n-jobs 4 --objective <OBJ> --listmle-target <TARGET> --phase baseline --batch-id <BATCH_ID> --config config/baseline_max_features.yaml
# Phase 1 (robust)
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective <OBJ> --listmle-target <TARGET> --phase phase1 --batch-id <BATCH_ID> --config config/baseline_max_features.yaml
```

---

## Sweep 1: baseline_spearman_final_rank

**Configuration:**
- **Objective:** spearman (maximize)
- **listmle_target:** final_rank (train on EOS standings, not playoff outcome)
- **Config:** baseline_max_features (max_lists_oof 100, max_final_batches 100)
- **Phase:** baseline (wide hyperparameter ranges)
- **n_trials:** 6
- **n_jobs:** 4

### Best combo (by spearman): combo 2

| Param | Value |
|-------|-------|
| model_a_epochs | 26 |
| max_depth | 4 |
| learning_rate | 0.094 |
| n_estimators_xgb | 209 |
| n_estimators_rf | 182 |
| min_samples_leaf | 6 |

### Best combo metrics vs run_022

| Metric | baseline_spearman_final_rank (combo 2) | run_022 |
|--------|----------------------------------------|---------|
| **Spearman** | 0.492 | 0.430 |
| **NDCG** | 0.483 | 0.482 |
| **NDCG10** | 0.483 | (same as ndcg) |
| **playoff_spearman** | 0.499 | 0.461 |
| **rank_mae_pred_vs_playoff** | 6.80 | 7.53 |
| **rank_rmse_pred_vs_playoff** | 8.72 | 9.24 |
| **ROC-AUC upset** | 0.777 | 0.728 |
| **MRR top-2** | 0.50 | 0.50 |
| **MRR top-4** | 0.50 | 0.50 |
| **Brier championship** | 0.032 | 0.032 |

**Interpretation:** Training on **standings** (final_rank) with max features and swept hyperparameters **improved** Spearman (+0.06), playoff_spearman (+0.04), rank_mae (−0.73), rank_rmse (−0.52), and ROC-AUC upset (+0.05) over run_022 (which trained on playoff_outcome with default config). NDCG is nearly unchanged. This suggests the wider hyperparameter search and max features helped; the standings target did not hurt and may have benefited from more stable training signal.

### Per-model comparison (best combo vs run_022)

| Model | Metric | Sweep best | run_022 |
|-------|--------|------------|---------|
| Ensemble | Spearman | 0.492 | 0.430 |
| Model A | Spearman | 0.492 | 0.426 |
| Model B | Spearman | 0.605 | 0.515 |
| Model C | Spearman | 0.294 | 0.313 |

Model A matches ensemble (stack dominated by A). Model B has higher Spearman than run_022. Model C is slightly lower.

### Per-conference (best combo)

| Conference | NDCG | Spearman |
|------------|------|----------|
| East | 0.263 | 0.346 |
| West | 0.749 | 0.614 |

West outperforms East on both metrics; West Spearman improved vs run_022 (0.50 → 0.61).

### Optuna parameter importance

| Param | Importance |
|-------|------------|
| learning_rate | 0.31 |
| min_samples_leaf | 0.22 |
| n_estimators_rf | 0.15 |
| n_estimators_xgb | 0.14 |
| model_a_epochs | 0.11 |
| max_depth | 0.07 |
| subsample | 0.00 |
| rolling_windows | 0.00 |
| colsample_bytree | 0.00 |

**Phase 1 planning:** Fix subsample, rolling_windows, colsample_bytree at default; focus search on learning_rate, min_samples_leaf, n_estimators_rf/xgb, model_a_epochs, max_depth.

---

## Sweep 2: baseline_spearman_playoff_outcome

**Configuration:**
- **Objective:** spearman
- **listmle_target:** playoff_outcome (train on EOS playoff result)
- **Config:** baseline_max_features
- **Phase:** baseline

### Best combo (by spearman): combo 5

| Param | Value |
|-------|-------|
| model_a_epochs | 16 |
| max_depth | 5 |
| learning_rate | 0.063 |
| n_estimators_xgb | 272 |
| n_estimators_rf | 185 |
| min_samples_leaf | 6 |

### Spearman (final_rank) vs spearman (playoff_outcome)

| Metric | final_rank (sweep 1) | playoff_outcome (sweep 2) | run_022 |
|--------|----------------------|---------------------------|---------|
| **Spearman** | 0.492 | **0.512** | 0.430 |
| **playoff_spearman** | 0.499 | **0.513** | 0.461 |
| **rank_mae_pred_vs_playoff** | 6.80 | **6.67** | 7.53 |
| **rank_rmse_pred_vs_playoff** | 8.72 | **8.55** | 9.24 |
| **NDCG** | 0.483 | 0.483 | 0.482 |
| **ROC-AUC upset** | 0.777 | 0.715 | 0.728 |

**Interpretation:** The **playoff_outcome** target sweep (sweep 2) outperformed the **final_rank** (standings) sweep (sweep 1) on Spearman (+0.02), playoff_spearman (+0.01), rank_mae (−0.13), and rank_rmse (−0.17). Both beat run_022. The hypothesis that playoff-optimized training gives better metrics is **supported** by these results. ROC-AUC upset is slightly lower in sweep 2 (0.715 vs 0.777).

---

## Sweep 3: baseline_ndcg_final_rank

**Configuration:**
- **Objective:** ndcg (maximize)
- **listmle_target:** final_rank (train on EOS standings)
- **Config:** baseline_max_features
- **Phase:** baseline
- **n_trials:** 6
- **n_jobs:** 4

### Best combo (by ndcg): combo 3

| Param | Value |
|-------|-------|
| model_a_epochs | 20 |
| max_depth | 5 |
| learning_rate | 0.092 |
| n_estimators_xgb | 269 |
| n_estimators_rf | 167 |
| min_samples_leaf | 6 |

### Best combo metrics

| Metric | baseline_ndcg_final_rank (combo 3) | run_022 |
|--------|------------------------------------|---------|
| **NDCG** | 0.486 | 0.482 |
| **NDCG10** | 0.486 | (same as ndcg) |
| **Spearman** | 0.483 | 0.430 |
| **playoff_spearman** | 0.511 | 0.461 |
| **rank_mae_pred_vs_playoff** | 6.93 | 7.53 |
| **rank_rmse_pred_vs_playoff** | 8.80 | 9.24 |

**Interpretation:** Optimizing for NDCG with standings target improved NDCG (+0.004) and playoff_spearman (+0.05) over run_022. Similar rank_mae/rmse gains to spearman sweeps.

---

## Sweep 4: baseline_ndcg_playoff_outcome

**Configuration:**
- **Objective:** ndcg (maximize)
- **listmle_target:** playoff_outcome (train on EOS playoff result)
- **Config:** baseline_max_features
- **Phase:** baseline
- **n_trials:** 6 planned; 4 completed (combos 4–5 interrupted; results aggregated via `scripts/aggregate_sweep_results.py`)

### Best combo (by ndcg): combo 1

| Param | Value |
|-------|-------|
| model_a_epochs | 14 |
| max_depth | 3 |
| learning_rate | 0.065 |
| n_estimators_xgb | 283 |
| n_estimators_rf | 228 |
| min_samples_leaf | 4 |

### Best combo metrics

| Metric | baseline_ndcg_playoff_outcome (combo 1) | baseline_ndcg_final_rank (sweep 3) | run_022 |
|--------|----------------------------------------|------------------------------------|---------|
| **NDCG** | 0.486 | 0.486 | 0.482 |
| **NDCG10** | 0.486 | 0.486 | (same as ndcg) |
| **Spearman** | 0.485 | 0.483 | 0.430 |
| **playoff_spearman** | 0.504 | 0.511 | 0.461 |
| **rank_mae_pred_vs_playoff** | 6.80 | 6.93 | 7.53 |
| **rank_rmse_pred_vs_playoff** | 8.78 | 8.80 | 9.24 |

**Interpretation:** NDCG-optimized playoff target (sweep 4) matches NDCG from sweep 3 (standings) at 0.486. Playoff_spearman slightly lower than sweep 3 (0.504 vs 0.511) but rank_mae better (6.80 vs 6.93). Both ndcg sweeps beat run_022 on all metrics.

---

## Standings vs playoff target comparison

| Sweep | listmle_target | Best spearman | Best ndcg | Best playoff_spearman | rank_mae |
|-------|----------------|---------------|-----------|-----------------------|----------|
| baseline_spearman_final_rank | final_rank | 0.492 | 0.483 | 0.499 | 6.80 |
| baseline_spearman_playoff_outcome | playoff_outcome | **0.512** | 0.483 | **0.513** | **6.67** |
| baseline_ndcg_final_rank | final_rank | 0.483 | **0.486** | 0.511 | 6.93 |
| baseline_ndcg_playoff_outcome | playoff_outcome | 0.485 | **0.486** | 0.504 | **6.80** |

---

## Conclusions

1. **Phase 0 complete.** All 4 baseline sweeps ran (sweep 4 aggregated from 4/6 combos after interruption).
2. **Standings-trained (sweep 1)** improved over run_022: Spearman, playoff_spearman, rank_mae, rank_rmse.
3. **Playoff-trained (sweep 2)** improved further over sweep 1: better Spearman, playoff_spearman, rank_mae, rank_rmse.
4. **NDCG sweeps (3 and 4)** both reached NDCG 0.486; playoff target tied on NDCG, slightly worse playoff_spearman, better rank_mae.
5. **Hypothesis supported:** Playoff-optimized training (sweep 2) gives best Spearman and playoff_spearman. NDCG-optimized configs are similar for standings vs playoff target.
6. **Next:** Phase 1 — full sweeps with narrowed ranges (see phased_sweep_roadmap), starting with `spearman` objective. Consider adding NDCG@12, NDCG@16, NDCG@20 sweeps per hypotheses below.

---

## Hypotheses for NDCG@k sweeps (future phases)

NBA playoff structure (12 guaranteed teams, 16 full field, 20 including play-in zone) motivates sweep objectives:

| Objective | Rationale | Expected outcome |
|-----------|-----------|------------------|
| **NDCG@12** | Optimizes rank ordering of the 12 teams guaranteed playoffs (seeds 1–6 per conference). Strongest teams; cares about rank ordering. | **Best at MRR** — most accurate at identifying/ordering lock-in playoff teams. |
| **NDCG@16** | Aligns with full 16-team playoff field. | **Best at determining who makes the playoffs** — predicts full bracket qualifiers. |
| **NDCG@20** | Includes play-in zone (seeds 7–10 per conference). | **Best at handling play-in noise** — more tolerant of uncertainty in seeds 7–10. |

**Playoff Spearman vs NDCG@k:**

- Models optimized for **playoff_spearman** may **suffer worse NDCG** but give **better RMSE and MAE**, because playoff Spearman better accounts for team strength across the full ranking.
- **Hypothesis:** Varying NDCG cutoff (NDCG@12, NDCG@16, NDCG@20) may **beat playoff-spearman-optimized models** on rank_mae/rank_rmse while maintaining or improving NDCG. Worth testing in Phase 1+ sweeps.

See `docs/HYPOTHESIZED_BEST_CONFIG_AND_METRIC_INSIGHTS.md` §3.6 for full rationale.
