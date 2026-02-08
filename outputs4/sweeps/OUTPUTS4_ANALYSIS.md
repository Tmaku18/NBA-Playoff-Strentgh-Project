# outputs4 Sweep Analysis

Comprehensive analysis of all sweeps in **outputs4** (run_024/run_025). outputs4 is the Phase I + Phase II sweep root; config: `config/outputs4_phase1.yaml`.

---

## 1. Overview

| Sweep | Trials | Objective | listmle_target | Status | Best Spearman | Best NDCG@4 | Best NDCG@16 |
|-------|--------|-----------|----------------|--------|---------------|------------|--------------|
| **phase3_fine_ndcg16_final_rank** | 20 | ndcg16 | final_rank | **Complete** | **0.557** | 0.506 | **0.550** |
| phase3_coarse_ndcg16_final_rank | 20 | ndcg16 | final_rank | Complete | 0.543 | 0.473 | 0.540 |
| **phase2_coarse_spearman_final_rank** | 15 | spearman | final_rank | Complete | 0.535 | 0.511 | 0.543 |
| phase3_coarse_ndcg4_final_rank | 20 | ndcg4 | final_rank | Complete | 0.492 | 0.506 | 0.547 |
| phase3_fine_ndcg4_final_rank | 20 | ndcg4 | final_rank | Complete | 0.513 | 0.506 | 0.545 |
| phase1_rolling_spearman_final_rank | 12 | spearman | final_rank | Complete | 0.496 | — | — |
| phase1_spearman_final_rank | 12 | spearman | final_rank | Partial | — | — | — |
| phase1_spearman_playoff_outcome | 12 | spearman | playoff_outcome | Partial | — | — | — |
| phase1_ndcg4_final_rank | 12 | ndcg4 | final_rank | Partial | — | — | — |
| phase1_ndcg4_playoff_outcome | 12 | ndcg4 | playoff_outcome | Partial | — | — | — |

---

## 2. Phase 2 Coarse Sweep (phase2_coarse_spearman_final_rank) — **Best Overall**

**Sweep:** Optuna 15 trials; `rolling_windows: [15, 30]` fixed; epochs 20–24, max_depth 5, min_leaf 5; lr, n_xgb, n_rf varied. Run ID: **run_025**.

### Best combo (8)

| Param | Value |
|-------|-------|
| rolling_windows | [15, 30] |
| model_a_epochs | 20 |
| max_depth | 5 |
| learning_rate | 0.0798 |
| n_estimators_xgb | 204 |
| n_estimators_rf | 209 |
| min_samples_leaf | 5 |

### Metrics vs prior best

| Metric | Phase 2 (combo 8) | Rolling best (combo 7) | run_022 |
|--------|-------------------|------------------------|---------|
| **Spearman** | **0.535** | 0.496 | 0.430 |
| **playoff_spearman** | **0.547** | 0.501 | 0.461 |
| **NDCG** | **0.543** | 0.490 | 0.48 |
| **NDCG@4** | **0.511** | 0.464 | 0.46 |
| **rank_mae_pred_vs_playoff** | **6.33** | 6.73 | 7.53 |
| **rank_rmse_pred_vs_playoff** | **8.35** | 8.69 | 9.24 |
| **ROC-AUC upset** | **0.764** | 0.747 | 0.728 |

### Interpretation

Phase 2 (coarse refinement) delivers a clear improvement:

- **Spearman +0.039** over rolling best (0.535 vs 0.496)
- **playoff_spearman +0.046** (0.547 vs 0.501)
- **NDCG +0.053** (0.543 vs 0.490)
- **rank_mae** improved to 6.33 (from 6.73)
- **rank_rmse** improved to 8.35 (from 8.69)

The best config uses **20 epochs** (vs 24 in rolling), **lr ≈ 0.08**, **n_xgb 204**, **n_rf 209** — close to the rolling best but with one fewer epoch and slightly different RF size.

### Optuna importances (Phase 2)

| Param | Importance |
|-------|------------|
| learning_rate | 0.40 |
| n_estimators_xgb | 0.24 |
| n_estimators_rf | 0.24 |
| model_a_epochs | 0.13 |
| max_depth, min_leaf, subsample, rolling_windows, colsample | 0 (fixed) |

**For phase2_fine:** Focus on learning_rate, n_xgb, n_rf, and model_a_epochs. Fix max_depth, min_leaf, rolling_windows at current best.

---

## 3. Rolling Sweep (phase1_rolling_spearman_final_rank)

**Best combo 7:** rolling [15, 30], epochs 24, lr 0.086, n_xgb 204, n_rf 226, Spearman 0.496, playoff_spearman 0.501.

Rolling windows [15, 30] outperformed [10], [10, 30], and [20, 30]. See `SWEEP_PHASE1_ANALYSIS.md` for details.

---

## 4. Phase 1 Sweeps (outputs4)

- **phase1_spearman_final_rank** — 12 trials; full combos if completed
- **phase1_spearman_playoff_outcome** — 12 trials
- **phase1_ndcg4_final_rank** — 12 trials
- **phase1_ndcg4_playoff_outcome** — 12 trials

These may have partial or Triton-affected results from earlier runs. Phase 2 coarse sweep supersedes them for the spearman objective.

---

## 5. Progression Summary

```
run_022 (baseline)        → Spearman 0.430, playoff_spearman 0.461
Phase 1 (outputs3)        → Spearman 0.499, playoff_spearman 0.518
Rolling sweep             → Spearman 0.496, playoff_spearman 0.501 (rolling [15,30])
Phase 2 coarse            → Spearman 0.535, playoff_spearman 0.547, NDCG@16 0.543
Phase 3 fine ndcg16       → Spearman 0.557, playoff_spearman 0.568, NDCG@16 0.550 ← Best
```

Phase 3 fine NDCG@16 (combo 18) is the best configuration across Spearman, playoff_spearman, and NDCG@16.

---

## 6. Phase 3 NDCG Sweeps — NDCG-Tuned Configs

Phase 3 sweeps target NDCG@4 and NDCG@16 optimization (coarse → fine). All use `rolling_windows: [15, 30]`, `listmle_target: final_rank`, and the Phase 2 search space.

### 6.1 Summary Table

| Sweep | Trials | Objective | Best Combo | NDCG@4 | NDCG@16 | Spearman | playoff_spearman |
|-------|--------|-----------|------------|--------|---------|----------|------------------|
| phase3_coarse_ndcg4 | 20 | ndcg4 | 8 | **0.506** | 0.547 | 0.492 | 0.489 |
| phase3_fine_ndcg4 | 20 | ndcg4 | 6 | **0.506** | 0.545 | 0.513 | 0.490 |
| phase3_coarse_ndcg16 | 20 | ndcg16 | 8 | 0.473 | **0.540** | **0.543** | **0.555** |
| phase3_fine_ndcg16 | 20 | ndcg16 | 18 | 0.506 | **0.550** | **0.557** | **0.568** |

### 6.2 Phase 3 Coarse NDCG@4 (phase3_coarse_ndcg4_final_rank)

**Best combo 8 (Optuna objective ndcg4):**
- **Params:** epochs 21, lr 0.066, n_xgb 267, n_rf 215, rolling [15, 30], max_depth 5, min_leaf 5
- **NDCG@4:** 0.506 | **NDCG@16:** 0.547 | NDCG@20: 0.551
- **Spearman:** 0.492 | playoff_spearman: 0.489 | rank_mae: 6.67 | ROC-AUC upset: 0.790

**Optuna importances:** n_rf 0.42, lr 0.41, n_xgb 0.12, epochs 0.05 (others 0).

**Trade-off:** NDCG@4-optimized config yields higher NDCG but lower Spearman vs Phase 2 Spearman-optimized (0.535).

### 6.3 Phase 3 Fine NDCG@4 (phase3_fine_ndcg4_final_rank)

**Best combo 6 (centered on phase3_coarse_ndcg4 best):**
- **Params:** epochs 21, lr 0.086, n_xgb 239, n_rf 202
- **NDCG@4:** 0.506 | NDCG@16: 0.545 | Spearman: 0.513 | playoff_spearman: 0.490

**Optuna importances:** epochs 0.60, n_xgb 0.18, n_rf 0.14, lr 0.08.

Fine sweep did not improve NDCG@4 beyond coarse (both 0.506); Spearman improved slightly (0.513 vs 0.492).

### 6.4 Phase 3 Coarse NDCG@16 (phase3_coarse_ndcg16_final_rank)

**Best combo 8 (Optuna objective ndcg16):**
- **Params:** epochs 20, lr 0.066, n_xgb 222, n_rf 164
- **NDCG@16:** 0.540 | NDCG@4: 0.473 | **Spearman:** 0.543 | **playoff_spearman:** 0.555
- **rank_mae:** 6.53 | **rank_rmse:** 8.27 | ROC-AUC upset: 0.778

**Optuna importances:** n_xgb 0.31, n_rf 0.29, lr 0.23, epochs 0.17.

Notable: NDCG@16 best also achieves the **strongest Spearman and playoff_spearman** among Phase 3 coarse sweeps.

### 6.5 Phase 3 Fine NDCG@16 (phase3_fine_ndcg16_final_rank) — **Best Overall Phase 3**

**Best combo 18 (Optuna objective ndcg16):**
- **Params:** epochs 22, lr 0.072, n_xgb 229, n_rf 173
- **NDCG@4:** 0.506 | **NDCG@16:** 0.550 | **Spearman:** 0.557 | **playoff_spearman:** 0.568
- **rank_mae:** 6.33 | **rank_rmse:** 8.15 | ROC-AUC upset: 0.773

**Optuna importances:** lr 0.55, n_xgb 0.21, epochs 0.15, n_rf 0.10.

**Phase 3 fine NDCG@16 combo 18 is the best config across all Phase 3 sweeps** and beats Phase 2 coarse Spearman best:
- Spearman: 0.557 vs 0.535 (+0.022)
- playoff_spearman: 0.568 vs 0.547 (+0.021)
- NDCG@16: 0.550 vs 0.543 (+0.007)
- NDCG@4: 0.506 vs 0.511 (-0.005, slight trade-off)
- rank_mae: 6.33 (tied) | rank_rmse: 8.15 vs 8.35 (better)

### 6.6 Phase 3 Progression and Recommendations

```
Phase 2 coarse (Spearman)  → Spearman 0.535, NDCG@4 0.511, NDCG@16 0.543
Phase 3 coarse ndcg4       → NDCG@4 0.506, Spearman 0.492
Phase 3 fine ndcg4         → NDCG@4 0.506, Spearman 0.513
Phase 3 coarse ndcg16      → NDCG@16 0.540, Spearman 0.543
Phase 3 fine ndcg16        → NDCG@16 0.550, Spearman 0.557 ← Best overall
```

**Recommendations:**
1. **Lock Phase 3 fine NDCG@16 combo 18** as the new best config for balanced ranking and playoff prediction.
2. **Config path:** `outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`
3. **Explain:** `python -m scripts.5b_explain --config outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`
4. For NDCG@4–focused use cases, use phase3_coarse_ndcg4 combo 8 or phase3_fine_ndcg4 combo 6 (both NDCG@4 0.506).

---

## 7. Recommendations (Updated)

1. **Lock Phase 3 fine NDCG@16 combo 18** as the current best config (Spearman 0.557, playoff_spearman 0.568, NDCG@16 0.550).
2. **Fallback:** Phase 2 coarse combo 8 remains strong for Spearman-focused runs.
3. **Explain on best combo:** `python -m scripts.5b_explain --config outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`
4. Update production/default config with combo 18 params.
