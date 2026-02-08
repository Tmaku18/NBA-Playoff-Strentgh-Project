# Sweep Analysis — Phase 2 & Phase 3 Results

**Project:** NBA True Strength Prediction  
**Sweep root:** outputs4 (run_024/run_025); config: `config/outputs4_phase1.yaml`  
**Last updated:** February 2026

---

## 1. Overview

Comprehensive analysis of Phase 2 (Spearman-optimized) and Phase 3 (NDCG-optimized) sweeps. Raw results: `outputs4/sweeps/OUTPUTS4_ANALYSIS.md`.

| Sweep | Trials | Objective | listmle_target | Status | Best Spearman | Best NDCG@4 | Best NDCG@16 |
|-------|--------|-----------|----------------|--------|---------------|------------|--------------|
| **phase3_fine_ndcg16_final_rank** | 20 | ndcg16 | final_rank | **Complete** | **0.557** | 0.506 | **0.550** |
| phase3_coarse_ndcg16_final_rank | 20 | ndcg16 | final_rank | Complete | 0.543 | 0.473 | 0.540 |
| **phase2_coarse_spearman_final_rank** | 15 | spearman | final_rank | Complete | 0.535 | 0.511 | 0.543 |
| phase3_coarse_ndcg4_final_rank | 20 | ndcg4 | final_rank | Complete | 0.492 | 0.506 | 0.547 |
| phase3_fine_ndcg4_final_rank | 20 | ndcg4 | final_rank | Complete | 0.513 | 0.506 | 0.545 |
| phase1_rolling_spearman_final_rank | 12 | spearman | final_rank | Complete | 0.496 | — | — |
| phase1_spearman_final_rank | 12 | spearman | final_rank | Partial | — | — | — |
| phase1_ndcg4_final_rank | 12 | ndcg4 | final_rank | Partial | — | — | — |

---

## 2. Progression Summary

```
run_022 (baseline)        → Spearman 0.430, playoff_spearman 0.461
Phase 1 (outputs3)        → Spearman 0.499, playoff_spearman 0.518
Rolling sweep             → Spearman 0.496, playoff_spearman 0.501 (rolling [15,30])
Phase 2 coarse            → Spearman 0.535, playoff_spearman 0.547, NDCG@16 0.543
Phase 3 fine ndcg16       → Spearman 0.557, playoff_spearman 0.568, NDCG@16 0.550 ← Best
```

**Phase 3 fine NDCG@16 (combo 18)** is the best configuration across Spearman, playoff_spearman, and NDCG@16. It is the production default in `config/defaults.yaml`.

---

## 3. Phase 2 Coarse Sweep (phase2_coarse_spearman_final_rank)

**Sweep:** Optuna 15 trials; `rolling_windows: [15, 30]` fixed; epochs 20–24, max_depth 5, min_leaf 5; lr, n_xgb, n_rf varied.

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

| Metric | Phase 2 (combo 8) | Rolling best | run_022 |
|--------|-------------------|--------------|---------|
| **Spearman** | **0.535** | 0.496 | 0.430 |
| **playoff_spearman** | **0.547** | 0.501 | 0.461 |
| **NDCG@16** | **0.543** | 0.490 | 0.48 |
| **NDCG@4** | **0.511** | 0.464 | 0.46 |
| rank_mae | 6.33 | 6.73 | 7.53 |
| rank_rmse | 8.35 | 8.69 | 9.24 |

### Optuna importances

| Param | Importance |
|-------|------------|
| learning_rate | 0.40 |
| n_estimators_xgb | 0.24 |
| n_estimators_rf | 0.24 |
| model_a_epochs | 0.13 |
| max_depth, min_leaf, subsample, rolling_windows, colsample | 0 (fixed) |

---

## 4. Phase 3 NDCG Sweeps

Phase 3 sweeps target NDCG@4 and NDCG@16 (coarse → fine). All use `rolling_windows: [15, 30]`, `listmle_target: final_rank`.

### 4.1 Summary Table

| Sweep | Trials | Objective | Best Combo | NDCG@4 | NDCG@16 | Spearman | playoff_spearman |
|-------|--------|-----------|------------|--------|---------|----------|------------------|
| phase3_coarse_ndcg4 | 20 | ndcg4 | 8 | **0.506** | 0.547 | 0.492 | 0.489 |
| phase3_fine_ndcg4 | 20 | ndcg4 | 6 | **0.506** | 0.545 | 0.513 | 0.490 |
| phase3_coarse_ndcg16 | 20 | ndcg16 | 8 | 0.473 | **0.540** | **0.543** | **0.555** |
| phase3_fine_ndcg16 | 20 | ndcg16 | 18 | 0.506 | **0.550** | **0.557** | **0.568** |

### 4.2 Phase 3 Coarse NDCG@4

**Best combo 8:** epochs 21, lr 0.066, n_xgb 267, n_rf 215  
- NDCG@4: 0.506 | NDCG@16: 0.547 | Spearman: 0.492  
- **Optuna importances:** n_rf 0.42, lr 0.41, n_xgb 0.12, epochs 0.05  

**Trade-off:** NDCG@4-optimized config yields higher NDCG but lower Spearman vs Phase 2 Spearman-optimized (0.535).

### 4.3 Phase 3 Fine NDCG@4

**Best combo 6:** epochs 21, lr 0.086, n_xgb 239, n_rf 202  
- NDCG@4: 0.506 | NDCG@16: 0.545 | Spearman: 0.513  
- **Optuna importances:** epochs 0.60, n_xgb 0.18, n_rf 0.14, lr 0.08  

Fine sweep did not improve NDCG@4 beyond coarse (both 0.506).

### 4.4 Phase 3 Coarse NDCG@16

**Best combo 8:** epochs 20, lr 0.066, n_xgb 222, n_rf 164  
- NDCG@16: 0.540 | NDCG@4: 0.473 | Spearman: 0.543 | playoff_spearman: 0.555  
- **Optuna importances:** n_xgb 0.31, n_rf 0.29, lr 0.23, epochs 0.17  

NDCG@16 best also achieves strong Spearman and playoff_spearman among Phase 3 coarse sweeps.

### 4.5 Phase 3 Fine NDCG@16 — **Best Overall**

**Best combo 18:** epochs 22, lr 0.072, n_xgb 229, n_rf 173  
- **NDCG@4:** 0.506 | **NDCG@16:** 0.550 | **Spearman:** 0.557 | **playoff_spearman:** 0.568  
- **rank_mae:** 6.33 | **rank_rmse:** 8.15 | ROC-AUC upset: 0.773  
- **Optuna importances:** lr 0.55, n_xgb 0.21, epochs 0.15, n_rf 0.10  

**Phase 3 fine NDCG@16 combo 18** beats Phase 2 coarse Spearman best:

| Metric | Phase 3 fine (combo 18) | Phase 2 coarse (combo 8) | Delta |
|--------|-------------------------|--------------------------|-------|
| Spearman | 0.557 | 0.535 | +0.022 |
| playoff_spearman | 0.568 | 0.547 | +0.021 |
| NDCG@16 | 0.550 | 0.543 | +0.007 |
| NDCG@4 | 0.506 | 0.511 | -0.005 |
| rank_rmse | 8.15 | 8.35 | better |

---

## 5. Recommendations

1. **Production default:** Phase 3 fine NDCG@16 combo 18 (Spearman 0.557, playoff_spearman 0.568, NDCG@16 0.550). Baked into `config/defaults.yaml`.
2. **Config path:** `outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`
3. **Explain on best combo:** `python -m scripts.5b_explain --config outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`
4. **NDCG@4–focused:** Use phase3_coarse_ndcg4 combo 8 or phase3_fine_ndcg4 combo 6 (both NDCG@4 0.506).
5. **Fallback:** Phase 2 coarse combo 8 remains strong for Spearman-focused runs.

---

## 6. Related Docs

- [HYPERPARAMETER_TESTING_EVOLUTION.md](HYPERPARAMETER_TESTING_EVOLUTION.md) — Sweep methodology, Optuna, phased grids
- [METRICS_USED.md](METRICS_USED.md) — NDCG, Spearman, rank_mae/rmse definitions
- `outputs4/sweeps/OUTPUTS4_ANALYSIS.md` — Full raw sweep analysis with all combo details
