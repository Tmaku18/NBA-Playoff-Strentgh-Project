# Hypothesized Best Configuration by Metric and Model

This document synthesizes online research, run_022 results, and project hypotheses to guide sweep strategy and evaluation. **Terminology:** Model A = Deep Set (attention-based ranker); Model B = XGBoost; Model C = Random Forest. The ensemble stacks A + B + C via RidgeCV meta-learner.

---

## 1. Summary

### Current state (run_022, baseline config)

Run_022 uses `listmle_target: playoff_outcome` and EOS source `eos_final_rank`. Ensemble metrics (2024-25): **NDCG 0.48**, **Spearman 0.43**, **playoff Spearman 0.46**, **NDCG@4 0.46**, **Brier 0.032**, **rank_mae pred vs playoff 7.53**, **rank_rmse pred vs playoff 9.24**. Standings vs playoff baseline: **rank_mae 3.13**, **rank_rmse 4.45** — standings currently outperform model predictions on rank distance, which is expected since regular-season win % correlates with playoff seeding and early outcomes.

**By model (2024-25):**

| Model     | NDCG | Spearman | MRR@2 | MRR@4 | ROC-AUC upset |
|-----------|------|----------|-------|-------|---------------|
| Ensemble  | 0.48 | 0.43     | 0.50  | 0.50  | 0.73          |
| Model A   | 0.48 | 0.43     | 0.50  | 0.50  | 0.73          |
| Model B   | 0.15 | 0.52     | 0.14  | 0.14  | 0.79          |
| Model C   | 0.05 | 0.31     | 0.07  | 0.20  | 0.82          |

Model A dominates the ensemble on NDCG/Spearman/MRR; Model B has higher Spearman and better upset AUC but weaker top-order ranking; Model C has best upset AUC but weakest ranking.

**Research context:** FiveThirtyEight uses CARM-Elo (player projections, Elo, schedule); research ensembles combine XGBoost, RF, MLP, etc. NDCG is recommended for playoff prediction when ordering teams by playoff success probability matters. Our goal is to show that **stats-based predictions (optimized for playoff outcome) beat standings vs actual** on every metric — proving stats matter more than raw wins and losses.

---

## 2. Hypothesized and Actual Best Configuration per Metric

### Phase 0 baseline results (baseline_spearman_final_rank)

**Sweep:** spearman, listmle_target=final_rank (standings). Best combo (2): epochs 26, max_depth 4, lr 0.094, n_xgb 209, n_rf 182, min_leaf 6.

| Metric | Best (combo 2) | run_022 |
|--------|----------------|---------|
| Spearman | 0.492 | 0.430 |
| playoff_spearman | 0.499 | 0.461 |
| rank_mae_pred_vs_playoff | 6.80 | 7.53 |
| rank_rmse_pred_vs_playoff | 8.72 | 9.24 |
| NDCG | 0.483 | 0.482 |

Standings-trained baseline sweep **outperformed** run_022 (playoff_outcome) on Spearman, playoff_spearman, rank_mae, rank_rmse. Full analysis: `outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md`.

### Phase 0 NDCG baseline results

**Sweep 3 (baseline_ndcg_final_rank):** Best combo (3): epochs 20, max_depth 5, lr 0.092, n_xgb 269, n_rf 167. NDCG 0.486, playoff_spearman 0.511, rank_mae 6.93.

**Sweep 4 (baseline_ndcg_playoff_outcome):** Best combo (1): epochs 14, max_depth 3, lr 0.065, n_xgb 283, n_rf 228. NDCG 0.486, playoff_spearman 0.504, rank_mae 6.80. (4/6 combos aggregated after interruption.)

**Phase 0 summary:** Playoff-optimized spearman (sweep 2) leads on Spearman (0.512) and playoff_spearman (0.513). NDCG sweeps both reached 0.486. Full comparison: `outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md`.

### Phase 1 spearman results (outputs3)

**phase1_spearman_final_rank** (listmle_target=standings): Best combo 10 — epochs 21, max_depth 5, lr 0.0704, n_xgb 291, n_rf 164, min_leaf 4. **Spearman 0.499**, **playoff_spearman 0.518**, rank_mae 6.73, rank_rmse 8.67, NDCG 0.485.

**phase1_spearman_playoff_outcome** (listmle_target=playoff): Best combo 3 — epochs 21, max_depth 4, lr 0.0669, n_xgb 280, n_rf 151, min_leaf 5. Spearman 0.482, playoff_spearman 0.487, rank_mae 6.87, rank_rmse 8.81.

**Phase 1 finding:** final_rank (standings) target **outperformed** playoff_outcome on spearman, playoff_spearman, rank_mae, rank_rmse — opposite of Phase 0. Full analysis: `outputs4/sweeps/SWEEP_PHASE1_ANALYSIS.md`.

### Hypothesized (pre-sweep expectations)

| Objective              | Best model (expected) | Rationale                                                                 |
|------------------------|-----------------------|---------------------------------------------------------------------------|
| **spearman**           | Ensemble or Model A   | Correlation benefits from stacking; Model A uses ListMLE with relevance. |
| **ndcg4 / ndcg16 / ndcg20** | Model A        | ListMLE is a ranking loss; NDCG@k aligns with listwise optimization (see §3.6). |
| **playoff_spearman**   | Ensemble or Model A   | Playoff-outcome target should drive playoff alignment; ensemble reduces noise. |
| **rank_rmse**          | Model A               | RMSE penalizes large errors; ListMLE optimizes ordering.                  |

*Eval-only (computed but not sweep objectives):* ndcg, ndcg10, ndcg12, rank_mae.

**Standings-target sweep:** If a separate sweep uses `listmle_target: final_rank` (standings) and `--objective playoff_spearman`, the hypothesis is that the best config from that sweep will have *higher* playoff_spearman than the playoff-outcome sweep, because training on standings aligns with the correlation structure of standings vs playoff outcome. Conversely, **playoff-outcome-target sweeps** should yield better ndcg, mrr, mae, and rmse across models, because the optimization target matches the evaluation target.

---

## 3. Personal Hypotheses

### 3.1 Model A under ranking objectives

When sweeps optimize for **NDCG**, **rank_mae**, or **rank_rmse**, Model A is expected to perform best. ListMLE directly optimizes ranking; those metrics measure ranking quality. Eventually the ensemble should be best once noise is reduced (better stacking weights, hyperparameter tuning).

### 3.2 Playoff vs standings optimization

- **Optimizing for spearman_playoff** (pred vs playoff rank): Should give better **ndcg**, **mrr**, **mae**, and **rmse** across models compared with optimizing for standings.
- **Optimizing for standings_playoff** (standings vs playoff): Will likely give better **spearman_playoff** across models, because the training signal matches the correlation between standings and playoff outcome.
- **Success criterion:** Predicted (optimized for playoff) should **beat standings vs actual** in every metric. That proves the model adds value beyond raw wins and losses.

### 3.3 Model B vs Model C

- Model B should outperform Model C across the board on ranking metrics (NDCG, Spearman, MRR, rank_mae, rank_rmse).
- Model C should be used as a **stabilizer** for the ensemble unless it hurts more than it helps. If Model C degrades ensemble ranking, consider reducing its weight or excluding it.

### 3.4 MRR top-2 vs top-4

- **MRR top-4** should be best for the ensemble: top 4 has more tolerance, and the ensemble can better account for noise across models.
- **MRR top-2** will be more consistent across models because it has less tolerance (champion + runner-up only); small changes in ordering can flip MRR@2.

### 3.5 Success definition and proof

- **Different hyperparameters** may be optimal for different metrics; sweeps will identify best configs per objective.
- There should exist a **best ensemble or single model (A, B, or C)** that beats the best standings-vs-actual baseline on all evaluated metrics.
- **Proving stats matter:** If our best playoff-optimized model outperforms standings vs actual on NDCG, Spearman, MRR, rank_mae, rank_rmse, and Brier, we have evidence that statistical strength estimates matter more than wins and losses for playoff prediction.
- **Comparative analysis and attention weights** should yield additional insights: per-conference behavior, which features drive Model A, and how Model B/C contribute to the stack.

### 3.6 NDCG@k and playoff structure (hypothesis)

NBA playoff structure motivates different NDCG cutoffs:

| NDCG@k | Rationale | Expected strength |
|--------|-----------|-------------------|
| **NDCG@12** | Matches the 12 teams that **guarantee** playoff berths (seeds 1–6 per conference, excluding play-in). These are the strongest teams. Optimizing for NDCG@12 focuses on rank ordering of lock-in playoff teams. | **Best at MRR** — predicts everyone guaranteed to make playoffs; cares about rank ordering of the strongest teams. |
| **NDCG@16** | Matches the full 16-team playoff field. | **Best at determining who makes the playoffs** — aligns with the full bracket. |
| **NDCG@20** | Extends beyond the field to include the play-in zone (seeds 7–10 per conference = 8 teams). | **Best at handling play-in noise** — captures the wider “playoff-adjacent” zone and uncertainty of seeds 7–10. |

**Playoff Spearman vs NDCG@k:**

- Models optimized for **playoff_spearman** may **suffer worse NDCG** but give **better RMSE and MAE**, because playoff Spearman better accounts for the strength of teams across the full ranking.
- **Hypothesis:** By changing the NDCG cutoff (e.g., NDCG@12, NDCG@16, NDCG@20), we may be able to **beat models optimized by playoff_spearman** on rank_mae/rank_rmse while maintaining or improving NDCG. Different k values target different aspects of the playoff structure.

---

## 4. Terminology and Script Naming

**From this document onward, use:**

- **Model A** = Deep Set (attention-based ranker; script 3)
- **Model B** = XGBoost (formerly “XGB”)
- **Model C** = Random Forest (formerly “RF”)

**Script naming for clarity:**

- `3_train_model_a.py` — unchanged
- `4_train_models_b_and_c.py` — trains both Model B (XGBoost) and Model C (RF). Script renamed from 4_train_model_b.py for clarity.
- Eval outputs use `test_metrics_model_b` and `test_metrics_model_c`; predictions include `model_b_rank` and `model_c_rank` in ensemble_diagnostics.
- Config keys `model_b.xgb` and `model_b.rf` can remain for backward compatibility, or be migrated to `model_b` and `model_c` under `model_b` / `model_c` sections.

---

## 5. Metrics Reference (quick)

| Metric                  | Interpretation                                      | Direction  |
|-------------------------|-----------------------------------------------------|------------|
| NDCG / NDCG@10          | Ranking quality (top-heavy); 1.0 = perfect order   | Higher     |
| NDCG@4 (final four)     | Same, truncated at top 4                           | Higher     |
| NDCG@12                 | Top 12 (guaranteed playoff teams); hypothesized best for MRR | Higher     |
| NDCG@16                 | Full 16-team playoff field; hypothesized best for “who makes playoffs” | Higher     |
| NDCG@20                 | Top 20 (includes play-in zone); hypothesized best for play-in noise | Higher     |
| Spearman                | Correlation of pred vs actual rank                 | Higher     |
| playoff_spearman        | Pred vs playoff outcome rank                       | Higher     |
| MRR top-2 / top-4       | 1 / (rank of first true top-2 or top-4 team)      | Higher     |
| rank_mae, rank_rmse     | Mean / RMSE of |pred_rank − actual_rank|         | Lower      |
| Brier (championship)    | Calibration of champion probabilities              | Lower      |
| ROC-AUC upset           | Ability to identify sleepers vs non-sleepers       | Higher     |

---

## 6. Next Steps

1. **Phase 0 complete.** All 4 baseline sweeps ran. Commit and push outputs.
2. **Phase 1:** Add `--phase phase1` with narrowed ranges; run full sweeps (20 trials each) for `spearman`, `ndcg4`, `ndcg16`, `ndcg20`, `playoff_spearman`, `rank_rmse` (each with final_rank + playoff_outcome).
3. **Eval/analytics only** (calculated but not sweep objectives): `ndcg`, `ndcg10`, `ndcg12`, `rank_mae` — appear in eval reports and sweep summaries for analysis.
4. Populate sweep summaries with `best_by_*` for each metric.
5. Compare best configs across objectives; validate which model (A, B, C, or ensemble) leads per metric.
6. Test standings-target vs playoff-target sweeps for playoff_spearman.
7. Update scripts and eval outputs to use Model B / Model C naming.
8. Use attention weights and comparative analysis to refine hypotheses and report findings.
