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

### Hypothesized (pre-sweep expectations)

| Objective              | Best model (expected) | Rationale                                                                 |
|------------------------|-----------------------|---------------------------------------------------------------------------|
| **spearman**           | Ensemble or Model A   | Correlation benefits from stacking; Model A uses ListMLE with relevance. |
| **ndcg / ndcg10**      | Model A               | ListMLE is a ranking loss; top-heavy NDCG aligns with listwise optimization. |
| **playoff_spearman**   | Ensemble or Model A   | Playoff-outcome target should drive playoff alignment; ensemble reduces noise. |
| **rank_mae**           | Model A               | Rank distance benefits from ListMLE’s explicit rank optimization.        |
| **rank_rmse**          | Model A               | Same as rank_mae; RMSE penalizes large errors, ListMLE optimizes ordering. |

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
| Spearman                | Correlation of pred vs actual rank                 | Higher     |
| playoff_spearman        | Pred vs playoff outcome rank                       | Higher     |
| MRR top-2 / top-4       | 1 / (rank of first true top-2 or top-4 team)      | Higher     |
| rank_mae, rank_rmse     | Mean / RMSE of |pred_rank − actual_rank|         | Lower      |
| Brier (championship)    | Calibration of champion probabilities              | Lower      |
| ROC-AUC upset           | Ability to identify sleepers vs non-sleepers       | Higher     |

---

## 6. Next Steps

1. **Phase 0 (in progress):** Run baseline_spearman_playoff_outcome, baseline_ndcg_final_rank, baseline_ndcg_playoff_outcome. Compare standings vs playoff target.
2. Run sweeps for each objective: `spearman`, `ndcg`, `ndcg10`, `playoff_spearman`, `rank_mae`, `rank_rmse`.
3. Populate sweep summaries with `best_by_*` for each metric.
4. Compare best configs across objectives; validate which model (A, B, C, or ensemble) leads per metric.
5. Test standings-target vs playoff-target sweeps for playoff_spearman.
6. Update scripts and eval outputs to use Model B / Model C naming.
7. Use attention weights and comparative analysis to refine hypotheses and report findings.
