# Run 022 — Results Analysis & Outputs Explained

This document explains what each output file is, what each metric means, and how to interpret run_022’s results.

---

## 1. Output layout (run_022)

| File | Source | Purpose |
|------|--------|--------|
| **predictions.json** | Script 6 (inference) | All teams, all test seasons combined (primary artifact for “latest run”). |
| **predictions_2023-24.json**, **predictions_2024-25.json** | Script 6 | Per-season predictions when `test_seasons` is used. |
| **eval_report.json** | Script 5 (evaluate) | Full evaluation report: ensemble + Model A / Model B / Model C metrics, by-conference, playoff metrics, notes. When multiple test seasons exist, the **summary** metrics in the report are from the **last** test season (2024-25). |
| **eval_report_2023-24.json**, **eval_report_2024-25.json** | Script 5 | Per-season metrics so you can compare 2023-24 vs 2024-25. |
| **ANALYSIS_01.md** | Script 5 | Short human-readable summary: run id, EOS source, and test metrics (ensemble) from the report. |
| **pred_vs_actual_*.png**, **pred_vs_playoff_rank_*.png** | Script 6 / 5 | Scatter plots: predicted strength/rank vs actual EOS/playoff rank, per season. |
| **eos_playoff_standings_vs_eos_global_rank_*.png** | Script 6 | EOS playoff standings vs EOS global rank. |
| **odds_top10_*.png**, **title_contender_scatter_*.png** | Script 6 | Championship odds and contender scatter, per season. |

**Pipeline order:** 3 (train Model A) → 4 (Model B) → 4b (stacking) → 6 (inference) → 5 (evaluate). So run_022 is produced by scripts 3–5 with outputs under `outputs2/`.

---

## 2. What each metric means

- **NDCG / NDCG@10 (ndcg10)**: Quality of the **ranking** of teams. Relevance = strength (e.g. derived from EOS global rank). Higher = predicted order matches true strength better. 1.0 = perfect order. ndcg and ndcg10 are the same (k=10).
- **Spearman**: Correlation between **predicted score** and **actual strength** (rank). Positive = higher predictions for better teams. Strong positive ≈ 0.5–0.8.
- **MRR top-2**: 1 / (rank of first team that is champion or runner-up in the predicted order). 1.0 = best team or runner-up is predicted #1.
- **MRR top-4**: Same idea for “top 4” (conference finals). 1.0 = one of the true top 4 is predicted first.
- **ROC-AUC upset**: Ability to separate “sleepers” (teams that ended up better than predicted) from non-sleepers. Upset = EOS global rank stronger than predicted. >0.5 = better than random; ~0.7+ = useful.
- **Playoff metrics** (when playoff data exists):
  - **spearman_pred_vs_playoff_rank**: Correlation of predicted strength vs **post-playoff** rank (champion=1, etc.).
  - **ndcg_at_4_final_four**: NDCG@4 using final four (playoff) outcome as relevance.
  - **ndcg10_pred_vs_playoff**: NDCG@10 with playoff rank as relevance (top 10 ranking quality).
  - **rank_mae_pred_vs_playoff**: Mean absolute error of predicted rank vs actual playoff rank (lower = better).
  - **rank_rmse_pred_vs_playoff**: RMSE of rank predictions; penalizes large errors more (lower = better).
  - **rank_mae_standings_vs_playoff**, **rank_rmse_standings_vs_playoff**: Baselines (standings vs playoff outcome).
  - **brier_championship_odds**: Brier score for “champion yes/no” vs model’s championship probabilities. Lower = better calibrated.

**EOS source:** `eos_final_rank` means ground truth is **end-of-season final rank** (playoff outcome when available), not just regular-season standings.

---

## 3. Run 022 — Summary (ensemble, last test season 2024-25)

The numbers in **ANALYSIS_01.md** and in the main **eval_report.json** summary are from the **last test season** (2024-25) when multiple seasons are evaluated.

| Metric | Value | Interpretation |
|--------|--------|----------------|
| **NDCG** | 0.482 | Moderate: predicted order is partly aligned with true strength; room to improve. |
| **Spearman** | 0.43 | Moderate positive correlation; model ranks stronger teams higher on average. |
| **MRR top-2** | 0.50 | First “top-2” team appears at predicted rank 2 on average (1/2). |
| **MRR top-4** | 0.50 | First “top-4” team at rank 2 on average. |
| **ROC-AUC upset** | 0.73 | Decent ability to identify sleepers vs non-sleepers. |
| **Playoff** | spearman_pred_vs_playoff_rank=0.46, ndcg_at_4_final_four=0.46, brier_championship_odds=0.032 | Playoff ordering and championship odds are reasonable; Brier is low (good). |

Overall: the ensemble is **learning** (non-trivial NDCG and Spearman, good upset AUC and playoff metrics). This run used the attention fallback and no all-masked batches; attention stayed non-zero in training and inference.

---

## 4. By model (2024-25)

| Model | NDCG | Spearman | MRR@2 | MRR@4 | ROC-AUC upset |
|-------|------|----------|-------|-------|----------------|
| **Ensemble** | 0.482 | 0.43 | 0.50 | 0.50 | 0.73 |
| **Model A (DeepSet)** | 0.482 | 0.43 | 0.50 | 0.50 | 0.73 |
| **Model B (XGBoost)** | 0.149 | 0.52 | 0.14 | 0.14 | 0.79 |
| **Model C (RF)** | 0.05 | 0.31 | 0.07 | 0.20 | 0.82 |

- **Model A** matches the ensemble on NDCG/Spearman/MRR here because the stacked ensemble is dominated by Model A on this run/split. (Terminology: Model B = XGBoost, Model C = Random Forest.)
- **Model B** has higher Spearman (0.52) and better upset AUC (0.79) but much lower NDCG/MRR — so it correlates with strength but its **ordering** (especially top slots) is worse.
- **Model C** has the best upset AUC (0.82) but weakest ranking (NDCG 0.05, Spearman 0.31). So ensemble/Model A are carrying ranking; tree models add some signal for upsets/calibration.

---

## 5. By season (2023-24 vs 2024-25)

| Metric | 2023-24 | 2024-25 |
|--------|---------|---------|
| NDCG | 0.559 | 0.482 |
| Spearman | 0.31 | 0.43 |
| MRR top-2 | 1.0 | 0.5 |
| MRR top-4 | 1.0 | 0.5 |
| ROC-AUC upset | 0.86 | 0.73 |
| spearman_pred_vs_playoff_rank | 0.35 | 0.46 |
| ndcg_at_4_final_four | 0.35 | 0.46 |
| brier_championship_odds | 0.032 | 0.032 |

- **2023-24**: Better NDCG, perfect MRR (top-2 and top-4 found at rank 1), higher upset AUC. Playoff alignment (Spearman, NDCG@4) slightly lower.
- **2024-25**: Lower NDCG and MRR, but **higher** Spearman and better playoff alignment (0.46 vs 0.35). Brier unchanged.
- So 2023-24 was “easier” for top-team ordering (MRR=1); 2024-25 is harder for top rank but the model’s playoff and strength correlation improved.

---

## 6. By conference (2024-25)

Relevance is now **per conference**: within each conference, actual rank is derived from EOS global rank (1 = best in conference, 15 = worst). NDCG uses that relevance vs ensemble score; Spearman uses derived actual conference rank vs predicted conference rank.

| Conference | NDCG | Spearman |
|------------|------|----------|
| East (E) | 0.25 | **0.25** |
| West (W) | 0.75 | **0.50** |

Both Spearman values are positive and interpretable: within East the model’s conference ordering correlates 0.25 with EOS strength; within West, 0.50. West NDCG is higher than East, reflecting better alignment of predicted order with EOS strength in the West.

---

## 7. Takeaways

1. **Outputs:** Predictions live in `predictions.json` / `predictions_<season>.json`; evaluation in `eval_report.json` and `eval_report_<season>.json`; ANALYSIS_01.md is the short summary (last test season).
2. **Metrics:** NDCG/Spearman measure ranking quality; MRR measures “where does the first true top-2 / top-4 appear”; ROC-AUC upset and playoff metrics measure sleepers and playoff alignment.
3. **Run 022:** Ensemble (and Model A) give moderate but meaningful ranking (NDCG ~0.48, Spearman ~0.43) and good upset/championship calibration; XGB/RF add more for upset detection than for top-order.
4. **Seasons:** 2023-24 has better MRR and upset AUC; 2024-25 has better playoff Spearman and NDCG@4.
5. **Conference:** Per-conference relevance is EOS-derived within conference (rank 1..n by EOS_global_rank); NDCG and Spearman are now comparable and interpretable per conference.

For full numbers and by-model/by-season breakdowns, use **eval_report.json** and **eval_report_2023-24.json** / **eval_report_2024-25.json** in this folder.
