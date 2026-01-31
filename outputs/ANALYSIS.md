# Pipeline Outputs Analysis

This document explains what each output means, interprets the current results, and flags potential errors or bugs.

---

## 1. Outputs Overview

| Output | Source | Purpose |
|--------|--------|---------|
| `outputs/run_010/predictions.json` | Script 6 (inference) | Per-team predictions, analysis, roster dependence, ensemble diagnostics |
| `outputs/eval_report.json` | Script 5 (evaluate) | Ranking and playoff metrics for a run’s predictions |
| `outputs/shap_summary.png` | Script 5b (explain) | SHAP feature importance for Model B (team-level) |
| `outputs/ig_model_a_attributions.txt` | Script 5b (explain) | Integrated Gradients attributions for Model A (player indices) |

**Important:** `eval_report.json` is produced by **5_evaluate**, which uses **the latest run** that has `predictions.json`. In the pipeline order (5 → 5b → 6), evaluation runs **before** inference. So the current `eval_report.json` was computed for **run_009**, not run_010. To get metrics for run_010, run `python scripts/5_evaluate.py` again after the pipeline.

---

## 2. Predictions (run_010)

### Structure

- **prediction**: `predicted_strength` (1–30 rank), `global_rank`, `ensemble_score` (0–1), `conference_rank` (1–15), `championship_odds`.
- **analysis**: `EOS_conference_rank`, `EOS_global_rank` (actual end-of-season ranks when available), `classification` (e.g. “Aligned”, “Under-ranked by X slots”), `playoff_rank`, `rank_delta_playoffs`.
- **ensemble_diagnostics**: `model_agreement` (High/Medium/Low), per-model ranks (deep_set, xgboost, random_forest).
- **roster_dependence**: `primary_contributors` (player name + `attention_weight` from Model A).

### Interpretations

- **Boston Celtics**: `predicted_strength` 5, `EOS_global_rank` 1 → “Under-ranked by 4 slots” (model predicted 5th, actual 1st). So the ensemble under-predicts the best team.
- **Milwaukee Bucks**: predicted 3, actual 3 → “Aligned”.
- **Classification**: “Under-ranked by X” = model rank worse (higher number) than actual; “Over-ranked by X” = model rank better (lower number) than actual.

### Potential errors

1. **All `attention_weight` values are 0.0**  
   Every team’s `primary_contributors` show `attention_weight: 0.0`. The code uses `np.nan_to_num` and skips non-finite or `w <= 0`; then a fallback adds players even when weight ≤ 0. So either:
   - Model A’s attention layer is outputting zero (or negative) weights for all players, or
   - There is a bug in how attention is read or mapped to player names.  
   Training log showed `epoch 1–3 loss=27.8993` (unchanged), which supports the hypothesis that Model A did not learn useful attention.

2. **Roster contamination**  
   Boston’s roster lists “Anfernee Simons” (Blazers); Milwaukee lists “Kyle Kuzma” (Wizards), “Myles Turner” (Pacers). These are not correct historical rosters for a single “as of” date. That suggests:
   - Rosters may be built from “latest team” in a way that mixes current (2024–25) affiliations with the inference `as_of_date`, or
   - Player–team assignment in `build_roster_set` / roster resolution may be wrong for the target date.  
   So **primary_contributors** (and any player-level interpretation) should be treated as suspect until roster logic is verified.

---

## 3. Evaluation Report (eval_report.json)

### What it refers to

- **Which run:** At the time 5_evaluate ran in the pipeline, the latest run was **run_009**. So the numbers below are for **run_009**, not run_010.

### Metrics

- **ndcg**: ~0.00026 — NDCG@10 of predicted order (by `ensemble_score`) vs. actual strength (relevance from EOS rank). Near zero means predicted ranking is almost unrelated to actual strength.
- **spearman**: 0.0 — No correlation between predicted scores and actual relevance. Can also occur if one array is constant (e.g. many teams with null `playoff_rank` in playoff_metrics); the ConstantInputWarning during the run is consistent with that in a different metric.
- **mrr**: 0.0 — Top-2 MRR: the first “best” team in predicted order is not in the top 2 positions (or relevance definition doesn’t match).
- **roc_auc_upset**: 0.5 — No ability to distinguish “sleeper” (under-ranked by standings) vs. non-sleeper using the ensemble score.
- **playoff_metrics**:
  - **spearman_pred_vs_playoff_rank**: -0.18 — Weak negative correlation between predicted global rank and playoff performance rank (e.g. higher predicted rank ↔ slightly worse playoff finish, or many null playoff ranks diluting the correlation).
  - **ndcg_at_4_final_four**: 0.36 — Some overlap between “top 4 by prediction” and “final four” by playoff outcome.
  - **brier_championship_odds**: ~0.001 — Very low Brier; model assigns low championship probability to almost everyone. If the true champion also had low probability, this can be low without meaning the model is “good” at identifying the champion.

### Deductions

- The **standings-based** metrics (ndcg, spearman, mrr, roc_auc_upset) indicate that, for **run_009**, the ensemble’s ordering of teams has almost no relationship to actual end-of-season strength. That is consistent with:
  - Model A not learning (flat loss),
  - Limited or mismatched training data,
  - Or a bug in how EOS ranks or scores are aligned (e.g. conference vs global, or wrong season).
- **playoff_metrics** mix predicted rank with playoff outcome; the negative Spearman and modest NDCG@4 suggest the model’s rank is only weakly related to playoff success, and Brier should be interpreted with care (low prob everywhere → low Brier).

---

## 4. Explain Outputs (5b_explain)

- **shap_summary.png**: SHAP for Model B (XGB/RF team-level features). Visually shows which team-level features push predictions up or down. No obvious error from the pipeline; interpretation is feature-specific.
- **ig_model_a_attributions.txt**: Integrated Gradients for Model A. Top-5 player indices all have **L2 norm = 0.0000**. So Model A’s input (player-level) is receiving zero attribution. This matches:
  - Zero attention weights in predictions, and
  - Flat training loss (model may not be using player information meaningfully).
- **Attention ablation**: Pipeline log reported “Attention ablation (top-2 masked) score mean: **NaN** (masked forward produced non-finite scores)”. So masking top-2 attention leads to non-finite scores — either numerical instability or the model’s forward pass is undefined when attention is masked.

---

## 5. Summary of Deductions and Potential Errors

| Item | Deduction / Error |
|------|--------------------|
| **Eval report refers to run_009** | Current `eval_report.json` is for run_009 because 5 runs before 6. Re-run 5_evaluate to get run_010 metrics. |
| **Near-zero ranking metrics** | For run_009, predicted order does not match actual strength. Consistent with Model A not learning (flat loss) and/or data/alignment issues. |
| **All attention weights 0** | Model A attention is zero or negative; fallback fills `primary_contributors` with 0.0. Likely cause: no learning in the attention layer. |
| **IG attributions all zero** | Same story: Model A is not attributing importance to players. |
| **Attention ablation NaN** | Masked forward pass yields non-finite scores; suggests numerical or implementation issues in ablation. |
| **Roster contamination** | Wrong-team players (e.g. Simons, Kuzma, Turner) in rosters suggest “latest team” or as_of_date logic is wrong for historical inference. |
| **ConstantInputWarning** | One of the Spearman inputs (e.g. playoff_metrics with many null playoff_rank) may be constant or near-constant; worth checking which series is passed to `spearman`. |

---

## 6. Recommended Next Steps

1. **Re-run evaluation for run_010**: `python scripts/5_evaluate.py` (with run_010 as latest) to see if metrics change.
2. **Diagnose Model A**: Check why loss is flat (learning rate, architecture, target definition, or data pipeline).
3. **Fix roster logic**: Verify `build_roster_set` / “latest team” and as_of_date so rosters match the inference date and avoid current-season contamination.
4. **Harden explain pipeline**: Handle non-finite scores in attention ablation and document when IG/attention are uninformative (e.g. all zeros).
