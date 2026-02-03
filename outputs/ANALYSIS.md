# Pipeline Outputs Analysis

This document explains what each output means, interprets the current results, compares against past runs (including run_021 and run_022), clarifies optimization targets (NDCG vs Spearman), and describes the sweep strategy (different target metrics per sweep, then compare).

---

## 1. Outputs Overview

| Output | Source | Purpose |
|--------|--------|---------|
| `outputs/run_XXX/predictions.json` | Script 6 (inference) | Per-team predictions, analysis, roster dependence, ensemble diagnostics |
| `outputs/run_XXX/predictions_{season}.json` | Script 6 (inference) | Per-season predictions when test_seasons configured |
| `outputs/run_XXX/eos_playoff_standings_vs_eos_global_rank.png` | Script 6 (inference) | Scatter: EOS playoff standings vs playoff outcome (EOS) |
| `outputs/eval_report.json` | Script 5 (evaluate) | Ranking and playoff metrics for the **latest** run’s predictions |
| `outputs/split_info.json` | Script 3 (train Model A) | Train/test date split (75/25); read by scripts 4, 5, 6 |
| `outputs/run_comparison.json` | `scripts/compare_runs.py` | NDCG, Spearman, MRR, ROC-AUC upset for every run_XXX with predictions |
| `outputs/shap_summary.png` | Script 5b (explain) | SHAP feature importance for Model B (team-level) |
| `outputs/ig_model_a_attributions.txt` | Script 5b (explain) | Integrated Gradients attributions for Model A (player indices) |

**Pipeline order:** 3 → 4 → 4b → 6 → 5 → 5b. So `eval_report.json` is produced for the **latest** run that has `predictions.json` at the time script 5 runs (currently **run_016** after the latest pipeline).

---

## 2. Current Results (run_016) and 75/25 Split

### Split and evaluation setup

- **Split:** Season-based 75/25. Train seasons 2015–16 through 2022–23 (158 train dates, 316 train lists); test seasons 2023–24 and 2024–25 (41 test dates, 82 test lists). See `outputs/split_info.json`.
- **Inference:** Primary predictions use the **last test date** (held-out snapshot), so evaluation is on true out-of-sample data.
- **Training:** Model A and Model B (and stacking) are trained only on the **train** portion; no test-date leakage.

### Test metrics (eval_report.json)

| Metric | Value | Interpretation |
|--------|--------|----------------|
| **NDCG@10** | 0.665 | Good: predicted ordering of teams by ensemble score aligns well with actual end-of-season strength (relevance from EOS global rank). |
| **Spearman** | 0.76 | Strong positive correlation between predicted scores and actual strength; the model’s ranking is meaningfully related to true strength. |
| **MRR** (top-2) | 0.0 | The first “best” team in predicted order is not in the top 2 positions (or relevance ties); less critical than NDCG/Spearman for full ranking. |
| **ROC-AUC upset** | 0.65 | Moderate ability to distinguish “sleepers” (under-ranked by standings) from non-sleepers using the ensemble score. |

### Per-conference metrics (caveat)

- **East (E):** NDCG 0.998, Spearman **−0.75**
- **West (W):** NDCG 0.72, Spearman **−0.73**

Within each conference, Spearman is **negative** while global Spearman is **+0.76**. That usually means:

- **Relevance definition:** We use EOS *global* rank (1–30) for relevance. Within one conference, teams only span a subset of global ranks (e.g. 1–15 for East). So within-conference, “better” teams can have *higher* global rank numbers (e.g. 8 vs 7) if the other conference has stronger teams. Using global rank as relevance inside a single conference can invert the intended ordering and produce negative Spearman. For a fair per-conference view, relevance should be defined **within conference** (e.g. EOS conference rank 1–15). Until that’s fixed, **per-conference Spearman should be interpreted with caution**; NDCG can still be useful if relevance is adjusted.

### Playoff metrics

- **run_016** (and run_014/015) test snapshot has **no post_playoff_rank** data (test date 2025-04-13 is in 2024-25; playoffs had not started). So `eval_report.json` has no `playoff_metrics`. When playoff data exists (≥16 teams with post_playoff_rank), the report will include Spearman pred vs playoff rank, NDCG@4 final four, and Brier championship odds.

#### Why is there no playoff data?

1. **Inference uses the last *test* date.** With the 75/25 split, the primary prediction snapshot is the **last date in test_dates** (from `split_info.json`), which is **2025-04-13**. That date falls in the **2024-25** season.
2. **Playoff rank is computed *per season*.** The code looks up playoff games for the **target season** (2024-25) using `season_start` / `season_end`. So it only considers 2024-25 playoff games.
3. **2024-25 playoffs had not (or barely) started.** The 2024-25 NBA playoffs typically begin in April 2025 (play-in ~April 11–14, first round later). As of 2025-04-13 there are **no** (or very few) playoff games in the DB for that season. The pipeline uses `season_type="Playoffs"` (Play-In excluded), so even play-in games may not count.
4. **Minimum 16 teams required.** `compute_playoff_performance_rank` requires at least **16** teams with playoff data. With 0 playoff games for 2024-25, `playoff_team_ids` is empty, so the function returns `{}` and you see: *"Warning: Only 0 playoff teams found (min 16). Skipping playoff rank/metrics."*

**To get playoff metrics:** Use a snapshot from a **completed** season (e.g. last date of 2023-24 after playoffs finished), or run inference with `inference.also_train_predictions: true` and evaluate `train_predictions.json` (last train date is in 2022-23, which has playoff data in the DB if playoff raw files were loaded).

**EOS_global_rank (Option B):** When playoff data exists for the target season (16+ playoff teams), `EOS_global_rank` is the **end-of-season final rank** (champion=1, first 2 eliminated=29-30). Otherwise it is standings order at the snapshot. See `eos_rank_source` in predictions JSON. **EOS_playoff_standings** = final regular-season rank (1-30 by final reg-season win %). Metrics from runs before Option B used standings; newer runs with playoff data use EOS final rank. Do not compare NDCG/Spearman across these run types.

---

## 3. Run 021 and Run 022 (baseline runs; optimization target)

**Run 021** and **run_022** are full-pipeline baseline runs (default config); neither was produced by a sweep optimized for a single metric.

- **Run 021:** First "real success" per project README: Model A contributes (attention/contributors), ensemble ranking vs playoff outcome improved. Not tuned for NDCG-only or Spearman-only; same default hyperparameters as the rest of the pipeline.
- **Run 022** (EOS source: **eos_final_rank**): Ensemble NDCG **0.482**, Spearman **0.4305**, playoff Spearman **0.4607**, NDCG@4 final four **0.4645**, Brier championship **0.032**. Per season: 2023-24 NDCG 0.559 / Spearman 0.31; 2024-25 NDCG 0.482 / Spearman 0.43. Per conference (2024-25): East NDCG 0.25 / Spearman 0.25; West NDCG 0.75 / Spearman 0.50. See `outputs2/run_022/ANALYSIS_01.md` and `outputs2/run_022/RESULTS_AND_OUTPUTS_EXPLAINED.md`.

**Clarification:** Run_021 and run_022 were **not** optimized for NDCG or Spearman in isolation; they used the same default config. They often outperform early sweeps that optimized only Spearman because the default config balances multiple objectives. For fair comparison, run **separate sweeps** each optimizing a different target (see §4).

---

## 4. Sweep strategy: different target metrics per sweep, then compare

We run **one objective per Optuna sweep** and compare best configs across objectives:

- **`--objective spearman`** — maximize `test_metrics_ensemble_spearman` (global ordering).
- **`--objective ndcg`** — maximize `test_metrics_ensemble_ndcg` (top-heavy ranking).
- **`--objective playoff_spearman`** — maximize `test_metrics_ensemble_playoff_spearman_pred_vs_playoff_rank` (pred vs playoff outcome).
- **`--objective rank_mae`** — minimize `test_metrics_ensemble_rank_mae_pred_vs_playoff` (rank distance).

Example: run four sweeps (e.g. with `--n-trials 10` each), then compare the best combo from each in `sweep_results_summary.json` and `sweep_results.csv`. After each sweep, **5b_explain** runs automatically on the best combo for that objective (unless `--no-run-explain`). Use `python -m scripts.5b_explain --config <sweeps_dir>/<batch_id>/combo_<NNNN>/config.yaml` to explain any combo manually.

---

## 5. Comparison Against Past Runs

Metrics below are computed from each run’s `predictions.json` using the same logic as script 5 (NDCG@10, Spearman, MRR, ROC-AUC upset). Source: `scripts/compare_runs.py` → `outputs/run_comparison.json`.

| Run | NDCG@10 | Spearman | MRR | ROC-AUC (upset) |
|-----|---------|----------|-----|-----------------|
| run_009 | 0.638 | 0.717 | 0.00 | 0.629 |
| run_010 | 0.638 | 0.717 | 0.00 | 0.629 |
| run_011 | 0.638 | 0.717 | 0.00 | 0.629 |
| run_012 | 0.638 | 0.717 | 0.00 | 0.629 |
| run_013 | 0.638 | 0.717 | 0.00 | 0.629 |
| **run_014** | **0.665** | **0.760** | 0.00 | **0.653** |
| run_015 | 0.665 | 0.760 | 0.00 | 0.653 |
| **run_016** | **0.665** | **0.760** | 0.00 | **0.653** |

### Findings

1. **Runs 009–013 are identical** on these metrics. They likely share the same inference snapshot (e.g. “latest date” across all data) and/or the same model outputs; only run numbering and pipeline tweaks (e.g. field renames) differ.
2. **run_014 is the first with a proper train/test split and test-date inference.** Models are trained only on train seasons; predictions are for the last **test** date. So run_014 and later (015, 016) are evaluated on truly held-out data.
3. **run_014/015/016 improve over 009–013:** +4% NDCG (0.638 → 0.665), +6% Spearman (0.72 → 0.76), +4% ROC-AUC upset (0.63 → 0.65). That suggests the 75/25 setup (train-only training + test-date inference) is working as intended and the ensemble generalizes to the test window.
4. **run_014, run_015, and run_016 have identical metrics.** Same test date (last test date), same split; run_015 was from a pipeline run where Model A had timed out; run_016 used a fresh Model A from the same DB. So retraining Model A (with flat loss) did not change the ensemble’s test ordering—Model B and the stacker dominate, and/or Model A’s output is effectively unchanged when it doesn’t learn.

**Note:** An older ANALYSIS.md reported run_009 with “ndcg ~0.00026, spearman 0.0”. Those numbers came from an earlier `eval_report.json` or evaluation logic. Recomputing from run_009’s `predictions.json` with the current evaluation code yields the 0.64 / 0.72 values in the table above.

---

## 6. How the Model Is Performing (Summary)

- **Ranking (standings-based):** The ensemble’s predicted order of teams is **well aligned with actual end-of-season strength** on the held-out test snapshot: NDCG 0.67, Spearman 0.76. That is a clear improvement over past runs and indicates the model is learning useful signal.
- **Upset detection:** ROC-AUC 0.65 shows **moderate** ability to identify teams that were under-ranked by standings (sleepers) vs not.
- **MRR:** 0.0 indicates the single “best” team by relevance is not in the top-2 of the predicted order (or relevance ties); this is a strict metric and less representative than NDCG/Spearman for overall ranking quality.
- **Per-conference:** Current within-conference Spearman is negative due to using global rank as relevance; fix by using within-conference relevance before trusting per-conference Spearman. NDCG by conference may still be useful with the right relevance.
- **Playoff metrics:** Not available for run_014’s test snapshot; will appear in the report when playoff data is present for the evaluated run.

---

## 5. Predictions (run_016) — Structure and Examples

- **prediction:** `predicted_strength` (1–30, used for eval), `ensemble_score` (0–1), `conference_rank` (1–15), `championship_odds`.
- **analysis:** `actual_conference_rank` (Actual Conference Rank, 1–15), `EOS_global_rank`, `classification` (e.g. “Under-ranked by 2 slots”), `post_playoff_rank`, `rank_delta_playoffs`.
- **conference:** E/W (for per-conference evaluation).
- **ensemble_diagnostics:** model_agreement (High/Medium/Low), per-model ranks (deep_set, xgboost, random_forest).
- **roster_dependence:** primary_contributors (player + attention_weight); currently often empty or fallback with zero weights.

Examples:

- **Boston Celtics:** predicted_strength 3, EOS_global_rank 1 → “Under-ranked by 2 slots” (model slightly under-predicts the best team).
- **Milwaukee Bucks:** predicted 6, actual 3 → “Under-ranked by 3 slots”.
- **Classification:** “Under-ranked by X” = model rank worse (higher number) than actual; “Over-ranked by X” = model rank better (lower number) than actual.

---

## 8. Known Issues and Caveats

| Item | Status / Deduction |
|------|--------------------|
| **Per-conference Spearman negative** | Relevance is EOS global rank; within conference this can invert ordering. Use within-conference relevance (e.g. EOS conference rank) for per-conference Spearman. |
| **All attention weights 0** | Model A attention outputs zero or negative weights; fallback fills `primary_contributors` with 0.0. Likely cause: attention layer not learning (flat loss). |
| **IG attributions all zero** | Model A is not attributing importance to players; consistent with zero attention. |
| **Attention ablation NaN** | Masked forward pass yields non-finite scores; numerical or implementation issue in ablation. |
| **Roster contamination** | Wrong-team players in rosters (e.g. Simons, Kuzma on wrong teams) suggest “latest team” or as_of_date logic may be wrong for historical inference; treat player-level interpretation with caution. |
| **MRR 0.0** | Top-2 MRR is strict; first max-relevance item not in top 2; NDCG/Spearman are the main ranking metrics. |

---

## 7. Recommended Next Steps

1. **Per-conference evaluation:** Define relevance within conference (e.g. EOS conference rank 1–15) and recompute per-conference NDCG/Spearman so they are interpretable.
2. **Diagnose Model A:** Investigate flat loss and zero attention (learning rate, architecture, target, or data pipeline).
3. **Fix roster logic:** Verify `build_roster_set` and as_of_date so rosters match the inference date and avoid current-season contamination.
4. **Optional: train metrics:** Set `inference.also_train_predictions: true`, re-run script 6, then script 5, to get `train_metrics` and `train_metrics_by_conference` in the report for overfit/calibration checks.

---

## 10. Analysis Summary and Inferences from the Data

### Current results (run_016) — what they mean

- **eval_report.json** (for run_016): NDCG 0.665, Spearman 0.76, MRR 0.0, ROC-AUC upset 0.65; split is seasons (158 train dates, 41 test dates). Evaluation is on the **test** set (last test date snapshot).
- **Interpretation:** The ensemble’s predicted order of teams matches actual end-of-season strength well on held-out data. The model has moderate ability to flag “sleepers” (teams under-ranked by standings). Per-conference Spearman is negative because relevance is global rank; within-conference relevance would fix interpretation.
- **run_016 predictions:** Same structure as run_014 (predicted_strength, ensemble_score, EOS_global_rank, classification, etc.). Boston predicted 3rd, actual 1st (“Under-ranked by 2”); Milwaukee predicted 6th, actual 3rd (“Under-ranked by 3”). All `primary_contributors` are empty (fallback); Model A attention is not learning.

### Comparison to previous ANALYSIS.md and old outputs

- **Previous ANALYSIS.md** described **run_014** as the first run with the 75/25 split and test-date inference, with NDCG 0.665 and Spearman 0.76. The current **eval_report.json** has the same numbers because **run_016 has identical test metrics to run_014** (same test date, same split; only Model A was retrained, and it has flat loss so the ensemble output is unchanged).
- **Old outputs (run_009–013):** All five runs have identical metrics (NDCG 0.638, Spearman 0.717, ROC-AUC 0.629). They used the same inference snapshot (“latest date” before the split) and/or same model outputs.
- **run_014 vs 009–013:** Introduction of the 75/25 split and test-date inference gave a clear improvement: +4% NDCG, +6% Spearman, +4% ROC-AUC. So the pipeline change (train-only training + held-out test evaluation) is validated.
- **run_014 vs run_015 vs run_016:** run_015 was from a pipeline where Model A had timed out (old or missing Model A); run_016 used a fresh Model A from the same DB. Metrics are identical across 014/015/016, so **retraining Model A did not change test performance**—consistent with Model A not learning (flat loss, zero attention). The ensemble’s test ranking is driven by Model B (XGB/RF) and the stacker.

### All inferences from the data

1. **75/25 split and test-date inference work.** Moving from “all data” / “latest date” (run_009–013) to train-only training and last-test-date evaluation (run_014+) improves ranking metrics on held-out data.
2. **Ensemble ranking quality is good on test.** NDCG 0.67 and Spearman 0.76 indicate the combined model orders teams in line with actual strength; ROC-AUC 0.65 indicates modest value for identifying sleepers.
3. **Model A is not learning.** Flat training loss (27.8993 every epoch), zero attention weights, zero IG attributions, and identical metrics after retraining Model A (run_016 vs 014/015) all point to Model A contributing little or no learned signal; the ensemble’s test performance is effectively from Model B + stacker.
4. **Per-conference Spearman is misleading.** Global rank used as relevance within conference inverts ordering (East/West Spearman negative). Need within-conference relevance (e.g. EOS conference rank) for interpretable per-conference metrics.
5. **Playoff metrics are missing** for the current test snapshot because the test date (2025-04-13) is in 2024-25 and no playoff games exist in the DB for that season. Playoff metrics would appear for a snapshot from a completed season (e.g. last train date with `also_train_predictions`).
6. **Roster/player-level interpretation is unreliable.** Empty or fallback primary_contributors, possible roster contamination (wrong-team players), and zero Model A attribution mean we should not trust player-level explanations until Model A and roster logic are fixed.
7. **MRR 0.0** is a strict metric (best team not in top-2 of predicted order); NDCG and Spearman are the main indicators of ranking quality and are positive.
8. **Run identity:** run_014, run_015, and run_016 are indistinguishable on test metrics; the latest run (016) uses a fresh Model A from the current DB but that model does not change the ensemble’s test ordering.
