# NBA True Strength Prediction — Checkpoint Project Report

**Tanaka Makuvaza**  
Georgia State University — Advanced Machine Learning  
February 2026

---

## Abstract

This project builds a multi-modal stacking ensemble to predict NBA team strength and playoff-relevant outcomes. The system combines a roster-aware Deep Set model (Model A) with a hybrid tabular ensemble of XGBoost and Random Forest (Model B), blended via K-fold out-of-sample (OOF) predictions and a RidgeCV meta-learner. The goal is to predict true team strength and identify *sleepers* (teams that outperform their predicted rank) versus *paper tigers* (teams that underperform), without circular evaluation or net-rating leakage. Evaluation evolved from regular-season standings to **playoff outcome** (end-of-season final rank). **Run 21 is the first real success:** Model A contributes (non-zero attention and primary contributors), and ensemble ranking versus playoff outcome improved (NDCG, Spearman, MRR) over run_020. From here, hyperparameter sweeps and future test runs write to `outputs3/` (sweeps → `outputs3/sweeps/<batch_id>/`). This report summarizes implementation findings, accuracy progress, metric development, methodology, key decisions, and issues encountered.

---

## 1. Introduction and Goals

### 1.1 Project aim

- **True team strength:** Predict team strength for future outcomes (e.g. playoff seed, championship odds), not merely historical efficiency.
- **Sleepers vs paper tigers:** Identify teams that will outperform or underperform their predicted rank (sleeper detection via ROC-AUC on upsets).
- **No circular evaluation:** Strict train/test split (75/25 by season); inference on last test date only; no test-date leakage.

### 1.2 Key design choices

- **Target:** Future W/L (next 5) or final playoff seed — never raw efficiency metrics as the primary target.
- **No net-rating leakage:** `net_rating` is excluded from Model B features and from targets; enforced in `src.features.team_context.FORBIDDEN` and training code. Allowed only in baselines (e.g. rank-by-Net-Rating).
- **ListMLE for Model A:** Listwise ranking loss over conference-date lists; teams in a list are ranked by strength; ListMLE encourages correct ordering of teams within each list. Model A ListMLE target = **playoff outcome** (`eos_final_rank`, champion=1) when `listmle_target: playoff_outcome`. Fallback to standings for seasons without playoff data. Previously `final_rank` (EOS standings); now `playoff_outcome`.
- **RidgeCV for stacking:** Level-2 meta-learner is RidgeCV on pooled OOF (not Logistic Regression) for stability and to avoid overfitting when blending Model A, XGB, and RF ranks.
- **Evaluation target evolution:** Early runs (009–017) used **regular-season standings** (or snapshot order) as ground truth. Later runs (020, 021) use **playoff outcome** (`eos_final_rank`: champion=1, runner-up=2, … first two eliminated=29–30). Metrics are **not comparable** across these two target types.

---

## 2. Methodology

### 2.1 Data and storage

- **Sources:** nba_api (official) for regular-season and playoff game/player logs; Kaggle (e.g. Wyatt Walsh) for SOS/SRS where used; Basketball-Reference as fallback.
- **Storage:** DuckDB with regular-season tables (`games`, `team_game_logs`, `player_game_logs`, `teams`, `players`) and playoff tables (`playoff_games`, `playoff_team_game_logs`, `playoff_player_game_logs`). Raw files (Parquet/CSV) hashed in `data/manifest.json`; DB rebuild is skipped when raw hashes are unchanged.
- **Build pipeline:** Scripts `1_download_raw` and `2_build_db`; `2_build_db` compares current raw hashes to manifest and rebuilds only when raw files change.

### 2.2 Train/test split

- **75/25 by season:** Train seasons 2015–16 through 2022–23 (e.g. 141 train dates); test seasons 2023–24 and 2024–25 (e.g. 36 test dates). Implemented in `src.utils.split`; split info written by script 3 and consumed by scripts 4, 5, 6.
- **Inference:** Predictions use the **last test date** (held-out snapshot) so evaluation is on true out-of-sample data. Per-season prediction files (e.g. `predictions_2023-24.json`, `predictions_2024-25.json`) support per-season evaluation.

### 2.3 Model A (Deep Set + ListMLE)

- **Architecture:** Deep Set over roster: player-level stats (e.g. L10/L30 rolling, on-court plus-minus approximation) encoded per player, then aggregated with set attention to produce a team strength score. See `src.models.deep_set_rank`, `src.models.set_attention`, `src.training.train_model_a`.
- **Training:** ListMLE loss over lists (conference-date or similar); each list is a set of teams with a ranking target (e.g. win rate or EOS playoff standings). Gradient clipping, input clamping, and `torch.logsumexp` for numerical stability. Hash-trick embeddings for unseen players. Epochs and early stopping configurable in `config/defaults.yaml`.
- **Output:** Per-team score used as a rank input to the stacker; also attention weights for explainability (primary_contributors).

### 2.4 Model B (XGBoost + Random Forest)

- **Features:** Team-context features only (no net_rating): rolling win rates, point differentials, SOS/SRS where enabled, optional Elo, team rolling, motivation, injury adjustment. Built in `src.features.team_context`; forbidden list enforced in training.
- **Training:** K-fold OOF; Model B (XGBoost) and Model C (RF) trained separately; OOF predictions passed to the stacker. See `src.models.xgb_model`, `src.models.rf_model`, `src.training.train_model_b`.

### 2.5 Stacking and inference

- **Stacking:** K-fold OOF from Model A and Model B (XGB, RF) are pooled; RidgeCV fits level-2 weights on OOF ranks/scores. See `src.models.stacking`, script `4b_train_stacking`.
- **Inference:** Script 6 loads DB, Model A checkpoint, Model B artifacts, and meta-learner; computes ensemble score and predicted strength (rank 1–30); writes `predictions.json`, per-season prediction files, and plots (pred_vs_actual, pred_vs_playoff_final_results, eos_playoff_standings_vs_eos_global_rank, title_contender_scatter, etc.).

### 2.6 Anti-leakage rules

- **Time rule:** All features use only rows with `game_date < as_of_date` (strict t-1). Rolling stats use shift(1) before aggregation.
- **Roster:** Players assigned to roster by latest team as of `as_of_date`; traded players appear only on their current team at that date.
- **Model B:** Feature set must not include `net_rating`; enforced in code.
- **Baselines only:** Net rating used only in rank-by-Net-Rating baseline, computed from offensive/defensive ratings, never as a model input.

---

## 3. Metrics Development

### 3.1 Ranking metrics

- **NDCG@10:** Normalized Discounted Cumulative Gain at 10. Relevance = higher is better (e.g. 31 − rank for 30 teams). Measures how well the predicted **order** of teams matches the ground-truth order. Implemented in `src.evaluation.metrics.ndcg_score` with `k=10`.
- **NDCG@4 (final four):** Same as NDCG but truncated at top 4; relevance = 30 − rank + 1. Used for “final four” playoff evaluation. See `ndcg_at_4` in `src.evaluation.metrics`.
- **Spearman:** Correlation between predicted scores and ground-truth relevance. Measures monotonic alignment of predicted strength with actual rank. See `spearman` in `src.evaluation.metrics`.

### 3.2 MRR (Mean Reciprocal Rank)

- **mrr_top2:** 1 / (rank of first team in top 2 by actual rank in predicted order). Top 2 = champion + runner-up. Strict: 0.0 if champion/runner-up not in predicted top-2.
- **mrr_top4:** Same idea for top 4 (conference finals). Implemented in `mrr(..., top_n_teams=2)` and `top_n_teams=4`.

### 3.3 Sleeper detection

- **ROC-AUC upset:** Binary label “upset” = team’s actual rank better (lower number) than predicted rank (sleeper). Continuous score = ensemble score (or derived). ROC-AUC measures ability to separate sleepers from non-sleepers. Returns 0.5 if constant labels. See `roc_auc_upset` in `src.evaluation.metrics`.

### 3.4 Playoff-specific metrics

- **Spearman (predicted vs playoff rank):** Correlation between predicted global rank and end-of-season playoff performance rank (champion=1, …, 30).
- **NDCG@10 (ndcg10):** Same as main ndcg; explicit key for clarity. In playoff_metrics: `ndcg10_pred_vs_playoff` (NDCG@10 with playoff rank as relevance).
- **Brier (championship odds):** One-hot champion vs predicted championship probabilities. See `brier_champion` in `src.evaluation.metrics`.
- **rank_mae, rank_rmse:** Rank-distance metrics (pred vs actual playoff rank; lower = better). Used for model and standings baselines (`pred_vs_playoff`, `standings_vs_playoff`). See `rank_mae`, `rank_rmse` in `src.evaluation.metrics`.

### 3.5 Per-conference caveat

- When relevance is **global** rank (1–30), computing Spearman **within one conference** can yield **negative** Spearman (e.g. East, West) because within a conference only a slice of global ranks appears; “better” teams in that conference can have worse (higher) global rank numbers. Run_020 showed negative per-conference Spearman; run_021 shows positive per-conference (E: 0.72, W: 0.68) after improvements. For fair per-conference evaluation, relevance should be defined within conference (e.g. EOS conference rank 1–15).

### 3.6 Ground truth (eos_rank_source)

- **standings:** Ground truth = regular-season standings or snapshot order at inference date.
- **eos_final_rank:** Ground truth = playoff outcome (champion=1, …, 30). Used when playoff data exists for the target season (≥16 teams with playoff data).
- **Do not compare NDCG/Spearman across run types:** Runs evaluated on standings vs runs evaluated on playoff outcome use different ground truth; metric values are not directly comparable. See `outputs2/ANALYSIS.md` Section 4.

---

## 4. Implementation Findings and Progress

Chronological narrative from commit history and plans:

1. **Scaffold and config (97f1e01):** Project layout, `config/defaults.yaml`, Plan.md.
2. **nba_api ingestion and DuckDB build (2821589):** Raw logs → DuckDB via `src.data.db_loader`; games built from team MATCHUP (e.g. @ = away, vs = home).
3. **Leakage-safe features and list construction (225d5cd):** Feature engineering and list building for Model A (conference-date lists).
4. **Model A + ListMLE (447d7a1):** Deep Set and ListMLE training loop.
5. **Model B without net_rating (85b8a3a):** XGB and RF on team-context features; net_rating forbidden.
6. **OOF stacking + RidgeCV (050bc78):** K-fold OOF, RidgeCV meta-learner.
7. **Inference and eval (62b6fd6, 5c34006):** Inference pipeline, evaluation metrics (NDCG, Spearman, MRR, ROC-AUC upset).
8. **SHAP and attention/IG (c84effa):** Explainability (script 5b): SHAP for Model B; Integrated Gradients and attention for Model A.
9. **Playoff-aware pipeline (0599d40):** Playoff tables, championship odds, conference rank output; skip-if-exists DB flag.
10. **Auto-increment run_id (5650835):** `inference.run_id: null` → next run folder; eval uses latest run.
11. **75/25 split and test-date inference (7e978e9, 5650835):** Season-based split; inference on last test date only.
12. **Update2 (bb78048):** IG batching (Captum auxiliary tensors for batched inputs); latest-team roster; EOS_global_rank; manifest db_path relative; ensemble agreement High/Medium/Low; attention/IG sanitized for valid JSON.
13. **Fix run_009 outputs (9c32b7f):** Playoff rank capping, NaN handling, classification labels, ensemble agreement.
14. **EOS final rank Option B (260cf61, 8dd8df3):** When playoff data exists (≥16 teams), ground truth = eos_final_rank (playoff outcome); walk-forward option; run_id reservation; stacking NaN fix; inference pts unpack fix; outputs2, run_019, ANALYSIS.md.
15. **Run_021 (16c00c5):** Full pipeline run; Model A contributing (attention, primary_contributors); stronger XGB; improved NDCG/Spearman/MRR vs playoff outcome; outputs3 for sweeps and future runs.

---

## 5. Accuracy Progress

### 5.1 Standings target (outputs/)

| Run(s)        | NDCG@10 | Spearman | MRR (top-2) | ROC-AUC upset |
|---------------|---------|----------|-------------|---------------|
| run_009–013   | 0.638   | 0.717    | 0.00        | 0.629         |
| run_014–016   | **0.665** | **0.760** | 0.00      | **0.653**     |

- Runs 009–013 used same inference snapshot (“latest date” or pre–75/25). Runs 014–016 use 75/25 split and last test date; metrics improve. Run_014/015/016 are identical (Model A had flat loss so retraining did not change ensemble ordering).

### 5.2 Playoff target (outputs2/)

| Season  | Run   | NDCG@10 | Spearman | MRR (top-2) | ROC-AUC upset | Playoff Spearman | NDCG@4 |
|---------|-------|---------|----------|-------------|---------------|------------------|--------|
| 2023-24 | 020   | 0.290   | 0.161    | 0.0         | 0.79          | 0.09             | 0.001  |
| 2023-24 | **021** | **0.417** | **0.306** | **0.25**   | 0.77          | **0.31**         | **0.289** |
| 2024-25 | 020   | 0.065   | 0.374    | 0.0         | 0.72          | 0.27             | 0.027  |
| 2024-25 | **021** | **0.167** | **0.571** | **0.14**   | 0.76          | **0.50**         | 0.024  |

- Run_021 improves ranking (NDCG, Spearman, playoff Spearman, MRR) on both test seasons. Model A contributes in 021 (non-zero attention, named primary_contributors). XGB in run_021 is much stronger on 2024–25 (e.g. Spearman 0.63 vs 0.34 in run_020). Per-conference Spearman is positive in 021 (E: 0.72, W: 0.68).

### 5.3 Interpretation

- **Drop from run_017 to run_020:** Largely due to **switching target** from standings to playoff outcome, not a worse model. Playoff order is noisier and harder to predict.
- **Run_021 vs run_020:** Same target (playoff outcome); run_021 is strictly better on ranking and MRR, with Model A now contributing and XGB stronger.

---

## 6. Key Decisions and Rationale

- **ListMLE:** Listwise ranking over conference-date lists captures relative strength within meaningful subsets; pairwise or pointwise losses would not model list structure as directly.
- **RidgeCV for stacking:** Stable blending of three rankers (Model A, XGB, RF); avoids overfitting that can occur with more flexible meta-models on limited OOF samples.
- **Exclude net_rating:** Prevents leakage of information that is highly correlated with outcomes we are trying to predict; keeps the task focused on strength derived from other signals.
- **Playoff outcome as target:** Aligns with the real-world question “who will go furthest in the playoffs?” rather than “who had the best regular-season record?”
- **75/25 by season:** Time-aware holdout; prevents future information from leaking into training.
- **Walk-forward option:** Allows sequential validation (train on seasons 1..k, validate on k+1) for tuning and diagnostics.
- **Run_id reservation and conditional DB rebuild:** Ensures a full pipeline run writes to a single run folder; avoids redundant DB rebuilds when raw data are unchanged, speeding iteration.

---

## 7. Issues Encountered and Resolutions

| Issue | Resolution / status |
|-------|----------------------|
| **Attention all-zero / Model A not contributing (run_020)** | Run_020 had empty primary_contributors and all-zero attention (fallback). Run_021: full pipeline retrain with aligned config/checkpoint; Model A now contributes (attention and named players). |
| **IG batching / Captum** | Update2: Captum auxiliary tensors expanded to match batched inputs so IG runs correctly in inference. |
| **Per-conference Spearman negative** | Documented: global rank as relevance within one conference inverts ordering. Run_021 shows positive per-conference after overall ranking improved; within-conference relevance (e.g. EOS conference rank) recommended for formal per-conference metrics. |
| **eos_rank_source and playoff metrics** | When playoff data exist for target season (≥16 teams), ground truth = eos_final_rank. Playoff_metrics (Spearman vs playoff rank, NDCG@4, Brier) only when playoff data present. Explained in outputs/ANALYSIS.md. |
| **Missing argparse** | Scripts `6_run_inference.py` and `5_evaluate.py` were missing `import argparse`; added during pipeline run. |
| **Config/checkpoint mismatch (5b_explain)** | Model A checkpoint from one run had different input size (e.g. stat_dim 49 vs 39) than current config; attention ablation in 5b failed to load. Documented; use matching config when loading checkpoint. |
| **Stacking NaN fix, inference pts unpack** | Addressed in commit 8dd8df3 (stacking NaN fix, inference pts unpack fix). |
| **Data loaders** | In-process cache by (db_path, mtime) added for `load_training_data` and `load_playoff_data` so the same DB is not reloaded within one process. |

---

## 8. Conclusion and Next Steps

**Run 21** is the first run where Model A contributes meaningfully (attention and primary_contributors) and ensemble ranking versus playoff outcome clearly improves (NDCG, Spearman, MRR). From here:

- **Hyperparameter sweeps:** Run `sweep_hparams`; results write to `outputs3/sweeps/<batch_id>/`. Grid over Model A epochs, Model B hyperparameters, rolling windows, etc.
- **Optional features:** Elo, team rolling, motivation, injury (e.g. nbainjuries), Monte Carlo championship odds, SOS/SRS (e.g. Team_Records.csv). See `.cursor/plans/enable_optional_features_7b94a57e.plan.md`.
- **Calibration and playoff residual:** Ideas in `.cursor/plans/comprehensive_feature_and_evaluation_expansion.plan.md` (ECE, Platt scaling, playoff residual model, NBA-specific XGBoost tuning).

---

## References and Repo Artifacts

- **Repository:** [NBA Playoff Strength Project](https://github.com/Tmaku18/NBA-Playoff-Strentgh-Project) (or NBA-Playoff-Strength-Project).
- **Key artifacts:**  
  - `README.md` — design choices, pipeline order, anti-leakage, outputs.  
  - `outputs2/ANALYSIS.md` — run_020/021 comparison, metrics interpretation, known issues.  
  - `outputs/ANALYSIS.md` — run_009–016 metrics, 75/25 split, playoff data caveat.  
  - `config/defaults.yaml` — paths, seasons, model_a/model_b/training/sweep config.  
  - `scripts/` — 1_download_raw, 2_build_db, 3_train_model_a, 4_train_models_b_and_c, 4b_train_stacking, 5_evaluate, 5b_explain, 6_run_inference, run_pipeline_from_model_a, sweep_hparams.  
  - `src/` — data, evaluation, features, inference, models, training, utils, viz.

---

*End of Checkpoint Project Report*
