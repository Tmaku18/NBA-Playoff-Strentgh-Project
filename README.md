# NBA "True Strength" Prediction

**Attention-Based Deep Set Network with Ensemble Validation**

Tanaka Makuvaza  
Georgia State University — Advanced Machine Learning  
January 28, 2026

---

## Overview
This project builds a **Multi-Modal Stacking Ensemble** to predict NBA **True Team Strength** using a Deep Set roster model plus a Hybrid tabular ensemble (XGBoost + Random Forest). The system targets **future outcomes** and identifies **Sleepers** versus **Paper Tigers** without circular evaluation.

---

## Key Design Choices
- **Target:** Future W/L (next 5) or Final Playoff Seed — **never** efficiency.
- **True Strength:** Model A produces a latent **Z** (penultimate layer); the **output** `ensemble_score` is the **ensemble** score (RidgeCV blend of A + XGB + RF) mapped to percentile (0–1 and 0–100).
- **No Net Rating leakage:** `net_rating` is excluded as a model input and never used as a target or evaluation metric (allowed only in baselines).
- **Stacking:** K-fold **OOF** across **all training seasons**; level-2 **RidgeCV** on pooled OOF (not Logistic Regression).
- **Game-level ListMLE:** lists per conference-date/week; **torch.logsumexp** and input clamping for numerical stability; gradient clipping in Model A training; hash-trick embeddings for new players.
- **Season config:** Hard-coded season date ranges in `defaults.yaml` to avoid play-in ambiguity.
- **Explainability:** SHAP on Model B only; Integrated Gradients or permutation ablation for Model A.

---

## Data Sources
- **nba_api** (official): play-by-play, player/team logs, tracking data; **playoff** game logs (SeasonType=Playoffs) for validation. Play-In games are excluded from playoff win counts.
- **Kaggle (Wyatt Walsh):** **primary** for SOS/SRS and historical validation.
- **Basketball-Reference:** **fallback** for SOS/SRS when Kaggle unavailable.
- **Proxy SOS:** If both are unavailable, compute from internal DB (e.g. opponent win-rate) and document.

**Storage:** DuckDB preferred (regular-season + separate playoff tables: `playoff_games`, `playoff_team_game_logs`, `playoff_player_game_logs`).

---

## Playoff performance rank (ground truth)
Used for training (optional) and evaluation when playoff data exists. **Phase 1:** Rank playoff teams by total playoff wins (desc). **Phase 2:** Tie-break by regular-season win %. **Phase 3:** Teams with 0 playoff wins are ranked 17–30 by regular-season win %. Config: `training.target_rank: standings | playoffs` (default `standings`). Playoff rank is computed from `playoff_team_game_logs` + `playoff_games`; it will be null if playoff data is missing or season mapping fails.

## Evaluation
- **Ranking:** NDCG, Spearman, MRR (MRR uses top_k=2 for two-conference “rank 1”).
- **Future outcomes:** Brier score.
- **Sleeper detection:** ROC-AUC on upsets (sleeper = actual rank worse than predicted rank); constant-label guard returns 0.5.
- **Playoff metrics** (when playoff data and predictions include playoff_rank): Spearman (predicted global rank vs playoff performance rank), NDCG@4 (final four), Brier score on championship odds (one-hot champion vs predicted odds). Section `playoff_metrics` in `eval_report.json`.
- **Report:** `eval_report.json` includes `notes` and, when applicable, `playoff_metrics`.
- **Baselines:** rank-by-SRS, rank-by-Net-Rating, **Dummy** (e.g. previous-season rank or rank-by-net-rating).

---

## Outputs (per run)
- **Global rank** (1–30, league-wide), **conference rank** (1–15 within East/West), **ensemble score** (0–1 and 0–100; derived from the stacked ensemble, not Model A’s Z alone), **championship odds** (softmax with `output.odds_temperature`).
- **EOS_conference_rank** in outputs and plots is conference standing (1–15 within East or West), from standings-to-date at the inference target date. When available, `EOS_global_rank` (1–30) is included for global evaluation/classification.
- **Playoff rank** and **rank_delta_playoffs** (when playoff data exists for the target season).
- Classification: **Over-ranked**, **Under-ranked**, **Aligned**.
- Delta (actual conference rank − predicted league rank) and ensemble agreement (Model A / XGB / RF ranks).
- Roster dependence (attention weights; IG contributors when enabled via `output.ig_inference_top_k` and Captum).
- **Plots:** `pred_vs_actual.png` — two panels (East/West), conference rank vs actual conference rank (1–15); `pred_vs_playoff_rank.png` — global rank vs playoff performance rank (1–30); `title_contender_scatter.png` — championship odds vs regular-season wins; `odds_top10.png` — top-10 championship odds bar chart.
- **Report assets:** `outputs/ANALYSIS.md` — human-readable analysis of pipeline outputs, interpretations, and known issues.

---

## How to Run the Pipeline

**Run order (production, real data only):**

1. **Setup:** `pip install -r requirements.txt`
2. **Config:** Edit `config/defaults.yaml` if needed (seasons, paths, model params, `build_db.skip_if_exists`, `inference.run_id`). DB path: `data/processed/nba_build_run.duckdb`. Set `inference.run_id: null` (default) to auto-increment runs (run_002, run_003, …); set e.g. `"run_001"` to fix a single run folder.
3. **Data:**  
   - `python -m scripts.1_download_raw` — fetch regular-season and playoff logs via nba_api (writes to `data/raw/`; reuses existing files when present).  
   - `python -m scripts.2_build_db` — build DuckDB from raw → `data/processed/nba_build_run.duckdb`, update `data/manifest.json`. If `build_db.skip_if_exists: true` (default) and the DB file already exists, the build is skipped to keep the current DB.
4. **Training (real DB):**  
   - `python -m scripts.3_train_model_a` — K-fold OOF → `outputs/oof_model_a.parquet`, then final model → `outputs/best_deep_set.pt`.  
   - `python -m scripts.4_train_model_b` — K-fold OOF → `outputs/oof_model_b.parquet`, then XGB + RF → `outputs/xgb_model.joblib`, `outputs/rf_model.joblib`.  
   - `python -m scripts.4b_train_stacking` — merge OOF parquets, RidgeCV → `outputs/ridgecv_meta.joblib`, `outputs/oof_pooled.parquet` (requires OOF from 3 and 4).
5. **Inference:** `python -m scripts.6_run_inference` — load DB and models, run Model A/B + meta → `outputs/<run_id>/predictions.json`, plots. With `inference.run_id: null`, run_id auto-increments (run_002, run_003, …) so each full pipeline run gets a new folder.
6. **Evaluation:** `python -m scripts.5_evaluate` — uses predictions from the latest (or configured) run_id → `outputs/eval_report.json` (NDCG, Spearman, MRR, ROC-AUC upset).
7. **Explainability:** `python -m scripts.5b_explain` — SHAP on Model B (team-context X) → `outputs/shap_summary.png`; attention ablation and Integrated Gradients (Model A) when Captum is installed → `outputs/ig_model_a_attributions.txt`. Attention ablation skips padded roster slots and reports clearly when the masked forward yields NaN.

**Optional:** `python -m scripts.run_manifest` (run manifest); `python -m scripts.run_leakage_tests` (before training).

**Pipeline behavior:** Script 1 reuses raw files that already exist (no re-download). Script 2 skips rebuilding the DB when `build_db.skip_if_exists` is true and the DB file exists; set it to false to force a full rebuild from raw. With `inference.run_id: null`, inference writes to the next run folder (run_002, run_003, …) and evaluation uses the latest run.

**Training notes:** Model A (script 3) subsamples conference-date lists for OOF and final training (`training.max_lists_oof`, `training.max_final_batches`) and `build_lists` subsamples dates (e.g. 200) for speed; use full list set by increasing these in config or adjusting `build_lists`.

---

## Anti-Leakage Checklist

- [ ] **Time rule:** All features use only rows with `game_date < as_of_date` (strict t-1). Rolling stats use `shift(1)` before aggregation.
- [ ] **Roster:** Minutes and roster selection use only games before `as_of_date`. Rosters use a **latest-team** map (player’s most recent team as of `as_of_date`) so traded players appear only on their current team; season boundaries from config scope games when building rosters.
- [ ] **Model B:** Feature set must **not** include `net_rating`. Enforced in `src.features.team_context.FORBIDDEN` and `train_model_b`.
- [ ] **ListMLE:** Targets are standings-to-date (win-rate), not season-end. Evaluation remains season-end.
- [ ] **Baselines only:** Net Rating is used only in `rank-by-Net-Rating` baseline, computed from off/def ratings, never as a model input.

---

## Report Assets (deliverables)

All paths under `outputs/` (or `config.paths.outputs`). Produced from real data when DB and models exist. With `inference.run_id: null`, each pipeline run writes to a new folder (`outputs/run_002/`, `outputs/run_003/`, …); evaluation uses the latest run.

- `outputs/eval_report.json` — NDCG, Spearman, MRR (top_k=2), ROC-AUC upset, `notes`; when playoff data exists, `playoff_metrics` (Spearman vs playoff rank, NDCG@4, Brier championship).
- `outputs/run_001/predictions.json` — per-team `predicted_strength` (rank), `global_rank` (1–30), `conference_rank` (1–15), `championship_odds`, `ensemble_score` (0–1 percentile), `EOS_conference_rank`, `EOS_global_rank` (1–30 when available), `playoff_rank`/`rank_delta_playoffs` (when playoff data exists), classification, ensemble diagnostics (model_agreement: High/Medium/Low), roster_dependence (attention + optional `ig_contributors`).
- `outputs/run_001/pred_vs_actual.png` — two panels (East/West): predicted vs actual conference rank (1–15); grid lines, team-colored points, legend.
- `outputs/run_001/pred_vs_playoff_rank.png` — predicted global rank (1–30) vs playoff performance rank (1–30).
- `outputs/run_001/title_contender_scatter.png` — championship odds vs regular-season wins (proxy).
- `outputs/run_001/odds_top10.png` — top-10 championship odds bar chart.
- `outputs/shap_summary.png` — Model B (RF) SHAP summary on real team-context features (script 5b).
- `outputs/ig_model_a_attributions.txt` — Model A Integrated Gradients top-5 player indices by attribution L2 norm (script 5b; requires Captum).
- `outputs/oof_pooled.parquet`, `outputs/ridgecv_meta.joblib` — stacking meta-learner and pooled OOF (script 4b).
- `outputs/oof_model_a.parquet`, `outputs/oof_model_b.parquet` — OOF from scripts 3 and 4 (Option A: K-fold, real data).
- `outputs/best_deep_set.pt`, `outputs/xgb_model.joblib`, `outputs/rf_model.joblib` — trained Model A and Model B.

---

## Reproducibility

- **Seeds:** `src.utils.repro.set_seeds(seed)` for random, numpy, torch.
- **Manifests:** `outputs/run_manifest.json` (config snapshot, git hash, data manifest hash). `data/manifest.json` (raw/processed hashes; `db_path` is stored relative to project root for portability).
- **OOF:** `outputs/oof_pooled.parquet` and `ridgecv_meta.joblib`.
- **Season boundaries:** Hard-coded in `config/defaults.yaml` to avoid play-in ambiguity.

---

## Recent implementation (Update2)

Implemented per [.cursor/plans/Update2.md](.cursor/plans/Update2.md): **IG batching fix** (Captum auxiliary tensors expanded to match batched inputs); **latest-team roster** (players only on current team as of `as_of_date`); **EOS_global_rank** (1–30) for evaluation/classification; **manifest db_path** stored relative to project root; **conference plot** uses only valid conference ranks (no global fallback); **attention/IG** sanitized so `predictions.json` is valid JSON (`allow_nan=False`); **ensemble agreement** High/Medium/Low with scaled thresholds and handling of missing models; **ensemble_score** percentile reaches 0.0/1.0. Optional: `output.ig_inference_top_k` and `output.ig_inference_steps` control IG in inference outputs.

---

## Full Plan

See `.cursor/plans/Plan.md`. Planned extensions: [.cursor/plans/Update1.md](.cursor/plans/Update1.md), [.cursor/plans/Update2.md](.cursor/plans/Update2.md).

---

## Implementation Roadmap
The full phased development plan and file-by-file checklist are in
`.cursor/plans/Plan.md` under **Development and Implementation Plan**.
Update1 roadmap: `.cursor/plans/Update1.md`. Update2 roadmap: `.cursor/plans/Update2.md`.
