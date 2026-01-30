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
- **True Strength:** Latent **Z** from Deep Set penultimate layer; score mapped to percentile within conference.
- **No Net Rating leakage:** `net_rating` is excluded as a model input and never used as a target or evaluation metric (allowed only in baselines).
- **Stacking:** K-fold **OOF** across **all training seasons**; level-2 **RidgeCV** on pooled OOF (not Logistic Regression).
- **Game-level ListMLE:** lists per conference-date/week; **torch.logsumexp** and input clamping for numerical stability; gradient clipping in Model A training; hash-trick embeddings for new players.
- **Season config:** Hard-coded season date ranges in `defaults.yaml` to avoid play-in ambiguity.
- **Explainability:** SHAP on Model B only; Integrated Gradients or permutation ablation for Model A.

---

## Data Sources
- **nba_api** (official): play-by-play, player/team logs, tracking data.
- **Kaggle (Wyatt Walsh):** **primary** for SOS/SRS and historical validation.
- **Basketball-Reference:** **fallback** for SOS/SRS when Kaggle unavailable.
- **Proxy SOS:** If both are unavailable, compute from internal DB (e.g. opponent win-rate) and document.

**Storage:** DuckDB preferred; SQLite allowed with pre-aggregation + indexing.

---

## Evaluation
- **Ranking:** NDCG, Spearman, MRR (MRR uses top_k=2 for two-conference “rank 1”).
- **Future outcomes:** Brier score.
- **Sleeper detection:** ROC-AUC on upsets (sleeper = actual conference rank > predicted league rank); constant-label guard returns 0.5.
- **Report:** `eval_report.json` includes a `notes` field (upset definition, MRR description).
- **Baselines:** rank-by-SRS, rank-by-Net-Rating, **Dummy** (e.g. previous-season rank or rank-by-net-rating).

---

## Outputs (per run)
- **Predicted rank** (1–30, league-wide) and true strength score (0–1).
- **Actual rank** in outputs and plots is conference standing (1–15 within East or West), from standings-to-date at the inference target date.
- Classification: **Sleeper** (under-ranked by standings), **Paper Tiger** (over-ranked), **Aligned**.
- Delta (actual conference rank − predicted league rank) and ensemble agreement (Model A / XGB / RF ranks).
- Roster dependence (attention weights when available).
- `pred_vs_actual.png`: two panels (East and West); x-axis = actual conference rank (1–15), y-axis = predicted league rank (1–30); grid lines, team-colored points, and legend.

---

## How to Run the Pipeline

**Run order (production, real data only):**

1. **Setup:** `pip install -r requirements.txt`
2. **Config:** Edit `config/defaults.yaml` if needed (seasons, paths, model params). DB path: `data/processed/nba_build_run.duckdb`.
3. **Data:**  
   - `python -m scripts.1_download_raw` — fetch player/team logs via nba_api (writes to `data/raw/` as parquet).  
   - `python -m scripts.2_build_db` — build DuckDB from raw → `data/processed/nba_build_run.duckdb`, update `data/manifest.json`.
4. **Training (real DB):**  
   - `python -m scripts.3_train_model_a` — K-fold OOF → `outputs/oof_model_a.parquet`, then final model → `outputs/best_deep_set.pt`.  
   - `python -m scripts.4_train_model_b` — K-fold OOF → `outputs/oof_model_b.parquet`, then XGB + RF → `outputs/xgb_model.joblib`, `outputs/rf_model.joblib`.  
   - `python -m scripts.4b_train_stacking` — merge OOF parquets, RidgeCV → `outputs/ridgecv_meta.joblib`, `outputs/oof_pooled.parquet` (requires OOF from 3 and 4).
5. **Inference:** `python -m scripts.6_run_inference` — load DB and models, run Model A/B + meta → `outputs/run_001/predictions.json`, `outputs/run_001/pred_vs_actual.png`.
6. **Evaluation:** `python -m scripts.5_evaluate` — uses predictions from step 6 (inference) → `outputs/eval_report.json` (NDCG, Spearman, MRR, ROC-AUC upset).
7. **Explainability:** `python -m scripts.5b_explain` — SHAP on real team-context X, attention ablation on real list batch → `outputs/shap_summary.png`.

**Optional:** `python -m scripts.run_manifest` (run manifest); `python -m scripts.run_leakage_tests` (before training).

---

## Anti-Leakage Checklist

- [ ] **Time rule:** All features use only rows with `game_date < as_of_date` (strict t-1). Rolling stats use `shift(1)` before aggregation.
- [ ] **Roster:** Minutes and roster selection use only games before `as_of_date`.
- [ ] **Model B:** Feature set must **not** include `net_rating`. Enforced in `src.features.team_context.FORBIDDEN` and `train_model_b`.
- [ ] **ListMLE:** Targets are standings-to-date (win-rate), not season-end. Evaluation remains season-end.
- [ ] **Baselines only:** Net Rating is used only in `rank-by-Net-Rating` baseline, computed from off/def ratings, never as a model input.

---

## Report Assets (deliverables)

All paths under `outputs/` (or `config.paths.outputs`). Produced from real data when DB and models exist.

- `outputs/eval_report.json` — NDCG, Spearman, MRR (top_k=2), ROC-AUC upset, and `notes` (definitions) from script 5.
- `outputs/run_001/predictions.json` — per-team predicted league rank (1–30), actual conference rank (1–15), true strength score, delta, classification, ensemble diagnostics.
- `outputs/run_001/pred_vs_actual.png` — two panels (East and West): x = actual conference rank (1–15), y = predicted league rank (1–30); grid lines, team-colored points, and legend (script 6).
- `outputs/shap_summary.png` — Model B (RF) SHAP summary on real team-context features (script 5b).
- `outputs/oof_pooled.parquet`, `outputs/ridgecv_meta.joblib` — stacking meta-learner and pooled OOF (script 4b).
- `outputs/oof_model_a.parquet`, `outputs/oof_model_b.parquet` — OOF from scripts 3 and 4 (Option A: K-fold, real data).
- `outputs/best_deep_set.pt`, `outputs/xgb_model.joblib`, `outputs/rf_model.joblib` — trained Model A and Model B.

---

## Reproducibility

- **Seeds:** `src.utils.repro.set_seeds(seed)` for random, numpy, torch.
- **Manifests:** `outputs/run_manifest.json` (config snapshot, git hash, data manifest hash). `data/manifest.json` (raw/processed hashes).
- **OOF:** `outputs/oof_pooled.parquet` and `ridgecv_meta.joblib`.
- **Season boundaries:** Hard-coded in `config/defaults.yaml` to avoid play-in ambiguity.

---

## Planned updates (Update1)

The following extensions are planned (see [.cursor/plans/Update1.md](.cursor/plans/Update1.md)):

- **Playoff data:** Ingest playoff game logs (nba_api, SeasonType=Playoffs) into separate DuckDB tables (`playoff_games`, `playoff_team_game_logs`, `playoff_player_game_logs`). Play-In games excluded from playoff win counts.
- **Playoff performance rank (1–30):** Ground truth by (1) playoff wins, (2) tie-break by regular-season win %, (3) non-playoff teams 17–30 by regular-season record. Used for training (optional) and evaluation.
- **Config:** `training.target_rank: standings | playoffs` (default `standings`); `output.odds_temperature` for championship odds.
- **Prediction outputs:** `global_rank` (1–30), `conference_rank` (1–15 per conference), `championship_odds` (softmax with temperature), `analysis.playoff_rank` and `rank_delta_playoffs` when playoff data exists.
- **Visuals:** `pred_vs_actual.png` updated to conference rank vs actual conference rank (same scale); new `pred_vs_playoff_rank.png`, `title_contender_scatter.png`, `odds_top10.png`, `sleeper_timeline.png`.
- **Evaluation:** Spearman vs playoff rank, NDCG@4 (final four), Brier score on championship odds; `eval_report.json` section `playoff_metrics`.

---

## Full Plan

See `.cursor/plans/Plan.md`. Planned extension: [.cursor/plans/Update1.md](.cursor/plans/Update1.md).

---

## Implementation Roadmap
The full phased development plan and file-by-file checklist are in
`.cursor/plans/Plan.md` under **Development and Implementation Plan**.
Update1 roadmap: `.cursor/plans/Update1.md`.
