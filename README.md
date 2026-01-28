# NBA "True Strength" Prediction

**Attention-Based Deep Set Network with Ensemble Validation**

Tanaka Makuvaza  
Georgia State University — Advanced Machine Learning  
January 27, 2026

---

## Overview
This project builds a **Multi-Modal Stacking Ensemble** to predict NBA **True Team Strength** using a Deep Set roster model plus a Hybrid tabular ensemble (XGBoost + Random Forest). The system targets **future outcomes** and identifies **Sleepers** versus **Paper Tigers** without circular evaluation.

---

## Key Design Choices
- **Target:** Future W/L (next 5) or Final Playoff Seed — **never** efficiency.
- **True Strength:** Latent **Z** from Deep Set penultimate layer; score mapped to percentile within conference.
- **No Net Rating leakage:** `net_rating` is excluded as a model input and never used as a target or evaluation metric (allowed only in baselines).
- **Stacking:** K-fold **OOF** across **all training seasons**; level-2 **RidgeCV** on pooled OOF (not Logistic Regression).
- **Game-level ListMLE:** lists per conference-date/week; **torch.logsumexp** for numerical stability; hash-trick embeddings for new players.
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
- **Ranking:** NDCG, Spearman, MRR.
- **Future outcomes:** Brier score.
- **Sleeper detection:** ROC-AUC on playoff upsets.
- **Baselines:** rank-by-SRS, rank-by-Net-Rating, **Dummy** (e.g. previous-season rank or rank-by-net-rating).

---

## Outputs
- Predicted rank (1–15)
- True strength score (0–1)
- Fraud/Sleeper delta
- Ensemble agreement
- Roster dependence (attention weights)
- **Sleeper Timeline** chart: true_strength_score vs actual rank over time (e.g. by week)

---

## How to Run the Pipeline

1. **Setup:** `pip install -r requirements.txt`
2. **Config:** Edit `config/defaults.yaml` if needed (seasons, paths, model params).
3. **Repro manifest:** `python -m scripts.run_manifest` — writes `outputs/run_manifest.json` (config snapshot, git hash, data manifest hash).
4. **Leakage tests:** `python -m scripts.run_leakage_tests` — run before training.
5. **Data:**  
   - `python -m scripts.1_download_raw` — fetch player/team logs via nba_api (optional; place CSVs in `data/raw/` as `team_logs_YYYY_YY.csv`, `player_logs_YYYY_YY.csv` to skip).  
   - `python -m scripts.2_build_db` — build `data/processed/nba.duckdb` and `data/manifest.json`.
6. **Models:**  
   - `python -m scripts.3_train_model_a` — Deep Set + ListMLE → `outputs/best_deep_set.pt`.  
   - `python -m scripts.4_train_model_b` — XGB + RF → `outputs/xgb_model.joblib`, `outputs/rf_model.joblib` (requires `xgboost`).  
   - `python -m scripts.4b_train_stacking` — RidgeCV on OOF → `outputs/ridgecv_meta.joblib`, `outputs/oof_pooled.parquet`.
7. **Eval:** `python -m scripts.5_evaluate` → `outputs/eval_report.json`.  
   **Explainability:** `python -m scripts.5b_explain` → SHAP summary (Model B), attention ablation (Model A).
8. **Inference:** `python -m scripts.6_run_inference` → `outputs/run_001/predictions.json`, `pred_vs_actual.png`.

---

## Anti-Leakage Checklist

- [ ] **Time rule:** All features use only rows with `game_date < as_of_date` (strict t-1). Rolling stats use `shift(1)` before aggregation.
- [ ] **Roster:** Minutes and roster selection use only games before `as_of_date`.
- [ ] **Model B:** Feature set must **not** include `net_rating`. Enforced in `src.features.team_context.FORBIDDEN` and `train_model_b`.
- [ ] **ListMLE:** Targets are standings-to-date (win-rate), not season-end. Evaluation remains season-end.
- [ ] **Baselines only:** Net Rating is used only in `rank-by-Net-Rating` baseline, computed from off/def ratings, never as a model input.

---

## Report Assets

- `outputs/eval_report.json` — NDCG, Spearman, MRR, ROC-AUC upset.
- `outputs/run_001/predictions.json` — per-team predicted rank, true strength, delta, ensemble diagnostics.
- `outputs/run_001/pred_vs_actual.png` — predicted vs actual rank scatter.
- `outputs/shap_summary.png` — Model B SHAP summary (after `5b_explain` with RF).
- `outputs/oof_pooled.parquet` — pooled OOF for stacking diagnostics.

---

## Reproducibility

- **Seeds:** `src.utils.repro.set_seeds(seed)` for random, numpy, torch.
- **Manifests:** `outputs/run_manifest.json` (config snapshot, git hash, data manifest hash). `data/manifest.json` (raw/processed hashes).
- **OOF:** `outputs/oof_pooled.parquet` and `ridgecv_meta.joblib`.
- **Season boundaries:** Hard-coded in `config/defaults.yaml` to avoid play-in ambiguity.

---

## Full Plan

See `.cursor/plans/Plan.md`.
