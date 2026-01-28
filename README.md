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

## Reproducibility
- Set seeds for torch/numpy/sklearn.
- Persist OOF predictions to `outputs/oof_*.parquet`.
- Version datasets with hashes.
- Hard-code season date ranges in `defaults.yaml` to avoid play-in ambiguity.

---

## Full Plan
See `.cursor/plans/Plan.md`.
