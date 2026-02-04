---
name: NBA True Strength Prediction
overview: Attention-Based Deep Set Network with Ensemble Validation for NBA "True Strength" Prediction.
isProject: false
---

# Project Report: Attention-Based Deep Set Network with Ensemble Validation for NBA "True Strength" Prediction

Tanaka Makuvaza  
Georgia State University  
Advanced Machine Learning  
January 27, 2026

---

## 1. Executive Summary

Traditional sports prediction models often rely on simple team-level season averages or basic linear regression, which fail to capture non-linear roster interactions and schedule difficulty. This project proposes a **Multi-Modal Stacking Ensemble** to predict **True Team Strength** and validate playoff rankings. By integrating a **Deep Set Neural Network** (player-level roster modeling) with a **Bagging-Boosting Hybrid** (XGBoost + Random Forest), the system aims to outperform standings-based baselines and identify **Sleeper** contenders versus **Paper Tigers**.

---

## Roadmap (Authoritative)

This is the top-level execution roadmap for the project. It is the authoritative plan to implement, and it supersedes any alternate planning docs (e.g., `Opus_Plan.md`) which should be treated as optional implementation references only.

1. **Phase 0** - Requirements lock and acceptance criteria  
2. **Phase 1** - Data layer and storage (DuckDB, ingestion, manifest)  
3. **Phase 2** - Feature engineering and leakage controls (t-1, DNP, roster)  
4. **Phase 3** - Model A (Deep Set) + stable ListMLE  
5. **Phase 4** - Model B (XGBoost + RF, no net_rating)  
6. **Phase 5** - OOF stacking meta-learner (RidgeCV on pooled OOF)  
7. **Phase 6** - Evaluation + baselines (NDCG/Spearman/MRR, Brier, ROC-AUC)  
8. **Phase 7** - Explainability + attention validation  
9. **Phase 8** - Visualization suite + reporting  
10. **Phase 9** - Integration and reproducibility controls  

Details for each phase and file-level tasks are defined in **Section 9** and **Section 10**.

---

## 2. Data Strategy and Scope

### 2.1 Historical Constraints: The "Modern Era"
Training data is restricted to **2015-2016 through 2025-2026**.
- **Justification:** The Three-Point Revolution (circa 2015) fundamentally changed shot profiles and winning efficiency. Pre-2015 data introduces concept drift.

### 2.2 Data Acquisition
A unified SQL database will be constructed from:
1. **nba_api (official):** play-by-play, player IDs, tracking data (speed, distance, touch time).
2. **Kaggle (Wyatt Walsh):** **primary** source for SOS/SRS and historical validation (preferred over live scraping).
3. **Basketball-Reference (scraped):** **fallback only** for SOS/SRS when Kaggle is unavailable; use sparingly due to anti-bot protection.

**Proxy SOS:** If neither Kaggle nor Basketball-Reference is available, compute an internal **Proxy SOS** from the unified DB (e.g., opponent win-rate or opponent SRS derived from stored results). Document the formula and mark outputs when Proxy SOS is used.

### 2.3 Storage and Performance
- **Preferred:** **DuckDB** for large joins (tracking + play-by-play over 10 years).
- **If SQLite:** pre-aggregate heavy tables and **index** `game_id`, `player_id`, `team_id`.

---

## 3. Methodology: The Architecture

### 3.1 Model A - Deep Set Network ("The Roster Expert")
A permutation-invariant Deep Set processes a team as a **set of player vectors**.

**Input:** `(batch, 15_players, stats + embeddings)`
- Rolling windows **Last 10 / Last 30** games for form. Enforce strict **`t-1`** (pre-game only): in `rolling.py`, apply **`.shift(1)`** before any rolling aggregation so features use only information available before the prediction target.
- **Player Embeddings:** `R^32` for latent traits. For **new/unseen players**, use a **hash-trick** index: `hash(player_id) % num_embeddings` to avoid out-of-vocabulary errors.
- **Roster selection:** top-N by minutes **as-of date only** (never full-season). **Embedding order and inclusion** also use the **as-of date** only.
- **DNP handling:** stats = per-game average over games played; availability = fraction of games played.

**Minutes-weighted attention ("The Coach"):**
- Use **Rolling Avg Minutes (Last 5)** or **Active Status (0/1)** strictly at `t-1`.
- **No projected minutes / Vegas**.
- **Either** explicit minutes-weighting **or** MP as encoder input (default: explicit minutes-weighting; no MP in encoder stats).

**Output:** A **Team Strength Vector (Z)** from the penultimate layer.

### 3.2 Model B - Hybrid Tabular Ensemble ("The Stats Validator")
- **XGBoost** (bias reduction) + **Random Forest** (variance reduction).
- Inputs: Four Factors, SOS, SRS, pace, SOS-adjusted stats.
- **No `net_rating`** as a feature to avoid leakage.

### 3.3 Meta-Learner (True Stacking)
Weighted averaging is **not** stacking. We use **OOF stacking**:
1. Train Deep Set on folds 1–4, predict fold 5 (repeat over K folds).
2. Train XGB/RF on folds 1–4, predict fold 5 (repeat over K folds).
3. **Level-2:** **RidgeCV** on OOF predictions (not Logistic Regression).
4. **OOF scope:** Collect OOF predictions **across all training seasons**; train the meta-learner on **pooled OOF data**, not a single season.

Persist OOF predictions (`outputs/oof_*.parquet`) for diagnostics.

### 3.4 Game-Level ListMLE Training
Season-level lists are too small (~40 lists). Instead:
- Build lists by **conference-date (or week)**.
- Features and rosters strictly **as of t-1**.
- Rank target = **standings-to-date** (or win-rate-to-date), not season-end totals.
- **Evaluation** remains **season-end only**.
- **ListMLE numerical stability:** Use **log-sum-exp** (e.g. `torch.logsumexp`) in the ListMLE loss to avoid overflow/underflow; do **not** use raw exp then log.

---

## 4. Evaluation Strategy

Primary evaluation focuses on ranking + predictive quality:
1. **NDCG** (top-heavy ranking quality).
2. **Spearman** (monotonic order alignment).
3. **MRR** (top-1 accuracy).
4. **Brier Score** on future game outcomes.
5. **Sleeper = Upset:** ROC-AUC on playoff series upsets (lower seed beats higher seed).

**Baselines:**
- Rank-by-SRS
- Rank-by-Net-Rating
- **Dummy:** e.g. previous-season rank or rank-by-net-rating (simple non-learned baseline).

No MAE on Net Rating; no efficiency alignment metric in evaluation. Net Rating is **not** a model target or evaluation metric; it may be used only in baselines (e.g. rank-by-Net-Rating, Dummy).

### 4.1 Target comparison and validation

- We will test targeting **both** `playoff_outcome` and `final_rank` (standings) in sweeps or dedicated runs.
- For **standings**, results will show relevance or irrelevance (e.g., if standings-targeted model underperforms vs playoff outcome).
- **Hypothesis:** Playoff-outcome target will be more accurate than standings consistently for playoff-relevant metrics (Spearman vs playoff rank, NDCG@4).
- **Final report:** Will calculate confidence levels (e.g., bootstrap CIs on Spearman/NDCG) and use statistical tests to demonstrate that the model's performance is above chance / baselines.

---

## 5. Explainability and Validation
- **SHAP only on Model B** (RF/XGBoost).
- **Model A:** Integrated Gradients (Captum) or permutation ablation.
- **Attention != explanation**. Validate with **ablation**: mask high-attention player vs random player and compare accuracy drop.

---

## 6. Defined Model Outputs

### 6.1 Primary Outputs
1. **Predicted Conference Rank (1-15)**
2. **True Strength Score (0.0-1.0)**
   - Default: **percentile within conference** by model score.
   - Alternatives: `softmax` probability of #1 or Platt scaling.
3. **Fraud/Sleeper Delta:** `Rank_actual - Rank_predicted`.
4. **Ensemble Agreement:** rank spread across Deep Set / XGB / RF.

### 6.2 Sample JSON Output
```json
{
  "team_name": "Indiana Pacers",
  "prediction": {
    "predicted_rank": 4,
    "true_strength_score": 0.782
  },
  "analysis": {
    "actual_rank": 6,
    "classification": "Sleeper (Under-ranked by 2 slots)",
    "explanation": "Efficiency exceeds win-loss record due to difficult schedule."
  },
  "ensemble_diagnostics": {
    "model_agreement": "High",
    "deep_set_rank": 4,
    "xgboost_rank": 3,
    "random_forest_rank": 5
  },
  "roster_dependence": {
    "star_reliance_score": "High",
    "primary_contributors": [
      {"player": "Tyrese Haliburton", "attention_weight": 0.35},
      {"player": "Pascal Siakam", "attention_weight": 0.28}
    ]
  }
}
```

---

## 7. Visualization Plan
1. **Accuracy Plot:** Predicted vs actual rank with identity line.
2. **Fraud/Sleeper Index:** Diverging bar chart of rank deltas.
3. **SHAP Summary:** Model B feature importance.
4. **Roster Attention Distribution:** attention weights per team.
5. **Sleeper Timeline:** true_strength_score vs actual rank over time (e.g. by week); highlights teams whose model score diverges from standings over the season.

---

## 8. Reproducibility and Diagnostics
- **Seeds:** Set seeds for **torch**, **numpy**, and **sklearn**.
- **OOF:** Persist OOF predictions (`outputs/oof_*.parquet`) for stacking analysis.
- **Data versioning:** Store hashes of raw + processed datasets.
- **Season boundaries:** In `defaults.yaml` (or equivalent config), **hard-code** season date ranges, e.g. `{season: {start: "YYYY-10-01", end: "YYYY-04-15"}}`, to avoid inferring "end of regular season" from logs (which can blur play-in vs. regular season).

---

## 9. Development and Implementation Plan

### Phase 0 - Requirements Lock and Acceptance Criteria
1. Freeze target definition (future W/L or final seed only).
2. Confirm no Net Rating as input or evaluation metric.
3. Finalize evaluation metrics and baselines.
4. Define True Strength score mapping (percentile default).
5. Validate data source policy (Kaggle primary, BRef fallback, Proxy SOS).

**Exit criteria:** Finalized Plan.md + README scope; updated metrics list; season calendar confirmed.

### Phase 1 - Data Layer and Storage
1. Define DB schema (games, player logs, team logs, standings, context).
2. Implement `nba_api` ingesters and Kaggle loader.
3. Implement Proxy SOS fallback (opponent win-rate or SRS-derived).
4. Choose DuckDB vs SQLite; add indexes if SQLite.
5. Add data manifest hashing for reproducibility.

**Deliverables:** `src/data/` loaders, schema, manifest builder.

### Phase 2 - Feature Engineering and Leakage Controls
1. Rolling windows with strict `t-1` (`shift(1)` before roll).
2. DNP handling: per-game averages + availability fraction.
3. Roster selection as-of date only; top-N by minutes.
4. Embeddings: hash-trick index for unseen players.
5. Build game-level lists by conference-date/week.

**Deliverables:** `src/features/rolling.py`, `src/data/build_roster_set.py`.

### Phase 3 - Model A (Deep Set)
1. Player encoder (shared MLP).
2. Minutes-weighted attention with padding mask.
3. ListMLE loss with `torch.logsumexp` stability.
4. Z extraction and prediction head.
5. DataLoader with custom collate_fn for variable-length rosters.

**Deliverables:** `src/models/player_encoder.py`, `set_attention.py`, `deep_set_rank.py`, `listmle_loss.py`.

### Phase 4 - Model B (Hybrid Tabular)
1. Build team-level features (Four Factors, SOS/SRS, pace).
2. Train XGBoost and Random Forest models.
3. Validate no Net Rating in feature set.

**Deliverables:** `src/models/xgboost_ensemble.py`, `rf_ensemble.py`, `features/team_context.py`.

### Phase 5 - OOF Stacking Meta-Learner
1. Generate OOF predictions across all training seasons.
2. Train RidgeCV on pooled OOF predictions.
3. Persist OOF outputs to `outputs/oof_*.parquet`.

**Deliverables:** `src/models/stacking.py`, `src/training/train_stacking.py`.

### Phase 6 - Evaluation and Baselines
1. Implement NDCG, Spearman, MRR.
2. Implement Brier score on future outcomes.
3. Implement ROC-AUC for Upset detection.
4. Baselines: rank-by-SRS, rank-by-Net-Rating, Dummy baseline.

**Deliverables:** `src/evaluation/metrics.py`, `src/evaluation/evaluate.py`.

### Phase 7 - Explainability and Validation
1. SHAP on Model B only.
2. Integrated Gradients or permutation ablation for Model A.
3. Attention validation via high-attention vs random masking.

**Deliverables:** `src/viz/shap_heatmap.py`, `src/viz/attention_roster.py`, ablation script.

### Phase 8 - Visualization Suite
1. Predicted vs Actual Rank scatter.
2. Fraud/Sleeper index bar chart.
3. SHAP summary (Model B).
4. Roster attention distribution.
5. Sleeper Timeline (score vs rank over season).

**Deliverables:** `src/viz/accuracy_plot.py`, `fraud_sleeper.py`, `sleeper_timeline.py`.

### Phase 9 - Integration and Reproducibility
1. End-to-end scripts (download -> build -> train -> evaluate -> inference).
2. Seed control in all scripts; deterministic DataLoader workers.
3. Data manifest versioning and run logs.

**Deliverables:** `scripts/1_download_raw.py` through `scripts/6_run_inference.py`.

---

## 10. File-by-File Implementation Checklist

### Data and Features
- `src/data/nba_api_client.py`: game logs, player logs, tracking.
- `src/data/kaggle_client.py`: SOS/SRS ingestion.
- `src/data/build_roster_set.py`: roster top-N as-of date; rolling stats; embedding indices.
- `src/features/rolling.py`: `shift(1)` before roll; DNP handling.
- `src/features/team_context.py`: Four Factors, SOS/SRS, pace; no net_rating.

### Models and Training
- `src/models/player_encoder.py`: shared MLP for player vectors.
- `src/models/set_attention.py`: masked attention; minutes-weighting rule.
- `src/models/deep_set_rank.py`: Z extraction + rank head.
- `src/models/listmle_loss.py`: logsumexp-stable ListMLE.
- `src/models/xgboost_ensemble.py`, `rf_ensemble.py`: tabular ensemble.
- `src/models/stacking.py`: RidgeCV meta-learner on OOF.
- `src/training/train_model_a.py`: Deep Set training, OOF support.
- `src/training/train_model_b.py`: XGB/RF training with proper splits.
- `src/training/train_stacking.py`: pooled OOF training.

### Evaluation and Outputs
- `src/evaluation/metrics.py`: NDCG, Spearman, MRR, Brier, ROC-AUC.
- `src/evaluation/evaluate.py`: baselines + diagnostics.
- `src/inference/predict.py`: JSON outputs, ensemble diagnostics.

### Visualization
- `src/viz/accuracy_plot.py`, `fraud_sleeper.py`, `shap_heatmap.py`
- `src/viz/attention_roster.py`, `sleeper_timeline.py`

### Config and Scripts
- `config/defaults.yaml`: seasons, paths, seeds, season boundaries.
- `scripts/1_download_raw.py` to `6_run_inference.py`: full pipeline.
