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
