# Model Performance Over Runs and Implications for Hyperparameters / Reverts

This document summarizes how pipeline performance evolved across runs (from `outputs/` and `.cursor/plans/`), what that implies about optimal hyperparameters, and which code or config might need reverting or re-checking.

---

## 1. Performance trajectory across runs

### 1.1 Computed metrics per run (from `outputs/*/predictions.json`)

| Run | NDCG@10 | Spearman | MRR (top‑2) | ROC‑AUC upset | Schema | Attention |
|-----|---------|----------|-------------|---------------|--------|-----------|
| run_001 | 0.7854 | 0.7425 | 0.50 | 0.8519 | Old (`predicted_rank`, `true_strength_score`, `actual_rank`) | empty |
| run_002 | 0.7854 | 0.7425 | 0.50 | 0.8519 | Old | empty |
| run_003 | 0.7854 | 0.7425 | 0.50 | 0.8519 | Old | empty |
| run_008 | 0.8063 | 0.7211 | 0.50 | 0.8393 | Old | NaN |
| run_009 | 0.6378 | 0.7175 | 0.00 | 0.6295 | New (`predicted_strength`, `ensemble_score`, `EOS_global_rank`) | empty |
| run_010 | 0.6378 | 0.7175 | 0.00 | 0.6295 | New | 0.0 |
| run_011 | 0.6378 | 0.7175 | 0.00 | 0.6295 | New | empty |
| run_012 | 0.6378 | 0.7175 | 0.00 | 0.6295 | New | empty |
| run_013 | 0.6378 | 0.7175 | 0.00 | 0.6295 | New | empty |

*Note: run_004–007 have no `predictions.json`. Metrics computed with unified eval logic (same as `scripts/5_evaluate.py` with schema fallbacks for old runs).*

### 1.2 Inferences from the trajectory

1. **Schema change (run_009)**  
   Run_001–008 use the old schema (`actual_rank`, `predicted_rank`, `true_strength_score`). Run_009 onward use the new schema (`EOS_global_rank`, `predicted_strength`, `ensemble_score`). The higher NDCG/ROC in run_001–008 likely reflect **scale mismatch** (e.g. conference 1–15 vs global 1–30) or different relevance assumptions, not genuinely better models. Plans cite mixed 1–15/1–30 evaluation for run_008.

2. **Post-fix performance (run_009–013)**  
   With consistent `EOS_global_rank`-based eval, **ndcg ≈ 0.64**, **spearman ≈ 0.72**, **roc_auc_upset ≈ 0.63** across run_009–013. These runs share **identical predictions**, implying the same checkpoint (Model A + Model B + stacking) is being used. This is the current best **honest** performance.

3. **Attention status**  
   In no run did the attention mechanism produce non-zero weights: run_008 has NaN, run_010 has 0.0, others have empty `primary_contributors` or fallback only. ANALYSIS.md and IG attributions (L2 norm 0) confirm Model A attention is not learning.

4. **Hyperparameter timeline (from plans)**  
   - run_009/010: Model A trained **3 epochs** → flat loss, no learning.  
   - run_011–013: Config has **epochs: 20**; eval fixes in place.  
   There is no per-run hyperparameter record in `outputs/`; sweep outputs have not been written yet (Update6 pending).

---

## 2. Best-performing configs (from `outputs/`)

**No sweep outputs exist yet.** `outputs/` contains only full-pipeline runs (run_001–013), and no per-run hyperparameter snapshots are saved. Therefore:

| Source | Best NDCG | Best Spearman | Best ROC‑AUC | Notes |
|--------|-----------|---------------|--------------|-------|
| **Post-schema runs (run_009–013)** | 0.64 | 0.72 | 0.63 | Honest metrics; all five runs identical (same checkpoint). |
| **Old-schema runs (run_001–008)** | 0.81 | 0.74 | 0.85 | Likely inflated by scale mismatch; do not compare directly. |

**Conclusion:** The best **comparable** performance from outputs is **ndcg 0.64, spearman 0.72, roc_auc 0.63** (run_009–013). Achieved with current `defaults.yaml` (Model A `epochs: 20`, XGB 500/6/0.05, RF 200/12/5). No sweep has been run to empirically select better hyperparameters.

---

## 3. What the plans say about optimal hyperparameters

*Plan-based recommendations; sweeps (Update6) have not been run yet.*

### Model A (Deep Set)

- **Epochs**
  - **run_009/010:** 3 epochs → flat loss, no learning (Update3, next_steps, ANALYSIS.md).
  - **Sweeps (Update4–6, not yet run):** Epoch grids 5,10,15,20,25 then 15,20,25,30,35,40 then **8–28 step 1** with **val_frac=0.25**.
  - **Recommended (next_full_pipeline_run):**
    - **NDCG-first:** `epochs=28`, `early_stopping_patience=0` so training reaches 28 (early stopping is val-loss-based, not NDCG).
    - **Spearman-first:** `epochs≈15`.
  - **Current config:** `epochs: 20`, `early_stopping_patience: 3`, `early_stopping_val_frac: 0.1` — between the two sweep sweet spots; val_frac is smaller than sweep’s 0.25.

- **Attention / convergence**
  - Plans (Update3, next_steps) attribute all-zero attention to: (a) **only 3 epochs**, (b) **query using mean over all positions** (including padding) in `set_attention.py`. Fix: masked mean for query (exclude padded positions). If attention is still zero after increasing epochs, **query-masking fix** is the next candidate; no plan suggests reverting it.

### Model B (XGB + RF)

- **Sweep recommendations (Update6, refined_sweep_rerun — not yet run):**
  - **Phase 1 (XGB local):** `max_depth=4`; `learning_rate` {0.08, 0.10, 0.12}; `n_estimators` {250, 300, 350}; subsample/colsample 0.8.
  - **Phase 2 (RF local):** RF `n_estimators` {150, 200, 250}; `min_samples_leaf` {4, 5, 6}; `max_depth=12`.
  - **Ranking-first (prior plans):** XGB 4/0.08/250; RF 200/12/5. **RMSE-first:** XGB 4/0.10/300; RF 150/12/4.
- **Current config:** XGB `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`; RF 200/12/5.
- Current config **differs** from sweep “ranking-first” choice (deeper XGB, more trees, lower LR). Spearman 0.72 may be driven by Model A + stacking; aligning Model B to sweep (e.g. 4/0.08/250) is recommended once sweeps run.

### Validation split

- Sweeps use **val_frac=0.25**; config has **early_stopping_val_frac: 0.1**. Larger val_frac (0.25) is intended for more robust epoch selection and less overfitting to a small val set.

---

## 4. Code or config that might need reverting or re-checking

Plans do **not** ask to revert any change by name; they do call out regressions and fragile behavior:

### Do not revert (these fixed the metrics)

- **EOS_global_rank and evaluation (fix_run_009, Update2):** Using `EOS_global_rank` (fallback `EOS_conference_rank`) and `predicted_strength` in `scripts/5_evaluate.py` and inference. Reverting would mix scales again and break ranking metrics.
- **Field renames (predicted_strength, ensemble_score, EOS_*):** Consumers (eval, plots, README) expect the new names. Reverting would break evaluation and docs.
- **NaN sanitization and `allow_nan=False` (Update2):** Prevents invalid JSON and parser failures; keep.
- **Model A epochs configurable and increased (e.g. 3 → 20):** Core driver of the improvement from run_009 to current; do not drop.

### Re-check (might still be wrong; plans flag them)

1. **Roster “latest-team” (Update2, fix_run_009, ANALYSIS.md)**  
   - Update2 chose “latest-team roster” (most recent team per player as of `as_of_date`). ANALYSIS.md still reports **roster contamination** (e.g. Simons on Boston, Kuzma/Turner on Milwaukee for a historical date).  
   - **Action:** Verify `build_roster_set` / `data_model_a` and inference use a single `as_of_date` and that “latest team” is derived from games **before** that date only. If contamination persists, consider a different roster rule (e.g. games-only-in-season up to `as_of_date`) rather than reverting; reverting to “no latest-team” would bring back players on wrong teams.

2. **Playoff rank and season mapping (Update3, next_steps)**  
   - Playoff rank was broken in run_010: season derived as `"2024"` while pipeline uses `"2023-24"` → empty filters → ranks 17–30 only. Plans say: use **date-range filtering** from config (`is_game_in_season(game_date, season_cfg)`) instead of string matching.  
   - **Action:** If `playoff_rank` or `playoff_metrics` are still wrong or missing, implement date-range filtering in `src/evaluation/playoffs.py`; do not revert to year-string matching.

3. **Conference plot fallback (Update2)**  
   - Fix: do not fall back to global rank for pred_vs_actual (keep 1–15 only). If someone reverted this, East/West panels could mix 1–30 with 1–15; re-apply the fix if needed.

4. **Model B hyperparameters**  
   - Current config does not match sweep “ranking-first” best. Not a revert: optionally **align** Model B to sweep (e.g. XGB 4/0.08/250, RF 200/12/5) and re-run eval to see if NDCG/Spearman improve further.

### Optional / not yet done (from plans)

- **Query masking in set_attention (Update3, next_steps):** Masked mean for query so padded positions do not dilute it. If attention is still all zero after 20+ epochs, implement this; it’s an additive fix, not a revert.
- **NDCG-tracked early stopping:** Plans suggest logging NDCG per epoch and stopping on plateau for a “balanced” setup; current early stopping is val-loss-based. Would be an addition, not a revert.

---

## 5. Summary table

| Topic | Recommendation |
|-------|----------------|
| **Performance trend** | run_001–008: old schema, inflated ndcg/roc. run_009–013: new schema, honest ndcg 0.64, spearman 0.72, roc_auc 0.63 (identical predictions). |
| **Optimal Model A epochs** | NDCG-first: 28 (patience=0). Spearman-first: ~15. Current 20 is in between. |
| **Optimal Model B** | From plans (sweeps not yet run): XGB 4/0.08/250, RF 200/12/5 (ranking-first). Current config differs; consider aligning once sweeps run. |
| **Validation fraction** | Sweeps use 0.25; config has 0.1. Prefer 0.25 for epoch/sweep comparability. |
| **Do not revert** | EOS_global_rank in eval, field renames, NaN sanitization, configurable/increased epochs. |
| **Re-check** | Roster latest-team (contamination); playoff date-range filtering; conference plot 1–15 only; Model B hparams vs sweep. |
| **Optional fixes** | Query masking in set_attention if attention still zero; NDCG-based early stopping. |
