# Model Performance Over Runs and Implications for Hyperparameters / Reverts

This document summarizes how pipeline performance evolved across runs (from `.cursor/plans/` and `outputs/ANALYSIS.md`), what that implies about optimal hyperparameters, and which code or config might need reverting or re-checking.

---

## 1. Performance trajectory across runs

| Run / period | Documented metrics | Main causes (from plans) |
|--------------|--------------------|---------------------------|
| **run_008** | Not reported numerically; plans cite **NaN attention**, **rank scale mismatch** (conference 1–15 vs global 1–30), IG batching failure. | Evaluation mixed 1–15 and 1–30; `actual_rank` overwritten across lists; conference plot fallback used global rank on 1–15 axes; NaN in JSON. |
| **run_009** | **ndcg ~0.00026**, **spearman 0.0**, **mrr 0.0**, **roc_auc_upset 0.5**. Playoff: spearman -0.18, ndcg_at_4 0.36, brier ~0.001. | Model A trained **3 epochs** with **flat loss 27.8993**; evaluation still tied to run_009 when run_010 was created (5_evaluate runs before 6). Possible EOS/rank alignment bugs. |
| **run_010** | No eval numbers in plans; **playoff_rank** all 17–30 (16 null, 14 non-null in 17–30); **attention weights 0.0** everywhere. | Same 3-epoch Model A; **playoff season mapping** bug (`"2024"` vs `"2023-24"`) → empty filters → lottery-only ranks; attention diluted by **query including padded positions** (Update3/next_steps). |
| **Current** (eval_report.json) | **ndcg 0.64**, **spearman 0.72**, **mrr 0.0**, **roc_auc_upset 0.63**. No playoff_metrics in report. | Eval uses **EOS_global_rank** and **predicted_strength** (fix_run_009 renames); **Model A epochs** increased (config has **epochs: 20**, early stopping). Likely run_011–013 or later. |

**Summary:** Performance improved sharply after (1) **evaluation fix** (EOS_global_rank, correct scale, field renames), (2) **more Model A training** (3 → 20 epochs). Run_008/009/010 suffered from both evaluation bugs and under-training; current metrics reflect fixed eval and longer training.

---

## 2. What the plans say about optimal hyperparameters

### Model A (Deep Set)

- **Epochs**
  - **run_009/010:** 3 epochs → flat loss, no learning (Update3, next_steps, ANALYSIS.md).
  - **Sweeps (Update4–6):** Epoch grids 5,10,15,20,25 then 15,20,25,30,35,40 then **8–28 step 1** with **val_frac=0.25**.
  - **Recommended (next_full_pipeline_run):**
    - **NDCG-first:** `epochs=28`, `early_stopping_patience=0` so training reaches 28 (early stopping is val-loss-based, not NDCG).
    - **Spearman-first:** `epochs≈15`.
  - **Current config:** `epochs: 20`, `early_stopping_patience: 3`, `early_stopping_val_frac: 0.1` — between the two sweep sweet spots; val_frac is smaller than sweep’s 0.25.

- **Attention / convergence**
  - Plans (Update3, next_steps) attribute all-zero attention to: (a) **only 3 epochs**, (b) **query using mean over all positions** (including padding) in `set_attention.py`. Fix: masked mean for query (exclude padded positions). If attention is still zero after increasing epochs, **query-masking fix** is the next candidate; no plan suggests reverting it.

### Model B (XGB + RF)

- **Sweep recommendations (next_full_pipeline_run, refined_sweep_rerun):**
  - **Ranking-first (best spearman_mean):** XGB `max_depth=4`, `learning_rate=0.08`, `n_estimators=250`; RF `n_estimators=200`, `max_depth=12`, `min_samples_leaf=5`.
  - **RMSE-first:** XGB `max_depth=4`, `lr=0.10`, `n_estimators=300`; RF 150/12/4.
- **Current config:** XGB `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`; RF 200/12/5.
- So current config is **different** from the sweep “ranking-first” choice (deeper XGB, more trees, lower LR). The good current eval (Spearman 0.72) could be from Model A + stacking rather than Model B tuning; trying sweep-style Model B (e.g. 4/0.08/250) is still recommended for comparability.

### Validation split

- Sweeps use **val_frac=0.25**; config has **early_stopping_val_frac: 0.1**. Larger val_frac (0.25) is intended for more robust epoch selection and less overfitting to a small val set.

---

## 3. Code or config that might need reverting or re-checking

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

## 4. Summary table

| Topic | Recommendation |
|-------|----------------|
| **Performance trend** | run_008/009/010: broken eval and/or 3 epochs → bad metrics. Current: fixed eval + 20 epochs → ndcg 0.64, spearman 0.72, roc_auc_upset 0.63. |
| **Optimal Model A epochs** | NDCG-first: 28 (patience=0). Spearman-first: ~15. Current 20 is in between. |
| **Optimal Model B** | From sweeps: XGB 4/0.08/250, RF 200/12/5 (ranking-first). Current config differs; consider aligning. |
| **Validation fraction** | Sweeps use 0.25; config has 0.1. Prefer 0.25 for epoch/sweep comparability. |
| **Do not revert** | EOS_global_rank in eval, field renames, NaN sanitization, configurable/increased epochs. |
| **Re-check** | Roster latest-team (contamination); playoff date-range filtering; conference plot 1–15 only; Model B hparams vs sweep. |
| **Optional fixes** | Query masking in set_attention if attention still zero; NDCG-based early stopping. |
