---
name: Fix Issues for Final Pipeline Run
overview: Comprehensive fixes for attention weights, playoff ranks, roster logic, and hyperparameter alignment to achieve trustworthy final results. Includes debugging, fixes, validation, and a pre-flight checklist.
todos:
  - id: update-hyperparams
    content: Update config/defaults.yaml with sweep-aligned hyperparameters (epochs=28, val_frac=0.25, XGB 250/4/0.08)
    status: pending
  - id: fix-attention-minutes
    content: Fix set_attention.py minutes reweighting (simplify to soft bias or make optional)
    status: pending
  - id: fix-playoff-threshold
    content: Keep playoffs.py threshold at 16; skip playoff rank/metrics when incomplete + debug logging
    status: pending
  - id: add-roster-debug
    content: Add roster debug logging to build_roster_set.py to verify latest-team logic
    status: pending
  - id: add-attention-debug
    content: Add final attention stats logging to train_model_a.py
    status: pending
  - id: create-preflight
    content: Create scripts/preflight_check.py validation script
    status: pending
  - id: pipeline-order
    content: Reorder run_full_pipeline.py so inference runs before evaluation; add run_leakage_tests before training
    status: pending
  - id: eval-scale
    content: In 5_evaluate.py use only EOS_global_rank; gate playoff_metrics on valid ranks
    status: pending
  - id: preflight-safe-retrain
    content: Preflight non-destructive retrain (archive/--force-retrain); run_id and model checks
    status: pending
  - id: delete-old-models
    content: "Optional: archive old checkpoints to force full retrain (non-destructive)"
    status: pending
  - id: run-pipeline
    content: Run full pipeline with new config and verify all metrics
    status: pending
  - id: update-performance-doc
    content: Update Performance_trajectory.md with final results and hyperparameters
    status: pending
isProject: false
---

# Fix All Issues for Trustworthy Final Pipeline Run

## Current State Summary

The **current eval_report.json** shows improved metrics (ndcg 0.64, spearman 0.72, roc_auc_upset 0.63), but several issues remain:

1. **Attention weights still 0.0** for all teams (ANALYSIS.md)
2. **Playoff metrics missing** from eval_report (no `playoff_metrics` section)
3. **MRR is 0** (expected given evaluation definition)
4. **Hyperparameters** not aligned with sweep recommendations
5. **Roster contamination** suspected (wrong-team players)

---

## Issue 1: Attention Weights All Zero

### Root Cause Analysis

The [set_attention.py](src/models/set_attention.py) already has masked mean for query (lines 30-35), so that fix is in place. The issue is likely:

1. **Minutes reweighting** (lines 38-45) may collapse weights when minutes are small:

```python
w = w * (0.5 + 0.5 * mins.unsqueeze(1))
w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
```

When `mins` is small (near 0), `0.5 + 0.5 * mins` approaches 0.5, but if all players have similar small minutes, the reweighted attention sums may become unstable.

1. **Training epochs** - ANALYSIS.md reports flat loss 27.8993 across 3 epochs. Current config has `epochs: 20`, but if training used an old checkpoint or the model was not retrained, attention could still be untrained.

### Fix

- [set_attention.py](src/models/set_attention.py): Simplify or make minutes reweighting optional (add minutes-sum guard; make configurable for A/B). If all weights non-finite/zero at inference, leave contributors empty and set flag.

```python
# Only apply minutes reweighting when minutes sum is meaningful (guard)
# Make minutes bias configurable (on/off) for A/B testing
if minutes is not None and w.shape[-1] == minutes.shape[-1]:
    mins = minutes
    if key_padding_mask is not None and key_padding_mask.shape == minutes.shape:
        mins = mins.masked_fill(key_padding_mask, 0.0)
    mins = mins.clamp(min=0.0)
    mins_sum = mins.sum(dim=-1, keepdim=True)
    if (mins_sum > 1e-6).any():  # only reweight when meaningful
        mins_norm = mins / mins_sum.clamp(min=1e-8)
        # Soft bias: 0.7 * attention + 0.3 * minutes (instead of multiplicative)
        w = 0.7 * w + 0.3 * mins_norm.unsqueeze(1)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
```

- [train_model_a.py](src/training/train_model_a.py): Add persistent attention diagnostics at end of training (gate behind debug flag):

```python
# After training, log final attention stats
with torch.no_grad():
    scores, attn_list = predict_batches_with_attention(model, batches[-1:], device)
    attn = attn_list[0]
    print(f"Final attention stats: min={attn.min():.6f}, max={attn.max():.6f}, sum={attn.sum(dim=-1).mean():.4f}")
```

---

## Issue 2: Playoff Rank Not Working (All 17-30)

### Root Cause Analysis

In [playoffs.py](src/evaluation/playoffs.py) line 166-167:

```python
if len(playoff_team_ids) < 16:
    return {}
```

This is too strict. If playoff data is incomplete or season filtering yields fewer than 16 teams, the function returns empty, causing all teams to get `playoff_rank: null`.

Additionally, [predict.py](src/inference/predict.py) lines 413-440 correctly pass `season_start` and `season_end` to `compute_playoff_performance_rank`, but we need to verify the dates match the target season.

### Fix

- [playoffs.py](src/evaluation/playoffs.py):
  - **Keep 16 as minimum** for trustworthy playoff ranks. When fewer than 16 playoff teams are found, skip playoff ranks/metrics and emit a warning (do not lower threshold to 8; partial ranks are misleading).
  - Guard downstream plots/metrics when `compute_playoff_performance_rank` returns `{}`.
  - Add logging when returning empty:

```python
# Line 166-167: Keep threshold at 16; log when skipping
MIN_PLAYOFF_TEAMS = 16  # Require full playoff field for valid ranks
if len(playoff_team_ids) < MIN_PLAYOFF_TEAMS:
    print(f"Warning: Only {len(playoff_team_ids)} playoff teams found (min {MIN_PLAYOFF_TEAMS}). Skipping playoff rank/metrics.")
    return {}
```

- Add debug logging to verify date filtering is working:

```python
# In _filtered_playoff_tgl, add:
print(f"Playoff filtering: season={season}, start={season_start}, end={season_end}, games_before={len(pg)}, games_after={len(result)}")
```

---

## Issue 3: Roster Contamination

### Root Cause Analysis

ANALYSIS.md reports wrong-team players (e.g., Simons on Boston). The [build_roster_set.py](src/features/build_roster_set.py) `latest_team_map_as_of()` function builds a map of player -> latest team as of `as_of_date`. The issue is:

1. `season_start` might not be passed, causing the map to include games from previous seasons
2. The `latest_team_map` might use current-season data when `as_of_date` is set to a date in a past season

### Fix

- [build_roster_set.py](src/features/build_roster_set.py): Enforce roster uses **games within season window and before as_of_date**. Add guard: warn or abort if `season_start` is missing or roster is empty. Add debug logging (gated):

```python
# In get_roster_as_of_date, after building latest_team_map:
print(f"Roster for team {team_id} as_of {as_of_date}: {len(latest_team_map)} players in map, season_start={season_start}")
```

- [data_model_a.py](src/training/data_model_a.py): Verify `season_start` is passed to `get_roster_as_of_date`; enforce season bounds.

---

## Issue 4: Pipeline Order and Run Consistency

### Root Cause Analysis

[run_full_pipeline.py](scripts/run_full_pipeline.py) currently runs **evaluation before inference**, so `eval_report.json` reflects a previous run, not the one just produced. Leakage tests exist but are not run in the pipeline.

### Fix

- [run_full_pipeline.py](scripts/run_full_pipeline.py): Reorder steps so **inference runs before evaluation**:
  - download → build_db → **run_leakage_tests** → train A → train B → stacking → **inference** → **evaluate** → explain
- Add **run_id consistency**: ensure evaluation uses the same run that produced the latest `predictions.json` (or config-specified run_id).

---

## Issue 5: Evaluation Scale Mixing

### Root Cause Analysis

[5_evaluate.py](scripts/5_evaluate.py) falls back from `EOS_global_rank` to `EOS_conference_rank` (and others), mixing 1–30 and 1–15 scales and corrupting metrics.

### Fix

- [5_evaluate.py](scripts/5_evaluate.py): Use **only** `analysis.EOS_global_rank` for ranking metrics (or fail/skip teams without it). Do not fall back to conference rank for global metrics.
- Gate `playoff_metrics` on a minimum number of valid playoff ranks; skip section when most ranks are null.

---

## Issue 6: Hyperparameters Not Aligned with Sweeps

### Current vs Recommended (from Performance_trajectory doc)


| Parameter                       | Current | Sweep Best (NDCG) | Sweep Best (Spearman) |
| ------------------------------- | ------- | ----------------- | --------------------- |
| model_a.epochs                  | 20      | 28                | 15                    |
| model_a.early_stopping_patience | 3       | 0 (no early stop) | 3                     |
| model_a.early_stopping_val_frac | 0.1     | 0.25              | 0.25                  |
| model_b.xgb.n_estimators        | 500     | 250               | 250                   |
| model_b.xgb.max_depth           | 6       | 4                 | 4                     |
| model_b.xgb.learning_rate       | 0.05    | 0.08              | 0.08                  |


### Fix

Update [config/defaults.yaml](config/defaults.yaml):

```yaml
model_a:
  epochs: 28                    # Sweep best for NDCG
  early_stopping_patience: 0    # Disable to reach epoch 28
  early_stopping_val_frac: 0.25 # Larger val for robustness

model_b:
  xgb:
    n_estimators: 250           # Sweep best
    max_depth: 4                # Sweep best
    learning_rate: 0.08         # Sweep best
```

---

## Issue 7: Pre-flight Validation Script

Create a validation script that checks all prerequisites before a full pipeline run.

### New file: [scripts/preflight_check.py](scripts/preflight_check.py)

```python
"""Pre-flight check: verify DB, config, and model prerequisites."""
# Checks:
# 1. DB exists and has required tables
# 2. Config has valid season boundaries
# 3. Raw data files exist
# 4. Model file presence (optional: --force-retrain to archive/rename old checkpoints, non-destructive)
# 5. Playoff data exists for target season (if running playoff metrics)
# 6. Run ID consistency (predictions.json and eval_report.json refer to same run)
```

---

## Implementation Order

1. **Pipeline order**: Reorder run_full_pipeline.py (inference before evaluate); add run_leakage_tests before training
2. **Evaluation scale**: In 5_evaluate.py use only EOS_global_rank; gate playoff_metrics
3. **Update hyperparameters** in defaults.yaml (sweep-aligned)
4. **Fix set_attention.py** minutes reweighting (guard + optional/configurable); inference fallback policy
5. **Fix playoffs.py** keep threshold 16, skip when incomplete, debug logging
6. **Roster guards** in build_roster_set.py and data_model_a (season window, as_of_date; warn if missing)
7. **Create preflight_check.py** (run_id, model presence, non-destructive retrain option)
8. **Retrain models** from scratch (optional: archive old checkpoints, non-destructive)
9. **Run full pipeline** and verify attention + playoff metrics
10. **Update Performance_trajectory.md** with final results

---

## Verification Checklist (After Pipeline Run)

- **Run ID consistency**: eval_report.json refers to the same run as the predictions just produced
- Attention weights in predictions.json are non-zero for some teams; **attention has non-trivial distribution** (not all zero)
- `attn_sum_mean` in training/inference logs is close to 1.0
- Playoff metrics present in eval_report.json only when playoff data is complete; **no partial playoff ranks** when data is incomplete
- `playoff_rank` in predictions.json includes values 1-16 when data is complete (not all 17-30 from partial data)
- No roster contamination (spot-check Boston, Milwaukee rosters)
- Loss decreases during training (not flat)
- NDCG >= 0.64, Spearman >= 0.72 (maintain or improve)

---

## Files to Modify


| File                                                                                                                       | Changes                                                 |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| [config/defaults.yaml](config/defaults.yaml)                                                                               | Update epochs, val_frac, XGB params                     |
| [src/models/set_attention.py](src/models/set_attention.py)                                                                 | Minutes guard, optional bias, fallback policy           |
| [src/evaluation/playoffs.py](src/evaluation/playoffs.py)                                                                   | Keep 16 threshold, skip when incomplete, debug logging  |
| [src/features/build_roster_set.py](src/features/build_roster_set.py)                                                       | Enforce season window, roster guards, debug logging     |
| [src/training/train_model_a.py](src/training/train_model_a.py)                                                             | Final attention stats (debug flag)                      |
| [scripts/preflight_check.py](scripts/preflight_check.py)                                                                   | Validation + run_id + non-destructive retrain option    |
| [scripts/run_full_pipeline.py](scripts/run_full_pipeline.py)                                                               | Reorder: inference before evaluation; add leakage tests |
| [scripts/5_evaluate.py](scripts/5_evaluate.py)                                                                             | Use only EOS_global_rank; gate playoff_metrics          |
| [.cursor/plans/Performance_trajectory_and_hyperparameters.md](.cursor/plans/Performance_trajectory_and_hyperparameters.md) | Update with fixes and final hyperparams                 |


