# Implementation Plan: Centralized Training, Config Architecture, Attention Hooks, and Evaluation Expansion

**Created:** 2025-02-03  
**Status:** Draft  
**Related:** Plan.md, CHECKPOINT_PROJECT_REPORT.md

---

## Overview

This plan implements seven improvements:

1. **Centralize Training Loops** — Move core training logic into a class-based Trainer in `src/training/`.
2. **Configuration-Driven Architecture** — Pass a `ModelConfig` object to model constructors; save architecture DNA with checkpoints.
3. **Attention Debugging as Integrated Hook** — Use PyTorch hooks or a logging callback to monitor attention entropy; auto-restart on collapse.
4. **Attention Stability Metrics** — Track attention diversity (variance) across rosters and epochs.
5. **Championship Win-Rate Calibration** — Evaluate calibration using Brier Score against Monte Carlo / historical championship outcomes.
6. **Comparative Explainability (RFX-Fuse)** — Validate DeepSet high-attention players against RFX-Fuse historical archetypes.
7. **Strength of Schedule (SoS) Normalization** — Add “Net Rating Adjusted for Opponent” features.

---

## 1. Centralize Training Loops

### Current State

- **scripts/3_train_model_a.py** — OOF fold loop, subsampling, walk-forward, device setup, `train_model_a_on_batches` calls.
- **src/training/train_model_a.py** — `train_epoch`, `eval_epoch`, `train_model_a_on_batches`, `train_model_a`, `_split_train_val`, `_log_attention_debug_stats`, `NOT_LEARNING_PATIENCE` logic.
- **scripts/4_train_model_b.py** — K-fold loop, data prep, `train_model_b` calls.
- **src/training/train_model_b.py** — `build_xgb`, `fit_xgb`, `build_rf`, `fit_rf`, leakage check.
- **scripts/4b_train_stacking.py** — OOF aggregation, `train_stacking` call.
- **src/training/train_stacking.py** — `build_oof`, `fit_ridgecv_on_oof`, `save_oof`.

**Duplication:** Script 3 repeats batching logic, fold splitting, and train/val flow. `train_model_a` and `train_model_a_on_batches` both contain nearly identical epoch loops, early stopping, and “not learning” handling.

### Target Architecture

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `BaseTrainer` | `src/training/base_trainer.py` | Abstract: batching, early stopping, checkpointing, callbacks. |
| `ModelATrainer(BaseTrainer)` | `src/training/train_model_a.py` | ListMLE loss, gradient clipping, attention callbacks. |
| `ModelBTrainer` | `src/training/train_model_b.py` | XGB/RF fit logic (wraps sklearn/XGB APIs). |
| `StackingTrainer` | `src/training/train_stacking.py` | OOF aggregation, RidgeCV fit. |
| Scripts | `scripts/3_*.py`, `scripts/4_*.py`, `scripts/4b_*.py` | CLI args, config load, data load, high-level orchestration only. |

### Implementation Steps

1. **Create `src/training/base_trainer.py`**
   - `BaseTrainer` with:
     - `fit(batches, val_batches=None)` — generic epoch loop.
     - `_train_epoch(batches)` — abstract; subclasses implement.
     - `_eval_epoch(batches)` — abstract.
     - Early stopping: `patience`, `min_delta`, `best_val`.
     - Callback hooks: `on_epoch_start`, `on_epoch_end`, `on_train_end`.
     - Checkpoint: `save_checkpoint(path)`, `load_checkpoint(path)`.
   - Optional: `Callback` base class with `on_epoch_start`, `on_epoch_end`, `on_batch_end`.

2. **Refactor `train_model_a.py`**
   - `ModelATrainer(BaseTrainer)`:
     - Implements `_train_epoch` (ListMLE, grad clip).
     - Implements `_eval_epoch`.
     - Registers attention callback (see Section 3).
     - Keeps `_split_train_val`, `get_dummy_batch`, `predict_batches`, `predict_batches_with_attention` as utilities.
   - `train_model_a(config, output_dir, ...)` and `train_model_a_on_batches(...)` become thin wrappers that instantiate `ModelATrainer` and call `fit`.

3. **Refactor scripts**
   - **3_train_model_a.py:** Load config/DB, build lists, compute split, call `build_batches_from_lists` / `build_batches_from_db`, instantiate `ModelATrainer`, call `trainer.fit(train_batches, val_batches)`, write OOF parquet. No inline epoch logic.
   - **4_train_model_b.py:** Load config/DB, build lists, build features, split by folds, call `ModelBTrainer.fit(X_train, y_train, X_val, y_val)` per fold. Script handles fold iteration; trainer handles single fit.
   - **4b_train_stacking.py:** Load OOF, call `StackingTrainer.fit(oof_deep_set, oof_xgb, oof_rf, y)`.

4. **Shared utilities**
   - Move `_next_run_id`, `_reserve_run_id` to `src/utils/run_id.py` if used by multiple scripts.
   - Fold-splitting logic: keep in scripts (it’s orchestration) or move to `src/training/splitters.py`.

---

## 2. Configuration-Driven Architecture

### Current State

- `DeepSetRank` is constructed with many positional/keyword args in:
  - `src/training/train_model_a.py` (`_build_model`)
  - `src/inference/predict.py` (`load_models`) — infers `stat_dim` from checkpoint because config can mismatch.
- Checkpoint saves `{"model_state": ..., "config": config}` but inference often overrides with runtime config.
- `SetAttention` does not accept `temperature`, `input_dropout`, `use_pre_norm`, `use_residual` (DeepSetRank passes them; they are dropped).

### Target Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| `ModelAConfig` | `src/models/config.py` | Dataclass: `stat_dim`, `num_embeddings`, `embedding_dim`, `encoder_hidden`, `attention_heads`, `dropout`, `minutes_bias_weight`, `minutes_sum_min`, `fallback_strategy`, `attention_temperature`, etc. |
| Checkpoint | `best_deep_set.pt` | `{"model_state": ..., "model_config": ModelAConfig.to_dict()}` — architecture DNA stored with weights. |
| Inference | `load_models` | Build `DeepSetRank` from `ck["model_config"]` if present; else fall back to config + inferred `stat_dim`. |

### Implementation Steps

1. **Create `src/models/config.py`**
   - `ModelAConfig` dataclass with all DeepSetRank/SetAttention-relevant fields.
   - `from_config_dict(config: dict) -> ModelAConfig`.
   - `to_dict() -> dict` for checkpoint serialization.
   - `ModelAConfig.from_checkpoint(path)` — load checkpoint, return config from `model_config` or infer from `model_state` + defaults.

2. **Update `DeepSetRank`**
   - Add `@classmethod from_config(config: ModelAConfig) -> DeepSetRank`.
   - Keep `__init__` for backward compatibility; `from_config` builds from config object.

3. **Update `SetAttention`**
   - Add optional `temperature`, `input_dropout`, `use_pre_norm`, `use_residual` if they affect behavior; otherwise remove from DeepSetRank call and document as unused.

4. **Update `train_model_a.py`**
   - `_build_model(config, device, stat_dim_override)` → `_build_model(model_config: ModelAConfig, device)`.
   - Build `ModelAConfig` from `config["model_a"]` at training start; persist in checkpoint.

5. **Update `load_models` in `predict.py`**
   - If checkpoint has `model_config`, use it to construct `DeepSetRank`.
   - Else use current infer-from-checkpoint logic as fallback.

---

## 3. Attention Debugging as Integrated Hook

### Current State

- `_log_attention_debug_stats` in `train_model_a.py` is called manually:
  - After “not learning” early stop.
  - At end of `train_model_a` when `attention_debug` is True.
- It runs a full forward pass with hooks to capture `z`, then logs `attn_sum_mean`, `attn_max_mean`, grad norms, etc.
- No automatic detection of “entropy too low” or restart on collapse.

### Target Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| `AttentionMonitorCallback` | `src/training/callbacks.py` | Registers forward hook on `model.attn` output; computes entropy per batch; logs. |
| Entropy threshold | Config: `model_a.attention_entropy_min` | If mean entropy < threshold for N consecutive batches, trigger “collapse” action. |
| Collapse action | Config: `model_a.attention_collapse_action` | `"log"` | `"stop_and_restart"` | `"stop"`. |
| Restart | `ModelATrainer` | On collapse, optionally re-seed and restart training from epoch 0 (new init). |

### Implementation Steps

1. **Create `src/training/callbacks.py`**
   - `AttentionMonitorCallback(Callback)`:
     - `on_train_start`: Register `register_forward_hook` on `model.attn` (or wherever attention weights are produced). For `SetAttention`, the output is `(out, attn_w)`; we need access to `attn_w`.
     - Alternative: Use a custom forward that captures attention. `DeepSetRank.forward` returns `(score, Z, attn_w)`; the trainer already has access. So the callback can run as `on_batch_end` if the trainer passes `attn_w` to it.
     - Simpler: In `train_epoch`, after each batch, pass `attn_w` to a callback. `ModelATrainer` calls `callback.on_batch_end(batch, attn_w=attn_w)`.
   - Compute entropy: `H = -sum(p * log(p + eps))` over valid positions per team. Mean over batch.
   - If `H < entropy_min` for `collapse_patience` batches, set `collapsed = True`.
   - On collapse: invoke `trainer.on_attention_collapse()`.

2. **Integrate into `ModelATrainer`**
   - Add `AttentionMonitorCallback` when `model_a.attention_debug` is True.
   - Config: `attention_entropy_min` (default 0.5), `attention_collapse_patience` (default 5), `attention_collapse_action` (default `"log"`).
   - If `attention_collapse_action == "stop_and_restart"`: set new seed, re-init model, restart `fit` from epoch 0. Cap restarts (e.g., max 2).

3. **PyTorch hook approach (alternative)**
   - Register hook on a module that produces attention: `model.attn` returns `(out, attn_w)`. Use `register_forward_hook` with a closure that captures `attn_w` and pushes to a buffer. Callback reads buffer at `on_epoch_end`.
   - Prefer batch-level callback if we want per-batch collapse detection.

---

## 4. Attention Stability Metrics

### Current State

- No formal metric for “attention diversity” in evaluation.
- `_log_attention_debug_stats` logs `attn_sum_mean`, `attn_max_mean` but not variance or entropy as a saved metric.

### Target Architecture

| Metric | Definition | Stored Where |
|--------|------------|--------------|
| `attention_variance` | Variance of attention weights across valid roster positions, mean over teams in batch. | Training log, `train_history.json` |
| `attention_entropy` | Mean entropy of attention distribution per team. | Training log, `train_history.json` |
| `attention_max_weight` | Max attention weight per team (detect over-focus). | Training log |

- **Collapse:** variance ≈ 0 (uniform) or entropy very low (single spike).
- **Over-focus:** max_weight > 0.9 on many teams.

### Implementation Steps

1. **Add to `AttentionMonitorCallback`**
   - Each batch: compute `variance`, `entropy`, `max_weight` over valid positions per team.
   - Append to `trainer.history["attention_variance"]`, etc.

2. **Add `train_history.json`**
   - `ModelATrainer` writes `output_dir / "train_history.json"` with per-epoch: `train_loss`, `val_loss`, `attention_variance_mean`, `attention_entropy_mean`, `attention_max_weight_mean`.

3. **Visualization**
   - Optional: `scripts/plot_training.py` reads `train_history.json`, plots attention metrics over epochs. Can be Phase 2.

---

## 5. Championship Win-Rate Calibration

### Current State

- `scripts/5_evaluate.py` computes `brier_championship_odds` when `post_playoff_rank` exists for ≥16 teams.
- `brier_champion(champion_onehot, odds_pct)` compares predicted championship probabilities to actual champion (1) vs non-champion (0).
- Monte Carlo simulation in `src/inference/monte_carlo_championship.py` produces championship probabilities.
- Calibration is evaluated only when playoff data exists; no historical calibration analysis.

### Target Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| `evaluate_championship_calibration` | `src/evaluation/calibration.py` | Brier score, ECE (Expected Calibration Error) for championship probabilities. |
| Historical bins | — | For seasons with playoff outcomes: bin teams by predicted prob (e.g., 0–5%, 5–10%, …), compute actual win rate per bin. |
| Report | `eval_report.json` | Add `championship_calibration`: `{brier, ece, reliability_diagram_bins}`. |

### Implementation Steps

1. **Create `src/evaluation/calibration.py`**
   - `championship_brier(actual_champion_onehot, predicted_probs) -> float`.
   - `championship_ece(actual_champion_onehot, predicted_probs, n_bins=10) -> float`.
   - `reliability_diagram_data(actual_champion_onehot, predicted_probs, n_bins=10) -> list[dict]` (bin_low, bin_high, pred_mean, actual_rate, count).

2. **Integrate into `5_evaluate.py`**
   - When `post_playoff_rank` exists: extend `playoff_metrics` with `brier_championship_odds` (already present), `ece_championship`, `reliability_bins`.
   - If historical seasons are evaluated: aggregate calibration across seasons (e.g., pooled Brier, or mean per-season Brier).

3. **Monte Carlo consistency**
   - When `championship_odds_method == "monte_carlo"`, ensure `championship_odds` in predictions use Monte Carlo. Calibration evaluation then directly validates the Monte Carlo outputs against actual outcomes.

---

## 6. Comparative Explainability (RFX-Fuse Integration)

### Current State

- `scripts/5b_explain.py` produces `primary_contributors` (player + attention_weight) from DeepSet.
- No external validation of whether high-attention players are “important” by another method.

### Target Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| RFX-Fuse integration | `src/explain/rfx_fuse.py` (new) or optional script | Run RFX-Fuse (or similar) to get “historical archetypes” / “comps” per player or lineup. |
| Comparison | `scripts/5b_explain.py` or new `scripts/compare_explainability.py` | Compare high-attention DeepSet players vs RFX-Fuse-flagged players. Output: agreement rate, concordance metric. |

**Note:** RFX-Fuse may be an external tool. The plan should:
- Define the interface: input (roster stats, player IDs), output (player importance / archetype scores).
- If RFX-Fuse is a Python package: add optional dependency, implement adapter.
- If it’s external (CLI, API): document how to run it and parse outputs; provide a comparison script that reads both DeepSet and RFX-Fuse outputs.

### Implementation Steps

1. **Research RFX-Fuse**
   - Identify: PyPI package, GitHub repo, or external service.
   - Determine input format (stats, player IDs) and output format (e.g., per-player importance scores or archetype assignments).

2. **Design interface**
   - `ExplainabilityComparator(deep_set_contributors, rfx_fuse_scores) -> dict`:
     - `agreement`: % of top-k DeepSet players that are also in top-k RFX-Fuse.
     - `rank_correlation`: Spearman between DeepSet attention rank and RFX-Fuse importance rank.
     - `per_team_summary`: list of `{team_id, agreement, correlation}`.

3. **Implement adapter**
   - If RFX-Fuse is available: `src/explain/rfx_fuse.py` with `run_rfx_fuse(roster_stats, ...) -> dict[player_id, score]`.
   - Else: stub that returns empty; document “RFX-Fuse integration pending.”

4. **Wire into 5b_explain**
   - Optional `--compare-rfx-fuse` flag. When set, run RFX-Fuse (if available) and append `rfx_fuse_comparison` to explain output.

---

## 7. Strength of Schedule (SoS) Normalization

### Current State

- `config.sos_srs.enabled: false`.
- `src/features/team_context.py` has `build_team_context(..., sos_srs=...)`; `build_team_context_as_of_dates` can load SOS/SRS via `load_team_records_srs` when enabled.
- `Team_Records.csv` provides SRS (and possibly SOS).
- No “Net Rating Adjusted for Opponent” (or opponent-adjusted net rating) feature.

### Target Architecture

| Feature | Definition | Integration |
|---------|------------|-------------|
| `srs` | Simple Rating System (already in Team_Records) | `sos_srs.enabled: true` → join into team_context. |
| `sos` | Strength of Schedule (opponent average SRS) | From Team_Records or derived. |
| `net_rating_opp_adjusted` | Offensive/Defensive rating adjusted for opponent strength. | New: `off_rtg_adj = off_rtg - k * opp_avg_def_rtg`, etc. Or use existing SRS as proxy. |

**Note:** “Net Rating” is forbidden for Model B (leakage). “Net Rating Adjusted for Opponent” would be a derived feature that adjusts for schedule strength. The plan should ensure we don’t reintroduce leakage—e.g., use only pre-game opponent strength, not future results.

### Implementation Steps

1. **Enable SOS/SRS**
   - Set `sos_srs.enabled: true` in config when `Team_Records.csv` is available.
   - Verify `load_team_records_srs` and merge in `build_team_context_as_of_dates`.

2. **Add `net_rating_opp_adjusted` (or equivalent)**
   - Formula: e.g. `adj_off_rtg = off_rtg - 0.5 * (opp_avg_def_rtg - league_avg_def_rtg)`.
   - Requires: per-game opponent DefRtg (or similar) at as-of date. May need to compute from tgl/games.
   - Add to `EXTENDED_FEATURE_COLS` or new `OPP_ADJUSTED_COLS` when `sos_srs.enabled` and a new `opponent_adjusted.enabled` config is True.

3. **Document leakage controls**
   - Opponent stats must use only games before `as_of_date`. Add to CHECKPOINT_PROJECT_REPORT.md.

---

## Implementation Order

| Phase | Items | Dependencies |
|-------|-------|--------------|
| **A** | 1. Centralize Training Loops, 2. Configuration-Driven Architecture | None |
| **B** | 3. Attention Hook, 4. Attention Stability Metrics | Phase A (Trainer exists) |
| **C** | 5. Championship Calibration | Existing eval pipeline |
| **D** | 7. SoS Normalization | Existing feature pipeline; Phase A optional |
| **E** | 6. RFX-Fuse Integration | Research; can be parallel |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/training/base_trainer.py` | BaseTrainer, Callback base |
| `src/training/callbacks.py` | AttentionMonitorCallback |
| `src/models/config.py` | ModelAConfig dataclass |
| `src/evaluation/calibration.py` | Brier, ECE, reliability diagram |
| `src/explain/rfx_fuse.py` | RFX-Fuse adapter (or stub) |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/training/train_model_a.py` | ModelATrainer, use ModelAConfig, integrate callbacks |
| `src/training/train_model_b.py` | Optional ModelBTrainer wrapper |
| `src/training/train_stacking.py` | Optional StackingTrainer wrapper |
| `scripts/3_train_model_a.py` | Thin orchestration only |
| `scripts/4_train_model_b.py` | Thin orchestration only |
| `scripts/4b_train_stacking.py` | Thin orchestration only |
| `src/models/deep_set_rank.py` | `from_config(ModelAConfig)` |
| `src/inference/predict.py` | Load from model_config in checkpoint |
| `scripts/5_evaluate.py` | Add calibration metrics |
| `scripts/5b_explain.py` | Optional RFX-Fuse comparison |
| `config/defaults.yaml` | `attention_entropy_min`, `attention_collapse_*`, `opponent_adjusted` |
| `src/features/team_context.py` | `net_rating_opp_adjusted` when enabled |

---

## Testing Strategy

- Unit tests for `ModelAConfig` serialization, `BaseTrainer` early stopping, `AttentionMonitorCallback` entropy computation.
- Integration: run `3_train_model_a` with new Trainer; verify `best_deep_set.pt` contains `model_config`; run inference and confirm no load errors.
- Regression: compare eval metrics before/after refactor (train loops, config).

---

## References

- `docs/CHECKPOINT_PROJECT_REPORT.md` — attention collapse, fix_attention plan.
- `.cursor/plans/Plan.md` — project roadmap.
- `src/models/set_attention.py` — COLLAPSE_THRESHOLD, fallback logic.
- PyTorch `register_forward_hook` — for attention capture (standard API).

---

## Appendix: Attention Analysis and Hyperparameter Tracking

**Living document:** [docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md](../docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md)

This document provides:

- Detailed walkthrough of the first working inference (run_023) and attention weights
- Key inferences about what contributes most to winning (star-dominant vs. distributed)
- **Hyperparameter tracking framework:** params → metrics by model (A, XGB, RF, ensemble)
- **Conference-specific vs. league-wide:** East NDCG, West NDCG, per-conference attention patterns
- **Future run template:** Each sweep combo logs best config, metrics, and attention behavior

By project end, the Analysis document will contain a full matrix of: which parameters improve which metric, for which model, with conference breakdowns.
