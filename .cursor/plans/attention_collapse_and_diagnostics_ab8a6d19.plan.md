---
name: Attention collapse and diagnostics
overview: Plan that prioritizes attention collapse fixes, then adds systematic diagnostics for encoder/output constant, LR, gradients, relevance, numerics, stat_dim, and batch diversity; creates Attention_Report with references and investigation history; updates README; and saves to GitHub.
todos:
  - id: attention-diagnostics
    content: Add encoder z, pooled Z, score stats, and optional entropy to attention debug and not-learning path
    status: pending
  - id: encoder-inputs-logging
    content: Log encoder/Z variance and player_stats summary when attention_debug or not learning
    status: pending
  - id: lr-config
    content: Add model_a.learning_rate (and optional grad_clip_max_norm) to config and training
    status: pending
  - id: grad-norm-logging
    content: Log global and per-module gradient norms before clip when attention_debug or not learning
    status: pending
  - id: rel-stats
    content: Log rel min/max/mean/std and list length in build or first-batch debug
    status: pending
  - id: score-nan-logging
    content: Log score min/max/mean and NaN count before loss when attention_debug or not learning
    status: pending
  - id: stat-dim-assert
    content: Assert stat_dim matches batch player_stats.shape[-1]; log resolved stat_dim at model build
    status: pending
  - id: batch-diversity-log
    content: Log batch count and list length stats at start of Model A training
    status: pending
  - id: attention-report-doc
    content: Create .cursor/plans/Attention_Report.md with refs, investigation, improvements, new changes
    status: pending
  - id: readme-update
    content: Add Model A / attention collapse subsection and link to Attention_Report in README
    status: pending
  - id: git-save
    content: Git add, commit, push (manual or follow-up)
    status: pending
isProject: false
---

# Attention Collapse and Full Diagnostics Plan

## Scope

- **Primary:** Harden and diagnose attention collapse (SetAttention + optional research-backed regularization).
- **Secondary:** Add checks for encoder/inputs constant, learning rate, gradients, relevance, numerics, stat_dim, and batch diversity in the order you specified.
- **Deliverables:** New doc `[.cursor/plans/Attention_Report.md](.cursor/plans/Attention_Report.md)` (references, investigation, improvements tried, new changes); README update; Git add/commit/push.

---

## 1. Attention collapse (first priority)

**Current behavior:** [src/models/set_attention.py](src/models/set_attention.py) uses `MultiheadAttention` with query from masked mean of `x`; when attention on valid positions is below `COLLAPSE_THRESHOLD` (1e-6) or non-finite, fallback weights (minutes or uniform) are applied so gradients flow. [src/models/deep_set_rank.py](src/models/deep_set_rank.py) does `encoder(x) -> z`, then `SetAttention(z) -> pooled Z`, then `scorer(Z) -> score`.

**Planned changes:**

- **Diagnostics:** When `model_a.attention_debug` is true (and when "not learning" triggers), log not only attention stats and attn grad norm but also:
  - **Encoder outputs:** Per-team `z` (before attention) variance or L2 norm spread across teams in a batch (e.g. first batch). If encoder outputs are nearly identical across teams, the problem is upstream of attention.
  - **Pooled Z:** Per-team `Z` variance/spread; if Z is constant, scores will be constant regardless of attention.
  - **Scores:** Per-batch score min/max/mean (already partially implied by loss; make explicit for numerics check).
- **Optional hardening (research-backed):** Consider adding a small **attention entropy regularizer** (e.g. encourage entropy of attention weights on valid positions to stay above a minimum) to discourage collapse, inspired by "Stabilizing Transformer Training by Preventing Attention Entropy Collapse" (Zhai et al., 2023, arXiv:2303.06296). Implementation: in [src/models/set_attention.py](src/models/set_attention.py), optionally add a loss term or a soft constraint so that attention does not collapse to zero or uniform. Alternative: **spectral normalization** on attention projection layers (σReparam) to bound logits and stabilize training. Decision: start with diagnostics only; add entropy or σReparam in a follow-up if diagnostics confirm attention collapse.

**Files:** [src/models/set_attention.py](src/models/set_attention.py), [src/models/deep_set_rank.py](src/models/deep_set_rank.py), [src/training/train_model_a.py](src/training/train_model_a.py) (extend `_log_attention_debug_stats` and any "not learning" logging to include encoder `z`, pooled `Z`, and score stats from a single batch).

---

## 2. Encoder output / inputs almost constant

**Cause:** Encoder weights collapsed (e.g. to zero) or inputs `player_stats` too similar across teams (many zeros, same rolling stats, or bad scale).

**Check:**

- **Log encoder outputs (or Z) per team:** In `_log_attention_debug_stats` (or a new helper), run one batch and record per-team `z` (encoder output) and/or `Z` (pooled). Compute variance of `z` across the team dimension and variance of `Z` across teams; log these. If variance is near zero, encoder or inputs are constant.
- **Inspect batch inputs:** Optionally log `player_stats` min/max/mean per batch (or per team) when `attention_debug` is true—e.g. in [src/training/train_model_a.py](src/training/train_model_a.py) when logging diagnostics—to see if stats are constant or on a bad scale.

**Files:** [src/training/train_model_a.py](src/training/train_model_a.py) (extend debug logging); optionally [src/training/data_model_a.py](src/training/data_model_a.py) or train script to log one-batch input stats.

---

## 3. Learning rate / optimization

**Current:** [src/training/train_model_a.py](src/training/train_model_a.py) uses `torch.optim.Adam(model.parameters(), lr=1e-3)` (hardcoded). LR too small → tiny updates; LR too large → explosion/oscillation; with clipping + nan_to_num, effective gradients can become zero.

**Check:**

- Add **configurable LR** in [config/defaults.yaml](config/defaults.yaml) under `model_a` (e.g. `learning_rate: 0.001`) and use it in `train_model_a.py` and `train_model_a_on_batches` instead of hardcoded `1e-3`.
- Document in Attention_Report and README: try `1e-3` vs `1e-4` (and optionally `3e-4`) when loss is flat; check for NaN or explosion.

**Files:** [config/defaults.yaml](config/defaults.yaml), [src/training/train_model_a.py](src/training/train_model_a.py).

---

## 4. Gradient vanishing or clipping

**Current:** `clip_grad_norm_(model.parameters(), max_norm=1.0)` in [src/training/train_model_a.py](src/training/train_model_a.py) (train_epoch). When "not learning" triggers, only **attention** grad norm is logged (`attn_grad_norm`).

**Check:**

- **Log gradient norms before clipping:** Compute global grad norm (and optionally per-module norms for encoder vs attention vs scorer) **before** `clip_grad_norm_`, and log when `attention_debug` is true or when "not learning" triggers. If encoder grad norm is near zero, the issue is upstream of attention.
- Optionally make **max_norm** configurable (e.g. `model_a.grad_clip_max_norm: 1.0`) so you can try looser clipping (e.g. 5.0) to see if flat loss is due to over-clipping.

**Files:** [src/training/train_model_a.py](src/training/train_model_a.py).

---

## 5. Uninformative or degenerate relevance (rel)

**Current:** Relevance comes from [src/training/data_model_a.py](src/training/data_model_a.py): `rel_values = [31.0 - float(r) for r in final_rank] if final_rank else win_rates`. ListMLE in [src/models/listmle_loss.py](src/models/listmle_loss.py) orders by `rel` descending; if `rel` is constant or almost constant within lists, the gradient w.r.t. scores is weak or degenerate.

**Check:**

- When building or loading batches, **log rel stats per list:** min, max, mean, std of `rel` per batch (or per list), and list length. If std is near zero or list length is always 1–2, add a warning or skip such batches.
- Optionally in `train_epoch` / debug path: log `rel` stats for the first batch (shape, min, max, mean per row) when `attention_debug` is true.

**Files:** [src/training/data_model_a.py](src/training/data_model_a.py) (during build) and/or [src/training/train_model_a.py](src/training/train_model_a.py) (first-batch rel stats in debug).

---

## 6. Numerics (scores or gradients killed)

**Current:** [src/models/listmle_loss.py](src/models/listmle_loss.py) clamps scores to [-50, 50] and uses `nan_to_num`. [src/training/train_model_a.py](src/training/train_model_a.py) applies `nan_to_num` to score before ListMLE. If all scores sit at a clamp boundary or NaNs are zeroed, gradients can be zero.

**Check:**

- **Log score stats per batch** (when attention_debug or "not learning"): before loss, log score min, max, mean, and count of NaN/inf. If scores are always at ±50 or NaNs are frequent, document in Attention_Report and consider relaxing clamp or fixing upstream NaNs.

**Files:** [src/training/train_model_a.py](src/training/train_model_a.py) (in train_epoch and in debug logging).

---

## 7. stat_dim / architecture vs data

**Current:** [config/defaults.yaml](config/defaults.yaml) has `model_a.stat_dim: 21`. [src/training/data_model_a.py](src/training/data_model_a.py) uses `PLAYER_STAT_COLS_WITH_ON_OFF` + team continuity → 21 features; [src/features/build_roster_set.py](src/features/build_roster_set.py) returns `len(stat_cols)+1`. Model is built with `stat_dim_override` from first batch’s `player_stats.shape[-1]` in [src/training/train_model_a.py](src/training/train_model_a.py).

**Check:**

- **Assert stat_dim at batch build:** In `build_batches_from_db` (or in script 3 when loading batches), assert that `player_stats.shape[-1] == config["model_a"]["stat_dim"]` (or log a clear error) to avoid silent mismatch.
- **Log at model build:** When building the model from config, log the resolved `stat_dim` (from override or config) so runs are auditable.

**Files:** [src/training/data_model_a.py](src/training/data_model_a.py), [src/training/train_model_a.py](src/training/train_model_a.py), [scripts/3_train_model_a.py](scripts/3_train_model_a.py).

---

## 8. Batch size / diversity

**Current:** [config/defaults.yaml](config/defaults.yaml) has `training.max_lists_oof: 30`, `training.max_final_batches: 50`. Too few or too similar batches can make loss appear flat.

**Check:**

- **Log batch count and list lengths** at start of Model A training (script 3 and train_model_a): number of batches, and min/max/mean list length (teams per list). Document in Attention_Report: if very low batch count or very short lists, suggest increasing `max_lists_oof` / `max_final_batches` or diversifying dates.

**Files:** [scripts/3_train_model_a.py](scripts/3_train_model_a.py), [src/training/train_model_a.py](src/training/train_model_a.py).

---

## 9. New doc: `.cursor/plans/Attention_Report.md`

Create a single report that includes:

- **References:** Papers and online findings (e.g. Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse," arXiv:2303.06296; PyTorch LTR / ListNet/NDCG training tips; debugging flat loss checklists).
- **Problem statement:** Model A "not learning" (flat ListMLE loss); attention collapse → constant Z → constant scores → fixed loss.
- **Existing docs and plans:** Pointers to [docs/ATTENTION_AND_BATCHES.md](docs/ATTENTION_AND_BATCHES.md), [docs/MODEL_A_NOT_LEARNING_ANALYSIS.md](docs/MODEL_A_NOT_LEARNING_ANALYSIS.md), [.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md](.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md).
- **Improvements already made:** Minutes reweighting only when meaningful; uniform/minutes fallback when attention is zero or non-finite; attention_debug and _log_attention_debug_stats; early stop with "not learning" message; roster/season fallbacks; inference contributors_are_fallback.
- **New proposed changes:** Summary of sections 1–8 above (attention diagnostics and optional entropy/σReparam, encoder/Z/score logging, configurable LR, gradient norm logging, rel stats, score stats, stat_dim assert, batch count/length logging).
- **Checklist:** Encoder/inputs constant, LR, gradients, relevance, numerics, stat_dim, batch diversity—what to check and where.

---

## 10. README update

- Add a short **"Model A and attention collapse"** subsection (e.g. under Key Design Choices or a new "Troubleshooting" section) that:
  - Explains that Model A can stop early with "Model A is not learning" when loss is flat.
  - Links to [.cursor/plans/Attention_Report.md](.cursor/plans/Attention_Report.md) for full investigation, references, and diagnostics.
  - Mentions config knobs: `model_a.attention_debug`, `model_a.learning_rate`, `model_a.attention_fallback_strategy`, and training batch limits.

**File:** [README.md](README.md).

---

## 11. Save to GitHub

- **Add** all modified and new files (including `.cursor/plans/Attention_Report.md`).
- **Commit** with a clear message (e.g. "Attention collapse diagnostics and Attention_Report; configurable LR; gradient/encoder/rel/score logging").
- **Push** to the remote (e.g. `origin`).

No automated Git execution in this plan; perform add/commit/push manually or in a follow-up step after applying edits.

---

## Implementation order (suggested)

1. Add configurable `model_a.learning_rate` and optional `model_a.grad_clip_max_norm`; use them in training.
2. Extend `_log_attention_debug_stats` (and "not learning" path) with encoder `z`, pooled `Z`, score min/max/mean, global and per-module gradient norms before clip, and first-batch `rel` and `player_stats` summary.
3. Add stat_dim assertion at batch build and log resolved stat_dim at model build.
4. Log batch count and list length stats at start of Model A training.
5. Write `.cursor/plans/Attention_Report.md` with references, investigation, improvements tried, and new changes.
6. Update README with Model A / attention subsection and link to Attention_Report.
7. Git add, commit, push.

