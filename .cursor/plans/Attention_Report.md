# Attention Collapse and Model A Diagnostics Report

## References and research

- **Zhai et al. (2023), "Stabilizing Transformer Training by Preventing Attention Entropy Collapse"**  
  [arXiv:2303.06296](https://arxiv.org/abs/2303.06296) | [PDF](https://arxiv.org/pdf/2303.06296)  
  - Low attention entropy (concentrated attention) correlates with training instability (oscillating loss, divergence).  
  - **σReparam:** reparameterize linear layers as W_hat = (γ / σ(W)) * W, with σ(W) from power iteration and γ learnable (init 1). Bounds spectral norm of attention logits and helps prevent entropy collapse.  
  - Code: [apple/ml-sigma-reparam](https://github.com/apple/ml-sigma-reparam).

- **PyTorch spectral normalization:** [torch.nn.utils.spectral_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html) — power iteration for σ(W); we use σReparam (γ learnable) rather than fixed SN.

- **Listwise ranking / flat loss:** ListNet/ListMLE; NDCG training tips (batch size, LR schedules). PyTorch LTR; debugging flat loss: LR, gradient norms, data/relevance variation, numerics (clamping/NaN), stat_dim/architecture, batch diversity.

---

## Problem statement

**Model A "not learning":** Training stops when train loss does not improve for several consecutive epochs. Typical chain:

1. **Attention collapse** (all-zero or nearly uniform weights on valid positions).  
2. **Constant pooled Z** — same (or nearly same) vector per team.  
3. **Constant scores** — ListMLE loss depends only on list length, not parameters.  
4. **Flat loss** → no learning.

Even with non-zero attention, **encoder output nearly constant** (no team signal) or **uninformative relevance** / **numerics** / **gradient clipping** can cause flat loss.

---

## Existing docs and plans

- [docs/ATTENTION_AND_BATCHES.md](docs/ATTENTION_AND_BATCHES.md) — NaN/collapse, all-masked batches, roster fallbacks.  
- [docs/MODEL_A_NOT_LEARNING_ANALYSIS.md](docs/MODEL_A_NOT_LEARNING_ANALYSIS.md) — Run summary, cause (attention → constant Z → constant scores → fixed loss), next steps.  
- [.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md](.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md) — Minutes reweighting, fallback policy, evaluation/pipeline fixes.

---

## Improvements already made

- **SetAttention:** Minutes reweighting only when minutes are meaningful; uniform/minutes fallback when attention on valid positions is below threshold or non-finite so gradients can flow.  
- **Attention debug:** `model_a.attention_debug` and `_log_attention_debug_stats`: mask/minutes, attn sums, attn grad norm.  
- **Early stop:** "Not learning" message and optional debug dump when train loss stops improving.  
- **Roster/season:** Fallbacks when roster lookup returns empty; skip all-masked batches.  
- **Inference:** `contributors_are_fallback` when attention is not usable.

---

## New changes (this implementation)

### 1. σReparam on attention projections (research-based, mandatory)

- **Reference:** Zhai et al., arXiv:2303.06296.  
- **Implementation:**  
  - [src/models/spectral_reparam.py](src/models/spectral_reparam.py): `SpectralReparamLinear` — effective weight = (γ / σ(W)) * W; σ(W) from one power iteration per forward (in `no_grad` so buffers are not in the autograd graph); γ learnable, init 1.  
  - [src/models/set_attention.py](src/models/set_attention.py): SetAttention uses custom multi-head attention with **σReparam on Q, K, V** (q_proj, k_proj, v_proj as `SpectralReparamLinear`); out_proj remains standard Linear.  
- **Goal:** Bound spectral norm of attention logits to stabilize training and reduce attention entropy collapse.

### 2. Configurable learning rate and gradient clipping

- **Config** ([config/defaults.yaml](config/defaults.yaml)):  
  - `model_a.learning_rate: 0.001`  
  - `model_a.grad_clip_max_norm: 1.0` (try e.g. 5.0 if flat loss may be due to over-clipping).  
- **Training** ([src/training/train_model_a.py](src/training/train_model_a.py)): Adam uses `learning_rate`; `clip_grad_norm_` uses `grad_clip_max_norm`.

### 3. Extended debug logging (when `model_a.attention_debug` is true)

- **First batch in train_epoch:**  
  - **rel:** shape; per-row min, max, mean.  
  - **player_stats:** shape, min, max, mean.  
  - **Grad norm before clip** (first batch only) and `max_norm` used.  
- **When "not learning" (and when debug runs at end):** `_log_attention_debug_stats` now logs:  
  - **Encoder z:** variance of encoder output across teams (via forward hook).  
  - **Z variance:** variance of pooled Z across teams.  
  - **Scores:** min, max, mean, NaN count.  
  - **Grad norms:** global, encoder, attention, scorer.  
  - **player_stats:** shape, min, max, mean.  
  - (Existing: attn sums, minutes, attn_grad_norm.)

### 4. Checklist for flat loss (order of suspicion)

| Suspect | What to check / try |
|--------|----------------------|
| Attention collapse | σReparam (done); debug: attn_sum_mean, Z_var, z_var. |
| Encoder/inputs constant | Log z_var, Z_var, player_stats min/max/mean. |
| Learning rate | Try `learning_rate: 0.0001` or `0.001`; watch for NaN/explosion. |
| Gradients | grad_norm_global, grad_norm_enc, grad_norm_attn, grad_norm_scorer; grad norm before clip. |
| Over-clipping | Increase `grad_clip_max_norm` (e.g. 5.0). |
| Relevance | rel shape and per-row min/max/mean; ensure variation and list length > 1. |
| Numerics | score_min, score_max, score_nan; ListMLE clamp ±50. |
| stat_dim | Ensure `model_a.stat_dim` matches `player_stats.shape[-1]` from batches. |
| Batch diversity | Increase `max_lists_oof` / `max_final_batches`; check batch count and list lengths. |

---

## File summary

- **New:** `src/models/spectral_reparam.py` (σReparam linear layer).  
- **Modified:** `src/models/set_attention.py` (σReparam on Q/K/V; custom multi-head); `src/training/train_model_a.py` (LR, grad_clip_max_norm, rel/player_stats/grad logging, extended _log_attention_debug_stats); `config/defaults.yaml` (learning_rate, grad_clip_max_norm).  
- **This report:** `.cursor/plans/Attention_Report.md`.
