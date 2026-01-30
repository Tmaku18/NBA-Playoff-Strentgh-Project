---
name: "Next steps: attention + playoff ranks"
overview: Fix playoff performance ranks/plot so playoff teams can receive ranks 1–16 (instead of everything landing in 17–30), and make Model A's attention outputs meaningful (non-zero, stable, and interpretable). Incorporates PyTorch MultiheadAttention behavior from Context7 docs.
todos:
  - id: fix-playoff-season-mapping
    content: Use date-range filtering instead of season-string matching in src/evaluation/playoffs.py; add helper is_game_in_season(game_date, season_cfg) and filter games by date range from config.
    status: pending
  - id: fix-query-masking
    content: In src/models/set_attention.py, compute query as masked mean (exclude padded positions) instead of including all positions.
    status: pending
  - id: increase-training-epochs
    content: Increase Model A training from 3 epochs to 10-30 epochs (or add early stopping) in src/training/train_model_a.py.
    status: pending
  - id: fix-playoff-team-detection
    content: Treat playoff teams as teams that appear in playoff logs (not playoff_wins > 0); keep ranks 1–16 for playoff teams.
    status: pending
  - id: add-playoff-sanity-check
    content: If playoff data looks missing/incomplete for season, return {} and suppress pred_vs_playoff_rank plot.
    status: pending
  - id: attention-debug
    content: Add diagnostics to confirm masks/minutes/attention sums and gradient flow for Model A attention.
    status: pending
  - id: attention-fallback-policy
    content: Make contributor fallback explicit and avoid emitting misleading 0.0 attention weights.
    status: pending
  - id: attention-minutes-reweight
    content: Adjust minutes reweighting so attention weights remain meaningful and non-degenerate.
    status: pending
  - id: ablation-stability
    content: Stabilize attention ablation to avoid non-finite scores by preventing degenerate masked inputs.
    status: pending
  - id: verify-end-to-end
    content: Re-run inference/evaluate on a completed season and confirm playoff ranks and attention outputs behave as expected.
    status: pending
isProject: false
---

# Next steps plan: make Model A attention work + fix playoff-rank plotting

## What we know from the current outputs

- In `outputs/run_010/predictions.json`, there are **30** `playoff_rank` entries but **16 are null** and only **14 are non-null** (these are currently all in the **17–30** range), which makes `pred_vs_playoff_rank.png` look like "everything is above 17".
- Model A attention/IG are not informative:
  - `primary_contributors[*].attention_weight` is effectively **0.0 everywhere**.
  - IG norms are **0.0000**.
  - Attention ablation can return **NaN**.

## Why `pred_vs_playoff_rank.png` is likely wrong

- In `[src/evaluation/playoffs.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\evaluation\playoffs.py)`, season filtering is done by deriving a `season` column as **year strings** (e.g. `"2024"`) via:
  - `dt.to_period("Y")` and `dt.year`
  - but the pipeline passes seasons like `"2023-24"`
- Result: playoff/regular-season filters can become empty → playoff wins and reg win% become empty/zero → everyone is treated like "lottery" and only ranks **17–30** are assigned.

## Why Model A attention is likely wrong/unhelpful

- The attention module itself is structurally correct: `[src/models/set_attention.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\models\set_attention.py)` uses `torch.nn.MultiheadAttention` with `batch_first=True` and `need_weights=True`, and returns weights shaped `(B, P)` after squeezing.
- Context7 confirms `MultiheadAttention.forward(..., need_weights=True, average_attn_weights=True)` returns weights shaped `(N, L, S)` (or `(N, num_heads, L, S)` when `average_attn_weights=False`). With query length `L=1` and key length `S=P`, your implementation should produce `(B, 1, P)` then squeeze to `(B, P)`.
- Therefore **all-zero weights** likely come from upstream data (minutes/masks/stats), post-processing, or model non-learning (flat loss).

## Implementation todos

- **fix-playoff-season-mapping** (HIGH PRIORITY)
  - Root cause: `get_reg_season_win_pct()` derives season as `dt.year` → `"2024"`, but pipeline passes `"2023-24"` → empty filter → all teams get 0 wins → ranks 17-30 only.
  - Fix: Use **date-range filtering** instead of string matching:
    - Add helper `is_game_in_season(game_date, season_start, season_end) -> bool`
    - In `get_playoff_wins()` and `get_reg_season_win_pct()`, filter games where `season_start <= game_date <= season_end` using dates from config
    - This avoids format mismatches entirely

- **fix-query-masking** (HIGH PRIORITY - NEW)
  - Root cause: In `[src/models/set_attention.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\models\set_attention.py)`, query is computed as `q = x.mean(dim=1)` which includes **padded positions** (zeros), diluting the query.
  - Fix: Compute masked mean excluding padded positions:
    ```python
    # Instead of: q = x.mean(dim=1, keepdim=True)
    mask_expanded = key_padding_mask.unsqueeze(-1).float()
    x_masked = x * (1 - mask_expanded)
    valid_count = (1 - mask_expanded).sum(dim=1, keepdim=True).clamp(min=1)
    q = x_masked.sum(dim=1, keepdim=True) / valid_count
    ```

- **increase-training-epochs** (HIGH PRIORITY - NEW)
  - Root cause: Training shows flat loss `27.8993` across all 3 epochs—insufficient iterations for convergence.
  - Fix in `[src/training/train_model_a.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\training\train_model_a.py)`:
    - Change `for epoch in range(3)` to `for epoch in range(20)` (or make configurable via `config["model_a"]["epochs"]`)
    - Optionally add validation-based early stopping

- **fix-playoff-team-detection**
  - Update `compute_playoff_performance_rank()` to define "playoff teams" as **teams that appear in the filtered playoff logs** (played at least one playoff game), not `playoff_wins > 0`.
  - This prevents swept teams (0 wins) from being incorrectly categorized as "lottery".
- **add-playoff-data-sanity-check**
  - If, after filtering, there are fewer than 16 playoff participants (or otherwise clearly incomplete), return `{}` so inference won't create a misleading `pred_vs_playoff_rank.png`.
- **validate-playoff-rank-range-and-coverage**
  - Re-run inference for a season with playoff data and confirm `playoff_rank` includes **some 1–16 values** and totals 30 assigned ranks (or cleanly suppressed if data is missing).
- **attention-debug-instrumentation**
  - Add lightweight diagnostics (log once per run) in:
    - `[src/training/train_model_a.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\training\train_model_a.py)` (during training), and
    - `[src/inference/predict.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)` (during inference),
  - Specifically capture:
    - fraction of teams with **all positions masked** in `key_padding_mask`
    - `minutes` min/mean/max for non-masked players
    - attention weight min/mean/max and `sum(attn_w)` per team (should be ~1 for any non-empty roster)
    - gradient norms for attention parameters (confirm learning signal)
- **fix-attention-weight-postprocessing**
  - In `[src/inference/predict.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)`, stop silently producing 0.0 contributor weights by tightening the fallback behavior:
    - only use fallback when weights are finite, and
    - include a clear flag/field (e.g. `contributors_are_fallback=true`) when fallback is used so interpretations aren't misleading.
- **make-attention-weights-meaningful**
  - If diagnostics show `minutes` is collapsing weights to ~0, adjust `[src/models/set_attention.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\models\set_attention.py)` minutes reweighting:
    - avoid `clamp(0,1)` on minutes if minutes are already normalized, and
    - normalize minutes across valid players and apply it as a *soft bias* (or return raw attention weights separately).
- **stabilize-attention-ablation**
  - In `[src/viz/integrated_gradients.py](c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\viz\integrated_gradients.py)`, handle cases where masking yields non-finite outputs by ensuring the ablated forward pass never receives an all-masked/all-zero degenerate input (masking strategy should preserve at least one valid player per team).

## Verification checklist

- `outputs/run_0XX/pred_vs_playoff_rank.png` shows **both** regions (1–16 and 17–30) for completed seasons, or is **omitted** when playoff data is unavailable.
- `outputs/run_0XX/predictions.json` contains `primary_contributors` with **non-zero, finite** attention weights that sum to ~1 over the roster for at least some teams.
- Training logs show Model A loss changing over epochs and non-zero gradient norms for attention parameters.
