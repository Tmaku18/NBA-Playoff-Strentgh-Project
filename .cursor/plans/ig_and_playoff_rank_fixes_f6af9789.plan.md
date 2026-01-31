---
name: IG and playoff rank fixes
overview: Address IG batching, playoff rank bounds, manifest portability, and run_008 quality issues (NaN attention weights, rank scale mismatch). Incorporate high/medium/low priority improvements with impact analysis, fixes, and tests.
todos: []
isProject: false
---

# IG, Playoff Rank, and Output Quality Improvements

This plan incorporates the requested improvements with explicit issue descriptions, code locations, impact, and fixes. It also reflects the latest decisions:

- **Eval remains in `scripts/5_evaluate.py**` (no eval_report inside inference).
- **Roster fix uses “latest-team roster”** (most recent team per player as of `as_of_date`).
- **IG fix uses closure + tensor expansion** (no `additional_forward_args` refactor).
- **IG results should also be embedded into `predictions.json`.**

---

## High Priority (critical bugs)

### 1) Integrated Gradients batching mismatch

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\viz\integrated_gradients.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\viz\integrated_gradients.py)
- **Impact:** IG runtime failure; no attributions produced (e.g., “Expected size 1 but got size 50”).
- **Fix:** In `ig_attr`, inside the forward wrapper, expand `emb_indices`, `minutes`, `key_padding_mask` to `B = stats.shape[0]` so Captum’s batched inputs match `(B, P)` / `(B, P, S)` for all tensors.

### 2) Actual rank overwriting across multiple lists

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- **Impact:** `analysis.actual_rank` can be overwritten when multiple lists/dates are used, producing inconsistent evaluation/classification.
- **Fix:** Restrict `target_lists` to a single date (latest) and/or guard against overwrites (only set `team_id_to_actual_rank` if not already assigned).

### 3) Conference plot fallback uses global rank

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- **Impact:** `pred_vs_actual.png` can mix 1–30 ranks on 1–15 axes, misleading the East/West plots.
- **Fix:** Do **not** fall back to global rank. If `conference_rank` is missing, either compute it for all teams or skip plotting that team.

### 4) NaN attention weights → invalid JSON

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- **Impact:** `predictions.json` contains `NaN` values, which is invalid JSON and breaks strict parsers; roster explainability unusable.
- **Fix:** Sanitize `attn_weights` via `np.nan_to_num(..., nan=0.0)` and skip non‑finite `w`. Write JSON with `allow_nan=False` after sanitization to prevent regressions.

### 5) Historical players on wrong teams (roster integrity)

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\features\build_roster_set.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\features\build_roster_set.py), [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\training\data_model_a.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\training\data_model_a.py), [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- **Impact:** Attributions show players on teams they no longer play for; rosters are historically contaminated.
- **Fix (chosen):** “Latest‑team roster” — compute each player’s latest team as of `as_of_date` from `pgl` (most recent game per player), then build each team roster using only players whose latest team == team_id; aggregate minutes from current season only. This prevents traded players appearing on old teams.

---

## Medium Priority (clarifications and quality)

### 6) Rank scale mismatch (conference vs global)

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py), [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5_evaluate.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5_evaluate.py)
- **Impact:** `classification`, NDCG/Spearman, and upset detection mix 1–15 vs 1–30 scales.
- **Fix (recommended):** Add `analysis.actual_global_rank` (1–30) based on league‑wide win_rate; use that for global evaluation and classification. Keep `analysis.actual_rank` for conference‑only uses.

### 7) Ensemble agreement when some models missing

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- **Impact:** Agreement can be misleading when a model is missing (zeroed scores still ranked).
- **Fix:** Compute ranks only for models that are present and have non-null predictions; compute spread using those only. Then apply a scaled threshold (e.g., High/Medium/Low based on `n_teams`).

### 8) Playoff‑rank plot axis order

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- **Impact:** If axes are inverted vs intended convention, interpretation suffers.
- **Fix:** Confirm convention (x=actual playoff rank, y=predicted global rank). If needed, swap axes and label accordingly.

### 9) Manifest db_path portability

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\2_build_db.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\2_build_db.py), [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\data\manifest.json](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\data\manifest.json)
- **Impact:** Absolute `db_path` reduces portability.
- **Fix:** Write `manifest["db_path"]` as relative to `ROOT` when possible; update current manifest to relative path.

---

## Low Priority (features / polish)

### 10) IG outputs embedded into predictions.json

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py), [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5b_explain.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5b_explain.py)
- **Impact:** IG attribution not present in inference outputs.
- **Fix:** Add a lightweight IG summary (e.g., top‑k players by attribution norm) into `predictions.json`. To limit runtime, compute for a configurable subset (default: top‑1 team per conference) and make scope configurable.

### 11) True strength score percentile endpoints

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- **Impact:** Percentiles never reach 0.0/1.0 due to `(rank)/(n+1)` formula.
- **Fix:** If strict [0,1] is desired, use `(rank‑1)/(n‑1)` (guard `n>1`).

### 12) Playoff rank edge cases (missing data / ties)

- **Location:** [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\evaluation\playoffs.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\evaluation\playoffs.py)
- **Impact:** Missing reg‑season data or ties can yield unstable ordering.
- **Fix:** Add deterministic tie‑break (e.g., team_id) and handle missing reg‑season win% explicitly.

---

## Completed

- **Playoff rank cap** already fixed in [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\evaluation\playoffs.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\evaluation\playoffs.py) by capping playoff to top 16 and lottery to top 14.

---

## Testing Strategy

1. Run `python -m scripts.5b_explain` and confirm IG completes without size mismatch.
2. Run `python -m scripts.6_run_inference` and verify:
  - `predictions.json` contains no `NaN` values (JSON is valid).
  - `playoff_rank` stays 1–30.
  - `pred_vs_actual.png` uses 1–15 scales only.
  - Roster contributors only show players on current teams.
3. Run `python -m scripts.5_evaluate` and confirm metrics use the intended rank scale (global vs conference).
4. Verify `data/manifest.json` contains a relative `db_path`.

