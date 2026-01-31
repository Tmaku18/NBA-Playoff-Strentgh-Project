---
name: Fix run_009 outputs
overview: Address empty roster contributors and IG NaNs, clarify playoff-rank plotting, and rename output fields for clarity while aligning evaluation/classification to new EOS rank names (breaking change as requested).
todos:
  - id: fix-contributors
    content: Populate primary_contributors + handle empty rosters and attention fallback
    status: completed
  - id: ig-nans
    content: Guard IG NaNs in inference and 5b
    status: completed
  - id: playoff-plot
    content: Drop null playoff_rank from plot + document source
    status: completed
  - id: rename-fields
    content: Rename output fields and update consumers (breaking)
    status: completed
  - id: classification-labels
    content: Fix classification wording to match delta direction
    status: completed
  - id: update-eval-readme
    content: Update evaluation + README for new field names
    status: completed
isProject: false
---

# Fix run_009 output issues + rename fields

## Scope decisions captured

- **Rename fields and remove old names** (breaking change).
- **Drop null playoff_rank teams** from `pred_vs_playoff_rank.png` (no x=0 points).

## Phase 1 — Explainability + roster contributors

### 1) Fix empty `primary_contributors`

**Files:**

- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\features\build_roster_set.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\features\build_roster_set.py)
- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\training\data_model_a.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\training\data_model_a.py)

**Plan:**

- Add logging/guard when `latest_team_map` filtering yields an empty roster; fall back to **season‑scoped roster without latest‑team filter** for that team/date to avoid empty contributor lists.
- Ensure attention weights are normalized/usable: if all weights are <=0 or sum to 0 after sanitization, take **top‑k by raw value** (still finite) so contributors are populated.
- Confirm `player_ids_per_team` length matches attention vector length; truncate safely if mismatched.

### 2) IG NaNs (inference + 5b)

**Files:**

- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5b_explain.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5b_explain.py)

**Plan:**

- Apply `torch.nan_to_num` to IG attributions before norm calculations; if still non‑finite, skip IG output and emit a clear message.
- Ensure inference IG only runs when model outputs are finite; add a guard that checks model forward is finite on that batch before calling IG.

## Phase 2 — Playoff rank plot + source clarity

### 3) Explain `playoff_rank` source in README + fix plot nulls

**Files:**

- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\README.md](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\README.md)

**Plan:**

- In `pred_vs_playoff_rank.png` generation, **filter to teams with `playoff_rank != null**` (no x=0 points).
- Add a README note that playoff rank is computed from playoff wins (then reg‑season win%), using `playoff_team_game_logs` and `playoff_games`, and is absent if playoff data is missing or season mapping fails.

## Phase 3 — Field renames (breaking change) + classification wording

### 4) Rename output fields

**Files:**

- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)
- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5_evaluate.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\scripts\5_evaluate.py)
- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\README.md](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\README.md)

**Renames (remove old fields):**

- `prediction.predicted_rank` → `prediction.predicted_strength`
- `prediction.true_strength_score` → `prediction.ensemble_score`
- `prediction.true_strength_score_100` → `prediction.ensemble_score_100`
- `analysis.actual_rank` → `analysis.EOS_conference_rank`
- `analysis.actual_global_rank` → `analysis.EOS_global_rank`

**Plan:**

- Update inference output keys, all plots and evaluation to read new names.
- Update README output descriptions and evaluation notes to use new field names.

### 5) Fix classification wording (label inversion)

**Files:**

- [C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py](C:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project\src\inference\predict.py)

**Plan:**

- Keep delta definition `delta = EOS_global_rank - predicted_strength`.
- Replace labels so:
  - `delta < 0` → **“Under‑ranked by X slots”** (model ranked them worse than reality)
  - `delta > 0` → **“Over‑ranked by X slots”** (model ranked them better than reality)
- Rename classification strings accordingly to avoid confusion.

## Phase 4 — Evaluation + documentation updates

### 6) Update `scripts/5_evaluate.py` to new names

**Plan:**

- Use `analysis.EOS_global_rank` (fallback `analysis.EOS_conference_rank`) and `prediction.predicted_strength` in evaluation.
- Update notes in `eval_report.json` to reference EOS names.

### 7) README updates

**Plan:**

- Update outputs section with renamed fields.
- Add note explaining why some playoff ranks are null.

---

## Testing strategy

- Run `python -m scripts.6_run_inference` and verify:
  - `predictions.json` uses new field names only.
  - `primary_contributors` populated for most teams.
  - `pred_vs_playoff_rank.png` has no x=0 points.
- Run `python -m scripts.5_evaluate` and verify metrics read new field names.
- Spot-check a few teams (e.g., Boston) for corrected classification wording.

