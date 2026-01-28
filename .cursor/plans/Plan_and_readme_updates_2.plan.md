---
name: Plan and README Updates
overview: "Update the plan and README to incorporate your new risk analysis: remove Net Rating head (Option B), switch Model A training to game-level lists, fix roster construction leakage, align interpretability methods, and tighten data/metrics/reproducibility guidance."
todos:
  - id: plan-net-rating
    content: Remove any Net Rating head references; keep NR out of Model B/stacking/eval
    status: completed
  - id: plan-game-level
    content: Switch Model A training to game-level lists and define targets
    status: completed
  - id: plan-roster-leak
    content: Enforce roster selection as-of date; fix embedding order/inclusion
    status: completed
  - id: plan-explain
    content: Limit SHAP to Model B; IG/ablation for Model A
    status: completed
  - id: plan-rolling
    content: Fix DNP handling and minutes weighting guidance
    status: completed
  - id: plan-true-strength
    content: Define true_strength_score scale/meaning explicitly
    status: completed
  - id: plan-db
    content: Add DuckDB or indexing/pre-agg guidance for large data
    status: completed
  - id: plan-repro
    content: Add seeding, persist OOF, and baseline comparisons in eval
    status: completed
  - id: readme-sync
    content: Mirror all plan updates in README
    status: completed
isProject: false
---

# Plan Update for Risk Fixes

## Scope

Update the plan and README to reflect your chosen fixes: **Option B (drop Net Rating head)** and **game‑level ListMLE training**, plus the additional leakage, explainability, data, and evaluation clarifications. Primary files:

- [C:\Users\tmakucursor\worktrees\NBA_Playoff_Strentgh_Project\mnucursor\plans\Plan.md](C:\Users\tmaku.cursor\worktrees\NBA_Playoff_Strentgh_Project\mnu.cursor\plans\Plan.md)
- [C:\Users\tmakucursor\worktrees\NBA_Playoff_Strentgh_Project\mnu\README.md](C:\Users\tmaku.cursor\worktrees\NBA_Playoff_Strentgh_Project\mnu\README.md)

## Planned Changes

### 1) Net Rating leakage resolution (Option B)

- Remove any remaining mention of a **Net Rating head** from Model A.
- Keep Net Rating out of Model B inputs and out of stacking signals entirely.
- Ensure evaluation never references Net Rating alignment or MAE.

### 2) Game‑level ListMLE training to fix small‑N lists

- Replace **season‑level list training** with **game‑date (or week) lists**.
- Define list construction: roster/rolling stats **as of t−1**, ranking target based on **standings to date** (or win‑rate to date), not season‑end totals.
- Keep **season‑end evaluation** separately as the final test for ranking performance.

### 3) Roster construction leakage fix

- Update roster selection and player inclusion to **as‑of date** (not full‑season minutes).
- Apply the same rule to player ordering, embedding index inclusion, and top‑N roster selection.

### 4) Explainability tooling correction

- Explicitly restrict **SHAP** to **Model B only** (RF/XGBoost).
- Use **Integrated Gradients (Captum)** or **permutation/ablation** for Model A.
- Clarify in both documents that **attention ≠ explanation**, validated via ablation.

### 5) Minutes weighting vs MP feature (avoid double‑count)

- Add a rule to use **either** explicit minutes weighting **or** MP as a feature, not both.
- Document which path the implementation should take (pick a default in the plan).

### 6) Rolling stats for DNP handling

- Switch rolling stats to **per‑game averages over games played**, not zeros for DNP.
- Track availability as a separate feature (e.g., fraction of games played in window).

### 7) True Strength score definition

- Define `true_strength_score` explicitly (e.g., percentile within conference, calibrated probability, or softmax‑derived score), and update outputs in the plan/README.

### 8) Storage/DB guidance

- Add a note that **DuckDB** is preferred for large joins, or that SQLite should use **pre‑aggregation + indexing** (game_id, player_id, team_id) if retained.

### 9) Reproducibility and diagnostics

- Require **random seeds** (torch/numpy/sklearn) and **persist OOF predictions**.
- Add **simple baselines** (rank by SRS/Net Rating) to evaluation section.

## Key Snippets to Update

- Model A training section: swap season‑level ListMLE for game‑level lists, update target definition.
- Data pipeline roster section: enforce **as‑of date** roster inclusion.
- Explainability section: SHAP only for Model B, IG/ablation for Model A.
- Evaluation section: remove efficiency alignment; add baselines + Upset ROC‑AUC.

## Notes from Context7

- scikit‑learn docs confirm **LogisticRegression** provides calibrated probabilities via `predict_proba` and is compatible with **Brier** and **ROC‑AUC** metrics.
- Captum Integrated Gradients is the standard method for Model A attribution in place of SHAP.
- PyTorch supports variable‑length batching with custom `collate_fn` (useful for game‑level lists with masks).

## Next Step

Once you approve, I'll make the plan and README edits in the two files above to match the selected options.
