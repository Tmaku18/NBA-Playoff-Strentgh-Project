# Notion update summary (copy into Notion — use 🤖 icon)

**Date:** 2026-01-28  
**Repo:** NBA Playoff Strength Project

---

## Tasks / goals to add or update

- [x] **Save performance trajectory doc to repo** — Done. `Performance_trajectory_and_hyperparameters.md` added; pushed to GitHub.
- [ ] **Review optimal hyperparameters** — Model A: epochs 28 (NDCG) or 15 (Spearman); val_frac 0.25. Model B: XGB 4/0.08/250, RF 200/12/5 (ranking-first).
- [ ] **Re-check roster logic** — Verify latest-team / as_of_date so no wrong-team players (e.g. Simons on Boston).
- [ ] **Re-check playoff rank** — Use date-range filtering in `playoffs.py` if playoff_rank still wrong.
- [ ] **Optional: align Model B to sweep** — Try XGB max_depth=4, lr=0.08, n_estimators=250; re-run eval.
- [ ] **Optional: query masking in set_attention** — If attention still zero after 20+ epochs, add masked mean for query.

---

## Run metrics (current)

| Metric           | Value  |
|------------------|--------|
| NDCG             | 0.64   |
| Spearman         | 0.72   |
| ROC-AUC upset    | 0.63   |
| MRR              | 0.0    |

*Source: `outputs/eval_report.json` (latest run).*

---

## Goals (high level)

1. **Ranking quality** — Keep NDCG/Spearman; consider epochs=28 and val_frac=0.25 from sweeps.
2. **Explainability** — Fix attention (query masking) and roster so primary_contributors and IG are meaningful.
3. **Playoff evaluation** — Fix season mapping so playoff_rank 1–16 and playoff_metrics in eval_report when data exists.
4. **Reproducibility** — Per-run eval_report copy or metrics log so runs are comparable.

---

*Use 🤖 as page/database icon for this project (see .cursor/rules/notion-robot-icon.mdc).*
