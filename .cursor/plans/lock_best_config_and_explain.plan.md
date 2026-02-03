---
name: Lock Best Config and Explain
overview: Lock the best sweep config, ensure sweep collects all metrics and evaluations correctly, run explain on the best run after each sweep (required), update README and ANALYSIS.md with run_021/022 insights and multi-target sweep strategy. Documents what is done vs what remains.
todos:
  - id: explain-config
    content: Add --config to 5b_explain.py so explain can run on a sweep best combo
    status: completed
  - id: run-explain-best
    content: Run 5b_explain on best combo after each sweep (required; implemented in sweep_hparams.py)
    status: completed
  - id: readme-analysis
    content: Update README and ANALYSIS.md (run_021/022 insights, sweep strategy, explain --config)
    status: completed
isProject: false
---

# Lock Best Config and Explain

## 1. Current state (done)

### Sweep metrics and evaluation

- `**_collect_metrics**` in `scripts/sweep_hparams.py` reads `eval_report.json` and collects:
  - **Ensemble / model_a / xgb / rf:** All scalar test metrics (ndcg, spearman, mrr_top2, mrr_top4, roc_auc_upset, rank_mae_pred_vs_playoff, rank_rmse_pred_vs_playoff, standings-vs-playoff when present).
  - **Playoff metrics:** Flattened from nested `playoff_metrics` (e.g. `test_metrics_ensemble_playoff_spearman_pred_vs_playoff_rank`, `test_metrics_ensemble_playoff_ndcg_at_4_final_four`, `test_metrics_ensemble_playoff_brier_championship_odds`).
  - **By-conference:** `test_metrics_by_conference_E_ndcg`, `test_metrics_by_conference_E_spearman`, `test_metrics_by_conference_W_ndcg`, `test_metrics_by_conference_W_spearman`.
- **Grid and Optuna:** Both paths call `_collect_metrics(combo_out / "eval_report.json")` and merge those metrics into each result row. Optuna rows therefore get the same full metrics as grid rows.
- **CSV columns:** Columns are the **union of all keys** across result rows (not just `results[0].keys()`), so metric columns are never dropped when the first Optuna trial fails or when some runs lack playoff metrics.
- **Summary:** `sweep_results_summary.json` includes `best_by_spearman`, `best_by_ndcg`, `best_by_rank_mae`, `best_by_playoff_spearman`, and for Optuna `best_optuna_trial`.

### Eval report path

- Evaluate writes the main report to `out_dir / "eval_report.json"` (i.e. `config.paths.outputs / "eval_report.json"`). The sweep sets `paths.outputs` to each combo’s `combo_dir/outputs`, so the sweep reads `combo_out / "eval_report.json"` correctly.

### Config for comparable evaluation

- To have the sweep evaluate against **playoff (eos_final_rank)** instead of standings, set in `config/defaults.yaml`:
  - `inference.require_eos_final_rank: true` (if your inference script enforces it).
- Ensure the DB and inference pipeline populate `post_playoff_rank` / EOS playoff data for test seasons so `playoff_metrics` and `best_by_playoff_spearman` are meaningful.

---

## 2. Needed changes (clarified)

### 2.1 Add `--config` to `scripts/5b_explain.py`

**Why:** To run explain on the **best sweep combo**, we must point 5b_explain at that combo’s outputs. Right now 5b_explain only reads `config/defaults.yaml` and uses `config["paths"]["outputs"]`, so it cannot use a sweep combo’s output directory.

**Change:**

- Add an optional CLI argument: `--config PATH` (path to a config YAML).
- If `--config` is provided, load that file and use its `paths.outputs` (and other settings). If not provided, keep current behavior: load `ROOT / "config" / "defaults.yaml"`.
- Resolve `paths.outputs` and `paths.db` the same way as other scripts (absolute vs relative to ROOT).

**Result:** After a sweep, you can run explain on the best combo with:

```bash
python -m scripts.5b_explain --config "outputs3/sweeps/<batch_id>/combo_0002/config.yaml"
```

Models are expected at `paths.outputs/best_deep_set.pt` and `paths.outputs/rf_model.joblib` (sweep writes these under each combo’s `outputs/`).

**File:** `scripts/5b_explain.py` — add `argparse` for `--config`; if set, use that path instead of `ROOT / "config" / "defaults.yaml"` for loading config.

### 2.2 Run explain on best run after sweep (required)

- After a sweep, read `sweep_results_summary.json` to get the best combo index (e.g. `best_optuna_trial` or `best_by_spearman`).
- Run: `python -m scripts.5b_explain --config "<sweeps_dir>/<batch_id>/combo_<NNNN>/config.yaml"`.
- This can be documented in README or a small “after sweep” checklist; Use `--no-run-explain` to skip. Config path: combo_/config.yaml; re-run manually with `--config <path>`.

---

## 3. Run 021 / Run 022 insights and multi-target sweeps (done)

- **Run 021:** First real success (Model A contributes); **not** optimized for NDCG or Spearman in isolation; default config.
- **Run 022** (EOS: eos_final_rank): NDCG 0.48, Spearman 0.43, playoff Spearman 0.46 — better than early Spearman-only sweeps; **not** single-metric optimized.
- **Strategy:** Run **separate Optuna sweeps** with `--objective spearman`, `--objective ndcg`, `--objective playoff_spearman`, or `--objective rank_mae`; compare best configs across objectives. See README and `outputs/ANALYSIS.md` (§3–4).

---

## 4. Optional / future

- **Phase 2 (attention / trust):** Model A attention fallback, set_attention behavior, training diagnostics (see e.g. `fix_attention_+_trustworthy_run_d52cdb1c.plan.md`, `model_a_attention_fix_and_phased_roadmap_1e5c219f.plan.md`).
- **README / ANALYSIS.md:** Updated with explain `--config`, run_021/022 insights, sweep strategy, and after-sweep explain (required).
- **Notion:** If the project is connected to Notion, update relevant tasks/databases when the best config is locked and when explain is run (per user rules).

---

## 4. Summary table


| Item                                                          | Status      | Notes                                    |
| ------------------------------------------------------------- | ----------- | ---------------------------------------- |
| Sweep collects all metrics (ensemble, playoff, by_conference) | Done        | `_collect_metrics` + Optuna merge        |
| Sweep CSV includes all metric columns                         | Done        | Union of keys across rows                |
| Sweep summary (best_by_spearman, playoff, etc.)               | Done        | `sweep_results_summary.json`             |
| 5b_explain accepts `--config`                                 | **Done**    | So explain can target best combo         |
| Run explain on best sweep combo                               | **Done**    | Sweep runs 5b_explain after each run     |
| README / ANALYSIS.md (run_021/022, sweep strategy)            | **Done**    | Insights and multi-target sweep strategy |
| Config: require_eos_final_rank for playoff eval               | Config-only | If you want sweep to use playoff ranks   |


---

## 5. Key files

- `scripts/sweep_hparams.py` — `_collect_metrics`, CSV columns union, summary keys.
- `scripts/5_evaluate.py` — Writes `eval_report.json` and per-season reports; structure matches what sweep collects.
- `scripts/5b_explain.py` — Accepts `--config`; sweep runs it on best combo after each run.
- `config/defaults.yaml` — `paths.outputs`, optional `inference.require_eos_final_rank`.

