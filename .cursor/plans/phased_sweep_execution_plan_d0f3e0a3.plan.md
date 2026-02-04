---
name: Phased Sweep Execution Plan
overview: "Actionable agent-executable plan for completing Phase 1 hyperparameter sweeps: commit/push current work, run 12 sweeps (6 objectives x 2 listmle_targets) with --phase phase1, and update docs after each sweep. Robust settings ensure each objective batch (2 sweeps + analysis) finishes in &lt; 4 hours with 4 parallel jobs."
todos:
  - id: git-commit-push
    content: Git add, commit, and push all Phase 0 outputs and code changes
    status: pending
  - id: sweep-spearman
    content: Run phase1_spearman_final_rank and phase1_spearman_playoff_outcome; analyze; update docs
    status: pending
  - id: sweep-ndcg4
    content: Run phase1_ndcg4_final_rank and phase1_ndcg4_playoff_outcome; analyze; update docs
    status: pending
  - id: sweep-ndcg16
    content: Run phase1_ndcg16_final_rank and phase1_ndcg16_playoff_outcome; analyze; update docs
    status: pending
  - id: sweep-ndcg20
    content: Run phase1_ndcg20_final_rank and phase1_ndcg20_playoff_outcome; analyze; update docs
    status: pending
  - id: sweep-playoff-spearman
    content: Run phase1_playoff_spearman sweeps; analyze; update docs
    status: pending
  - id: sweep-rank-rmse
    content: Run phase1_rank_rmse sweeps; analyze; update docs
    status: pending
  - id: phase2-plan
    content: Create PHASE2_SWEEP_PLAN.md after all Phase 1 sweeps complete
    status: pending
isProject: false
---

# Phased Sweep Execution Plan (Agent-Mode)

Executable plan derived from [.cursor/plans/phased_sweep_roadmap_3hr_6b5aa588.plan.md](.cursor/plans/phased_sweep_roadmap_3hr_6b5aa588.plan.md). Phase 0 complete; Phase 1 pending.

---

## Robustness and 4-Hour Budget (per objective)

**Constraint:** Each objective batch (2 sweeps + analysis) must finish in **&lt; 4 hours** with **4 parallel jobs**.

### Time model


| Setting          | Value     | Rationale                                                            |
| ---------------- | --------- | -------------------------------------------------------------------- |
| n_trials         | **12**    | ceil(12/4) = 3 batches; worst-case 3 × 30 min = **90 min per sweep** |
| n_jobs           | 4         | 4 pipelines in parallel                                              |
| --no-run-explain | **yes**   | Saves ~5–10 min per sweep (explain run manually on best combo later) |
| Per-trial time   | 15–30 min | Pipeline 3→4→4b→6→5; baseline avg ~15 min, worst ~30 min             |


**Per objective:** 2 sweeps × 90 min + 2 × 15 min analysis = **210 min (~3.5 hours)** — safely under 4 hours.

### Robustness tactics

1. **n_trials=12** — Guarantees completion even if trials run 25–30 min each.
2. `**--no-run-explain**` — Skip SHAP/IG on best combo; run `5b_explain` manually after sweep if needed.
3. **Aggregate on interruption** — If sweep stops early (Wi‑Fi, Ctrl+C), run `aggregate_sweep_results.py` to build summary from completed combos.
4. **Optional fast config** — If trials regularly exceed 25 min, use lower `max_lists_oof` / `max_final_batches` in config (trade: fewer lists, faster Model A).

### If trials run faster (15 min avg)

With 15 min trials: ceil(12/4) × 15 = **45 min per sweep**. To use extra time for more trials, increase to `n_trials=16` (4 × 15 = 60 min per sweep, 2 × 60 + 30 = 150 min total). Use `n_trials=12` as the default for robustness.

---

## Prerequisites (verify before starting)

- [scripts/sweep_hparams.py](scripts/sweep_hparams.py): `--phase phase1` exists with narrowed ranges (epochs 14-26, max_depth 3-5, lr 0.06-0.10, n_xgb 200-300, n_rf 150-230, min_leaf 4-6)
- [config/baseline_max_features.yaml](config/baseline_max_features.yaml): exists with max_lists_oof, max_final_batches raised
- cwd: project root; `PYTHONPATH` set to project root
- No other heavy jobs on the machine (sweep uses 4 cores)

---

## Step 1: Git commit and push (before any sweeps)

1. `git add` modified files: scripts, docs, plans, outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md, scripts/aggregate_sweep_results.py
2. Add Phase 0 sweep outputs: `outputs3/sweeps/baseline_ndcg_playoff_outcome/` (if not in .gitignore)
3. Optionally add: data/modern_RAPTOR_by_team.csv, docs/Team_Records.csv, docs/modern_RAPTOR_by_team.csv (user decision)
4. `git commit -m "Phase 0 complete; add phase1, ndcg4/16/20, aggregate script; doc updates"`
5. `git push origin main`

**Rule:** outputs3/sweeps must be pushed before starting sweeps ([outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md](outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md) workflow).

---

## Step 2: Sweep execution loop (per objective)

For each objective in order: `spearman`, `ndcg4`, `ndcg16`, `ndcg20`, `playoff_spearman`, `rank_rmse`:

### 2a. Sweep 1 — final_rank (standings)

```powershell
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective <OBJ> --listmle-target final_rank --phase phase1 --batch-id phase1_<OBJ>_final_rank --config config/baseline_max_features.yaml
```

Replace `<OBJ>` with the objective (e.g. `spearman`, `ndcg4`).

**Timing:** ~45–90 min per sweep (12 trials, 4 jobs). If interrupted — use Step 2c to aggregate partial results.

### 2b. Analyze and update docs (after Sweep 1)

1. Read `outputs3/sweeps/phase1_<OBJ>_final_rank/sweep_results_summary.json`, `sweep_results.csv`, `optuna_importances.json`
2. Compare best combo to run_022, run_023, and previous sweeps
3. Add a section to [outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md](outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md) (or create `SWEEP_PHASE1_ANALYSIS.md`) with:
  - batch_id, objective, listmle_target
  - Best combo params and metrics
  - Comparison table
4. Update [docs/HYPOTHESIZED_BEST_CONFIG_AND_METRIC_INSIGHTS.md](docs/HYPOTHESIZED_BEST_CONFIG_AND_METRIC_INSIGHTS.md) with actual results if material
5. (Optional) Run explain on best combo: `python -m scripts.5b_explain --config outputs3/sweeps/<batch_id>/combo_XXXX/config.yaml`
6. `git add`; `git commit -m "Phase 1: <OBJ> final_rank sweep results"`; `git push origin main`

### 2c. If sweep was interrupted

```powershell
python -m scripts.aggregate_sweep_results --batch-id phase1_<OBJ>_final_rank --objective <OBJ>
```

Then proceed with 2b using the generated summary files.

### 2d. Sweep 2 — playoff_outcome

```powershell
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective <OBJ> --listmle-target playoff_outcome --phase phase1 --batch-id phase1_<OBJ>_playoff_outcome --config config/baseline_max_features.yaml
```

Then repeat 2b (and 2c if needed).

### 2e. Proceed to next objective

Only after both sweeps (final_rank and playoff_outcome) for the current objective are analyzed and committed.

---

## Step 3: Sweep invocation table

**Robust command template** (n_trials=12, n_jobs=4, --no-run-explain; &lt; 4 h per objective):


| Objective        | Sweep 1 batch_id                   | Sweep 2 batch_id                        |
| ---------------- | ---------------------------------- | --------------------------------------- |
| spearman         | phase1_spearman_final_rank         | phase1_spearman_playoff_outcome         |
| ndcg4            | phase1_ndcg4_final_rank            | phase1_ndcg4_playoff_outcome            |
| ndcg16           | phase1_ndcg16_final_rank           | phase1_ndcg16_playoff_outcome           |
| ndcg20           | phase1_ndcg20_final_rank           | phase1_ndcg20_playoff_outcome           |
| playoff_spearman | phase1_playoff_spearman_final_rank | phase1_playoff_spearman_playoff_outcome |
| rank_rmse        | phase1_rank_rmse_final_rank        | phase1_rank_rmse_playoff_outcome        |


---

## Step 4: Long-running sweep handling

- Each sweep is **45–90 min** (12 trials, 4 jobs). Agent `run_terminal_cmd` may timeout (~10 min).
- **Recommendation:** Agent performs Step 1 (commit/push). For sweeps, agent outputs the command; user runs it in foreground (no timeout); agent performs analysis and doc updates when user indicates the sweep finished.
- **If interrupted:** Run `scripts/aggregate_sweep_results.py` (Step 2c) to reconstruct sweep_results.csv and sweep_results_summary.json from completed combo outputs; then proceed with analysis.

---

## Step 5: Phase 2 planning (after all 12 sweeps)

1. Review which objective + listmle_target performed best on each metric
2. Use optuna_importances to decide Phase 2 search ranges
3. Create `.cursor/plans/PHASE2_SWEEP_PLAN.md` with narrowed ranges and objectives

---

## Files modified per iteration

- [outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md](outputs3/sweeps/BASELINE_SWEEP_ANALYSIS.md) or `outputs3/sweeps/SWEEP_PHASE1_ANALYSIS.md` — add sweep section
- [docs/HYPOTHESIZED_BEST_CONFIG_AND_METRIC_INSIGHTS.md](docs/HYPOTHESIZED_BEST_CONFIG_AND_METRIC_INSIGHTS.md) — update best config table
- [.cursor/plans/phased_sweep_roadmap_3hr_6b5aa588.plan.md](.cursor/plans/phased_sweep_roadmap_3hr_6b5aa588.plan.md) — mark completed todos

---

## Mermaid: Execution flow

```mermaid
flowchart TD
    subgraph prep [Pre-Sweep]
        G1[Git commit and push]
    end
    subgraph loop [Per Objective]
        S1[Sweep 1: final_rank]
        A1[Analyze and update docs]
        C1[Git commit and push]
        S2[Sweep 2: playoff_outcome]
        A2[Analyze and update docs]
        C2[Git commit and push]
    end
    G1 --> S1 --> A1 --> C1 --> S2 --> A2 --> C2
    C2 --> Next[Next objective]
```



