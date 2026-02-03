# Hyperparameter Testing Evolution: Methodology, Decisions, and Speed

**Project:** NBA Playoff Strength — True Strength Prediction  
**Scope:** From initial sweep design through grid, Bayesian (Optuna), successive halving, and phased Model B grids.  
**Last updated:** February 2026

---

## 1. Executive Summary

This document traces the evolution of hyperparameter testing in the NBA True Strength pipeline from the first sweep designs to the current multi-method setup. It explains **methodology changes**, **decision steps**, **pros and cons** of each choice, and the **persistent challenge of testing speed**, with references to research on optimization techniques.

**Current capabilities (as of this document):**

- **Grid search** over Model A epochs, rolling windows, and Model B (XGB + RF) params; configurable validation fraction and phased grids.
- **Optuna (Bayesian)** with TPE sampler; multiple objectives (spearman, ndcg, playoff_spearman, rank_mae); parameter importance (Fanova) post-run.
- **Successive halving** (custom): cheap-epoch round then full-epoch round on top 1/factor survivors.
- **Parallelism:** `--n-jobs` for both grid (ProcessPoolExecutor) and Optuna trials.
- **Smaller default grid** in config for feasible full runs; phased grids (`--phase phase1_xgb`, `phase2_rf`) for targeted exploration.

**Key constraint:** A single combo runs the full pipeline (Model A → Model B → stacking → inference → evaluation), typically **10–30+ minutes** per combo. Full grids (e.g. 1,728 combos) are infeasible without parallelism, pruning, or smaller search spaces.

---

## 2. Timeline and Decision Steps

### 2.1 Pre-sweep: Model A “not learning” and attention collapse

**Context (from `.cursor/plans/` and docs):**

- Early runs (e.g. run_009–013) showed **flat train loss** and **all-zero or degenerate attention** in Model A. Sweeps could not produce meaningful comparisons until Model A was trainable.
- Plans (e.g. `model_a_attention_fix_and_phased_roadmap_1e5c219f.plan.md`, `Attention_Report.md`) prioritized:
  1. **Harden SetAttention** (minutes/uniform fallback when raw attention is zero).
  2. **σReparam** on Q/K/V (Zhai et al., arXiv:2303.06296) to bound spectral norm and reduce attention entropy collapse.
  3. **Configurable LR and gradient clipping** (`learning_rate`, `grad_clip_max_norm`).
  4. **Expanded debug logging** (encoder Z variance, gradient norms, relevance) when `attention_debug: true`.

**Decision:** Fix Model A before investing in large sweeps.  
**Pros:** Sweep metrics (NDCG, Spearman, etc.) reflect real model differences, not “which combo happened to get a non-collapsed run.”  
**Cons:** Delayed availability of sweep results; early batches (e.g. 20260201, 20260203) failed or timed out at Model A.

---

### 2.2 Sweep script introduction (outputs2, run_019, Update8)

**Source:** `.cursor/plans/outputs2_run_019_sweeps_update8_d5fca612.plan.md`

**Decisions:**

- **Sweeps write to configured outputs path** (e.g. `outputs2/sweeps/<batch_id>/`), not a fixed folder. Enables multiple batches without overwriting.
- **Run in foreground, no artificial timeout.** Long sweeps can complete; user controls environment (e.g. server, no 10-minute limit).
- **No background/daemon mode.** Simplifies debugging and log inspection; user runs the script explicitly.

**Pros:** Robust, reproducible, and predictable behavior; no hidden timeouts killing runs.  
**Cons:** User must plan for long wall-clock time (hours/days for large grids).

---

### 2.3 Initial grid design and first failures

**Source:** `outputs2/sweeps/SWEEP_ANALYSIS_REPORT.md`, `config/defaults.yaml` (historical)

**Original grid (from SWEEP_ANALYSIS_REPORT):**

| Dimension           | Values | Count |
|--------------------|--------|-------|
| model_a_epochs     | [8, 16, 24, 28] | 4 |
| rolling_windows    | [5,10], [10,20], [10,30], [15,30] | 4 |
| max_depth (XGB)    | [3, 4, 5] | 3 |
| learning_rate      | [0.05, 0.08, 0.10] | 3 |
| n_estimators_xgb   | [200, 250, 300, 350] | 4 |
| n_estimators_rf    | [150, 200, 250] | 3 |
| (subsample, colsample, min_leaf fixed) | | |

- **Full grid size:** 4×4×3×3×4×3 = **1,728 combos**.
- **Observed:** Batches 20260201_165611, 20260201_165650 **failed at Model A** (exit non-zero). Batch 20260203_021923 (4 combos with `--max-combos`) **timed out** during combo 0 (Model A completed; pipeline did not reach script 4 in time). Optuna batch `optuna_3trial` **timed out** during trial 1.

**Decision:** Treat “no successful sweep combos” as a combination of (1) Model A failures and (2) single-combo runtime (10–25+ minutes) making even small runs vulnerable to environment timeouts.

**Pros:** Clear evidence that full grid is impractical without either fewer combos or faster evaluation.  
**Cons:** No sweep_results.csv with metrics yet; baseline comparison (run_022) remains the only completed full-pipeline reference.

---

### 2.4 Configurable validation fraction and batch id

**Source:** `.cursor/plans/refined_sweep_rerun_bc8afb8f.plan.md`

**Decisions:**

- **`--val-frac`:** Model A early-stopping validation fraction (default 0.25). Aligns with plan recommendation (Performance_trajectory_and_hyperparameters.md: “Sweeps use val_frac=0.25; config has 0.1. Prefer 0.25 for epoch/sweep comparability”).
- **`--batch-id`:** User-settable batch folder name (default: timestamp). Enables reproducible batch names and multiple reruns without overwriting.

**Pros:** Consistent validation setup across sweeps; easier to compare epochs and Model B across batches.  
**Cons:** None significant; purely additive.

---

### 2.5 Phased Model B grids (Phase 1 XGB, Phase 2 RF)

**Source:** `.cursor/plans/refined_sweep_rerun_bc8afb8f.plan.md`, `Performance_trajectory_and_hyperparameters.md`

**Rationale:** XGBoost tuning: `learning_rate` trades off with `n_estimators`; `subsample`/`colsample_bytree` control overfitting. Random Forest: `n_estimators` and `min_samples_leaf`/`max_depth` are primary levers. Rather than one huge joint grid, use two **local** grids around sensible defaults.

**Phase 1 (XGB local):**

- Fix: max_depth=4, RF fixed (n_estimators=200, min_samples_leaf=5).
- Vary: learning_rate ∈ {0.08, 0.10, 0.12}, n_estimators_xgb ∈ {250, 300, 350}, subsample=0.8, colsample_bytree=0.8.
- Model A epochs: 8–28 (step 1).

**Phase 2 (RF local):**

- Fix: XGB at “best” from Phase 1 (e.g. max_depth=4, lr=0.10, n_xgb=300, subsample=0.8, colsample=0.8).
- Vary: n_estimators_rf ∈ {150, 200, 250}, min_samples_leaf ∈ {4, 5, 6}.
- Model A epochs: 8–28 (step 1).

**Decision:** Implement as `--phase full | phase1_xgb | phase2_rf`. `full` uses config sweep section; `phase1_xgb` / `phase2_rf` override lists in the script.

**Pros:** Cuts combinatorial explosion; focuses compute on high-impact dimensions; matches “ranking-first” vs “RMSE-first” intuition from plans.  
**Cons:** Best Phase 2 depends on Phase 1; joint interactions between XGB and RF are not fully explored.

---

### 2.6 Bayesian tuning (Optuna)

**Source:** `.cursor/plans/model_a_attention_fix_and_phased_roadmap_1e5c219f.plan.md` (“Bayesian tuning (immediately after attention fix)”)

**Decisions:**

- **Method:** Optuna with default TPE (Tree-structured Parzen Estimator) sampler.
- **Objectives:** `--objective spearman | ndcg | playoff_spearman | rank_mae`; rank_mae is minimized, others maximized.
- **Outputs:** `optuna_study.json` (best_value, best_params), `optuna_importances.json` (Fanova parameter importance) for post-hoc analysis.
- **Usage:** `--method optuna --n-trials N`. Same pipeline per trial; no pruning within a trial (each trial runs full pipeline).

**Pros:** Fewer evaluations than grid to reach good configs; importance scores help fix low-importance params and shrink search space in follow-up runs.  
**Cons:** Each trial is still a full pipeline (10–30+ min); TPE needs a minimum number of trials to learn; no early stopping within a trial (e.g. no pruning at script 3).

---

### 2.7 Successive halving (custom)

**Decision:** Add `--method halving` that (1) runs **all** combos (or a random subset) with a **cheap** Model A epoch count (`--halving-epochs-cheap`, default 8), (2) ranks by a single metric (e.g. test_metrics_ensemble_spearman), (3) keeps top **1/factor** (e.g. 1/3), (4) re-runs those **survivors** with **full** Model A epochs (`--halving-epochs-full`, default max of epoch list).

**Pros:** Reduces total compute when many configs are bad; aligns with bandit-based methods (Hyperband, BOHB) that allocate more resource to promising configs.  
**Cons:** Cheap epoch may reorder configs vs full epoch (approximation); two rounds still require many full pipelines for survivors; no Bayesian guidance for which configs to try.

---

### 2.8 Parallelism (`--n-jobs`)

**Decision:** For grid: run combos in parallel with `concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs)`. For Optuna: pass `n_jobs` to `study.optimize()`. Default 1 (sequential).

**Pros:** Linear speedup with cores (subject to I/O and DB contention); makes larger grids or trial counts feasible on multi-core machines.  
**Cons:** Memory scales with workers (each worker loads config, runs full pipeline); need enough RAM and stable DB/disk.

---

### 2.9 Smaller default grid in config

**Source:** Recommendation in `SWEEP_ANALYSIS_REPORT.md` and plans: “reduce grid for a quick full grid e.g. model_a_epochs: [16, 28], rolling_windows: [[10, 30]], max_depth: [4], single values for the rest ⇒ 2×1×1×1 = 2 combos.”

**Decision:** In `config/defaults.yaml`, sweep section was reduced to a **smaller default grid**, e.g.:

- model_a_epochs: [16, 28]
- rolling_windows: [[10, 30]]
- model_b: max_depth [3, 4], learning_rate [0.08], n_estimators_xgb [250, 300], n_estimators_rf [200], etc.

So **default combo count** is much smaller (e.g. 8) for a feasible end-to-end test without `--max-combos` or `--phase`.

**Pros:** `python -m scripts.sweep_hparams` (no args) can complete in a few hours; CI or quick validation possible.  
**Cons:** Default does not explore phases or full historical grid; user must explicitly set phase or expand config for large sweeps.

---

### 2.10 Post-sweep explain on best combo

**Decision:** After writing sweep_results and summary, the script runs `5b_explain` on the best combo (by Spearman for grid/halving, by Optuna value for Optuna) unless `--no-run-explain`.

**Pros:** Best config gets SHAP and attention/IG artifacts for reporting.  
**Cons:** Adds one full explain run after the sweep; can be skipped if not needed.

---

## 3. Methodology Summary Table

| Method        | How it works | Best for | Speed vs full grid | Pros | Cons |
|---------------|--------------|----------|--------------------|------|------|
| **Grid**      | Exhaustive product of lists in config (or phase overrides). | Small grids, reproducibility, full coverage. | 1× (baseline) | Deterministic; no approximation. | Combinatorial explosion; slow. |
| **Optuna**    | TPE sampler suggests params; each trial = full pipeline. | Medium trial counts (e.g. 20–50); single objective. | Fewer evaluations to “good” config (literature: ~5–20× fewer than grid for similar quality). | Importance output; flexible objectives. | Per-trial cost unchanged; needs several trials to learn. |
| **Halving**   | Round 1: all combos, cheap epochs; round 2: top 1/factor, full epochs. | Large combo lists when cheap epoch correlates with full. | Depends on factor and correlation; can save ~(1 - 1/factor) of full-epoch runs. | No extra tuning library; simple. | Ranking can change with full epochs; two rounds. |
| **Phased**    | phase1_xgb or phase2_rf overrides grid to local XGB or RF subspace. | Targeted Model B tuning after a baseline. | Fewer combos than full joint grid. | Focused search; interpretable. | Not joint XGB+RF; Phase 2 conditional on Phase 1. |
| **Parallel**  | `--n-jobs` runs multiple pipelines at once (grid or Optuna). | Multi-core machines; same method, less wall time. | Up to n_jobs× fewer wall-clock time. | Easy to use. | Memory and I/O scale with n_jobs. |

---

## 4. The Challenge of Speed and Research on Optimization Techniques

### 4.1 Why speed is a bottleneck

Each “combo” or “trial” runs the **full pipeline**:

1. **3_train_model_a** — K-fold OOF (Model A) then final training (often 10–20+ minutes).
2. **4_train_model_b** — OOF and training for XGB + RF.
3. **4b_train_stacking** — RidgeCV meta-learner.
4. **6_run_inference** — Predictions for test seasons.
5. **5_evaluate** — eval_report.json.

So one evaluation is **10–30+ minutes**. Implications:

- **Full grid of 1,728:** ~288–864 hours (12–36 days) at 1 combo at a time.
- **Even 8 combos:** ~2–4 hours sequential.
- **Optuna 20 trials:** Same 20 × (10–30 min) if sequential; with `n_jobs=4`, wall time is still significant.

Hence the need for: **smaller default grid**, **fewer trials** (Optuna), **early pruning** (halving, or future pruning inside Model A), and **parallelism**.

### 4.2 Research on speed of techniques (with references)

**Grid search**

- Exhaustively evaluates every point in the discrete product of hyperparameters. Simple and reproducible but **cost = product of sizes** of each dimension. No guidance from past evaluations.
- No single “grid search” paper; it is the baseline in most HPO comparisons.

**Random search**

- Bergstra & Bengio (2012), “Random Search for Hyper-Parameter Optimization”: often **as good or better than grid** in practice because many dimensions are low-impact; cost is linear in number of trials.
- Reference: *Journal of Machine Learning Research*, 13:281–305.

**Bayesian optimization (e.g. TPE, Gaussian processes)**

- Uses a surrogate model (e.g. TPE, GP) to suggest the next hyperparameters from previous trial results. **Fewer evaluations** to reach good regions than grid or random.
- **Optuna:** Akiba et al. (2019), “Optuna: A Next-generation Hyperparameter Optimization Framework,” *KDD 2019*. Define-by-run API, TPE default, pruning support. [KDD 2019](https://www.kdd.org/kdd2019/accepted-papers/view/optuna-a-next-generation-hyperparameter-optimization-framework); [arXiv:1907.10902](https://arxiv.org/abs/1907.10902).
- In practice, Bayesian methods often need **5–20× fewer evaluations** than grid to reach similar or better best values (depending on space and budget).

**Successive halving and Hyperband**

- **Successive halving:** Evaluate all configs on a small budget; keep top 1/η; repeat with budget × η until max budget. **Resource efficiency** when “good” configs tend to be good early.
- **Hyperband:** Li et al. (2017), “Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization,” *JMLR* 18:1–52. Uses successive halving as a subroutine with multiple brackets. [JMLR](https://jmlr.org/papers/v18/16-558.html).
- **BOHB:** Falkner, Klein, Hutter (2018), “BOHB: Robust and Efficient Hyperparameter Optimization at Scale,” *ICML 2018*. Combines Bayesian models (for sampling) with Hyperband (for resource allocation). Often **outperforms** both vanilla Bayesian optimization and Hyperband alone; strong **anytime** performance. [MLR proceedings](https://proceedings.mlr.press/v80/falkner18a.html).

**Implications for our pipeline**

- **Grid:** Only feasible with a **small grid** (e.g. default 8 combos) or **parallelism**.
- **Optuna:** Aligns with “fewer trials for good configs”; our constraint is **per-trial cost** (full pipeline). Pruning *within* a trial (e.g. after Model A) is not implemented but would further improve speed (Optuna supports pruning callbacks).
- **Halving:** Our custom implementation approximates “more resource to promising configs” by a **two-stage** (cheap vs full epochs). True Hyperband would require multiple brackets and budget levels; BOHB would add a surrogate model for sampling. We kept the design simple and avoid new dependencies.

### 4.3 References (concise)

1. Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13, 281–305.
2. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *Journal of Machine Learning Research*, 18(1), 1–52.
3. Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80.
4. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD)*.
5. Zhai, S., et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. *arXiv:2303.06296* (for Model A σReparam, not HPO speed).

---

## 5. Current Usage (Quick Reference)

```bash
# Dry-run (preview combo count)
python -m scripts.sweep_hparams --dry-run

# Grid with smaller default (from config)
python -m scripts.sweep_hparams

# Grid with limit and validation fraction
python -m scripts.sweep_hparams --max-combos 8 --val-frac 0.25

# Phased Model B grids
python -m scripts.sweep_hparams --phase phase1_xgb --batch-id phase1_xgb_001
python -m scripts.sweep_hparams --phase phase2_rf --batch-id phase2_rf_001

# Optuna (Bayesian)
python -m scripts.sweep_hparams --method optuna --n-trials 20 --objective spearman
python -m scripts.sweep_hparams --method optuna --n-trials 15 --objective rank_mae --batch-id optuna_rank_mae

# Successive halving
python -m scripts.sweep_hparams --method halving --halving-epochs-cheap 8 --halving-epochs-full 28 --halving-factor 3

# Parallel (grid or Optuna)
python -m scripts.sweep_hparams --n-jobs 4 --max-combos 16
python -m scripts.sweep_hparams --method optuna --n-trials 20 --n-jobs 2
```

Outputs (per batch): `sweep_results.csv`, `sweep_results_summary.json`, `sweep_config.json`; for Optuna also `optuna_study.json`, `optuna_importances.json`. Best combo gets `5b_explain` unless `--no-run-explain`.

---

## 6. Document History and Related Files

- **Plans:** `.cursor/plans/model_a_attention_fix_and_phased_roadmap_1e5c219f.plan.md`, `refined_sweep_rerun_bc8afb8f.plan.md`, `outputs2_run_019_sweeps_update8_d5fca612.plan.md`, `sweep_rerun_+_attention_check_7aa587c2.plan.md`, `Attention_Report.md`, `Performance_trajectory_and_hyperparameters.md`.
- **Sweep analysis:** `outputs2/sweeps/SWEEP_ANALYSIS_REPORT.md`.
- **Implementation:** `scripts/sweep_hparams.py`, `config/defaults.yaml` (sweep section).
- **README:** `README.md` (sweep usage and outputs3).

This document is the single place for **methodology evolution**, **decision rationale**, **pros/cons**, and **speed/research** for hyperparameter testing in this project.
