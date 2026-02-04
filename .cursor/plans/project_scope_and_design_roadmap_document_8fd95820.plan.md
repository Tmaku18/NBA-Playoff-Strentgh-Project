---
name: Project Scope and Design Roadmap Document
overview: Create a single new document (e.g. docs/PROJECT_SCOPE_AND_DESIGN_ROADMAP.md) that (1) defines project scope and key research findings as a "start-from-scratch" report, (2) states the original plan from Plan.md, and (3) organizes all design alterations from Update1–Update8 and related plans into a logical chronological roadmap with references and rationale (including why alternatives were rejected).
todos: []
isProject: false
---

# Project Scope and Design Roadmap Document

## Objective

Create one new document that serves as a **report-style plan** for the NBA True Strength project: scope, original design, and a **chronological roadmap of design alterations** with clear references and rationale (including why alternatives were not chosen).

---

## Document location and structure

**Suggested path:** [docs/PROJECT_SCOPE_AND_DESIGN_ROADMAP.md](docs/PROJECT_SCOPE_AND_DESIGN_ROADMAP.md)

**High-level sections:**

1. **Project scope and key research findings** — Written as if starting from scratch: what the project does, target (true team strength, sleepers vs paper tigers), data scope (2015–16 onward, modern era), and key research/prior work (ListMLE, Deep Sets, stacking, Zhai et al. σReparam, etc.).
2. **Original plan** — Condensed but faithful summary of [.cursor/plans/Plan.md](.cursor/plans/Plan.md): 10 phases (Phase 0–9), architecture (Model A Deep Set + ListMLE, Model B XGB+RF, RidgeCV stacking), evaluation (NDCG, Spearman, MRR, Brier, ROC-AUC upset), anti-leakage rules, and file-level checklist.
3. **Chronological roadmap of design alterations** — Grouped revisions in order, each with: what changed, where (files/plans), why (rationale + source), and why alternatives were rejected.

---

## Content blueprint

### Part 1: Scope and research (start-from-scratch framing)

- **Scope:** Multi-modal stacking ensemble for NBA “true team strength”; predict future outcomes and identify sleepers vs paper tigers; no circular evaluation; no net_rating leakage.
- **Targets:** Future W/L (e.g. next 5) or final playoff seed — never raw efficiency as primary target.
- **Data:** nba_api, Kaggle (SOS/SRS), Basketball-Reference fallback; DuckDB; 2015–16 through 2025–26 (modern era); playoff tables separate.
- **Key research / prior work to cite:**
  - ListMLE / listwise ranking (e.g. ListNet, ListMLE) for list-based strength ordering.
  - Deep Sets (permutation invariance) for roster aggregation.
  - Stacking with OOF and RidgeCV (not simple averaging; not Logistic Regression meta) — stability and avoidance of overfitting on limited OOF.
  - Zhai et al. (arXiv:2303.06296) — σReparam for attention stability; [.cursor/plans/Attention_Report.md](.cursor/plans/Attention_Report.md).
  - Bergstra & Bengio (2012), Optuna (Akiba et al. 2019), Hyperband/BOHB (for hyperparameter evolution) — [docs/HYPERPARAMETER_TESTING_EVOLUTION.md](docs/HYPERPARAMETER_TESTING_EVOLUTION.md).
  - Four Factors, SRS, Pythagorean expectation — [.cursor/plans/comprehensive_feature_and_evaluation_expansion.plan.md](.cursor/plans/comprehensive_feature_and_evaluation_expansion.plan.md).

---

### Part 2: Original plan summary

- **Source:** [.cursor/plans/Plan.md](.cursor/plans/Plan.md).
- **Phases:** Phase 0 (requirements lock) → Phase 1 (data/storage) → Phase 2 (features/leakage) → Phase 3 (Model A) → Phase 4 (Model B) → Phase 5 (OOF stacking) → Phase 6 (evaluation/baselines) → Phase 7 (explainability) → Phase 8 (viz) → Phase 9 (integration/reproducibility).
- **Architecture:** Model A (Deep Set, player encoder, set attention, ListMLE over conference-date lists); Model B (XGB + RF, no net_rating); RidgeCV on pooled OOF; hash-trick for unseen players; logsumexp in ListMLE; season boundaries in config.
- **Evaluation:** NDCG, Spearman, MRR, Brier, ROC-AUC upset; baselines SRS, Net Rating, Dummy.
- **Outputs:** Conference rank 1–15, true strength score 0–1, fraud/sleeper delta, ensemble agreement, roster dependence (attention/IG).
- **Anti-leakage:** t-1 only; roster as-of date; no net_rating in Model B; baselines only for net rating.

---

### Part 3: Chronological roadmap of design alterations

Organize into **logical time-ordered groups** (not strictly by Update number where order is clearer otherwise). For each group: **what changed**, **where**, **why (with reference)**, **why alternatives were not chosen**.

**Group A — Data and evaluation target (playoff-aware)**

- **Update1 ([.cursor/plans/Update1.md](.cursor/plans/Update1.md)):** Playoff tables, playoff-performance rank (3-phase: playoff wins → tie-break reg-season win % → non-playoff 17–30), config `training.target_rank: standings | playoffs`, championship odds (softmax + temperature), conference rank 1–15, new plots (pred_vs_playoff_rank, title_contender_scatter, odds_top10), playoff_metrics (Spearman vs playoff rank, NDCG@4, Brier champion).
- **Rationale:** Align evaluation with “who goes furthest in playoffs” and support conference-level outputs; Play-In excluded by design.
- **Alternatives not chosen:** Keeping only standings would not answer playoff outcome; using net rating as target would reintroduce leakage.

**Group B — Train/test and inference discipline**

- **75/25 split plan ([.cursor/plans/75-25_split_and_richer_metrics_78a2db80.plan.md](.cursor/plans/75-25_split_and_richer_metrics_78a2db80.plan.md)):** Season-based train/test (e.g. train 2015–16–2022–23, test 2023–24, 2024–25); `split_info.json`; inference on **last test date** only; scripts 3/4 use train lists only.
- **Rationale:** Time-order holdout; no future leakage; reproducible split.
- **Alternatives not chosen:** Random split would leak time; using “latest date overall” would mix train/test.

**Group C — EOS final rank and walk-forward**

- **Update7 ([.cursor/plans/Update7.md](.cursor/plans/Update7.md)):** EOS final rank (Option B): when playoff data exists (≥16 teams), set `EOS_global_rank` to playoff outcome (champion=1 … first two eliminated=29–30); `eos_rank_source` in outputs; EOS_playoff_standings; per-season inference (`predictions_{season}.json`) and evaluation; optional walk-forward training (train on 1..k, validate on k+1).
- **Rationale:** Ground truth for evaluation = playoff outcome when available; per-season outputs for clear comparison.
- **Alternatives not chosen:** Option A (separate playoff rank field only) would leave evaluation scale ambiguous; no walk-forward by default to keep pipeline simple.

**Group D — Output and inference quality (Update2)**

- **Update2 ([.cursor/plans/Update2.md](.cursor/plans/Update2.md)):** IG batching fix (Captum auxiliary tensors expanded to batch size); latest-team roster (player’s most recent team as of `as_of_date`); `EOS_global_rank` (1–30) for evaluation; manifest `db_path` relative to project root; conference plot uses only valid conference ranks (no global fallback); NaN attention/IG sanitized, JSON `allow_nan=False`; ensemble agreement High/Medium/Low with scaled thresholds and handling of missing models; ensemble_score percentile can reach 0.0/1.0; optional IG in predictions (`ig_inference_top_k`, `ig_inference_steps`).
- **Rationale:** IG failed with batched inputs; roster showed traded players on wrong teams; NaN broke JSON parsers; scale mixing (1–15 vs 1–30) confused metrics.
- **Alternatives not chosen:** Keeping absolute db_path reduces portability; using global rank on conference plot would mislead; emitting NaN would break strict JSON.

**Group E — Playoff rank computation and season mapping (Update3)**

- **Update3 ([.cursor/plans/Update3.md](.cursor/plans/Update3.md)):** Date-range filtering for playoff/reg-season (from config season start/end) instead of season-string matching (`"2024"` vs `"2023-24"`); playoff teams = teams that appear in playoff logs (not only playoff_wins > 0); masked query in set attention (exclude padded positions from mean for query); configurable Model A epochs (and early stopping); attention debug and fallback policy (`contributors_are_fallback`).
- **Rationale:** Season format mismatch caused all teams to get ranks 17–30; swept teams (0 wins) must still be playoff teams; padded positions diluted query; flat loss required more epochs and diagnostics.
- **Alternatives not chosen:** String normalization alone is fragile across sources; including padding in query keeps attention degenerate; fixed 3 epochs were insufficient.

**Group F — Model A attention collapse and σReparam (Attention_Report, fix_attention plan)**

- **Attention_Report ([.cursor/plans/Attention_Report.md](.cursor/plans/Attention_Report.md)), fix_attention ([.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md](.cursor/plans/fix_attention_+_trustworthy_run_d52cdb1c.plan.md)), MODEL_A_NOT_LEARNING ([docs/MODEL_A_NOT_LEARNING_ANALYSIS.md](docs/MODEL_A_NOT_LEARNING_ANALYSIS.md)):** σReparam on Q/K/V (Zhai et al., arXiv:2303.06296) in SetAttention; configurable learning_rate and grad_clip_max_norm; minutes reweighting only when meaningful; uniform/minutes fallback when raw attention is zero or non-finite; extended attention debug (encoder Z variance, scores, grad norms); pipeline order (inference before evaluate); evaluation scale-consistent (EOS_global_rank only, no conference fallback for global metrics); roster and playoff rank guards.
- **Rationale:** Collapsed attention → constant Z → constant scores → flat ListMLE loss; Zhai et al. bounds spectral norm to stabilize attention; fallbacks keep gradients finite.
- **Alternatives not chosen:** PyTorch spectral_norm alone (no learnable γ) is less flexible than σReparam; no fallback would leave NaN in batches; evaluating on conference rank for global metrics would mix scales.

**Group G — Sweeps and hyperparameter methodology (Update4–6, Update8, HYPERPARAMETER_TESTING_EVOLUTION)**

- **Update4 ([.cursor/plans/Update4.md](.cursor/plans/Update4.md)):** Sweep script with real DB data, configurable epochs, Model B grid; outputs under `outputs/sweeps/`.
- **Update5–6:** Sweep rerun with attention diagnostics; phased Model B (phase1_xgb, phase2_rf); `--val-frac`, `--batch-id`.
- **Update8 ([.cursor/plans/Update8.md](.cursor/plans/Update8.md)):** Outputs to outputs2; run_id from 19; sweeps to `outputs2/sweeps/<batch_id>/`; foreground, no timeout; per-season inference and conference rank as primary.
- **HYPERPARAMETER_TESTING_EVOLUTION ([docs/HYPERPARAMETER_TESTING_EVOLUTION.md](docs/HYPERPARAMETER_TESTING_EVOLUTION.md)):** Grid → smaller default grid; Optuna (TPE) with multiple objectives; successive halving (cheap vs full epochs); `--n-jobs`; phased grids; post-sweep explain on best combo. References: Bergstra & Bengio 2012, Optuna (Akiba et al. 2019), Hyperband/BOHB.
- **Rationale:** Full grid (e.g. 1,728 combos) infeasible; Bayesian and halving reduce evaluations; phased grids focus on high-impact params; foreground avoids hidden timeouts.
- **Alternatives not chosen:** Single huge grid without parallelism would not finish; random search only (no Optuna) would give no importance analysis; background daemon would complicate debugging.

**Group H — Outputs3, run_022, and reporting**

- **README / CHECKPOINT_PROJECT_REPORT:** outputs3 for sweeps and new runs; run_021/022 as baseline (Model A contributing, playoff Spearman 0.46); sweep strategy: separate objectives (spearman, ndcg, playoff_spearman, rank_mae); clone classifier (4c) optional; manifest and reproducibility (WSL/GPU note).
- **Rationale:** Single place for sweep and test runs; clear baseline and metric definitions; clone classifier for playoff vs non-playoff comparison.
- **Alternatives not chosen:** Mixing sweep and legacy runs in one folder would be confusing; one objective for all sweeps would not show trade-offs.

---

## References to cite in the document

- **Zhai et al. (2023):** Stabilizing Transformer Training by Preventing Attention Entropy Collapse. arXiv:2303.06296. (σReparam; Model A.)
- **Bergstra & Bengio (2012):** Random Search for Hyper-Parameter Optimization. JMLR 13.
- **Akiba et al. (2019):** Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.
- **Li et al. (2017):** Hyperband. JMLR 18(1).
- **Falkner et al. (2018):** BOHB. ICML.
- **Plan and updates:** `.cursor/plans/Plan.md`, `Update1.md`–`Update8.md`, and the specific plan files named in each group above.
- **Project docs:** `docs/CHECKPOINT_PROJECT_REPORT.md`, `docs/ATTENTION_AND_BATCHES.md`, `docs/MODEL_A_NOT_LEARNING_ANALYSIS.md`, `docs/HYPERPARAMETER_TESTING_EVOLUTION.md`, `docs/METRICS_USED.md`, `docs/PLATT_CALIBRATION.md`, `README.md`.

---

## Implementation notes

- **Single file:** One markdown file in `docs/` (e.g. `PROJECT_SCOPE_AND_DESIGN_ROADMAP.md`) containing all three parts.
- **Cross-links:** Use relative links to `.cursor/plans/*.md` and other `docs/*.md` and `README.md` so the document doubles as a map of the repo.
- **Tables/diagrams:** Optional: one table summarizing “revision group → main files changed” and one simple flowchart (e.g. original phases vs post-revision pipeline) if it aids clarity; keep mermaid rules (no spaces in node IDs, no HTML in labels).
- **Tone:** Report-style, third person or “the project”; avoid emojis; concise but complete enough for someone starting from scratch or writing a report.

---

## Deliverable

- **New file:** [docs/PROJECT_SCOPE_AND_DESIGN_ROADMAP.md](docs/PROJECT_SCOPE_AND_DESIGN_ROADMAP.md) with:
  1. Project scope and key research findings (start-from-scratch).
  2. Original plan summary (Plan.md).
  3. Chronological roadmap of design alterations (Groups A–H), each with what changed, where, why (with references), and why alternatives were not chosen.
  4. References section listing all cited plans and papers.

No code or config changes; README can add one line linking to this doc under “Report Assets” or “Documentation” if desired.
