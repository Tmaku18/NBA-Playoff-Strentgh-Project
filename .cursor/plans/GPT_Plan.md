Scope & current docs
Primary reference: C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\cab\.cursor\plans\Plan.md
README: C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\cab\README.md
Plan update history (numbered):
C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\cab\.cursor\plans\Plan_and_readme_updates_1.plan.md
C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\cab\.cursor\plans\Plan_and_readme_updates_2.plan.md
Key methodology decisions (locked)
Primary target (phase 1): playoff_seed / conference rank (ListMLE training; season-end evaluation)
Stacking level-2 default: RidgeCV over base model scores (OOF-only training)
No circularity:
No net_rating feature input to Model B
No Net Rating head in Model A
Net Rating is allowed only as a baseline ranker (rank-by-Net-Rating) in evaluation
Strict time rule: all features computed at t-1 (never include target game / future games)
Phase 0 — Documentation, repo scaffolding, and guardrails
Doc alignment:
Update Plan.md and README wording where needed to reflect: playoff_seed as primary target, RidgeCV as default level-2, and the no-Net-Rating policy.
Add a short “Methodology decisions” section in README to prevent drift.
Repository skeleton (minimum viable):
config/ with defaults.yaml
src/ split into data/, features/, models/, training/, evaluation/, inference/, viz/
scripts/ pipeline entrypoints
outputs/ and data/ directories with .gitignore
Reproducibility defaults:
repro.seed in config, used consistently across numpy/torch/sklearn.
Persist run metadata: config snapshot + git commit hash into outputs/run_manifest.json.
Phase 1 — Data ingestion & storage (nba_api first, SOS/SRS without scraping traps)
Storage choice:
Default to DuckDB for analytics joins; keep an SQLite compatibility path only if necessary.
Ingestion (nba_api):
Pull game schedule, team game logs, player game logs.
If tracking endpoints are too slow/unreliable, defer tracking features until Phase 6.
SOS/SRS:
Prefer Kaggle Wyatt Walsh dataset for historical SOS/SRS.
If SOS/SRS missing, compute a proxy SOS from your own schedule + opponent strength.
Data versioning:
Store raw pulls and derived tables in data/raw/ and data/processed/.
Write data/manifest.json (hash + download timestamp).
Deliverables:
src/data/nba_api_client.py, src/data/db_schema.py, src/data/db_loader.py
scripts/1_download_raw.py, scripts/2_build_db.py
Phase 2 — Feature engineering (the leakage-proof core)
Roster construction (as-of date):
For each conference/date (training) and conference/season-end (eval), select top-N players by minutes up to as-of date only.
Keep variable-length masking; pad to 15.
Rolling stats:
Implement rolling windows (10, 30) with an explicit shift(1) before rolling aggregation.
DNP handling: compute per-game averages over games played; store availability fraction as separate feature.
Minutes signal:
Use either explicit minutes-weighting in attention or MP in encoder (default: explicit weighting; do not include MP in encoder stats).
Deliverables:
src/features/rolling.py
src/data/build_roster_set.py (produces roster tensors + masks + per-player features)
Phase 3 — Model A (Deep Set ranker) + stable ListMLE
Player encoder: shared MLP mapping [stats; embedding] -> player_vec.
Attention: multi-head attention over player_vec with padding mask.
Output: team score scalar (for ranking) + penultimate Z.
ListMLE:
Implement numerically stable ListMLE using the log-sum-exp trick (torch.logsumexp).
Lists are conference-sized (15) but generated at game-date/week granularity to get thousands of lists.
Deliverables:
src/models/player_encoder.py, src/models/set_attention.py, src/models/deep_set_rank.py
src/models/listmle_loss.py
src/training/train_model_a.py
Phase 4 — Model B (tabular ensemble without Net Rating)
Feature set: Four Factors, SOS, SRS, pace, and derived features; exclude net_rating.
Models:
xgboost.XGBRegressor for non-linearities
sklearn.ensemble.RandomForestRegressor for variance reduction
Deliverables:
src/features/four_factors.py, src/features/team_context.py
src/models/xgboost_ensemble.py, src/models/rf_ensemble.py
src/training/train_model_b.py
Phase 5 — True stacking with OOF + RidgeCV (default)
OOF generation:
For each training season (walk-forward), generate OOF predictions from Model A + Model B components.
Persist OOF to outputs/oof_*.parquet.
Level-2:
Fit sklearn.linear_model.RidgeCV on OOF predictions.
Produce a final scalar score per team; rank teams by this score.
Notes from docs:
Use scikit-learn probability tooling where applicable (e.g. ROC-AUC uses score/prob inputs); keep calibration optional for later.
Deliverables:
src/models/stacking.py (OOF builder + RidgeCV fitter)
src/training/train_stacking.py
Phase 6 — Evaluation (walk-forward) + baselines
Temporal splits:
Walk-forward seasons: Train -> Val -> Test by year.
Ranking metrics:
NDCG, Spearman, MRR for season-end eval.
Sleeper = Upset:
Build an “upset event” dataset from playoff outcomes (lower seed beats higher seed) and compute ROC-AUC using model score.
Baselines:
Rank-by-SRS
Rank-by-Net-Rating
Optional: previous-season rank baseline
Deliverables:
src/evaluation/metrics.py, src/evaluation/evaluate.py
scripts/5_evaluate.py
Phase 7 — Explainability + attention validation
Model B:
SHAP with TreeExplainer (XGB/RF) for global feature importance.
Model A:
Integrated Gradients (Captum) for attribution.
Ablation validation: mask high-attention player vs random player; quantify delta in ranking quality.
Docs consulted:
Captum IntegratedGradients supports baselines and additional forward args (masks) as needed.
Deliverables:
src/viz/shap_heatmap.py
src/viz/attention_roster.py + ablation experiment code
Optional: src/viz/ig_attribution.py
Phase 8 — Inference + outputs schema
Inference:
Load trained A/B/stacker and output per-team JSON.
Output schema:
Include predicted rank, true_strength_score, agreement diagnostics, roster contributors.
Define true_strength_score mapping (default percentile; alternatives softmax/platt in config).
Deliverables:
src/inference/predict.py
scripts/6_run_inference.py
Implementation notes (from Context7)
PyTorch:
Use a custom collate_fn (padding + masks) when batching variable-length inputs.
Use attention masking patterns similar to Transformer examples.
Captum:
Use IntegratedGradients.attribute(inputs, baselines=..., additional_forward_args=..., n_steps=...).
scikit-learn:
ROC-AUC uses classifier probabilities or continuous scores; Brier score is available for calibrated probability evaluation when you add future W/L later.