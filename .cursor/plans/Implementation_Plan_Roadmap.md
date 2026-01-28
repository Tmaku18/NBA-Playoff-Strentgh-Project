# Implementation Plan Roadmap: NBA True Strength Prediction

**Source of truth:** [`.cursor/plans/Plan.md`](.cursor/plans/Plan.md)

**Key locked decisions (from Plan.md):**
- Primary target: playoff_seed / conference rank (ListMLE training; season-end evaluation)
- Stacking level-2: **RidgeCV** on OOF predictions (not Logistic Regression)
- No net_rating as model input (Model A or B); allowed only as baseline ranker
- Strict t-1: all features and rosters as-of date only; never target or future games

---

## Phase 0 — Repo scaffolding + guardrails

- Directory skeleton: `config/`, `src/{data,features,models,training,evaluation,inference,viz,utils}`, `scripts/`, `data/raw`, `data/processed`, `outputs/`
- `requirements.txt`, `config/defaults.yaml` (seasons, DB path, `repro.seed`, model params, `true_strength_score` mapping)
- `src/utils/repro.py`: `set_seeds(seed)` for random/numpy/torch
- Run manifests: `outputs/run_manifest.json` (config snapshot, git hash, data hash)
- Guardrails: as-of semantics helper, leakage-test module

**Exit:** `python -m scripts.<entrypoint>` imports project, reads config, writes manifest.

---

## Phase 1 — Data acquisition + storage

- nba_api: `LeagueGameLog` with `player_or_team_abbreviation='P'` or `'T'`; rate limiting + caching
- DuckDB: `src/data/db.py`, `src/data/db_schema.py`; `read_csv`/`read_parquet`, `CREATE TABLE AS SELECT * FROM df`, `INSERT INTO … BY NAME`
- Kaggle SOS/SRS loader; `data/manifest.json` (hashes, timestamps)
- **Context7 nba_api:** `leaguegamelog.LeagueGameLog(season=..., season_type_all_star='Regular Season', sorter='DATE', direction='ASC', player_or_team_abbreviation='T'|'P', league_id='00')`; `get_data()` or `get_data_frames()`. `PlayerOrTeam` accepts `^(P)|(T)$`.

**Exit:** One command builds `data/processed/nba.duckdb` and `data/manifest.json`.

---

## Phase 2 — Feature engineering (leakage-proof)

- **Rolling:** `src/features/rolling.py` — `shift(1)` before rolling (L10/L30); DNP = per-game avg over games played + availability fraction
- **Roster:** `src/features/build_roster_set.py` — top-N by minutes as-of date only; pad to 15; masks; hash-trick for unseen players
- **Model B features:** `four_factors.py`, `team_context.py` (SOS, SRS, pace, derived); **exclude net_rating**
- **Lists:** `src/training/build_lists.py` — conference-date/week lists; target = standings-to-date (win-rate-to-date)
- Leakage tests: no `game_date >= as_of_date`; no `net_rating` in Model B features

**Exit:** One season of conference-date lists + features; leakage tests pass.

---

## Phase 3 — Model A (Deep Set) + stable ListMLE

- `player_encoder.py`, `player_embedding.py` (hash-trick), `set_attention.py`, `deep_set_rank.py` → `(score, Z, attention_weights)`
- **Attention:** `nn.MultiheadAttention(..., batch_first=True)`; `key_padding_mask` shape `(batch, num_players)`, `True` = ignore (padded)
- **ListMLE:** `torch.logsumexp` / `torch.logcumsumexp` (no raw exp then log); custom `collate_fn` with `pad_sequence(..., batch_first=True)` for variable-length rosters
- `src/training/train_model_a.py`: seeds, checkpointing, walk-forward train/val

**Exit:** Model A trains on conference-date lists; sensible val rankings.

---

## Phase 4 — Model B (XGB + RF, no Net Rating)

- `XGBRegressor(early_stopping_rounds=50, ...)`; `fit(..., eval_set=[(X_val,y_val)], verbose=False)`; `model.best_iteration`; optional `predict(..., iteration_range=(0, best_iteration))`
- `src/training/train_model_b.py`; same ranking target as Model A for list construction

**Exit:** Model B trains and evaluates on same val splits as A.

---

## Phase 5 — OOF stacking + RidgeCV meta-learner

- OOF for Model A score, XGB score, RF score; persist `outputs/oof_*.parquet`
- **RidgeCV** on stacked OOF: `X_meta = np.column_stack([oof_deep_set, oof_xgb, oof_rf])`; `RidgeCV().fit(X_meta, y)`. Do **not** use `StackingRegressor`’s internal CV; we use manual temporal OOF.
- Final score → rank → `true_strength_score` (percentile default)

**Exit:** Ensemble rankings for a held-out season from OOF-trained RidgeCV.

---

## Phase 6 — Evaluation + baselines

- `src/evaluation/metrics.py`: NDCG, Spearman, MRR; optional Brier when game-outcome head exists
- Baselines: rank-by-SRS, rank-by-Net-Rating (from off/def, not as feature), optional previous-season rank
- `src/evaluation/evaluate.py`: walk-forward by season; playoff upset ROC-AUC (lower seed beats higher)

**Exit:** Repeatable eval script and metrics report.

---

## Phase 7 — Explainability

- **Model B (SHAP):** `shap.TreeExplainer(model)`; `explainer(X)` or `explainer.shap_values(X)`. Use official SHAP docs; `feature_perturbation` default `"auto"`; if `"interventional"`, provide background `data`.
- **Model A (Captum):** `IntegratedGradients.attribute(inputs, baselines=..., additional_forward_args=(minutes, mask), n_steps=100, return_convergence_delta=True)`; use `additional_forward_args` for non-primary forward inputs.
- **Attention ablation:** mask top-attention vs random; compare ranking metric drop.

**Exit:** SHAP summary plot; attention-ablation result.

---

## Phase 8 — Inference + outputs + viz

- `src/inference/predict.py`: load A/B/stacker; produce JSON per team (predicted_rank, true_strength_score, delta, ensemble agreement, primary_contributors)
- Plots: predicted vs actual rank; fraud/sleeper delta; SHAP summary; attention distribution; sleeper timeline

**Exit:** One command writes `outputs/<run_id>/predictions.json` and plots.

---

## Phase 9 — Polish

- README: methodology decisions, run instructions, anti-leakage checklist
- Manifests: config + data hashes + code version
- Export figures/tables for report

---

## Context7 implementation notes (Jan 2026)

**Libraries:** `/swar/nba_api`, `/websites/duckdb_stable`, `/pytorch/pytorch`, `/dmlc/xgboost`, `/websites/scikit-learn_stable`, `/pytorch/captum`. SHAP: use upstream docs.

### nba_api LeagueGameLog
- Param: `player_or_team_abbreviation` → `PlayerOrTeam`; values `'P'`, `'T'`. Rate-limit (e.g. ≥0.6s) and retry/timeout handling.

### DuckDB Python
- `duckdb.read_csv/read_parquet/read_json`; `CREATE TABLE t AS SELECT * FROM df`; `INSERT INTO t BY NAME SELECT * FROM df`; `con.register("v", df)` for non-global refs. Parquet: `INSERT INTO t SELECT * FROM read_parquet('f.parquet')`.

### PyTorch MultiheadAttention + ListMLE
- `key_padding_mask` `(B,L)`, `True`=ignore. Variable-length: `pad_sequence(..., batch_first=True)` in `collate_fn`; feed matching `key_padding_mask`. ListMLE: `torch.logsumexp`/`torch.logcumsumexp` + max-shift; no raw exp-then-log.

### XGBRegressor early stopping
- `early_stopping_rounds` in **constructor**; `eval_set` in `fit()`; `best_iteration` after fit; `iteration_range=(0, best_iteration)` in `predict` optional.

### Stacking
- Manual temporal OOF + `RidgeCV` as sole meta-learner; no `StackingRegressor` CV. RidgeCV does internal alpha CV; do not nest `cross_val_predict` around it.

### Captum IntegratedGradients
- `ig.attribute(inputs, baselines=..., additional_forward_args=(minutes, mask), n_steps=100, return_convergence_delta=True)` → `(attributions, delta)`. Use `additional_forward_args` for Deep Set’s `minutes`, `mask`, etc.

### SHAP (Model B)
- `TreeExplainer(model, data?)`; `explainer(X)`. `feature_perturbation="auto"`; if `"interventional"`, pass `data`. One explainer per XGB/RF or combined view; `shap.summary_plot` / `Explanation`.

---

## Build strategy

Thin vertical slice first: Phase 0 → 1 (one season) → 2 (one conference-date list) → 3 (Model A few epochs) → 6 (one-season eval). Then multi-season walk-forward, Model B, stacking, explainability, inference.
