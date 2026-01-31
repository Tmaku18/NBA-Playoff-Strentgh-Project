# Playoff-Aware Rankings & Odds Plan

## Goal

Extend the pipeline to compare the league-wide 1–30 rank against playoff performance, add championship odds from strength scores, and add conference-specific predicted ranks (1–15) that align with the pred-vs-actual plot. Keep all current behavior and add new functionality behind configuration.

## Key choices confirmed

- **Play-In games:** Exclude from playoff win counts.
- **Odds method:** Softmax with temperature.
- **Playoff data storage:** Separate playoff tables.
- **Targeting:** Add a config switch to train/evaluate on playoff performance rank while keeping existing standings-based targets.

## Plan

### 1) Add playoff tables and raw ingestion

- Update **src/data/db_schema.py** to add `playoff_games`, `playoff_team_game_logs`, `playoff_player_game_logs` (mirroring existing tables). Include `season` and `game_date` in `playoff_games` for easier joins.
- Extend **src/data/db_loader.py**:
  - Add loader logic for raw playoff parquet/CSV files (naming like `playoffs_team_game_logs_{season}.parquet`, `playoffs_player_game_logs_{season}.parquet`).
  - Build `playoff_games` from playoff team logs (same MATCHUP parsing as regular season).
  - Provide a new helper `load_playoff_data()` or extend `load_training_data()` to optionally return playoffs in addition to regular season.
- Extend **scripts/1_download_raw.py** to pull playoff logs for each season (2015–16 through 2025–26) using `nba_api` (SeasonType=`Playoffs`) and save to `data/raw/` with clear filenames.
- Extend **scripts/2_build_db.py** to ingest those new playoff files into the new tables.

### 2) Compute playoff-performance ground truth

Create a dedicated module, e.g. **src/evaluation/playoffs.py** (or `src/utils/playoffs.py`), to compute:

- **Playoff wins per team per season** from `playoff_team_game_logs` (exclude Play-In).
- **Playoff performance rank (1–30)** using the specified phases:
  1. Rank playoff teams by **total playoff wins**.
  2. Tie-break by **regular-season win %** (from regular season logs).
  3. Rank non-playoff teams (0 playoff wins) as **17–30** by regular-season win %.
- Expose a helper to return `team_id -> playoff_rank` (per season), and optionally `team_id -> playoff_wins` and `team_id -> playoff_finish_label` for reporting.

### 3) Config switch for targets + odds temperature

- Add to **config/defaults.yaml**:
  - `training.target_rank: standings | playoffs` (default `standings` to preserve current behavior).
  - `output.odds_temperature: <float>` for softmax temperature.
- Update **scripts/4b_train_stacking.py** and **src/training/train_stacking.py** to use the playoff-performance rank as the training target when `training.target_rank == "playoffs"`.
- Keep all existing targets and outputs when the flag is `standings`.

### 4) Prediction outputs (JSON)

Update **src/inference/predict.py** to add new fields without removing existing ones:

- **prediction.global_rank**: league-wide 1–30 (existing `predicted_rank` can remain for backwards compatibility).
- **prediction.conference_rank**: 1–15 within East/West (ranked by ensemble score within each conference).
- **prediction.championship_odds**: softmax over final scores with temperature from config.
- **prediction.true_strength_score**: keep existing scale (0–1) and optionally add a 0–100 scaled field if desired (`true_strength_score_100`).
- **analysis.playoff_rank** (when available) and **analysis.rank_delta_playoffs** = playoff_rank − global_rank.

### 5) Visuals for new comparisons

Extend plotting in **src/inference/predict.py** or add a small viz helper module:

- **Update pred_vs_actual.png** to use **conference_rank vs actual conference rank** (rank-to-rank on same scale).
- **Add pred_vs_playoff_rank.png**: global rank (1–30) vs playoff performance rank (1–30).
- **Add title_contender_scatter.png**: championship odds vs regular-season wins (or win %), highlighting low-wins/high-odds sleepers.
- **Add odds_top10.png**: bar chart for top-10 championship odds.
- **Add sleeper_timeline.png**: global rank over time for a configurable list of teams (use `build_lists` dates and per-date inference). Add config `viz.sleeper_teams` or `viz.timeline_top_n`.

### 6) Evaluation updates

Update **src/evaluation/metrics.py** and **src/evaluation/evaluate.py** to add:

- **Spearman** between predicted global rank and playoff performance rank.
- **NDCG@4** for identifying the final four.
- **Brier score** for championship odds (one-hot champion vs predicted odds).
- Keep existing metrics and add a new section in `eval_report.json` (e.g., `playoff_metrics`).

### 7) README and outputs documentation

Update **README.md**:

- Define **playoff performance rank** and the three-phase logic.
- Document new fields in `predictions.json` (`global_rank`, `conference_rank`, `championship_odds`).
- Describe new plots and metrics.

## Files likely to change

- **Schema & loader:** src/data/db_schema.py, src/data/db_loader.py, scripts/1_download_raw.py, scripts/2_build_db.py
- **Playoff rank logic:** src/evaluation/playoffs.py (new)
- **Training + inference:** src/training/train_stacking.py, scripts/4b_train_stacking.py, src/inference/predict.py
- **Evaluation + docs:** src/evaluation/metrics.py, src/evaluation/evaluate.py, README.md

## Implementation todos

- Add playoff tables + ingestion + loader support.
- Implement playoff-performance rank computation.
- Add config switch for playoff targets + odds temperature.
- Extend inference outputs (global/conference ranks, odds) and visuals.
- Add new playoff-aware evaluations + README updates.
