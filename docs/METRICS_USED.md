# Metrics used in the project

Documentation of metrics added for model inputs and evaluation (see NBA_ANALYST_METRICS and ANALYSIS.md).

## Player rolling features (Model A)

- **TS% (True Shooting %):** `PTS / (2 * (FGA + 0.44 * FTA))` per rolling window (L10, L30). Computed in [src/features/rolling.py](../src/features/rolling.py); columns `ts_pct_L10`, `ts_pct_L30`.
- **Usage rate:** `FGA + 0.44 * FTA + TOV` per game over rolling window (L10, L30). Columns `usage_L10`, `usage_L30`.
- **Stat dim:** With TS% and usage, Model A stat_dim is 21 (18 base L10+L30 + 2 on_court_pm_approx + 1 pct_min_returning).

## Playoff contribution (evaluation)

- **Game Score (per player):** `PTS + 0.5*REB + 1.5*AST + 2*STL + 2*BLK - TOV` per game; aggregated by (player_id, team_id, season). Implemented in `compute_playoff_contribution_per_player` in [src/evaluation/playoffs.py](../src/evaluation/playoffs.py). Returns `contribution_per_game` and `total_contribution`.
- **BPM-style:** Not yet implemented; could be added as an alternative to Game Score using box-score regression (see NBA_ANALYST_METRICS).

## Rank-distance metrics (evaluation)

- **rank_mae**: Mean absolute error of predicted vs actual rank (1=best, 30=worst). Lower = better. Usage: `pred_vs_playoff` (model predictions vs playoff outcome) and `standings_vs_playoff` (baseline: regular-season standings vs playoff outcome). Implemented in [src/evaluation/metrics.py](../src/evaluation/metrics.py).
- **rank_rmse**: Root mean squared error of rank predictions. Penalizes large errors more than MAE. Lower = better. Same usage as rank_mae.

## NDCG variants

- **ndcg / ndcg10**: NDCG@10 — ranking quality over top 10 positions. Main `ndcg` key uses k=10; `ndcg10` is an explicit alias. Relevance = strength (higher = better).
- **ndcg_at_4 (final four)**: NDCG truncated at top 4; used for playoff "final four" evaluation. Implemented in `ndcg_at_4`, `ndcg_at_10` in [src/evaluation/metrics.py](../src/evaluation/metrics.py).

## Calibration and Phase 3 metrics

- **ECE (Expected Calibration Error):** Implemented in [src/evaluation/metrics.py](../src/evaluation/metrics.py) as `ece(y_true, y_prob, n_bins=10)`. Use with championship probabilities vs one-hot champion label.
- **Platt scaling:** Two variants in [src/models/calibration.py](../src/models/calibration.py): (1) on meta-learner output, (2) on raw model outputs then combine. See [docs/PLATT_CALIBRATION.md](PLATT_CALIBRATION.md).
- **Pythagorean expectation:** Per comprehensive plan, Win_Pyth = PointsScored^13.91 / (PointsScored^13.91 + PointsAllowed^13.91). Regression candidate: Actual Win% − Pythagorean Win%. Not yet in team_context; add when needed.
