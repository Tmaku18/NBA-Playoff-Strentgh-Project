---
name: ""
overview: ""
todos: []
isProject: false
---

# Comprehensive Feature and Evaluation Expansion Plan

## Part A: Data Evolution Context and Design Philosophy

Understanding the evolution of basketball analytics clarifies why certain features are chosen.

### Order of Metrics (Historical Progression)


| Order      | Type             | Examples                             | Use for Prediction                                           |
| ---------- | ---------------- | ------------------------------------ | ------------------------------------------------------------ |
| **First**  | Box Score        | PTS, REB, AST                        | Useful for history; poor for prediction due to pace bias     |
| **Second** | Possession-Based | Four Factors, Off/Def Rating per 100 | Removes pace noise; bedrock of feature engineering           |
| **Third**  | Adjusted Impact  | RAPM, BPM, RPM, LEBRON               | Solves lineup collinearity; isolates individual contribution |
| **Fourth** | Tracking/ML      | Second Spectrum XY, Shot Quality     | Enables spatial features; where available                    |


**Design principle:** For predicting game outcomes and championships, the most effective models operate at the intersection of **Second Order** (efficiency features) and **Third Order** (player impact), using Fourth Order concepts as engineered features where available.

---

## Part B: Canonical Feature Engineering (Second Order)

### B.1 Four Factors (Dean Oliver)

The Four Factors compress the game into four components that mathematically explain a win. Use **rates**, not totals, to satisfy IID assumptions.


| Factor        | Metric      | Calculation                  | Predictive Weight | Rationale                                             |
| ------------- | ----------- | ---------------------------- | ----------------- | ----------------------------------------------------- |
| Shooting      | eFG%        | (FG + 0.5 × 3PM) / FGA       | ~40%              | Strongest correlate; accounts for 3pt geometric value |
| Ball Security | TOV%        | TOV / (FGA + 0.44×FTA + TOV) | ~25%              | Non-linear; threshold (~18%) correlates with collapse |
| Rebounding    | ORB% / DRB% | ORB / (ORB + OppDRB)         | ~20%              | DRB% more stable than ORB%                            |
| Free Throws   | FTR         | FTA / FGA                    | ~15%              | Uncapped efficiency, foul trouble                     |


**Implementation:** [src/features/four_factors.py](src/features/four_factors.py) already has Four Factors. Ensure all are **per-possession normalized**. Add **differential features** (team − opponent) for denser signal.

### B.2 Advanced Team Strength

#### Pythagorean Expectation

$$\text{Win}_{Pyth} = \frac{\text{PointsScored}^{13.91}}{\text{PointsScored}^{13.91} + \text{PointsAllowed}^{13.91}}$$

- Exponent 13.91 minimizes MSE for NBA.
- **Regression Candidate feature:** Actual Win% − Pythagorean Win%. If Actual > Pyth, team is "lucky"; predict downturn.

#### Simple Rating System (SRS)

$$SRS = \text{Margin of Victory} + \text{Strength of Schedule}$$

- SOS = average SRS of opponents. Solve simultaneously for all 30 teams.
- **Predictive power:** SRS often outperforms seeding for playoff success.
- **Implementation:** Add SRS to Model B features. Solve linear system or use iterative method.

---

## Part C: Player Impact (Third Order)

### C.1 RAPM (Regularized Adjusted Plus-Minus)

**Math:** Linear regression per stint (no substitutions):
$$Y = X\beta + \epsilon$$

- $Y$ = point differential per possession
- $X$ = binary indicators (1 home, -1 away)

**Regularization:** Ridge ($L2$): $\lambda \sum \beta^2$ to handle multicollinearity.

**Implementation:** [src/features/on_off.py](src/features/on_off.py) — RAPM-lite using plus_minus. Full RAPM requires stint-level data; approximate with lineup-based +/- or player plus_minus rolling averages.

### C.2 RAPTOR / LEBRON (External Data)

- **RAPTOR:** Integrates tracking; separate Offensive and Defensive RAPTOR for lineup.
- **LEBRON:** Luck-adjusted; useful for early-season (October–December) when sample sizes are small.
- **Implementation:** If external API/CSV available, add `off_raptor_sum_top5`, `def_raptor_sum_top5` to team context. Document data source in config.

---

## Part D: Rating Systems as Features

### D.1 Elo Rating

$$R_{new} = R_{old} + K \cdot (S_{actual} - S_{expected})$$

- **Design:** Dynamic K-factor: higher K (e.g., 20) early season; lower K late season.
- **Implementation:** Add [src/features/elo.py](src/features/elo.py). Compute Elo per team as-of each date. Add to Model B.

### D.2 Massey Ratings

- Matrix $M$: diagonal = games played; off-diagonal = -1 for opponents.
- Solve $Mr = p$ where $p$ = point differential.
- **Massey Rank Differential:** Feed into XGBoost as team strength signal.
- **Implementation:** Add [src/features/massey.py](src/features/massey.py). Compute per season/date.

---

## Part E: Calibration (Platt Scaling)

**Recommendation:** Train Logistic Regression on raw XGBoost logits to calibrate probabilities.

- **Method:** Platt Scaling. Ensures "60% probability" ≈ 60% actual wins.
- **Implementation:** In stacking or post-processing, add calibration layer. Tune on validation Brier. Add to [src/models/stacking.py](src/models/stacking.py) or separate calibration step.

---

## Part F: Hyperparameter Tuning Strategy

### F.1 Bayesian Optimization (Replace Grid Search)

**Rationale:** Grid search is inefficient for high-dimensional spaces. Bayesian Optimization builds a probability model of the objective and explores the hyperparameter space more intelligently than random search.

**Libraries:** Optuna or Hyperopt.

**Implementation:** Extend [scripts/sweep_hparams.py](scripts/sweep_hparams.py) to use Optuna (or Hyperopt) instead of brute-force grid. Define an objective function that runs one pipeline config and returns validation metric (e.g., NDCG, Spearman, or Brier). Use `optuna.create_study(direction="maximize")` for NDCG/Spearman or `direction="minimize"` for Brier. Run `study.optimize(objective, n_trials=N)`.

### F.2 XGBoost Hyperparameters for NBA Data


| Parameter             | Recommended Range | Description                           | Why for NBA Data                                                                                                              |
| --------------------- | ----------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `learning_rate` (eta) | 0.005 – 0.05      | Step size shrinkage                   | NBA data is noisy. Low LR + more trees prevents over-committing to noise (e.g., buzzer-beater luck).                          |
| `max_depth`           | 3 – 5             | Max tree depth                        | Keep shallow. Deep trees memorize 2018 roster matchups; shallow trees learn broad patterns (e.g., "Efficiency beats Volume"). |
| `min_child_weight`    | 5 – 15            | Min sum of instance weight in a child | High values prevent leaves with 1–2 games. Forces patterns that exist across many games.                                      |
| `colsample_bytree`    | 0.5 – 0.7         | Subsample ratio of columns            | Diversity. If one stat (e.g., Net Rating) dominates, model becomes brittle.                                                   |
| `lambda` (reg_l2)     | 1.0 – 10.0        | L2 regularization                     | Strong regularization shrinks feature weights, reduces outlier impact (e.g., 150 pts in triple-OT).                           |
| `alpha` (reg_l1)      | 1.0 – 10.0        | L1 regularization                     | Same rationale as lambda.                                                                                                     |


**Optuna suggest:** Use `suggest_float("learning_rate", 0.005, 0.05, log=True)`, `suggest_int("max_depth", 3, 5)`, etc.

---

## Part G: Original Phase-Based Implementation Plan

### Phase 1: Foundation Features

- Lineup continuity ([src/features/lineup_continuity.py](src/features/lineup_continuity.py))
- Rest/fatigue ([src/features/fatigue.py](src/features/fatigue.py))
- Momentum deltas ([src/features/momentum.py](src/features/momentum.py))
- Sleeper threshold (|delta| >= 2), IG L1 normalize

### Phase 2: On/Off and Team Strength

- RAPM-lite ([src/features/on_off.py](src/features/on_off.py))
- SRS ([src/features/srs.py](src/features/srs.py))
- Pythagorean expectation and regression candidate ([src/features/team_context.py](src/features/team_context.py))

### Phase 3: Rating Systems and Four Factors Enhancement

- Elo ([src/features/elo.py](src/features/elo.py))
- Massey ([src/features/massey.py](src/features/massey.py))
- Four Factors differentials; ensure possession normalization

### Phase 4: Metrics

- ECE, reliability diagram, Precision@k, Kendall Tau, rank volatility, ensemble lift

### Phase 5: Model A

- Learnable attention temperature, MC Dropout

### Phase 6: Training and Calibration

- Progressive training, odds temperature tune, Platt Scaling calibration
- **Hyperparameter tuning:** Replace sweep grid with Bayesian Optimization (Optuna); use NBA-specific XGBoost ranges (Part F)

### Phase 7: Playoff Residual

- Delta model, matchup features, playoff contribution per-100

### Phase 8: Playoff Evaluation

- Round advancement, upset ROC-AUC, champion log loss, calibration by round

---

## File Change Summary (Updated)


| File                                                                   | Changes                                                      |
| ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| [src/features/lineup_continuity.py](src/features/lineup_continuity.py) | New — continuity metrics                                     |
| [src/features/fatigue.py](src/features/fatigue.py)                     | New — rest/fatigue                                           |
| [src/features/momentum.py](src/features/momentum.py)                   | New — delta features                                         |
| [src/features/on_off.py](src/features/on_off.py)                       | New — RAPM-lite                                              |
| [src/features/elo.py](src/features/elo.py)                             | New — Elo with dynamic K                                     |
| [src/features/massey.py](src/features/massey.py)                       | New — Massey ratings                                         |
| [src/features/srs.py](src/features/srs.py)                             | New — SRS + SOS                                              |
| [src/features/matchup.py](src/features/matchup.py)                     | New — opponent features (playoff)                            |
| [src/features/four_factors.py](src/features/four_factors.py)           | Add differentials, possession norm                           |
| [src/features/team_context.py](src/features/team_context.py)           | Integrate SRS, Pythagorean, Elo, Massey, continuity, fatigue |
| [src/evaluation/metrics.py](src/evaluation/metrics.py)                 | ECE, Precision@k, Kendall Tau                                |
| [src/models/stacking.py](src/models/stacking.py)                       | Platt Scaling calibration                                    |
| [config/defaults.yaml](config/defaults.yaml)                           | New flags, K-factor schedule                                 |
| [scripts/sweep_hparams.py](scripts/sweep_hparams.py)                   | Optuna Bayesian Optimization; NBA XGBoost ranges             |


---

## Recommended Execution Order

1. **Four Factors + SRS + Pythagorean** (Part B) — foundation
2. **Phase 1** (continuity, fatigue, momentum) — high ROI
3. **Elo + Massey** (Part D) — strong team strength signals
4. **Phase 3** (metrics), **Phase 4** (Model A)
5. **Platt Scaling** (Part E) — calibration
6. **RAPM-lite** (Part C) — player impact
7. **Phases 6–8** (playoff residual and eval)

