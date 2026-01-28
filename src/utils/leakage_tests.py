"""Lightweight leakage tests: run before training."""

from __future__ import annotations

import pandas as pd

from src.features.rolling import compute_rolling_stats
from src.features.team_context import FORBIDDEN, build_team_context


def test_no_future_leakage_in_features():
    """Assert no feature computation uses game_date >= as_of_date."""
    df = pd.DataFrame({
        "player_id": [1, 1, 1],
        "game_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "pts": [10, 12, 14],
        "reb": [3, 4, 5],
        "min": [30, 32, 28],
    })
    out = compute_rolling_stats(df, as_of_date="2024-01-02", stat_cols=["pts", "reb"], windows=[2])
    # Output must only include rows with game_date < 2024-01-02
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    bad = out[out["game_date"] >= pd.to_datetime("2024-01-02").date()]
    assert len(bad) == 0, "rolling must not include rows with game_date >= as_of_date"


def test_model_b_excludes_net_rating():
    """Assert Model B feature set and FORBIDDEN include no net_rating."""
    assert "net_rating" in FORBIDDEN or any("net_rating" in str(f).lower() for f in FORBIDDEN)

    # build_team_context must not produce net_rating
    games = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "game_date": ["2024-01-01", "2024-01-02"],
        "home_team_id": [1, 2],
        "away_team_id": [2, 1],
    })
    tgl = pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"],
        "team_id": [1, 2, 2, 1],
        "fgm": [40, 38, 42, 39],
        "fga": [85, 88, 86, 84],
        "fg3m": [10, 9, 11, 10],
        "ftm": [8, 12, 7, 11],
        "fta": [10, 14, 9, 13],
        "tov": [12, 10, 11, 13],
        "oreb": [8, 7, 9, 8],
        "dreb": [32, 34, 33, 31],
    })
    ctx = build_team_context(tgl, games)
    for c in ctx.columns:
        assert not any(f in str(c).lower() for f in FORBIDDEN), f"Model B must not include {c}"


def run_all():
    test_no_future_leakage_in_features()
    test_model_b_excludes_net_rating()
