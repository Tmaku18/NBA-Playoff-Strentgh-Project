"""Kaggle SOS/SRS loader. Expects CSV(s) already downloaded (no credentials in repo)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_kaggle_sos_srs(
    path: str | Path,
    *,
    team_col: str = "Team",
    season_col: str = "Season",
    sos_col: str = "SOS",
    srs_col: str = "SRS",
) -> pd.DataFrame:
    """
    Load a Kaggle (or similar) CSV with SOS and SRS.
    Returns DataFrame with normalized team/season keys and sos, srs columns.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: str(c).strip())
    # map to expected names if present
    for dst, candidates in [
        (team_col, ["Team", "team", "TEAM", "Team Abbreviation", "Abbreviation"]),
        (season_col, ["Season", "season", "SEASON", "Year"]),
        (sos_col, ["SOS", "sos", "Strength of Schedule"]),
        (srs_col, ["SRS", "srs", "Simple Rating"]),
    ]:
        for c in candidates:
            if c in df.columns and c != dst:
                df = df.rename(columns={c: dst})
    return df


def normalize_sos_srs_to_team_season(
    df: pd.DataFrame,
    *,
    team_col: str = "Team",
    season_col: str = "Season",
    sos_col: str = "SOS",
    srs_col: str = "SRS",
) -> pd.DataFrame:
    """
    Normalize to team_id/season where possible. If team_col is abbreviation, keep it;
    caller can join to teams.team_id via teams.abbreviation.
    Returns DataFrame with columns: team (or team_abbreviation), season, sos, srs.
    """
    out = df[[c for c in [team_col, season_col, sos_col, srs_col] if c in df.columns]].copy()
    out = out.rename(columns={team_col: "team_abbreviation", season_col: "season", sos_col: "sos", srs_col: "srs"})
    out = out.dropna(subset=["team_abbreviation", "season"], how="all")
    return out
